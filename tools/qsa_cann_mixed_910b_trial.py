# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Same FP32/N64/Top2048 algorithm with one or two outer-block ring slots."""

from functools import partial
import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle
import triton.language.extra.cann.extension as al
from probe_qsa_int8_mixed_pipeline_910b import SORT_IMPL_BASE
from qsa_int8_fused_binary_910b_trial import pack_mixed_queries

pipe = al.PIPE


@triton.jit
def _merge_equal2048(old, local, packed_storage, MERGE_KIND: tl.constexpr):
    if MERGE_KIND == 1:
        old_slot = tle.dsa.subview(packed_storage, [0], [4096], [1])
        local_slot = tle.dsa.subview(packed_storage, [4096], [4096], [1])
        tle.dsa.to_buffer(old, space=tle.dsa.ascend.UB, bind_buffer=old_slot)
        tle.dsa.to_buffer(local, space=tle.dsa.ascend.UB, bind_buffer=local_slot)
        tl.debug_barrier()
        packed = tle.dsa.to_tensor(packed_storage)
    else:
        WORDS: tl.constexpr = 16384 if MERGE_KIND == 2 else 8192
        packed = tl.zeros((WORDS,), tl.float32)
        packed = tle.dsa.insert_slice(packed, old, (0,), (4096,), (1,))
        packed = tle.dsa.insert_slice(packed, local, (4096,), (4096,), (1,))
    THIRD: tl.constexpr = 4096 if MERGE_KIND == 2 else 0
    FOURTH: tl.constexpr = 6144 if MERGE_KIND == 2 else 0
    merged, consumed = tle.dsa.ascend.raw(
        "merge_exhaust_sort4", packed, 2, 0, 2048, THIRD, FOURTH, 2048, 2048, 0, 0,
        out=[tl.zeros((8192,), tl.float32), tl.zeros((4,), tl.int32)],
    )
    return al.extract_slice(merged, (0,), (4096,), (1,)), consumed


@triton.jit
def _merge_representation_probe(Old, Local, Output, Counts, MERGE_KIND: tl.constexpr):
    word = tl.arange(0, 4096)
    old = tl.load(Old + word)
    local = tl.load(Local + word)
    if MERGE_KIND == 1:
        scratch = tle.dsa.alloc([8192], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    else:
        scratch: tl.constexpr = None
    state, consumed = _merge_equal2048(old, local, scratch, MERGE_KIND)
    tl.store(Output + word, state)
    tl.store(Counts + tl.arange(0, 4), consumed)


@triton.jit
def _consume_outer2048(
    QS, KS, Weight, Table, Visible, Ring, TDots, TScores, TLocal, TState, TCounts,
    score_storage, state_storage, packed_storage, outer,
    ROWS: tl.constexpr, COLUMNS: tl.constexpr, WIDTH: tl.constexpr,
    BLOCKS: tl.constexpr, SLOTS: tl.constexpr, TRACE: tl.constexpr, FIRST: tl.constexpr, STAGE: tl.constexpr,
    TABLE_PLAIN: tl.constexpr, RING_TILE_MAJOR: tl.constexpr, MERGE_KIND: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.arange(0, 4)
    n = tl.arange(0, 64)
    vector_slot = outer % SLOTS
    al.sync_block_wait("cube", "vector", vector_slot, pipe.PIPE_FIX, pipe.PIPE_MTE2)
    for inner in range(32):
        start = outer * 2048 + inner * 64
        if TABLE_PLAIN:
            page = tl.load(Table + tl.minimum(start // 192, COLUMNS - 1))
        else:
            page = tl.load(Table + tl.minimum(start // 192, COLUMNS - 1), mask=start < WIDTH, other=0)
        token = page * 192 + start % 192 + n
        key_scale = tl.load(KS + token, mask=start + n < WIDTH, other=0).to(tl.float32)
        for local_query in range(2):
            row = pid * 4 + al.sub_vec_id() * 2 + local_query
            head = al.sub_vec_id() * 8 + local_query * 4 + h
            if RING_TILE_MAJOR:
                src = Ring + (pid * SLOTS + vector_slot) * 16 * 2048 + inner * 16 * 64
                dots = tl.load(src + head[:, None] * 64 + n[None, :])
            else:
                src = Ring + (pid * SLOTS + vector_slot) * 16 * 2048 + inner * 64
                dots = tl.load(src + head[:, None] * 2048 + n[None, :])
            if STAGE >= 0:
                qs = tl.load(QS + row * 4 + h, mask=row < ROWS, other=0).to(tl.float32)
                weight = tl.load(Weight + row * 4 + h, mask=row < ROWS, other=0).to(tl.float32)
                visible = tl.load(Visible + row, mask=row < ROWS, other=0)
                live = (start + n < WIDTH) & (start + n < visible)
                dequant = dots.to(tl.float32) * qs[:, None] * key_scale[None, :]
                scores = tl.where(live, tl.sum(tl.maximum(dequant, 0.0) * weight[:, None], 0), -float("inf"))
                score_slot = tle.dsa.subview(score_storage, [local_query * 2048 + inner * 64], [64], [1])
                tle.dsa.to_buffer(scores, space=tle.dsa.ascend.UB, bind_buffer=score_slot)
                if TRACE:
                    base = (row * BLOCKS + outer) * 2048 + inner * 64
                    tl.store(TScores + base + n, scores, mask=row < ROWS)
                    tl.store(TDots + (base + n[None, :]) * 4 + h[:, None],
                             tl.where(live[None, :], dots, 0), mask=row < ROWS)
            else:
                base = (row * BLOCKS + outer) * 2048 + inner * 64
                tl.store(TDots + (base + n[None, :]) * 4 + h[:, None], dots, mask=row < ROWS)
    tl.debug_barrier()
    if STAGE >= 1:
        for local_query in range(2):
            row = pid * 4 + al.sub_vec_id() * 2 + local_query
            score_slot = tle.dsa.subview(score_storage, [local_query * 2048], [2048], [1])
            scores = tle.dsa.to_tensor(score_slot)
            pairs = tle.dsa.ascend.raw(
                "sort_1d_pack", scores, tl.zeros((8192,), tl.float32), True,
                2048, outer * 2048, SORT_IMPL_BASE, out=tl.zeros((4096,), tl.float32),
            )
            if TRACE:
                tl.store(TLocal + (row * BLOCKS + outer) * 4096 + tl.arange(0, 4096), pairs, mask=row < ROWS)
                tl.debug_barrier()
            state_slot = tle.dsa.subview(state_storage, [local_query * 4096], [4096], [1])
            if not FIRST and STAGE >= 2:
                old = tle.dsa.to_tensor(state_slot)
                new_state, counts = _merge_equal2048(old, pairs, packed_storage, MERGE_KIND)
            else:
                new_state = pairs
                counts = tl.zeros((4,), tl.int32)
            tle.dsa.to_buffer(new_state, space=tle.dsa.ascend.UB, bind_buffer=state_slot)
            tl.debug_barrier()
            if TRACE:
                tl.store(TState + (row * BLOCKS + outer) * 4096 + tl.arange(0, 4096), new_state, mask=row < ROWS)
                tl.store(TCounts + (row * BLOCKS + outer) * 4 + tl.arange(0, 4), counts, mask=row < ROWS)
    tl.debug_barrier()
    al.sync_block_set("vector", "cube", vector_slot, pipe.PIPE_MTE3, pipe.PIPE_FIX)


@triton.jit
def _mixed_outer2048(
    Q, K, QS, KS, Weight, Table, Visible, Ring, OutValues, OutIndices,
    TDots, TScores, TLocal, TState, TCounts,
    ROWS: tl.constexpr, COLUMNS: tl.constexpr, WIDTH: tl.constexpr,
    BLOCKS: tl.constexpr, SLOTS: tl.constexpr, TRACE: tl.constexpr, STAGE: tl.constexpr = 2,
    TABLE_PLAIN: tl.constexpr = False, RING_TILE_MAJOR: tl.constexpr = False, MERGE_KIND: tl.constexpr = 0,
):
    with al.scope(core_mode="cube"):
        pid = tl.program_id(0)
        m = tl.arange(0, 16)
        d = tl.arange(0, 128)
        n = tl.arange(0, 64)
        query = tl.load(Q + (pid * 16 + m[:, None]) * 128 + d[None, :])
        for outer in range(BLOCKS):
            cube_slot = outer % SLOTS
            al.sync_block_wait("vector", "cube", cube_slot, pipe.PIPE_MTE3, pipe.PIPE_FIX)
            for inner in range(32):
                start = outer * 2048 + inner * 64
                if TABLE_PLAIN:
                    page = tl.load(Table + tl.minimum(start // 192, COLUMNS - 1))
                else:
                    page = tl.load(Table + tl.minimum(start // 192, COLUMNS - 1), mask=start < WIDTH, other=0)
                token = page * 192 + start % 192 + n
                key = tl.load(K + token[:, None] * 128 + d[None, :], mask=start + n[:, None] < WIDTH, other=0)
                dot = tl.dot(query, tl.trans(key), out_dtype=tl.int32)
                if RING_TILE_MAJOR:
                    dst = Ring + (pid * SLOTS + cube_slot) * 16 * 2048 + inner * 16 * 64
                    tl.store(dst + m[:, None] * 64 + n[None, :], dot)
                else:
                    dst = Ring + (pid * SLOTS + cube_slot) * 16 * 2048 + inner * 64
                    tl.store(dst + m[:, None] * 2048 + n[None, :], dot)
            al.sync_block_set("cube", "vector", cube_slot, pipe.PIPE_FIX, pipe.PIPE_MTE2)
        for drain in tl.static_range(SLOTS):
            al.sync_block_wait("vector", "cube", drain, pipe.PIPE_MTE3, pipe.PIPE_FIX)

    with al.scope(core_mode="vector"):
        pid = tl.program_id(0)
        h = tl.arange(0, 4)
        n = tl.arange(0, 64)
        score_storage = tle.dsa.alloc([4096], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
        state_storage = tle.dsa.alloc([8192], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
        if MERGE_KIND == 1 and STAGE >= 2:
            packed_storage = tle.dsa.alloc([8192], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
        else:
            packed_storage: tl.constexpr = None
        for initial in tl.static_range(SLOTS):
            al.sync_block_set("vector", "cube", initial, pipe.PIPE_MTE3, pipe.PIPE_FIX)
        _consume_outer2048(
            QS, KS, Weight, Table, Visible, Ring, TDots, TScores, TLocal, TState, TCounts,
            score_storage, state_storage, packed_storage, 0,
            ROWS, COLUMNS, WIDTH, BLOCKS, SLOTS, TRACE, True, STAGE, TABLE_PLAIN, RING_TILE_MAJOR, MERGE_KIND,
        )
        for outer in range(1, BLOCKS):
            _consume_outer2048(
                QS, KS, Weight, Table, Visible, Ring, TDots, TScores, TLocal, TState, TCounts,
                score_storage, state_storage, packed_storage, outer,
                ROWS, COLUMNS, WIDTH, BLOCKS, SLOTS, TRACE, False, STAGE, TABLE_PLAIN, RING_TILE_MAJOR, MERGE_KIND,
            )
        if STAGE >= 1:
            for local_query in range(2):
                row = pid * 4 + al.sub_vec_id() * 2 + local_query
                state_slot = tle.dsa.subview(state_storage, [local_query * 4096], [4096], [1])
                final = al.extract_slice(tle.dsa.to_tensor(state_slot), (0,), (1024,), (1,))
                values, indices = tle.dsa.ascend.raw(
                    "unpack_sort", final, 512,
                    out=[tl.zeros((512,), tl.float32), tl.zeros((512,), tl.int32)],
                )
                rank = tl.arange(0, 512)
                live = rank < tl.load(Visible + row, mask=row < ROWS, other=0)
                tl.store(OutValues + row * 512 + rank, tl.where(live, values, -float("inf")), mask=row < ROWS)
                tl.store(OutIndices + row * 512 + rank, tl.where(live, indices, -1), mask=row < ROWS)


class PreparedMixedOuter:
    def __init__(self, values, width, *, slots, request_ids, structure=None):
        from qsa_cann_serial_910b_trial import PreparedCANNSerial
        proof = PreparedCANNSerial(values, width, query_group=4, request_ids=request_ids)
        self.structure = dict(TABLE_PLAIN=False, RING_TILE_MAJOR=False)
        if structure is not None:
            if set(structure) != set(self.structure) or any(type(value) is not bool for value in structure.values()):
                raise ValueError("Expected boolean TABLE_PLAIN and RING_TILE_MAJOR")
            self.structure.update(structure)
        if width <= 0 or values["row_table"].shape[1] < triton.cdiv(width, 192):
            raise ValueError("Mixed probe requires nonempty width and sufficient page-table columns")
        self.rows, self.width, self.slots = proof.rows, width, slots
        self.blocks = triton.cdiv(width, 2048)
        device = values["q"].device
        self.output = torch.empty((self.rows, 512), device=device, dtype=torch.int32)
        self.output_values = torch.empty_like(self.output, dtype=torch.float32)
        self.traces = (
            torch.empty((self.rows, self.blocks, 2048, 4), device=device, dtype=torch.int32),
            torch.empty((self.rows, self.blocks, 2048), device=device, dtype=torch.float32),
            torch.empty((self.rows, self.blocks, 4096), device=device, dtype=torch.float32),
            torch.empty((self.rows, self.blocks, 4096), device=device, dtype=torch.float32),
            torch.empty((self.rows, self.blocks, 4), device=device, dtype=torch.int32),
        )
        self.launches, self.compiled = [], {}
        cube_cores = int(triton.runtime.driver.active.utils.get_device_properties(device.index)["num_aicore"])
        for first, count, _ in proof.segments:
            cores = triton.cdiv(count, 4)
            if cores > cube_cores:
                raise ValueError("Diagnostic chunk exceeds physical Cube core count")
            packed = pack_mixed_queries(values["q"][first:first + count], 4)
            ring = torch.empty((cores, slots, 16, 2048), device=device, dtype=torch.int32)
            args = (packed, values["cache"], values["qs"][first:first + count], values["ks"],
                    values["weights"][first:first + count], values["row_table"][first],
                    values["visible"][first:first + count], ring,
                    self.output_values[first:first + count], self.output[first:first + count],
                    *(tensor[first:first + count] for tensor in self.traces))
            self.launches.append(partial(_mixed_outer2048[(cores,)], *args,
                ROWS=count, COLUMNS=values["row_table"].shape[1], WIDTH=width, BLOCKS=self.blocks,
                SLOTS=slots, **self.structure, use_bytecode=True, disable_auto_cv_work_space_manage=True,
                disable_auto_inject_block_sync=True, enable_auto_bind_sub_block=False,
                multibuffer=False, sync_solver=False, unit_flag=False))

    def compile_variant(self, trace, stage=2):
        compiled = []
        for launch in self.launches:
            options = dict(launch.keywords)
            grid = (triton.cdiv(options["ROWS"], 4),)
            compiled.append(_mixed_outer2048.warmup(*launch.args, grid=grid, TRACE=trace, STAGE=stage, **options))
        return compiled

    def run(self, check=None):
        for index, launch in enumerate(self.launches):
            self.compiled[f"mixed{index}"] = launch(TRACE=check is not None)
        if check:
            dots, scores, local, state, counts = self.traces
            for block in range(self.blocks):
                check("score", block * 2048, scores[:, block], dots[:, block])
                check("sort", block * 2048, local[:, block], None)
                check("state", block * 2048, state[:, block], counts[:, block] if block else None)
        return self.output

    def __call__(self):
        return self.run()
