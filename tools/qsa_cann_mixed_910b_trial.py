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
def _merge_equal2048(old, local):
    packed = tl.zeros((16384,), tl.float32)
    packed = tle.dsa.insert_slice(packed, old, (0,), (4096,), (1,))
    packed = tle.dsa.insert_slice(packed, local, (4096,), (4096,), (1,))
    merged, consumed = tle.dsa.ascend.raw(
        "merge_exhaust_sort4", packed, 2, 0, 2048, 4096, 6144, 2048, 2048, 0, 0,
        out=[tl.zeros((8192,), tl.float32), tl.zeros((4,), tl.int32)],
    )
    return al.extract_slice(merged, (0,), (4096,), (1,)), consumed


@triton.jit
def _mixed_outer2048(
    Q, K, QS, KS, Weight, Table, Visible, Ring, OutValues, OutIndices,
    TDots, TScores, TLocal, TState, TCounts,
    ROWS: tl.constexpr, COLUMNS: tl.constexpr, WIDTH: tl.constexpr,
    BLOCKS: tl.constexpr, SLOTS: tl.constexpr, TRACE: tl.constexpr,
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
                page = tl.load(Table + tl.minimum(start // 192, COLUMNS - 1), mask=start < WIDTH, other=0)
                token = page * 192 + start % 192 + n
                key = tl.load(K + token[:, None] * 128 + d[None, :], mask=start + n[:, None] < WIDTH, other=0)
                dot = tl.dot(query, tl.trans(key), out_dtype=tl.int32)
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
        for initial in tl.static_range(SLOTS):
            al.sync_block_set("vector", "cube", initial, pipe.PIPE_MTE3, pipe.PIPE_FIX)
        for outer in range(BLOCKS):
            vector_slot = outer % SLOTS
            al.sync_block_wait("cube", "vector", vector_slot, pipe.PIPE_FIX, pipe.PIPE_MTE2)
            for inner in range(32):
                start = outer * 2048 + inner * 64
                page = tl.load(Table + tl.minimum(start // 192, COLUMNS - 1), mask=start < WIDTH, other=0)
                token = page * 192 + start % 192 + n
                key_scale = tl.load(KS + token, mask=start + n < WIDTH, other=0).to(tl.float32)
                for local_query in range(2):
                    row = pid * 4 + al.sub_vec_id() * 2 + local_query
                    head = al.sub_vec_id() * 8 + local_query * 4 + h
                    src = Ring + (pid * SLOTS + vector_slot) * 16 * 2048 + inner * 64
                    dots = tl.load(src + head[:, None] * 2048 + n[None, :])
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
            tl.debug_barrier()
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
                if outer > 0:
                    old = tle.dsa.to_tensor(state_slot)
                    new_state, counts = _merge_equal2048(old, pairs)
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
    def __init__(self, values, width, *, slots, request_ids):
        from qsa_cann_serial_910b_trial import PreparedCANNSerial
        proof = PreparedCANNSerial(values, width, query_group=4, request_ids=request_ids)
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
                SLOTS=slots, use_bytecode=True, disable_auto_cv_work_space_manage=True,
                disable_auto_inject_block_sync=True, enable_auto_bind_sub_block=False,
                multibuffer=False, sync_solver=False, unit_flag=False))

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
