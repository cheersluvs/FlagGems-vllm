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

"""FP32 serial structural control: N64 score, sort2048, persistent Top2048."""

from functools import partial

import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle
import triton.language.extra.cann.extension as al
from probe_qsa_int8_mixed_pipeline_910b import SORT_IMPL_BASE

@triton.jit
def _score_outer_n64(
    Q,
    K,
    QS,
    KS,
    Weights,
    Table,
    Visible,
    Out,
    RawDot,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    COLUMNS: tl.constexpr,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASK_RAW_DOT: tl.constexpr,
    WRITE_RAW_DOT: tl.constexpr,
    KEY_START: tl.constexpr, SCAN_WIDTH: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(WIDTH, BLOCK_N)
    dims = tl.arange(0, 128)
    lanes = tl.arange(0, 16)
    for work in range(tl.program_id(0), ROWS * tiles, tl.num_programs(0)):
        row = work // tiles
        local_columns = (work % tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
        columns = KEY_START + local_columns
        visible = tl.load(Visible + row)
        live = (columns < SCAN_WIDTH) & (columns < visible)
        physical = tl.load(Table + row * COLUMNS + columns // PAGE, mask=live, other=0)
        token = physical.to(tl.int64) * PAGE + columns % PAGE
        keys = tl.load(
            K + token[:, None] * 128 + dims[None, :],
            mask=live[:, None],
            other=0,
        )
        key_scale = tl.load(KS + token, mask=live, other=0).to(tl.float32)
        scores = tl.full((BLOCK_N,), 0, tl.float32)
        for group in tl.static_range(triton.cdiv(HEADS, 16)):
            heads = group * 16 + lanes
            query = tl.load(
                Q + (row * HEADS + heads[None, :]) * 128 + dims[:, None],
                mask=heads[None, :] < HEADS,
                other=0,
            )
            dot = tl.dot(keys, query, out_dtype=tl.int32)
            if WRITE_RAW_DOT:
                if MASK_RAW_DOT:
                    debug_dot = tl.where(live[:, None], dot, 0)
                else:
                    debug_dot = dot
                tl.store(
                    RawDot + (row * WIDTH + local_columns[:, None]) * HEADS + heads[None, :],
                    debug_dot,
                    mask=(local_columns[:, None] < WIDTH) & (heads[None, :] < HEADS),
                )
            query_scale = tl.load(
                QS + row * HEADS + heads, mask=heads < HEADS, other=0
            ).to(tl.float32)
            weights = tl.load(
                Weights + row * HEADS + heads, mask=heads < HEADS, other=0
            ).to(tl.float32)
            dequant = dot.to(tl.float32) * query_scale[None, :] * key_scale[:, None]
            scores += tl.sum(tl.maximum(dequant, 0.0) * weights[None, :], axis=1)
        tl.store(
            Out + row * WIDTH + local_columns,
            tl.where(live, scores, -float("inf")),
            mask=local_columns < WIDTH,
        )


@triton.jit
def _score_outer_shared_n64(
    Q,
    K,
    QS,
    KS,
    Weights,
    Table,
    Visible,
    Out,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KEY_TILES: tl.constexpr,
    QUERY_GROUP: tl.constexpr, KEY_START: tl.constexpr, SCAN_WIDTH: tl.constexpr,
    WRITE_RAW_DOT: tl.constexpr, RawDot,
):
    tl.static_assert(BLOCK_N == 64 and PAGE == 192)
    groups: tl.constexpr = triton.cdiv(ROWS, QUERY_GROUP)
    tiles: tl.constexpr = triton.cdiv(WIDTH, BLOCK_N)
    dims = tl.arange(0, 128)
    lanes = tl.arange(0, QUERY_GROUP * 4)
    queries = tl.arange(0, QUERY_GROUP)
    blocks: tl.constexpr = triton.cdiv(tiles, KEY_TILES)
    for unit in range(tl.program_id(0), groups * blocks, tl.num_programs(0)):
        group = unit // blocks
        first_tile = unit % blocks * KEY_TILES
        base = group * QUERY_GROUP
        rows = base + queries
        flat_heads = base * 4 + lanes
        query = tl.load(
            Q + flat_heads[:, None] * 128 + dims[None, :],
            mask=flat_heads[:, None] < ROWS * 4,
            other=0,
        )
        query_scale = tl.load(QS + flat_heads, mask=flat_heads < ROWS * 4, other=0).to(
            tl.float32
        )
        weights = tl.load(Weights + flat_heads, mask=flat_heads < ROWS * 4, other=0).to(
            tl.float32
        )
        visible = tl.load(Visible + rows, mask=rows < ROWS, other=0)
        max_visible = tl.max(visible, axis=0)
        for local_tile in range(KEY_TILES):
            tile = first_tile + local_tile
            local_columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
            columns = KEY_START + local_columns
            key_live = (columns < SCAN_WIDTH) & (columns < max_visible)
            start = KEY_START + tile * BLOCK_N
            page = start // PAGE
            offset = start % PAGE
            physical = tl.load(
                Table + base * COLUMNS + tl.minimum(page, COLUMNS - 1),
                mask=(start < SCAN_WIDTH) & (start < max_visible) & (page < COLUMNS),
                other=0,
            )
            token = physical.to(tl.int64) * PAGE + offset + tl.arange(0, BLOCK_N)
            keys = tl.load(
                K + token[:, None] * 128 + dims[None, :],
                mask=key_live[:, None],
                other=0,
            )
            key_scale = tl.load(KS + token, mask=key_live, other=0).to(tl.float32)
            dot = tl.dot(query, tl.trans(keys), out_dtype=tl.int32)
            dequant = dot.to(tl.float32) * query_scale[:, None] * key_scale[None, :]
            weighted = tl.maximum(dequant, 0.0) * weights[:, None]
            scores = tl.sum(tl.reshape(weighted, (QUERY_GROUP, 4, BLOCK_N)), axis=1)
            live = (columns[None, :] < SCAN_WIDTH) & (columns[None, :] < visible[:, None])
            if WRITE_RAW_DOT:
                raw = tl.reshape(dot, (QUERY_GROUP, 4, BLOCK_N))
                heads = tl.arange(0, 4)
                raw_out = RawDot + ((rows[:, None, None] * WIDTH + local_columns[None, None, :]) * 4 + heads[None, :, None])
                tl.store(raw_out, tl.where(live[:, None, :], raw, 0),
                         mask=(rows[:, None, None] < ROWS) & (local_columns[None, None, :] < WIDTH))
            tl.store(
                Out + rows[:, None] * WIDTH + local_columns[None, :],
                tl.where(live, scores, -float("inf")),
                mask=(local_columns[None, :] < WIDTH) & (rows[:, None] < ROWS),
            )


@triton.jit
def _sort_outer(Source, Pairs, Visible, STRIDE: tl.constexpr, START: tl.constexpr,
                WIDTH: tl.constexpr, FULL_SOURCE: tl.constexpr):
    row = tl.program_id(0)
    n = tl.arange(0, 2048)
    index = START + n
    address = row * STRIDE + (index if FULL_SOURCE else n)
    live = (index < WIDTH) & (index < tl.load(Visible + row))
    score = tl.load(Source + address, mask=live, other=-float("inf"))
    pairs = tle.dsa.ascend.raw(
        "sort_1d_pack", score, tl.zeros((8192,), tl.float32),
        True, 2048, START, SORT_IMPL_BASE, out=tl.zeros((4096,), tl.float32),
    )
    tl.store(Pairs + row * 4096 + tl.arange(0, 4096), pairs)


@triton.jit
def _merge_outer(Old, Local, New, Counts):
    row = tl.program_id(0)
    words = tl.arange(0, 4096)
    old = tl.load(Old + row * 4096 + words)
    local = tl.load(Local + row * 4096 + words)
    packed = tl.zeros((16384,), tl.float32)
    packed = tle.dsa.insert_slice(packed, old, (0,), (4096,), (1,))
    packed = tle.dsa.insert_slice(packed, local, (4096,), (4096,), (1,))
    merged, consumed = tle.dsa.ascend.raw(
        "merge_exhaust_sort4", packed, 2, 0, 2048, 4096, 6144,
        2048, 2048, 0, 0,
        out=[tl.zeros((8192,), tl.float32), tl.zeros((4,), tl.int32)],
    )
    prefix = al.extract_slice(merged, (0,), (4096,), (1,))
    tl.store(New + row * 4096 + words, prefix)
    tl.store(Counts + row * 4 + tl.arange(0, 4), consumed)


@triton.jit
def _finish_outer(State, Visible, Values, Indices):
    row = tl.program_id(0)
    pairs = tl.load(State + row * 4096 + tl.arange(0, 1024))
    values, indices = tle.dsa.ascend.raw(
        "unpack_sort", pairs, 512,
        out=[tl.zeros((512,), tl.float32), tl.zeros((512,), tl.int32)],
    )
    rank = tl.arange(0, 512)
    live = rank < tl.load(Visible + row)
    tl.store(Values + row * 512 + rank, tl.where(live, values, -float("inf")))
    tl.store(Indices + row * 512 + rank, tl.where(live, indices, -1))


class PreparedCANNSerial:
    """Kernel-separated serial control; buffers and launch sequence are explicit."""

    def __init__(self, values, width, *, query_group=1, request_ids=None):
        self.values, self.width = values, width
        self.rows = values["q"].shape[0]
        if query_group not in (1, 4, 16):
            raise ValueError("Query group must be 1, 4, or 16")
        self.query_group = query_group
        self.segments = []
        if query_group != 1:
            if request_ids is None or len(request_ids) != self.rows:
                raise ValueError("Explicit request ownership is required for shared K")
            mapping = values["row_table"].cpu()
            begin = 0
            while begin < self.rows:
                end = begin + 1
                while end < self.rows and request_ids[end] == request_ids[begin] and torch.equal(mapping[end], mapping[begin]):
                    end += 1
                group = 4 if query_group == 4 or end - begin <= 4 else 16
                self.segments.append((begin, end - begin, group))
                begin = end
        self.starts = tuple(range(0, width, 2048))
        device = values["q"].device
        self.score_buffer = torch.empty((self.rows, 2048), device=device, dtype=torch.float32)
        self.raw_dot = torch.empty((self.rows, 2048, 4), device=device, dtype=torch.int32)
        self.states = [torch.empty((self.rows, 4096), device=device, dtype=torch.float32) for _ in range(2)]
        self.local = torch.empty_like(self.states[0])
        self.counts = torch.empty((self.rows, 4), device=device, dtype=torch.int32)
        self.output = torch.empty((self.rows, 512), device=device, dtype=torch.int32)
        self.output_values = torch.empty((self.rows, 512), device=device, dtype=torch.float32)
        self.compiled = {}

    def score_block(self, start, debug=False):
        v = self.values
        if self.query_group != 1:
            compiled = None
            for first, count, group in self.segments:
                compiled = _score_outer_shared_n64[(min(20, triton.cdiv(count, group) * 4),)](
                    v["q"][first:first + count], v["cache"], v["qs"][first:first + count], v["ks"],
                    v["weights"][first:first + count], v["row_table"][first:first + count], v["visible"][first:first + count],
                    self.score_buffer[first:first + count], ROWS=count, COLUMNS=v["row_table"].shape[1],
                    WIDTH=2048, PAGE=192, BLOCK_N=64, KEY_TILES=8,
                    QUERY_GROUP=group, KEY_START=start, SCAN_WIDTH=self.width,
                    WRITE_RAW_DOT=debug, RawDot=self.raw_dot[first:first + count], num_warps=4,
                )
            return compiled
        return _score_outer_n64[(min(20, self.rows * 32),)](
            v["q"], v["cache"], v["qs"], v["ks"], v["weights"],
            v["row_table"], v["visible"], self.score_buffer, self.raw_dot,
            HEADS=4, PAGE=192, COLUMNS=v["row_table"].shape[1], ROWS=self.rows,
            WIDTH=2048, BLOCK_N=64, MASK_RAW_DOT=True, WRITE_RAW_DOT=debug,
            KEY_START=start, SCAN_WIDTH=self.width, num_warps=4,
        )

    def score_only(self):
        for start in self.starts:
            self.score_block(start)

    def run(self, full_source=None, check=None):
        for block, start in enumerate(self.starts):
            if full_source is None:
                self.compiled["score"] = self.score_block(start, debug=check is not None)
                if check:
                    check("score", start, self.score_buffer, self.raw_dot)
            target = self.states[0] if block == 0 else self.local
            self.compiled["sort"] = _sort_outer[(self.rows,)](
                self.score_buffer if full_source is None else full_source, target,
                self.values["visible"], STRIDE=2048 if full_source is None else self.width,
                START=start, WIDTH=self.width, FULL_SOURCE=full_source is not None,
            )
            if check:
                check("sort", start, target, None)
            if block:
                self.compiled["merge"] = _merge_outer[(self.rows,)](
                    self.states[(block - 1) % 2], self.local, self.states[block % 2], self.counts,
                )
            if check:
                check("state", start, self.states[block % 2], self.counts if block else None)
        self.compiled["finish"] = _finish_outer[(self.rows,)](
            self.states[(len(self.starts) - 1) % 2], self.values["visible"], self.output_values, self.output,
        )
        return self.output

    def __call__(self):
        return self.run()
