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

    def __init__(self, values, width):
        self.values, self.width = values, width
        self.rows = values["q"].shape[0]
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
