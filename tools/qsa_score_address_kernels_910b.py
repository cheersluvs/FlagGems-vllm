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

"""Test-only K-address probes derived from the production score AST."""

import triton
import triton.language as tl

PRODUCTION_AST_SHA256 = (
    "8f4c8d07542fe9ac0e83519f31ed95ca68009a68a4fa0ca40ba55eb903b4bad0"
)


@triton.jit
def _paged_score_910b_scalar_2d(
    Q,
    K,
    Table,
    Owners,
    Visible,
    Out,
    SQ0: tl.constexpr,
    SQ1: tl.constexpr,
    SQ2: tl.constexpr,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SK3: tl.constexpr,
    ST0: tl.constexpr,
    ST1: tl.constexpr,
    SO0: tl.constexpr,
    SO1: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    WIDTH: tl.constexpr,
    OFFSET: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(WIDTH, BLOCK_N)
    dims = tl.arange(0, 128)
    lanes = tl.arange(0, 16)
    for work in range(tl.program_id(0), ROWS * tiles, tl.num_programs(0)):
        local_row = work // tiles
        row = local_row + OFFSET
        columns = work % tiles * BLOCK_N + tl.arange(0, BLOCK_N)
        request = tl.load(Owners + row)
        visible = tl.load(Visible + row)
        live = (columns < WIDTH) & (columns < visible)
        tile_start = work % tiles * BLOCK_N
        physical = tl.load(
            Table + request * ST0 + tile_start // PAGE * ST1,
            mask=(tile_start < WIDTH) & (tile_start < visible),
            other=0,
        )
        base = K + physical.to(tl.int64) * SK0 + (tile_start % PAGE).to(tl.int64) * SK1
        offsets = tl.arange(0, BLOCK_N)[:, None] * SK1 + dims[None, :] * SK3
        keys = tl.load(base + offsets, mask=live[:, None], other=0.0)
        scores = tl.full((BLOCK_N,), 0.0, tl.float32)
        for group in tl.static_range(triton.cdiv(HEADS, 16)):
            heads = group * 16 + lanes
            query = tl.load(
                Q + row * SQ0 + heads[None, :] * SQ1 + dims[:, None] * SQ2,
                mask=heads[None, :] < HEADS,
                other=0.0,
            )
            dot = tl.dot(keys, query, out_dtype=tl.float32)
            positive = tl.where(heads[None, :] < HEADS, tl.maximum(dot, 0.0), 0.0)
            scores += tl.sum(positive, axis=1)
        tl.store(
            Out + local_row * SO0 + columns * SO1,
            tl.where(live, scores, -float("inf")),
            mask=columns < WIDTH,
        )


@triton.jit
def _paged_score_910b_scalar_flat(
    Q,
    K,
    Table,
    Owners,
    Visible,
    Out,
    SQ0: tl.constexpr,
    SQ1: tl.constexpr,
    SQ2: tl.constexpr,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SK3: tl.constexpr,
    ST0: tl.constexpr,
    ST1: tl.constexpr,
    SO0: tl.constexpr,
    SO1: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    WIDTH: tl.constexpr,
    OFFSET: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(WIDTH, BLOCK_N)
    dims = tl.arange(0, 128)
    lanes = tl.arange(0, 16)
    for work in range(tl.program_id(0), ROWS * tiles, tl.num_programs(0)):
        local_row = work // tiles
        row = local_row + OFFSET
        columns = work % tiles * BLOCK_N + tl.arange(0, BLOCK_N)
        request = tl.load(Owners + row)
        visible = tl.load(Visible + row)
        live = (columns < WIDTH) & (columns < visible)
        tile_start = work % tiles * BLOCK_N
        physical = tl.load(
            Table + request * ST0 + tile_start // PAGE * ST1,
            mask=(tile_start < WIDTH) & (tile_start < visible),
            other=0,
        )
        base = K + physical.to(tl.int64) * SK0 + (tile_start % PAGE).to(tl.int64) * SK1
        flat = tl.arange(0, BLOCK_N * 128)
        flat_n = flat // 128
        flat_d = flat % 128
        offsets = flat_n * SK1 + flat_d * SK3
        flat_live = (tile_start + flat_n < WIDTH) & (tile_start + flat_n < visible)
        keys = tl.load(base + offsets, mask=flat_live, other=0.0).reshape(
            (BLOCK_N, 128)
        )
        scores = tl.full((BLOCK_N,), 0.0, tl.float32)
        for group in tl.static_range(triton.cdiv(HEADS, 16)):
            heads = group * 16 + lanes
            query = tl.load(
                Q + row * SQ0 + heads[None, :] * SQ1 + dims[:, None] * SQ2,
                mask=heads[None, :] < HEADS,
                other=0.0,
            )
            dot = tl.dot(keys, query, out_dtype=tl.float32)
            positive = tl.where(heads[None, :] < HEADS, tl.maximum(dot, 0.0), 0.0)
            scores += tl.sum(positive, axis=1)
        tl.store(
            Out + local_row * SO0 + columns * SO1,
            tl.where(live, scores, -float("inf")),
            mask=columns < WIDTH,
        )
