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

"""Four-query shared-cache score diagnostic, restricted to validated row maps."""

import bisect

import triton
import triton.language as tl


def validate_shared_groups(table_rows, starts, first):
    """Validate CPU fixture metadata; no NPU tensor contents are inspected."""
    if (
        not table_rows
        or not table_rows[0]
        or any(len(row) != len(table_rows[0]) for row in table_rows)
    ):
        raise ValueError("Nonempty rectangular CPU page table required")
    if (
        len(starts) < 2
        or starts[0] != 0
        or any(b < a for a, b in zip(starts, starts[1:]))
        or first < 0
        or first + len(table_rows) > starts[-1]
    ):
        raise ValueError("Invalid CPU request boundaries")
    for base in range(0, len(table_rows), 4):
        anchor = table_rows[base]
        owner = bisect.bisect_right(starts, first + base) - 1
        for row in range(base, min(base + 4, len(table_rows))):
            if (
                table_rows[row] != anchor
                or bisect.bisect_right(starts, first + row) - 1 != owner
            ):
                raise NotImplementedError(
                    f"Query group crosses a request or page mapping at row {first+row}"
                )
    return dict(
        groups=triton.cdiv(len(table_rows), 4),
        all_group_maps_equal=True,
        same_request_per_group=True,
    )


@triton.jit
def _qsa_query_pack4_score(
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
    SCALE_PRODUCT: tl.constexpr,
    STORE_MODE: tl.constexpr = 0,
    FLAT_QUERY: tl.constexpr = False,
    QUERY_LEFT: tl.constexpr = False,
    LATE_KEY_SCALE: tl.constexpr = False,
    COMBINE_QUERY_WEIGHT: tl.constexpr = False,
    CUBE_REDUCE: tl.constexpr = False,
):
    tiles: tl.constexpr = triton.cdiv(WIDTH, BLOCK_N)
    groups: tl.constexpr = triton.cdiv(ROWS, 4)
    dims = tl.arange(0, 128)
    lanes = tl.arange(0, 16)
    queries = tl.arange(0, 4)
    for work in range(tl.program_id(0), groups * tiles, tl.num_programs(0)):
        base = (work // tiles) * 4
        rows = base + queries
        columns = (work % tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
        visible = tl.load(Visible + rows, mask=rows < ROWS, other=0)
        max_visible = tl.max(visible, axis=0)
        key_live = (columns < WIDTH) & (columns < max_visible)
        physical = tl.load(
            Table + base * COLUMNS + columns // PAGE, mask=key_live, other=0
        )
        token = physical.to(tl.int64) * PAGE + columns % PAGE
        keys = tl.load(
            K + token[:, None] * 128 + dims[None, :], mask=key_live[:, None], other=0
        )
        key_scale = tl.load(KS + token, mask=key_live, other=0).to(tl.float32)
        query_rows = base + lanes // 4
        heads = lanes % 4
        if FLAT_QUERY:
            # Index consecutive query/head rows without division in Q's mask.
            flat_heads = base * 4 + lanes
            query = tl.load(
                Q + flat_heads[None, :] * 128 + dims[:, None],
                mask=flat_heads[None, :] < ROWS * 4,
                other=0,
            )
        else:
            query = tl.load(
                Q + (query_rows[None, :] * 4 + heads[None, :]) * 128 + dims[:, None],
                mask=query_rows[None, :] < ROWS,
                other=0,
            )
        if QUERY_LEFT:
            dot = tl.dot(tl.trans(query), tl.trans(keys), out_dtype=tl.int32)
        else:
            dot = tl.dot(keys, query, out_dtype=tl.int32)
        if CUBE_REDUCE:
            flat_heads = base * 4 + lanes
            query_scale = tl.load(
                QS + flat_heads, mask=flat_heads < ROWS * 4, other=0
            ).to(tl.float32)
            weights = tl.load(
                Weights + flat_heads, mask=flat_heads < ROWS * 4, other=0
            ).to(tl.float32)
        else:
            query_scale = tl.load(
                QS + query_rows * 4 + heads, mask=query_rows < ROWS, other=0
            ).to(tl.float32)
            weights = tl.load(
                Weights + query_rows * 4 + heads, mask=query_rows < ROWS, other=0
            ).to(tl.float32)
        if QUERY_LEFT:
            if CUBE_REDUCE:
                # Match the native raw-dot scaling; restore score units at output.
                activated = tl.maximum(dot.to(tl.float32) * (1.0 / 1024.0), 0.0).to(
                    tl.float16
                )
                coefficient = (query_scale * weights).to(tl.float16)
                output_queries = tl.arange(0, 16)
                coefficient_matrix = tl.where(
                    (output_queries[:, None] < 4)
                    & (lanes[None, :] >= output_queries[:, None] * 4)
                    & (lanes[None, :] < output_queries[:, None] * 4 + 4),
                    coefficient[None, :],
                    0.0,
                ).to(tl.float16)
                summed = tl.dot(coefficient_matrix, activated, out_dtype=tl.float32)
                scores = (summed * key_scale[None, :]) * 1024.0
                output_rows = base + output_queries
                output_visible = tl.sum(
                    tl.where(
                        output_queries[:, None] == queries[None, :], visible[None, :], 0
                    ),
                    axis=1,
                )
                live = (columns[None, :] < WIDTH) & (
                    columns[None, :] < output_visible[:, None]
                )
                tl.store(
                    Out + output_rows[:, None] * WIDTH + columns[None, :],
                    tl.where(live, scores, -float("inf")),
                    mask=(output_queries[:, None] < 4)
                    & (output_rows[:, None] < ROWS)
                    & (columns[None, :] < WIDTH),
                )
            else:
                if COMBINE_QUERY_WEIGHT:
                    # Caller proves finite nonnegative query scales for reassociation.
                    coefficient = query_scale * weights
                    weighted = (
                        tl.maximum(dot.to(tl.float32) * key_scale[None, :], 0.0)
                        * coefficient[:, None]
                    )
                else:
                    if LATE_KEY_SCALE:
                        # Caller proves finite nonnegative key scales before this probe.
                        dequant = dot.to(tl.float32) * query_scale[:, None]
                    elif SCALE_PRODUCT:
                        dequant = dot.to(tl.float32) * (
                            query_scale[:, None] * key_scale[None, :]
                        )
                    else:
                        dequant = (
                            dot.to(tl.float32)
                            * query_scale[:, None]
                            * key_scale[None, :]
                        )
                    weighted = tl.maximum(dequant, 0.0) * weights[:, None]
                scores = tl.sum(tl.reshape(weighted, (4, 4, BLOCK_N)), axis=1)
                if LATE_KEY_SCALE:
                    scores = scores * key_scale[None, :]
                live = (columns[None, :] < WIDTH) & (
                    columns[None, :] < visible[:, None]
                )
                tl.store(
                    Out + rows[:, None] * WIDTH + columns[None, :],
                    tl.where(live, scores, -float("inf")),
                    mask=(columns[None, :] < WIDTH) & (rows[:, None] < ROWS),
                )

        else:
            if SCALE_PRODUCT:
                dequant = dot.to(tl.float32) * (
                    query_scale[None, :] * key_scale[:, None]
                )
            else:
                dequant = dot.to(tl.float32) * query_scale[None, :] * key_scale[:, None]
            weighted = tl.maximum(dequant, 0.0) * weights[None, :]
            if STORE_MODE == 2:
                # Keep four independent FP32 reductions and contiguous output rows.
                for local in tl.static_range(4):
                    scores = tl.sum(
                        tl.where(lanes[None, :] // 4 == local, weighted, 0.0), axis=1
                    )
                    row = base + local
                    row_visible = tl.load(Visible + row, mask=row < ROWS, other=0)
                    live = (columns < WIDTH) & (columns < row_visible)
                    tl.store(
                        Out + row * WIDTH + columns,
                        tl.where(live, scores, -float("inf")),
                        mask=(columns < WIDTH) & (row < ROWS),
                    )
            else:
                scores = tl.sum(tl.reshape(weighted, (BLOCK_N, 4, 4)), axis=2)
                if STORE_MODE == 1:
                    scores = tl.trans(scores)
                    live = (columns[None, :] < WIDTH) & (
                        columns[None, :] < visible[:, None]
                    )
                    tl.store(
                        Out + rows[:, None] * WIDTH + columns[None, :],
                        tl.where(live, scores, -float("inf")),
                        mask=(columns[None, :] < WIDTH) & (rows[:, None] < ROWS),
                    )
                else:
                    live = (columns[:, None] < WIDTH) & (
                        columns[:, None] < visible[None, :]
                    )
                    tl.store(
                        Out + rows[None, :] * WIDTH + columns[:, None],
                        tl.where(live, scores, -float("inf")),
                        mask=(columns[:, None] < WIDTH) & (rows[None, :] < ROWS),
                    )


@triton.jit
def _qsa_qk_group_reuse_score(
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
):
    groups: tl.constexpr = triton.cdiv(ROWS, 4)
    tiles: tl.constexpr = triton.cdiv(WIDTH, BLOCK_N)
    dims = tl.arange(0, 128)
    lanes = tl.arange(0, 16)
    queries = tl.arange(0, 4)
    for group in range(tl.program_id(0), groups, tl.num_programs(0)):
        base = group * 4
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
        for tile in range(tiles):
            columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
            key_live = (columns < WIDTH) & (columns < max_visible)
            physical = tl.load(
                Table + base * COLUMNS + columns // PAGE, mask=key_live, other=0
            )
            token = physical.to(tl.int64) * PAGE + columns % PAGE
            keys = tl.load(
                K + token[:, None] * 128 + dims[None, :],
                mask=key_live[:, None],
                other=0,
            )
            key_scale = tl.load(KS + token, mask=key_live, other=0).to(tl.float32)
            dot = tl.dot(query, tl.trans(keys), out_dtype=tl.int32)
            dequant = dot.to(tl.float32) * query_scale[:, None] * key_scale[None, :]
            weighted = tl.maximum(dequant, 0.0) * weights[:, None]
            scores = tl.sum(tl.reshape(weighted, (4, 4, BLOCK_N)), axis=1)
            live = (columns[None, :] < WIDTH) & (columns[None, :] < visible[:, None])
            tl.store(
                Out + rows[:, None] * WIDTH + columns[None, :],
                tl.where(live, scores, -float("inf")),
                mask=(columns[None, :] < WIDTH) & (rows[:, None] < ROWS),
            )


@triton.jit
def _qsa_qk_key_group_score(
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
):
    groups: tl.constexpr = triton.cdiv(ROWS, 4)
    tiles: tl.constexpr = triton.cdiv(WIDTH, BLOCK_N)
    dims = tl.arange(0, 128)
    lanes = tl.arange(0, 16)
    queries = tl.arange(0, 4)
    blocks: tl.constexpr = triton.cdiv(tiles, KEY_TILES)
    for unit in range(tl.program_id(0), groups * blocks, tl.num_programs(0)):
        group = unit // blocks
        first_tile = (unit % blocks) * KEY_TILES
        base = group * 4
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
            columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
            key_live = (columns < WIDTH) & (columns < max_visible)
            physical = tl.load(
                Table + base * COLUMNS + columns // PAGE, mask=key_live, other=0
            )
            token = physical.to(tl.int64) * PAGE + columns % PAGE
            keys = tl.load(
                K + token[:, None] * 128 + dims[None, :],
                mask=key_live[:, None],
                other=0,
            )
            key_scale = tl.load(KS + token, mask=key_live, other=0).to(tl.float32)
            dot = tl.dot(query, tl.trans(keys), out_dtype=tl.int32)
            dequant = dot.to(tl.float32) * query_scale[:, None] * key_scale[None, :]
            weighted = tl.maximum(dequant, 0.0) * weights[:, None]
            scores = tl.sum(tl.reshape(weighted, (4, 4, BLOCK_N)), axis=1)
            live = (columns[None, :] < WIDTH) & (columns[None, :] < visible[:, None])
            tl.store(
                Out + rows[:, None] * WIDTH + columns[None, :],
                tl.where(live, scores, -float("inf")),
                mask=(columns[None, :] < WIDTH) & (rows[:, None] < ROWS),
            )


@triton.jit
def _qsa_qk_key_prefetch_score(
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
):
    groups: tl.constexpr = triton.cdiv(ROWS, 4)
    tiles: tl.constexpr = triton.cdiv(WIDTH, BLOCK_N)
    dims = tl.arange(0, 128)
    lanes = tl.arange(0, 16)
    queries = tl.arange(0, 4)
    blocks: tl.constexpr = triton.cdiv(tiles, KEY_TILES)
    for unit in range(tl.program_id(0), groups * blocks, tl.num_programs(0)):
        group = unit // blocks
        first_tile = (unit % blocks) * KEY_TILES
        base = group * 4
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
        columns = first_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        key_live = (columns < WIDTH) & (columns < max_visible)
        physical = tl.load(
            Table + base * COLUMNS + columns // PAGE, mask=key_live, other=0
        )
        token = physical.to(tl.int64) * PAGE + columns % PAGE
        keys = tl.load(
            K + token[:, None] * 128 + dims[None, :],
            mask=key_live[:, None],
            other=0,
        )
        key_scale = tl.load(KS + token, mask=key_live, other=0).to(tl.float32)
        for local_tile in range(KEY_TILES):
            tile = first_tile + local_tile
            columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
            next_columns = (tile + 1) * BLOCK_N + tl.arange(0, BLOCK_N)
            next_key_live = (
                (next_columns < WIDTH)
                & (next_columns < max_visible)
                & (local_tile + 1 < KEY_TILES)
            )
            next_physical = tl.load(
                Table + base * COLUMNS + next_columns // PAGE,
                mask=next_key_live,
                other=0,
            )
            next_token = next_physical.to(tl.int64) * PAGE + next_columns % PAGE
            next_keys = tl.load(
                K + next_token[:, None] * 128 + dims[None, :],
                mask=next_key_live[:, None],
                other=0,
            )
            next_key_scale = tl.load(KS + next_token, mask=next_key_live, other=0).to(
                tl.float32
            )
            dot = tl.dot(query, tl.trans(keys), out_dtype=tl.int32)
            dequant = dot.to(tl.float32) * query_scale[:, None] * key_scale[None, :]
            weighted = tl.maximum(dequant, 0.0) * weights[:, None]
            scores = tl.sum(tl.reshape(weighted, (4, 4, BLOCK_N)), axis=1)
            live = (columns[None, :] < WIDTH) & (columns[None, :] < visible[:, None])
            tl.store(
                Out + rows[:, None] * WIDTH + columns[None, :],
                tl.where(live, scores, -float("inf")),
                mask=(columns[None, :] < WIDTH) & (rows[:, None] < ROWS),
            )
            keys = next_keys
            key_scale = next_key_scale


@triton.jit
def _qsa_qk_page_segment_score(
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
):
    tl.static_assert(BLOCK_N == 128 and PAGE == 192)
    groups: tl.constexpr = triton.cdiv(ROWS, 4)
    tiles: tl.constexpr = triton.cdiv(WIDTH, BLOCK_N)
    dims = tl.arange(0, 128)
    lanes = tl.arange(0, 16)
    queries = tl.arange(0, 4)
    blocks: tl.constexpr = triton.cdiv(tiles, KEY_TILES)
    for unit in range(tl.program_id(0), groups * blocks, tl.num_programs(0)):
        group = unit // blocks
        first_tile = (unit % blocks) * KEY_TILES
        base = group * 4
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
            columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
            key_live = (columns < WIDTH) & (columns < max_visible)
            # N128/page192 means at most two contiguous page fragments.
            start = tile * BLOCK_N
            page = start // PAGE
            offset = start % PAGE
            lanes_n = tl.arange(0, BLOCK_N)
            first_count = tl.minimum(BLOCK_N, PAGE - offset)
            first_live = key_live & (lanes_n < first_count)
            second_live = key_live & (lanes_n >= first_count)
            physical0 = tl.load(
                Table + base * COLUMNS + page,
                mask=(start < WIDTH) & (start < max_visible) & (page < COLUMNS),
                other=0,
            )
            physical1 = tl.load(
                Table + base * COLUMNS + page + 1,
                mask=(first_count < BLOCK_N)
                & (start + first_count < WIDTH)
                & (start + first_count < max_visible)
                & (page + 1 < COLUMNS),
                other=0,
            )
            token0 = physical0.to(tl.int64) * PAGE + offset + lanes_n
            token1 = physical1.to(tl.int64) * PAGE + offset + lanes_n - PAGE
            keys0 = tl.load(
                K + token0[:, None] * 128 + dims[None, :],
                mask=first_live[:, None],
                other=0,
            )
            keys1 = tl.load(
                K + token1[:, None] * 128 + dims[None, :],
                mask=second_live[:, None],
                other=0,
            )
            scale0 = tl.load(KS + token0, mask=first_live, other=0)
            scale1 = tl.load(KS + token1, mask=second_live, other=0)
            keys = tl.where(lanes_n[:, None] < first_count, keys0, keys1)
            key_scale = tl.where(lanes_n < first_count, scale0, scale1).to(tl.float32)
            dot = tl.dot(query, tl.trans(keys), out_dtype=tl.int32)
            dequant = dot.to(tl.float32) * query_scale[:, None] * key_scale[None, :]
            weighted = tl.maximum(dequant, 0.0) * weights[:, None]
            scores = tl.sum(tl.reshape(weighted, (4, 4, BLOCK_N)), axis=1)
            live = (columns[None, :] < WIDTH) & (columns[None, :] < visible[:, None])
            tl.store(
                Out + rows[:, None] * WIDTH + columns[None, :],
                tl.where(live, scores, -float("inf")),
                mask=(columns[None, :] < WIDTH) & (rows[:, None] < ROWS),
            )
