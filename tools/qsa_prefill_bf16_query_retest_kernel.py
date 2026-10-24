# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Private full-head BF16 query-half retest; production imports no probe."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_rows", "query_offset", "max_query_len"])
def _query_split_score_kernel(
    q_ptr,
    cache_ptr,
    table_ptr,
    starts_ptr,
    visible_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_table_page,
    num_rows,
    query_offset,
    max_query_len,
    logits_width,
    num_requests,
    num_pages,
    table_width,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    HEAD_GROUP: tl.constexpr = False,
):
    request = tl.program_id(0)
    query_base = tl.load(starts_ptr)
    request_start = tl.maximum(tl.load(starts_ptr + request) - query_base, query_offset)
    request_end = tl.minimum(
        tl.load(starts_ptr + request + 1) - query_base, query_offset + num_rows
    )
    row_start = request_start + tl.program_id(1) * TILE_R
    if row_start >= request_end:
        return

    tl.static_assert(not IS_FP8 and NUM_HEADS == 4 and HEAD_DIM == 128)
    tl.static_assert(TILE_R == 64 and K_TILES == 16 and not HEAD_GROUP)
    HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires N>=16 even when TILE_R * HEADS_PADDED is smaller.
    # Fake rows do not read queries/metadata or write outside this query tile.
    BLOCK_R: tl.constexpr = max(TILE_R, triton.cdiv(16, HEADS_PADDED))
    row_lanes = tl.arange(0, BLOCK_R)
    rows = row_start + row_lanes
    valid_rows = (row_lanes < TILE_R) & (rows < request_end)
    visible = tl.load(visible_ptr + rows, mask=valid_rows, other=0)
    max_visible = tl.minimum(tl.max(visible, axis=0), logits_width)
    first_tile = tl.program_id(2) * K_TILES
    if first_tile * BLOCK_N >= max_visible:
        return
    last_tile = tl.minimum(first_tile + K_TILES, tl.cdiv(max_visible, BLOCK_N))

    HALF_R: tl.constexpr = TILE_R // 2
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    dims = tl.arange(0, BLOCK_D)
    half_lanes = tl.arange(0, HALF_R * HEADS_PADDED)
    half_rows = half_lanes // HEADS_PADDED
    half_heads = half_lanes % HEADS_PADDED
    query_left = tl.load(
        q_ptr
        + (row_start + 0 + half_rows)[None, :] * stride_q_row
        + half_heads[None, :] * stride_q_head
        + dims[:, None],
        mask=((row_start + 0 + half_rows)[None, :] < request_end)
        & (half_heads[None, :] < NUM_HEADS)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    query_right = tl.load(
        q_ptr
        + (row_start + HALF_R + half_rows)[None, :] * stride_q_row
        + half_heads[None, :] * stride_q_head
        + dims[:, None],
        mask=((row_start + HALF_R + half_rows)[None, :] < request_end)
        & (half_heads[None, :] < NUM_HEADS)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    visible_left, visible_right = tl.split(tl.trans(tl.reshape(visible, (2, HALF_R))))
    column_lanes = tl.arange(0, BLOCK_N)
    for tile in range(first_tile, last_tile):
        columns = tile * BLOCK_N + column_lanes
        logical_page = columns // PAGE_SIZE
        page_offset = columns % PAGE_SIZE
        live = (columns < max_visible) & (logical_page < table_width)
        physical_page = tl.load(
            table_ptr + request * stride_table_req + logical_page * stride_table_page,
            mask=live,
            other=0,
        )
        live &= (physical_page >= 0) & (physical_page < num_pages)
        keys = tl.load(
            cache_ptr
            + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        )
        scores_left = tl.dot(keys, query_left, out_dtype=tl.float32)
        scores_left = tl.reshape(scores_left, (BLOCK_N, HALF_R, HEADS_PADDED))
        scores_left = tl.maximum(scores_left, 0.0)
        scores_left = tl.sum(scores_left, axis=2)
        rows_left = row_start + 0 + tl.arange(0, HALF_R)
        tl.store(
            logits_ptr
            + (rows_left - query_offset)[None, :] * logits_width
            + columns[:, None],
            tl.where(live[:, None], scores_left, -float("inf")),
            mask=(rows_left[None, :] < request_end)
            & (columns[:, None] < visible_left[None, :])
            & (columns[:, None] < logits_width),
        )
        scores_right = tl.dot(keys, query_right, out_dtype=tl.float32)
        scores_right = tl.reshape(scores_right, (BLOCK_N, HALF_R, HEADS_PADDED))
        scores_right = tl.maximum(scores_right, 0.0)
        scores_right = tl.sum(scores_right, axis=2)
        rows_right = row_start + HALF_R + tl.arange(0, HALF_R)
        tl.store(
            logits_ptr
            + (rows_right - query_offset)[None, :] * logits_width
            + columns[:, None],
            tl.where(live[:, None], scores_right, -float("inf")),
            mask=(rows_right[None, :] < request_end)
            & (columns[:, None] < visible_right[None, :])
            & (columns[:, None] < logits_width),
        )
