# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Private fixed-choice head pairing; no production import of this module."""

import triton
import triton.language as tl

from flaggems_vllm.ops.qsa_select_paged_prefill import _fp8_bulk_tail_store


@triton.jit(do_not_specialize=["num_rows", "query_offset", "max_query_len"])
def _head_group_score_kernel(
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
    # Compatibility with captured FP8 configs; candidate groups heads, not rows.
    QUERY_PARTS: tl.constexpr = 1,
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

    tl.static_assert(NUM_HEADS == 4, "Head-group probe supports H4 only")
    HEADS_PADDED: tl.constexpr = 2
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

    # Each pair has two adjacent heads; the high pair loads heads2 and3.
    query_lanes = tl.arange(0, BLOCK_R * HEADS_PADDED)
    query_row_lanes = query_lanes // HEADS_PADDED
    query_rows = row_start + query_row_lanes
    heads = query_lanes % HEADS_PADDED
    # Pad the reduction dimension to the native dot instruction's minimum.
    BLOCK_D: tl.constexpr = max(32 if IS_FP8 else 16, triton.next_power_of_2(HEAD_DIM))
    dims = tl.arange(0, BLOCK_D)
    query_low = tl.load(
        q_ptr
        + query_rows[None, :] * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None],
        mask=(query_row_lanes[None, :] < TILE_R)
        & (query_rows[None, :] < request_end)
        & (heads[None, :] < NUM_HEADS)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    query_high = tl.load(
        q_ptr
        + query_rows[None, :] * stride_q_row
        + (heads[None, :] + 2) * stride_q_head
        + dims[:, None],
        mask=(query_row_lanes[None, :] < TILE_R)
        & (query_rows[None, :] < request_end)
        & (heads[None, :] < NUM_HEADS)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
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
        low = tl.dot(keys, query_low, out_dtype=tl.float32)
        low = tl.sum(tl.maximum(tl.reshape(low, (BLOCK_N, BLOCK_R, 2)), 0.0), axis=2)
        high = tl.dot(keys, query_high, out_dtype=tl.float32)
        high = tl.sum(tl.maximum(tl.reshape(high, (BLOCK_N, BLOCK_R, 2)), 0.0), axis=2)
        scores = low + high
        if IS_FP8:
            _fp8_bulk_tail_store(
                logits_ptr
                + (rows - query_offset)[None, :] * logits_width
                + columns[:, None],
                tl.where(live[:, None], scores, -float("inf")),
                valid_rows[None, :],
                columns[:, None],
                visible[None, :],
                logits_width,
            )
        else:
            tl.store(
                logits_ptr
                + (rows - query_offset)[None, :] * logits_width
                + columns[:, None],
                tl.where(live[:, None], scores, -float("inf")),
                mask=valid_rows[None, :]
                & (columns[:, None] < visible[None, :])
                & (columns[:, None] < logits_width),
            )
