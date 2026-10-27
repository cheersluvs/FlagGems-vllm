import triton
import triton.language as tl

@triton.jit(do_not_specialize=["num_rows", "query_offset", "max_query_len"])
def _qsa_prefill_compact_score_kernel(
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
):
    # Monotone slot boundaries include one padding slot per request. This
    # avoids the rectangular requests * max_query_len grid without a prefix
    # scan, host metadata reads, atomics or an extra launch/workspace.
    slot = tl.program_id(0)
    query_base = tl.load(starts_ptr)
    low, high = 0, num_requests
    while low + 1 < high:
        middle = (low + high) // 2
        start = tl.load(starts_ptr + middle) - query_base - query_offset
        start = tl.minimum(tl.maximum(start, 0), num_rows)
        boundary = start // TILE_R + middle
        if boundary <= slot:
            low = middle
        else:
            high = middle
    request = low
    start = tl.load(starts_ptr + request) - query_base - query_offset
    clipped_start = tl.minimum(tl.maximum(start, 0), num_rows)
    first_slot = clipped_start // TILE_R + request
    request_start = query_offset + clipped_start
    request_end = tl.minimum(
        tl.load(starts_ptr + request + 1) - query_base, query_offset + num_rows
    )
    row_start = request_start + (slot - first_slot) * TILE_R
    if row_start >= request_end:
        return

    HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires M>=16 even when TILE_R * HEADS_PADDED is smaller.
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

    # Padded heads are adjacent in the dot result. Mask them before reduction.
    query_lanes = tl.arange(0, BLOCK_R * HEADS_PADDED)
    query_row_lanes = query_lanes // HEADS_PADDED
    query_rows = row_start + query_row_lanes
    heads = query_lanes % HEADS_PADDED
    # Pad the reduction dimension to the native dot instruction's minimum.
    BLOCK_D: tl.constexpr = max(32 if IS_FP8 else 16, triton.next_power_of_2(HEAD_DIM))
    dims = tl.arange(0, BLOCK_D)
    query = tl.load(
        q_ptr
        + query_rows[:, None] * stride_q_row
        + heads[:, None] * stride_q_head
        + dims[None, :],
        mask=(query_row_lanes[:, None] < TILE_R)
        & (query_rows[:, None] < request_end)
        & (heads[:, None] < NUM_HEADS)
        & (dims[None, :] < HEAD_DIM),
        other=0.0,
    )
    HALF_R: tl.constexpr = BLOCK_R // 2
    HALF_M: tl.constexpr = HALF_R * HEADS_PADDED
    query_left, query_right = tl.split(
        tl.permute(tl.reshape(query, (2, HALF_M, BLOCK_D)), (1, 2, 0))
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
            + physical_page[None, :].to(tl.int64) * stride_cache_block
            + page_offset[None, :] * stride_cache_token
            + dims[:, None],
            mask=live[None, :] & (dims[:, None] < HEAD_DIM),
            other=0.0,
        )
        scores_left = tl.dot(query_left, keys, out_dtype=tl.float32)
        scores_left = tl.reshape(scores_left, (HALF_R, HEADS_PADDED, BLOCK_N))
        real_heads_left = tl.arange(0, HEADS_PADDED) < NUM_HEADS
        scores_left = tl.where(real_heads_left[None, :, None], tl.maximum(scores_left, 0.0), 0.0)
        scores_left = tl.sum(scores_left, axis=1)
        rows_left = row_start + 0 + tl.arange(0, HALF_R)
        tl.store(
            logits_ptr + (rows_left - query_offset)[:, None] * logits_width + columns[None, :],
            tl.where(live[None, :], scores_left, -float("inf")),
            mask=(rows_left[:, None] < request_end)
            & (columns[None, :] < visible_left[:, None])
            & (columns[None, :] < logits_width),
        )
        scores_right = tl.dot(query_right, keys, out_dtype=tl.float32)
        scores_right = tl.reshape(scores_right, (HALF_R, HEADS_PADDED, BLOCK_N))
        real_heads_right = tl.arange(0, HEADS_PADDED) < NUM_HEADS
        scores_right = tl.where(real_heads_right[None, :, None], tl.maximum(scores_right, 0.0), 0.0)
        scores_right = tl.sum(scores_right, axis=1)
        rows_right = row_start + HALF_R + tl.arange(0, HALF_R)
        tl.store(
            logits_ptr + (rows_right - query_offset)[:, None] * logits_width + columns[None, :],
            tl.where(live[None, :], scores_right, -float("inf")),
            mask=(rows_right[:, None] < request_end)
            & (columns[None, :] < visible_right[:, None])
            & (columns[None, :] < logits_width),
        )
