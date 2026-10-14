# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""BF16/FP8 QSA paged prefill selection for NVIDIA SM90.

API/semantics reference: vllm-project/vllm, commit
7a877ae34dcf4a7d46de08eb16db4699e1e0047c,
vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py.

The score kernel uses request-local query tiles and reuses keys across queries.
Top-K is provided by this repository, without a vLLM extension dependency.
Selection order and tie-breaking are unspecified. Device-resident metadata
must satisfy the documented value preconditions; validation does not copy it
to the host. Autograd is not supported.
"""

import os

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.persistent_topk import persistent_topk
from flaggems_vllm.utils import libentry, libtuner


# Fixed-size bandwidth-only fills: no autotune. They do not perform selection.
@triton.jit
def _select_all_kernel(visible_ptr, output_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    visible = tl.load(visible_ptr + row)
    tl.store(
        output_ptr + row * K + columns,
        tl.where(columns < visible, columns, -1),
        mask=columns < K,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("qsa_select_paged_prefill_score"),
    key=[
        "num_rows",
        "max_query_len",
        "logits_width",
        "num_requests",
        "PAGE_SIZE",
        "NUM_HEADS",
        "HEAD_DIM",
        "IS_FP8",
        "stride_q_row",
        "stride_q_head",
        "stride_cache_block",
        "stride_cache_token",
        "stride_table_req",
        "stride_table_page",
    ],
    strategy="default",
)
@triton.jit(do_not_specialize=["num_rows", "query_offset", "max_query_len"])
def _qsa_prefill_score_kernel(
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
    request = tl.program_id(0)
    query_base = tl.load(starts_ptr)
    request_start = tl.maximum(tl.load(starts_ptr + request) - query_base, query_offset)
    request_end = tl.minimum(
        tl.load(starts_ptr + request + 1) - query_base, query_offset + num_rows
    )
    row_start = request_start + tl.program_id(1) * TILE_R
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
        scores = tl.dot(query, keys, out_dtype=tl.float32)
        scores = tl.reshape(scores, (BLOCK_R, HEADS_PADDED, BLOCK_N))
        real_heads = tl.arange(0, HEADS_PADDED) < NUM_HEADS
        scores = tl.where(real_heads[None, :, None], tl.maximum(scores, 0.0), 0.0)
        scores = tl.sum(scores, axis=1)
        tl.store(
            logits_ptr
            + (rows - query_offset)[:, None] * logits_width
            + columns[None, :],
            tl.where(live[None, :], scores, -float("inf")),
            mask=valid_rows[:, None]
            & (columns[None, :] < visible[:, None])
            & (columns[None, :] < logits_width),
        )


def _check_nonoverlapping_strides(tensor, name):
    extent = 1
    for stride, size in sorted(zip(tensor.stride(), tensor.shape)):
        if size > 1:
            if stride < extent:
                raise NotImplementedError(f"{name} must have nonoverlapping strides")
            extent += (size - 1) * stride


def _overlaps(a, b):
    if not a.numel() or not b.numel():
        return False
    a_start, b_start = a.data_ptr(), b.data_ptr()
    a_end = (
        a_start
        + (1 + sum((size - 1) * stride for size, stride in zip(a.shape, a.stride())))
        * a.element_size()
    )
    b_end = (
        b_start
        + (1 + sum((size - 1) * stride for size, stride in zip(b.shape, b.stride())))
        * b.element_size()
    )
    return a_start < b_end and b_start < a_end


def _validate_inputs(
    q,
    k_cache,
    page_table,
    query_start_loc,
    visible_blocks,
    token_topk,
    compress_ratio,
    max_query_len,
    block_indices,
    max_seq_len,
):
    tensors = (q, k_cache, page_table, query_start_loc, visible_blocks, block_indices)
    if q.device.type != "cuda" or runtime.device.vendor_name != "nvidia":
        raise NotImplementedError("QSA prefill currently requires NVIDIA CUDA SM90")
    if any(t.device != q.device for t in tensors):
        raise ValueError("QSA prefill tensors must be on the same device")
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise NotImplementedError("QSA prefill currently supports SM90 (H800/H100)")
    if q.dtype != k_cache.dtype:
        raise ValueError("Q and the compressed K cache must match dtype")
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise NotImplementedError("QSA prefill supports BF16/FP8 E4M3FN query/cache")
    if q.ndim != 3:
        raise NotImplementedError("QSA prefill query must be [tokens, heads, head_dim]")
    if q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA prefill heads and head_dim must be positive")
    if k_cache.ndim != 4 or k_cache.shape[2:] != (1, q.shape[2]):
        raise NotImplementedError(
            "QSA prefill cache must be [pages, page_size, 1, D] matching Q"
        )
    if q.requires_grad or k_cache.requires_grad:
        raise NotImplementedError("QSA prefill is inference-only")
    for name, value in (
        ("token_topk", token_topk),
        ("compress_ratio", compress_ratio),
        ("max_query_len", max_query_len),
        ("max_seq_len", max_seq_len),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if compress_ratio <= 0 or token_topk <= 0 or token_topk % compress_ratio:
        raise ValueError("token_topk must be positive and divisible by compress_ratio")
    k = token_topk // compress_ratio
    if k not in (512, 1024, 2048):
        raise NotImplementedError("QSA prefill supports block Top-K 512/1024/2048")
    rows = q.shape[0]
    if page_table.ndim != 2:
        raise ValueError("page_table must be [requests, max_pages]")
    if page_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("page_table must use int32/int64")
    if query_start_loc.shape != (page_table.shape[0] + 1,):
        raise ValueError("query_start_loc must contain requests + 1 offsets")
    if query_start_loc.dtype not in (torch.int32, torch.int64):
        raise TypeError("query_start_loc must use int32/int64")
    if visible_blocks.shape != (rows,) or visible_blocks.dtype != torch.int32:
        raise ValueError("visible_blocks must be int32 [tokens]")
    if block_indices.shape != (rows, k) or block_indices.dtype != torch.int32:
        raise ValueError("block_indices must be int32 [tokens, block_topk]")
    if max_query_len < 0 or max_seq_len < 0:
        raise ValueError("maximum lengths must be nonnegative")
    if rows and (
        not all(k_cache.shape[:2])
        or not all(page_table.shape)
        or max_query_len == 0
        or max_seq_len == 0
    ):
        raise ValueError(
            "nonempty queries require positive cache/table/maximum lengths"
        )
    if q.stride(-1) != 1 or k_cache.stride(-1) != 1:
        raise NotImplementedError("QSA query/cache require unit innermost stride")
    for name, tensor in (("q", q), ("cache", k_cache), ("page_table", page_table)):
        _check_nonoverlapping_strides(tensor, name)
    for tensor in (query_start_loc, visible_blocks, block_indices):
        if not tensor.is_contiguous():
            raise NotImplementedError("QSA metadata/output must be contiguous")
    if any(_overlaps(block_indices, tensor) for tensor in tensors[:-1]):
        raise ValueError("block_indices must not overlap any input")
    return rows, k


def _prefill_logits(
    q,
    k_cache,
    page_table,
    query_start_loc,
    visible_blocks,
    logits_workspace,
    max_query_len,
    logits_width,
    query_offset,
    num_queries,
):
    """Internal score launcher; inputs have already passed host validation."""
    logits = logits_workspace[: num_queries * logits_width].view(
        num_queries, logits_width
    )
    grid = lambda meta: (
        page_table.shape[0],
        triton.cdiv(min(num_queries, max_query_len), meta["TILE_R"]),
        triton.cdiv(logits_width, meta["BLOCK_N"] * meta["K_TILES"]),
    )
    _qsa_prefill_score_kernel[grid](
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        page_table.stride(0),
        page_table.stride(1),
        num_queries,
        query_offset,
        max_query_len,
        logits_width,
        page_table.shape[0],
        k_cache.shape[0],
        page_table.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        IS_FP8=q.dtype == torch.float8_e4m3fn,
    )
    return logits


def qsa_select_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    max_query_len: int,
    block_indices: torch.Tensor,
    max_seq_len: int,
) -> None:
    """Write logical compressed-block indices selected for packed prefill Q.

    Matching BF16/FP8 Q is [tokens,H,D], K is [pages,page_size,1,D], H,D>0.
    Heads and dimensions are inferred from Q; compilation must fit device
    resources. Padded head/dimension lanes do not contribute to the score.
    ``query_start_loc``
    is [requests+1] and its first entry may be nonzero. Caller-provided
    ``visible_blocks`` is int32 [tokens]. ``block_indices`` is contiguous
    int32 [tokens,token_topk//compress_ratio]. Short rows are padded with -1.
    The caller must supply valid device-resident metadata and finite Q/K.
    Only logits workspace is limited by VLLM_SPARSE_INDEXER_MAX_LOGITS_MB;
    zero requests one row per chunk, as in vLLM. At least one logits row is
    allocated, and the existing Top-K operator allocates additional scratch.
    """
    rows, k = _validate_inputs(
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        token_topk,
        compress_ratio,
        max_query_len,
        block_indices,
        max_seq_len,
    )
    if rows == 0:
        return
    capacity = page_table.shape[1] * k_cache.shape[1]
    visible_bound = min(capacity, triton.cdiv(max_seq_len, compress_ratio))
    with torch.cuda.device(q.device):
        if visible_bound <= k:
            _select_all_kernel[(rows, triton.cdiv(k, 256))](
                visible_blocks, block_indices, K=k, BLOCK=256, num_warps=4
            )
            return
        try:
            budget_mb = int(os.environ.get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512"))
        except ValueError as exc:
            raise ValueError(
                "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB must be an integer"
            ) from exc
        if budget_mb < 0:
            raise ValueError("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB must be nonnegative")
        logits_width = min(capacity, triton.cdiv(max(64, visible_bound), 64) * 64)
        rows_per_chunk = min(
            rows, max(1, budget_mb * 1024 * 1024 // (logits_width * 4))
        )
        workspace = torch.empty(
            (rows_per_chunk * logits_width,), dtype=torch.float32, device=q.device
        )
        topk_workspace = torch.empty((1024 * 1024,), dtype=torch.uint8, device=q.device)
        for query_offset in range(0, rows, rows_per_chunk):
            num_queries = min(rows_per_chunk, rows - query_offset)
            logits = _prefill_logits(
                q,
                k_cache,
                page_table,
                query_start_loc,
                visible_blocks,
                workspace,
                max_query_len,
                logits_width,
                query_offset,
                num_queries,
            )
            query_slice = slice(query_offset, query_offset + num_queries)
            persistent_topk(
                logits,
                visible_blocks[query_slice],
                block_indices[query_slice],
                topk_workspace,
                k=k,
                max_seq_len=logits_width,
            )


__all__ = ["qsa_select_paged_prefill"]
