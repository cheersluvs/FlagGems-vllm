# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Frozen pre-preparation host code; benchmark only, never production-imported."""

from __future__ import annotations

import hashlib
import inspect
import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os

    import torch
    import triton

    from flaggems_vllm.ops.qsa_select_paged_prefill import (
        _qsa_prefill_score_kernel,
        _select_all_kernel,
        _validate_inputs,
        persistent_topk,
    )

BASELINE_COMMIT = "ab50d7a03ec3d5e683691e0f2f8bb46c57307de8"
BASELINE_OP_SHA256 = "c424d97849f6276a0716dd573e8387acea3eaefb222fd62d334d49931e3b4a58"
SOURCE_HASHES = {
    "_prefill_logits": "435ed1f271e6c0eb571b210091503b7d54beb841b671e5143933ef4ca274e483",
    "qsa_select_paged_prefill": "277945267d171b819dc6c282e41368d259ed3831a26faed8d04b13e7f75e508a",
    "_check_nonoverlapping_strides": "f287198d67f5d66ee6799854739f5bae190c834d8b68ea416210515c319fea29",
    "_validate_inputs": "c6d4d33f2de988026e2c51b3d13c3ee75c2c109f65b8fb5eba756486b621ba49",
    "_output_overlaps_inputs": "abfd0e120bc8839579d13974e97b8f596ec28cac6da0c15646d45d77f4926697",
}


CANDIDATE_SOURCE_HASHES = {
    "_prefill_score_metadata": "0ab64038742906c5ef167e02adbfb2948c37ace37fd09ec85f12f759c63ed1a7",
    "_prefill_logits": "417a41e1386141617939b8ddc545c72cc782a785e1cbdc1ed9a0f57ab3ef90d8",
    "qsa_select_paged_prefill": "9dc80c8e7f8ee0ca77883d07dd9c5ee0cc073383108a54e0df3453af0b62cad6",
}


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


def baseline_entry(op):
    for name, expected in CANDIDATE_SOURCE_HASHES.items():
        source = inspect.getsource(getattr(op, name)).strip()
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            raise RuntimeError(
                f"Chunk-setup baseline requires pinned candidate source: {name}"
            )
    frozen = {
        "_prefill_logits": _prefill_logits,
        "qsa_select_paged_prefill": qsa_select_paged_prefill,
    }
    for name, expected in SOURCE_HASHES.items():
        function = frozen[name] if name in frozen else getattr(op, name)
        source = inspect.getsource(function).strip()
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            raise RuntimeError(f"Chunk-setup baseline requires pinned source: {name}")
    namespace = dict(op.qsa_select_paged_prefill.__globals__)
    namespace["_prefill_logits"] = types.FunctionType(
        _prefill_logits.__code__,
        namespace,
        _prefill_logits.__name__,
        _prefill_logits.__defaults__,
    )
    for name in (
        "_validate_inputs",
        "_qsa_prefill_score_kernel",
        "persistent_topk",
        "_select_all_kernel",
    ):
        if namespace[name] is not getattr(op, name):
            raise RuntimeError(f"A/B must share the same callable: {name}")
    return types.FunctionType(
        qsa_select_paged_prefill.__code__,
        namespace,
        "qsa_full_host_baseline",
        qsa_select_paged_prefill.__defaults__,
    )
