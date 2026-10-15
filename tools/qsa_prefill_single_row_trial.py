# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Experimental single-query scoring; production defaults are unchanged."""

import hashlib
import inspect
import types
from pathlib import Path
from typing import TYPE_CHECKING

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

if TYPE_CHECKING:
    from flaggems_vllm.ops.qsa_select_paged_prefill import (
        _prefill_score_metadata,
        _qsa_prefill_score_kernel,
    )

BASELINE_COMMIT = "454074da0c079cc649d139489ce11fd3141585b7"
BASELINE_OP_SHA256 = "d70faeb6928c2c7a56636ded833f3359368871e353ecf4c66c6dfd35cae93e73"
SOURCE_HASHES = {
    "_prefill_logits": "417a41e1386141617939b8ddc545c72cc782a785e1cbdc1ed9a0f57ab3ef90d8",
    "_prefill_score_metadata": "0ab64038742906c5ef167e02adbfb2948c37ace37fd09ec85f12f759c63ed1a7",
    "qsa_select_paged_prefill": "9dc80c8e7f8ee0ca77883d07dd9c5ee0cc073383108a54e0df3453af0b62cad6",
    "_validate_inputs": "c6d4d33f2de988026e2c51b3d13c3ee75c2c109f65b8fb5eba756486b621ba49",
    "_output_overlaps_inputs": "abfd0e120bc8839579d13974e97b8f596ec28cac6da0c15646d45d77f4926697",
    "_check_nonoverlapping_strides": "f287198d67f5d66ee6799854739f5bae190c834d8b68ea416210515c319fea29",
}


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("qsa_select_paged_prefill_single_row_score"),
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
def _qsa_prefill_single_row_score_kernel(
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
    tl.static_assert(TILE_R == 1 and K_TILES == 1)
    request = tl.program_id(0)
    query_base = tl.load(starts_ptr)
    start = tl.load(starts_ptr + request) - query_base
    end = tl.load(starts_ptr + request + 1) - query_base
    if num_rows != 1 or query_offset < start or query_offset >= end:
        return
    visible = tl.minimum(tl.load(visible_ptr + query_offset), logits_width)
    if tl.program_id(2) * BLOCK_N >= visible:
        return
    columns = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, triton.next_power_of_2(HEAD_DIM))
    logical_page = columns // PAGE_SIZE
    live = (columns < visible) & (logical_page < table_width)
    physical_page = tl.load(
        table_ptr + request * stride_table_req + logical_page * stride_table_page,
        mask=live,
        other=0,
    )
    live &= (physical_page >= 0) & (physical_page < num_pages)
    keys = tl.load(
        cache_ptr
        + physical_page[:, None].to(tl.int64) * stride_cache_block
        + (columns % PAGE_SIZE)[:, None] * stride_cache_token
        + dims[None, :],
        mask=live[:, None] & (dims[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    scores = tl.full((BLOCK_N,), 0.0, tl.float32)
    for head in tl.static_range(NUM_HEADS):
        query = tl.load(
            q_ptr
            + query_offset.to(tl.int64) * stride_q_row
            + head * stride_q_head
            + dims,
            mask=dims < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        per_head = tl.sum(keys * query[None, :], axis=1)
        scores += tl.maximum(per_head, 0.0)
    tl.store(
        logits_ptr + columns,
        tl.where(live, scores, -float("inf")),
        mask=columns < visible,
    )


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
    *,
    launch_metadata=None,
):
    """Internal score launcher; inputs have already passed host validation."""
    if logits_workspace.ndim == 2:
        logits = (
            logits_workspace
            if num_queries == logits_workspace.shape[0]
            else logits_workspace[:num_queries]
        )
    else:
        # Preserve direct private-helper calls using the original flat buffer.
        logits = logits_workspace[: num_queries * logits_width].view(
            num_queries, logits_width
        )
    if launch_metadata is None:
        launch_metadata = _prefill_score_metadata(q, k_cache, page_table)
    strides, layout, constants = launch_metadata
    grid = lambda meta: (
        layout[0],
        triton.cdiv(min(num_queries, max_query_len), meta["TILE_R"]),
        triton.cdiv(logits_width, meta["BLOCK_N"] * meta["K_TILES"]),
    )
    kernel = (
        _qsa_prefill_single_row_score_kernel
        if num_queries == 1
        and constants["NUM_HEADS"] <= 8
        and constants["HEAD_DIM"] <= 256
        else _qsa_prefill_score_kernel
    )
    kernel[grid](
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        logits,
        *strides,
        num_queries,
        query_offset,
        max_query_len,
        logits_width,
        *layout,
        **constants,
    )
    return logits


def candidate_module(op):
    if hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest() != BASELINE_OP_SHA256:
        raise RuntimeError(
            "Single-row trial requires the pinned production source file"
        )
    for name, expected in SOURCE_HASHES.items():
        source = inspect.getsource(getattr(op, name)).strip()
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            raise RuntimeError(f"Single-row trial requires pinned source: {name}")
    module = types.ModuleType(op.__name__ + ".single_row_trial")
    namespace = module.__dict__
    namespace.update(vars(op))
    namespace["__name__"] = op.__name__ + ".single_row_trial"
    namespace["__file__"] = __file__
    namespace["_qsa_prefill_single_row_score_kernel"] = (
        _qsa_prefill_single_row_score_kernel
    )
    launcher = types.FunctionType(
        _prefill_logits.__code__,
        namespace,
        _prefill_logits.__name__,
        _prefill_logits.__defaults__,
    )
    launcher.__kwdefaults__ = dict(_prefill_logits.__kwdefaults__)
    namespace["_prefill_logits"] = launcher
    original = op.qsa_select_paged_prefill
    namespace["qsa_select_paged_prefill"] = types.FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__
    )
    for name in (
        "_qsa_prefill_score_kernel",
        "persistent_topk",
        "_select_all_kernel",
        "_validate_inputs",
        "_prefill_score_metadata",
    ):
        if namespace[name] is not getattr(op, name):
            raise RuntimeError(f"Candidate must share production callable: {name}")
    return module
