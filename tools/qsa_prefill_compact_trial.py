# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Experimental compact request scheduling; production and TopK stay frozen."""

import copy
import hashlib
import types
from pathlib import Path
from typing import TYPE_CHECKING

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner
from tools.qsa_prefill_triton_ab import PRODUCTION_SHA256

if TYPE_CHECKING:
    from flaggems_vllm.ops.qsa_select_paged_prefill import _prefill_score_metadata

    _production_prefill_logits = None


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
    # Equal-length envelopes (including every upstream benchmark) keep the
    # original kernel/grid. Small chunks also keep the original path: their
    # compact lookup overhead is unlikely to amortize. Metadata only, no sync.
    if (
        q.shape[0] >= page_table.shape[0] * max_query_len
        or page_table.shape[0] <= 1
        or num_queries <= max_query_len
    ):
        return _production_prefill_logits(
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
            launch_metadata=launch_metadata,
        )
    if logits_workspace.ndim == 2:
        logits = (
            logits_workspace
            if num_queries == logits_workspace.shape[0]
            else logits_workspace[:num_queries]
        )
    else:
        logits = logits_workspace[: num_queries * logits_width].view(
            num_queries, logits_width
        )
    if launch_metadata is None:
        launch_metadata = _prefill_score_metadata(q, k_cache, page_table)
    strides, layout, constants = launch_metadata
    grid = lambda meta: (
        num_queries // meta["TILE_R"] + layout[0],
        1,
        triton.cdiv(logits_width, meta["BLOCK_N"] * meta["K_TILES"]),
    )
    _qsa_prefill_compact_score_kernel[grid](
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


def isolated_score_entry(kernel):
    """Keep production's dispatch cache independent of the trial's fallback.

    LibEntry keys omit the values of do_not_specialize scalars, including row
    count. A trial's small fallback chunk must not prime the production entry
    with a small-row tile. JIT code and the inner, row-keyed tuner are shared;
    only the outer per-device compiled-entry dictionaries are independent.
    """
    entry = copy.copy(kernel)
    entry.kernel_cache = tuple({} for _ in kernel.kernel_cache)
    entry._cpu_cache = {}
    return entry


def candidate_module(production):
    if (
        hashlib.sha256(Path(production.__file__).read_bytes()).hexdigest()
        != PRODUCTION_SHA256
    ):
        raise RuntimeError("Compact trial requires the pinned production source")
    module = types.ModuleType(production.__name__ + ".compact_trial")
    namespace = module.__dict__
    namespace.update(vars(production))
    namespace["__name__"] = module.__name__ = production.__name__ + ".compact_trial"
    namespace["_qsa_prefill_score_kernel"] = isolated_score_entry(
        production._qsa_prefill_score_kernel
    )
    original_launcher = production._prefill_logits
    fallback = types.FunctionType(
        original_launcher.__code__,
        namespace,
        original_launcher.__name__,
        original_launcher.__defaults__,
        original_launcher.__closure__,
    )
    fallback.__kwdefaults__ = original_launcher.__kwdefaults__
    namespace["_production_prefill_logits"] = fallback
    namespace["_qsa_prefill_compact_score_kernel"] = _qsa_prefill_compact_score_kernel
    for original in (_prefill_logits, production.qsa_select_paged_prefill):
        copy = types.FunctionType(
            original.__code__,
            namespace,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        copy.__kwdefaults__ = original.__kwdefaults__
        namespace[original.__name__] = copy
    module.production_baseline = production
    return module
