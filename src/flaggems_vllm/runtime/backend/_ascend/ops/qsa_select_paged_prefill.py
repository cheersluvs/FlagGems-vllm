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

"""BF16 paged QSA selection for Ascend with FP32 scores and group indices."""

import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._ascend.ops.persistent_topk import persistent_topk

_QSA_TOPK_BLOCK = 256


@lru_cache(maxsize=None)
def _score_programs(device_index):
    """Bound the validated launch by the device Cube count; retain a safe fallback."""
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(device_index)
        for key in ("cube_core_num", "aic_core_num", "num_aicore"):
            count = props.get(key)
            if type(count) is int and count > 0:
                return min(20, count)
    except (AttributeError, RuntimeError, TypeError):
        pass
    return 8


@triton.jit
def _request_owners(
    Starts,
    Owners,
    STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    REQUESTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base = tl.load(Starts)
    owners = tl.full((BLOCK,), 0, tl.int32)
    for request in range(REQUESTS):
        end = tl.load(Starts + (request + 1) * STRIDE) - base
        owners += (rows < ROWS) & (rows >= end)
    tl.store(Owners + rows, owners, mask=rows < ROWS)


def _build_owners(q, table, starts):
    owners = torch.empty((q.shape[0],), device=q.device, dtype=torch.int32)
    _request_owners[(triton.cdiv(q.shape[0], 128),)](
        starts,
        owners,
        starts.stride(0),
        q.shape[0],
        table.shape[0],
        128,
        num_warps=4,
    )
    return owners


@triton.jit
def _paged_score_910b(
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
    # Limit launched programs; each program processes independent work tiles.
    for work in range(tl.program_id(0), ROWS * tiles, tl.num_programs(0)):
        local_row = work // tiles
        row = local_row + OFFSET
        columns = (work % tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
        request = tl.load(Owners + row)
        visible = tl.load(Visible + row)
        live = (columns < WIDTH) & (columns < visible)
        physical = tl.load(
            Table + request * ST0 + (columns // PAGE) * ST1,
            mask=live,
            other=0,
        )
        keys = tl.load(
            K
            + physical[:, None].to(tl.int64) * SK0
            + (columns % PAGE)[:, None] * SK1
            + dims[None, :] * SK3,
            mask=live[:, None],
            other=0.0,
        )
        scores = tl.full((BLOCK_N,), 0.0, tl.float32)
        # All dot dimensions are multiples of 16; only real heads contribute.
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


def _score(
    q, cache, table, starts, visible, output, *, offset=0, programs=None, owners=None
):
    """Write unscaled FP32 scores; output rows select a chunk of packed Q."""
    if q.device.type != "npu" or q.dtype != torch.bfloat16 or cache.dtype != q.dtype:
        raise NotImplementedError("Ascend QSA requires NPU BF16 Q/K")
    if q.ndim != 3 or q.shape[-1] != 128 or not 1 <= q.shape[1] <= 64:
        raise ValueError("Ascend QSA supports D128 and H1..64")
    if cache.ndim != 4 or cache.shape[2:] != (1, 128):
        raise ValueError("Expected [pages,page_size,1,128] cache")
    if output.ndim != 2 or output.dtype != torch.float32:
        raise ValueError("Output must be FP32 [chunk_rows,width]")
    rows, width = output.shape
    if (
        offset < 0
        or offset + rows > q.shape[0]
        or (programs is not None and programs < 1)
    ):
        raise ValueError("Invalid chunk or program count")
    if table.ndim != 2 or starts.shape != (table.shape[0] + 1,):
        raise ValueError("Invalid request metadata shapes")
    if table.dtype not in (torch.int32, torch.int64) or starts.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Request metadata must be int32/int64")
    if visible.shape != (q.shape[0],) or visible.dtype != torch.int32:
        raise ValueError("Visibility must be int32 [packed_rows]")
    if not 0 < width <= table.shape[1] * cache.shape[1]:
        raise ValueError("Width exceeds page-table capacity")
    if any(t.device != q.device for t in (cache, table, starts, visible, output)):
        raise ValueError("All operands must be on the same NPU")
    if output.stride(1) != 1 or output.stride(0) < width:
        raise ValueError("Output rows must be nonoverlapping with unit column stride")
    if not rows:
        return output
    if not visible.is_contiguous():
        raise ValueError("Visibility must be contiguous")
    if owners is None:
        owners = _build_owners(q, table, starts)
    # Conservative 32x128 key and 128x16 query tiles keep buffers small.
    # Use at most the validated 20-program launch, bounded by available Cube cores.
    block_n = 32
    if programs is None:
        index = q.device.index
        if index is None:
            index = torch.npu.current_device()
        programs = _score_programs(index)
    grid = (min(programs, rows * triton.cdiv(width, block_n)),)
    _paged_score_910b[grid](
        q,
        cache,
        table,
        owners,
        visible,
        output,
        *q.stride(),
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        *table.stride(),
        *output.stride(),
        HEADS=q.shape[1],
        PAGE=cache.shape[1],
        WIDTH=width,
        OFFSET=offset,
        ROWS=rows,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return output


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
    """Write compressed-group indices for packed BF16 prefill queries.

    Q is [rows,H,128], H=1..64; K is [pages,page_size,1,128]. Metadata
    and all operands reside on the same NPU. Request offsets may start
    above zero. Visibility counts compressed groups and must not exceed
    the page-table capacity or ceil(max_seq_len/compress_ratio). Metadata
    values and finite Q/K are caller preconditions, without host reads.
    Visibility must be contiguous. Output must not overlap inputs.
    Output is contiguous int32 [rows,token_topk//compress_ratio], padded
    with -1. Selection order and tie-breaking are unspecified.
    """
    for name, value in (
        ("token_topk", token_topk),
        ("compress_ratio", compress_ratio),
        ("max_query_len", max_query_len),
        ("max_seq_len", max_seq_len),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if compress_ratio <= 0 or token_topk <= 0 or token_topk % compress_ratio:
        raise ValueError("Invalid token TopK or compression ratio")
    k = token_topk // compress_ratio
    if k not in (512, 1024, 2048):
        raise NotImplementedError("Group TopK must be 512, 1024 or 2048")
    if q.device.type != "npu" or q.dtype != torch.bfloat16:
        raise NotImplementedError("Ascend QSA currently supports BF16 on NPU")
    if q.ndim != 3 or q.shape[2] != 128 or not 1 <= q.shape[1] <= 64:
        raise NotImplementedError("Ascend QSA supports H1..64 and D128")
    if k_cache.dtype != q.dtype or k_cache.ndim != 4 or k_cache.shape[2:] != (1, 128):
        raise ValueError("Expected BF16 cache [pages,page_size,1,128]")
    if q.requires_grad or k_cache.requires_grad:
        raise NotImplementedError("QSA selection is inference-only")
    if any(
        t.device != q.device
        for t in (k_cache, page_table, query_start_loc, visible_blocks, block_indices)
    ):
        raise ValueError("All operands must be on the same NPU")
    if page_table.ndim != 2 or query_start_loc.shape != (page_table.shape[0] + 1,):
        raise ValueError("Invalid request metadata shapes")
    if any(
        t.dtype not in (torch.int32, torch.int64) for t in (page_table, query_start_loc)
    ):
        raise ValueError("Request metadata must be int32/int64")
    rows = q.shape[0]
    if (
        visible_blocks.shape != (rows,)
        or visible_blocks.dtype != torch.int32
        or not visible_blocks.is_contiguous()
    ):
        raise ValueError("Visibility must be int32 [rows]")
    if (
        block_indices.shape != (rows, k)
        or block_indices.dtype != torch.int32
        or not block_indices.is_contiguous()
    ):
        raise ValueError("Output must be contiguous int32 [rows,group_topk]")
    if max_query_len < 0 or max_seq_len < 0:
        raise ValueError("Maximum lengths must be nonnegative")
    if not rows:
        return
    if (
        not all(k_cache.shape[:2])
        or not all(page_table.shape)
        or not max_query_len
        or not max_seq_len
    ):
        raise ValueError(
            "Nonempty queries require positive cache/table/maximum lengths"
        )
    capacity = page_table.shape[1] * k_cache.shape[1]
    width = min(
        capacity, triton.cdiv(triton.cdiv(max_seq_len, compress_ratio), 64) * 64
    )
    try:
        budget_mb = int(os.environ.get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512"))
    except ValueError as exc:
        raise ValueError("Invalid logits budget") from exc
    if budget_mb < 0:
        raise ValueError("Logits budget must be nonnegative")
    chunk_rows = min(rows, 16, max(1, budget_mb * 1024 * 1024 // (width * 4)))
    with torch.npu.device(q.device):
        owners = _build_owners(q, page_table, query_start_loc)
        logits = torch.empty((chunk_rows, width), device=q.device, dtype=torch.float32)
        workspace = torch.empty((1024 * 1024,), device=q.device, dtype=torch.uint8)
        for first in range(0, rows, chunk_rows):
            count = min(chunk_rows, rows - first)
            chunk = logits[:count]
            _score(
                q,
                k_cache,
                page_table,
                query_start_loc,
                visible_blocks,
                chunk,
                offset=first,
                owners=owners,
            )
            persistent_topk(
                chunk,
                visible_blocks[first : first + count],
                block_indices[first : first + count],
                workspace,
                k=k,
                max_seq_len=width,
                block=_QSA_TOPK_BLOCK,
            )


__all__ = ["qsa_select_paged_prefill"]
