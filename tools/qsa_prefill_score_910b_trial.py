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

"""BF16 paged QSA score compile/correctness trial; no performance claim."""

import torch
import torch_npu  # noqa: F401 -- initialize NPU before importing Triton
import triton
import triton.language as tl


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


def score(q, cache, table, starts, visible, output, *, offset=0, programs=8):
    """Write unscaled FP32 scores; output rows select a chunk of packed Q."""
    if q.device.type != "npu" or q.dtype != torch.bfloat16 or cache.dtype != q.dtype:
        raise NotImplementedError("Score trial requires NPU BF16 Q/K")
    if q.ndim != 3 or q.shape[-1] != 128 or not 1 <= q.shape[1] <= 64:
        raise ValueError("Score trial supports D128 and H1..64")
    if cache.ndim != 4 or cache.shape[2:] != (1, 128):
        raise ValueError("Expected [pages,page_size,1,128] cache")
    if output.ndim != 2 or output.dtype != torch.float32:
        raise ValueError("Output must be FP32 [chunk_rows,width]")
    rows, width = output.shape
    if offset < 0 or offset + rows > q.shape[0] or programs < 1:
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
    ends = starts[1:] - starts[0]
    row_ids = torch.arange(q.shape[0], device=q.device, dtype=ends.dtype)
    owners = (row_ids[:, None] >= ends[None, :]).sum(1).to(torch.int32).contiguous()
    visible = visible.contiguous()
    # Conservative 32x128 key and 128x16 query tiles keep buffers small.
    # Eight programs is a compile probe cap, not a tuned production policy.
    block_n = 32
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
