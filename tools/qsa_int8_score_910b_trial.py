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

"""Standalone INT8 scores with optional raw INT32 dot diagnostics."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_score_trial(
    Q,
    K,
    QS,
    KS,
    Weights,
    Table,
    Visible,
    Out,
    RawDot,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    COLUMNS: tl.constexpr,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASK_RAW_DOT: tl.constexpr,
    WRITE_RAW_DOT: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(WIDTH, BLOCK_N)
    dims = tl.arange(0, 128)
    lanes = tl.arange(0, 16)
    for work in range(tl.program_id(0), ROWS * tiles, tl.num_programs(0)):
        row = work // tiles
        columns = (work % tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
        visible = tl.load(Visible + row)
        live = (columns < WIDTH) & (columns < visible)
        physical = tl.load(Table + row * COLUMNS + columns // PAGE, mask=live, other=0)
        token = physical.to(tl.int64) * PAGE + columns % PAGE
        keys = tl.load(
            K + token[:, None] * 128 + dims[None, :],
            mask=live[:, None],
            other=0,
        )
        key_scale = tl.load(KS + token, mask=live, other=0).to(tl.float32)
        scores = tl.full((BLOCK_N,), 0, tl.float32)
        for group in tl.static_range(triton.cdiv(HEADS, 16)):
            heads = group * 16 + lanes
            query = tl.load(
                Q + (row * HEADS + heads[None, :]) * 128 + dims[:, None],
                mask=heads[None, :] < HEADS,
                other=0,
            )
            dot = tl.dot(keys, query, out_dtype=tl.int32)
            if WRITE_RAW_DOT:
                if MASK_RAW_DOT:
                    debug_dot = tl.where(live[:, None], dot, 0)
                else:
                    debug_dot = dot
                tl.store(
                    RawDot + (row * WIDTH + columns[:, None]) * HEADS + heads[None, :],
                    debug_dot,
                    mask=(columns[:, None] < WIDTH) & (heads[None, :] < HEADS),
                )
            query_scale = tl.load(
                QS + row * HEADS + heads, mask=heads < HEADS, other=0
            ).to(tl.float32)
            weights = tl.load(
                Weights + row * HEADS + heads, mask=heads < HEADS, other=0
            ).to(tl.float32)
            dequant = dot.to(tl.float32) * query_scale[None, :] * key_scale[:, None]
            scores += tl.sum(tl.maximum(dequant, 0.0) * weights[None, :], axis=1)
        tl.store(
            Out + row * WIDTH + columns,
            tl.where(live, scores, -float("inf")),
            mask=columns < WIDTH,
        )


def score(
    q,
    cache,
    qs,
    ks,
    weights,
    row_table,
    visible,
    output,
    raw_dot=None,
    *,
    mask_debug=True,
):
    """Score contiguous operands; raw_dot=None removes the debug store."""
    if q.device.type != "npu" or q.dtype != torch.int8 or cache.dtype != torch.int8:
        raise NotImplementedError("INT8 NPU Q/K required")
    if "910B" not in torch.npu.get_device_name(q.device.index):
        raise NotImplementedError("This trial targets Ascend 910B")
    if q.ndim != 3 or q.shape[2] != 128 or not 1 <= q.shape[1] <= 64:
        raise NotImplementedError("Trial requires D128 and H1..64")
    if cache.ndim != 4 or cache.shape[2:] != (1, 128):
        raise ValueError("Cache must be [pages,page,1,128]")
    rows, heads, _ = q.shape
    page = cache.shape[1]
    if page <= 0 or page > 1024 or page % 16:
        raise ValueError("Page must be a multiple of 16, <=1024")
    if (
        row_table.ndim != 2
        or row_table.shape[0] != rows
        or row_table.dtype != torch.int32
    ):
        raise ValueError("Row page table must be INT32 [rows,columns]")
    if visible.shape != (rows,) or visible.dtype != torch.int32:
        raise ValueError("Visibility must be INT32 [rows]")
    if qs.shape != (rows, heads) or weights.shape != qs.shape:
        raise ValueError("Query scales/weights must be [rows,heads]")
    if ks.shape != cache.shape[:3] or any(
        t.dtype != torch.float16 for t in (qs, ks, weights)
    ):
        raise ValueError("Scales and weights must be FP16 with matching shapes")
    if output.ndim != 2 or output.shape[0] != rows or output.dtype != torch.float32:
        raise ValueError("Scores must be FP32 [rows,width]")
    width = output.shape[1]
    if not 0 < width <= row_table.shape[1] * page:
        raise ValueError("Score width must fit page table capacity")
    if raw_dot is not None and (
        raw_dot.shape != (rows, width, heads) or raw_dot.dtype != torch.int32
    ):
        raise ValueError("Debug dots must be INT32 [rows,width,heads]")
    tensors = (q, cache, qs, ks, weights, row_table, visible, output)
    if raw_dot is not None:
        tensors += (raw_dot,)
    if any(not t.is_contiguous() or t.device != q.device for t in tensors):
        raise NotImplementedError("Contiguous operands on one NPU required")
    if rows == 0:
        return None
    return _quant_score_trial[(min(8, rows * triton.cdiv(width, 32)),)](
        q,
        cache,
        qs,
        ks,
        weights,
        row_table,
        visible,
        output,
        raw_dot if raw_dot is not None else output,
        HEADS=heads,
        PAGE=page,
        COLUMNS=row_table.shape[1],
        ROWS=rows,
        WIDTH=width,
        BLOCK_N=32,
        MASK_RAW_DOT=mask_debug,
        WRITE_RAW_DOT=raw_dot is not None,
        num_warps=4,
    )
