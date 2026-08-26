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

"""Multi-block prefill top-K for Moore Threads, to test one hypothesis.

The generic prefill launches grid=(num_rows,) with no intra-row parallelism. On a
60-SM S5000 that leaves the card idle whenever num_rows is small, and it is
exactly there that the generic kernel loses to vLLM:

    num_rows=4    0.1 programs/SM   0.669 / 0.853
    num_rows=64   1.1 programs/SM   0.638
    num_rows>=4100  68-273/SM       1.109 - 1.538   (generic already wins)

Decode does not have this problem because it splits a row across
MULTIPLE_BLOCKS_PER_ROW_CONFIG blocks and merges. Prefill has the *same*
machinery in its `_top_k_per_row_job` -- byte-identical signature -- but its outer
kernel never plumbs the flags through, and SPLIT_WORK_THRESHOLD,
MULTIPLE_BLOCKS_PER_ROW_CONFIG and NUM_THREADS_PER_BLOCK_MERGE all sit in
top_k_per_row_prefill.py unreferenced.

This override supplies the missing outer kernel and dispatcher while reusing the
generic `_top_k_per_row_job` unchanged, so the generic op keeps a zero diff.

HYPOTHESIS UNDER TEST: the small-num_rows deficit is an occupancy problem, and
splitting the row fixes it. Falsified if (64, 129280) does not move toward 1.0.

This is a probe, not a shipping candidate. If it works the mechanism belongs in
the generic op, where every vendor gets it -- the weakness is not MUSA-specific.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.top_k_per_row_prefill import (
    NUM_THREADS_PER_BLOCK,
    NUM_THREADS_PER_BLOCK_MERGE,
    _top_k_per_row_job,
    _use_radix_final_for_prefill,
)
from flaggems_vllm.ops.top_k_per_row_prefill import (
    top_k_per_row_prefill as _generic_prefill,
)
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE = True
    except ImportError:
        tle = None
        HAS_TLE = False
else:
    tle = None
    HAS_TLE = False


# Mirrors MULTIPLE_BLOCKS_PER_ROW_CONFIG in the generic op.
SPLIT = 10

# Split only when the grid cannot fill the card. S5000 has 60 SMs; below 2x that
# the generic launch leaves most of it idle. Above it, generic already wins and
# the extra merge launch would be pure cost.
SPLIT_MAX_ROWS = 120

# ...and only when each block still has real work. Below this the two extra
# launches dominate, which is why the num_rows=4 shapes (span 8193 / 16385, so
# 819 / 1638 per block) must NOT take this path -- they are launch-bound, not
# occupancy-bound, and measured at 0.3-0.4% of peak bandwidth.
SPLIT_MIN_CHUNK = 4096


@triton.jit
def _mtt_multi_block_prefill(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    stride0,
    stride1,
    vocab_size,
    out_logits_ptr,
    indices_ptr,
    TOPK: tl.constexpr,
    TOPKP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
    MULTIPLE_BLOCKS_NUM: tl.constexpr,
    MERGE_BLOCKS: tl.constexpr,
):
    """Outer kernel for the two-pass split. Handles ONLY split and merge.

    The single-block case is never routed here -- the dispatcher returns the
    generic function for it -- which keeps this kernel to two modes.
    """
    NUM_FILNAL_ITEMS: tl.constexpr = 2048
    NUM_BINS: tl.constexpr = 2048
    VEC: tl.constexpr = 4

    row_id = tl.program_id(0)

    if MERGE_BLOCKS:
        # Pass 2. `logits_ptr` is the aux logits [num_rows, SPLIT*TOPK] and
        # `indices_ptr` the aux indices, which already carry indices relative to
        # row_starts[row_id] (see the pointer shift below). So the merge emits
        # the caller's convention directly, with no offset to undo.
        row_start = 0
        row_end = MULTIPLE_BLOCKS_NUM * TOPK
        indices_ptr += row_id * MULTIPLE_BLOCKS_NUM * TOPK
        out_indices_ptr += row_id * TOPK
        logits_ptr += row_id * stride0
        skip_elems = 0
    else:
        # Pass 1. In multi-block mode `_top_k_per_row_job` emits `pos +
        # row_start`, i.e. indices relative to whatever origin the caller uses.
        # Decode can pass an absolute offset because its rows start at 0; prefill
        # rows start at row_starts[i], so we shift the base pointer to that start
        # and give the job *span-relative* bounds. Passing the absolute start
        # instead would skew every index by row_starts[i].
        rs = tl.load(row_starts + row_id)
        re = tl.load(row_ends + row_id)
        span = re - rs
        logits_ptr += row_id * stride0 + rs * stride1

        blk_id = tl.program_id(1)
        blk = span // MULTIPLE_BLOCKS_NUM
        row_start = blk * blk_id
        # Last block absorbs the remainder. When span < MULTIPLE_BLOCKS_NUM this
        # degenerates to empty ranges for blocks 0..N-2 and the whole span for the
        # last -- wasteful but correct, and the gate keeps us out of that regime.
        row_end = span if blk_id == MULTIPLE_BLOCKS_NUM - 1 else row_start + blk

        out_indices_ptr += row_id * MULTIPLE_BLOCKS_NUM * TOPK + blk_id * TOPK
        out_logits_ptr += row_id * MULTIPLE_BLOCKS_NUM * TOPK + blk_id * TOPK

        # float4 alignment is a property of the true memory offset, so it must be
        # computed from the absolute position, not the span-relative one.
        x_off_mod = (row_id * stride0 + rs + row_start) % VEC
        skip_elems = 0 if x_off_mod == 0 else VEC - x_off_mod

    s_histogram = tle.gpu.alloc(
        [NUM_BINS], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_logits = tle.gpu.alloc(
        [NUM_FILNAL_ITEMS], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_out_indices = tle.gpu.alloc(
        [TOPKP], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_cnt = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_threshold_bin_idx = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_bin_size = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_found_topk_values = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_histogram_ptr = tle.gpu.local_ptr(s_histogram, (0,))
    s_final_logits_ptr = tle.gpu.local_ptr(s_final_logits, (0,))
    s_out_indices_ptr = tle.gpu.local_ptr(s_out_indices, (0,))
    s_final_cnt_ptr = tle.gpu.local_ptr(s_final_cnt, (0,))
    s_threshold_bin_idx_ptr = tle.gpu.local_ptr(s_threshold_bin_idx, (0,))
    s_final_bin_size_ptr = tle.gpu.local_ptr(s_final_bin_size, (0,))
    s_found_topk_values_ptr = tle.gpu.local_ptr(s_found_topk_values, (0,))

    if MERGE_BLOCKS:
        s_out_logits_ptr = None
    else:
        s_out_logits = tle.gpu.alloc(
            [TOPKP], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        s_out_logits_ptr = tle.gpu.local_ptr(s_out_logits, (0,))

    _top_k_per_row_job(
        logits_ptr,
        out_indices_ptr,
        row_start,
        row_end,
        stride1,
        vocab_size,
        skip_elems,
        out_logits_ptr,
        indices_ptr,
        s_histogram_ptr,
        s_final_logits_ptr,
        s_final_cnt_ptr,
        s_threshold_bin_idx_ptr,
        s_final_bin_size_ptr,
        s_found_topk_values_ptr,
        s_out_indices_ptr,
        s_out_logits_ptr,
        TOPK=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_RADIX_FINAL=USE_RADIX_FINAL,
        HAS_TLE=True,
        MULTIPLE_BLOCKS_PER_ROW=not MERGE_BLOCKS,
        MERGE_BLOCKS=MERGE_BLOCKS,
    )


def _should_split(num_rows: int, vocab_size: int) -> bool:
    """Gate, derived from the measured occupancy table rather than guessed.

    vocab_size is used as the span bound instead of row_ends: reading the real
    per-row span would need a device sync, and over-estimating only costs
    parallelism, never correctness.
    """
    if not HAS_TLE:
        return False
    if num_rows >= SPLIT_MAX_ROWS:
        return False
    return vocab_size // SPLIT >= SPLIT_MIN_CHUNK


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for DeepSeek V4 prefill, split across blocks when the grid
    would otherwise leave the card idle.

    Falls back to the generic kernel by *calling it* -- not by claiming to in a
    docstring -- whenever the split is not worthwhile or TLE is unavailable.
    """
    vocab_size = logits.shape[1]

    if not _should_split(num_rows, vocab_size):
        return _generic_prefill(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        )

    topkp = triton.next_power_of_2(top_k)
    use_radix_final = _use_radix_final_for_prefill(vocab_size)
    device = logits.device

    out_indices_aux = torch.empty(
        (num_rows, SPLIT, top_k), device=device, dtype=torch.int32
    )
    out_logits_aux = torch.empty(
        (num_rows, SPLIT, top_k), device=device, dtype=torch.float32
    )

    # Pass 1: split each row into SPLIT chunks, local top-k in each
    _mtt_multi_block_prefill[(num_rows, SPLIT)](
        logits,
        out_indices_aux,
        row_starts,
        row_ends,
        stride0,
        stride1,
        vocab_size,
        out_logits_aux,
        None,
        TOPK=top_k,
        TOPKP=topkp,
        BLOCK_SIZE=NUM_THREADS_PER_BLOCK,
        USE_RADIX_FINAL=use_radix_final,
        MULTIPLE_BLOCKS_NUM=SPLIT,
        MERGE_BLOCKS=False,
        num_warps=NUM_THREADS_PER_BLOCK // 32,
    )

    # Pass 2: merge the SPLIT*top_k candidates back down to top_k
    _mtt_multi_block_prefill[(num_rows,)](
        out_logits_aux,
        indices,
        row_starts,
        row_ends,
        SPLIT * top_k,
        1,
        SPLIT * top_k,
        None,
        out_indices_aux,
        TOPK=top_k,
        TOPKP=topkp,
        BLOCK_SIZE=NUM_THREADS_PER_BLOCK_MERGE,
        USE_RADIX_FINAL=use_radix_final,
        MULTIPLE_BLOCKS_NUM=SPLIT,
        MERGE_BLOCKS=True,
        num_warps=NUM_THREADS_PER_BLOCK_MERGE // 32,
    )
