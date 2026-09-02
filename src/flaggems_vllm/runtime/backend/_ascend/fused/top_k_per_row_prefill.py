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

"""Ascend rewrites for top_k_per_row_prefill, kept out of the generic operator.

Seven defects in this backend stand between the generic operator and a correct
result. Every rewrite for them lives here, so the generic operator carries
nothing that exists only because of this card -- and nothing that would need
validating on NVIDIA, which cannot be done from here.

    1  tl.reduce_or absent in Triton 3.2.0        frontend trace
    2  early `return` inside an `if`              TritonToLinalgIncubated abort
    3  llvm.intr.assume has no assembly form      ConvertLinalgRToBinary
    4  uint32 as a pointer offset                 BiShengHIR abort
    5  atomic returns non-unique per-lane values  silent wrong results
    6  the scan needs more UB than BLOCK=512 fits ub overflow
    7  tl.uint16 >> lowered as an ARITHMETIC shift silent, half the input lost

The substitution works by REBINDING onto the generic module rather than by
redefining here. A Triton jit function resolves the jit functions it calls
through its OWN module's globals, so a copy defined in this file would simply be
ignored -- _distribute_to_bins lives over there and would keep seeing the
generic _extract_bin_idx. Assigning the module attribute before anything is
traced is what makes it take, and it is verified: with the generic operator
reverted, this backend's histogram and output are exact.

Three functions are copied because they must differ. _process_histogram_step,
at 395 lines, deliberately is NOT: _compact_pos derives the scalar counter from
the broadcast pointer it is already given, so _process_bins keeps its signature
and its caller needs no change.

Correct but slow. One histogram pass costs 76 ms against torch.topk's 0.78 ms
for the whole operation, because without TLE the 2048-bin histogram lives in
global memory and every element pays a global atomic. That is the algorithm
being mismatched to the backend, not something these rewrites cause.
"""

import os
from importlib import import_module

import torch

import triton
import triton.language as tl

from flaggems_vllm import runtime

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")

# Bound into this module so the copied functions below resolve them. These are
# unchanged; only the ones copied further down needed to differ.
_convert_to_uint32 = _generic._convert_to_uint32
_distribute_to_bins = _generic._distribute_to_bins
_final_select_radix = _generic._final_select_radix
_process_histogram_step = _generic._process_histogram_step
SORTING_ALGORITHM_THRESHOLD = _generic.SORTING_ALGORITHM_THRESHOLD
NUM_BINS = _generic.NUM_BINS
NUM_FILNAL_ITEMS = _generic.NUM_FILNAL_ITEMS
_num_warps = _generic._num_warps
non_tle_top_k_per_row_prefill = _generic.non_tle_top_k_per_row_prefill

# 6. The scan materialises prefix sums the atomic never needed, so its tiles
# cost more unified buffer: at 512 with VEC=4 the backend wants 2589952 bits
# against 1572864 and the compile does not finish. 256 fits and builds in ~20 s.
SCAN_BLOCK_SIZE = int(os.environ.get("FLAGGEMS_SCAN_BLOCK_SIZE", "256"))


# 1. tl.reduce_or does not exist in Triton 3.2.0. The one call site is inside
# _process_histogram_step, which is 395 lines and otherwise needs no change, so
# supplying the missing builtin is far cheaper than copying it. The guard means
# this only ever ADDS a name the build does not have; where the symbol exists it
# is left alone.
#
# The call is a block-wide "did any lane find it" -- what vLLM's CUDA writes as
# __syncthreads_or -- and a max over the mask as int32 says the same thing with
# primitives every build has.
if not hasattr(tl, "reduce_or"):

    @triton.jit
    def _reduce_or(x, axis):
        return tl.max(x.to(tl.int32), axis=axis) != 0

    tl.reduce_or = _reduce_or


# 3. tl.assume lowers to llvm.intr.assume, which this backend's IR round trip
# cannot print. It is a pure optimisation hint, so dropping it changes nothing
# but the code the compiler is allowed to assume.
@triton.jit
def _assume(cond):
    pass


# 5. tl.atomic_add accumulates correctly here but returns non-unique per-lane
# old values -- 512 lanes adding 1 leave the counter at 512 while the returns
# hold 65 distinct numbers. Used as store addresses those collide, and a masked
# store with duplicate lane addresses is dropped, so the output came out with
# 7 of 64 entries, every one of them valid.
#
# The counters here are per-row and the grid is one program per row, so the
# atomic was never serialising across programs on this path: it only supplied
# unique offsets within a tile and a running base across tiles. An exclusive
# prefix sum gives the first and a read-modify-write of the counter gives the
# second.
#
# Takes the BROADCAST pointer only, and reduces it to a scalar itself. That is
# what lets _process_bins keep its signature, which in turn is what keeps the
# 395-line _process_histogram_step out of this file.
@triton.jit
def _compact_pos(cnt_ptrs, ones, take):
    t = take.to(tl.int32)
    if len(t.shape) == 2:
        # Vectorised tiles are [BLOCK, VEC]. Reducing over axis 0 alone leaves a
        # [VEC] block, so do it in two levels: a prefix down each column plus
        # the total of every column to its left, numbering the tile
        # column-major. Any order will do; only uniqueness and density matter.
        col_tot = tl.sum(t, axis=0)
        col_excl = tl.cumsum(col_tot, axis=0) - col_tot
        excl = (tl.cumsum(t, axis=0) - t) + col_excl[None, :]
        total = tl.sum(col_tot, axis=0)
        cur = tl.load(cnt_ptrs)
        base = tl.min(tl.min(cur, axis=0), axis=0)
        first = (tl.arange(0, t.shape[0])[:, None] == 0) & (
            tl.arange(0, t.shape[1])[None, :] == 0
        )
    else:
        excl = tl.cumsum(t, axis=0) - t
        total = tl.sum(t, axis=0)
        cur = tl.load(cnt_ptrs)
        base = tl.min(cur, axis=0)
        first = tl.arange(0, t.shape[0]) == 0
    # Barriers, or lanes in different warps read the same base and are handed
    # the same destinations. Triton needs an explicit barrier for a store-then-
    # load of one address inside a single program; without it this wrote 511 of
    # 512 entries on a few percent of rows.
    tl.debug_barrier()
    tl.store(cnt_ptrs, base + total, mask=first)
    tl.debug_barrier()
    return base + excl


# Site 3 needs this variant: there the counter address is `s_histogram_ptr +
# bin_idx`, which is per-lane, not broadcast. Only the taken lanes have
# bin_idx == threshold_bin_idx; the rest point at other bins, so reducing a load
# across all lanes would fold in histogram entries that are not the counter and
# the single-lane write-back could land on the wrong bin. The caller knows the
# scalar address, so it passes it.
@triton.jit
def _compact_pos_scalar(cnt_scalar_ptr, ones, take):
    t = take.to(tl.int32)
    if len(t.shape) == 2:
        col_tot = tl.sum(t, axis=0)
        col_excl = tl.cumsum(col_tot, axis=0) - col_tot
        excl = (tl.cumsum(t, axis=0) - t) + col_excl[None, :]
        total = tl.sum(col_tot, axis=0)
    else:
        excl = tl.cumsum(t, axis=0) - t
        total = tl.sum(t, axis=0)
    tl.debug_barrier()
    base = tl.load(cnt_scalar_ptr)
    tl.debug_barrier()
    tl.store(cnt_scalar_ptr, base + total)
    tl.debug_barrier()
    return base + excl


@triton.jit
def _extract_bin_idx(x, in_range, pattern, STEP: tl.constexpr):
    is_partial_match = in_range
    if STEP == 0:
        h = x.to(tl.float16)
        bits = h.to(tl.uint16, bitcast=True)
        sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
        sign_set = (bits & sign_mask) != 0
        inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
        mapped = tl.where(sign_set, bits, inv)
        bin_idx = (mapped.to(tl.int32) & 0xFFFF) >> 5
    else:
        bits = _convert_to_uint32(x)
        if STEP == 1:
            bin_idx = (bits >> 21).to(tl.int32)
        elif STEP == 2:
            bin_idx = ((bits >> 10) & 0x7FF).to(tl.int32)
            is_partial_match &= ((bits ^ pattern) >> 21) == 0
        elif STEP == 3:
            bin_idx = (bits & 0x3FF).to(tl.int32)
            is_partial_match &= ((bits ^ pattern) >> 10) == 0
    return bin_idx, is_partial_match


@triton.jit
def _process_bins(
    logits,
    in_range,
    ones,
    offs,  # row_start based
    found_topk_values_ptrs,
    final_cnt_ptrs,
    logit_pattern,
    threshold_bin_idx,
    write_directly,
    use_final,
    row_start,
    indices_ptr,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_out_indices_ptr,
    s_out_logits_ptr,
    STEP: tl.constexpr,
    TOPK: tl.constexpr,
    MULTIPLE_BLOCKS_PER_ROW: tl.constexpr,
    MERGE_BLOCKS: tl.constexpr,
):
    NUM_FINAL_ITEMS: tl.constexpr = 2048

    bin_idx, is_partial_match = _extract_bin_idx(
        logits,
        in_range,
        logit_pattern,
        STEP=STEP,
    )
    take_lt = is_partial_match & (bin_idx < threshold_bin_idx) & write_directly
    out_pos_lt = _compact_pos(found_topk_values_ptrs, ones, take_lt)
    if MERGE_BLOCKS:
        indices = tl.load(
            indices_ptr + offs,
            mask=take_lt,
        )
        tl.store(
            s_out_indices_ptr + out_pos_lt,
            indices,
            mask=take_lt,
        )
    elif MULTIPLE_BLOCKS_PER_ROW:
        tl.store(
            s_out_indices_ptr + out_pos_lt,
            (offs + row_start).to(tl.int32),
            mask=take_lt,
        )
        tl.store(
            s_out_logits_ptr + out_pos_lt,
            logits,
            mask=take_lt,
        )
    else:
        tl.store(
            s_out_indices_ptr + out_pos_lt,
            offs.to(tl.int32),
            mask=take_lt,
        )

    if STEP < 3:
        if use_final:
            take_eq_final = is_partial_match & (bin_idx == threshold_bin_idx)
            final_pos = _compact_pos(final_cnt_ptrs, ones, take_eq_final)
            tl.store(
                s_final_logits_ptr + final_pos,
                logits,
                mask=take_eq_final & (final_pos < NUM_FINAL_ITEMS),
            )
            # s_histogram_ptr being used for indices in final sort
            if MERGE_BLOCKS:
                indices = tl.load(
                    indices_ptr + offs,
                    mask=take_eq_final & (final_pos < NUM_FINAL_ITEMS),
                )
                tl.store(
                    s_histogram_ptr + final_pos,
                    indices,
                    mask=take_eq_final & (final_pos < NUM_FINAL_ITEMS),
                )
            elif MULTIPLE_BLOCKS_PER_ROW:
                tl.store(
                    s_histogram_ptr + final_pos,
                    (offs + row_start).to(tl.int32),
                    mask=take_eq_final & (final_pos < NUM_FINAL_ITEMS),
                )
            else:
                tl.store(
                    s_histogram_ptr + final_pos,
                    offs.to(tl.int32),
                    mask=take_eq_final & (final_pos < NUM_FINAL_ITEMS),
                )
    else:
        take_eq = is_partial_match & (bin_idx == threshold_bin_idx)
        # s_histogram_ptr being used for exclude prefix sum
        out_pos_eq = _compact_pos_scalar(
            s_histogram_ptr + threshold_bin_idx, ones, take_eq
        )
        if MERGE_BLOCKS:
            indices = tl.load(
                indices_ptr + offs,
                mask=take_eq & (out_pos_eq < TOPK),
            )
            tl.store(
                s_out_indices_ptr + out_pos_eq,
                indices,
                mask=take_eq & (out_pos_eq < TOPK),
            )
        elif MULTIPLE_BLOCKS_PER_ROW:
            tl.store(
                s_out_indices_ptr + out_pos_eq,
                (offs + row_start).to(tl.int32),
                mask=take_eq & (out_pos_eq < TOPK),
            )
            tl.store(
                s_out_logits_ptr + out_pos_eq,
                logits,
                mask=take_eq & (out_pos_eq < TOPK),
            )
        else:
            tl.store(
                s_out_indices_ptr + out_pos_eq,
                offs.to(tl.int32),
                mask=take_eq & (out_pos_eq < TOPK),
            )


@triton.jit
def _top_k_per_row_job(
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
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
    HAS_TLE: tl.constexpr,
    MULTIPLE_BLOCKS_PER_ROW: tl.constexpr,
    MERGE_BLOCKS: tl.constexpr,
):
    NUM_FINAL_ITEMS: tl.constexpr = 2048

    assume_aligned = (
        (row_start == 0)
        & (row_end == vocab_size)
        & (stride1 == 1)
        & ((vocab_size % BLOCK_SIZE) == 0)
    )
    if assume_aligned:
        _assume(row_start == 0)
        _assume(row_end == vocab_size)
        _assume(stride1 == 1)
        vocab_size = tl.multiple_of(vocab_size, BLOCK_SIZE)
    elif stride1 == 1:
        _assume(stride1 == 1)

    lane = tl.arange(0, BLOCK_SIZE)
    row_len = row_end - row_start
    if row_len <= TOPK:
        chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
        for chunk_idx in tl.range(0, chunks):
            pos = chunk_idx * BLOCK_SIZE + lane
            take_row = pos < row_len
            if MULTIPLE_BLOCKS_PER_ROW:
                tl.store(
                    out_indices_ptr + pos,
                    (pos + row_start).to(tl.int32),
                    mask=take_row,
                )
                logits = tl.load(logits_ptr + pos + row_start, mask=take_row)
                tl.store(out_logits_ptr + pos, logits, mask=take_row)
            else:
                tl.store(
                    out_indices_ptr + pos,
                    pos.to(tl.int32),
                    mask=take_row,
                )
            take_pad = (pos >= row_len) & (pos < TOPK)
            tl.store(out_indices_ptr + pos, -1, mask=take_pad)
            if MULTIPLE_BLOCKS_PER_ROW:
                tl.store(out_logits_ptr + pos, float("-inf"), mask=take_pad)
    else:
        tl.store(s_final_cnt_ptr, 0)
        tl.store(s_found_topk_values_ptr, 0)
        tl.debug_barrier()
        logit_pattern = tl.zeros((), dtype=tl.uint32)
        continue_to_next_step = tl.full((), True, dtype=tl.int1)
        threshold_bin_idx = tl.full((), -1, dtype=tl.int32)
        for step_idx in tl.static_range(0, 4):
            if continue_to_next_step:
                (
                    continue_to_next_step,
                    logit_pattern,
                    threshold_bin_idx,
                ) = _process_histogram_step(
                    logits_ptr,
                    row_start,
                    row_end,
                    stride1,
                    vocab_size,
                    skip_elems,
                    indices_ptr,
                    logit_pattern,
                    threshold_bin_idx,
                    assume_aligned,
                    s_histogram_ptr,
                    s_final_logits_ptr,
                    s_final_cnt_ptr,
                    s_threshold_bin_idx_ptr,
                    s_final_bin_size_ptr,
                    s_found_topk_values_ptr,
                    s_out_indices_ptr,
                    s_out_logits_ptr,
                    STEP=step_idx,
                    TOPK=TOPK,
                    BLOCK_SIZE=BLOCK_SIZE,
                    HAS_TLE=HAS_TLE,
                    MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                    MERGE_BLOCKS=MERGE_BLOCKS,
                )

        if not continue_to_next_step:
            if USE_RADIX_FINAL and HAS_TLE:
                _final_select_radix(
                    s_histogram_ptr,
                    s_final_logits_ptr,
                    s_final_cnt_ptr,
                    s_found_topk_values_ptr,
                    s_out_indices_ptr,
                    s_out_logits_ptr,
                    TOPK=TOPK,
                    BLOCK_SIZE=BLOCK_SIZE,
                    MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                )
            else:
                base_idx = tl.load(s_found_topk_values_ptr)
                # Guard against stale/oversized counts to avoid out-of-bounds accesses
                # in the shared-memory final buffers.
                final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), NUM_FINAL_ITEMS)
                sort_chunks = tl.cdiv(final_cnt, BLOCK_SIZE)
                for sort_chunk in tl.range(0, sort_chunks):
                    pos = sort_chunk * BLOCK_SIZE + lane
                    valid = pos < final_cnt
                    logit_i = tl.load(
                        s_final_logits_ptr + pos,
                        mask=valid,
                        other=0,
                    )
                    out_rank = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
                    for j in tl.range(0, final_cnt):
                        logit_j = tl.load(s_final_logits_ptr + j)
                        better = (logit_i < logit_j) | ((logit_i == logit_j) & (pos < j))
                        out_rank = out_rank + (valid & better).to(tl.int32)
                    dst_pos = base_idx + out_rank
                    take = valid & (dst_pos < TOPK)
                    idx_i = tl.load(
                        s_histogram_ptr + pos,
                        mask=take,
                        other=0,
                    )
                    tl.store(s_out_indices_ptr + dst_pos, idx_i, mask=take)
                    if MULTIPLE_BLOCKS_PER_ROW:
                        tl.store(s_out_logits_ptr + dst_pos, logit_i, mask=take)
                tl.debug_barrier()

        # out_indices_ptr is identical to s_out_indices_ptr for non-tle
        if HAS_TLE:
            flush_chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
            for flush_chunk in tl.static_range(flush_chunks):
                pos = flush_chunk * BLOCK_SIZE + lane
                mask = pos < TOPK
                out_vals = tl.load(s_out_indices_ptr + pos, mask=mask, other=-1)
                tl.store(out_indices_ptr + pos, out_vals, mask=mask)
                if MULTIPLE_BLOCKS_PER_ROW:
                    split_logits = tl.load(
                        s_out_logits_ptr + pos, mask=mask, other=float("-inf")
                    )
                    tl.store(out_logits_ptr + pos, split_logits, mask=mask)


# Rebind BEFORE anything is traced. Callers in the generic module resolve these
# names through that module's globals, so the assignments are what make the
# replacements reach _distribute_to_bins, _process_histogram_step and
# non_tle_top_k_per_row_prefill.
_generic._extract_bin_idx = _extract_bin_idx
_generic._process_bins = _process_bins
_generic._top_k_per_row_job = _top_k_per_row_job

def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """The generic host's non-TLE branch, launched at this backend's block size.

    Delegating to the generic host and mutating its NUM_THREADS_PER_BLOCK would
    have been shorter, but that writes into the module we are trying to leave
    alone. This launches the same kernel with the same scratch, only with the
    block size the scan path fits in unified buffer, so nothing outside this
    file changes at all.

    Only the non-TLE branch exists here: HAS_TLE is False on this backend --
    Triton 3.2.0 ships no tle module -- so the other branch is unreachable.

    Registering under this name also makes the suite's _OVERRIDE_ACTIVE guard
    run instead of skip.
    """
    vocab_size = logits.shape[1]
    assert num_rows == logits.shape[0]
    device = logits.device
    s_histogram_ptr = torch.empty(
        (num_rows, NUM_BINS), device=device, dtype=torch.int32
    )
    s_final_logits_ptr = torch.empty(
        (num_rows, NUM_FILNAL_ITEMS), device=device, dtype=torch.float32
    )
    s_final_cnt_ptr = torch.empty((num_rows,), device=device, dtype=torch.int32)
    s_threshold_bin_idx_ptr = torch.empty(
        (num_rows,), device=device, dtype=torch.int32
    )
    s_final_bin_size_ptr = torch.empty(
        (num_rows,), device=device, dtype=torch.int32
    )
    s_found_topk_values_ptr = torch.empty(
        (num_rows,), device=device, dtype=torch.int32
    )
    non_tle_top_k_per_row_prefill[(num_rows,)](
        logits,
        indices,
        row_starts,
        row_ends,
        stride0,
        stride1,
        vocab_size,
        s_histogram_ptr,
        s_final_logits_ptr,
        s_final_cnt_ptr,
        s_threshold_bin_idx_ptr,
        s_final_bin_size_ptr,
        s_found_topk_values_ptr,
        TOPK=top_k,
        BLOCK_SIZE=SCAN_BLOCK_SIZE,
        ROW_OFFSET=0,
        num_warps=_num_warps(SCAN_BLOCK_SIZE),
    )
