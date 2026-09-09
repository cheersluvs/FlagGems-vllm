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

"""Build prefill's histogram with tl.histogram instead of per-element atomics.

The generic operator counts with one GLOBAL `tl.atomic_add` per element into a
2048-entry buffer. On this card atomic address contention is a first-order
cost: forcing 4096 of those onto a single address cost 31 us against a whole
operator of 19.4 us per program.

`tl.histogram` accumulates in registers, so the whole row needs one flush
instead of one atomic per element. The accumulator has to outlive every tile
loop -- flushing per tile would be slower than what it replaces whenever
BLOCK_SIZE < NBINS.

This is the Ascend override's implementation, unchanged. It was written there
because that backend has no way to scatter into unified buffer at all, and it
measured 3.0x on the histogram pass; on Moore Threads the same code measured
1.42x. Whether it wins HERE is an open question and the reason this file
exists: a microbenchmark put `tl.histogram` at 2048 bins near 7.8 us, which is
the same order as the atomics it replaces -- but that microbenchmark also wrote
2048 counters back to global memory per program, so it may have been pricing
the store rather than the histogram.

What is NOT taken from Ascend is `_extract_bin_idx`: that copy carries
workarounds for two silent miscompilations on that card (uint16 and uint32 `>>`
lowered as arithmetic shifts) which do not apply here and would be a behaviour
change. It is bound to the generic one below, along with everything else the
step calls.

Nothing else changes. `_process_histogram_step` is rebound onto the generic
module before anything is traced, so the generic operator carries no diff.
"""

from importlib import import_module

import triton
import triton.language as tl

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")

# Resolved through THIS module's globals by the jit functions below, so bind
# every one of them to the generic version explicitly.
_extract_bin_idx = _generic._extract_bin_idx
_distribute_to_bins = _generic._distribute_to_bins
_process_bins = _generic._process_bins


@triton.jit
def _hist_counts(
    x,
    in_range,
    logit_pattern,
    STEP: tl.constexpr,
    NBINS: tl.constexpr,
    NELEM: tl.constexpr,
):
    """Count one 1-D tile with tl.histogram instead of one global atomic per element.

    Every tile reaching here is 1-D by construction.  Reshaping the vectorised
    [BLOCK_SIZE, VEC] tile instead would be the obvious route and does not
    compile: a [256, 4] i32 buffer has a 16-byte row stride, the UB allocator
    wants 32, and BiShengIR rejects it with "cannot align 0 axis".  Those loads
    address a contiguous run anyway, so the counting loops read them as one wide
    1-D tile and no reshape is needed.  The loops that feed _process_bins keep
    their 2-D tiles -- they need offs to build indices.

    Two more measured facts shape this.  There is no masked tl.histogram; and
    parking dead lanes on an out-of-range value does NOT discard them here --
    511 were counted where only 259 lanes were live, i.e. they fold into a real
    bin and silently corrupt it.  So dead lanes are parked in bin 0 and bin 0 is
    corrected by how many were parked: exact, and every shape stays a power of
    two (NBINS + 1 would not be).

    in_range is a plain Python True at five of the seven call sites, so the mask
    is broadcast to a tensor before it can be summed.
    """
    bin_idx, ok = _extract_bin_idx(x, in_range, logit_pattern, STEP=STEP)
    m = tl.full([NELEM], 1, tl.int1) & ok
    counts = tl.histogram(tl.where(m, bin_idx, 0), NBINS)
    parked = NELEM - tl.sum(m.to(tl.int32), axis=0)
    return counts - tl.where(tl.arange(0, NBINS) == 0, parked, 0)


@triton.jit
def _process_histogram_step(
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
    STEP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_TLE: tl.constexpr,
    MULTIPLE_BLOCKS_PER_ROW: tl.constexpr,
    MERGE_BLOCKS: tl.constexpr,
):
    VEC: tl.constexpr = 4
    NUM_FINAL_ITEMS: tl.constexpr = 2048
    RADIX11_SIZE: tl.constexpr = 2048
    RADIX11_MASK: tl.constexpr = 0x7FF
    RADIX10_SIZE: tl.constexpr = 1024

    lane = tl.arange(0, BLOCK_SIZE)
    # The two counting loops address a contiguous run, so they read one wide
    # 1-D tile: a [BLOCK_SIZE, VEC] tile cannot be reshaped on this backend
    # (the UB allocator rejects the 16-byte row stride).  The loops feeding
    # _process_bins keep their 2-D tiles, which they need for offs.
    wide = tl.arange(0, BLOCK_SIZE * VEC)
    vec = tl.arange(0, VEC)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    ones_vec_2d = tl.full([BLOCK_SIZE, VEC], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    zeros_vec_2d = tl.zeros([BLOCK_SIZE, VEC], dtype=tl.int32)

    # One histogram for the whole row, flushed once, instead of one global
    # atomic per element: measured 3.0x on this card (51.0 -> 17.2 ms for
    # 64x131072 into 2048 bins).  The accumulator has to outlive every tile
    # loop below -- flushing per tile would be slower than what it replaces
    # whenever BLOCK_SIZE < NBINS.
    NBINS: tl.constexpr = RADIX10_SIZE if STEP == 3 else RADIX11_SIZE
    acc = tl.zeros([NBINS], tl.int32)

    threshold_rounds: tl.constexpr = (
        RADIX10_SIZE // BLOCK_SIZE if STEP == 3 else RADIX11_SIZE // BLOCK_SIZE
    )
    for clear_round in tl.static_range(0, threshold_rounds):
        clear_bins = clear_round * BLOCK_SIZE + lane
        tl.store(s_histogram_ptr + clear_bins, 0)
    tl.debug_barrier()

    if STEP == 2:
        logit_pattern = (threshold_bin_idx.to(tl.uint32) & RADIX11_MASK) << 21
    elif STEP == 3:
        logit_pattern |= (threshold_bin_idx.to(tl.uint32) & RADIX11_MASK) << 10

    if assume_aligned:
        n_tiles = tl.cdiv(vocab_size, BLOCK_SIZE)
        n_vec_full = vocab_size // (BLOCK_SIZE * VEC)
        rem_tiles = (vocab_size - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            offs = t * BLOCK_SIZE * VEC + wide
            x_vec = tl.load(logits_ptr + offs)
            acc += _hist_counts(
                x_vec,
                True,
                logit_pattern,
                STEP=STEP,
                NBINS=NBINS,
                NELEM=BLOCK_SIZE * VEC,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(logits_ptr + offs)
            acc += _hist_counts(
                x,
                True,
                logit_pattern,
                STEP=STEP,
                NBINS=NBINS,
                NELEM=BLOCK_SIZE,
            )
    elif stride1 == 1:
        aligned_row_ptr = tl.multiple_of(logits_ptr + row_start + skip_elems, VEC * 4)
        row_len = row_end - row_start - skip_elems
        n_vec_full = row_len // (BLOCK_SIZE * VEC)
        rem_tiles = (row_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        rem_elems = row_len % BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            offs = t * BLOCK_SIZE * VEC + wide
            x_vec = tl.load(aligned_row_ptr + offs)
            acc += _hist_counts(
                x_vec,
                True,
                logit_pattern,
                STEP=STEP,
                NBINS=NBINS,
                NELEM=BLOCK_SIZE * VEC,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(aligned_row_ptr + offs)
            acc += _hist_counts(
                x,
                True,
                logit_pattern,
                STEP=STEP,
                NBINS=NBINS,
                NELEM=BLOCK_SIZE,
            )
        if skip_elems > 0:
            offs = lane
            in_range = lane < skip_elems
            x = tl.load(
                logits_ptr + row_start + offs, mask=in_range, other=float("-inf")
            )
            acc += _hist_counts(
                x,
                in_range,
                logit_pattern,
                STEP=STEP,
                NBINS=NBINS,
                NELEM=BLOCK_SIZE,
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(aligned_row_ptr + offs, mask=in_range, other=float("-inf"))
            acc += _hist_counts(
                x,
                in_range,
                logit_pattern,
                STEP=STEP,
                NBINS=NBINS,
                NELEM=BLOCK_SIZE,
            )
    else:
        row_len = row_end - row_start
        n_tiles = tl.cdiv(row_len, BLOCK_SIZE)
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = offs < row_len
            x = tl.load(
                logits_ptr + row_start + offs * stride1,
                mask=in_range,
                other=float("-inf"),
            )
            acc += _hist_counts(
                x,
                in_range,
                logit_pattern,
                STEP=STEP,
                NBINS=NBINS,
                NELEM=BLOCK_SIZE,
            )
    # The one write the whole distribution phase costs.  Still atomic: with
    # MULTIPLE_BLOCKS_PER_ROW several programs share this histogram.
    tl.atomic_add(
        s_histogram_ptr + tl.arange(0, NBINS),
        acc,
        sem="relaxed",
        scope="cta",
    )

    last_value = tl.load(s_found_topk_values_ptr)
    tl.debug_barrier()

    threshold_bin_ptrs = s_threshold_bin_idx_ptr + zeros
    final_bin_size_ptrs = s_final_bin_size_ptr + zeros
    threshold_found = tl.full((), False, dtype=tl.int1)
    for round_idx in tl.static_range(0, threshold_rounds):
        if not threshold_found:
            bins = round_idx * BLOCK_SIZE + lane
            counts = tl.load(s_histogram_ptr + bins)
            if HAS_TLE:
                prefix_sum, counts_total = tle.cumsum(counts, axis=0, reverse=False)
            else:
                counts_total = tl.sum(counts)
                prefix_sum = counts_total - tl.cumsum(counts, axis=0, reverse=True)
            prefix_sum = prefix_sum + last_value
            total_sum = last_value + counts_total
            next_prefix_sum = prefix_sum + counts
            threshold_mask = (prefix_sum < TOPK) & (next_prefix_sum >= TOPK)
            threshold_bin = bins
            threshold_bin_size = next_prefix_sum - prefix_sum
            if STEP == 3:
                tl.store(s_histogram_ptr + bins, prefix_sum)
            tl.store(threshold_bin_ptrs, threshold_bin, mask=threshold_mask)
            tl.store(final_bin_size_ptrs, threshold_bin_size, mask=threshold_mask)
            found_round = tl.reduce_or(threshold_mask, axis=0)
            threshold_found = found_round
            last_value = total_sum

    tl.debug_barrier()
    threshold_bin_idx = tl.load(s_threshold_bin_idx_ptr)
    final_bin_size = tl.load(s_final_bin_size_ptr)
    use_final = final_bin_size <= NUM_FINAL_ITEMS
    write_directly = ((STEP == 0) & (final_bin_size <= NUM_FINAL_ITEMS)) | (STEP >= 1)

    found_ptrs = s_found_topk_values_ptr + zeros
    final_cnt_ptrs = s_final_cnt_ptr + zeros
    if assume_aligned:
        found_ptrs_vec_2d = s_found_topk_values_ptr + zeros_vec_2d
        final_cnt_ptrs_vec_2d = s_final_cnt_ptr + zeros_vec_2d
        n_tiles = tl.cdiv(vocab_size, BLOCK_SIZE)
        n_vec_full = vocab_size // (BLOCK_SIZE * VEC)
        rem_tiles = (vocab_size - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(logits_ptr + offs)
            _process_bins(
                x_vec,
                True,
                ones_vec_2d,
                offs,
                found_ptrs_vec_2d,
                final_cnt_ptrs_vec_2d,
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
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(logits_ptr + offs)
            _process_bins(
                x,
                True,
                ones,
                offs,
                found_ptrs,
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
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
    elif stride1 == 1:
        found_ptrs_vec_2d = s_found_topk_values_ptr + zeros_vec_2d
        final_cnt_ptrs_vec_2d = s_final_cnt_ptr + zeros_vec_2d
        aligned_row_ptr = tl.multiple_of(logits_ptr + row_start + skip_elems, VEC * 4)
        row_len = row_end - row_start - skip_elems
        n_vec_full = row_len // (BLOCK_SIZE * VEC)
        rem_tiles = (row_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE
        rem_elems = row_len % BLOCK_SIZE
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(aligned_row_ptr + offs)
            _process_bins(
                x_vec,
                True,
                ones_vec_2d,
                offs + skip_elems,
                found_ptrs_vec_2d,
                final_cnt_ptrs_vec_2d,
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
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(aligned_row_ptr + offs)
            _process_bins(
                x,
                True,
                ones,
                offs + skip_elems,
                found_ptrs,
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
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
        if skip_elems > 0:
            offs = lane
            in_range = lane < skip_elems
            x = tl.load(
                logits_ptr + row_start + offs, mask=in_range, other=float("-inf")
            )
            _process_bins(
                x,
                in_range,
                ones,
                offs,
                found_ptrs,
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
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            x = tl.load(aligned_row_ptr + offs, mask=in_range, other=float("-inf"))
            _process_bins(
                x,
                in_range,
                ones,
                offs + skip_elems,
                found_ptrs,
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
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
    else:
        row_len = row_end - row_start
        n_tiles = tl.cdiv(row_len, BLOCK_SIZE)
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = offs < row_len
            x = tl.load(
                logits_ptr + row_start + offs * stride1,
                mask=in_range,
                other=float("-inf"),
            )
            _process_bins(
                x,
                in_range,
                ones,
                offs,
                found_ptrs,
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
                STEP=STEP,
                TOPK=TOPK,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                MERGE_BLOCKS=MERGE_BLOCKS,
            )
    tl.debug_barrier()
    return final_bin_size > NUM_FINAL_ITEMS, logit_pattern, threshold_bin_idx


# Rebind BEFORE anything is traced. Callers in the generic module resolve these
# names through that module's globals, so the assignments are what make the
# replacements reach _distribute_to_bins, _process_histogram_step and
# non_tle_top_k_per_row_prefill.


# Before anything is traced: a jit function resolves its callees through its
# OWN module's globals, so the generic kernel would otherwise keep calling the
# generic step.
_generic._process_histogram_step = _process_histogram_step

top_k_per_row_prefill = _generic.top_k_per_row_prefill
