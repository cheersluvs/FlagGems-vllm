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

"""Allocate output slots with a scan instead of an atomic, on MetaX C550.

`_process_bins` opens by handing each selected element an output slot through
`tl.atomic_add` on ONE counter. That fires about top_k times per row, all to a
single address, and on this card it serialises: ablating it out of the shipped
kernel saved **7.75 of 19.4 microseconds per program**, which is the whole of a
cost term measured at ~17 ns per unit of top_k -- proportional to top_k,
independent of vocabulary, independent of input distribution.

How that term was found is worth recording, because five earlier attributions
were wrong and each was refuted by measurement rather than by argument: the
O(n^2) rank sort (13x its loop bound changed nothing), bin walking (its slope
grows with vocabulary where the model says it must shrink), the output stores
(disabling all seven of them moved 0.1 us), the 65536 radix-final gate (no step
in the curve) and a 2048-bin `tl.histogram` cliff, which is real on this card
but which this operator never triggers, because it builds its histogram from
global atomics instead.

The replacement is not new code. The Ascend override already computes these
slots with a prefix sum, for a completely different reason -- there
`tl.atomic_add` returns non-unique values per lane, so the atomic was wrong
rather than slow. Only one program handles a row, so the counter has no
cross-program contention to protect and a scan is free to replace it:
measured, a cumsum-derived position costs 0.8 ns per item against the atomic's
17.

Nothing else changes. `_process_bins` is rebound onto the generic module before
anything is traced, so the generic operator carries no diff.
"""

import functools  # noqa: F401  (kept for parity with the decode override)
from importlib import import_module

import triton
import triton.language as tl

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")

_extract_bin_idx = _generic._extract_bin_idx
NUM_FILNAL_ITEMS = _generic.NUM_FILNAL_ITEMS


# On Ascend this replacement was written because the atomic RETURNED the wrong
# values; here it is written because it is slow. The code is the same either
# way, and the reason it is safe is the same too:
#
# The counters here are per-row and the grid is one program per row, so the
# atomic was never serialising across programs on this path: it only supplied
# unique offsets within a tile and a running base across tiles. An exclusive
# prefix sum gives the first and a read-modify-write of the counter gives the
# second. Measured on the C550: 0.8 ns per item against the atomic's 17.
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


# Rebind before anything is traced: a jit function resolves the jit functions it
# calls through its OWN module's globals, so _process_histogram_step over there
# would otherwise keep seeing the atomic version.
_generic._process_bins = _process_bins

top_k_per_row_prefill = _generic.top_k_per_row_prefill
