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

"""top_k_per_row_prefill on Hygon BW1000: allocate output slots by prefix sum
where a tile selects many elements, by per-element atomic where it selects few.

WHY. prefill loses on all seven benchmark shapes against vLLM's C++ kernel
here (geomean 0.36), worst on the small-vocabulary ones. Its per-program cost
fits base + ~19 ns x top_k + ~1.3 ns x vocab, and the k term is
`tl.atomic_add(found_topk_values_ptrs, ones, mask=take_lt)` in _process_bins:
one atomic per selected element, all to one address, ~12 ns each on this card
(the ablation removed 7.55 of 18.07 us per program with it). Shared memory does
not help here -- smem scatter atomics measured 2.5-3.5x slower than global.

WHAT. A prefix sum over the tile's take-mask gives every selected element a
distinct offset in lane order; one atomic of the tile's COUNT to the same
address gives the base. That costs a flat ~0.4 us per 512-lane tile whatever
the tile selects, against ~12 ns per selected element for the atomics, so it
wins above a crossover and loses below it. Measured on 512-lane tiles
(tools/hygon_slot_alloc_cost.py), allocation cost per program:

    selected/tile   atomic   prefix-sum   adaptive
               8      1.56        4.73       3.00
              64      6.61        4.85       4.99
             256     27.02        4.79       4.90

The operator's main tiles are [512, VEC=4] = 2048 elements, so its small-vocab
shapes select ~200-256 per tile and land deep in the prefix-sum region, while
(64, 129280) with k=1024 selects ~16 and stays on atomics.

WHY ADAPTIVE, AND WHY IN THE KERNEL. The choice cannot be made per shape from
the host: Triton resolves module globals at compile time and caches the kernel,
so rebinding a helper between calls silently keeps whichever variant compiled
first. The tile's own count is needed by the prefix sum anyway, so branch on it.

Nothing else in _process_bins changes -- the copy below is generic's own, and
the import-time assert refuses to install it if generic's atomic ever changes
shape. FLAGGEMS_HYGON_TOPK_SLOTSCAN=0 keeps the generic function, which is how
to A/B this on one box.
"""

import os
from importlib import import_module

import triton
import triton.language as tl

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
_extract_bin_idx = _generic._extract_bin_idx

# Selected elements per tile above which the prefix sum is cheaper than one
# atomic per element; the measured crossover on 512-lane tiles is ~48.
SLOT_SCAN_MIN = tl.constexpr(48)


@triton.jit
def _alloc_slots(ptrs, ones, take):
    """Same contract as tl.atomic_add(ptrs, ones, mask=take): every taken lane
    gets a distinct slot, returned in a tensor shaped like `take`. `ptrs` all
    point at one counter. Works on 1-D tiles and on [BLOCK, VEC] tiles."""
    ti = take.to(tl.int32)
    flat = tl.reshape(ti, (ti.numel,))
    total = tl.sum(flat, axis=0)
    if total >= SLOT_SCAN_MIN:
        # One atomic, on lane 0 only, adds the whole count and returns the
        # counter's previous value -- the base for this tile's run of slots.
        first = tl.arange(0, ti.numel) == 0
        flat_ptrs = tl.reshape(ptrs, (ti.numel,))
        prev = tl.atomic_add(
            flat_ptrs, flat * 0 + total, mask=first, sem="relaxed", scope="cta"
        )
        start = tl.sum(tl.where(first, prev, 0), axis=0)
        pos = tl.reshape(start + tl.cumsum(flat, axis=0) - flat, ti.shape)
    else:
        pos = tl.atomic_add(ptrs, ones, mask=take, sem="relaxed", scope="cta")
    return pos


@triton.jit
def _process_bins_slotscan(
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
    # The only change from the generic function: slots for the definitely-in
    # elements are allocated by _alloc_slots, not one atomic per element.
    out_pos_lt = _alloc_slots(found_topk_values_ptrs, ones, take_lt)
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
            final_pos = tl.atomic_add(
                final_cnt_ptrs,
                ones,
                mask=take_eq_final,
                sem="relaxed",
                scope="cta",
            )
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
        out_pos_eq = tl.atomic_add(
            s_histogram_ptr + bin_idx,
            ones,
            mask=take_eq,
            sem="relaxed",
            scope="cta",
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


def _slotscan_enabled():
    raw = os.environ.get("FLAGGEMS_HYGON_TOPK_SLOTSCAN", "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


# Rebind before anything compiles: the generic kernels resolve _process_bins
# from their module globals at compile time.
if _slotscan_enabled():
    _generic._process_bins = _process_bins_slotscan


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """The generic operator, with the slot allocation above when enabled."""
    return _generic.top_k_per_row_prefill(
        logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
    )
