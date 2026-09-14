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

"""top_k_per_row_prefill on Hygon BW1000: on DENSE rows, allocate output slots
by prefix sum instead of one atomic per selected element.

WHY. prefill loses on all seven benchmark shapes against vLLM's C++ kernel here
(geomean 0.355), worst on the small-vocabulary ones. Per program it fits
base + ~19 ns x top_k + ~1.3 ns x vocab, and the k term is
`tl.atomic_add(found_topk_values_ptrs, ones, mask=take_lt)` in _process_bins:
one atomic per selected element, all to one address, ~12 ns each here (the
ablation removed 7.55 of 18.07 us per program with it). Shared memory does not
help on this card: smem scatter atomics measured 2.5-3.5x slower than global.

WHAT. A prefix sum over the tile's take-mask gives each selected element a
distinct offset; one atomic of the tile's COUNT gives the base. Its cost scales
with the tile's SIZE, the atomics' with how many it selects -- so the deciding
quantity is DENSITY, the fraction of elements selected, which for a full-range
row is top_k / vocab. Measured crossover ~9.4% (48 of 512); dispatched here at
vocab <= 10 * top_k.

WHY TWO MODULE COPIES, NOT A BRANCH. The first version branched inside the
kernel on each tile's own count. The A/B on this box (kernel mode, both passes
agreeing to 0.03):

    shape            dense?  branch-in-kernel
    (4100, 1025)       yes        1.97x
    (12961, 4100)      yes        1.57x
    (16380, 5115)      yes        1.41x
    (16383, 4095)      yes        1.40x
    (4, 8193)          no         0.77x   threshold was per-count, not density
    (4, 16385)         no         0.71x   same
    (64, 129280)       no         0.85x   took the atomic branch and STILL lost

The last row is the branch's own cost, ~0.18 us per call whether taken or not,
on the production shape. And the choice cannot move to the host by rebinding:
Triton binds module globals at compile time and caches, so one module can only
ever hold one _process_bins. So the generic module is loaded a SECOND time
under another name, the copy gets the prefix-sum _process_bins, and the host
picks a module per call. Sparse rows run the untouched generic kernel.

Density from vocab is conservative for partial-range rows (a shorter row is
denser than vocab suggests), so a miss falls back to generic, never to
something slower. FLAGGEMS_HYGON_TOPK_SLOTSCAN=0 always uses generic.
"""

import importlib.util
import os
import sys
from importlib import import_module

import triton
import triton.language as tl

_GENERIC_NAME = "flaggems_vllm.ops.top_k_per_row_prefill"
_DENSE_NAME = "flaggems_vllm.ops._top_k_per_row_prefill_hygon_dense"

_generic = import_module(_GENERIC_NAME)

# Dense iff vocab_size <= DENSE_VOCAB_PER_TOPK * top_k, i.e. density >= 10%.
DENSE_VOCAB_PER_TOPK = 10


def _load_copy(name):
    """The generic module, executed again as a separate module. @triton.jit
    needs its functions' source on disk, which the generic file provides."""
    spec = importlib.util.spec_from_file_location(name, _generic.__file__)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_dense = _load_copy(_DENSE_NAME)
_extract_bin_idx = _dense._extract_bin_idx


@triton.jit
def _alloc_slots(ptrs, take):
    """Same contract as tl.atomic_add(ptrs, 1, mask=take) with `ptrs` all at
    one counter: every taken lane gets a distinct slot. 1-D or [BLOCK, VEC]."""
    ti = take.to(tl.int32)
    flat = tl.reshape(ti, (ti.numel,))
    total = tl.sum(flat, axis=0)
    # One atomic, on lane 0 only, adds the tile's count and returns the
    # counter's previous value: the base of this tile's run of slots.
    first = tl.arange(0, ti.numel) == 0
    prev = tl.atomic_add(
        tl.reshape(ptrs, (ti.numel,)),
        flat * 0 + total,
        mask=first,
        sem="relaxed",
        scope="cta",
    )
    start = tl.sum(tl.where(first, prev, 0), axis=0)
    return tl.reshape(start + tl.cumsum(flat, axis=0) - flat, ti.shape)


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
    # elements come from a prefix sum, not from one atomic per element.
    out_pos_lt = _alloc_slots(found_topk_values_ptrs, take_lt)
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


# Rebind in the COPY only, before anything compiles.
_dense._process_bins = _process_bins_slotscan


def _slotscan_enabled():
    raw = os.environ.get("FLAGGEMS_HYGON_TOPK_SLOTSCAN", "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


_ENABLED = _slotscan_enabled()


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Dense rows through the prefix-sum copy, everything else through generic."""
    if _ENABLED and logits.shape[1] <= DENSE_VOCAB_PER_TOPK * top_k:
        op = _dense.top_k_per_row_prefill
    else:
        op = _generic.top_k_per_row_prefill
    return op(logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k)
