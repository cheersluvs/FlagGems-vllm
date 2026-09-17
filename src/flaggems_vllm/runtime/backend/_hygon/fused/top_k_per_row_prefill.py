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

import functools
import importlib.util
import os
import sys
import threading
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


# ---------------------------------------------------------------------------
# Launch geometry by occupancy.
#
# Which BLOCK_SIZE x num_warps is fastest depends on how many rows share this
# card's SMs, not on elements per lane. Full operator, every point checked
# against torch.topk, ratio vs vLLM (tools/hygon_prefill_launch_sweep.py and
# tools/hygon_prefill_rows_sweep.py):
#
#   rows/SM         row 4096, k 512                row 129280, k 1024
#   < 4             all configs within noise       B512 w8 (today) best
#   4 - 16          B512 w4: +3% .. +16%           B512 w4: +3% .. +10%
#   32 - 52         B256 w2: +47% .. +57%          B256 w4: +13% .. +32%
#   204 (16383 r)   B256 w2: 1.95x                 --
#
# Many rows want narrow programs: past one row per SM the grid is the
# parallelism, and wider programs only crowd each other out. Long rows keep
# more threads per program than short ones at high occupancy. Where the switch
# between w2 and w4 falls for row lengths between 8192 and 129280 is UNMEASURED;
# those take w4, the choice measured for long rows.
#
# num_warps=1 is never used: it returned WRONG answers in the sweep.
#
# NUM_THREADS_PER_BLOCK and _num_warps are host-side globals read at each
# launch -- unlike the jit globals above, they are not baked into a compiled
# kernel -- so they are set per call. A lock keeps set-and-launch atomic.
# FLAGGEMS_HYGON_TOPK_GEOMETRY=0 leaves them at generic's values.

SHORT_ROW_MAX = 8192


@functools.lru_cache(maxsize=1)
def _sm_count():
    try:
        import torch

        props = torch.cuda.get_device_properties(0)
        return int(getattr(props, "multi_processor_count", 0)) or 80
    except Exception:  # noqa: BLE001 - detection must never break dispatch
        return 80


def _geometry(num_rows, row_len):
    """(BLOCK_SIZE, num_warps) for this call, or None for generic's own."""
    sms = _sm_count()
    if num_rows < 4 * sms:
        return None
    if num_rows < 32 * sms:
        return 512, 4
    return (256, 2) if row_len <= SHORT_ROW_MAX else (256, 4)


def _geometry_enabled():
    raw = os.environ.get("FLAGGEMS_HYGON_TOPK_GEOMETRY", "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


_GEOMETRY = _geometry_enabled()
_LAUNCH_LOCK = threading.Lock()
_GENERIC_DEFAULTS = {
    id(m): (m.NUM_THREADS_PER_BLOCK, m._num_warps) for m in (_generic, _dense)
}


# --------------------------------------------------------------------------
# A sampled threshold for very sparse rows.
#
# The histogram pass is 82% of this operator at (64,129280) -- measured by
# splitting T(top_k=1) at row_end = vocab against vocab/2, which separates the
# per-element pass (189.0 us) from fixed cost (15.0). Its collection pass is
# only ~26 us for the same bytes: the 7x gap is the per-element atomic.
#
# So estimate the threshold from 1/SSTRIDE of the row, spend the saving on a
# deliberately LOOSE threshold, and let one pass collect about TARGET_MULT *
# top_k candidates instead of exactly top_k. The design is the MTT override's
# (_mthreads/fused/top_k_per_row_prefill.py); the implementation is not, because
# that one keeps its histogram and candidates in shared memory through TLE and
# on this card TLE prefill measures 0.21x and a shared-memory histogram is
# 2.5-3.5x slower than a global one.
#
# WHEN IT PAYS. Collecting m * top_k candidates means ranking them down to
# top_k afterwards, which the generic operator does not pay -- its final stage
# sees the threshold bin alone, 27-36 elements. That ranking costs about
# 0.06 us per row at 1024 candidates (tools/hygon_merge_kernel.py), and it
# scales with rows * candidates while the saving scales with rows * vocab. So
# the figure of merit is vocab / (m * top_k):
#
#     shape            ratio   saves   ranking   net
#     (64,129280)        63     165       40     1.73x
#     (4,16385)          16      18       14     0.98
#     (4,8193)            8      10       14     <1
#     (16380,5115)        5    1030     1000     ~1.0
#     (16383,4095)        4    1090     1000     ~1.0
#
# Only the first qualifies. MTT's own MIN_SPAN = 16384 says the same thing in
# absolute vocabulary rather than as a ratio; the ratio is what actually
# decides it, and it excludes every many-row shape here.
SAMPLED_MIN_VOCAB_PER_TOPK = int(
    os.environ.get("FLAGGEMS_HYGON_PREFILL_SAMPLED_RATIO", "64")
)
SSTRIDE = int(os.environ.get("FLAGGEMS_HYGON_PREFILL_SSTRIDE", "8"))
# Collect about this many times top_k. 2 put the target at the geometric
# midpoint of the acceptance window [top_k, CAP] = [1024, 4096], which is
# where MTT's notes say it belongs -- but the two stages that scale with the
# collected count are 194 of the pipeline's 228 us, so the midpoint is not
# free. At 1.5 the ranking drops about a quarter, and the target sits 24%
# above the window's lower edge against a measured per-row spread of +-18%, so
# an undershoot into the retry stays unlikely. Watch "rows outside the window"
# in tools/hygon_prefill_sampled_stages.py: aiming below the midpoint is the
# direction MTT's notes warn about.
TARGET_MULT = float(os.environ.get("FLAGGEMS_HYGON_PREFILL_TARGET_MULT", "1.5"))
CAP_MULT = 4  # candidate buffer; the acceptance window is [top_k, CAP]
SBLOCK = 512
SWARPS = 8
SRADIX = 256

_key11 = _generic._convert_to_trt_uint16_hi11
_key32 = _generic._convert_to_uint32


@triton.jit
def _s_scan(base, target, NB: tl.constexpr, BLOCK: tl.constexpr):
    """Lowest bin whose inclusive prefix reaches `target`, and that bin's
    exclusive prefix. Bin 0 holds the largest values."""
    lane = tl.arange(0, BLOCK)
    carry = tl.zeros([], tl.int32)
    tb = tl.full([], NB - 1, tl.int32)
    lt = tl.zeros([], tl.int32)
    found = tl.full([], False, tl.int1)
    for t in tl.static_range(NB // BLOCK):
        bins = t * BLOCK + lane
        c = tl.load(base + bins)
        pre = carry + tl.cumsum(c, axis=0) - c
        hit = (pre < target) & (pre + c >= target) & (not found)
        cand = tl.min(tl.where(hit, bins, NB - 1), axis=0)
        candlt = tl.max(tl.where(hit, pre, 0), axis=0)
        if (not found) & (tl.max(hit.to(tl.int32), axis=0) > 0):
            tb = cand
            lt = candlt
            found = tl.full([], True, tl.int1)
        carry += tl.sum(c, axis=0)
    return tb, lt


@triton.jit
def _s_hist(
    logits_ptr, base, row, stride0, s, e, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    """Histogram every STRIDE-th TILE of [s, e). Tiles rather than strided
    elements: at SSTRIDE 8 a strided element sample is 32 bytes apart and so
    touches every other cache line, reading half the bytes for an eighth of
    the values. Tiles read exactly 1/STRIDE. The cost is that the sample is
    spatially clustered, which is only unbiased if the row has no spatial
    structure -- true of the benchmark's iid inputs, and the reason the
    fallback below is not optional."""
    lane = tl.arange(0, BLOCK)
    for t in tl.range(0, tl.cdiv(e - s, BLOCK * STRIDE)):
        i = s + t * BLOCK * STRIDE + lane
        m = i < e
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        tl.atomic_add(
            base + _key11(x),
            tl.full([BLOCK], 1, tl.int32),
            mask=m,
            sem="relaxed",
            scope="cta",
        )


@triton.jit
def _s_prepare(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    thr_ptr,
    cnt_ptr,
    stride0,
    TARGET: tl.constexpr,
    NB: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Zero, sample and threshold, one program per row -- so the histogram is
    this program's alone and a barrier is all the ordering needed."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = hist_ptr + row * NB
    for t in tl.static_range(NB // BLOCK):
        tl.store(base + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))
    tl.store(cnt_ptr + row, 0)
    tl.debug_barrier()
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    _s_hist(logits_ptr, base, row, stride0, s, e, STRIDE, BLOCK)
    tl.debug_barrier()
    # Take the WHOLE boundary bin (+1, exclusive): coarser estimates should
    # over-collect, since falling short of top_k forces the exact retry while
    # overshooting only costs a slightly larger ranking.
    tb, _ = _s_scan(base, tl.cdiv(TARGET, STRIDE), NB, BLOCK)
    tl.store(thr_ptr + row, tb + 1)


@triton.jit
def _s_collect(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    thr_ptr,
    cnt_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    stride0,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
):
    """One pass: append every element strictly better than the threshold bin.
    Indices are stored relative to row_start, which is what the operator
    returns.

    The bulk loop is UNMASKED and the remainder is handled separately, which
    is how the generic operator writes its own passes. Masking every iteration
    instead measured 131.3 us here against a modelled 55 -- 252 GB/s where the
    generic's collection pass reaches ~1270, essentially read speed -- and a
    mask on every load is the one structural difference between them.
    """
    row = tl.program_id(0)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    span = e - s
    thr = tl.load(thr_ptr + row)
    base = logits_ptr + row * stride0 + s
    lane = tl.arange(0, BLOCK)
    off = lane[:, None] * VEC + tl.arange(0, VEC)[None, :]
    ones2 = tl.full([BLOCK, VEC], 1, tl.int32)
    ones1 = tl.full([BLOCK], 1, tl.int32)
    cnt2 = cnt_ptr + row + tl.zeros([BLOCK, VEC], tl.int32)
    cnt1 = cnt_ptr + row + tl.zeros([BLOCK], tl.int32)

    n_vec = span // (BLOCK * VEC)
    for t in tl.range(0, n_vec):
        i = t * BLOCK * VEC + off
        x = tl.load(base + i)
        # Cast explicitly: the key is uint32 and thr int32, and leaving that
        # promotion implicit selects every element (the MTT override records
        # the same bug).
        take = _key11(x).to(tl.int32) < thr
        pos = tl.atomic_add(cnt2, ones2, mask=take, sem="relaxed", scope="cta")
        keep = take & (pos >= 0) & (pos < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos, i.to(tl.int32), mask=keep)
        tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)

    tail = n_vec * BLOCK * VEC
    for t in tl.range(0, tl.cdiv(span - tail, BLOCK)):
        i = tail + t * BLOCK + lane
        m = i < span
        x = tl.load(base + i, mask=m, other=0.0)
        take = m & (_key11(x).to(tl.int32) < thr)
        pos = tl.atomic_add(cnt1, ones1, mask=take, sem="relaxed", scope="cta")
        keep = take & (pos >= 0) & (pos < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos, i.to(tl.int32), mask=keep)
        tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)


@triton.jit
def _s_exact_pass(
    logits_ptr,
    row,
    stride0,
    s,
    e,
    thr,
    cnt_ptrs,
    cand_idx_ptr,
    cand_val_ptr,
    EQUAL: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    for t in tl.range(0, tl.cdiv(e - s, BLOCK)):
        i = t * BLOCK + lane
        m = i < e - s
        x = tl.load(logits_ptr + row * stride0 + s + i, mask=m, other=0.0)
        k = _key11(x).to(tl.int32)
        if EQUAL:
            take = m & (k == thr)
        else:
            take = m & (k < thr)
        pos = tl.atomic_add(
            cnt_ptrs,
            tl.full([BLOCK], 1, tl.int32),
            mask=take,
            sem="relaxed",
            scope="cta",
        )
        keep = take & (pos >= 0) & (pos < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos, i.to(tl.int32), mask=keep)
        tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)


@triton.jit
def _s_finish(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    cnt_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    out_ptr,
    counts_ptr,
    slot_ptr,
    stride0,
    TOPK: tl.constexpr,
    NB: tl.constexpr,
    CAP: tl.constexpr,
    RADIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The retry decision and the exact answer, one program per row.

    A sample can under- or overshoot, so a row whose collected count falls
    outside [TOPK, CAP] is redone here from a full histogram at rank TOPK --
    what the generic operator does anyway, so a bad estimate is slow, never
    wrong. The redo appends the strictly-better bins BEFORE the boundary bin,
    so a buffer that still overflows can only drop elements sharing an 11-bit
    key with the k-th.

    Then the exact top-k of the candidates: four 8-bit radix rounds over the
    FULL 32-bit ordered key, which is what the generic operator's own final
    select does. Exact by construction -- no fp16 granularity to reason about.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    bins = tl.arange(0, RADIX)
    ones = tl.full([BLOCK], 1, tl.int32)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    span = e - s
    c = tl.load(cnt_ptr + row)
    cnt_ptrs = cnt_ptr + row + tl.zeros([BLOCK], tl.int32)
    if (c < tl.minimum(TOPK, span)) | (c > CAP):
        base = hist_ptr + row * NB
        for t in tl.static_range(NB // BLOCK):
            tl.store(base + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))
        tl.debug_barrier()
        _s_hist(logits_ptr, base, row, stride0, s, e, 1, BLOCK)
        tl.debug_barrier()
        tb, _ = _s_scan(base, TOPK, NB, BLOCK)
        tl.store(cnt_ptr + row, 0)
        tl.debug_barrier()
        _s_exact_pass(
            logits_ptr,
            row,
            stride0,
            s,
            e,
            tb,
            cnt_ptrs,
            cand_idx_ptr,
            cand_val_ptr,
            False,
            CAP,
            BLOCK,
        )
        tl.debug_barrier()
        _s_exact_pass(
            logits_ptr,
            row,
            stride0,
            s,
            e,
            tb,
            cnt_ptrs,
            cand_idx_ptr,
            cand_val_ptr,
            True,
            CAP,
            BLOCK,
        )
        tl.debug_barrier()

    n = tl.minimum(tl.load(cnt_ptr + row), CAP)
    vbase = cand_val_ptr + row * CAP
    ibase = cand_idx_ptr + row * CAP
    obase = out_ptr + row * TOPK
    cbase = counts_ptr + row * RADIX
    tiles = tl.cdiv(n, BLOCK)

    if n <= TOPK:
        for t in tl.static_range((TOPK + BLOCK - 1) // BLOCK):
            j = t * BLOCK + lane
            idx = tl.load(ibase + j, mask=j < n, other=-1)
            tl.store(obase + j, tl.where(j < n, idx, -1), mask=j < TOPK)
        return

    desired = tl.zeros((), tl.uint32)
    desired_mask = tl.zeros((), tl.uint32)
    k_to_find = TOPK + 1
    for digit_pos in tl.static_range(24, -1, -8):
        if k_to_find > 1:
            tl.store(cbase + bins, tl.zeros([RADIX], tl.int32))
            tl.debug_barrier()
            for t in tl.range(0, tiles):
                pos = t * BLOCK + lane
                valid = pos < n
                key = _key32(tl.load(vbase + pos, mask=valid, other=0.0))
                digit = ((key >> digit_pos) & (RADIX - 1)).to(tl.int32)
                tl.atomic_add(
                    cbase + digit,
                    ones,
                    mask=valid & ((key & desired_mask) == desired),
                    sem="relaxed",
                    scope="cta",
                )
            tl.debug_barrier()
            cnts = tl.load(cbase + bins)
            prefix = tl.cumsum(cnts, axis=0) - cnts
            hit = (prefix < k_to_find) & (prefix + cnts >= k_to_find)
            rb = tl.min(tl.where(hit, bins, RADIX), axis=0).to(tl.int32)
            rb = tl.where(rb == RADIX, RADIX - 1, rb)
            lt = tl.max(tl.where(bins == rb, prefix, 0), axis=0).to(tl.int32)
            desired = desired | (rb.to(tl.uint32) << digit_pos)
            desired_mask = desired_mask | (
                tl.full((), RADIX - 1, tl.uint32) << digit_pos
            )
            k_to_find = k_to_find - lt

    thr_key = desired
    tl.store(slot_ptr + row, 0)
    tl.debug_barrier()
    slots = slot_ptr + row + tl.zeros([BLOCK], tl.int32)
    for equal in tl.static_range(2):
        for t in tl.range(0, tiles):
            pos = t * BLOCK + lane
            valid = pos < n
            key = _key32(tl.load(vbase + pos, mask=valid, other=0.0))
            if equal == 0:
                take = valid & (key < thr_key)
            else:
                take = valid & (key == thr_key)
            q = tl.atomic_add(slots, ones, mask=take, sem="relaxed", scope="cta")
            idx = tl.load(ibase + pos, mask=take, other=-1)
            tl.store(obase + q, idx, mask=take & (q < TOPK))
        tl.debug_barrier()


class _SLaunch:
    """One kernel: JIT on first use, direct afterwards. Same recipe as the
    decode override's; copied so the two operators stay independent."""

    __slots__ = ("jit", "grid", "grid3", "constexprs", "num_warps", "runner")

    def __init__(self, jit, grid, constexprs, num_warps):
        self.jit = jit
        self.grid = grid
        self.grid3 = tuple(grid) + (1,) * (3 - len(grid))
        self.constexprs = constexprs
        self.num_warps = num_warps
        self.runner = None

    def __call__(self, *args):
        if self.runner is not None:
            self.runner(*args, *self.constexprs.values())
            return
        ck = self.jit.run(
            *args,
            **self.constexprs,
            num_warps=self.num_warps,
            grid=self.grid,
            warmup=False,
        )
        if ck is not None:
            self.runner = ck[self.grid3]


class _SPlan:
    """Buffers and the three launches for one sampled shape."""

    def __init__(self, dev, dtype, num_rows, vocab, top_k):
        import torch

        cap = max(SBLOCK, triton.next_power_of_2(top_k * CAP_MULT))
        self.cap = cap
        nb = _generic.NUM_BINS
        self.hist = torch.empty((num_rows, nb), dtype=torch.int32, device=dev)
        self.thr = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.cnt = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.cand_idx = torch.empty((num_rows, cap), dtype=torch.int32, device=dev)
        self.cand_val = torch.empty((num_rows, cap), dtype=dtype, device=dev)
        self.counts = torch.empty((num_rows, SRADIX), dtype=torch.int32, device=dev)
        self.slot = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.prepare = _SLaunch(
            _s_prepare,
            (num_rows,),
            {
                "TARGET": int(top_k * TARGET_MULT),
                "NB": nb,
                "STRIDE": SSTRIDE,
                "BLOCK": SBLOCK,
            },
            SWARPS,
        )
        self.collect = _SLaunch(
            _s_collect,
            (num_rows,),
            {"CAP": cap, "BLOCK": SBLOCK, "VEC": 4},
            SWARPS,
        )
        self.finish = _SLaunch(
            _s_finish,
            (num_rows,),
            {"TOPK": top_k, "NB": nb, "CAP": cap, "RADIX": SRADIX, "BLOCK": SBLOCK},
            SWARPS,
        )

    def run(self, logits, starts, ends, indices, stride0):
        self.prepare(logits, starts, ends, self.hist, self.thr, self.cnt, stride0)
        self.collect(
            logits,
            starts,
            ends,
            self.thr,
            self.cnt,
            self.cand_idx,
            self.cand_val,
            stride0,
        )
        self.finish(
            logits,
            starts,
            ends,
            self.hist,
            self.cnt,
            self.cand_idx,
            self.cand_val,
            indices,
            self.counts,
            self.slot,
            stride0,
        )


_SPLANS = {}
_SPLANS_MAX = 8
_SPLAN_LOCK = threading.Lock()


def _s_aligned(t):
    return t.data_ptr() % 16 == 0


def _can_sample(logits, row_starts, row_ends, num_rows, stride0, stride1, top_k):
    import torch

    vocab = logits.shape[1]
    return (
        SAMPLED_MIN_VOCAB_PER_TOPK > 0
        and vocab >= SAMPLED_MIN_VOCAB_PER_TOPK * top_k
        and stride1 == 1
        and num_rows > 0
        and num_rows == logits.shape[0]
        and logits.dtype == torch.float32
        and row_starts.dtype == torch.int32
        and row_ends.dtype == torch.int32
        and not getattr(_generic, "HAS_TLE", False)
        and num_rows * triton.next_power_of_2(top_k * CAP_MULT) <= (1 << 24)
    )


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Very sparse rows through a sampled threshold, dense rows through the
    prefix-sum copy, everything else through generic, each launched at the
    geometry its occupancy wants."""
    if _can_sample(logits, row_starts, row_ends, num_rows, stride0, stride1, top_k):
        key = (
            logits.device,
            num_rows,
            logits.shape[1],
            top_k,
            stride0,
            _s_aligned(logits),
            _s_aligned(row_starts),
            _s_aligned(row_ends),
            _s_aligned(indices),
        )
        with _SPLAN_LOCK:
            plan = _SPLANS.get(key)
            if plan is None:
                if len(_SPLANS) >= _SPLANS_MAX:
                    _SPLANS.pop(next(iter(_SPLANS)))
                plan = _SPLANS[key] = _SPlan(
                    logits.device, logits.dtype, num_rows, logits.shape[1], top_k
                )
            plan.run(logits, row_starts, row_ends, indices, stride0)
        return indices

    if _ENABLED and logits.shape[1] <= DENSE_VOCAB_PER_TOPK * top_k:
        mod = _dense
    else:
        mod = _generic
    geo = _geometry(num_rows, logits.shape[1]) if _GEOMETRY else None
    with _LAUNCH_LOCK:
        if geo is None:
            mod.NUM_THREADS_PER_BLOCK, mod._num_warps = _GENERIC_DEFAULTS[id(mod)]
        else:
            block, warps = geo
            mod.NUM_THREADS_PER_BLOCK = block
            mod._num_warps = lambda block_size, w=warps: w
        return mod.top_k_per_row_prefill(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        )
