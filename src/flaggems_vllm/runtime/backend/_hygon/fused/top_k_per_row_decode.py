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

"""top_k_per_row_decode on Hygon BW1000: pick the threshold from a sample, so
one pass over the logits replaces the radix algorithm's two.

WHY. The generic non-TLE path launches one program per row and runs the full
radix algorithm: a histogram pass over the row, a threshold scan, then a second
pass writing the elements at or above the threshold. Below one row per SM the
card is mostly idle, and every Triton launch costs ~110 us of host dispatch on
this box (~12 us for a cached CompiledKernel launched directly), so the first
lever was a row split with direct launches -- 0.82 of vLLM, geomean over the
benchmark's eleven shapes.

Stage one then WAS the operator: 91-94% of device time, with geometry, split
factor and the helper kernels all exhausted. But it does more work than the
answer needs. The merge only needs a SUPERSET of the row's top-k, and a
threshold that admits a few times k is enough to produce one:

    prepare  histogram ~8 tiles of the row, then scan it for the bin that
             holds rank k scaled to the sample, times a safety factor
    select   ONE pass over the row, appending every element at or above that
             bin to a per-row candidate buffer (value and index)
    tail     the fallback decision (see below) and then the exact top-k of the
             candidates -- four 8-bit radix rounds over the full 32-bit
             ordered key, writing cand_idx[pos] straight out

That is ~1.02 passes against 2, and it wins at every shape the benchmark runs
(tools/hygon_decode_sampled_threshold.py, vocab 262144, top_k 512):

    rows      1     4     8    16    24    32    40    48    56   496   512
    split  .886  .732  .631  .555  .537  .621  .863  .762  .606 1.947 1.984
    this  1.020 1.072  .780  .915 1.059  .997 1.466 1.645 1.477 2.705 2.766

geomean 0.823 -> 1.324. The split path is gone; this replaces it outright.

THE ESTIMATE IS NOT A BOUND, and the operator does not sync, so the fallback
decision is made on the device, at the top of `_tail`, which reads each row's
candidate count:
a row that admitted between k and CAP candidates already holds a superset of
its top-k and only needs its merge length written, which is one scalar load and
an early return. A row outside that range -- too few, or more than the buffer
holds -- is redone exactly inside the kernel: full histogram, threshold at rank
k, then the strictly-better bins appended BEFORE the threshold bin, so that a
buffer which still overflows can only drop elements sharing an 11-bit key with
the k-th. Checked against torch.topk on random normals (which never leave the
range) and on rounded logits (which always do), for every shape above and for
top_k 64/256/1024, vocab down to 8192, and seq_len below vocab.

DIRECT LAUNCH SAFETY. Triton specialises a compiled kernel on pointer
alignment (data_ptr % 16) and on integer values (== 1, % 16). A cached kernel
must never be launched with arguments it was not specialised for, so the plan
key holds every integer argument and the alignment of each caller tensor.
Internal buffers are fresh allocations. If a Triton version returns no
CompiledKernel from `run`, the plan falls back to ordinary JIT launches.

next_n != 1, non-unit stride1, a strided row layout, a dtype other than float32
and shapes too small or too large for the candidate buffer go to the generic
operator. FLAGGEMS_HYGON_TOPK_DECODE_SAMPLED=0 disables the override;
FLAGGEMS_HYGON_TOPK_DECODE_SPLIT=n forces the select pass's programs per row,
for sweeping it.
"""

import functools
import os
import threading
from importlib import import_module

import torch
import triton
import triton.language as tl

_generic = import_module("flaggems_vllm.ops.top_k_per_row_decode")

# The operator's own STEP-0 key: fp16 bits mapped so that ascending uint16
# means descending float, then the top 11 bits. Taken from the generic module
# rather than copied, so the two cannot drift apart.
_key = _generic._convert_to_trt_uint16_hi11
_key32 = _generic._convert_to_uint32
NUM_BINS = _generic.NUM_BINS  # 2048 == 1 << 11

BLOCK = 512
WARPS = 8
SAMPLE_TILES = 8  # tiles the sample reads, whatever the row length
SAFETY = 4  # admit about SAFETY * top_k elements
CAP_FACTOR = 16  # candidate buffer, as a multiple of top_k
RADIX = 256  # bins per round of the tail's exact radix
MAX_CAND = 1 << 24  # refuse shapes whose buffers would be absurd
MIN_VOCAB = 2048
MAX_TOP_K = 2048

# Programs per row for the select pass. One constant, not a table: the old
# table (16 below five rows, 8 below twenty-five, else 4) was swept on the
# two-pass split pipeline, where each program ran the whole radix algorithm
# over its chunk. This pipeline's select only compares and appends, and a
# re-sweep (tools/hygon_decode_split_resweep.py, ratio vs vLLM) says every row
# count wants more programs than that table gave:
#
#   rows        1     4     8    16    24    32    40    48    56
#   split  1  .232  .235  .197  .227  .287  .403  .490  .559  .649
#   split  4  .660  .632  .578  .665  .801 1.123 1.339 1.446 1.667
#   split  8  .940  .889  .855  .966 1.143 1.480 1.633 1.410 1.664
#   split 16 1.074 1.128 1.065 1.164  .968 1.314 1.562 1.528 1.744
#   split 32 1.121 1.119 1.159 1.097 1.134 1.388 1.545 1.568 1.650
#
# Across 8, 16 and 32 the surface is flat to about 10% and not monotonic (24
# rows dips at 16 and recovers at 32), so picking a best per row band fits
# noise: geomean over these nine shapes is 1.295 for "32 up to 24 rows then
# 8", 1.290 for "32 then 16", and 1.292 for a flat 32. Take the flat one.
# At or beyond one row per SM the rows alone fill the card.
_SPLIT = 32
MIN_CHUNK = 8192  # smallest chunk worth its own program


@triton.jit
def _scan_threshold(base, target, NB: tl.constexpr, BLOCK: tl.constexpr):
    """Lowest bin whose prefix count reaches `target`. Bin 0 holds the largest
    values, so 'at or above the threshold' means bin <= thr."""
    lane = tl.arange(0, BLOCK)
    carry = tl.zeros([], tl.int32)
    thr = tl.full([], NB - 1, tl.int32)
    found = tl.full([], False, tl.int1)
    for t in tl.static_range(NB // BLOCK):
        bins = t * BLOCK + lane
        c = tl.load(base + bins)
        pre = carry + tl.cumsum(c, axis=0)
        hit = (pre >= target) & (not found)
        cand = tl.min(tl.where(hit, bins, NB - 1), axis=0)
        if (not found) & (tl.max(hit.to(tl.int32), axis=0) > 0):
            thr = cand
            found = tl.full([], True, tl.int1)
        carry += tl.sum(c, axis=0)
    return thr


@triton.jit
def _hist_total(base, NB: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    total = tl.zeros([], tl.int32)
    for t in tl.static_range(NB // BLOCK):
        total += tl.sum(tl.load(base + t * BLOCK + lane), axis=0)
    return total


@triton.jit
def _hist_pass(
    logits_ptr, base, row, stride0, n, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    """Histogram every STRIDE-th tile of the row into `base` (STRIDE 1: all of
    it). Whole tiles, not strided elements: a strided element sample touches
    one cache line per value and so costs a full pass for a fraction of the
    data."""
    lane = tl.arange(0, BLOCK)
    for t in tl.range(0, tl.cdiv(n, BLOCK * STRIDE)):
        i = t * BLOCK * STRIDE + lane
        m = i < n
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        tl.atomic_add(
            base + _key(x),
            tl.full([BLOCK], 1, tl.int32),
            mask=m,
            sem="relaxed",
            scope="cta",
        )


@triton.jit
def _prepare(
    logits_ptr,
    seq_lens_ptr,
    hist_ptr,
    thr_ptr,
    cnt_ptr,
    stride0,
    TOPK: tl.constexpr,
    SAFETY: tl.constexpr,
    NB: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Zero, sample and threshold, in one program per row.

    One program, so the row's histogram is this program's alone and a
    tl.debug_barrier() is all the ordering the three phases need -- and so the
    atomics stay inside one CTA. Splitting the sample across programs would pay
    the ~19 us per-program floor again for a pass that reads 1/STRIDE.

    The admit rank is derived from the sample actually taken, not from STRIDE,
    so that a row far shorter than the vocabulary still gets a usable estimate.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = hist_ptr + row * NB
    for t in tl.static_range(NB // BLOCK):
        tl.store(base + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))
    tl.store(cnt_ptr + row, 0)
    tl.debug_barrier()
    n = tl.load(seq_lens_ptr + row)
    _hist_pass(logits_ptr, base, row, stride0, n, STRIDE, BLOCK)
    tl.debug_barrier()
    total = _hist_total(base, NB, BLOCK)
    target = tl.maximum(tl.cdiv(TOPK * total, tl.maximum(n, 1)), 1) * SAFETY
    tl.store(thr_ptr + row, _scan_threshold(base, target, NB, BLOCK))


@triton.jit
def _select(
    logits_ptr,
    seq_lens_ptr,
    thr_ptr,
    cnt_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    stride0,
    CHUNK: tl.constexpr,
    SPLIT: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The one pass: append every element at or above the row's threshold bin.

    SPLIT programs share a row, so the counter they append through is written
    from several CTAs and the atomic has to be scoped to the whole device.
    """
    pid = tl.program_id(0)
    row = pid // SPLIT
    chunk = pid % SPLIT
    lane = tl.arange(0, BLOCK)
    thr = tl.load(thr_ptr + row)
    n = tl.load(seq_lens_ptr + row)
    start = chunk * CHUNK
    end = tl.minimum(start + CHUNK, n)
    cnt_ptrs = cnt_ptr + row + tl.zeros([BLOCK], tl.int32)
    for t in tl.range(0, tl.cdiv(CHUNK, BLOCK)):
        i = start + t * BLOCK + lane
        m = i < end
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        take = m & (_key(x) <= thr)
        pos = tl.atomic_add(
            cnt_ptrs,
            tl.full([BLOCK], 1, tl.int32),
            mask=take,
            sem="relaxed",
            scope="gpu",
        )
        keep = take & (pos < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos, i.to(tl.int32), mask=keep)
        tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)


@triton.jit
def _select_exact(
    logits_ptr,
    row,
    stride0,
    n,
    thr,
    cnt_ptrs,
    cand_idx_ptr,
    cand_val_ptr,
    EQUAL: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program's pass over a row, appending the elements whose key is below
    `thr` (EQUAL False) or exactly `thr` (EQUAL True)."""
    lane = tl.arange(0, BLOCK)
    for t in tl.range(0, tl.cdiv(n, BLOCK)):
        i = t * BLOCK + lane
        m = i < n
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        k = _key(x)
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
        keep = take & (pos < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos, i.to(tl.int32), mask=keep)
        tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)


@triton.jit
def _tail(
    logits_ptr,
    seq_lens_ptr,
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
    """The fallback decision and the exact top-k of the candidates, in one
    program per row -- so one launch where there were three.

    First the decision a host sync would otherwise make: a row that admitted
    between TOPK and CAP candidates already holds a superset of its top-k; one
    outside that range is redone exactly here, appending the strictly-better
    bins BEFORE the threshold bin so that a buffer which still overflows can
    only ever drop elements sharing an 11-bit key with the k-th.

    Then the answer, by four 8-bit radix rounds over the FULL 32-bit ordered
    key (ascending uint32 is descending float). That is what the generic
    kernel's own final select does; running it over the candidates directly
    costs no 11-bit pre-pass and no threshold-bin special case, and is exact by
    construction -- there is no fp16 granularity left to reason about. The
    output is cand_idx[pos], so the remap disappears as well.

    Measured against the generic merge plus a remap at the candidate counts
    this pipeline produces (tools/hygon_merge_kernel.py): at or above parity
    from 4 to 512 rows, and 1.2-1.8x where the candidates are few. A variant
    that narrows with an 11-bit histogram first is flatter in the candidate
    count and better above 64 rows, but loses here at the low row counts this
    is meant to fix.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    bins = tl.arange(0, RADIX)
    ones = tl.full([BLOCK], 1, tl.int32)
    n = tl.load(seq_lens_ptr + row)
    c = tl.load(cnt_ptr + row)
    cnt_ptrs = cnt_ptr + row + tl.zeros([BLOCK], tl.int32)
    if (c < tl.minimum(TOPK, n)) | (c > CAP):
        base = hist_ptr + row * NB
        for t in tl.static_range(NB // BLOCK):
            tl.store(base + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))
        tl.debug_barrier()
        _hist_pass(logits_ptr, base, row, stride0, n, 1, BLOCK)
        tl.debug_barrier()
        thr = _scan_threshold(base, TOPK, NB, BLOCK)
        tl.store(cnt_ptr + row, 0)
        tl.debug_barrier()
        _select_exact(
            logits_ptr,
            row,
            stride0,
            n,
            thr,
            cnt_ptrs,
            cand_idx_ptr,
            cand_val_ptr,
            False,
            CAP,
            BLOCK,
        )
        tl.debug_barrier()
        _select_exact(
            logits_ptr,
            row,
            stride0,
            n,
            thr,
            cnt_ptrs,
            cand_idx_ptr,
            cand_val_ptr,
            True,
            CAP,
            BLOCK,
        )
        tl.debug_barrier()

    m = tl.minimum(tl.load(cnt_ptr + row), CAP)
    vbase = cand_val_ptr + row * CAP
    ibase = cand_idx_ptr + row * CAP
    obase = out_ptr + row * TOPK
    cbase = counts_ptr + row * RADIX
    tiles = tl.cdiv(m, BLOCK)

    if m <= TOPK:
        # fewer candidates than asked for: all of them go out, -1 pads
        for t in tl.static_range((TOPK + BLOCK - 1) // BLOCK):
            j = t * BLOCK + lane
            idx = tl.load(ibase + j, mask=j < m, other=-1)
            tl.store(obase + j, tl.where(j < m, idx, -1), mask=j < TOPK)
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
                valid = pos < m
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
            counts = tl.load(cbase + bins)
            prefix = tl.cumsum(counts, axis=0) - counts
            hit = (prefix < k_to_find) & (prefix + counts >= k_to_find)
            rb = tl.min(tl.where(hit, bins, RADIX), axis=0).to(tl.int32)
            rb = tl.where(rb == RADIX, RADIX - 1, rb)
            counts_lt = tl.max(tl.where(bins == rb, prefix, 0), axis=0).to(tl.int32)
            desired = desired | (rb.to(tl.uint32) << digit_pos)
            desired_mask = desired_mask | (
                tl.full((), RADIX - 1, tl.uint32) << digit_pos
            )
            k_to_find = k_to_find - counts_lt

    thr_key = desired
    tl.store(slot_ptr + row, 0)
    tl.debug_barrier()
    slots = slot_ptr + row + tl.zeros([BLOCK], tl.int32)
    # everything strictly better than the k-th, then its equals; the rounds
    # above leave the first group short of TOPK and the two together at least
    # TOPK, so this lands exactly on TOPK
    for equal in tl.static_range(2):
        for t in tl.range(0, tiles):
            pos = t * BLOCK + lane
            valid = pos < m
            key = _key32(tl.load(vbase + pos, mask=valid, other=0.0))
            if equal == 0:
                take = valid & (key < thr_key)
            else:
                take = valid & (key == thr_key)
            q = tl.atomic_add(slots, ones, mask=take, sem="relaxed", scope="cta")
            idx = tl.load(ibase + pos, mask=take, other=-1)
            tl.store(obase + q, idx, mask=take & (q < TOPK))
        tl.debug_barrier()


@functools.lru_cache(maxsize=1)
def _sm_count():
    try:
        props = torch.cuda.get_device_properties(0)
        return int(getattr(props, "multi_processor_count", 0)) or 80
    except Exception:  # noqa: BLE001 - detection must never break dispatch
        return 80


def _enabled():
    return os.environ.get("FLAGGEMS_HYGON_TOPK_DECODE_SAMPLED", "1") != "0"


def _forced_split():
    raw = os.environ.get("FLAGGEMS_HYGON_TOPK_DECODE_SPLIT")
    if raw is None:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def _split_factor(num_rows, vocab_size, top_k):
    """Programs per row for the select pass."""
    forced = _forced_split()
    if forced is None and num_rows >= _sm_count():
        return 1
    split = forced or _SPLIT
    while split > 1 and (
        vocab_size % split or vocab_size // split < max(MIN_CHUNK, top_k)
    ):
        split //= 2
    return split


def _sample_stride(vocab_size):
    return max(1, vocab_size // (BLOCK * SAMPLE_TILES))


def _cap(top_k):
    return max(BLOCK, triton.next_power_of_2(top_k * CAP_FACTOR))


class _Launch:
    """One kernel of the pipeline: JIT on first use, direct afterwards."""

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


class _Plan:
    """Buffers and launchers for one (shape, specialisation). Three launches:
    sample the row for a threshold, select against it in one pass, then the
    fallback decision and the exact answer together."""

    def __init__(self, dev, dtype, num_rows, vocab, top_k, split):
        cap = _cap(top_k)
        chunk = vocab // split
        self.hist = torch.empty((num_rows, NUM_BINS), dtype=torch.int32, device=dev)
        self.thr = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.cnt = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.cand_idx = torch.empty((num_rows, cap), dtype=torch.int32, device=dev)
        self.cand_val = torch.empty((num_rows, cap), dtype=dtype, device=dev)
        self.counts = torch.empty((num_rows, RADIX), dtype=torch.int32, device=dev)
        self.slot = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.prepare = _Launch(
            _prepare,
            (num_rows,),
            {
                "TOPK": top_k,
                "SAFETY": SAFETY,
                "NB": NUM_BINS,
                "STRIDE": _sample_stride(vocab),
                "BLOCK": BLOCK,
            },
            WARPS,
        )
        self.select = _Launch(
            _select,
            (num_rows * split,),
            {"CHUNK": chunk, "SPLIT": split, "CAP": cap, "BLOCK": BLOCK},
            WARPS,
        )
        self.tail = _Launch(
            _tail,
            (num_rows,),
            {
                "TOPK": top_k,
                "NB": NUM_BINS,
                "CAP": cap,
                "RADIX": RADIX,
                "BLOCK": BLOCK,
            },
            WARPS,
        )

    def run(self, logits, seq_lens, indices, stride0):
        self.prepare(logits, seq_lens, self.hist, self.thr, self.cnt, stride0)
        self.select(
            logits,
            seq_lens,
            self.thr,
            self.cnt,
            self.cand_idx,
            self.cand_val,
            stride0,
        )
        self.tail(
            logits,
            seq_lens,
            self.hist,
            self.cnt,
            self.cand_idx,
            self.cand_val,
            indices,
            self.counts,
            self.slot,
            stride0,
        )


_PLANS = {}
_PLANS_MAX = 32
_LOCK = threading.Lock()


def _aligned(t):
    return t.data_ptr() % 16 == 0


def top_k_per_row_decode(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    """One pass over the logits, with the threshold picked from a sample."""
    vocab_size = logits.shape[1]
    if (
        not _enabled()
        or next_n != 1
        or stride1 != 1
        or stride0 != vocab_size
        or logits.dtype != torch.float32
        or seq_lens.dtype != torch.int32
        or vocab_size < MIN_VOCAB
        or top_k > MAX_TOP_K
        or top_k > vocab_size
        or num_rows * _cap(top_k) > MAX_CAND
    ):
        return _generic.top_k_per_row_decode(
            logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
        )

    split = _split_factor(num_rows, vocab_size, top_k)
    key = (
        logits.device,
        num_rows,
        vocab_size,
        top_k,
        split,
        _aligned(logits),
        _aligned(seq_lens),
        _aligned(indices),
    )
    with _LOCK:
        plan = _PLANS.get(key)
        if plan is None:
            if len(_PLANS) >= _PLANS_MAX:
                _PLANS.pop(next(iter(_PLANS)))
            plan = _PLANS[key] = _Plan(
                logits.device, logits.dtype, num_rows, vocab_size, top_k, split
            )
        plan.run(logits, seq_lens, indices, stride0)
    return indices
