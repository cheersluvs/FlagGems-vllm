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

"""Sampled-threshold prefill top-K for Moore Threads.

The generic operator reads each row TWICE per refinement step: once to build a
2048-bin histogram so the top-K threshold can be found, and again to compact the
elements at or above it. Measured on S5000 at (64, 129280):

    Pass A  build histogram      75.2 us
    Pass B  re-read and compact 131.8 us
    final select                 10.2 us
    operator                    217.2 us      vLLM 141.6   -> 0.651

Pass A exists only to find a threshold. This estimates that threshold from
1/SSTRIDE of the row instead and spends the saving on a deliberately loose
threshold, so the single remaining pass collects several times top_k candidates
rather than exactly top_k.

Measured by driving the generic _process_bins at a loose threshold:

    rank target   trigger   pass    sample + pass + final   speedup
      2 x top_k    1.6%    135.7            146.8            0.965
      4 x top_k    3.2%    139.0            150.2            0.943
      8 x top_k    6.3%    141.6            152.7            0.927

It pays because compaction barely tracks the trigger rate -- four times the hits
cost 4% more -- so the atomics are per-lane issue overhead, not per-hit traffic,
and trading a looser threshold for a whole scan is close to free.

Two parameters decide whether that theoretical win survives contact with a real
sample, and the first version got both wrong (0.383 against the generic 0.652):
the sample must be large enough that the estimate's error is small compared with
the acceptance window, and the target must sit in the MIDDLE of that window
rather than against its edge. See SSTRIDE and TARGET_RANK below.

Correctness does not depend on the estimate being good. A sample can under- or
over-shoot, so the row's collected count is checked against [TOPK,
NUM_FINAL_ITEMS] and a miss redoes the threshold exactly, from a full histogram,
before compacting again. That fallback costs about what the generic operator
costs, so a bad estimate is slow, never wrong.
"""

import math
import os

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.top_k_per_row_prefill import (
    NUM_BINS,
    NUM_FILNAL_ITEMS,
    NUM_THREADS_PER_BLOCK,
    _extract_bin_idx,
    _final_select_radix,
    _num_warps,
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


# One element in SSTRIDE feeds the estimate.
#
# 64 was the first choice and it was wrong: it leaves only ~2000 samples, of
# which ~32 land in the tail being estimated, for a relative standard deviation
# near 18%. The collected count then sits 95% inside [1324, 2772] against a
# buffer of 2048, so the retry fired almost every row and the operator ran
# sample + collect + full histogram + collect again -- 357 us predicted against
# 373 measured, worse than the generic two passes it replaced.
#
# 8 gives ~16000 samples and ~256 in the tail, cutting the deviation to ~6%,
# while the sample pass still costs 75.2/8 = 9.4 us against the 75.2 it removes.
SSTRIDE = int(os.environ.get("FLAGGEMS_MTT_PREFILL_SSTRIDE", "8"))

# Sampling needs enough elements for the estimate to mean anything; below this
# the sample pass costs more than the scan it replaces anyway.
MIN_SPAN = 8192


@triton.jit
def _sampled_prefill(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    stride0,
    stride1,
    TOPK: tl.constexpr,
    TOPKP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
    SSTRIDE: tl.constexpr,
    TARGET_RANK: tl.constexpr,
    NBINS: tl.constexpr,
    NFINAL: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    span = row_end - row_start
    # Base at the row's valid start, so every offset below is already in the
    # caller's convention: indices relative to row_starts[row_id].
    base = logits_ptr + row_id * stride0 + row_start * stride1
    out = out_indices_ptr + row_id * TOPK

    hist = tle.gpu.alloc(
        [NBINS], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    fin = tle.gpu.alloc(
        [NFINAL], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    oidx = tle.gpu.alloc(
        [TOPKP], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    ccnt = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    cfound = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    hp = tle.gpu.local_ptr(hist, (0,))
    fp = tle.gpu.local_ptr(fin, (0,))
    op = tle.gpu.local_ptr(oidx, (0,))
    cp = tle.gpu.local_ptr(ccnt, (0,))
    fvp = tle.gpu.local_ptr(cfound, (0,))

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC)
    bins = tl.arange(0, NBINS)
    one1 = tl.full([BLOCK_SIZE], 1, tl.int32)
    one2 = tl.full([BLOCK_SIZE, VEC], 1, tl.int32)

    # ---- pass 1: histogram of every SSTRIDE-th element -------------------
    for z in tl.range(0, NBINS, BLOCK_SIZE):
        tl.store(hp + z + lane, 0)
    tl.debug_barrier()

    n_s = span // SSTRIDE
    for t in tl.range(0, tl.cdiv(n_s, BLOCK_SIZE)):
        i = (t * BLOCK_SIZE + lane) * SSTRIDE
        m = i < span
        b, _ = _extract_bin_idx(tl.load(base + i * stride1, mask=m, other=0.0),
                                m, 0, STEP=0)
        tl.atomic_add(hp + b, one1, mask=m, sem="relaxed", scope="cta")
    tl.debug_barrier()

    # A lower bin is a larger value, so the prefix count over bins is the count
    # of the largest elements. TARGET_RANK aims at the GEOMETRIC MIDPOINT of the
    # acceptance window [TOPK, NFINAL] rather than its upper edge, which is what
    # the first version did -- aiming at the edge meant half the sampling error
    # pushed the count straight out of the window and into the retry.
    cum = tl.cumsum(tl.load(hp + bins), axis=0)
    target = TARGET_RANK // SSTRIDE + 1
    thr = tl.min(tl.where(cum >= target, bins, NBINS - 1), axis=0)

    # ---- pass 2: collect everything below the threshold -------------------
    # Two attempts. The first uses the sampled threshold; if the count lands
    # outside [TOPK, NFINAL] the estimate was bad, and the retry derives the
    # threshold exactly from a full histogram, which is what the generic
    # operator does anyway.
    for attempt in tl.static_range(0, 2):
        redo = attempt == 1
        if (attempt == 0) or (tl.load(cp) < TOPK) or (tl.load(cp) > NFINAL):
            if redo:
                for z in tl.range(0, NBINS, BLOCK_SIZE):
                    tl.store(hp + z + lane, 0)
                tl.debug_barrier()
                for t in tl.range(0, tl.cdiv(span, BLOCK_SIZE)):
                    i = t * BLOCK_SIZE + lane
                    m = i < span
                    b, _ = _extract_bin_idx(
                        tl.load(base + i * stride1, mask=m, other=0.0), m, 0,
                        STEP=0,
                    )
                    tl.atomic_add(hp + b, one1, mask=m, sem="relaxed",
                                  scope="cta")
                tl.debug_barrier()
                cum2 = tl.cumsum(tl.load(hp + bins), axis=0)
                thr = tl.min(tl.where(cum2 >= TOPK, bins, NBINS - 1), axis=0) + 1

            # hist doubles as the candidate index buffer from here on
            for z in tl.range(0, NBINS, BLOCK_SIZE):
                tl.store(hp + z + lane, 0)
            tl.store(cp, 0)
            tl.store(fvp, 0)
            tl.debug_barrier()

            n_vec = span // (BLOCK_SIZE * VEC)
            for t in tl.range(0, n_vec):
                offs = (t * BLOCK_SIZE * VEC + lane * VEC)[:, None] + vec[None, :]
                x = tl.load(base + offs * stride1)
                b, _ = _extract_bin_idx(x, True, 0, STEP=0)
                # Cast explicitly: b is uint32 and thr int32, and leaving that
                # promotion implicit is what silently selected every element in
                # an earlier version of this kernel.
                take = b.to(tl.int32) < thr
                pos = tl.atomic_add(cp + tl.zeros([BLOCK_SIZE, VEC], tl.int32),
                                    one2, mask=take, sem="relaxed", scope="cta")
                keep = take & (pos < NFINAL)
                tl.store(fp + pos, x, mask=keep)
                tl.store(hp + pos, offs.to(tl.int32), mask=keep)
            tail = n_vec * BLOCK_SIZE * VEC
            for t in tl.range(0, tl.cdiv(span - tail, BLOCK_SIZE)):
                i = tail + t * BLOCK_SIZE + lane
                m = i < span
                x = tl.load(base + i * stride1, mask=m, other=0.0)
                b, _ = _extract_bin_idx(x, m, 0, STEP=0)
                take = m & (b.to(tl.int32) < thr)
                pos = tl.atomic_add(cp + tl.zeros([BLOCK_SIZE], tl.int32),
                                    one1, mask=take, sem="relaxed", scope="cta")
                keep = take & (pos < NFINAL)
                tl.store(fp + pos, x, mask=keep)
                tl.store(hp + pos, i.to(tl.int32), mask=keep)
            tl.debug_barrier()

    # ---- select TOPK out of the candidates --------------------------------
    _final_select_radix(
        hp, fp, cp, fvp, op, None,
        TOPK=TOPK, BLOCK_SIZE=BLOCK_SIZE, MULTIPLE_BLOCKS_PER_ROW=False,
    )
    tl.debug_barrier()

    n_have = tl.minimum(tl.load(cp), TOPK)
    for z in tl.range(0, TOPK, BLOCK_SIZE):
        o = z + lane
        m = o < TOPK
        v = tl.load(op + o, mask=m & (o < n_have), other=-1)
        tl.store(out + o, tl.where(o < n_have, v, -1), mask=m)


def _can_sample(num_rows, vocab_size, stride1, top_k):
    """Route to the sampled kernel only where it was measured to pay."""
    if not HAS_TLE:
        return False
    if stride1 != 1:
        return False
    # The candidate buffer is the generic op's, so the margin has to fit in it.
    if top_k <= 0 or NUM_FILNAL_ITEMS // top_k < 2:
        return False
    return vocab_size >= MIN_SPAN


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for DeepSeek V4 prefill, with a sampled threshold.

    Falls back by *calling* the generic implementation, not by claiming to.
    """
    vocab_size = logits.shape[1]
    if not _can_sample(num_rows, vocab_size, stride1, top_k):
        return _generic_prefill(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        )

    # Midpoint of [top_k, NUM_FILNAL_ITEMS] in log space: maximum room on both
    # sides for the sample's error, instead of hugging one edge.
    target_rank = int(math.sqrt(top_k * NUM_FILNAL_ITEMS))
    _sampled_prefill[(num_rows,)](
        logits,
        indices,
        row_starts,
        row_ends,
        stride0,
        stride1,
        TOPK=top_k,
        TOPKP=triton.next_power_of_2(top_k),
        BLOCK_SIZE=NUM_THREADS_PER_BLOCK,
        VEC=4,
        SSTRIDE=SSTRIDE,
        TARGET_RANK=target_rank,
        NBINS=NUM_BINS,
        NFINAL=NUM_FILNAL_ITEMS,
        num_warps=_num_warps(NUM_THREADS_PER_BLOCK),
    )
