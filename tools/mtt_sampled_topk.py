#!/usr/bin/env python3
"""Does a sampled threshold turn prefill's two full passes into one?

Measured structure of the operator at (64, 129280) on MTT:

    Pass A  build the 2048-bin histogram      75.6 us
    (threshold scan over the bins)
    Pass B  RE-READ the row, compact candidates 132.6 us
            of which: re-read+extract 59.6, two atomic/store paths 73.0
    final select                               10.2 us
    operator 217.5                             vLLM 141.6

Every local fix failed: privatised histogram atomics 0.97x, num_stages 0.76x,
row splitting 1.03x at best, tile-aggregated atomics 5.3x WORSE (tl.cumsum over
a 2048-wide tile costs far more than the atomics it replaces -- MTT's
same-address atomics are already cheap). And the ceiling on simply deleting the
second read is 0.897, i.e. barely the bar even if the mechanism were free.

The idea here is different: estimate the threshold from a SAMPLE, then make one
full pass that both filters and compacts. Cost would be sample (~1/64 of a read)
+ one full read + compaction, against today's two full reads.

The estimate must be loose, not tight: taking the sample rank at MARGIN x the
expected position collects roughly MARGIN x top_k candidates, so the true top_k
is inside with room. If it is not -- fewer than top_k collected -- the row is
flagged and the real operator would have to fall back, so the flag count is
reported as a correctness signal, not hidden.

This probe deliberately stops before the final select: that stage is ~10 us and
identical under both designs, so it cannot decide anything. What it decides is
whether one pass plus a sample beats two passes.

    VLLM_PLUGINS=musa PYTHONPATH=src python tools/mtt_sampled_topk.py

Measurement only. Registers nothing.
"""

import sys

import torch
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.ops.top_k_per_row_prefill import (
    NUM_THREADS_PER_BLOCK,
    _extract_bin_idx,
    _num_warps,
)

DEV = flaggems_vllm.device

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = None


@triton.jit
def _sampled_pass(
    X, OUTC, OVER, DBG_THR, DBG_SMP, N,
    TOPK: tl.constexpr, BLOCK: tl.constexpr, NUM_BINS: tl.constexpr,
    VEC: tl.constexpr, SSTRIDE: tl.constexpr, MARGIN: tl.constexpr,
    CAP: tl.constexpr,
):
    pid = tl.program_id(0)
    X += pid * N
    lane = tl.arange(0, BLOCK)
    vec = tl.arange(0, VEC)

    hist = tle.gpu.alloc(
        [NUM_BINS], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    cv = tle.gpu.alloc(
        [CAP], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    ci = tle.gpu.alloc(
        [CAP], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    cnt = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    hp = tle.gpu.local_ptr(hist, (0,))
    cvp = tle.gpu.local_ptr(cv, (0,))
    cip = tle.gpu.local_ptr(ci, (0,))
    cp = tle.gpu.local_ptr(cnt, (0,))

    for z in tl.range(0, NUM_BINS, BLOCK):
        tl.store(hp + z + lane, 0)
    tl.store(cp, 0)
    tl.debug_barrier()

    # --- 1. sample pass: every SSTRIDE-th element ---
    n_s = N // SSTRIDE
    ones = tl.full([BLOCK], 1, tl.int32)
    for t in tl.range(0, tl.cdiv(n_s, BLOCK)):
        i = (t * BLOCK + lane) * SSTRIDE
        m = i < N
        b, _ = _extract_bin_idx(tl.load(X + i, mask=m, other=0.0), m, 0, STEP=0)
        tl.atomic_add(hp + b, ones, mask=m, sem="relaxed", scope="cta")
    tl.debug_barrier()

    # --- 2. threshold: smallest bin whose sample prefix reaches MARGIN x rank ---
    bins = tl.arange(0, NUM_BINS)
    h = tl.load(hp + bins)
    smp_total = tl.sum(h, axis=0)
    cum = tl.cumsum(h, axis=0)
    target = (TOPK * MARGIN) // SSTRIDE + 1
    # axis=0 explicitly: a bare tl.min was one of two suspects for the threshold
    # pinning to NUM_BINS-1, the other being the sample histogram never filling.
    # DBG_SMP separates them -- ~N/SSTRIDE means the histogram is fine and the
    # reduction was at fault; 0 means the sample pass never landed.
    thr = tl.min(tl.where(cum >= target, bins, NUM_BINS - 1), axis=0)
    tl.store(DBG_THR + pid, thr.to(tl.int32))
    tl.store(DBG_SMP + pid, smp_total)
    tl.debug_barrier()

    # --- 3. ONE full pass: filter and compact ---
    n_vec = N // (BLOCK * VEC)
    ones2 = tl.full([BLOCK, VEC], 1, tl.int32)
    for t in tl.range(0, n_vec):
        base = t * BLOCK * VEC + lane * VEC
        offs = base[:, None] + vec[None, :]
        x = tl.load(X + offs)
        b, ok = _extract_bin_idx(x, True, 0, STEP=0)
        take = ok & (b < thr)
        pos = tl.atomic_add(cp + tl.zeros([BLOCK, VEC], tl.int32), ones2,
                            mask=take, sem="relaxed", scope="cta")
        keep = take & (pos < CAP)
        tl.store(cvp + (pos % CAP), x, mask=keep)
        tl.store(cip + (pos % CAP), offs.to(tl.int32), mask=keep)
    tail = n_vec * BLOCK * VEC
    for t in tl.range(0, tl.cdiv(N - tail, BLOCK)):
        i = tail + t * BLOCK + lane
        m = i < N
        x = tl.load(X + i, mask=m, other=0.0)
        b, ok = _extract_bin_idx(x, m, 0, STEP=0)
        take = ok & (b < thr)
        pos = tl.atomic_add(cp + tl.zeros([BLOCK], tl.int32), ones,
                            mask=take, sem="relaxed", scope="cta")
        keep = take & (pos < CAP)
        tl.store(cvp + (pos % CAP), x, mask=keep)
        tl.store(cip + (pos % CAP), i.to(tl.int32), mask=keep)
    tl.debug_barrier()

    c = tl.load(cp)
    tl.store(OUTC + pid, c)
    # too few collected -> the estimate was too tight and the real op must retry;
    # more than CAP -> the buffer dropped candidates. Both are failures.
    tl.store(OVER + pid, tl.where((c < TOPK) | (c > CAP), 1, 0))


def main():
    print("=" * 78)
    print("  采样阈值 + 单遍压缩  vs  现状两遍")
    print("=" * 78)
    if tle is None:
        print("  !! 无 TLE, 退出")
        return 1

    CASES = [(64, 129280, 1024), (16383, 4095, 512), (12961, 4100, 512)]
    BLOCK = NUM_THREADS_PER_BLOCK
    nw = _num_warps(BLOCK)
    print(f"  {'shape':>15}{'SS':>5}{'MG':>4}{'CAP':>6}{'采样us':>9}"
          f"{'现状us':>9}{'thr':>6}{'样本数':>8}{'期望':>7}{'候选':>9}{'失败':>7}")
    for rows, vocab, top_k in CASES:
        torch.manual_seed(42)
        x = torch.randn((rows, vocab), dtype=torch.float32, device=DEV)
        starts = torch.zeros((rows,), dtype=torch.int32, device=DEV)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=DEV)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=DEV)
        full = triton.testing.do_bench(
            lambda: flaggems_vllm.top_k_per_row_prefill(
                x, starts, ends, idx, rows, x.stride(0), x.stride(1), top_k
            ), warmup=25, rep=100, return_mode="median") * 1000

        for sstride, margin in ((64, 4), (64, 2), (256, 4)):
            cap = triton.next_power_of_2(top_k * margin)
            smem = (2048 * 4 + cap * 8) / 1024
            if smem > 100:
                print(f"  {f'({rows},{vocab})':>15}{sstride:>9}{margin:>8}{cap:>6}"
                      f"   跳过: 需 {smem:.0f} KB smem")
                continue
            c = torch.zeros((rows,), dtype=torch.int32, device=DEV)
            over = torch.zeros((rows,), dtype=torch.int32, device=DEV)
            dthr = torch.zeros((rows,), dtype=torch.int32, device=DEV)
            dsmp = torch.zeros((rows,), dtype=torch.int32, device=DEV)
            try:
                _sampled_pass[(rows,)](
                    x, c, over, dthr, dsmp, vocab, TOPK=top_k, BLOCK=BLOCK,
                    NUM_BINS=2048, VEC=4, SSTRIDE=sstride, MARGIN=margin,
                    CAP=cap, num_warps=nw,
                )
                flaggems_vllm.runtime.torch_device_fn.synchronize()
                t = triton.testing.do_bench(
                    lambda: _sampled_pass[(rows,)](
                        x, c, over, dthr, dsmp, vocab, TOPK=top_k,
                        BLOCK=BLOCK, NUM_BINS=2048, VEC=4, SSTRIDE=sstride,
                        MARGIN=margin, CAP=cap, num_warps=nw,
                    ), warmup=25, rep=100, return_mode="median") * 1000
            except Exception as e:  # noqa: BLE001
                print(f"  {f'({rows},{vocab})':>15}{sstride:>9}{margin:>8}{cap:>6}"
                      f"   失败 {type(e).__name__}: {str(e)[:30]}")
                continue
            med = int(c.median())
            bad = int(over.sum())
            exp = vocab // sstride
            print(f"  {f'({rows},{vocab})':>15}{sstride:>5}{margin:>4}{cap:>6}"
                  f"{t:>9.1f}{full:>9.1f}{int(dthr.median()):>6}"
                  f"{int(dsmp.median()):>8}{exp:>7}{med:>9}{bad:>7}", flush=True)
    print("\n  样本数 ~= 期望 而 thr = 2047  => 直方图没问题, 是归约算错")
    print("  样本数 = 0                    => 采样那一遍根本没写进直方图")
    print("  thr 合理(几十~几百) 而候选 ~= MARGIN*top_k => 修好了, 看时间")
    print("\n  采样版明显低于现状 且 失败行=0  => 结构成立, 值得写完整 override")
    print("  失败行 > 0                       => 估计太紧, 加大 MARGIN 再看")
    print("  采样版不低于现状                 => 单遍省不出来, 这条也关掉")
    print("\n  注意: 采样版不含 final select(~10us), 两边比较时记得加回去。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
