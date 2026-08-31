#!/usr/bin/env python3
"""Is one 2048-wide tl.cumsum worse than rounds of 512, as vLLM does it?

The only structural difference left between the shipped override and vLLM's
sampler.cu is the threshold scan. vLLM does it in kNumBins/kNumThreadsPerBlock
rounds of a 512-wide cub::BlockScan, carrying the running total, and breaks out
of the loop as soon as any thread finds the threshold bin:

    for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++) {
        Scan(...).ExclusiveSum(binCount, prefixSum, totalSum);
        ...
        if (__syncthreads_or(foundThreshold)) break;
        lastValue = totalSum;
    }

The override does one tl.cumsum across all 2048 bins, measured at ~10 us of the
157.4 us the operator takes at (64, 129280) -- 6.4%.

Everything else about the two implementations matches: same shared-memory
atomics for both the histogram and the compaction (atomicAdd on smem in vLLM,
tl.atomic_add scope="cta" here), same vectorized reads, same four-step 11-bit
refinement. So this is the last portable idea; the remaining 4.3x on the
non-read work is Triton codegen, not a writing style.

This measures ONLY the rounds, not the early exit. A data-dependent break inside
a Triton loop is a compiler risk on this backend and is worth attempting only if
rounds alone already pay -- cumsum cost is usually superlinear in width, so they
may. The rounds are EXACT: same 2048 bins, same answer, just accumulated in
pieces, which is why correctness is checked at every point.

An earlier attempt to make this scan cheaper by narrowing the histogram to 512
or 256 bins was catastrophic (0.856 -> 0.379 / 0.368) because coarse bins
overshoot and the exact-retry then fires on every row. That failure does not
apply here: no precision is lost.

    VLLM_PLUGINS=musa PYTHONPATH=src python tools/mtt_scan_rounds.py

Measurement only. Registers nothing, changes no shipped file.
"""

import math
import sys

import torch
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.ops.top_k_per_row_prefill import (
    NUM_BINS,
    NUM_FILNAL_ITEMS,
    NUM_THREADS_PER_BLOCK,
    _extract_bin_idx,
    _final_select_radix,
    _num_warps,
)

DEV = flaggems_vllm.device
OV = "flaggems_vllm.runtime.backend._mthreads.fused.top_k_per_row_prefill"

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = None

try:
    import vllm._custom_ops  # noqa: F401

    HAS_VLLM = hasattr(torch.ops._C, "top_k_per_row_prefill")
except (ImportError, AttributeError, RuntimeError):
    HAS_VLLM = False


@triton.jit
def _scan_probe(
    logits_ptr, out_indices_ptr, row_starts, row_ends, stride0, stride1,
    TOPK: tl.constexpr, TOPKP: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr, SSTRIDE: tl.constexpr, TARGET_RANK: tl.constexpr,
    NBINS: tl.constexpr, NFINAL: tl.constexpr, SCAN_W: tl.constexpr,
    CONDW: tl.constexpr, TLEW: tl.constexpr,
    dbg_ptr, DBG: tl.constexpr,
):
    """The shipped sampled kernel, with the threshold scan switchable.

    SCAN_W == 0 is one cumsum across NBINS, exactly what ships. Anything else is
    NBINS // SCAN_W rounds of that width, carrying the running total forward --
    the same arithmetic, split up.
    """
    row_id = tl.program_id(0)
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    span = row_end - row_start
    base = logits_ptr + row_id * stride0 + row_start * stride1
    out = out_indices_ptr + row_id * TOPK

    hist = tle.gpu.alloc([NBINS], dtype=tl.int32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    fin = tle.gpu.alloc([NFINAL], dtype=tl.float32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    oidx = tle.gpu.alloc([TOPKP], dtype=tl.int32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    ccnt = tle.gpu.alloc([1], dtype=tl.int32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    cfound = tle.gpu.alloc([1], dtype=tl.int32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    thrs = tle.gpu.alloc([1], dtype=tl.int32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)
    hp = tle.gpu.local_ptr(hist, (0,))
    fp = tle.gpu.local_ptr(fin, (0,))
    op = tle.gpu.local_ptr(oidx, (0,))
    cp = tle.gpu.local_ptr(ccnt, (0,))
    fvp = tle.gpu.local_ptr(cfound, (0,))
    tp = tle.gpu.local_ptr(thrs, (0,))

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC)
    bins = tl.arange(0, NBINS)
    one1 = tl.full([BLOCK_SIZE], 1, tl.int32)
    one2 = tl.full([BLOCK_SIZE, VEC], 1, tl.int32)

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

    # ---- the thing under test -------------------------------------------
    target = TARGET_RANK // SSTRIDE + 1
    if TLEW > 0:
        # Verbatim transcription of the GENERIC operator's own threshold search
        # (ops/top_k_per_row_prefill.py, the `threshold_found` loop): rounds of
        # TLEW bins, tle.cumsum carrying the running total, a masked store to
        # pick out the single hit, and `if not threshold_found` as the early
        # exit -- a block-uniform scalar, not a data-dependent break.
        #
        # tle.cumsum is a DIFFERENT primitive from tl.cumsum: it returns
        # (prefix_sum, total), and the total is what a carried rounds loop
        # needs. The earlier rounds experiment used tl.cumsum and measured a
        # fixed 1.5 us per round; that verdict says nothing about this one.
        zeros1 = tl.zeros([TLEW], tl.int32)
        tl.store(tp, NBINS - 1)
        tl.debug_barrier()
        last_value = 0
        threshold_found = tl.full((), False, dtype=tl.int1)
        for r in tl.static_range(0, NBINS // TLEW):
            if not threshold_found:
                b_idx = r * TLEW + tl.arange(0, TLEW)
                counts = tl.load(hp + b_idx)
                ps, tot = tle.cumsum(counts, axis=0, reverse=False)
                ps = ps + last_value
                nxt = ps + counts
                tmask = (ps < target) & (nxt >= target)
                tl.store(tp + zeros1, b_idx, mask=tmask)
                threshold_found = tl.reduce_or(tmask, axis=0)
                last_value = last_value + tot
        tl.debug_barrier()
        thr_c = tl.load(tp)
    elif CONDW > 0:
        # Measured: thr_c is 506-529 across every shape tried, so a first pass
        # over [0, CONDW) finds it every time and the second is dead code at
        # run time. The condition is a block-uniform SCALAR, not a break inside
        # a loop, and the fallback is exact, so a distribution that lands past
        # CONDW costs 1.6 us and is still right.
        i0 = tl.arange(0, CONDW)
        c0 = tl.cumsum(tl.load(hp + i0), axis=0)
        f0 = tl.min(tl.where(c0 >= target, i0, NBINS), axis=0)
        thr_c = f0
        if f0 == NBINS:
            i1 = CONDW + tl.arange(0, NBINS - CONDW)
            c1 = tl.cumsum(tl.load(hp + i1), axis=0) + tl.max(c0, axis=0)
            thr_c = tl.min(tl.where(c1 >= target, i1, NBINS - 1), axis=0)
    elif SCAN_W == 0:
        cum = tl.cumsum(tl.load(hp + bins), axis=0)
        thr_c = tl.min(tl.where(cum >= target, bins, NBINS - 1), axis=0)
    else:
        carry = tl.zeros([1], tl.int32)
        thr_c = NBINS - 1
        for r in tl.static_range(0, NBINS // SCAN_W):
            idx = r * SCAN_W + tl.arange(0, SCAN_W)
            c = tl.cumsum(tl.load(hp + idx), axis=0) + tl.sum(carry, axis=0)
            found = tl.min(tl.where(c >= target, idx, NBINS - 1), axis=0)
            thr_c = tl.minimum(thr_c, found)
            # counts are non-negative so the prefix sum's maximum IS its total
            carry = tl.full([1], 1, tl.int32) * tl.max(c, axis=0)
    if DBG:
        # constexpr, so the timing builds compile this away entirely
        tl.store(dbg_ptr + row_id, thr_c.to(tl.int32))
    thr = thr_c + 1

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
                        STEP=0)
                    tl.atomic_add(hp + b, one1, mask=m, sem="relaxed",
                                  scope="cta")
                tl.debug_barrier()
                cum2 = tl.cumsum(tl.load(hp + bins), axis=0)
                thr = tl.min(tl.where(cum2 >= TOPK, bins, NBINS - 1), axis=0) + 1

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
                take = b.to(tl.int32) < thr
                pos = tl.atomic_add(cp + tl.zeros([BLOCK_SIZE, VEC], tl.int32),
                                    one2, mask=take, sem="relaxed", scope="cta")
                keep = take & (pos < NFINAL)
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
                tl.store(hp + pos, i.to(tl.int32), mask=keep)
            tl.debug_barrier()

    c_have = tl.minimum(tl.load(cp), NFINAL)
    for t in tl.range(0, tl.cdiv(NFINAL, BLOCK_SIZE)):
        j = t * BLOCK_SIZE + lane
        m = j < c_have
        gi = tl.load(hp + j, mask=m, other=0)
        tl.store(fp + j, tl.load(base + gi * stride1, mask=m, other=0.0), mask=m)
    tl.debug_barrier()

    _final_select_radix(hp, fp, cp, fvp, op, None, TOPK=TOPK,
                        BLOCK_SIZE=BLOCK_SIZE, MULTIPLE_BLOCKS_PER_ROW=False)
    tl.debug_barrier()
    n_have = tl.minimum(tl.load(cp), TOPK)
    for z in tl.range(0, TOPK, BLOCK_SIZE):
        o = z + lane
        m = o < TOPK
        v = tl.load(op + o, mask=m & (o < n_have), other=-1)
        tl.store(out + o, tl.where(o < n_have, v, -1), mask=m)


SHAPES = [
    (64, 129280, 1024, 129280, 1),   # the decisive one, 0.902 today
    (4, 16385, 512, 16648, 1),
    (16, 65536, 1024, 65536, 1),     # the num_rows 16-60 range the suite skips
]
# (标签, SCAN_W, CONDW, TLEW) -- 只有一个非零
SCANS = (
    ("一次 2048 tl.cumsum",      0,    0,    0),
    ("通用式 tle 分轮 512",       0,    0,  512),
    ("通用式 tle 分轮 1024",      0,    0, 1024),
    ("条件两轮 tl 首轮1024",      0, 1024,    0),
    ("tl 分轮 2 x 1024",       1024,    0,    0),
    ("tl 分轮 4 x 512",         512,    0,    0),
)


def make_inputs(num_rows, vocab, top_k, stride0, stride1):
    torch.manual_seed(42)
    buf = torch.randn(
        (num_rows - 1) * stride0 + (vocab - 1) * stride1 + 1,
        device=DEV, dtype=torch.float32,
    )
    logits = torch.as_strided(buf, (num_rows, vocab), (stride0, stride1))
    starts = torch.zeros(num_rows, dtype=torch.int32, device=DEV)
    ends = torch.full((num_rows,), vocab, dtype=torch.int32, device=DEV)
    idx = torch.empty((num_rows, top_k), dtype=torch.int32, device=DEV)
    return logits, starts, ends, idx


def block_for(num_rows):
    """Reproduce the shipped dispatcher's block choice, or the probe is testing
    a configuration the operator never launches."""
    import importlib

    ov = importlib.import_module(OV)
    wide = ov._wide_max_rows()
    return ov._WIDE_BLOCK if 0 < num_rows <= wide else NUM_THREADS_PER_BLOCK


def launch(logits, starts, ends, idx, num_rows, vocab, top_k, s0, s1, scan_w,
           dbg=None, condw=0, tlew=0):
    block = block_for(num_rows)
    _scan_probe[(num_rows,)](
        logits, idx, starts, ends, s0, s1,
        TOPK=top_k, TOPKP=triton.next_power_of_2(top_k), BLOCK_SIZE=block,
        VEC=4, SSTRIDE=8,
        TARGET_RANK=int(math.sqrt(top_k * NUM_FILNAL_ITEMS)),
        NBINS=NUM_BINS, NFINAL=NUM_FILNAL_ITEMS, SCAN_W=scan_w,
        CONDW=condw, TLEW=tlew, dbg_ptr=dbg, DBG=dbg is not None,
        num_warps=_num_warps(block),
    )


def correct(logits, idx, vocab, top_k):
    got = idx.to(torch.int64)
    if int(got.min()) < 0 or int(got.max()) >= vocab:
        return "索引越界"
    a = torch.sort(torch.gather(logits, 1, got), dim=1).values
    b = torch.sort(torch.topk(logits, top_k, dim=1, largest=True,
                              sorted=False).values, dim=1).values
    return "OK" if torch.equal(a, b) else "不符"


def bench(fn, reps=100):
    return triton.testing.do_bench(fn, warmup=25, rep=reps,
                                   return_mode="median") * 1000


# Fitted from this card's own four widths (2048/1024/512/256 -> 10.0/5.95/3.63/
# 2.64 us): a fixed ~1.6 us per round -- one cross-thread reduction tree and a
# barrier -- plus a linear term. Every point lands within 3%.
def scan_cost_us(width):
    return 1.6 + 0.0041 * width


def thr_distribution():
    """Where does the threshold bin actually land?

    Rounds lose because each one costs a fixed 1.6 us on top of the linear part,
    so vLLM's version only pays off through its early exit -- it stops at the
    round that contains the threshold bin. Whether that helps here is decided
    entirely by the distribution of thr_c, and nothing about measuring it
    touches control flow, so there is no compiler risk in finding out.
    """
    print("  阈值 bin 的实际分布（决定提前退出有没有意义）\n")
    for num_rows, vocab, top_k, s0, s1 in SHAPES:
        logits, starts, ends, idx = make_inputs(num_rows, vocab, top_k, s0, s1)
        dbg = torch.full((num_rows,), -1, dtype=torch.int32, device=DEV)
        try:
            launch(logits, starts, ends, idx, num_rows, vocab, top_k, s0, s1,
                   0, dbg)
            flaggems_vllm.runtime.torch_device_fn.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  ({num_rows},{vocab})  失败 {type(e).__name__}: "
                  f"{str(e).splitlines()[-1][:44]}")
            continue
        ok = correct(logits, idx, vocab, top_k)
        # Off the device before any statistics: torch.quantile dies on MUSA
        # ("NOT_SUPPORTED in Sort::Run"), and it is 64 integers.
        d = sorted(dbg.cpu().tolist())
        n = len(d)

        def pct(f):
            return d[min(n - 1, max(0, int(f * (n - 1) + 0.5)))]

        print(f"  ({num_rows},{vocab},k={top_k})   正确性 {ok}   "
              f"target_rank={int(math.sqrt(top_k * NUM_FILNAL_ITEMS))}")
        print(f"    thr_c  min={d[0]}  中位={pct(0.5)}  p90={pct(0.9)}  "
              f"max={d[-1]}")
        # What each round width would actually cost, given this distribution.
        print(f"    {'轮宽':>6}{'首轮命中率':>12}{'平均轮数':>10}"
              f"{'期望代价µs':>12}{'vs 现状10.0':>12}")
        for w in (1024, 512, 256):
            rounds = [-(-(x + 1) // w) for x in d]
            hit0 = sum(1 for r in rounds if r <= 1) / n
            avg = sum(rounds) / n
            cost = avg * scan_cost_us(w)
            print(f"    {w:>6}{hit0:>11.1%}{avg:>10.2f}{cost:>12.2f}"
                  f"{cost - 10.0:>+12.2f}")
        print()
    print("  全负输入下的 thr_c（真实 log-prob 是负的，bin 会落到正数区之后）")
    for num_rows, vocab, top_k, s0, s1 in SHAPES[:1]:
        logits, starts, ends, idx = make_inputs(num_rows, vocab, top_k, s0, s1)
        # abs() then negate on the HOST-side tensor is a device op but a trivial
        # elementwise one; quantile was the unsupported call, not this.
        neg = -torch.abs(logits)
        dbg = torch.full((num_rows,), -1, dtype=torch.int32, device=DEV)
        try:
            launch(neg, starts, ends, idx, num_rows, vocab, top_k, s0, s1, 0,
                   dbg)
            flaggems_vllm.runtime.torch_device_fn.synchronize()
            d = sorted(dbg.cpu().tolist())
            print(f"    ({num_rows},{vocab}) 全负   thr_c min={d[0]} "
                  f"中位={d[len(d) // 2]} max={d[-1]}   "
                  f"{'仍 < 1024 ✓' if d[-1] < 1024 else '>= 1024 ← 条件两轮会每行走第二轮'}")
        except Exception as e:  # noqa: BLE001
            print(f"    全负检查失败 {type(e).__name__}: "
                  f"{str(e).splitlines()[-1][:44]}")
    print()
    print("  读法")
    print("    某个轮宽的期望代价明显低于 10.0  => 提前退出值得试, 省的就是那个差值")
    print("    都不低于 10.0                    => thr_c 落得太靠后, 这条路死透,")
    print("      不必冒 Triton 数据依赖 break 的编译风险")
    print("    注意这只是扫描一段的账。整算子 157.5µs, 所以省 7µs 也只到 0.945。")
    return 0

def main():
    print("=" * 80)
    print("  阈值扫描：一次 2048 宽 cumsum  vs  分轮 + 进位（vLLM 的写法）")
    print("=" * 80)
    if tle is None:
        print("  !! 无 TLE, 退出")
        return 1
    print(f"  vLLM 基线: {'有' if HAS_VLLM else '无'}")
    print(f"  出货绑定: {flaggems_vllm.top_k_per_row_prefill.__module__}\n")

    if "--thr" in sys.argv:
        return thr_distribution()

    for num_rows, vocab, top_k, s0, s1 in SHAPES:
        logits, starts, ends, idx = make_inputs(num_rows, vocab, top_k, s0, s1)
        reps = 400 if num_rows <= 8 else 100
        v = None
        if HAS_VLLM:
            vidx = torch.empty_like(idx)
            v = bench(lambda: torch.ops._C.top_k_per_row_prefill(
                logits, starts, ends, vidx, num_rows, s0, s1, top_k), reps)

        idx.fill_(-1)
        flaggems_vllm.top_k_per_row_prefill(
            logits, starts, ends, idx, num_rows, s0, s1, top_k)
        flaggems_vllm.runtime.torch_device_fn.synchronize()
        ship_ok = correct(logits, idx, vocab, top_k)
        ship = bench(lambda: flaggems_vllm.top_k_per_row_prefill(
            logits, starts, ends, idx, num_rows, s0, s1, top_k), reps)

        print(f"  ({num_rows},{vocab},k={top_k})  BLOCK={block_for(num_rows)}"
              + (f"   vLLM {v:.1f} µs" if v else ""))
        print(f"    {'扫描方式':<24}{'µs':>9}{'vs 出货':>9}{'speedup':>9}   正确性")
        print(f"    {'出货算子':<24}{ship:>9.1f}{1.0:>9.3f}"
              f"{(v / ship if v else 0):>9.3f}   {ship_ok}")

        base = None
        for name, w, cw, tw in SCANS:
            try:
                idx.fill_(-1)
                launch(logits, starts, ends, idx, num_rows, vocab, top_k, s0,
                       s1, w, condw=cw, tlew=tw)
                flaggems_vllm.runtime.torch_device_fn.synchronize()
                ok = correct(logits, idx, vocab, top_k)
                t = bench(lambda ww=w, cc=cw, tt=tw: launch(
                    logits, starts, ends, idx, num_rows, vocab, top_k, s0, s1,
                    ww, condw=cc, tlew=tt), reps)
            except Exception as e:  # noqa: BLE001
                print(f"    {name:<24}   失败 {type(e).__name__}: "
                      f"{str(e).splitlines()[-1][:40]}")
                continue
            if w == 0 and cw == 0 and tw == 0:
                base = t
            print(f"    {name:<24}{t:>9.1f}{t / ship:>9.3f}"
                  f"{(v / t if v else 0):>9.3f}   {ok}"
                  + (f"   (对基准 {t / base:.3f})"
                     if base and (w or cw or tw) else ""))
        print()

    print("  读法")
    print("    「探针 一次 2048」应当 ≈ 出货算子; 差得多说明这份复制不忠实,")
    print("      后面所有比较都不能信")
    print("    某个分轮宽度快于它 => 分轮本身就有收益, 再考虑加提前退出")
    print("    分轮都不更快 => cumsum 不随宽度超线性, 提前退出也不用试了,")
    print("      那 4.3 倍非读开销就是 Triton 代码生成, 换写法不解决")
    return 0


if __name__ == "__main__":
    sys.exit(main())
