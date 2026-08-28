#!/usr/bin/env python3
"""Is the candidate VALUE store worth its cost, or should values be re-gathered?

The shipped MTT override reaches 0.860 at (64, 129280) against vLLM's 141.6 us,
still short of the 0.9 acceptance bar. Its single remaining pass does, per hit,
one atomic and TWO scattered shared-memory stores:

    pos = atomic_add(cnt)
    store(fp + pos, x)              <- the value
    store(hp + pos, offs)           <- the index

The value store is removable: _final_select_radix needs the values, but they can
be re-read from global memory afterwards using the indices that were kept. That
trades ~NFINAL scattered smem stores spread through the hot loop for NFINAL
scattered GLOBAL loads in a short, separate, fully parallel pass.

Whether that is a win is not obvious and has not been measured, so this probe
measures it instead of arguing about it.

Four variants share ONE kernel body, selected by a constexpr, so nothing but the
store/atomic structure differs between them:

    base      atomic + store value + store index      (what ships; correct)
    gather    atomic + store index, values re-read    (the candidate; correct)
    nostore   atomic, neither store                   (INVALID: bounds store cost)
    noatomic  deterministic pos, both stores          (INVALID: bounds atomic cost)

Each is timed twice:

    --nofinal   kernel stops after collection -> isolates the three costs
    full        the whole operator            -> the number that decides 0.9

The truncated form folds a checksum of the candidate buffers into its output.
Without it the stores feed nothing and the compiler is free to delete them --
that exact mistake once produced 3.4 TB/s on a card whose peak is 1.3.

    VLLM_PLUGINS=musa PYTHONPATH=src python tools/mtt_store_split.py

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

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = None

try:
    import vllm._custom_ops  # noqa: F401

    HAS_VLLM = hasattr(torch.ops._C, "top_k_per_row_prefill")
except (ImportError, AttributeError, RuntimeError):
    HAS_VLLM = False

# Host-side names only. Inside the kernel these are written as integer
# LITERALS: Triton refuses to read a plain-int module global from a jit
# function ("Cannot access global variable ... only constexpr"), and a
# tl.constexpr(3) would then not work as a dict key out here.
BASE, GATHER, NOSTORE, NOATOMIC = 0, 1, 2, 3
NAMES = {BASE: "base", GATHER: "gather", NOSTORE: "nostore", NOATOMIC: "noatomic"}
VALID = (BASE, GATHER)


@triton.jit
def _probe(
    logits_ptr, out_indices_ptr, row_starts, row_ends, stride0, stride1,
    TOPK: tl.constexpr, TOPKP: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr, SSTRIDE: tl.constexpr, TARGET_RANK: tl.constexpr,
    SBINS: tl.constexpr, SSHIFT: tl.constexpr, NBINS: tl.constexpr,
    NFINAL: tl.constexpr, MODE: tl.constexpr, NOFINAL: tl.constexpr,
):
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

    # ---- pass 1: sample histogram (identical in every mode) ---------------
    for z in tl.range(0, SBINS, BLOCK_SIZE):
        tl.store(hp + z + lane, 0)
    tl.debug_barrier()
    n_s = span // SSTRIDE
    for t in tl.range(0, tl.cdiv(n_s, BLOCK_SIZE)):
        i = (t * BLOCK_SIZE + lane) * SSTRIDE
        m = i < span
        b, _ = _extract_bin_idx(tl.load(base + i * stride1, mask=m, other=0.0),
                                m, 0, STEP=0)
        tl.atomic_add(hp + (b >> SSHIFT), one1, mask=m, sem="relaxed",
                      scope="cta")
    tl.debug_barrier()
    sbins = tl.arange(0, SBINS)
    cum = tl.cumsum(tl.load(hp + sbins), axis=0)
    target = TARGET_RANK // SSTRIDE + 1
    thr_c = tl.min(tl.where(cum >= target, sbins, SBINS - 1), axis=0)
    thr = (thr_c + 1) << SSHIFT

    # ---- pass 2: collect -------------------------------------------------
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
                if MODE == 3:  # noatomic
                    # Same store traffic, same mask, no counter contention.
                    pos = offs % NFINAL
                    keep = take
                else:
                    pos = tl.atomic_add(
                        cp + tl.zeros([BLOCK_SIZE, VEC], tl.int32), one2,
                        mask=take, sem="relaxed", scope="cta")
                    keep = take & (pos < NFINAL)
                if MODE == 0 or MODE == 3:  # base / noatomic
                    tl.store(fp + pos, x, mask=keep)
                    tl.store(hp + pos, offs.to(tl.int32), mask=keep)
                if MODE == 1:  # gather
                    tl.store(hp + pos, offs.to(tl.int32), mask=keep)
            tail = n_vec * BLOCK_SIZE * VEC
            for t in tl.range(0, tl.cdiv(span - tail, BLOCK_SIZE)):
                i = tail + t * BLOCK_SIZE + lane
                m = i < span
                x = tl.load(base + i * stride1, mask=m, other=0.0)
                b, _ = _extract_bin_idx(x, m, 0, STEP=0)
                take = m & (b.to(tl.int32) < thr)
                if MODE == 3:  # noatomic
                    pos = i % NFINAL
                    keep = take
                else:
                    pos = tl.atomic_add(cp + tl.zeros([BLOCK_SIZE], tl.int32),
                                        one1, mask=take, sem="relaxed",
                                        scope="cta")
                    keep = take & (pos < NFINAL)
                if MODE == 0 or MODE == 3:  # base / noatomic
                    tl.store(fp + pos, x, mask=keep)
                    tl.store(hp + pos, i.to(tl.int32), mask=keep)
                if MODE == 1:  # gather
                    tl.store(hp + pos, i.to(tl.int32), mask=keep)
            if MODE == 3:  # noatomic
                # No counter was maintained; pin it so the retry does not fire
                # and the timing stays comparable with the other modes.
                tl.store(cp, NFINAL)
            tl.debug_barrier()

    # ---- the candidate's extra pass: re-read the values ------------------
    if MODE == 1:  # gather
        c_have = tl.minimum(tl.load(cp), NFINAL)
        for t in tl.range(0, tl.cdiv(NFINAL, BLOCK_SIZE)):
            j = t * BLOCK_SIZE + lane
            m = j < c_have
            gi = tl.load(hp + j, mask=m, other=0)
            tl.store(fp + j, tl.load(base + gi * stride1, mask=m, other=0.0),
                     mask=m)
        tl.debug_barrier()

    if NOFINAL:
        # Consume both candidate buffers, or the stores above have no reader
        # and the compiler may delete the very thing being measured.
        #
        # Written as a masked store of a TENSOR over the same output loop the
        # shipped override already uses, not as a scalar store into out+lane:
        # this block is the only code here with no counterpart in a kernel
        # known to compile on this card, so it copies a proven shape.
        acc = tl.zeros([BLOCK_SIZE], tl.int32)
        for t in tl.range(0, tl.cdiv(NFINAL, BLOCK_SIZE)):
            j = t * BLOCK_SIZE + lane
            acc += tl.load(hp + j) + tl.load(fp + j).to(tl.int32)
        acc += tl.load(cp)
        for z in tl.range(0, TOPK, BLOCK_SIZE):
            o = z + lane
            tl.store(out + o, acc, mask=o < TOPK)
    else:
        _final_select_radix(hp, fp, cp, fvp, op, None, TOPK=TOPK,
                            BLOCK_SIZE=BLOCK_SIZE,
                            MULTIPLE_BLOCKS_PER_ROW=False)
        tl.debug_barrier()
        n_have = tl.minimum(tl.load(cp), TOPK)
        for z in tl.range(0, TOPK, BLOCK_SIZE):
            o = z + lane
            m = o < TOPK
            v = tl.load(op + o, mask=m & (o < n_have), other=-1)
            tl.store(out + o, tl.where(o < n_have, v, -1), mask=m)


# num_rows, vocab, top_k, stride0, stride1.  The first two are the benchmark's
# own shapes that the sampled path actually serves; the last two fill the
# num_rows 16-60 hole the benchmark set has no sample for.
SHAPES = [
    (64, 129280, 1024, 129280, 1),
    (4, 16385, 512, 16648, 1),
    (32, 65536, 1024, 65536, 1),
    (16, 65536, 1024, 65536, 1),
]


def make_inputs(num_rows, vocab, top_k, stride0, stride1):
    """Exactly the benchmark's construction: a strided view, not a fresh 2-D."""
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


def launch(mode, nofinal, logits, starts, ends, idx, num_rows, vocab, top_k,
           stride0, stride1, block=NUM_THREADS_PER_BLOCK):
    _probe[(num_rows,)](
        logits, idx, starts, ends, stride0, stride1,
        TOPK=top_k, TOPKP=triton.next_power_of_2(top_k),
        BLOCK_SIZE=block, VEC=4, SSTRIDE=8,
        TARGET_RANK=int(math.sqrt(top_k * NUM_FILNAL_ITEMS)),
        SBINS=NUM_BINS, SSHIFT=0, NBINS=NUM_BINS, NFINAL=NUM_FILNAL_ITEMS,
        MODE=mode, NOFINAL=nofinal,
        num_warps=_num_warps(block),
    )


def bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100,
                                   return_mode="median") * 1000


def correct(logits, idx, num_rows, vocab, top_k):
    """Selected VALUES must match torch.topk's as a multiset (ties reorder)."""
    got = idx.to(torch.int64)
    if int(got.min()) < 0 or int(got.max()) >= vocab:
        return "索引越界"
    picked = torch.gather(logits, 1, got)
    ref = torch.topk(logits, top_k, dim=1, largest=True, sorted=False).values
    a = torch.sort(picked, dim=1).values
    b = torch.sort(ref, dim=1).values
    return "OK" if torch.equal(a, b) else "不符"


SWEEP_SPANS = (16384, 24576, 32768, 49152, 65536, 98304, 129280)
# (num_rows, top_k). 4 rows is where the benchmark's regressed shape lives and
# where only 4 of 60 SMs have work, so the gather's latency is fully exposed;
# 64 rows is the shape that crossed 0.9. Both top_k values matter because the
# gather reads NFINAL entries regardless of top_k while the saving tracks the
# trigger rate, which does depend on it.
SWEEP_CASES = ((4, 512), (64, 512), (64, 1024))


def sweep():
    """Where does re-reading stop paying?

    The gather costs a fixed NFINAL scattered global loads; the store it removes
    costs in proportion to the span. So there is a crossover, and the shipped
    gate should sit at a measured one rather than at the lowest span that
    happened to win.
    """
    print("  span 扫描: gather 相对 base 的完整耗时比 (< 1.000 = gather 更快)\n")
    for num_rows, top_k in SWEEP_CASES:
        # A 4-row shape runs in tens of microseconds and this box's spread on
        # those is over 10%, so give it four times the reps.
        reps = 400 if num_rows <= 8 else 100
        print(f"  num_rows={num_rows}  top_k={top_k}   (rep={reps})")
        print(f"    {'span':>8}{'base us':>10}{'gather us':>11}{'比值':>9}"
              f"{'vLLM us':>10}{'base sp':>9}{'gather sp':>11}")
        for span in SWEEP_SPANS:
            logits, starts, ends, idx = make_inputs(
                num_rows, span, top_k, span, 1)
            try:
                ts = {}
                for mode in (BASE, GATHER):
                    launch(mode, False, logits, starts, ends, idx, num_rows,
                           span, top_k, span, 1)
                    flaggems_vllm.runtime.torch_device_fn.synchronize()
                    ts[mode] = triton.testing.do_bench(
                        lambda m=mode: launch(m, False, logits, starts, ends,
                                              idx, num_rows, span, top_k,
                                              span, 1),
                        warmup=25, rep=reps, return_mode="median") * 1000
                v = None
                if HAS_VLLM:
                    vidx = torch.empty_like(idx)
                    v = triton.testing.do_bench(
                        lambda: torch.ops._C.top_k_per_row_prefill(
                            logits, starts, ends, vidx, num_rows, span, 1,
                            top_k),
                        warmup=25, rep=reps, return_mode="median") * 1000
            except Exception as e:  # noqa: BLE001
                print(f"    {span:>8}   失败 {type(e).__name__}: "
                      f"{str(e).splitlines()[-1][:44]}")
                continue
            vs = f"{v:.1f}" if v else "—"
            bs = f"{v / ts[BASE]:.3f}" if v else ""
            gs = f"{v / ts[GATHER]:.3f}" if v else ""
            print(f"    {span:>8}{ts[BASE]:>10.1f}{ts[GATHER]:>11.1f}"
                  f"{ts[GATHER] / ts[BASE]:>9.3f}{vs:>10}{bs:>9}{gs:>11}",
                  flush=True)
        print()
    print("  比值第一次降到 1.000 以下的 span = 门控该放的位置。")
    print("  若三组的交叉点不一致, 门控就不该只看 span -- 说明还有别的变量。")
    return 0


# The sampled kernel hardcodes BLOCK_SIZE=512 -- the SM-derived wide-block gate
# lives in the generic operator and never reaches this path. 512 threads is 16
# warps against this part's 32-warp SM ceiling, so two programs fit per SM and
# the card holds 120. Below that, SMs sit idle and widening the block is the
# only way to put more threads on a row.
BLOCK_CASES = (
    # span, top_k -- the two benchmark shapes the sampled path actually serves
    (16384, 512),
    (129280, 1024),
)
BLOCK_ROWS = (4, 16, 32, 60, 64, 96)
BLOCK_WIDTHS = (512, 1024)


def block_sweep(cases=BLOCK_CASES, rows=BLOCK_ROWS):
    """Does a wider block pay where the grid cannot fill the card?

    At num_rows=4 the whole kernel runs at ~18 GB/s against 645 achievable, so
    it is latency-bound, not bandwidth-bound, and more threads per row is the
    obvious lever. 60 SMs x 2 programs means widening should stop paying
    somewhere above 60 rows -- if it does not, the occupancy model is wrong and
    the gate should not be derived from it.
    """
    warp, maxt = None, None
    try:
        from flaggems_vllm.ops.top_k_per_row_prefill import _launch_geometry

        warp, maxt = _launch_geometry()
        print(f"  warp={warp}  max_threads/block={maxt}  "
              f"-> BLOCK 512 用 {_num_warps(512)} warp, "
              f"1024 用 {_num_warps(1024)} warp")
    except Exception as e:  # noqa: BLE001
        print(f"  设备几何读取失败: {e}")
    print("  gather 变体, 完整算子。比值 = 1024 耗时 / 512 耗时\n")

    for span, top_k in cases:
        print(f"  span={span}  top_k={top_k}")
        print(f"    {'rows':>6}{'512 us':>10}{'1024 us':>10}{'比值':>9}"
              f"{'vLLM us':>10}{'sp@512':>9}{'sp@1024':>10}")
        for num_rows in rows:
            reps = 400 if num_rows <= 8 else 100
            logits, starts, ends, idx = make_inputs(
                num_rows, span, top_k, span, 1)
            ts, failed = {}, None
            for blk in BLOCK_WIDTHS:
                try:
                    launch(GATHER, False, logits, starts, ends, idx, num_rows,
                           span, top_k, span, 1, block=blk)
                    flaggems_vllm.runtime.torch_device_fn.synchronize()
                    ts[blk] = triton.testing.do_bench(
                        lambda b=blk: launch(
                            GATHER, False, logits, starts, ends, idx, num_rows,
                            span, top_k, span, 1, block=b),
                        warmup=25, rep=reps, return_mode="median") * 1000
                except Exception as e:  # noqa: BLE001
                    failed = f"BLOCK={blk} {type(e).__name__}: " \
                             f"{str(e).splitlines()[-1][:40]}"
            if failed and len(ts) < 2:
                print(f"    {num_rows:>6}   {failed}")
                continue
            # Correctness is not incidental here: a wider block changes the
            # candidate-buffer indexing, so a faster wrong answer is the thing
            # to watch for.
            idx.fill_(-1)
            launch(GATHER, False, logits, starts, ends, idx, num_rows, span,
                   top_k, span, 1, block=1024)
            flaggems_vllm.runtime.torch_device_fn.synchronize()
            verdict = correct(logits, idx, num_rows, span, top_k)
            v = None
            if HAS_VLLM:
                vidx = torch.empty_like(idx)
                v = triton.testing.do_bench(
                    lambda: torch.ops._C.top_k_per_row_prefill(
                        logits, starts, ends, vidx, num_rows, span, 1, top_k),
                    warmup=25, rep=reps, return_mode="median") * 1000
            vs = f"{v:.1f}" if v else "—"
            s5 = f"{v / ts[512]:.3f}" if v else ""
            s10 = f"{v / ts[1024]:.3f}" if v else ""
            mark = "" if verdict == "OK" else f"   1024 {verdict}"
            print(f"    {num_rows:>6}{ts[512]:>10.1f}{ts[1024]:>10.1f}"
                  f"{ts[1024] / ts[512]:>9.3f}{vs:>10}{s5:>9}{s10:>10}{mark}",
                  flush=True)
        print()
    print("  比值 < 1.000 = 加宽更快。若它在 60 行以上才转正, 占用率模型成立,")
    print("  门控按 SM 数推导; 若转折点对不上 60, 就不能用那个模型定门控。")
    return 0


def compile_check():
    """Compile every (mode, nofinal) once and print the FULL error.

    Without this a single compile fault is reported once per shape per variant
    -- sixteen truncated copies of one message, none of them readable.
    """
    print("  编译自检 (最小形状, 每个组合一次)")
    num_rows, vocab, top_k, s0, s1 = 2, 20480, 512, 20480, 1
    logits, starts, ends, idx = make_inputs(num_rows, vocab, top_k, s0, s1)
    ok = {}
    for nofinal in (True, False):
        for mode in (BASE, GATHER, NOSTORE, NOATOMIC):
            if not nofinal and mode not in VALID:
                continue
            try:
                launch(mode, nofinal, logits, starts, ends, idx, num_rows,
                       vocab, top_k, s0, s1)
                flaggems_vllm.runtime.torch_device_fn.synchronize()
                ok[(mode, nofinal)] = True
            except Exception as e:  # noqa: BLE001
                ok[(mode, nofinal)] = False
                tail = "截断" if nofinal else "完整"
                print(f"\n  ---- {NAMES[mode]} / {tail}: {type(e).__name__} ----")
                print(str(e))
                print("  " + "-" * 60)
    good = sum(1 for v in ok.values() if v)
    print(f"\n  {good}/{len(ok)} 个组合编译通过\n")
    return ok


def main():
    print("=" * 78)
    print("  候选缓冲: 存值+存索引  vs  只存索引后回读")
    print("=" * 78)
    if tle is None:
        print("  !! 无 TLE, 退出")
        return 1
    print(f"  vLLM 基线: {'有' if HAS_VLLM else '无 (只报相对 base 的比值)'}\n")

    ok = compile_check()
    if "--block" in sys.argv or "--blockspan" in sys.argv:
        if not ok.get((GATHER, False)):
            print("  gather 完整版未编译通过, 扫描无意义。")
            return 1
        if "--blockspan" in sys.argv:
            # The row cliff is now measured at exactly 60/64, so sweep span
            # instead, at two row counts safely below it. top_k is held at 1024
            # so the only variable is span: widening speeds up the SCAN, and
            # whether that pays depends on how much of the time the scan is.
            return block_sweep(cases=tuple((s, 1024) for s in SWEEP_SPANS),
                               rows=(4, 32))
        return block_sweep()
    if "--sweep" in sys.argv:
        if not ok.get((BASE, False)) or not ok.get((GATHER, False)):
            print("  base/gather 完整版未编译通过, 扫描无意义。")
            return 1
        return sweep()
    if not any(ok.values()):
        print("  编译全数失败, 不进行任何计时。上面第一段完整错误就是原因。")
        return 1

    for num_rows, vocab, top_k, s0, s1 in SHAPES:
        tag = f"({num_rows},{vocab},k={top_k})"
        logits, starts, ends, idx = make_inputs(num_rows, vocab, top_k, s0, s1)

        vllm_us = None
        if HAS_VLLM:
            vidx = torch.empty_like(idx)
            try:
                vllm_us = bench(lambda: torch.ops._C.top_k_per_row_prefill(
                    logits, starts, ends, vidx, num_rows, s0, s1, top_k))
            except Exception as e:  # noqa: BLE001
                print(f"  {tag}  vLLM 基线失败: {type(e).__name__}: {str(e)[:40]}")

        print(f"  {tag}" + (f"   vLLM {vllm_us:.1f} us" if vllm_us else ""))
        print(f"    {'变体':<10}{'截断us':>9}{'完整us':>9}{'vs base':>9}"
              f"{'speedup':>9}   正确性")

        base_full = None
        for mode in (BASE, GATHER, NOSTORE, NOATOMIC):
            if not ok.get((mode, True)):
                print(f"    {NAMES[mode]:<10}  截断: 编译未通过, 跳过")
                continue
            try:
                launch(mode, True, logits, starts, ends, idx, num_rows, vocab,
                       top_k, s0, s1)
                flaggems_vllm.runtime.torch_device_fn.synchronize()
                t_cut = bench(lambda m=mode: launch(
                    m, True, logits, starts, ends, idx, num_rows, vocab, top_k,
                    s0, s1))
            except Exception as e:  # noqa: BLE001
                print(f"    {NAMES[mode]:<10}  截断失败 {type(e).__name__}: "
                      f"{str(e).splitlines()[-1][:60]}")
                continue

            t_full = None
            verdict = ("—  (无效变体, 仅作上界)" if mode not in VALID
                       else "完整版编译未通过")
            if mode in VALID and ok.get((mode, False)):
                try:
                    idx.fill_(-1)
                    launch(mode, False, logits, starts, ends, idx, num_rows,
                           vocab, top_k, s0, s1)
                    flaggems_vllm.runtime.torch_device_fn.synchronize()
                    verdict = correct(logits, idx, num_rows, vocab, top_k)
                    t_full = bench(lambda m=mode: launch(
                        m, False, logits, starts, ends, idx, num_rows, vocab,
                        top_k, s0, s1))
                    if mode == BASE:
                        base_full = t_full
                except Exception as e:  # noqa: BLE001
                    verdict = (f"完整失败 {type(e).__name__}: "
                               f"{str(e).splitlines()[-1][:40]}")

            rel = f"{t_full / base_full:.3f}" if (t_full and base_full) else ""
            sp = f"{vllm_us / t_full:.3f}" if (t_full and vllm_us) else ""
            fu = f"{t_full:.1f}" if t_full else "—"
            print(f"    {NAMES[mode]:<10}{t_cut:>9.1f}{fu:>9}{rel:>9}{sp:>9}"
                  f"   {verdict}", flush=True)
        print()

    print("  读法")
    print("    base-nostore   = 两次散写的代价")
    print("    base-noatomic  = 原子的代价")
    print("    gather 的 speedup >= 0.9 且正确性 OK  => 改, 越线")
    print("    gather 快于 base 但仍 < 0.9          => 改, 但要如实说明未达标")
    print("    gather 不快于 base                   => 回读不划算, MTT 收在 0.860")
    print("    nostore 与 base 相差无几             => 瓶颈不在 store, 这条思路作废")
    return 0


if __name__ == "__main__":
    sys.exit(main())
