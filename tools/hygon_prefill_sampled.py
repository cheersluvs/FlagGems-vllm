"""Does the sampled threshold that carried decode also carry prefill?

Decode went 0.847 -> 1.28 by replacing the radix algorithm's two passes with
one: sample ~8 tiles of the row, take the bin holding rank top_k scaled to the
sample, then make ONE pass appending everything at or above it and merge the
few thousand candidates. Prefill has the same two-pass shape, so the same lever
should exist -- but the arithmetic is different, because the merge is itself a
radix over the candidates:

    sampled ~ n + n/STRIDE + 2C        against        exact ~ 2n

with C the candidates admitted, about 6 * top_k. So it pays only when
n / top_k is bigger than ~12, and the benchmark's shapes sit on both sides:

    (64,129280) k=1024   126     (16380,5115)  10
    (4,16385)   k=512     32     (12961,4100)   8
    (4,8193)    k=512     16     (16383,4095)   8
                                 (4100,1025)    2

which is exactly the split the shipped override already makes: the three
sparse shapes are the ones its prefix-sum slot allocation does nothing for,
and they are its three worst ratios (0.463, 0.551, 0.606).

Two things make this a measurement rather than the arithmetic above. The
per-program kernel floor is ~19 us and this pipeline has five launches, which
the 4-row shapes may not be able to pay; and prefill rows are a range
[row_start, row_end) inside a padded row, so every kernel carries the offset.

DEVICE time, from the profiler, not wall: at 4 rows the shipped operator's own
wall time is host dispatch, and comparing a five-launch pipeline against it in
wall time is the trap this card has already sprung three times.

    tools/vendor_probe.sh tools/hygon_prefill_sampled.py hygon_prefill_sampled
"""

import sys
from importlib import import_module

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

import flaggems_vllm

# (num_rows, vocab, top_k, stride0, stride1) -- the benchmark's own shapes
SHAPES = [
    (64, 129280, 1024, 129280, 1),
    (4, 8193, 512, 8456, 1),
    (4, 16385, 512, 16648, 1),
    (16383, 4095, 512, 4352, 1),
    (12961, 4100, 512, 4360, 1),
    (16380, 5115, 512, 5376, 1),
    (4100, 1025, 512, 1288, 1),
]

NB = 2048
BLOCK = 512
WARPS = 8
SAMPLE_TILES = 8
SAFETY = 4
CAP_FACTOR = 16
MIN_CHUNK = 512


def device_us(fn, iters=20, warmup=5):
    """Total device time per call, summed over every kernel it launches."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = 0.0
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0)
        total += t or 0.0
    return total / iters


@triton.jit
def _scan_threshold(base, target, NB: tl.constexpr, BLOCK: tl.constexpr):
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
def _hist_pass(
    logits_ptr, base, row, stride0, s, e, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    lane = tl.arange(0, BLOCK)
    for t in tl.range(0, tl.cdiv(e - s, BLOCK * STRIDE)):
        i = s + t * BLOCK * STRIDE + lane
        m = i < e
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
    starts_ptr,
    ends_ptr,
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
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = hist_ptr + row * NB
    for t in tl.static_range(NB // BLOCK):
        tl.store(base + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))
    tl.store(cnt_ptr + row, 0)
    tl.debug_barrier()
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    _hist_pass(logits_ptr, base, row, stride0, s, e, STRIDE, BLOCK)
    tl.debug_barrier()
    total = tl.zeros([], tl.int32)
    for t in tl.static_range(NB // BLOCK):
        total += tl.sum(tl.load(base + t * BLOCK + lane), axis=0)
    n = tl.maximum(e - s, 1)
    target = tl.maximum(tl.cdiv(TOPK * total, n), 1) * SAFETY
    tl.store(thr_ptr + row, _scan_threshold(base, target, NB, BLOCK))


@triton.jit
def _select(
    logits_ptr,
    starts_ptr,
    ends_ptr,
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
    """One pass. SPLIT programs share a row's counter, so that atomic is
    scoped to the device; the indices stored are relative to row_start, which
    is what the operator returns."""
    pid = tl.program_id(0)
    row = pid // SPLIT
    chunk = pid % SPLIT
    lane = tl.arange(0, BLOCK)
    thr = tl.load(thr_ptr + row)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    start = s + chunk * CHUNK
    end = tl.minimum(start + CHUNK, e)
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
        tl.store(cand_idx_ptr + row * CAP + pos, (i - s).to(tl.int32), mask=keep)
        tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)


@triton.jit
def _select_exact(
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
        i = s + t * BLOCK + lane
        m = i < e
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
        tl.store(cand_idx_ptr + row * CAP + pos, (i - s).to(tl.int32), mask=keep)
        tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)


@triton.jit
def _fixup(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    cnt_ptr,
    merge_lens_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    stride0,
    TOPK: tl.constexpr,
    NB: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.load(cnt_ptr + row)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    if (c >= tl.minimum(TOPK, e - s)) & (c <= CAP):
        tl.store(merge_lens_ptr + row, c)
        return
    lane = tl.arange(0, BLOCK)
    base = hist_ptr + row * NB
    for t in tl.static_range(NB // BLOCK):
        tl.store(base + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))
    tl.debug_barrier()
    _hist_pass(logits_ptr, base, row, stride0, s, e, 1, BLOCK)
    tl.debug_barrier()
    thr = _scan_threshold(base, TOPK, NB, BLOCK)
    tl.store(cnt_ptr + row, 0)
    tl.debug_barrier()
    cnt_ptrs = cnt_ptr + row + tl.zeros([BLOCK], tl.int32)
    _select_exact(
        logits_ptr,
        row,
        stride0,
        s,
        e,
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
        s,
        e,
        thr,
        cnt_ptrs,
        cand_idx_ptr,
        cand_val_ptr,
        True,
        CAP,
        BLOCK,
    )
    tl.debug_barrier()
    tl.store(merge_lens_ptr + row, tl.minimum(tl.load(cnt_ptr + row), CAP))


@triton.jit
def _remap(
    cand_idx_ptr,
    merged_ptr,
    out_ptr,
    CAP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    m = j < TOPK
    pos = tl.load(merged_ptr + row * TOPK + j, mask=m, other=-1)
    live = m & (pos >= 0)
    idx = tl.load(cand_idx_ptr + row * CAP + pos, mask=live, other=-1)
    tl.store(out_ptr + row * TOPK + j, tl.where(live, idx, -1), mask=m)


def split_factor(rows, vocab, sms):
    split = 1
    while rows * split * 2 <= 4 * sms and vocab // (split * 2) >= MIN_CHUNK:
        split *= 2
    return split


def main():
    gen = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    import vllm._custom_ops  # noqa: F401

    global _key
    _key = gen._convert_to_trt_uint16_hi11
    globals()["_key"] = _key

    dev = "cuda"
    sms = ov._sm_count()
    names = ("prepare", "select", "fixup", "merge", "remap")
    print(f"device us from the profiler; {sms} SMs\n")
    print(
        f"  {'rows':>6} {'vocab':>7} {'k':>5} {'n/k':>5} {'split':>5} "
        f"{'vllm':>9} {'shipped':>9} {'sampled':>9} {'ratio now':>10} "
        f"{'ratio new':>10} {'cands':>7} {'ok':>5}"
    )
    rowsum = []
    for rows, vocab, top_k, stride0, stride1 in SHAPES:
        torch.manual_seed(42)
        buf = torch.randn(
            (rows - 1) * stride0 + (vocab - 1) * stride1 + 1,
            device=dev,
            dtype=torch.float32,
        )
        logits = torch.as_strided(buf, (rows, vocab), (stride0, stride1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)

        cap = max(BLOCK, triton.next_power_of_2(top_k * CAP_FACTOR))
        split = split_factor(rows, vocab, sms)
        chunk = triton.cdiv(vocab, split)
        stride = max(1, vocab // (BLOCK * SAMPLE_TILES))

        hist = torch.empty((rows, NB), dtype=torch.int32, device=dev)
        thr = torch.empty((rows,), dtype=torch.int32, device=dev)
        cnt = torch.empty((rows,), dtype=torch.int32, device=dev)
        cand_idx = torch.empty((rows, cap), dtype=torch.int32, device=dev)
        cand_val = torch.empty((rows, cap), dtype=torch.float32, device=dev)
        merged = torch.empty((rows, top_k), dtype=torch.int32, device=dev)
        mlens = torch.empty((rows,), dtype=torch.int32, device=dev)
        zeros = torch.zeros((rows,), dtype=torch.int32, device=dev)
        block = gen.NUM_THREADS_PER_BLOCK
        scratch = (
            torch.empty((rows, gen.NUM_BINS), dtype=torch.int32, device=dev),
            torch.empty((rows, gen.NUM_FILNAL_ITEMS), dtype=torch.float32, device=dev),
            torch.empty((rows,), dtype=torch.int32, device=dev),
            torch.empty((rows,), dtype=torch.int32, device=dev),
            torch.empty((rows,), dtype=torch.int32, device=dev),
            torch.empty((rows,), dtype=torch.int32, device=dev),
        )

        L = ov._Launch
        lprep = L(
            _prepare,
            (rows,),
            {
                "TOPK": top_k,
                "SAFETY": SAFETY,
                "NB": NB,
                "STRIDE": stride,
                "BLOCK": BLOCK,
            },
            WARPS,
        )
        lsel = L(
            _select,
            (rows * split,),
            {"CHUNK": chunk, "SPLIT": split, "CAP": cap, "BLOCK": BLOCK},
            WARPS,
        )
        lfix = L(
            _fixup,
            (rows,),
            {"TOPK": top_k, "NB": NB, "CAP": cap, "BLOCK": BLOCK},
            WARPS,
        )
        lmerge = L(
            gen.non_tle_top_k_per_row_prefill,
            (rows,),
            {"TOPK": top_k, "BLOCK_SIZE": block, "ROW_OFFSET": 0},
            gen._num_warps(block),
        )
        lremap = L(
            _remap,
            (rows,),
            {"CAP": cap, "TOPK": top_k, "BLOCK": triton.next_power_of_2(top_k)},
            4,
        )

        def sampled():
            lprep(logits, starts, ends, hist, thr, cnt, stride0)
            lsel(logits, starts, ends, thr, cnt, cand_idx, cand_val, stride0)
            lfix(logits, starts, ends, hist, cnt, mlens, cand_idx, cand_val, stride0)
            lmerge(cand_val, merged, zeros, mlens, cap, 1, cap, *scratch)
            lremap(cand_idx, merged, idx)

        sampled()
        torch.cuda.synchronize()
        hi = int(cnt.max())
        want = torch.topk(logits.float(), top_k, dim=1).values.sort(dim=1).values
        got = logits.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
        ok = torch.allclose(got, want) and bool((idx >= 0).all())

        t_vllm = device_us(
            lambda: torch.ops._C.top_k_per_row_prefill(
                logits, starts, ends, idx, rows, stride0, stride1, top_k
            )
        )
        t_ship = device_us(
            lambda: flaggems_vllm.top_k_per_row_prefill(
                logits, starts, ends, idx, rows, stride0, stride1, top_k
            )
        )
        t_new = device_us(sampled)
        rowsum.append(
            (
                rows,
                [
                    device_us(f)
                    for f in (
                        lambda: lprep(logits, starts, ends, hist, thr, cnt, stride0),
                        lambda: lsel(
                            logits, starts, ends, thr, cnt, cand_idx, cand_val, stride0
                        ),
                        lambda: lfix(
                            logits,
                            starts,
                            ends,
                            hist,
                            cnt,
                            mlens,
                            cand_idx,
                            cand_val,
                            stride0,
                        ),
                        lambda: lmerge(
                            cand_val, merged, zeros, mlens, cap, 1, cap, *scratch
                        ),
                        lambda: lremap(cand_idx, merged, idx),
                    )
                ],
            )
        )
        print(
            f"  {rows:>6} {vocab:>7} {top_k:>5} {vocab // top_k:>5} {split:>5} "
            f"{t_vllm:>9.1f} {t_ship:>9.1f} {t_new:>9.1f} "
            f"{t_vllm / t_ship:>10.3f} {t_vllm / t_new:>10.3f} {hi:>7} "
            f"{'OK' if ok else 'WRONG':>5}"
        )
    print("\n  per-stage device us\n")
    print("  " + f"{'rows':>6}" + "".join(f"{n:>9}" for n in names) + f"{'sum':>9}")
    for rows, ts in rowsum:
        print(
            "  " + f"{rows:>6}" + "".join(f"{t:>9.1f}" for t in ts) + f"{sum(ts):>9.1f}"
        )


if __name__ == "__main__":
    sys.exit(main())
