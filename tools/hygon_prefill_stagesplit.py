"""More waves, without copying the radix into each of them.

The counters say the sparse prefill shape is not atomic-bound at all: 64
workgroups of 8 waves is 512 waves against this card's 3200 wave slots -- 16%
occupancy -- with arch_vgpr 44, so registers are not the limit and there is
simply not enough work in flight. Its waves wait 3.3x longer each than the
dense shape's.

The row split I measured earlier went the wrong way (249 -> 317 us) because
each chunk ran the WHOLE radix: its own 2048-bin histogram, its own threshold
scan, its own selection, and then a merge of per-chunk top-k. That multiplies
the per-program cost by the split.

Split by STAGE instead. One histogram and one threshold per row, shared:

    hist     rows x SPLIT programs, each only reading its chunk and adding
             into the ROW's histogram -- so the atomics are cross-CTA and
             scoped to the device. Nothing per-program to duplicate.
    thresh   one program per row: scan 2048 bins for the bin holding rank
             top_k, record how many are strictly better, and zero the
             histogram for the next call so no separate clear is needed
    select   rows x SPLIT programs again. Because thresh knows nlt (the count
             strictly better, necessarily < top_k), the better elements can
             take slots [0, nlt) and the threshold bin's take [nlt, CAP)
             through two counters -- disjoint by construction, no ordering
             needed between programs
    tail     the decode override's kernel, unchanged, for the exact top-k

Against the shipped operator on the two shapes that are starved (64 rows, and
4 rows) with a many-row shape as a control, over split and geometry. Reported
in device time and in the event-bracketed time do_bench uses, because this
trades one launch for four.

    tools/vendor_probe.sh tools/hygon_prefill_stagesplit.py hygon_prefill_stagesplit
"""

import sys
from importlib import import_module

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

import flaggems_vllm

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
_key11 = _generic._convert_to_trt_uint16_hi11

# (num_rows, vocab, top_k, stride0)
SHAPES = [
    (64, 129280, 1024, 129280),
    (4, 16385, 512, 16648),
    (4, 8193, 512, 8456),
    (16383, 4095, 512, 4352),
]
SPLITS = (1, 2, 4, 8, 16, 32)
GEOMS = ((256, 4), (512, 8))
NB = 2048
VEC = 4
RADIX = 256
MIN_CHUNK = 2048


def wall_us(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000


def device_us(fn, iters=20, warmup=5):
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
def _hist_split(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    stride0,
    CHUNK: tl.constexpr,
    SPLIT: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
):
    """SPLIT programs share one row's histogram, so the atomic is device
    scoped. Vectorised like the operator's own loop: each lane takes VEC
    consecutive elements."""
    pid = tl.program_id(0)
    row = pid // SPLIT
    chunk = pid % SPLIT
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    start = s + chunk * CHUNK
    end = tl.minimum(start + CHUNK, e)
    base = hist_ptr + row * NB
    lane = tl.arange(0, BLOCK)[:, None] * VEC + tl.arange(0, VEC)[None, :]
    for t in tl.range(0, tl.cdiv(CHUNK, BLOCK * VEC)):
        i = start + t * BLOCK * VEC + lane
        m = i < end
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        tl.atomic_add(
            base + _key11(x),
            tl.full([BLOCK, VEC], 1, tl.int32),
            mask=m,
            sem="relaxed",
            scope="gpu",
        )


@triton.jit
def _thresh(
    hist_ptr,
    thr_ptr,
    cnt_a_ptr,
    cnt_b_ptr,
    TOPK: tl.constexpr,
    NB: tl.constexpr,
):
    """One program per row: the bin holding rank TOPK and how many are
    strictly better, then zero the histogram for the next call."""
    row = tl.program_id(0)
    bins = tl.arange(0, NB)
    base = hist_ptr + row * NB
    counts = tl.load(base + bins)
    pre = tl.cumsum(counts, axis=0) - counts
    hit = (pre < TOPK) & (pre + counts >= TOPK)
    thr = tl.min(tl.where(hit, bins, NB - 1), axis=0).to(tl.int32)
    nlt = tl.max(tl.where(bins == thr, pre, 0), axis=0).to(tl.int32)
    tl.store(thr_ptr + row, thr)
    tl.store(cnt_a_ptr + row, 0)
    tl.store(cnt_b_ptr + row, nlt)
    tl.store(base + bins, tl.zeros([NB], tl.int32))


@triton.jit
def _select_split(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    thr_ptr,
    cnt_a_ptr,
    cnt_b_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    stride0,
    CHUNK: tl.constexpr,
    SPLIT: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
):
    """Strictly-better elements take slots [0, nlt) through counter A and the
    threshold bin's take [nlt, CAP) through counter B, which thresh seeded at
    nlt. The two regions cannot collide, so the programs need no ordering."""
    pid = tl.program_id(0)
    row = pid // SPLIT
    chunk = pid % SPLIT
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    thr = tl.load(thr_ptr + row)
    start = s + chunk * CHUNK
    end = tl.minimum(start + CHUNK, e)
    lane = tl.arange(0, BLOCK)[:, None] * VEC + tl.arange(0, VEC)[None, :]
    a_ptrs = cnt_a_ptr + row + tl.zeros([BLOCK, VEC], tl.int32)
    b_ptrs = cnt_b_ptr + row + tl.zeros([BLOCK, VEC], tl.int32)
    ones = tl.full([BLOCK, VEC], 1, tl.int32)
    for t in tl.range(0, tl.cdiv(CHUNK, BLOCK * VEC)):
        i = start + t * BLOCK * VEC + lane
        m = i < end
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        k = _key11(x)
        better = m & (k < thr)
        pos = tl.atomic_add(a_ptrs, ones, mask=better, sem="relaxed", scope="gpu")
        keep = better & (pos >= 0) & (pos < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos, (i - s).to(tl.int32), mask=keep)
        tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)
        eq = m & (k == thr)
        pos2 = tl.atomic_add(b_ptrs, ones, mask=eq, sem="relaxed", scope="gpu")
        keep2 = eq & (pos2 >= 0) & (pos2 < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos2, (i - s).to(tl.int32), mask=keep2)
        tl.store(cand_val_ptr + row * CAP + pos2, x, mask=keep2)


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    dev = "cuda"
    sms = ov._sm_count()
    print(f"{sms} SMs; device us, and the event-bracketed us do_bench uses\n")
    for rows, vocab, top_k, stride0 in SHAPES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)
        want = torch.topk(logits, top_k, dim=1).values.sort(dim=1).values
        cap = triton.next_power_of_2(top_k * 8)
        hist = torch.zeros((rows, NB), dtype=torch.int32, device=dev)
        thr = torch.empty((rows,), dtype=torch.int32, device=dev)
        cnt_a = torch.empty((rows,), dtype=torch.int32, device=dev)
        cnt_b = torch.empty((rows,), dtype=torch.int32, device=dev)
        cand_idx = torch.empty((rows, cap), dtype=torch.int32, device=dev)
        cand_val = torch.empty((rows, cap), dtype=torch.float32, device=dev)
        counts = torch.empty((rows, RADIX), dtype=torch.int32, device=dev)
        slot = torch.empty((rows,), dtype=torch.int32, device=dev)

        d_ship = device_us(
            lambda: flaggems_vllm.top_k_per_row_prefill(
                logits, starts, ends, idx, rows, stride0, 1, top_k
            )
        )
        w_ship = wall_us(
            lambda: flaggems_vllm.top_k_per_row_prefill(
                logits, starts, ends, idx, rows, stride0, 1, top_k
            )
        )
        print(
            f"  {rows} x {vocab}, top_k {top_k}: shipped {d_ship:.1f} us device, "
            f"{w_ship:.1f} us wall; {rows} workgroups today",
            flush=True,
        )
        for split in SPLITS:
            chunk = triton.cdiv(vocab, split)
            if split > 1 and chunk < MIN_CHUNK:
                continue
            for block, warps in GEOMS:
                tag = f"split {split:>2}, {block}x{warps}"
                lh = ov._Launch(
                    _hist_split,
                    (rows * split,),
                    {
                        "CHUNK": chunk,
                        "SPLIT": split,
                        "NB": NB,
                        "BLOCK": block,
                        "VEC": VEC,
                    },
                    warps,
                )
                lt = ov._Launch(_thresh, (rows,), {"TOPK": top_k, "NB": NB}, 8)
                ls = ov._Launch(
                    _select_split,
                    (rows * split,),
                    {
                        "CHUNK": chunk,
                        "SPLIT": split,
                        "CAP": cap,
                        "BLOCK": block,
                        "VEC": VEC,
                    },
                    warps,
                )
                lx = ov._Launch(
                    ov._tail,
                    (rows,),
                    {
                        "TOPK": top_k,
                        "NB": NB,
                        "CAP": cap,
                        "RADIX": RADIX,
                        "BLOCK": 512,
                    },
                    8,
                )

                def run():
                    lh(logits, starts, ends, hist, stride0)
                    lt(hist, thr, cnt_a, cnt_b)
                    ls(
                        logits,
                        starts,
                        ends,
                        thr,
                        cnt_a,
                        cnt_b,
                        cand_idx,
                        cand_val,
                        stride0,
                    )
                    lx(
                        logits,
                        ends,
                        hist,
                        cnt_b,
                        cand_idx,
                        cand_val,
                        idx,
                        counts,
                        slot,
                        stride0,
                    )

                idx.fill_(-9)
                try:
                    run()
                    torch.cuda.synchronize()
                except Exception as exc:  # noqa: BLE001 - keep sweeping
                    print(f"    {tag:<22} FAILED {exc!r:.60}", flush=True)
                    continue
                got = (
                    logits.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
                )
                ok = torch.allclose(got, want) and bool((idx >= 0).all())
                d = device_us(run)
                w = wall_us(run)
                print(
                    f"    {tag:<22} {d:>8.1f} us device ({d_ship / d:>5.2f}x), "
                    f"{w:>8.1f} us wall ({w_ship / w:>5.2f}x), "
                    f"{rows * split:>6} workgroups, "
                    f"{'OK' if ok else 'WRONG'}",
                    flush=True,
                )
        print(flush=True)


if __name__ == "__main__":
    sys.exit(main())
