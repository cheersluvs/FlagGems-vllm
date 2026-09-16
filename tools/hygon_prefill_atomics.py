"""The two atomics prefill is paying for, and whether either can be made cheap.

tools/hygon_prefill_bw.py found the cliff: a pure read runs at 1.4 TB/s on
this card and the key is free, but one 2048-bin histogram atomic per element
drops it to ~320 GB/s, and the shipped operator's two passes come out at ~400
GB/s -- i.e. it is already running at the histogram's rate. Prefill is not
bandwidth-bound and not pass-count-bound; it is bound by scattered atomics.

Two measured facts about this card point at the fix
([[hygon-bw1000-topk-per-row]]): an atomic where every lane hits ONE address
costs 0.05 ns, and a 2048-bin scatter costs 1.6 ns -- 32x. The hardware
coalesces same-address atomics and serialises scattered ones. So:

  1. does the histogram get cheaper with FEWER bins, where more lanes share an
     address? 2048 vs 256 vs 64.
  2. the candidate append is one atomic per SELECTED element on a single
     counter. The shipped prefill override already replaced exactly that
     pattern with a prefix sum over the tile's take-mask plus ONE atomic
     (_alloc_slots, worth 1.4-1.86x on the dense shapes) -- and I wrote the
     sampled select with per-element atomics anyway. Measure both, at the
     density the exact algorithm's select pass sees (top_k/n) and at the one
     the sampled algorithm sees (6*top_k/n).

The logits are standard normal, so a density is turned into a threshold with
the normal inverse CDF rather than by sorting. The compare is on the float
rather than the key: the key costs nothing (1015 vs 1078 GB/s) and leaving it
out keeps this about the atomics.

    tools/vendor_probe.sh tools/hygon_prefill_atomics.py hygon_prefill_atomics
"""

import math
import sys
from importlib import import_module

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
_key = _generic._convert_to_trt_uint16_hi11

# (num_rows, vocab, top_k, stride0, split) -- one shape per regime, at the
# split that read best in the bandwidth sweep.
SHAPES = [
    (64, 129280, 1024, 129280, 2),
    (16383, 4095, 512, 4352, 1),
    (12961, 4100, 512, 4360, 1),
    (4100, 1025, 512, 1288, 1),
]
GEOMS = [(256, 2), (256, 4), (512, 4), (512, 8), (1024, 8), (1024, 16)]


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


def normal_threshold(density):
    """t with P(x > t) = density, for standard normal x."""
    return math.sqrt(2.0) * torch.erfinv(torch.tensor(1.0 - 2.0 * density)).item()


@triton.jit
def _alloc_slots(ptrs, take, SCOPE: tl.constexpr):
    """Same contract as tl.atomic_add(ptrs, 1, mask=take) on one counter, but
    one atomic per TILE instead of one per taken lane. Ported from the shipped
    prefill override; SCOPE must be "gpu" when programs share a row."""
    ti = take.to(tl.int32)
    total = tl.sum(ti, axis=0)
    first = tl.arange(0, ti.numel) == 0
    prev = tl.atomic_add(ptrs, ti * 0 + total, mask=first, sem="relaxed", scope=SCOPE)
    start = tl.sum(tl.where(first, prev, 0), axis=0)
    return start + tl.cumsum(ti, axis=0) - ti


@triton.jit
def k_pass(
    logits_ptr,
    ends_ptr,
    hist_ptr,
    cnt_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    out_ptr,
    stride0,
    thr,
    LEVEL: tl.constexpr,
    BINS: tl.constexpr,
    SHIFT: tl.constexpr,
    CHUNK: tl.constexpr,
    SPLIT: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """LEVEL 0 read, 1 histogram into BINS bins, 2 append by per-element
    atomic, 3 append by prefix-sum slot allocation."""
    pid = tl.program_id(0)
    row = pid // SPLIT
    chunk = pid % SPLIT
    lane = tl.arange(0, BLOCK)
    e = tl.load(ends_ptr + row)
    start = chunk * CHUNK
    end = tl.minimum(start + CHUNK, e)
    base = hist_ptr + row * BINS
    cnt_ptrs = cnt_ptr + row + tl.zeros([BLOCK], tl.int32)
    acc = tl.zeros([BLOCK], tl.float32)
    for t in tl.range(0, tl.cdiv(CHUNK, BLOCK)):
        i = start + t * BLOCK + lane
        m = i < end
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        if LEVEL == 0:
            acc += x
        elif LEVEL == 1:
            tl.atomic_add(
                base + (_key(x) >> SHIFT),
                tl.full([BLOCK], 1, tl.int32),
                mask=m,
                sem="relaxed",
                scope="cta",
            )
        else:
            take = m & (x > thr)
            if LEVEL == 2:
                pos = tl.atomic_add(
                    cnt_ptrs,
                    tl.full([BLOCK], 1, tl.int32),
                    mask=take,
                    sem="relaxed",
                    scope="gpu",
                )
            else:
                pos = _alloc_slots(cnt_ptrs, take, "gpu")
            keep = take & (pos < CAP)
            tl.store(cand_idx_ptr + row * CAP + pos, i.to(tl.int32), mask=keep)
            tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)
    if LEVEL == 0:
        tl.store(out_ptr + pid, tl.sum(acc, axis=0))


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    dev = "cuda"
    print(f"{ov._sm_count()} SMs; GB/s over the bytes read, one pass\n")
    for rows, vocab, top_k, stride0, split in SHAPES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        d_exact = top_k / vocab
        d_sampled = min(0.5, 6.0 * top_k / vocab)
        cap = min(65536, triton.next_power_of_2(int(d_sampled * vocab) + 64))
        hist = torch.zeros((rows, 2048), dtype=torch.int32, device=dev)
        cnt = torch.zeros((rows,), dtype=torch.int32, device=dev)
        cand_idx = torch.empty((rows, cap), dtype=torch.int32, device=dev)
        cand_val = torch.empty((rows, cap), dtype=torch.float32, device=dev)
        out = torch.empty((rows * split,), dtype=torch.float32, device=dev)
        chunk = triton.cdiv(vocab, split)
        gbytes = rows * vocab * 4 / 1e9

        cols = [("read", 0, 2048, 0.0)]
        cols += [(f"h{b}", 1, b, 0.0) for b in (64, 256, 2048)]
        for tag, d in (("x", d_exact), ("s", d_sampled)):
            cols.append((f"atom{tag}", 2, 2048, d))
            cols.append((f"scan{tag}", 3, 2048, d))
        print(
            f"  {rows} x {vocab}, top_k {top_k}, split {split}: "
            f"{gbytes * 1e3:.1f} MB; select density {d_exact:.3f} (exact) "
            f"{d_sampled:.3f} (sampled), buffer {cap}"
        )
        print(f"    {'B x w':>9}" + "".join(f"{c[0]:>9}" for c in cols))
        best = {}
        for block, warps in GEOMS:
            if block > chunk:
                continue
            cells = []
            for name, level, bins, d in cols:
                launch = ov._Launch(
                    k_pass,
                    (rows * split,),
                    {
                        "LEVEL": level,
                        "BINS": bins,
                        "SHIFT": 11 - int(math.log2(bins)),
                        "CHUNK": chunk,
                        "SPLIT": split,
                        "CAP": cap,
                        "BLOCK": block,
                    },
                    warps,
                )
                thr = normal_threshold(d) if d else 0.0

                def run(launch=launch, thr=thr):
                    cnt.zero_()
                    launch(
                        logits,
                        ends,
                        hist,
                        cnt,
                        cand_idx,
                        cand_val,
                        out,
                        stride0,
                        thr,
                    )

                t = max(device_us(run) - device_us(lambda: cnt.zero_()), 1e-6)
                gbs = gbytes / t * 1e6
                cells.append(gbs)
                if gbs > best.get(name, 0):
                    best[name] = gbs
            print(
                f"    {f'{block} x {warps}':>9}" + "".join(f"{c:>9.0f}" for c in cells)
            )
        print("    best: " + ", ".join(f"{k} {v:.0f}" for k, v in best.items()) + "\n")


if __name__ == "__main__":
    sys.exit(main())
