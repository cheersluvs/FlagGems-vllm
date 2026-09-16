"""One global read: keep the row in registers across both passes.

The generic radix reads every element twice -- once to histogram it, once to
compare it against the threshold the histogram produced. The second read is
268 MB at (16383,4095), and on this card a read runs at ~1.4 TB/s, so it is
worth roughly 190 us of the shipped 1348. That is the last idea left for
prefill; everything else decode taught us has been tried and refuted.

The trick is to avoid rewriting the operator. Triton unrolls static_range, so
tiles loaded inside a loop cannot be named again after it -- there is no way
to "keep the tiles from pass one". But a row that FITS can be loaded as a
single [BLOCK, VEC] tile, and then the histogram, the threshold scan and the
selection all run against that one register-resident tile:

    resident   one masked load of the whole row; zero the 2048 bins;
               histogram from registers; scan for the bin holding rank top_k;
               emit every element at or above it into a candidate buffer,
               strictly-better bins BEFORE the threshold bin so a buffer that
               overflows can only drop elements sharing an 11-bit key with the
               k-th. Slots come from a prefix sum over the whole tile, so ONE
               atomic per program -- the mechanism the shipped override
               already uses for these shapes.
    tail       the decode override's kernel, unchanged, for the exact top-k

Only the dense shapes can fit: rows of 1025-5115 elements need tiles of
2048-8192, about 32 floats per thread. The sparse shapes (8193 and up) cannot,
and are not attempted.

The risk is not capacity, it is occupancy: 32 KB of registers per program cuts
how many the SM can hold, and these shapes have thousands of programs. So
(BLOCK, VEC, warps) is swept rather than chosen.

Round 1 took a VM fault on the first configuration -- (16383,4095) at
256 x 16 x 2 -- and a fault kills the process, so the report came back empty
and the failing step had to be inferred from the AQL dump. Round 2 therefore
does three things:

  * every configuration and STAGE prints before it runs, flushed, so the last
    line of the report names what died
  * STAGE picks how far the kernel goes: 0 loads and histograms and scans but
    emits nothing, 1 emits through a per-element atomic (the mechanism the
    shipped decode select uses), 2 emits through the prefix-sum tile
    allocator. Run in that order, the first failing stage is the answer.
  * pos is guarded with >= 0 as well as < CAP. A masked atomic's return is
    undefined on inactive lanes, and a negative slot passes a bare `< CAP`
    and stores wherever it points -- which is what "beyond the largest legal
    address" describes.

The prefix-sum allocator is the prime suspect: the AQL dump shows
group_segment_size 8192, and a 4096-element cumsum needs 16 KB. The shipped
override only ever runs it on [512, 4] tiles.

Device time from the profiler, which is what the benchmark's kernel mode
reports. vLLM is NOT timed here: the profiler double-counts it on this card.

    tools/vendor_probe.sh tools/hygon_prefill_resident.py hygon_prefill_resident
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

# the dense shapes: (num_rows, vocab, top_k, stride0)
SHAPES = [
    (16383, 4095, 512, 4352),
    (12961, 4100, 512, 4360),
    (16380, 5115, 512, 5376),
    (4100, 1025, 512, 1288),
]
NB = 2048
RADIX = 256
WARPS = (2, 4, 8)
BLOCKS = (256, 512, 1024)


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
def _alloc_slots(ptrs, take):
    """One atomic per TILE instead of one per taken lane: the prefix sum over
    the take-mask hands out the same distinct slots."""
    ti = take.to(tl.int32)
    flat = tl.reshape(ti, (ti.numel,))
    total = tl.sum(flat, axis=0)
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
def _resident(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    cnt_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    stride0,
    thr_ptr,
    TOPK: tl.constexpr,
    NB: tl.constexpr,
    CAP: tl.constexpr,
    STAGE: tl.constexpr,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
):
    row = tl.program_id(0)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    n = e - s
    off = tl.arange(0, BLOCK)[:, None] * VEC + tl.arange(0, VEC)[None, :]
    m = off < n
    x = tl.load(logits_ptr + row * stride0 + s + off, mask=m, other=0.0)

    bins = tl.arange(0, NB)
    hbase = hist_ptr + row * NB
    tl.store(hbase + bins, tl.zeros([NB], tl.int32))
    tl.store(cnt_ptr + row, 0)
    tl.debug_barrier()
    k = _key11(x)
    tl.atomic_add(
        hbase + k,
        tl.full([BLOCK, VEC], 1, tl.int32),
        mask=m,
        sem="relaxed",
        scope="cta",
    )
    tl.debug_barrier()
    counts = tl.load(hbase + bins)
    pre = tl.cumsum(counts, axis=0) - counts
    hit = (pre < TOPK) & (pre + counts >= TOPK)
    thr = tl.min(tl.where(hit, bins, NB - 1), axis=0).to(tl.int32)
    tl.store(thr_ptr + row, thr)
    if STAGE == 0:
        return

    cnt_ptrs = cnt_ptr + row + tl.zeros([BLOCK, VEC], tl.int32)
    for equal in tl.static_range(2):
        if equal == 0:
            take = m & (k < thr)
        else:
            take = m & (k == thr)
        if STAGE == 1:
            pos = tl.atomic_add(
                cnt_ptrs,
                tl.full([BLOCK, VEC], 1, tl.int32),
                mask=take,
                sem="relaxed",
                scope="cta",
            )
        else:
            pos = _alloc_slots(cnt_ptrs, take)
        keep = take & (pos >= 0) & (pos < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos, off.to(tl.int32), mask=keep)
        tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)
        tl.debug_barrier()


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    dev = "cuda"
    print("device us; the whole operator, against the shipped override\n")
    for rows, vocab, top_k, stride0 in SHAPES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)
        want = torch.topk(logits, top_k, dim=1).values.sort(dim=1).values
        cap = triton.next_power_of_2(top_k * 8)
        hist = torch.empty((rows, NB), dtype=torch.int32, device=dev)
        cnt = torch.empty((rows,), dtype=torch.int32, device=dev)
        cand_idx = torch.empty((rows, cap), dtype=torch.int32, device=dev)
        cand_val = torch.empty((rows, cap), dtype=torch.float32, device=dev)
        counts = torch.empty((rows, RADIX), dtype=torch.int32, device=dev)
        slot = torch.empty((rows,), dtype=torch.int32, device=dev)

        thr = torch.empty((rows,), dtype=torch.int32, device=dev)
        t_ship = device_us(
            lambda: flaggems_vllm.top_k_per_row_prefill(
                logits, starts, ends, idx, rows, stride0, 1, top_k
            )
        )
        need = triton.next_power_of_2(vocab)
        print(
            f"  {rows} x {vocab}, top_k {top_k}: shipped {t_ship:.1f} us; "
            f"row needs a {need}-element tile, buffer {cap}",
            flush=True,
        )
        names = {0: "hist only", 1: "atomic", 2: "prefix sum"}
        for block in BLOCKS:
            vec = need // block
            if vec < 1 or block * vec < vocab:
                continue
            for warps in WARPS:
                for stage in (0, 1, 2):
                    tag = f"{block} x {vec} x {warps}, {names[stage]}"
                    print(f"    {tag:<32} running...", end="", flush=True)
                    lr = ov._Launch(
                        _resident,
                        (rows,),
                        {
                            "TOPK": top_k,
                            "NB": NB,
                            "CAP": cap,
                            "STAGE": stage,
                            "BLOCK": block,
                            "VEC": vec,
                        },
                        warps,
                    )
                    lt = ov._Launch(
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

                    def resident(lr=lr):
                        lr(
                            logits,
                            starts,
                            ends,
                            hist,
                            cnt,
                            cand_idx,
                            cand_val,
                            stride0,
                            thr,
                        )

                    def run(lr=lr):
                        resident(lr)
                        lt(
                            logits,
                            ends,
                            hist,
                            cnt,
                            cand_idx,
                            cand_val,
                            idx,
                            counts,
                            slot,
                            stride0,
                        )

                    idx.fill_(-9)
                    try:
                        resident()
                        torch.cuda.synchronize()
                    except Exception as exc:  # noqa: BLE001 - keep sweeping
                        print(f" FAILED {exc!r:.70}", flush=True)
                        continue
                    if stage == 0:
                        t_r = device_us(resident)
                        print(f" {t_r:>8.1f} us", flush=True)
                        continue
                    run()
                    torch.cuda.synchronize()
                    got = (
                        logits.gather(1, idx.long().clamp(0, vocab - 1))
                        .sort(dim=1)
                        .values
                    )
                    ok = torch.allclose(got, want) and bool((idx >= 0).all())
                    hi = int(cnt.max())
                    t_r = device_us(resident)
                    t = device_us(run)
                    print(
                        f" {t_r:>8.1f} + tail = {t:>8.1f} us, "
                        f"{t_ship / t:>5.2f}x shipped, {hi:>5} cands, "
                        f"{'OK' if ok else 'WRONG'}",
                        flush=True,
                    )
        print(flush=True)


if __name__ == "__main__":
    sys.exit(main())
