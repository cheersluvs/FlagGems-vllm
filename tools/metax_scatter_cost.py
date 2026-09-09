"""What does a data-dependent scatter cost on this card, per item?

prefill's per-program cost carries a term of ~17 ns per unit of top_k that is
independent of vocabulary and of input distribution, and the O(n^2) rank sort
has been measured innocent. What is left in _process_bins is a pile of masked
stores to computed positions:

    tl.store(out_ptr + out_pos_eq, ..., mask=take_eq & (out_pos_eq < TOPK))

with out_pos_eq derived from a cumsum -- a scatter whose addresses are only
known at run time. If those serialise, 512 outputs at ~17 ns each is exactly
the 8.7 us the fit attributes to k.

Cheaper to test the mechanism directly than to ablate a 300-line kernel: build
the same store four ways and see which one costs 17 ns an item.

    python tools/metax_scatter_cost.py
"""

import sys

import torch
import triton
import triton.language as tl

import flaggems_vllm

DEV = flaggems_vllm.device
ROWS = 4160          # 40 waves on 104 SMs, same as the prefill sweep
BLOCK = 512


@triton.jit
def k_contig(out_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
    """Each program writes K items to consecutive addresses."""
    row = tl.program_id(0)
    for c in tl.static_range((K + BLOCK - 1) // BLOCK):
        pos = c * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + row * K + pos, pos, mask=pos < K)


@triton.jit
def k_scatter_perm(out_ptr, perm_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
    """Same K items, but each to an address read from memory."""
    row = tl.program_id(0)
    for c in tl.static_range((K + BLOCK - 1) // BLOCK):
        pos = c * BLOCK + tl.arange(0, BLOCK)
        m = pos < K
        dst = tl.load(perm_ptr + pos, mask=m, other=0)
        tl.store(out_ptr + row * K + dst, pos, mask=m)


@triton.jit
def k_scatter_cumsum(out_ptr, src_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
    """The shape _process_bins actually uses: positions from a cumsum, and the
    store masked by both a predicate and a bound."""
    row = tl.program_id(0)
    base = tl.zeros([], tl.int32)
    for c in tl.static_range((K + BLOCK - 1) // BLOCK):
        pos = c * BLOCK + tl.arange(0, BLOCK)
        m = pos < K
        take = m & (tl.load(src_ptr + pos, mask=m, other=0) > 0)
        out_pos = base + tl.cumsum(take.to(tl.int32), axis=0) - 1
        tl.store(out_ptr + row * K + out_pos, pos, mask=take & (out_pos < K))
        base += tl.sum(take.to(tl.int32), axis=0)


@triton.jit
def k_hist(out_ptr, src_ptr, NB: tl.constexpr, K: tl.constexpr,
           BLOCK: tl.constexpr):
    """Histogram then scan -- both, as the operator does it."""
    row = tl.program_id(0)
    h = tl.zeros([NB], tl.int32)
    for c in tl.static_range((K + BLOCK - 1) // BLOCK):
        pos = c * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(src_ptr + pos, mask=pos < K, other=0)
        h += tl.histogram(v % NB, NB)
    tl.store(out_ptr + row * NB + tl.arange(0, NB), tl.cumsum(h, axis=0))


@triton.jit
def k_hist_only(out_ptr, src_ptr, NB: tl.constexpr, K: tl.constexpr,
                BLOCK: tl.constexpr):
    """Histogram without the scan."""
    row = tl.program_id(0)
    h = tl.zeros([NB], tl.int32)
    for c in tl.static_range((K + BLOCK - 1) // BLOCK):
        pos = c * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(src_ptr + pos, mask=pos < K, other=0)
        h += tl.histogram(v % NB, NB)
    tl.store(out_ptr + row * NB + tl.arange(0, NB), h)


@triton.jit
def k_scan_only(out_ptr, NB: tl.constexpr):
    """The scan without the histogram."""
    row = tl.program_id(0)
    bins = tl.arange(0, NB)
    tl.store(out_ptr + row * NB + bins, tl.cumsum(bins.to(tl.int32), axis=0))


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000 / (ROWS / 104)   # us per program


def main():
    print(f"device {DEV} | {ROWS} rows = 40 waves | per-program microseconds\n")
    print("The scatter hypothesis is dead: 1.8 ns/k against the 17.1 the fit")
    print("wants. But a 2048-bin histogram measured 8.0 us flat, and prefill's")
    print("k increments are +2.33, +4.97, +8.01 us -- the last one being exactly")
    print("one histogram. So the k term is the STEP COUNT, and the question is")
    print("whether that 8 us is proportional to the bin count or fixed.\n")

    K = 512
    src = torch.randint(0, 4096, (K,), dtype=torch.int32, device=DEV)
    print(f"  {'bins':>6} {'hist+scan':>10} {'hist only':>10} {'scan only':>10}"
          f" {'us/1k bins':>11}")
    for NB in (64, 128, 256, 512, 1024, 2048, 4096):
        out = torch.zeros(ROWS * NB, dtype=torch.int32, device=DEV)
        t_both = timed(lambda: k_hist[(ROWS,)](out, src, NB=NB, K=K, BLOCK=BLOCK))
        t_hist = timed(lambda: k_hist_only[(ROWS,)](out, src, NB=NB, K=K, BLOCK=BLOCK))
        t_scan = timed(lambda: k_scan_only[(ROWS,)](out, NB=NB))
        print(f"  {NB:>6} {t_both:>10.3f} {t_hist:>10.3f} {t_scan:>10.3f}"
              f" {t_both / (NB / 1024):>11.3f}")

    print("\n  A flat us/1k-bins column means the cost is proportional to bins,")
    print("  so 2048 -> 256 buys 8x and more steps are affordable. A rising one")
    print("  means most of it is fixed per call, and only fewer STEPS help.")
    print()
    print(f"  {'K':>6} {'contig':>9} {'scatter':>9} {'cumsum':>9}   (stores, for the record)")
    for K2 in (128, 512, 1024):
        out = torch.zeros(ROWS * max(K2, 64), dtype=torch.int32, device=DEV)
        perm = torch.randperm(K2, device=DEV).to(torch.int32)
        s2 = torch.randint(0, 4, (K2,), dtype=torch.int32, device=DEV)
        print(f"  {K2:>6} "
              f"{timed(lambda: k_contig[(ROWS,)](out, K=K2, BLOCK=BLOCK)):>9.3f} "
              f"{timed(lambda: k_scatter_perm[(ROWS,)](out, perm, K=K2, BLOCK=BLOCK)):>9.3f} "
              f"{timed(lambda: k_scatter_cumsum[(ROWS,)](out, s2, K=K2, BLOCK=BLOCK)):>9.3f}")


if __name__ == "__main__":
    sys.exit(main())
