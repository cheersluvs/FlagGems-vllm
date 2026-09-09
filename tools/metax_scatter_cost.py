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
    """A 2048-bin zero, fill and scan -- the cost I wrongly blamed earlier."""
    row = tl.program_id(0)
    bins = tl.arange(0, NB)
    h = tl.zeros([NB], tl.int32)
    for c in tl.static_range((K + BLOCK - 1) // BLOCK):
        pos = c * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(src_ptr + pos, mask=pos < K, other=0)
        h += tl.histogram(v % NB, NB)
    tl.store(out_ptr + row * NB + bins, tl.cumsum(h, axis=0))


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
    print("prefill's k term is 17.1 ns per unit of k. Whichever line below has")
    print("that slope is the mechanism.\n")
    print(f"  {'K':>6} {'contig':>9} {'scatter':>9} {'cumsum':>9} {'2048-hist':>10}")

    prev = {}
    for K in (128, 256, 512, 1024):
        out = torch.zeros(ROWS * max(K, 2048), dtype=torch.int32, device=DEV)
        perm = torch.randperm(K, device=DEV).to(torch.int32)
        src = torch.randint(0, 4, (K,), dtype=torch.int32, device=DEV)
        t1 = timed(lambda: k_contig[(ROWS,)](out, K=K, BLOCK=BLOCK))
        t2 = timed(lambda: k_scatter_perm[(ROWS,)](out, perm, K=K, BLOCK=BLOCK))
        t3 = timed(lambda: k_scatter_cumsum[(ROWS,)](out, src, K=K, BLOCK=BLOCK))
        t4 = timed(lambda: k_hist[(ROWS,)](out, src, NB=2048, K=K, BLOCK=BLOCK))
        prev[K] = (t1, t2, t3, t4)
        print(f"  {K:>6} {t1:>9.3f} {t2:>9.3f} {t3:>9.3f} {t4:>10.3f}")

    print(f"\n  {'slope ns/k':>12}", end="")
    for i, name in enumerate(("contig", "scatter", "cumsum", "hist")):
        s = (prev[1024][i] - prev[128][i]) / (1024 - 128) * 1000
        print(f" {name}={s:.1f}", end="")
    print("\n\n  prefill wants 17.1 ns/k. A microbenchmark that reaches it names")
    print("  the mechanism; one that stays far below says the term is not a")
    print("  store at all and the ablation has to run on the real kernel.")


if __name__ == "__main__":
    sys.exit(main())
