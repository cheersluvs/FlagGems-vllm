"""Is a masked global atomic really 60x a maskless one on Hygon?

tools/metax_smem_atomic_cost.py on BW1000 measured, per program, 4096 atomics
to ONE global address:

    all lanes taken (no mask effect)   0.21 us
    1/4 of lanes taken, masked        12.56 us

Fewer atomics costing 60x more points at the mask itself -- a maskless atomic
looks warp-aggregated, a masked one falls back to per-lane. The operator's
`_process_bins` uses the masked form for every selected element, and prefill
loses on all seven shapes here, so this is worth pinning down before anything
is rewritten.

The candidate rewrite, validated for semantics on MetaX: drop the mask and add
`take.to(tl.int32)`. Untaken lanes add 0 and their returned value is unused;
taken lanes still get a unique slot each. Measured here at the operator's own
geometry (BLOCK=512 on 8 warps of 64), on GLOBAL memory, which is where this
card's non-TLE path keeps the counter.

    tools/vendor_probe.sh tools/hygon_masked_atomic_cost.py hygon_masked_atomic
"""

import sys

import torch
import triton
import triton.language as tl

SMS = int(getattr(torch.cuda.get_device_properties(0), "multi_processor_count", 80))
ROWS = 40 * SMS
BLOCK = 512
WARPS = 8
CHUNKS = 8  # 4096 items per program


@triton.jit
def k_atomic(
    scr_ptr,
    src_ptr,
    out_ptr,
    DENS: tl.constexpr,
    MODE: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """MODE 0: masked, the operator's form. MODE 1: maskless, +take.
    MODE 2: no atomic at all, so the atomic's own share can be read off."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    zeros = tl.zeros([BLOCK], tl.int32)
    tl.store(scr_ptr + row, 0)
    tl.debug_barrier()
    p = scr_ptr + row + zeros
    acc = tl.zeros([BLOCK], tl.int32)
    for c in range(CHUNKS):
        v = tl.load(src_ptr + c * BLOCK + lane)
        take = (v % DENS) == 0
        if MODE == 0:
            r = tl.atomic_add(p, zeros + 1, mask=take, sem="relaxed", scope="cta")
        elif MODE == 1:
            r = tl.atomic_add(p, take.to(tl.int32), sem="relaxed", scope="cta")
        else:
            r = tl.where(take, v, 0)
        acc += tl.where(take, r, 0)
    tl.store(out_ptr + row, tl.sum(acc))


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
    return a.elapsed_time(b) / iters * 1000 / (ROWS / SMS)  # us per program


def main():
    dev = "cuda"
    torch.manual_seed(0)
    src = torch.randint(0, 1 << 20, (CHUNKS * BLOCK,), dtype=torch.int32, device=dev)
    scr = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    out = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    print(
        f"{ROWS} rows = 40 waves on {SMS} SMs | BLOCK={BLOCK} x {WARPS} warps | "
        f"{CHUNKS * BLOCK} atomics/program to one address | microseconds per program\n"
    )
    print(
        f"  {'taken':>6} {'masked':>9} {'maskless+take':>14} {'no atomic':>10} "
        f"{'masked-none':>12} {'maskless-none':>14} {'speedup':>8}  check"
    )
    for dens in (1, 2, 4, 8, 32):
        t = {}
        for mode in (0, 1, 2):
            out.zero_()
            t[mode] = timed(
                lambda m=mode: k_atomic[(ROWS,)](
                    scr,
                    src,
                    out,
                    DENS=dens,
                    MODE=m,
                    CHUNKS=CHUNKS,
                    BLOCK=BLOCK,
                    num_warps=WARPS,
                )
            )
            if mode == 0:
                first = out.clone()
            elif mode == 1:
                same = bool(torch.equal(out, first))
        n = int(((src.cpu() % dens) == 0).sum())
        want = n * (n - 1) // 2
        ok = bool((out.cpu() == want).all()) and same
        sp = (t[0] - t[2]) / (t[1] - t[2]) if t[1] > t[2] else float("inf")
        print(
            f"  {'1/' + str(dens):>6} {t[0]:>9.2f} {t[1]:>14.2f} {t[2]:>10.2f} "
            f"{t[0] - t[2]:>12.2f} {t[1] - t[2]:>14.2f} {sp:>8.1f}  "
            f"{'same sums' if ok else 'MISMATCH'}"
        )
    print("\n  'masked-none' and 'maskless-none' are the atomic's own share.")
    print("  A large speedup column means _process_bins should drop its mask on")
    print("  this card; 'same sums' is the semantics check (identical slot sums).")


if __name__ == "__main__":
    sys.exit(main())
