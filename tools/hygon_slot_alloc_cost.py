"""Three ways to allocate an output slot per selected element, priced.

prefill's k term is the whole story on the small-vocab shapes here: the fit
gives ~19 ns per unit of top_k at vocab 4096, i.e. 9.7 of that shape's 17.9 us
per program, and the ablation independently removed 7.55 us with the atomic.
The term is `tl.atomic_add(found_topk_values_ptrs, ones, mask=take)` in
_process_bins -- one atomic per SELECTED element, all to one address.

On MetaX the cumsum replacement measured 1.6-3x SLOWER, and the recorded
reason was three tl.debug_barrier() plus a load and a store of the base per
call. That reason does not apply to the form below: tl.cumsum is
program-wide, so every lane already sees the same total and the running base
can live in a REGISTER -- no shared counter, no barrier.

    atomic      today: one masked atomic per selected lane
    scan_reg    tl.cumsum over the mask, base carried in a register
    scan_atomic tl.cumsum plus ONE atomic per tile (needed only if the base
                must be visible to other programs)

Swept over how many lanes a tile actually selects, which is what decides the
crossover: 512 lanes, one program per row, 40 waves.

    tools/vendor_probe.sh tools/hygon_slot_alloc_cost.py hygon_slot_alloc
"""

import sys

import torch
import triton
import triton.language as tl

SMS = int(getattr(torch.cuda.get_device_properties(0), "multi_processor_count", 80))
ROWS = 40 * SMS
BLOCK = 512
WARPS = 8
TILES = 8  # 4096 elements per program, the vocab that loses worst here


@triton.jit
def k_slots(
    src_ptr,
    scr_ptr,
    out_ptr,
    DENS: tl.constexpr,
    MODE: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    THRESH: tl.constexpr,
):
    """MODE 0 atomic per selected lane, 1 cumsum + register carry,
    2 cumsum + one atomic per tile, 3 no allocation at all (the floor),
    4 adaptive: the tile's own count picks the branch.

    Mode 4 exists because the choice cannot be made per shape from the host:
    Triton resolves module globals at COMPILE time and caches the kernel, so
    rebinding a helper between calls silently keeps the first variant. The
    count is free in the cumsum path anyway."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    zeros = tl.zeros([BLOCK], tl.int32)
    tl.store(scr_ptr + row, 0)
    tl.debug_barrier()
    p = scr_ptr + row + zeros
    base = tl.zeros([], tl.int32)
    acc = tl.zeros([BLOCK], tl.int32)
    for t in range(TILES):
        v = tl.load(src_ptr + t * BLOCK + lane)
        take = (v % DENS) == 0
        if MODE == 0:
            pos = tl.atomic_add(p, zeros + 1, mask=take, sem="relaxed", scope="cta")
        elif MODE == 1:
            ti = take.to(tl.int32)
            pos = base + tl.cumsum(ti, axis=0) - ti
            base += tl.sum(ti, axis=0)
        elif MODE == 2:
            ti = take.to(tl.int32)
            total = tl.sum(ti, axis=0)
            start = tl.atomic_add(scr_ptr + row, total, sem="relaxed", scope="cta")
            pos = start + tl.cumsum(ti, axis=0) - ti
        elif MODE == 4:
            ti = take.to(tl.int32)
            total = tl.sum(ti, axis=0)
            if total >= THRESH:
                start = tl.atomic_add(scr_ptr + row, total, sem="relaxed", scope="cta")
                pos = start + tl.cumsum(ti, axis=0) - ti
            else:
                pos = tl.atomic_add(p, zeros + 1, mask=take, sem="relaxed", scope="cta")
        else:
            pos = lane
        # consume pos the way the operator does: a masked scatter store
        tl.store(out_ptr + row * 4096 + pos, v, mask=take & (pos < 4096))
        acc += tl.where(take, pos, 0)
    tl.store(scr_ptr + row, tl.sum(acc, axis=0))


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
    src = torch.randint(0, 1 << 20, (TILES * BLOCK,), dtype=torch.int32, device=dev)
    scr = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    out = torch.zeros(ROWS * 4096, dtype=torch.int32, device=dev)
    print(
        f"{ROWS} programs = 40 waves on {SMS} SMs | {TILES} tiles x {BLOCK} lanes "
        f"per program | microseconds per program\n"
    )
    print(
        f"  {'selected/tile':>13} {'atomic':>9} {'scan_reg':>9} {'scan_atomic':>12} "
        f"{'no alloc':>9} {'best vs atomic':>15}"
    )
    for dens in (64, 16, 8, 4, 2):
        sel = BLOCK // dens
        t = {}
        for mode in (0, 1, 2, 3):
            t[mode] = timed(
                lambda m=mode: k_slots[(ROWS,)](
                    src,
                    scr,
                    out,
                    DENS=dens,
                    MODE=m,
                    TILES=TILES,
                    BLOCK=BLOCK,
                    num_warps=WARPS,
                )
            )
        floor = t[3]
        best = min(t[1], t[2])
        gain = (t[0] - floor) / (best - floor) if best > floor else float("inf")
        print(
            f"  {sel:>13} {t[0]:>9.2f} {t[1]:>9.2f} {t[2]:>12.2f} {t[3]:>9.2f} "
            f"{gain:>14.2f}x"
        )
    print("\n  Costs are per program; 'adaptive vs best' divides the adaptive")
    print("  column's own cost (minus the floor) by the best fixed strategy's,")
    print("  so 1.0x means the branch is free and picks right. The operator's")
    print("  tiles select about top_k*BLOCK/vocab lanes each: 64 at vocab 4096")
    print("  with k=512, 4 at the (64,129280) production shape.")


if __name__ == "__main__":
    sys.exit(main())
