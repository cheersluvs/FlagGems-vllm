"""Are mctle smem buffers private to their CTA on MetaX, and does a barrier
order the zero-fill before the atomics?

metax_smem_atomic_cost found a single-address smem atomic that is a perfect
permutation in ONE program but wrong in some rows of a big grid: 0 bad at
grid 104 (one CTA per SM), 18 at 416, ~170 of 4160 with 1/4 lanes taken.
metadata.shared is right (256 B), so the driver is not over-packing. Two of
the bad 1/1 rows were every return shifted by one constant -- 6845.0 and
12541.0 exactly -- i.e. the counter did not start at 0.

Two mechanisms fit, and they need different fixes:

    ISOLATION  a co-resident CTA writes into this CTA's buffer
    ORDERING   the zero-fill has not landed when the barrier releases the
               atomics (a barrier that does not wait for outstanding smem
               stores)

Each CTA writes its own row id into buf[1] as a signature and zeroes the
counter buf[0], then reads both back after the barrier (init0, sig0), runs the
atomics, and reads both again at the end (fin, sig1):

    sig0/sig1 != row           -> ISOLATION broken
    init0 != 0, signature ok   -> ORDERING broken
    fin != n but both ok       -> lost/extra increments inside the CTA

    /data/wuyuqing/workspace/mctle-test/bin/python tools/metax_smem_isolation.py
"""

import sys

import torch
import triton
import triton.language as tl

BLOCK = 512
CHUNKS = 8
BIG = tl.constexpr(1 << 30)     # a plain int global is rejected inside @jit
REC = 8


def _mctle_ok():
    from triton._C import libtriton as L
    try:
        import triton.backends.metax.compiler as c
        enabled = getattr(c, "enable_mctle", None)
    except Exception:  # noqa: BLE001
        enabled = None
    return hasattr(L.ir.builder, "make_swizzled_shared_encoding_attr") and enabled is True


if not _mctle_ok():
    print("!! not an mctle build")
    sys.exit(3)

import triton.experimental.tle.language as tle  # noqa: E402


@triton.jit
def k_diag(src_ptr, rec_ptr, DENS: tl.constexpr, CHUNKS: tl.constexpr,
           BLOCK: tl.constexpr, NBUF: tl.constexpr, BARRIERS: tl.constexpr,
           PROBE: tl.constexpr, SIGPOS: tl.constexpr):
    """PROBE=False is the OPERATOR's shape: fill, one barrier, atomics -- no
    read-back and no second barrier in between, which round 1 suspected of
    hiding an ordering bug (the cost tool, written that way, had bad rows at
    8 warps; this kernel with the read-back had none). init0/sig0 are then
    recorded as -5 and the atomic's min return is the witness for init."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    zeros = tl.zeros([BLOCK], tl.int32)
    buf = tle.gpu.alloc([NBUF], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    cells = tl.arange(0, NBUF)
    tl.store(tle.gpu.local_ptr(buf), tl.where(cells == SIGPOS, row, 0))
    for _ in tl.static_range(BARRIERS):
        tl.debug_barrier()
    c0 = tle.gpu.local_ptr(buf, (0,))
    s1 = tle.gpu.local_ptr(buf, (SIGPOS,))
    if PROBE:
        init0 = tl.load(c0)
        sig0 = tl.load(s1)
        tl.debug_barrier()
    else:
        init0 = tl.full((), -5, tl.int32)
        sig0 = row
    p = c0 + zeros
    acc = zeros
    mn = zeros + BIG
    for c in range(CHUNKS):
        v = tl.load(src_ptr + c * BLOCK + lane)
        take = (v % DENS) == 0
        r = tl.atomic_add(p, zeros + 1, mask=take, sem="relaxed", scope="cta")
        acc += tl.where(take, r, 0)
        mn = tl.minimum(mn, tl.where(take, r, BIG))
    tl.debug_barrier()
    fin = tl.load(c0)
    sig1 = tl.load(s1)
    base = rec_ptr + row * 8
    tl.store(base + 0, init0)
    tl.store(base + 1, sig0)
    tl.store(base + 2, fin)
    tl.store(base + 3, sig1)
    tl.store(base + 4, tl.sum(acc))
    tl.store(base + 5, tl.min(mn))


def run(src, grid, dens, nbuf, warps, barriers, probe, sigpos):
    rec = torch.full((grid * REC,), -7, dtype=torch.int32, device="cuda")
    k_diag[(grid,)](src, rec, DENS=dens, CHUNKS=CHUNKS, BLOCK=BLOCK, NBUF=nbuf,
                    BARRIERS=barriers, PROBE=probe, SIGPOS=sigpos, num_warps=warps)
    torch.cuda.synchronize()
    r = rec.view(grid, REC).cpu()
    rows = torch.arange(grid, dtype=torch.int32)
    n = int(((src.cpu() % dens) == 0).sum())
    want_sum = n * (n - 1) // 2
    # Round 2 showed the "other CTA" reading was wrong: the only failing
    # config broke EVERY row, with its own signature zeroed at the end -- the
    # CTA clobbering itself. So name the columns by what they observe.
    sig = (r[:, 1] != rows) | (r[:, 3] != rows)       # signature cell clobbered
    init = (r[:, 5] != 0) | ((r[:, 0] != 0) & (r[:, 0] != -5))   # counter not 0 at start
    count = (r[:, 2] != n)                            # final count wrong
    sumbad = (r[:, 4] != want_sum)
    ex = sumbad.nonzero().flatten()[:2].tolist()
    detail = "; ".join(f"row{i}: init0={int(r[i,0])} fin={int(r[i,2])} "
                       f"sig1={int(r[i,3])} min={int(r[i,5])}" for i in ex)
    return (int(sig.sum()), int(init.sum()), int(count.sum()), int(sumbad.sum()),
            n, detail)


# (nbuf, warps, barriers, probe, sigpos, why)
CONFIGS = (
    (64, 8, 1, False, 1, "OPERATOR shape: fill, 1 barrier, atomics"),
    (64, 8, 2, False, 1, "same + a second barrier"),
    (64, 8, 1, True, 1, "round-2 shape (read-back between)"),
    (64, 4, 1, True, 1, "2 elems/thread, round-2 failure"),
    (64, 4, 1, True, 32, "2 elems/thread, signature far from the counter"),
    (64, 4, 1, False, 1, "2 elems/thread, operator shape"),
    (64, 2, 1, True, 1, "4 elems/thread"),
    (2048, 8, 1, False, 1, "operator shape, histogram-sized buffer"),
)


def main():
    torch.manual_seed(0)
    src = torch.randint(0, 1 << 20, (CHUNKS * BLOCK,), dtype=torch.int32, device="cuda")
    print(f"BLOCK={BLOCK}  {CHUNKS * BLOCK} items/program  up to 3 reps; "
          f"counts are bad rows\n")
    print(f"  {'grid':>5} {'take':>5} {'nbuf':>5} {'w':>2} {'bar':>3} {'probe':>5} "
          f"{'sig@':>4}  {'SIG':>5} {'INIT':>5} {'COUNT':>5} {'SUM':>5}  example")
    for grid in (104, 4160):
        for dens in (1, 4):
            for nbuf, warps, bars, probe, sigpos, why in CONFIGS:
                worst = None
                for rep in range(3):
                    res = run(src, grid, dens, nbuf, warps, bars, probe, sigpos)
                    if worst is None or res[3] > worst[3]:
                        worst = res
                sig, init, cnt, sb, n, detail = worst
                print(f"  {grid:>5} {'1/' + str(dens):>5} {nbuf:>5} {warps:>2} {bars:>3} "
                      f"{str(probe):>5} {sigpos:>4}  {sig:>5} {init:>5} {cnt:>5} {sb:>5}  "
                      f"{detail[:110] if sb else why}")
        print()


if __name__ == "__main__":
    sys.exit(main())
