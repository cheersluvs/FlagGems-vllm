"""Is the TLE path worth fixing on MetaX? Price a shared-memory atomic against
the global one the non-TLE path uses.

prefill's bottleneck is `tl.atomic_add(found_topk_values_ptrs, ...)` in
_process_bins: every taken lane of every chunk hitting ONE address, ablated at
7.75 of 19.4 us per program. Without TLE that address is global scratch; the
TLE path puts it in shared memory.

Atomics on local_ptr now verify and run correctly (the -D__MCTLE__ build), but
moving the operator onto TLE on MetaX is real work: tle.cumsum has no metax
binding, and the operator's smem loads at 1 element per thread hit the
plugin's vec assert. So price the prize before paying for it.

Two patterns, each global vs smem, at the operator's geometry (BLOCK=512,
8 warps of 64 lanes, 4160 rows = 40 waves on 104 SMs), plus the same loop with
no atomic at all so the atomic's own share can be read off:

    single  one address, masked, returned value consumed   (_process_bins)
    hist    2048-bin scatter, masked                        (_distribute_to_bins)

Needs the mctle wheel:

    /data/wuyuqing/workspace/mctle-test/bin/python tools/metax_smem_atomic_cost.py
"""

import sys

import torch
import triton
import triton.language as tl

ROWS = 4160
SMS = 104
BLOCK = 512
WARPS = 8
NB = 2048
CHUNKS = 8           # 4096 items per program


def _mctle_ok():
    from triton._C import libtriton as L
    try:
        import triton.backends.metax.compiler as c
        enabled = getattr(c, "enable_mctle", None)
    except Exception:  # noqa: BLE001
        enabled = None
    ok = hasattr(L.ir.builder, "make_swizzled_shared_encoding_attr") and enabled is True
    print(f"libtriton {L.__file__}  mctle={'yes' if ok else 'NO'}")
    return ok


if not _mctle_ok():
    print("!! not an mctle build; the smem kernels cannot compile here")
    sys.exit(3)

import triton.experimental.tle.language as tle  # noqa: E402


@triton.jit
def k_loop_only(src_ptr, out_ptr, DENS: tl.constexpr, CHUNKS: tl.constexpr,
                BLOCK: tl.constexpr):
    """The loop both atomic kernels share, with the atomic removed."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.int32)
    for c in range(CHUNKS):
        v = tl.load(src_ptr + c * BLOCK + lane)
        take = (v % DENS) == 0
        acc += tl.where(take, v, 0)
    tl.store(out_ptr + row, tl.sum(acc))


@triton.jit
def k_single_global(scr_ptr, src_ptr, out_ptr, DENS: tl.constexpr,
                    CHUNKS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    zeros = tl.zeros([BLOCK], tl.int32)
    ones = zeros + 1
    tl.store(scr_ptr + row, 0)
    tl.debug_barrier()
    p = scr_ptr + row + zeros
    acc = tl.zeros([BLOCK], tl.int32)
    for c in range(CHUNKS):
        v = tl.load(src_ptr + c * BLOCK + lane)
        take = (v % DENS) == 0
        r = tl.atomic_add(p, ones, mask=take, sem="relaxed", scope="cta")
        acc += tl.where(take, r, 0)
    tl.store(out_ptr + row, tl.sum(acc))


@triton.jit
def k_single_smem(src_ptr, out_ptr, DENS: tl.constexpr, CHUNKS: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    zeros = tl.zeros([BLOCK], tl.int32)
    ones = zeros + 1
    buf = tle.gpu.alloc([64], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    tl.store(tle.gpu.local_ptr(buf), tl.zeros([64], tl.int32))
    tl.debug_barrier()
    p = tle.gpu.local_ptr(buf, (0,)) + zeros
    acc = tl.zeros([BLOCK], tl.int32)
    for c in range(CHUNKS):
        v = tl.load(src_ptr + c * BLOCK + lane)
        take = (v % DENS) == 0
        r = tl.atomic_add(p, ones, mask=take, sem="relaxed", scope="cta")
        acc += tl.where(take, r, 0)
    tl.store(out_ptr + row, tl.sum(acc))


@triton.jit
def k_hist_global(h_ptr, src_ptr, out_ptr, NB: tl.constexpr, DENS: tl.constexpr,
                  CHUNKS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = h_ptr + row * NB
    for c in tl.static_range(NB // BLOCK):
        tl.store(base + c * BLOCK + lane, tl.zeros([BLOCK], tl.int32))
    tl.debug_barrier()
    ones = tl.full([BLOCK], 1, tl.int32)
    for c in range(CHUNKS):
        v = tl.load(src_ptr + c * BLOCK + lane)
        take = (v % DENS) == 0
        tl.atomic_add(base + (v // DENS) % NB, ones, mask=take,
                      sem="relaxed", scope="cta")
    tl.debug_barrier()
    s = tl.zeros([BLOCK], tl.int32)
    for c in tl.static_range(NB // BLOCK):
        b = c * BLOCK + lane
        s += tl.load(base + b) * b
    tl.store(out_ptr + row, tl.sum(s))


@triton.jit
def k_hist_smem(src_ptr, out_ptr, NB: tl.constexpr, DENS: tl.constexpr,
                CHUNKS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    buf = tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    view = tle.gpu.local_ptr(buf)
    tl.store(view, tl.zeros([NB], tl.int32))
    tl.debug_barrier()
    sp = tle.gpu.local_ptr(buf, (0,))
    ones = tl.full([BLOCK], 1, tl.int32)
    for c in range(CHUNKS):
        v = tl.load(src_ptr + c * BLOCK + lane)
        take = (v % DENS) == 0
        tl.atomic_add(sp + (v // DENS) % NB, ones, mask=take,
                      sem="relaxed", scope="cta")
    tl.debug_barrier()
    h = tl.load(view)                       # full view: the load form that works
    tl.store(out_ptr + row, tl.sum(h * tl.arange(0, NB)))


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
    return a.elapsed_time(b) / iters * 1000 / (ROWS / SMS)   # us per program


def main():
    dev = "cuda"
    torch.manual_seed(0)
    src = torch.randint(0, 1 << 20, (CHUNKS * BLOCK,), dtype=torch.int32, device=dev)
    out_g = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    out_s = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    scr = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    hist = torch.zeros(ROWS * NB, dtype=torch.int32, device=dev)
    kw = dict(CHUNKS=CHUNKS, BLOCK=BLOCK, num_warps=WARPS)

    print(f"{ROWS} rows = {ROWS // SMS} waves | BLOCK={BLOCK} x {WARPS} warps | "
          f"{CHUNKS * BLOCK} items/program | per-program microseconds\n")
    print(f"  {'pattern':<8} {'take':>5} {'loop':>7} {'global':>8} {'smem':>8}"
          f" {'atomic g':>9} {'atomic s':>9} {'ratio':>6}  check")
    for pattern in ("single", "hist"):
        for dens in (1, 4):
            t_loop = timed(lambda: k_loop_only[(ROWS,)](src, out_g, DENS=dens, **kw))
            if pattern == "single":
                t_g = timed(lambda: k_single_global[(ROWS,)](scr, src, out_g, DENS=dens, **kw))
                t_s = timed(lambda: k_single_smem[(ROWS,)](src, out_s, DENS=dens, **kw))
                n = int(((src.cpu() % dens) == 0).sum())
                want = n * (n - 1) // 2          # sum of old values, any order
                ok = bool((out_g.cpu() == want).all() and (out_s.cpu() == want).all())
            else:
                t_g = timed(lambda: k_hist_global[(ROWS,)](hist, src, out_g, NB=NB, DENS=dens, **kw))
                t_s = timed(lambda: k_hist_smem[(ROWS,)](src, out_s, NB=NB, DENS=dens, **kw))
                ok = bool(torch.equal(out_g, out_s)) and int(out_g[0]) != 0
            ag, as_ = t_g - t_loop, t_s - t_loop
            ratio = ag / as_ if as_ > 0.05 else float("inf")
            print(f"  {pattern:<8} {'1/' + str(dens):>5} {t_loop:>7.2f} {t_g:>8.2f}"
                  f" {t_s:>8.2f} {ag:>9.2f} {as_:>9.2f} {ratio:>6.1f}  "
                  f"{'OK' if ok else 'MISMATCH'}")

    print("\n  'atomic g/s' = kernel minus the atomic-free loop. prefill's single-")
    print("  address term is ~7.75 us of 19.4 per program; if 'single' shows most")
    print("  of that vanishing in smem, the TLE path is worth its two fixes.")


if __name__ == "__main__":
    sys.exit(main())
