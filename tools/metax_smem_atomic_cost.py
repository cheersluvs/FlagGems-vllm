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
import torch as _torch  # noqa: E402
import triton
import triton.language as tl

# 40 waves on whatever card this is: C550 has 104 SMs, BW1000 has 80.
SMS = int(getattr(_torch.cuda.get_device_properties(0), "multi_processor_count", 104))
ROWS = 40 * SMS
BLOCK = 512
WARPS = 8
NB = 2048
CHUNKS = 8  # 4096 items per program


def _tle_ok():
    """Vendor-neutral: this needs the local_ptr bindings, nothing metax-specific.
    On metax that means an mctle build; on hcu the stock wheel already has them."""
    from triton._C import libtriton as L

    ok = hasattr(L.ir.builder, "make_swizzled_shared_encoding_attr")
    print(f"libtriton {L.__file__}  local_ptr bindings={'yes' if ok else 'NO'}")
    return ok


if not _tle_ok():
    print("!! no TLE bindings in this build; the smem kernels cannot compile here")
    sys.exit(3)

import triton.experimental.tle.language as tle  # noqa: E402


@triton.jit
def k_loop_only(
    src_ptr, out_ptr, DENS: tl.constexpr, CHUNKS: tl.constexpr, BLOCK: tl.constexpr
):
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
def k_single_global(
    scr_ptr,
    src_ptr,
    out_ptr,
    DENS: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
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
def k_single_smem(
    src_ptr, out_ptr, DENS: tl.constexpr, CHUNKS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    zeros = tl.zeros([BLOCK], tl.int32)
    ones = zeros + 1
    buf = tle.gpu.alloc(
        [64],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
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
def k_hist_global(
    h_ptr,
    src_ptr,
    out_ptr,
    NB: tl.constexpr,
    DENS: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
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
        tl.atomic_add(
            base + (v // DENS) % NB, ones, mask=take, sem="relaxed", scope="cta"
        )
    tl.debug_barrier()
    s = tl.zeros([BLOCK], tl.int32)
    for c in tl.static_range(NB // BLOCK):
        b = c * BLOCK + lane
        s += tl.load(base + b) * b
    tl.store(out_ptr + row, tl.sum(s))


@triton.jit
def k_hist_smem(
    src_ptr,
    out_ptr,
    NB: tl.constexpr,
    DENS: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    buf = tle.gpu.alloc(
        [NB],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    view = tle.gpu.local_ptr(buf)
    tl.store(view, tl.zeros([NB], tl.int32))
    tl.debug_barrier()
    sp = tle.gpu.local_ptr(buf, (0,))
    ones = tl.full([BLOCK], 1, tl.int32)
    for c in range(CHUNKS):
        v = tl.load(src_ptr + c * BLOCK + lane)
        take = (v % DENS) == 0
        tl.atomic_add(
            sp + (v // DENS) % NB, ones, mask=take, sem="relaxed", scope="cta"
        )
    tl.debug_barrier()
    h = tl.load(view)  # full view: the load form that works
    tl.store(out_ptr + row, tl.sum(h * tl.arange(0, NB)))


@triton.jit
def k_dump_global(
    scr_ptr,
    src_ptr,
    r_ptr,
    DENS: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program; every lane's returned old value, -1 where not taken."""
    lane = tl.arange(0, BLOCK)
    zeros = tl.zeros([BLOCK], tl.int32)
    tl.store(scr_ptr, 0)
    tl.debug_barrier()
    p = scr_ptr + zeros
    for c in range(CHUNKS):
        v = tl.load(src_ptr + c * BLOCK + lane)
        take = (v % DENS) == 0
        r = tl.atomic_add(p, zeros + 1, mask=take, sem="relaxed", scope="cta")
        tl.store(r_ptr + c * BLOCK + lane, tl.where(take, r, -1))


@triton.jit
def k_dump_smem(
    src_ptr, r_ptr, DENS: tl.constexpr, CHUNKS: tl.constexpr, BLOCK: tl.constexpr
):
    lane = tl.arange(0, BLOCK)
    zeros = tl.zeros([BLOCK], tl.int32)
    buf = tle.gpu.alloc(
        [64],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    tl.store(tle.gpu.local_ptr(buf), tl.zeros([64], tl.int32))
    tl.debug_barrier()
    p = tle.gpu.local_ptr(buf, (0,)) + zeros
    for c in range(CHUNKS):
        v = tl.load(src_ptr + c * BLOCK + lane)
        take = (v % DENS) == 0
        r = tl.atomic_add(p, zeros + 1, mask=take, sem="relaxed", scope="cta")
        tl.store(r_ptr + c * BLOCK + lane, tl.where(take, r, -1))


def dump_check(name, r, take):
    """Returned old values of the taken lanes must be exactly 0..n-1."""
    r, take = r.cpu(), take.cpu()
    got = r[take]
    n = int(take.sum())
    uniq = int(got.unique().numel())
    perm = bool(torch.equal(got.sort().values, torch.arange(n, dtype=got.dtype)))
    stray = int((r[~take] != -1).sum())
    print(
        f"  {name:<7} taken={n:<5} unique={uniq:<5} permutation={perm}  "
        f"untaken!=-1: {stray}  min={int(got.min())} max={int(got.max())}"
    )
    if not perm:
        # per-warp view of the first chunk: a warp-aggregated atomic that hands
        # every lane the same old value shows up as unique << taken here
        for w in range(min(4, BLOCK // 64)):
            seg_r, seg_t = r[w * 64 : (w + 1) * 64], take[w * 64 : (w + 1) * 64]
            vals = seg_r[seg_t]
            print(
                f"      warp {w}: taken={int(seg_t.sum()):<3} "
                f"unique={int(vals.unique().numel()):<3} first={vals[:6].tolist()}"
            )


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
    out_g = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    out_s = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    scr = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    hist = torch.zeros(ROWS * NB, dtype=torch.int32, device=dev)
    kw = dict(CHUNKS=CHUNKS, BLOCK=BLOCK, num_warps=WARPS)
    bad = []

    print(
        f"{ROWS} rows = {ROWS // SMS} waves | BLOCK={BLOCK} x {WARPS} warps | "
        f"{CHUNKS * BLOCK} items/program | per-program microseconds\n"
    )
    print(
        f"  {'pattern':<8} {'take':>5} {'loop':>7} {'global':>8} {'smem':>8}"
        f" {'atomic g':>9} {'atomic s':>9} {'ratio':>6}  check"
    )
    for pattern in ("single", "hist"):
        for dens in (1, 4):
            t_loop = timed(lambda: k_loop_only[(ROWS,)](src, out_g, DENS=dens, **kw))
            if pattern == "single":
                t_g = timed(
                    lambda: k_single_global[(ROWS,)](scr, src, out_g, DENS=dens, **kw)
                )
                t_s = timed(lambda: k_single_smem[(ROWS,)](src, out_s, DENS=dens, **kw))
                n = int(((src.cpu() % dens) == 0).sum())
                want = n * (n - 1) // 2  # sum of old values, any order
                okg = bool((out_g.cpu() == want).all())
                oks = bool((out_s.cpu() == want).all())
                ok = okg and oks
                if not ok:
                    bad.append(
                        (dens, okg, oks, want, out_g[:3].tolist(), out_s[:3].tolist())
                    )
            else:
                t_g = timed(
                    lambda: k_hist_global[(ROWS,)](
                        hist, src, out_g, NB=NB, DENS=dens, **kw
                    )
                )
                t_s = timed(
                    lambda: k_hist_smem[(ROWS,)](src, out_s, NB=NB, DENS=dens, **kw)
                )
                ok = bool(torch.equal(out_g, out_s)) and int(out_g[0]) != 0
            ag, as_ = t_g - t_loop, t_s - t_loop
            ratio = ag / as_ if as_ > 0.05 else float("inf")
            print(
                f"  {pattern:<8} {'1/' + str(dens):>5} {t_loop:>7.2f} {t_g:>8.2f}"
                f" {t_s:>8.2f} {ag:>9.2f} {as_:>9.2f} {ratio:>6.1f}  "
                f"{'OK' if ok else 'MISMATCH'}"
            )

    for dens, okg, oks, want, g3, s3 in bad:
        print(
            f"\n  MISMATCH single 1/{dens}: global {'OK' if okg else 'BAD'}"
            f" {g3}  smem {'OK' if oks else 'BAD'} {s3}  want {want}"
        )

    # One program is a perfect permutation, 4160 are not: suspect CTAs sharing
    # smem. If mctle's alloc is not counted in metadata.shared, the driver
    # packs more CTAs per SM than the smem allows and their buffers overlap.
    print("\n  multi-program smem single, which rows go wrong:")
    for dens in (1, 4):
        n = int(((src.cpu() % dens) == 0).sum())
        want = n * (n - 1) // 2
        for rep in range(3):
            out_s.zero_()
            k_single_smem[(ROWS,)](src, out_s, DENS=dens, **kw)
            torch.cuda.synchronize()
            o = out_s.cpu()
            badrows = (o != want).nonzero().flatten()
            print(
                f"  1/{dens} rep{rep}: bad rows {badrows.numel()}/{ROWS}"
                f"  first={badrows[:8].tolist()}"
                f"  values={o[badrows[:4]].tolist()} want {want}"
            )
        for grid in (104, 208, 416):
            out_s.zero_()
            k_single_smem[(grid,)](src, out_s, DENS=dens, **kw)
            torch.cuda.synchronize()
            nb = int((out_s[:grid].cpu() != want).sum())
            print(f"  1/{dens} grid={grid:<4} bad {nb}")

    def smem_of(k, *a, **kk):
        ck = k[(1,)](*a, **kk)
        md = getattr(ck, "metadata", None)
        return getattr(md, "shared", "?"), getattr(ck, "n_regs", "?")

    print("\n  compiled kernel shared-memory size (bytes) and regs:")
    print(
        f"  single_smem  alloc 256 B   -> shared, regs = "
        f"{smem_of(k_single_smem, src, out_s, DENS=1, **kw)}"
    )
    print(
        f"  hist_smem    alloc 8192 B  -> shared, regs = "
        f"{smem_of(k_hist_smem, src, out_s, NB=NB, DENS=1, **kw)}"
    )
    print(
        f"  single_global (no alloc)   -> shared, regs = "
        f"{smem_of(k_single_global, scr, src, out_g, DENS=1, **kw)}"
    )

    # The operator USES the returned value (out_pos_lt is a write position),
    # so a masked atomic that returns wrong old values is a correctness bug,
    # not a timing footnote. Look at the values themselves.
    print("\n  masked single-address atomic, returned old values (one program):")
    r = torch.empty(CHUNKS * BLOCK, dtype=torch.int32, device=dev)
    for dens in (1, 4):
        take = (src % dens) == 0
        print(f"  take 1/{dens}")
        k_dump_global[(1,)](scr, src, r, DENS=dens, **kw)
        torch.cuda.synchronize()
        dump_check("global", r, take)
        k_dump_smem[(1,)](src, r, DENS=dens, **kw)
        torch.cuda.synchronize()
        dump_check("smem", r, take)

    print("\n  'atomic g/s' = kernel minus the atomic-free loop. prefill's single-")
    print("  address term is ~7.75 us of 19.4 per program; if 'single' shows most")
    print("  of that vanishing in smem, the TLE path is worth its two fixes.")


if __name__ == "__main__":
    sys.exit(main())
