"""The masked smem atomic at the operator's REAL shape, and the workaround.

metax_smem_isolation found: a MASKED smem atomic with >= 2 elements per
thread writes wrong byte offsets; unmasked is fine, 1 element/thread is fine.
The operator's main loop tiles x_vec as [BLOCK, VEC] = [512, 4] on 8 warps --
4 elements per thread -- and does exactly that twice:

    _distribute_to_bins  tl.atomic_add(s_histogram_ptr + bin_idx, ones,
                                       mask=is_partial_match)        scatter
    _process_bins        out_pos = tl.atomic_add(found_ptrs, ones,
                                                 mask=take)          single
                                                                     address

STEP 0's mask is ~all-true (in_range), which is why random full rows mostly
survive; later steps and the merge have sparse masks.

Workaround under test (MODE 1): drop the mask and add take.to(int32) -- an
untaken lane adds 0, changing nothing, and its returned value is unused.

    /data/wuyuqing/workspace/mctle-test/bin/python tools/metax_smem_atomic_2d.py
"""

import sys

import torch
import triton
import triton.language as tl

BLOCK = 512
WARPS = 8
NB = 2048


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
def _tile(src_ptr, BLOCK: tl.constexpr, VEC: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    if VEC == 1:
        offs = lane
    else:
        offs = (lane * VEC)[:, None] + tl.arange(0, VEC)[None, :]
    return offs, tl.load(src_ptr + offs)


@triton.jit
def k_scatter(src_ptr, out_ptr, NB: tl.constexpr, BLOCK: tl.constexpr,
              VEC: tl.constexpr, DENS: tl.constexpr, MODE: tl.constexpr):
    row = tl.program_id(0)
    buf = tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    view = tle.gpu.local_ptr(buf)
    tl.store(view, tl.zeros([NB], tl.int32))
    tl.debug_barrier()
    offs, v = _tile(src_ptr, BLOCK, VEC)
    b = v % NB
    take = ((v // NB) % DENS) == 0
    sp = tle.gpu.local_ptr(buf, (0,))
    if MODE == 0:
        tl.atomic_add(sp + b, v * 0 + 1, mask=take, sem="relaxed", scope="cta")
    else:
        tl.atomic_add(sp + b, take.to(tl.int32), sem="relaxed", scope="cta")
    tl.debug_barrier()
    tl.store(out_ptr + row * NB + tl.arange(0, NB), tl.load(view))


@triton.jit
def k_single(src_ptr, r_ptr, BLOCK: tl.constexpr, VEC: tl.constexpr,
             DENS: tl.constexpr, MODE: tl.constexpr):
    buf = tle.gpu.alloc([64], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    tl.store(tle.gpu.local_ptr(buf), tl.zeros([64], tl.int32))
    tl.debug_barrier()
    p = tle.gpu.local_ptr(buf, (0,))
    offs, v = _tile(src_ptr, BLOCK, VEC)
    zeros = v * 0
    take = (v % DENS) == 0
    if MODE == 0:
        r = tl.atomic_add(p + zeros, zeros + 1, mask=take, sem="relaxed", scope="cta")
    else:
        r = tl.atomic_add(p + zeros, take.to(tl.int32), sem="relaxed", scope="cta")
    tl.store(r_ptr + offs, tl.where(take, r, -1))
    tl.debug_barrier()
    tl.store(r_ptr + BLOCK * VEC, tl.load(p))


def main():
    dev = "cuda"
    torch.manual_seed(0)
    print(f"BLOCK={BLOCK} x {WARPS} warps (64 lanes) -> elems/thread = VEC\n")
    print(f"  {'kind':<8} {'VEC':>3} {'take':>5} {'mode':<16} {'result':<8} detail")
    for kind in ("scatter", "single"):
        for vec in (1, 4):
            n_items = BLOCK * vec
            src = torch.randint(0, 1 << 20, (n_items,), dtype=torch.int32, device=dev)
            for dens in (1, 4):
                for mode, mname in ((0, "masked"), (1, "unmasked+0/1")):
                    if kind == "scatter":
                        grid = 416
                        out = torch.full((grid * NB,), -9, dtype=torch.int32, device=dev)
                        k_scatter[(grid,)](src, out, NB=NB, BLOCK=BLOCK, VEC=vec,
                                           DENS=dens, MODE=mode, num_warps=WARPS)
                        torch.cuda.synchronize()
                        s = src.cpu().long()
                        take = ((s // NB) % dens) == 0
                        want = torch.bincount(s[take] % NB, minlength=NB).to(torch.int32)
                        o = out.view(grid, NB).cpu()
                        bad = int((o != want).any(dim=1).sum())
                        ok = bad == 0
                        detail = f"bad rows {bad}/{grid}  total={int(o[0].sum())} want {int(want.sum())}"
                    else:
                        r = torch.full((n_items + 1,), -9, dtype=torch.int32, device=dev)
                        k_single[(1,)](src, r, BLOCK=BLOCK, VEC=vec, DENS=dens,
                                       MODE=mode, num_warps=WARPS)
                        torch.cuda.synchronize()
                        rc = r.cpu()
                        take = (src.cpu() % dens) == 0
                        n = int(take.sum())
                        got = rc[:n_items][take]
                        perm = bool(torch.equal(got.sort().values, torch.arange(n, dtype=got.dtype)))
                        fin = int(rc[n_items])
                        ok = perm and fin == n
                        detail = (f"taken={n} unique={int(got.unique().numel())} "
                                  f"permutation={perm} final={fin}")
                    print(f"  {kind:<8} {vec:>3} {'1/' + str(dens):>5} {mname:<16} "
                          f"{'OK' if ok else 'BROKEN':<8} {detail}")
    print("\n  If masked/VEC=4/take 1/4 is BROKEN and unmasked+0/1 is OK on the same")
    print("  row, the workaround is a MetaX rebinding of _distribute_to_bins and")
    print("  _process_bins -- no generic diff.")


if __name__ == "__main__":
    sys.exit(main())
