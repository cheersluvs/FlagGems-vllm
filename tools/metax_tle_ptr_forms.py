"""Does the full-view form of tle.gpu.local_ptr dodge MetaX's load assert?

With indices, local_ptr returns a SCALAR pointer and the kernel builds a
pointer tensor itself -- tt.splat then tt.addptr -- and MetaX's
LoadStoreOpToLLVM.cpp:427 then asserts
`wordNElems * nWords * numVecs == numElems` on the load.

Without indices the docstring says it emits "a full-view pointer tensor over
buffer shape", so the pointer tensor arrives from a different path and is used
directly, with no splat and no addptr. Whether that avoids the same assert is
the question; it is a different IR shape, not merely a shorter call.

Store and load are separated too, because the assert is in LoadOpConversion:
if stores lower and only loads fail, that is a sharper report and possibly a
narrower workaround.

    python tools/metax_tle_ptr_forms.py
"""

import os
import subprocess
import sys

CASES = ("scalar_store", "scalar_roundtrip", "view_store", "view_roundtrip",
         "atomic_view_plain", "atomic_scalar_arange", "atomic_masked_operator_form",
         "atomic_scatter_view_read", "atomic_scatter_scalar_read",
         # --- why does scalar+offs LOAD assert? ------------------------------
         # Hypothesis: AxisInfo proves the arange contiguous, so vec=4, but
         # NB=256 over 4 warps x 64 lanes is ONE element per thread, and
         # numVecs = 1/4 = 0 fails `wordNElems*nWords*numVecs == numElems`.
         # The full view works because its contiguity is unknown -> vec=1.
         # If that is right: >=4 elems/thread passes, an unprovable stride
         # passes, and the operator's shapes can be predicted from geometry.
         "load_nb256_w1",      # 4 elems/thread          -> predict CORRECT
         "load_nb1024_w4",     # 4 elems/thread          -> predict CORRECT
         "load_nb2048_w8",     # operator's histogram: 2048 bins, 8 warps
         "load_nb256_w8",      # operator's radix: 256 bins, 8 warps (0.5/thr)
         "load_nb512_w4",      # 2 elems/thread          -> predict CRASH
         "load_opaque_stride", # NB=256 w4, stride unprovable -> predict CORRECT
         "load_0d",            # tl.load(scalar_ptr): s_found_topk_values etc.
         "gather_masked")      # tl.load(scalar + idx, mask) with random idx

def _whoami():
    """Say which build this is, before any result is read.

    Every verdict here depends on WHICH libtriton is loaded, and on this box
    that silently changed under us once: $HOME was wiped, the throwaway venv
    with the mctle wheel went with it, and the metadata still reported a
    flagtree version. A result is only meaningful next to the build it came
    from, so print the loaded library and whether mctle is live, and refuse
    to run the cases at all if it is not -- a run on the wrong build produces
    plausible-looking failures that mean nothing.
    """
    import triton
    from triton._C import libtriton as L
    has_mctle = hasattr(L, "mctle")
    has_swz = hasattr(L.ir.builder, "make_swizzled_shared_encoding_attr")
    try:
        import triton.backends.metax.compiler as c
        enabled = getattr(c, "enable_mctle", None)
    except Exception:  # noqa: BLE001
        enabled = None
    print(f"python     {sys.executable}")
    print(f"libtriton  {L.__file__}")
    print(f"mctle      module={has_mctle}  swizzled_binding={has_swz}  "
          f"enable_mctle={enabled}")
    return has_mctle and has_swz and enabled is True


if len(sys.argv) == 1:
    if not _whoami():
        print("\n!! this is NOT an mctle build -- every case would fail for "
              "that reason alone. Install the mctle wheel into this python "
              "first; refusing to run.")
        raise SystemExit(3)
    print()
    print("Each case in its own process: an assert failure aborts the "
          "interpreter, so one case cannot be allowed to take the rest.\n")
    for c in CASES:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), c],
                           capture_output=True, text=True, timeout=600)
        out = (r.stdout + r.stderr).strip().splitlines()
        verdict = next((l for l in out if l.startswith("RESULT")), None)
        if verdict is None:
            detail = next((l for l in out if "Assertion" in l or "error:" in l),
                          out[-1] if out else "no output")
            verdict = f"RESULT {c}: CRASHED -- {detail.strip()[:150]}"
        print(f"  {verdict}")
    raise SystemExit(0)

CASE = sys.argv[1]

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
import triton.experimental.tle.language as tle  # noqa: E402

NB = 256


def _alloc(NB):
    return tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                         scope=tle.gpu.smem, nv_mma_shared_layout=False)


@triton.jit
def k_scalar_store(out_ptr, NB: tl.constexpr):
    buf = tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    p = tle.gpu.local_ptr(buf, (0,))
    lane = tl.arange(0, NB)
    tl.store(p + lane, lane * 2)
    tl.store(out_ptr + lane, lane * 2)      # answer from elsewhere: no smem read


@triton.jit
def k_scalar_roundtrip(out_ptr, NB: tl.constexpr):
    buf = tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    p = tle.gpu.local_ptr(buf, (0,))
    lane = tl.arange(0, NB)
    tl.store(p + lane, lane * 2)
    tl.debug_barrier()
    tl.store(out_ptr + lane, tl.load(p + lane))


@triton.jit
def k_view_store(out_ptr, NB: tl.constexpr):
    buf = tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    p = tle.gpu.local_ptr(buf)              # full view: already a [NB] tensor
    lane = tl.arange(0, NB)
    tl.store(p, lane * 2)                   # no + lane
    tl.store(out_ptr + lane, lane * 2)


@triton.jit
def k_view_roundtrip(out_ptr, NB: tl.constexpr):
    buf = tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    p = tle.gpu.local_ptr(buf)
    lane = tl.arange(0, NB)
    tl.store(p, lane * 2)
    tl.debug_barrier()
    tl.store(out_ptr + lane, tl.load(p))


@triton.jit
def k_atomic_scatter_view_read(idx_ptr, out_ptr, NB: tl.constexpr):
    """The shape top_k_per_row actually needs.

    Scatter with atomic_add through a scalar pointer plus computed indices --
    a different conversion from the one that asserts, AtomicRMWOpConversion
    rather than LoadOpConversion -- and read the whole buffer back through the
    full-view pointer, which is the form measured to work for loads.
    """
    buf = tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    view = tle.gpu.local_ptr(buf)
    scalar = tle.gpu.local_ptr(buf, (0,))
    bins = tl.arange(0, NB)
    tl.store(view, tl.zeros([NB], tl.int32))
    tl.debug_barrier()
    tl.atomic_add(scalar + tl.load(idx_ptr + bins), tl.full([NB], 1, tl.int32),
                  sem="relaxed", scope="cta")
    tl.debug_barrier()
    tl.store(out_ptr + bins, tl.load(view))


@triton.jit
def k_atomic_scatter_scalar_read(idx_ptr, out_ptr, NB: tl.constexpr):
    """Same scatter, read back the way that asserts -- to confirm the read is
    what breaks and the atomic is not."""
    buf = tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    scalar = tle.gpu.local_ptr(buf, (0,))
    bins = tl.arange(0, NB)
    tl.store(scalar + bins, tl.zeros([NB], tl.int32))
    tl.debug_barrier()
    tl.atomic_add(scalar + tl.load(idx_ptr + bins), tl.full([NB], 1, tl.int32),
                  sem="relaxed", scope="cta")
    tl.debug_barrier()
    tl.store(out_ptr + bins, tl.load(scalar + bins))


@triton.jit
def k_atomic_view_plain(out_ptr, NB: tl.constexpr):
    """The simplest possible atomic on shared memory: a full-view pointer, no
    arithmetic, no mask. If even this fails to verify then atomics on an
    address-space-3 pointer are rejected outright, and no indexing scheme
    rescues them."""
    buf = tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    view = tle.gpu.local_ptr(buf)
    bins = tl.arange(0, NB)
    tl.store(view, tl.zeros([NB], tl.int32))
    tl.debug_barrier()
    tl.atomic_add(view, tl.full([NB], 2, tl.int32))
    tl.debug_barrier()
    tl.store(out_ptr + bins, tl.load(view))


@triton.jit
def k_atomic_scalar_arange(out_ptr, NB: tl.constexpr):
    """Scalar pointer plus a contiguous range -- the same construction the
    operator uses, minus the data-dependent indices."""
    buf = tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    view = tle.gpu.local_ptr(buf)
    scalar = tle.gpu.local_ptr(buf, (0,))
    bins = tl.arange(0, NB)
    tl.store(view, tl.zeros([NB], tl.int32))
    tl.debug_barrier()
    tl.atomic_add(scalar + bins, tl.full([NB], 2, tl.int32))
    tl.debug_barrier()
    tl.store(out_ptr + bins, tl.load(view))


@triton.jit
def k_atomic_masked_operator_form(idx_ptr, out_ptr, NB: tl.constexpr):
    """Exactly what _distribute_to_bins writes, mask and all.

    The four unmasked cases all failed the tt.atomic_rmw verifier, and the
    generic operator uses this same construction on NVIDIA, where it verifies.
    A mask should not enter a ptr-vs-value type check -- but "should not" is
    the reason to test it rather than assert it.
    """
    buf = tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    view = tle.gpu.local_ptr(buf)
    scalar = tle.gpu.local_ptr(buf, (0,))
    bins = tl.arange(0, NB)
    ones = tl.full([NB], 1, tl.int32)
    tl.store(view, tl.zeros([NB], tl.int32))
    tl.debug_barrier()
    bin_idx = tl.load(idx_ptr + bins)
    in_range = bin_idx >= 0
    tl.atomic_add(scalar + bin_idx, ones, mask=in_range,
                  sem="relaxed", scope="cta")
    tl.debug_barrier()
    tl.store(out_ptr + bins, tl.load(view))


@triton.jit(do_not_specialize=["one"])
def k_load_opaque_stride(out_ptr, one, NB: tl.constexpr):
    """scalar + offs load where AxisInfo cannot prove contiguity: `one` is a
    runtime 1 (do_not_specialize, else Triton folds 1 into a constant)."""
    buf = tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    p = tle.gpu.local_ptr(buf, (0,))
    lane = tl.arange(0, NB)
    tl.store(tle.gpu.local_ptr(buf), lane * 2)
    tl.debug_barrier()
    tl.store(out_ptr + lane, tl.load(p + lane * one))


@triton.jit
def k_load_0d(out_ptr, NB: tl.constexpr):
    """A 0-d load through the scalar pointer, broadcast to the output --
    the form of every tl.load(s_found_topk_values_ptr) in the operator."""
    buf = tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    lane = tl.arange(0, NB)
    tl.store(tle.gpu.local_ptr(buf), lane * 2)
    tl.debug_barrier()
    v = tl.load(tle.gpu.local_ptr(buf, (0,)))          # element 0 == 0
    v1 = tl.load(tle.gpu.local_ptr(buf, (1,)))         # element 1 == 2
    tl.store(out_ptr + lane, lane * v1 + v)            # == lane*2 iff both right


@triton.jit
def k_gather_masked(idx_ptr, out_ptr, NB: tl.constexpr):
    """What _final_select_radix does: tl.load(s_histogram_ptr + pos, mask)."""
    buf = tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    scalar = tle.gpu.local_ptr(buf, (0,))
    lane = tl.arange(0, NB)
    tl.store(tle.gpu.local_ptr(buf), lane * 2)
    tl.debug_barrier()
    idx = tl.load(idx_ptr + lane)
    tl.store(out_ptr + lane, tl.load(scalar + idx, mask=idx >= 0, other=-1))


# case -> (kernel, NB, num_warps) for the load sweep; all expect lane*2
LOAD_SWEEP = {
    "load_nb256_w1": (k_scalar_roundtrip, 256, 1),
    "load_nb1024_w4": (k_scalar_roundtrip, 1024, 4),
    "load_nb2048_w8": (k_scalar_roundtrip, 2048, 8),
    "load_nb256_w8": (k_scalar_roundtrip, 256, 8),
    "load_nb512_w4": (k_scalar_roundtrip, 512, 4),
    "load_opaque_stride": (k_load_opaque_stride, 256, 4),
    "load_0d": (k_load_0d, 256, 4),
}

KERNELS = {
    "scalar_store": k_scalar_store,
    "scalar_roundtrip": k_scalar_roundtrip,
    "view_store": k_view_store,
    "view_roundtrip": k_view_roundtrip,
    "atomic_view_plain": k_atomic_view_plain,
    "atomic_scalar_arange": k_atomic_scalar_arange,
    "atomic_masked_operator_form": k_atomic_masked_operator_form,
    "atomic_scatter_view_read": k_atomic_scatter_view_read,
    "atomic_scatter_scalar_read": k_atomic_scatter_scalar_read,
}

print(f"--- {CASE} | triton {triton.__version__}", flush=True)
out = torch.zeros(NB, dtype=torch.int32, device="cuda")
if CASE in LOAD_SWEEP:
    kern, nb, nw = LOAD_SWEEP[CASE]
    out = torch.zeros(nb, dtype=torch.int32, device="cuda")
    if kern is k_load_opaque_stride:
        kern[(1,)](out, 1, NB=nb, num_warps=nw)
    else:
        kern[(1,)](out, NB=nb, num_warps=nw)
    torch.cuda.synchronize()
    ok = torch.equal(out, torch.arange(nb, dtype=torch.int32, device="cuda") * 2)
    extra = f" | NB={nb} warps={nw} elems/thread={nb / (nw * 64):g}"
elif CASE == "gather_masked":
    torch.manual_seed(0)
    idx = torch.randint(0, NB, (NB,), dtype=torch.int32, device="cuda")
    k_gather_masked[(1,)](idx, out, NB=NB)
    torch.cuda.synchronize()
    ok = torch.equal(out, idx * 2)
    extra = ""
elif CASE in ("atomic_view_plain", "atomic_scalar_arange"):
    KERNELS[CASE][(1,)](out, NB=NB)
    torch.cuda.synchronize()
    exp = torch.full((NB,), 2, dtype=torch.int32, device="cuda")
    ok = torch.equal(out, exp)
    extra = f" | first={out[:4].tolist()} expected all 2"
elif CASE.startswith("atomic_"):
    torch.manual_seed(0)
    idx = torch.randint(0, NB, (NB,), dtype=torch.int32, device="cuda")
    KERNELS[CASE][(1,)](idx, out, NB=NB)
    torch.cuda.synchronize()
    exp = torch.bincount(idx.cpu().long(), minlength=NB).to(torch.int32)
    ok = torch.equal(out.cpu(), exp)
    extra = f" | sum={int(out.sum())} expected {NB}"
else:
    KERNELS[CASE][(1,)](out, NB=NB)
    torch.cuda.synchronize()
    exp = torch.arange(NB, dtype=torch.int32, device="cuda") * 2
    ok = torch.equal(out, exp)
    extra = ""
print(f"RESULT {CASE}: {'CORRECT' if ok else 'WRONG ' + str(out[:6].tolist())}{extra}")
