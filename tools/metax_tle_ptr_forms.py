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
         "atomic_scatter_view_read", "atomic_scatter_scalar_read")

if len(sys.argv) == 1:
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


KERNELS = {
    "scalar_store": k_scalar_store,
    "scalar_roundtrip": k_scalar_roundtrip,
    "view_store": k_view_store,
    "view_roundtrip": k_view_roundtrip,
    "atomic_scatter_view_read": k_atomic_scatter_view_read,
    "atomic_scatter_scalar_read": k_atomic_scatter_scalar_read,
}

print(f"--- {CASE} | triton {triton.__version__}", flush=True)
out = torch.zeros(NB, dtype=torch.int32, device="cuda")
if CASE.startswith("atomic_"):
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
