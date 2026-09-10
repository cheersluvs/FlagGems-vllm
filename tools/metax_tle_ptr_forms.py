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

CASES = ("scalar_store", "scalar_roundtrip", "view_store", "view_roundtrip")

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


KERNELS = {
    "scalar_store": k_scalar_store,
    "scalar_roundtrip": k_scalar_roundtrip,
    "view_store": k_view_store,
    "view_roundtrip": k_view_roundtrip,
}

print(f"--- {CASE} | triton {triton.__version__}", flush=True)
out = torch.zeros(NB, dtype=torch.int32, device="cuda")
KERNELS[CASE][(1,)](out, NB=NB)
torch.cuda.synchronize()
exp = torch.arange(NB, dtype=torch.int32, device="cuda") * 2
ok = torch.equal(out, exp)
print(f"RESULT {CASE}: {'CORRECT' if ok else 'WRONG ' + str(out[:6].tolist())}")
