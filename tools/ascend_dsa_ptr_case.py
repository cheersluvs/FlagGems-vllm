"""Does tle.dsa.local_ptr give a real pointer into UB -- and can it be scattered into?

On 0.5.0+ascend3.2 the dsa namespace had no local_ptr at all: only value-semantic
alloc/copy/to_tensor/subview, which is why the histogram had to stay in global
memory.  0.6.0+ascend3.5 adds one.  `tle.gpu.local_ptr` is present too but its
dialect op is not registered in the Ascend MLIRContext and hard-aborts:

    LLVM ERROR: Building op `tle.local_pointers` but it isn't known in this
    MLIRContext

So the question is specifically whether the *dsa* one lowers where the *gpu* one
does not.  If tl.atomic_add through it works, the 2048-bin radix histogram moves
on chip and tl.histogram stops being the answer on Ascend.
"""

import os
import sys
import traceback

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl
import triton.experimental.tle.language.dsa as dsa

CASE = sys.argv[1]
NBINS = 256
BLOCK = 128
UB = dsa.ascend.UB


@triton.jit
def k_store(out_ptr, NB: tl.constexpr):
    buf = dsa.alloc([NB], tl.int32, dsa.ascend.UB)
    p = dsa.local_ptr(buf, (0,))
    lane = tl.arange(0, NB)
    tl.store(p + lane, lane * 2)
    tl.debug_barrier()
    tl.store(out_ptr + lane, tl.load(p + lane))


@triton.jit
def k_atomic(idx_ptr, out_ptr, BLK: tl.constexpr, NB: tl.constexpr):
    """The scatter the operator needs and UB could not express."""
    buf = dsa.alloc([NB], tl.int32, dsa.ascend.UB)
    p = dsa.local_ptr(buf, (0,))
    bins = tl.arange(0, NB)
    tl.store(p + bins, 0)
    tl.debug_barrier()
    tl.atomic_add(p + tl.load(idx_ptr + tl.arange(0, BLK)), 1)
    tl.debug_barrier()
    tl.store(out_ptr + bins, tl.load(p + bins))


def run():
    if CASE == "sig":
        import inspect
        names = sorted(a for a in dir(dsa) if not a.startswith("_"))
        out = [f"dsa exports {len(names)}: {names}"]
        for n in ("local_ptr", "alloc", "to_tensor", "copy"):
            o = getattr(dsa, n, None)
            if o is None:
                out.append(f"  {n}: ABSENT")
                continue
            try:
                sig = str(inspect.signature(o))
            except Exception:
                sig = "(?)"
            doc = (inspect.getdoc(o) or "").strip().splitlines()
            out.append(f"  {n}{sig}")
            for line in doc[:6]:
                out.append(f"      | {line}")
        return "\n".join(out)

    if CASE == "store":
        out = torch.zeros(NBINS, dtype=torch.int32, device="npu")
        k_store[(1,)](out, NB=NBINS)
        torch.npu.synchronize()
        exp = torch.arange(NBINS, dtype=torch.int32, device="npu") * 2
        ok = torch.equal(out, exp)
        return f"dsa.local_ptr store/load {'CORRECT' if ok else 'WRONG ' + str(out[:8].tolist())}"

    if CASE == "atomic":
        idx = torch.randint(0, NBINS, (BLOCK,), dtype=torch.int32, device="npu")
        out = torch.zeros(NBINS, dtype=torch.int32, device="npu")
        k_atomic[(1,)](idx, out, BLK=BLOCK, NB=NBINS)
        torch.npu.synchronize()
        ref = torch.bincount(idx.cpu().long(), minlength=NBINS).to(torch.int32)
        ok = torch.equal(out.cpu(), ref)
        return (f"UB scatter {'CORRECT' if ok else 'WRONG'} | sum={int(out.sum())}"
                f" expected {BLOCK}")

    return f"!! unknown case {CASE}"


print(f"--- dsa_ptr {CASE} | triton {triton.__version__}")
sys.stdout.flush()
try:
    print(f"RESULT {CASE}: {run()}")
except Exception:
    print(f"RESULT {CASE}: FAILED")
    sys.stdout.flush()
    traceback.print_exc(file=sys.stdout)
sys.stdout.flush()
