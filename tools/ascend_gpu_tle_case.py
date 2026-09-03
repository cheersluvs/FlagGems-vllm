"""Does the tle.gpu shared-memory surface actually lower on Ascend?

The generic operator's TLE path needs exactly four things, in this order:
alloc a smem buffer, take a local_ptr into it, scatter into that pointer with
tl.atomic_add, and tle.cumsum the result.  The symbols exist on
0.6.0+ascend3.5 -- but MetaX C550 is the standing reminder that a symbol which
imports is not a symbol the backend can generate code for.

If all four lower, the histogram can live in shared memory and the whole
tl.histogram workaround in the Ascend override becomes unnecessary.
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
import triton.experimental.tle.language as tle

# TLE_PATCH=1 loads the TLE dialect into the Ascend context first, so the same
# kernels can be run with and without it and the difference attributed.
if os.environ.get("TLE_PATCH") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tle_dialect_patch

    print(f"--- tle dialect patch: {tle_dialect_patch.apply()}")

CASE = sys.argv[1]
NBINS = 256
BLOCK = 128


@triton.jit
def k_alloc(out_ptr, NB: tl.constexpr):
    buf = tle.gpu.alloc((NB,), tl.int32, scope=tle.gpu.smem)
    p = tle.gpu.local_ptr(buf, (0,))
    lane = tl.arange(0, NB)
    tl.store(p + lane, lane * 2)
    tl.debug_barrier()
    tl.store(out_ptr + lane, tl.load(p + lane))


@triton.jit
def k_atomic(idx_ptr, out_ptr, BLK: tl.constexpr, NB: tl.constexpr):
    """The histogram scatter the operator cannot express on UB."""
    buf = tle.gpu.alloc((NB,), tl.int32, scope=tle.gpu.smem)
    p = tle.gpu.local_ptr(buf, (0,))
    bins = tl.arange(0, NB)
    tl.store(p + bins, 0)
    tl.debug_barrier()
    tl.atomic_add(p + tl.load(idx_ptr + tl.arange(0, BLK)), 1)
    tl.debug_barrier()
    tl.store(out_ptr + bins, tl.load(p + bins))


@triton.jit
def k_cumsum(in_ptr, pre_ptr, tot_ptr, NB: tl.constexpr):
    bins = tl.arange(0, NB)
    prefix, total = tle.cumsum(tl.load(in_ptr + bins), axis=0, reverse=False)
    tl.store(pre_ptr + bins, prefix)
    tl.store(tot_ptr + bins, total)


def run():
    if CASE == "gpu_alloc":
        out = torch.zeros(NBINS, dtype=torch.int32, device="npu")
        k_alloc[(1,)](out, NB=NBINS)
        torch.npu.synchronize()
        exp = torch.arange(NBINS, dtype=torch.int32, device="npu") * 2
        return f"alloc+local_ptr lower, {'CORRECT' if torch.equal(out, exp) else 'WRONG ' + str(out[:8].tolist())}"

    if CASE == "gpu_atomic":
        idx = torch.randint(0, NBINS, (BLOCK,), dtype=torch.int32, device="npu")
        out = torch.zeros(NBINS, dtype=torch.int32, device="npu")
        k_atomic[(1,)](idx, out, BLK=BLOCK, NB=NBINS)
        torch.npu.synchronize()
        ref = torch.bincount(idx.cpu().long(), minlength=NBINS).to(torch.int32)
        ok = torch.equal(out.cpu(), ref)
        return (f"smem scatter {'CORRECT' if ok else 'WRONG'} | sum={int(out.sum())}"
                f" expected {BLOCK}")

    if CASE == "gpu_cumsum":
        src = torch.ones(NBINS, dtype=torch.int32, device="npu")
        pre = torch.zeros(NBINS, dtype=torch.int32, device="npu")
        tot = torch.zeros(NBINS, dtype=torch.int32, device="npu")
        k_cumsum[(1,)](src, pre, tot, NB=NBINS)
        torch.npu.synchronize()
        return (f"tle.cumsum lowers | prefix[:5]={pre[:5].tolist()}"
                f" total={int(tot[0])} (expected {NBINS})")

    return f"!! unknown case {CASE}"


print(f"--- {CASE} | triton {triton.__version__}")
sys.stdout.flush()
try:
    print(f"RESULT {CASE}: {run()}")
except Exception:
    print(f"RESULT {CASE}: FAILED")
    sys.stdout.flush()
    traceback.print_exc(file=sys.stdout)
sys.stdout.flush()
