"""Is there a pointer into UB under tle.dsa, and can it be scattered into?

Discovery first, on purpose.  Checking `dsa/__init__.py`'s exports is not enough:
0.5.0 defined `from_buffer_to_tensor_pointer` in core.py without exporting it,
and it turned out to return a value tensor rather than a pointer despite the
name.  So enumerate every public name in every dsa submodule, grep the installed
sources for anything pointer-shaped, and only then try to build a kernel.

The operator's TLE path needs exactly one thing dsa has never offered: a pointer
it can `tl.atomic_add` into at a per-lane index.  Three of its seven atomics
scatter (`s_histogram_ptr + bin_idx`), four return the old value of a shared
counter.  alloc has a dsa equivalent and tle.cumsum has a plain-Triton one, so
the pointer is the whole question.
"""

import inspect
import os
import sys
import traceback
import types

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
# tle/__init__ imports dsa.ascend.communication, which imports `shmem`
# (Ascend SHMEM) -- absent here and unrelated to what we are testing.
sys.modules.setdefault("shmem", types.ModuleType("shmem"))

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl

CASE = sys.argv[1] if len(sys.argv) > 1 else "discover"
NBINS, BLOCK = 256, 128

try:
    import triton.experimental.tle.language.dsa as dsa
except Exception:
    dsa = None


def discover():
    import importlib

    if dsa is None:
        print("  tle.language.dsa 导入失败:")
        traceback.print_exc(file=sys.stdout)
        return

    WANT = ("ptr", "pointer", "local", "addr", "atomic", "scatter", "index")
    for name in ("triton.experimental.tle.language.dsa",
                 "triton.experimental.tle.language.dsa.core",
                 "triton.experimental.tle.language.dsa.ascend",
                 "triton.experimental.tle.language.dsa.semantic",
                 "triton.experimental.tle.language.dsa.types"):
        try:
            m = importlib.import_module(name)
        except Exception as e:
            print(f"  {name}: {type(e).__name__}: {e}")
            continue
        pub = sorted(a for a in dir(m) if not a.startswith("_"))
        hits = [a for a in pub if any(w in a.lower() for w in WANT)]
        print(f"\n  {name}  ({len(pub)} 个公开名)")
        print(f"      全部: {pub}")
        print(f"      指针相关: {hits or '无'}")
        for h in hits:
            o = getattr(m, h)
            try:
                print(f"          {h}{inspect.signature(getattr(o, 'fn', o))}")
            except Exception:
                pass

    print("\n  安装目录里 grep 指针相关符号:")
    root = os.path.dirname(dsa.__file__)
    import subprocess

    r = subprocess.run(
        ["grep", "-rn", "-E", "local_ptr|local_pointers|create_addptr|to_tensor_pointer",
         root], capture_output=True, text=True)
    print("      " + (r.stdout.strip().replace("\n", "\n      ") or "无匹配"))


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
    buf = dsa.alloc([NB], tl.int32, dsa.ascend.UB)
    p = dsa.local_ptr(buf, (0,))
    bins = tl.arange(0, NB)
    tl.store(p + bins, 0)
    tl.debug_barrier()
    tl.atomic_add(p + tl.load(idx_ptr + tl.arange(0, BLK)), 1)
    tl.debug_barrier()
    tl.store(out_ptr + bins, tl.load(p + bins))


def run():
    if CASE == "discover":
        discover()
        return "见上"

    if CASE == "store":
        out = torch.zeros(NBINS, dtype=torch.int32, device="npu")
        k_store[(1,)](out, NB=NBINS)
        torch.npu.synchronize()
        exp = torch.arange(NBINS, dtype=torch.int32, device="npu") * 2
        return f"dsa.local_ptr 读写 {'CORRECT' if torch.equal(out, exp) else 'WRONG ' + str(out[:8].tolist())}"

    if CASE == "atomic":
        idx = torch.randint(0, NBINS, (BLOCK,), dtype=torch.int32, device="npu")
        out = torch.zeros(NBINS, dtype=torch.int32, device="npu")
        k_atomic[(1,)](idx, out, BLK=BLOCK, NB=NBINS)
        torch.npu.synchronize()
        ref = torch.bincount(idx.cpu().long(), minlength=NBINS).to(torch.int32)
        ok = torch.equal(out.cpu(), ref)
        return (f"UB 散射累加 {'CORRECT' if ok else 'WRONG'} | sum={int(out.sum())}"
                f" 期望 {BLOCK}")

    return f"!! unknown case {CASE}"


print(f"--- dsa_ptr {CASE} | triton {triton.__version__} | dsa "
      f"{'可用' if dsa else '不可用'}")
sys.stdout.flush()
try:
    print(f"RESULT {CASE}: {run()}")
except Exception:
    print(f"RESULT {CASE}: FAILED")
    sys.stdout.flush()
    traceback.print_exc(file=sys.stdout)
sys.stdout.flush()
