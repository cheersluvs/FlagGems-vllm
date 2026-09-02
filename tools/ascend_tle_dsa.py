"""Can the Ascend DSA surface put the histogram on chip?

Probe 1 established that this build has a full buffer/memref DSA language with
an Ascend address space (dsa.ascend.UB), and that the earlier "no TLE" reading
was wrong twice over: the symbols are under dsa, not gpu, and my alloc call
failed only because it omitted the required mem_addr_space argument.

Three things decide whether the 2048-bin histogram can leave global memory:
  A. does alloc(shape, dtype, UB) lower at all
  B. what is the idiom for getting data in and out -- to_tensor / to_buffer /
     copy / subview are all exposed, but the semantics are not documented here
  C. how much UB is actually usable.  The measured ceiling on this card was
     ~36KB, not the nominal 192KB, and the histogram alone wants 8KB.

Sections run in that order so a hang in the capacity sweep -- one earlier
attempt on this card hung the AICore -- still leaves the rest in the report.
"""

import importlib
import os
import sys
import traceback

import torch
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except Exception:
    pass


def show(tag):
    print(f"  [{tag}] FAILED")
    sys.stdout.flush()
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    sys.stdout.flush()


def dump(path, first=None, start=1):
    if not os.path.exists(path):
        print(f"  (missing: {path})")
        return
    with open(path, errors="replace") as fh:
        lines = fh.readlines()
    end = len(lines) if first is None else min(len(lines), start - 1 + first)
    print(f"  {path}  ({len(lines)} lines, showing {start}..{end})")
    for i in range(start - 1, end):
        print(f"{i + 1:5d}| {lines[i].rstrip()}")
    sys.stdout.flush()


print(f"triton {triton.__version__} | torch {torch.__version__}")

# ------------------------------------------------------- A. the rest of core
rule("A. dsa/core.py from line 113, dsa/semantic.py, and the CANN address space")
try:
    import triton.experimental.tle as T

    PKG = os.path.dirname(T.__file__)
    dump(os.path.join(PKG, "language/dsa/core.py"), start=113)
    print()
    dump(os.path.join(PKG, "language/dsa/semantic.py"))
except Exception:
    show("core/semantic dump")

try:
    ext = importlib.import_module("triton.language.extra.cann.extension.core")
    print()
    dump(ext.__file__)
except Exception:
    show("cann extension dump")

# --------------------------------------------- B. is the operator importable
rule("B. does flaggems_vllm still import on this box?")
try:
    import flaggems_vllm  # noqa: F401

    print("  import flaggems_vllm OK")
    from flaggems_vllm.utils.triton_version_utils import has_triton_tle

    for v in ((0, 0, 0), (3, 2, 0), (3, 6, 0)):
        print(f"  has_triton_tle{v} = {has_triton_tle(*v)}")
    from flaggems_vllm import runtime

    info = runtime.device.info
    print(f"  vendor {getattr(info, 'name', '?')} "
          f"tle_enabled={getattr(info, 'tle_enabled', 'ABSENT')}")
except Exception:
    show("import flaggems_vllm")
    # The failure in probe 1 was an unrelated operator's Ascend autotuner
    # rejecting its config at import.  Whether that is new matters, because the
    # suite passed 25 tests on this same commit.
    for mod, attr in (("triton.backends.ascend.runtime.autotuner", "__file__"),):
        try:
            m = importlib.import_module(mod)
            p = getattr(m, attr)
            print(f"  {mod} at {p}")
            print(f"      mtime {os.path.getmtime(p)}")
        except Exception:
            show(f"locate {mod}")

# ------------------------------------------------------ C. does UB alloc work
rule("C. alloc/to_tensor/copy/subview against dsa.ascend.UB")
dsa = None
try:
    import triton.experimental.tle.language.dsa as dsa

    print(f"  address spaces: {[x for x in dir(dsa.ascend) if not x.startswith('_')]}")
    print(f"  UB = {dsa.ascend.UB!r}")
except Exception:
    show("dsa import")

if dsa is not None:
    B = 128

    @triton.jit
    def kA(out_ptr, BLOCK: tl.constexpr):
        buf = dsa.alloc([BLOCK], tl.int32, dsa.ascend.UB)
        t = dsa.to_tensor(buf)
        tl.store(out_ptr + tl.arange(0, BLOCK), t)

    @triton.jit
    def kB(in_ptr, out_ptr, BLOCK: tl.constexpr):
        lane = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + lane)
        buf = dsa.alloc([BLOCK], tl.int32, dsa.ascend.UB)
        src = dsa.to_buffer(x)
        dsa.copy(src, buf, [BLOCK])
        tl.store(out_ptr + lane, dsa.to_tensor(buf) * 2)

    @triton.jit
    def kC(in_ptr, out_ptr, BLOCK: tl.constexpr, HALF: tl.constexpr):
        lane = tl.arange(0, BLOCK)
        x = tl.load(in_ptr + lane)
        buf = dsa.alloc([BLOCK], tl.int32, dsa.ascend.UB)
        dsa.copy(dsa.to_buffer(x), buf, [BLOCK])
        sub = dsa.subview(buf, [0], [HALF], [1])
        tl.store(out_ptr + tl.arange(0, HALF), dsa.to_tensor(sub))

    src = torch.arange(B, dtype=torch.int32, device="npu")
    for tag, fn, args, kw in (
        ("A alloc+to_tensor", kA, (torch.zeros(B, dtype=torch.int32, device="npu"),),
         dict(BLOCK=B)),
        ("B to_buffer+copy", kB, (src, torch.zeros(B, dtype=torch.int32, device="npu")),
         dict(BLOCK=B)),
        ("C subview", kC, (src, torch.zeros(B, dtype=torch.int32, device="npu")),
         dict(BLOCK=B, HALF=B // 2)),
    ):
        try:
            fn[(1,)](*args, **kw)
            torch.npu.synchronize()
            out = args[-1]
            print(f"  [{tag}] compiled and ran | out[:8]={out[:8].tolist()}")
        except Exception:
            show(tag)

# ---------------------------------------------------------- D. how much UB
rule("D. usable UB capacity (int32 elements) -- LAST, it may hang")
if dsa is not None:

    @triton.jit
    def kCap(out_ptr, N: tl.constexpr):
        buf = dsa.alloc([N], tl.int32, dsa.ascend.UB)
        t = dsa.to_tensor(buf)
        tl.store(out_ptr + tl.arange(0, 128), tl.reshape(t, (N // 128, 128))[0, :])

    for n in (2048, 4096, 8192, 16384):
        kb = n * 4 // 1024
        try:
            out = torch.zeros(128, dtype=torch.int32, device="npu")
            kCap[(1,)](out, N=n)
            torch.npu.synchronize()
            print(f"  N={n:6d} ({kb:3d} KB) OK")
        except Exception:
            print(f"  N={n:6d} ({kb:3d} KB) failed:")
            show(f"cap {n}")
            break

print("\ndone.")
