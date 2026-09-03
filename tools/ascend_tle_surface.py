"""What TLE surface does the Ascend Triton build actually expose?

The generic operator asks for tle.gpu.alloc / gpu.local_ptr / gpu.smem /
tle.cumsum -- the NVIDIA and Moore Threads shared-memory surface.  This build
has none of those; it has triton.experimental.tle.language.dsa.alloc instead.

Two questions decide whether the histogram can live on chip:
  1. what does dsa.alloc take, and is there a way to address what it returns
  2. does a kernel using it compile and run, or is the symbol decorative
     (MetaX C550 is the second case for the gpu.* surface)

Everything is dumped: nothing here is truncated to a fixed width, because four
separate times on this operator a truncated error hid the actual cause.
"""

# torch_npu must be imported explicitly, before triton.  On FlagTree 0.6.1 the
# import graph is circular under torch's automatic backend loading: triton pulls
# in torch, torch auto-loads torch_npu, torch_npu re-enters triton, and triton
# dies with "cannot import name 'backends' from partially initialized module".
# TORCH_DEVICE_BACKEND_AUTOLOAD=0 (set by the verify script) disables the
# autoload; this import is what replaces it.
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass
try:
    import torch_musa  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl

import inspect
import sys
import traceback


def show(tag):
    print(f"  [{tag}] FAILED")
    sys.stdout.flush()
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


print(f"triton {triton.__version__} | torch {torch.__version__}")
print(f"python {sys.version.split()[0]}")

# ---------------------------------------------------------------- 1. sources
rule("1. every source file of the TLE package")
PKG = None
try:
    import triton.experimental.tle as T

    PKG = os.path.dirname(T.__file__)
    print(f"package at {PKG}\n")
    files = []
    for root, _dirs, names in os.walk(PKG):
        for n in sorted(names):
            if n.endswith(".py"):
                p = os.path.join(root, n)
                files.append((sum(1 for _ in open(p, errors="replace")), p))
    for n_lines, p in sorted(files):
        print(f"  {n_lines:6d}  {os.path.relpath(p, PKG)}")

    # The language surface is what the operator has to target, so print it
    # whole.  The rest is compiler plumbing: enough of it to see the shape.
    FULL = {"language/__init__.py", "language/dsa.py", "language/builder.py",
            "__init__.py"}
    for n_lines, p in sorted(files):
        rel = os.path.relpath(p, PKG)
        limit = None if rel in FULL else 120
        rule(f"1.{rel}  ({n_lines} lines"
             + (")" if limit is None else f", first {limit})"))
        with open(p, errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if limit and i > limit:
                    print(f"    ... {n_lines - limit} more lines")
                    break
                print(f"{i:5d}| {line.rstrip()}")
except Exception:
    show("sources")

# ------------------------------------------------------------ 2. signatures
rule("2. callable surface, with signatures")
for name in ("triton.experimental.tle",
             "triton.experimental.tle.language",
             "triton.experimental.tle.language.dsa",
             "triton.experimental.tle.language.builder",
             "triton.experimental.tle.dsa"):
    try:
        mod = __import__(name, fromlist=["*"])
    except Exception as e:
        print(f"\n  {name}: import failed {type(e).__name__}: {e}")
        continue
    print(f"\n  {name}")
    for a in sorted(x for x in dir(mod) if not x.startswith("_")):
        obj = getattr(mod, a)
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            sig = ""
        kind = type(obj).__name__
        print(f"      {a}{sig}   <{kind}>")
        doc = (inspect.getdoc(obj) or "").strip()
        if doc:
            for line in doc.splitlines()[:6]:
                print(f"          | {line}")

# ------------------------------------------------- 3. does dsa.alloc lower?
rule("3. does a kernel using dsa.alloc compile and run?")
try:
    import triton.experimental.tle.language.dsa as dsa

    print(f"  dsa.alloc signature: {inspect.signature(dsa.alloc)}")
    print(f"  dsa.alloc doc:\n{inspect.getdoc(dsa.alloc)}")
except Exception:
    show("dsa import")
    dsa = None

if dsa is not None:
    B = 128

    @triton.jit
    def k_alloc(out_ptr, BLOCK: tl.constexpr):
        buf = dsa.alloc((BLOCK,), tl.int32)
        lane = tl.arange(0, BLOCK)
        tl.store(buf + lane, lane * 2)
        tl.debug_barrier()
        tl.store(out_ptr + lane, tl.load(buf + lane))

    try:
        out = torch.zeros(B, dtype=torch.int32, device="npu")
        k_alloc[(1,)](out, BLOCK=B)
        torch.npu.synchronize()
        exp = torch.arange(B, dtype=torch.int32, device="npu") * 2
        ok = torch.equal(out, exp)
        print(f"  [alloc as pointer] compiled and ran, result "
              f"{'CORRECT' if ok else 'WRONG: ' + str(out[:8].tolist())}")
    except Exception:
        show("alloc as pointer")

# --------------------------------------------------- 4. what FlagGems sees
rule("4. the gates the operator actually evaluates")
try:
    from flaggems_vllm.utils.triton_version_utils import has_triton_tle

    for v in ((0, 0, 0), (3, 2, 0), (3, 6, 0)):
        print(f"  has_triton_tle{v} = {has_triton_tle(*v)}")
except Exception:
    show("has_triton_tle")

try:
    from flaggems_vllm import runtime

    info = runtime.device.info
    print(f"  vendor {getattr(info, 'name', '?')} "
          f"tle_enabled={getattr(info, 'tle_enabled', 'ABSENT')}")
except Exception:
    show("vendor descriptor")

try:
    import flaggems_vllm.ops.top_k_per_row_prefill as _m
    from importlib import import_module

    m = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
    print(f"  operator module HAS_TLE = {m.HAS_TLE}")
except Exception:
    show("operator HAS_TLE")

print("\ndone.")
