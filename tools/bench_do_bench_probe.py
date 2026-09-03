"""Does triton.testing.do_bench time the whole callable, or just its first kernel?

`--mode kernel` in this repo's benchmark is nothing but do_bench (see
benchmark/performance_utils.py), so if do_bench under-counts, every kernel-mode
number is wrong -- and wrong in the direction that flatters whichever side
launches more kernels.  torch.topk launches several; our operator launches one
or two.  That is exactly the shape that would corrupt a ratio.

The test needs no knowledge of the internals: launch the SAME kernel 1, 2 and 4
times inside one callable.  Real elapsed time must scale with the count.  If
do_bench reports a flat number while manual timing scales, it is only timing the
first launch.
"""

import inspect
import sys
import time

import torch
import triton
import triton.language as tl

for m in ("torch_npu", "torch_musa"):
    try:
        __import__(m)
    except Exception:
        pass


def _device():
    for name in ("npu", "musa", "cuda"):
        mod = getattr(torch, name, None)
        if mod is not None and mod.is_available():
            return name, mod
    raise RuntimeError("no accelerator found")


DEV, DEVMOD = _device()
do_bench = (triton.musa_testing.do_bench if DEV == "musa"
            else triton.testing.do_bench)

print(f"device {DEV} | triton {triton.__version__}")
try:
    print(f"do_bench from {inspect.getsourcefile(do_bench)}")
    src = inspect.getsource(do_bench)
    print(f"--- do_bench source ({len(src.splitlines())} lines) ---")
    print(src)
except Exception as e:
    print(f"  (source unavailable: {type(e).__name__}: {e})")


@triton.jit
def k_busy(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    tl.store(out_ptr + off, tl.load(x_ptr + off, mask=m) * 2, mask=m)


N = 1 << 24
BLOCK = 1024
grid = (triton.cdiv(N, BLOCK),)
x = torch.randn(N, dtype=torch.float32, device=DEV)
out = torch.empty_like(x)


def make(k):
    def f():
        for _ in range(k):
            k_busy[grid](x, out, N, BLOCK=BLOCK)
    return f


def manual(f, reps=20):
    for _ in range(5):
        f()
    DEVMOD.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        DEVMOD.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


print("\n=== launches per callable vs reported time ===")
print(f"{'launches':>9} | {'manual ms':>10} | {'do_bench ms':>12} | {'manual/1x':>9} | {'bench/1x':>9}")
base_m = base_b = None
for k in (1, 2, 4, 8):
    f = make(k)
    m = manual(f)
    b = do_bench(f, warmup=25, rep=100, return_mode="median")
    if base_m is None:
        base_m, base_b = m, b
    print(f"{k:>9} | {m:>10.3f} | {b:>12.3f} | {m / base_m:>8.2f}x | {b / base_b:>8.2f}x")

print("\nIf manual scales ~linearly and do_bench stays flat, do_bench is timing")
print("only the first launch, and every --mode kernel number is unusable.")
