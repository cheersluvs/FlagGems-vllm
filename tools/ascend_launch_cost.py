"""What does a program cost before it does any work, and where is the wave edge?

Two questions decide whether a persistent rewrite of top_k_per_row is worth it:

  1. Launch cost per program.  The decode timings imply ~2.6 us (1 row 46.1 ms
     vs 40 rows 46.2 ms), which over 16383 rows would be ~42 ms against a 327 ms
     operator -- 13%, worth having if real.  Measure it directly instead of
     inferring it from an operator that also does work.

  2. The concurrency width.  The same timings step at 40 (496 and 512 rows both
     take 600.4 ms = 13 x 46.2), so 40 programs appear to run at once.  A sweep
     of an empty kernel should show the same staircase, and its step is the
     number to use for a persistent grid.

Both are measured with an empty kernel, so the answer is launch and scheduling
only.  A second kernel does a trivial store, to show the cost is not being
optimised away.
"""

import os
import sys
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl


def _device():
    for name in ("npu", "musa", "cuda"):
        mod = getattr(torch, name, None)
        if mod is not None and mod.is_available():
            return name, mod
    raise RuntimeError("no accelerator found")


DEV, DEVMOD = _device()


@triton.jit
def k_empty(out_ptr):
    pass


@triton.jit
def k_touch(out_ptr):
    tl.store(out_ptr + tl.program_id(0), tl.program_id(0))


def timeit(fn, grid, arg, reps=20, warm=5):
    for _ in range(warm):
        fn[(grid,)](arg)
    DEVMOD.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn[(grid,)](arg)
        DEVMOD.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


GRIDS = [1, 8, 16, 24, 32, 40, 41, 48, 56, 64, 80, 128, 256, 512,
         1024, 2048, 4096, 8192, 16384]

print(f"device {DEV} | triton {triton.__version__}")
big = torch.zeros(max(GRIDS), dtype=torch.int32, device=DEV)

for name, fn in (("empty", k_empty), ("store 1 int", k_touch)):
    print(f"\n=== {name} kernel ===")
    print(f"{'grid':>7} | {'median ms':>10} {'min':>8} {'max':>8} | {'us/program':>11} | 相对 grid=1")
    base = None
    for g in GRIDS:
        med, lo, hi = timeit(fn, g, big)
        if base is None:
            base = med
        per = (med - base) / g * 1e3 if g else 0.0
        print(f"{g:>7} | {med:>10.4f} {lo:>8.4f} {hi:>8.4f} | {per:>11.3f} | {med / base:>6.2f}x")

print("\n阶梯若出现在 40/41 之间，即为并发宽度；us/program 是 persistent 化能省下的上限。")
print("对照：top_k_per_row prefill (16383, 4095) 现为 327 ms，decode 单波 46.2 ms。")
