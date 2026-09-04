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
# Ascend ships its own do_bench_npu, and FlagGems switched the ascend
# benchmark over to it (FlagGems#4857, #5451) -- but FlagGems-vllm's
# performance_utils.py still special-cases only musa, so --mode kernel on
# Ascend uses the generic do_bench.  Comparing the two is the point.
do_bench_npu = None
if DEV == "npu":
    try:
        from triton.backends.ascend.testing import do_bench_npu
    except Exception as e:
        print(f"  (do_bench_npu unavailable: {type(e).__name__}: {e})")

if DEV == "musa":
    try:
        import triton.musa_testing  # noqa: F401  (attribute only exists once imported)
        do_bench = triton.musa_testing.do_bench
    except Exception:
        do_bench = triton.testing.do_bench
else:
    do_bench = triton.testing.do_bench

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
print(f"{'launches':>9} | {'manual ms':>10} | {'do_bench ms':>12} | "
      f"{'do_bench_npu':>12} | {'manual/1x':>9} | {'bench/1x':>9} | {'npu/1x':>7}")
base_m = base_b = base_n = None
for k in (1, 2, 4, 8):
    f = make(k)
    m = manual(f)
    b = do_bench(f, warmup=25, rep=100, return_mode="median")
    n = None
    if do_bench_npu is not None:
        try:
            n = do_bench_npu(f, warmup=25, rep=100, return_mode="median")
        except TypeError:
            try:
                n = do_bench_npu(f)
            except Exception as e:
                print(f"    (do_bench_npu failed: {type(e).__name__}: {e})")
    if base_m is None:
        base_m, base_b, base_n = m, b, n
    fn_ = lambda v: f"{v:12.3f}" if isinstance(v, float) else f"{'-':>12}"
    rn = f"{n / base_n:6.2f}x" if isinstance(n, float) and base_n else f"{'-':>7}"
    print(f"{k:>9} | {m:>10.3f} | {b:>12.3f} | {fn_(n)} | "
          f"{m / base_m:>8.2f}x | {b / base_b:>8.2f}x | {rn}")

# The claim is about the BASELINE's first kernel, and a baseline launches
# several DIFFERENT kernels.  Repeating one kernel would not expose an
# attribution bug that keys on the kernel itself, so compare like for like:
# four distinct kernels once each, against one kernel four times.


@triton.jit
def k_a(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    tl.store(out_ptr + off, tl.load(x_ptr + off, mask=m) * 2, mask=m)


@triton.jit
def k_b(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    tl.store(out_ptr + off, tl.load(x_ptr + off, mask=m) * 3, mask=m)


@triton.jit
def k_c(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    tl.store(out_ptr + off, tl.load(x_ptr + off, mask=m) * 4, mask=m)


@triton.jit
def k_d(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    tl.store(out_ptr + off, tl.load(x_ptr + off, mask=m) * 5, mask=m)


def hetero():
    for kf in (k_a, k_b, k_c, k_d):
        kf[grid](x, out, N, BLOCK=BLOCK)


print("\n=== four DIFFERENT kernels vs the same kernel four times ===")
hm, hb = manual(hetero), do_bench(hetero, warmup=25, rep=100, return_mode="median")
sm, sb = manual(make(4)), do_bench(make(4), warmup=25, rep=100, return_mode="median")
print(f"  4 distinct kernels : manual {hm:8.3f} ms | do_bench {hb:8.3f} ms")
print(f"  1 kernel x4        : manual {sm:8.3f} ms | do_bench {sb:8.3f} ms")
print(f"  do_bench/manual    : distinct {hb / hm:.2f}  same {sb / sm:.2f}"
      "   (a much smaller ratio for distinct kernels = attribution bug)")

print("\n=== the actual baseline: torch.topk ===")
for rows, vocab in ((64, 131072), (4, 8192)):
    t = torch.randn(rows, vocab, dtype=torch.float32, device=DEV)

    def tk():
        torch.topk(t, 64, dim=-1)

    tm, tb = manual(tk), do_bench(tk, warmup=25, rep=100, return_mode="median")
    print(f"  torch.topk {rows}x{vocab}: manual {tm:8.3f} ms | do_bench {tb:8.3f} ms"
          f" | ratio {tb / tm:.2f}")

print("\nManual includes ~0.5 ms of launch+sync per call, so do_bench reading a")
print("little lower is expected.  A do_bench/manual ratio far below the others")
print("is what would betray a baseline being under-counted.")
