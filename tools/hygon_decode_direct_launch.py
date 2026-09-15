"""How low can one-row decode's HOST cost go on Hygon?

hygon_launch_host_device.py: a one-row decode call spends ~108 us on the host
(Triton dispatch ~28 us, the 15-argument list another ~28, the operator's own
host code ~52: six scratch allocations and geometry lookups) and ~2 us on the
device when the kernel has nothing to do. At vocab 262144 the device's 322 us
hides that -- but it is exactly what makes splitting a row pay nothing in wall
time: split's device work models at ~85 us against 322, yet every extra launch
costs another ~110 us of serial host time.

So measure the host floor with the dispatch removed, in three levels, each in
host submit time (no synchronise) and in wall time at vocab 4096 and 262144:

    A  op        flaggems_vllm top_k_per_row_decode as shipped
    B  jit       scratch preallocated; the non-TLE kernel launched via JIT
    C  direct    scratch preallocated; the CompiledKernel cached and launched
                 directly: compiled[(1, 1, 1)](*args in signature order)

C is the recipe recorded for launch-bound ops on Triton 3.3-3.6. Its HIT path is
validated separately: the answer of a cached launch (not the first, compiling
one) is checked against torch.topk.

    tools/vendor_probe.sh tools/hygon_decode_direct_launch.py hygon_decode_direct_launch
"""

import sys
import time
from importlib import import_module

import torch

import flaggems_vllm  # noqa: F401  (runtime init)

K = 512


def submit_us(fn, iters=300, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / iters * 1e6


def wall_us(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000


def main():
    dec = import_module("flaggems_vllm.ops.top_k_per_row_decode")
    kernel = dec.non_tle_top_k_per_row_decode
    block = dec.NUM_THREADS_PER_BLOCK
    warps = dec._num_warps(block)
    print(f"geometry BLOCK_SIZE={block} num_warps={warps}\n")
    print(
        f"  {'vocab':>7} {'level':<8} {'submit us':>10} {'wall us':>9}  hit-path answer"
    )

    for v in (4096, 262144):
        torch.manual_seed(0)
        logits = torch.randn(1, v, dtype=torch.float32, device="cuda")
        lens = torch.full((1,), v, dtype=torch.int32, device="cuda")
        idx = torch.empty(1, K, dtype=torch.int32, device="cuda")
        want = torch.topk(logits, K, dim=1).values.sort(dim=1).values
        sc = [
            torch.empty(s, dtype=d, device="cuda")
            for s, d in (
                ((1, dec.NUM_BINS), torch.int32),
                ((1, dec.NUM_FILNAL_ITEMS), torch.float32),
                ((1,), torch.int32),
                ((1,), torch.int32),
                ((1,), torch.int32),
                ((1,), torch.int32),
            )
        ]
        args = (logits, idx, lens, 1, v, 1, v, *sc)

        def check():
            got = logits.gather(1, idx.long().clamp(0, v - 1)).sort(dim=1).values
            return "OK" if torch.allclose(got, want) else "WRONG"

        op = lambda: dec.top_k_per_row_decode(
            logits, 1, lens, idx, 1, v, 1, K
        )  # noqa: E731
        jit = lambda: kernel[(1,)](
            *args, TOPK=K, BLOCK_SIZE=block, num_warps=warps
        )  # noqa: E731

        compiled = kernel.run(
            *args, TOPK=K, BLOCK_SIZE=block, num_warps=warps, grid=(1,), warmup=False
        )
        direct = None
        if compiled is not None:
            runner = compiled[(1, 1, 1)]
            direct = lambda: runner(*args, K, block)  # noqa: E731

        for name, fn in (("A op", op), ("B jit", jit), ("C direct", direct)):
            if fn is None:
                print(
                    f"  {v:>7} {name:<8} {'-':>10} {'-':>9}  kernel.run returned None"
                )
                continue
            try:
                fn()
                torch.cuda.synchronize()
                idx.fill_(-3)
                fn()  # the HIT path: second launch, answer checked
                torch.cuda.synchronize()
                ans = check()
                s = submit_us(fn)
                w = wall_us(fn)
                print(f"  {v:>7} {name:<8} {s:>10.1f} {w:>9.1f}  {ans}")
            except Exception as e:  # noqa: BLE001
                print(f"  {v:>7} {name:<8} FAILED {type(e).__name__}: {str(e)[:110]}")
    print("\n  At 262144 wall stays at the device's ~322 us whatever the host does;")
    print("  what matters is the submit column, which is what every extra launch")
    print("  of a split decode would pay.")


if __name__ == "__main__":
    sys.exit(main())
