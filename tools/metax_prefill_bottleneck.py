"""Where inside prefill does the time go?

A profiler cannot see inside a Triton kernel, and prefill is one kernel, so the
decomposition has to come from varying inputs and fitting.

From the benchmark shapes the per-program cost already fits
    12.56 us + 1.389 ns/element
to within 2% at vocab 1025, 4095, 4100 and 5115 -- so at vocab 1025 the fixed
part is 90% of the time and the 1025 elements are 10%. That fixed part is the
2048-bin histogram, the 2048-wide final buffer and the four-step scaffolding,
all sized for a 262144-element row.

But the production shape (64, 129280, top_k=1024) sits 91 us ABOVE that model,
and it is the only shape that is both top_k=1024 and past
RADIX_FINAL_PREFILL_VOCAB_THRESHOLD=65536, which switches the final selection
from insertion sort to a radix pass. Those two have to be separated before
anything is attributed to either.

    python tools/metax_prefill_bottleneck.py
"""

import sys

import torch

import flaggems_vllm

DEV = flaggems_vllm.device
SMS = int(
    getattr(torch.cuda.get_device_properties(0), "multi_processor_count", 104)
)  # C550 104, BW1000 80


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def per_program(rows, vocab, top_k):
    """Wall time divided by waves -- what one program costs."""
    lg = torch.randn(rows, vocab, device=DEV, dtype=torch.float32)
    st = torch.zeros(rows, dtype=torch.int32, device=DEV)
    en = torch.full((rows,), vocab, dtype=torch.int32, device=DEV)
    out = torch.empty((rows, top_k), dtype=torch.int32, device=DEV)
    t = timed(
        lambda: flaggems_vllm.top_k_per_row_prefill(
            lg, st, en, out, rows, lg.stride(0), lg.stride(1), top_k
        )
    )
    waves = rows / SMS
    return t, t * 1000 / waves


def profile_one(rows, vocab, top_k):
    from torch.profiler import ProfilerActivity, profile

    lg = torch.randn(rows, vocab, device=DEV, dtype=torch.float32)
    st = torch.zeros(rows, dtype=torch.int32, device=DEV)
    en = torch.full((rows,), vocab, dtype=torch.int32, device=DEV)
    out = torch.empty((rows, top_k), dtype=torch.int32, device=DEV)
    fn = lambda: flaggems_vllm.top_k_per_row_prefill(  # noqa: E731
        lg, st, en, out, rows, lg.stride(0), lg.stride(1), top_k
    )
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=acts) as prof:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    evs = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    evs.sort(key=lambda e: -e.self_device_time_total)
    tot = sum(e.self_device_time_total for e in evs) / 10 / 1000
    print(
        f"  ({rows},{vocab}) k={top_k}: device {tot:.4f} ms over "
        f"{sum(e.count for e in evs) / 10:.1f} launches"
    )
    for e in evs[:4]:
        print(
            f"      {e.self_device_time_total / 10 / 1000:8.4f} ms  "
            f"x{e.count / 10:4.1f}  {e.key[:52]}"
        )


def main():
    print(f"device {DEV} | {SMS} SMs\n")

    print("=" * 76)
    print("=== 1. is it really all one kernel?")
    print("=" * 76)
    for shape in ((4160, 1024, 512), (4160, 4096, 512), (208, 131072, 1024)):
        profile_one(*shape)
    print()

    print("=" * 76)
    print("=== 2. vocab sweep at top_k=512 -- fixed cost and the 65536 gate")
    print("=" * 76)
    print(
        f"  {'vocab':>8} {'rows':>6} {'ms':>9} {'us/prog':>9} {'model':>8} {'diff':>8}"
    )
    for vocab in (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072):
        rows = max(SMS * 2, min(4160, 2**26 // vocab))
        rows = (rows // SMS) * SMS or SMS
        t, pp = per_program(rows, vocab, 512)
        model = 12.56 + vocab * 0.001389
        print(
            f"  {vocab:>8} {rows:>6} {t:>9.4f} {pp:>9.2f} {model:>8.2f} "
            f"{pp - model:>+8.2f}"
        )
    print("  A jump at 65536 is the insertion-sort -> radix-final switch.")
    print()

    print("=" * 76)
    print("=== 3. top_k sweep -- how much of the production shape is k=1024?")
    print("=" * 76)
    print(f"  {'vocab':>8} {'top_k':>6} {'us/prog':>9}  vs k=512")
    for vocab in (4096, 32768, 131072):
        base = None
        for top_k in (128, 256, 512, 1024):
            rows = max(SMS * 2, min(4160, 2**26 // vocab))
            rows = (rows // SMS) * SMS or SMS
            _t, pp = per_program(rows, vocab, top_k)
            base = base if top_k != 512 else pp
            print(
                f"  {vocab:>8} {top_k:>6} {pp:>9.2f}"
                f"{'' if base is None or top_k == 512 else f'  {pp / base:+.2f}x'}"
            )
        print()


if __name__ == "__main__":
    sys.exit(main())
