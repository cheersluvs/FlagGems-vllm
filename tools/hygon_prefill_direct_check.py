"""The prefill override's cached direct launch: correct, and what it saves.

This change is invisible to the acceptance metric. `--mode kernel` times the
kernel, and the kernel is untouched -- what moves is the HOST cost of reaching
it, measured at ~125 us per call on the small shapes (the operator's own
dispatch read 155 us against 27 us of device work). So this checks two things
the benchmark cannot:

  correctness on the CACHED path. The functional tests call each shape once,
  which only exercises the JIT launch that compiles; every later call of the
  same shape goes through a cached CompiledKernel, a different path the tests
  never reach. Each shape is called three times with FRESH inputs, including
  a partial row range and a row shorter than top_k, and every call is checked
  against torch.topk.

  the saving, as wall time per call with the direct path on and off, next to
  device time both ways -- which must not move, since it is the same kernel at
  the same geometry.

    tools/vendor_probe.sh tools/hygon_prefill_direct_check.py hygon_prefill_direct
"""

import sys
from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, profile

import flaggems_vllm

# (num_rows, vocab, top_k, stride0) -- the benchmark's shapes
SHAPES = [
    (64, 129280, 1024, 129280),
    (4, 8193, 512, 8456),
    (4, 16385, 512, 16648),
    (16383, 4095, 512, 4352),
    (12961, 4100, 512, 4360),
    (16380, 5115, 512, 5376),
    (4100, 1025, 512, 1288),
]


def wall_us(fn, iters=30, warmup=10):
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


def device_us(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = 0.0
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0)
        total += t or 0.0
    return total / iters


def check(logits, starts, ends, idx, top_k):
    bad = 0
    for r in range(logits.shape[0]):
        s, e = int(starts[r]), int(ends[r])
        n = e - s
        kk = min(top_k, n)
        want = torch.topk(logits[r, s:e], kk).values.sort().values
        sel = idx[r, :top_k]
        live = sel[(sel >= 0) & (sel < n)].long()
        got = logits[r, s + live].sort().values
        pad_ok = int((sel < 0).sum()) == top_k - kk
        if not (live.numel() == kk and torch.equal(got, want) and pad_ok):
            bad += 1
    return bad


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    dev = "cuda"
    print(f"direct launch enabled: {ov._DIRECT}\n")
    print(f"  {'shape':>18} {'range':<9} {'call1':>6} {'call2':>6} {'call3':>6}")
    total_bad = 0
    for rows, vocab, top_k, stride0 in SHAPES:
        for label in ("full", "partial", "short"):
            cells = []
            for call in range(3):
                torch.manual_seed(1000 * rows + call)
                buf = torch.randn(
                    (rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32
                )
                logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
                starts = torch.zeros(rows, dtype=torch.int32, device=dev)
                ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
                if label == "partial":
                    starts += 7
                    ends -= torch.arange(rows, dtype=torch.int32, device=dev) % 11
                elif label == "short":
                    ends = starts + min(vocab, max(1, top_k - 3))
                idx = torch.full((rows, top_k), -9, dtype=torch.int32, device=dev)
                flaggems_vllm.top_k_per_row_prefill(
                    logits, starts, ends, idx, rows, stride0, 1, top_k
                )
                torch.cuda.synchronize()
                bad = check(logits, starts, ends, idx, top_k)
                total_bad += bad
                cells.append(" ok " if bad == 0 else f"{bad:>3}!")
            print(
                f"  {f'{rows}x{vocab}':>18} {label:<9} " + "  ".join(cells),
                flush=True,
            )
    print(f"\n  plans cached: {len(ov._PLANS)}")
    print(f"  {'ALL CORRECT' if total_bad == 0 else f'{total_bad} WRONG ROWS'}\n")

    print(
        f"  {'shape':>18} {'wall on':>9} {'wall off':>9} {'saved':>8} "
        f"{'dev on':>9} {'dev off':>9}"
    )
    for rows, vocab, top_k, stride0 in SHAPES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)

        def call():
            flaggems_vllm.top_k_per_row_prefill(
                logits, starts, ends, idx, rows, stride0, 1, top_k
            )

        ov._DIRECT = True
        w_on, d_on = wall_us(call), device_us(call)
        ov._DIRECT = False
        w_off, d_off = wall_us(call), device_us(call)
        ov._DIRECT = True
        print(
            f"  {f'{rows}x{vocab}':>18} {w_on:>9.1f} {w_off:>9.1f} "
            f"{w_off - w_on:>8.1f} {d_on:>9.1f} {d_off:>9.1f}",
            flush=True,
        )
    print(
        "\n  'saved' is host time per call, which --mode kernel cannot see."
        "\n  The device columns are the same kernel at the same geometry and"
        "\n  should not move; if they do, something else changed."
    )
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
