#!/usr/bin/env python3
"""Measure where prefill's time actually goes, on ANY backend.

Two hypotheses are on the table after multi-block failed:

  H1  the kernel has a large per-program fixed cost (2048-bin histogram, threshold
      scan, final radix), which dominates at small vocab
  H2  vLLM wins the small shapes because ITS fixed cost is lower, not because it
      moves data faster

Both are testable without writing a single line of kernel code. Sweep vocab_size
at fixed num_rows and fit

      t(N) = a + b * N

`a` is the per-launch fixed cost, `b` the per-element cost. Then the ceiling on
any fixed-cost optimisation is exactly (a_gems - a_vllm): even a perfect fix
cannot beat that, and if it is small the whole line of work is not worth starting.

Sweep 2 varies num_rows at fixed vocab against the device's SM count, since
grid=(num_rows,) means a small num_rows cannot fill the machine.

Run this on NVIDIA to settle whether the 1.72x per-element gap measured on MTT is
a property of this operator versus vLLM's CUDA kernel, or something MUSA-specific.
If NVIDIA shows a similar per-element ratio, it is an operator-level issue and
not a vendor one.

    PYTHONPATH=src python tools/topk_prefill_cost.py
    VLLM_PLUGINS=musa PYTHONPATH=src python tools/topk_prefill_cost.py   # MTT

Reports numbers only. It changes nothing and proposes nothing.
"""

import os
import sys

import torch
import triton

import flaggems_vllm

DEV = flaggems_vllm.device


def _sm_count():
    try:
        p = flaggems_vllm.runtime.torch_device_fn.get_device_properties(0)
        return getattr(p, "multi_processor_count", 0) or 60
    except Exception:  # noqa: BLE001
        return 60


SM = _sm_count()

HAS_VLLM = False
try:
    import vllm._custom_ops  # noqa: F401 - loads torch.ops._C

    if hasattr(torch.ops._C, "top_k_per_row_prefill"):
        HAS_VLLM = True
except (ImportError, AttributeError, RuntimeError):
    pass


def _bench(fn):
    """Median kernel time in ms. do_bench handles warmup and repeat itself."""
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


def _make(num_rows, vocab, top_k):
    torch.manual_seed(42)
    logits = torch.randn((num_rows, vocab), dtype=torch.float32, device=DEV)
    starts = torch.zeros((num_rows,), dtype=torch.int32, device=DEV)
    ends = torch.full((num_rows,), vocab, dtype=torch.int32, device=DEV)
    idx = torch.empty((num_rows, top_k), dtype=torch.int32, device=DEV)
    return logits, starts, ends, idx


def _time_both(num_rows, vocab, top_k):
    logits, starts, ends, idx = _make(num_rows, vocab, top_k)
    s0, s1 = logits.stride(0), logits.stride(1)

    g = _bench(
        lambda: flaggems_vllm.top_k_per_row_prefill(
            logits, starts, ends, idx, num_rows, s0, s1, top_k
        )
    )
    v = None
    if HAS_VLLM:
        v = _bench(
            lambda: torch.ops._C.top_k_per_row_prefill(
                logits, starts, ends, idx, num_rows, s0, s1, top_k
            )
        )
    return g, v


def _fit(xs, ys):
    """Least-squares y = a + b x. Returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return my - b * mx, b


def sweep_vocab(num_rows, top_k):
    print("\n" + "=" * 78)
    print(f"  SWEEP 1: vocab at num_rows={num_rows} (fixed vs per-element)")
    print("=" * 78)
    vocabs = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    print(f"  {'vocab':>8}{'gems ms':>11}{'vLLM ms':>11}{'ratio':>8}")
    gx, gy, vy = [], [], []
    for v in vocabs:
        if v < top_k:
            continue
        g, vl = _time_both(num_rows, v, top_k)
        gx.append(v)
        gy.append(g)
        if vl is not None:
            vy.append(vl)
        r = f"{vl/g:>8.3f}" if vl else "     n/a"
        print(f"  {v:>8}{g:>11.5f}{(vl if vl else float('nan')):>11.5f}{r}")

    ga, gb = _fit(gx, gy)
    print(f"\n  gems : fixed {ga*1000:8.2f} us   per-elem {gb*1e6:.5f} us/K")
    if vy:
        va, vb = _fit(gx, vy)
        print(f"  vLLM : fixed {va*1000:8.2f} us   per-elem {vb*1e6:.5f} us/K")
        print(f"\n  --> fixed-cost gap = {(ga-va)*1000:.2f} us per launch")
        print("      That gap is the CEILING on any fixed-cost optimisation.")
        for v in (8192, 16384):
            if v >= top_k:
                tot = ga + gb * v
                print(
                    f"      vocab={v:>6}: fixed = {ga/tot*100:5.1f}% of gems time"
                )


def sweep_rows(vocab, top_k):
    print("\n" + "=" * 78)
    print(f"  SWEEP 2: num_rows at vocab={vocab} (wave quantisation, {SM} SMs)")
    print("=" * 78)
    print(f"  {'rows':>7}{'waves':>8}{'gems ms':>11}{'vLLM ms':>11}{'ratio':>8}{'us/row':>9}")
    for r in (4, 16, 30, 60, 61, 64, 120, 121, 240):
        g, vl = _time_both(r, vocab, top_k)
        rat = f"{vl/g:>8.3f}" if vl else "     n/a"
        print(
            f"  {r:>7}{r/SM:>8.2f}{g:>11.5f}"
            f"{(vl if vl else float('nan')):>11.5f}{rat}{g*1000/r:>9.2f}"
        )
    print("\n  A jump between 60->61 and 120->121 is wave quantisation, not algorithm.")


def sweep_distribution(num_rows, vocab, top_k):
    """Does the refinement depth drive the per-element cost?

    The main path re-scans the whole row inside `tl.static_range(0, 4)`, guarded
    by `continue_to_next_step` -- so it makes 1 to 4 passes depending on how hard
    the data is to separate. If pass count is what makes us 1.72x slower per
    element than vLLM, then feeding distributions that need different refinement
    depths must move OUR time a lot and vLLM's comparatively little.

    If all three land within a few percent, pass count is NOT the lever and the
    gap is plain scan throughput.
    """
    print("\n" + "=" * 78)
    print(f"  SWEEP 3: data distribution at ({num_rows}, {vocab}), top_k={top_k}")
    print("=" * 78)
    torch.manual_seed(7)
    starts = torch.zeros((num_rows,), dtype=torch.int32, device=DEV)
    ends = torch.full((num_rows,), vocab, dtype=torch.int32, device=DEV)
    idx = torch.empty((num_rows, top_k), dtype=torch.int32, device=DEV)

    cases = {}
    # 1. well spread -- should separate in few steps
    cases["randn (spread)"] = torch.randn(
        (num_rows, vocab), dtype=torch.float32, device=DEV
    )
    # 2. only the low 8 mantissa bits differ -> 256 distinct values, maximum
    #    refinement depth (this is the test_logits_diff_in_8LSBits distribution)
    bits = torch.randint(
        0, 2**8, (num_rows, vocab), dtype=torch.int32, device=DEV
    )
    cases["8-LSB ties (hard)"] = (0x3F900000 | (bits & 0xFF)).view(torch.float32)
    # 3. a handful of distinct values -> extreme ties
    q = torch.randint(0, 4, (num_rows, vocab), dtype=torch.int32, device=DEV)
    cases["4 distinct (extreme)"] = q.to(torch.float32)

    print(f"  {'distribution':>22}{'gems ms':>11}{'vLLM ms':>11}{'ratio':>8}")
    base = None
    for name, logits in cases.items():
        s0, s1 = logits.stride(0), logits.stride(1)
        g = _bench(
            lambda lg=logits, a=s0, b=s1: flaggems_vllm.top_k_per_row_prefill(
                lg, starts, ends, idx, num_rows, a, b, top_k
            )
        )
        v = float("nan")
        if HAS_VLLM:
            v = _bench(
                lambda lg=logits, a=s0, b=s1: torch.ops._C.top_k_per_row_prefill(
                    lg, starts, ends, idx, num_rows, a, b, top_k
                )
            )
        rel = "" if base is None else f"   ({g/base:+.2f}x vs randn)"
        if base is None:
            base = g
        print(f"  {name:>22}{g:>11.5f}{v:>11.5f}{v/g:>8.3f}{rel}")
    print("\n  Big spread across rows -> refinement depth is the lever.")
    print("  All within a few percent -> it is plain scan throughput, not passes.")


def main():
    print("=" * 78)
    print("  MTT prefill fixed-cost probe -- measurement only")
    print("=" * 78)
    print(f"  device   : {DEV}")
    print(f"  HAS_VLLM : {HAS_VLLM}")
    if not HAS_VLLM:
        print("  !! no vLLM baseline; run with VLLM_PLUGINS=musa")

    top_k = int(os.environ.get("PROBE_TOPK", "512"))
    print(f"  top_k    : {top_k}")

    sweep_vocab(num_rows=SM, top_k=top_k)
    sweep_rows(vocab=129280, top_k=1024)
    sweep_distribution(num_rows=60, vocab=131072, top_k=512)


if __name__ == "__main__":
    sys.exit(main())
