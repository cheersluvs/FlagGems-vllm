"""Does prefill's compaction track how many elements it collects?

The MTT prefill override samples 1/8 of the row to estimate a deliberately
LOOSE threshold, so its single remaining pass collects several times top_k
instead of exactly top_k. Its whole case rests on one measured claim:

    "compaction barely tracks the trigger rate -- four times the hits cost 4%
     more -- so the atomics are per-lane issue overhead, not per-hit traffic"

If that held here, a looser threshold would be nearly free and trading it for a
whole histogram pass would pay. My atomics probe says it does NOT hold on
BW1000: the same kernel with only the threshold changed went 477 GB/s at 0.008
density to 115 at 0.048 -- six times the hits for 4.1 times the cost, per-hit
rather than per-issue.

But that was a REPLICA kernel, and replicas have now been wrong here twice:
256 bins beat 2048 in a replica and lost on the operator, and a replica's
histogram pass read 167 GB/s where the operator's reaches ~278. So the claim
gets tested on the operator itself.

No new kernel is needed. Raising top_k at a fixed shape lowers the threshold
and raises the collected count in proportion -- exactly the trigger rate in
question. If doubling top_k costs a few percent, the sampled design is
available here; if it costs something like double, it is not.

Round 1 answered nothing, because of two confounders I failed to control.

Raising top_k moves the override's OWN density routing: it sends a shape to
the prefix-sum allocator once vocab <= 10 * top_k, and that allocator issues
one atomic per TILE, so its cost cannot track the hits by construction. All
three points of (16383,4095) landed there -- flat to 1.3%, and meaningless for
this question -- while both four-row shapes switched modules mid-series.

That left (64,129280) as the only all-generic series, and it is the bimodal
shape: within one series 2048 -> 4096 cost 1.0x (per-issue) and 4096 -> 8192
cost 1.8x (per-hit), with 512 slower than 1024.

So: pin the module by turning the slot-scan copy off, and interleave k against
2k in one process over several rounds, which is the technique that has already
corrected two readings on this shape. Only the two shapes a sampled path could
serve are measured -- MTT's own crossover (MIN_SPAN 16384) puts the other five
out of reach regardless of the answer.

The final select also grows with top_k -- 10 us of 217 on MTT -- so the
numbers are reported rather than adjusted.

    tools/vendor_probe.sh tools/hygon_prefill_trigger.py hygon_prefill_trigger
"""

import sys

from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, profile

import flaggems_vllm

# (num_rows, vocab, stride0, top_k values) -- long rows first, since the MTT
# crossover (MIN_SPAN 16384) puts the short ones out of reach anyway
# (num_rows, vocab, stride0, (k, 2k) pairs) -- only the shapes at or above
# MTT's 16384 crossover, since no answer here helps the others
CASES = [
    (64, 129280, 129280, ((1024, 2048), (2048, 4096), (4096, 8192))),
    (4, 16385, 16648, ((512, 1024), (1024, 2048))),
]
ROUNDS = 4


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


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    ov._ENABLED = False  # pin the generic module: no prefix-sum allocator
    dev = "cuda"
    print(
        "slot-scan copy OFF, so every point uses the per-element atomic.\n"
        f"k against 2k, interleaved, {ROUNDS} rounds\n"
    )
    for rows, vocab, stride0, pairs in CASES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        print(f"  {rows} x {vocab}")
        for k1, k2 in pairs:
            calls = []
            ok = True
            for k in (k1, k2):
                idx = torch.empty((rows, k), dtype=torch.int32, device=dev)

                def call(k=k, idx=idx):
                    flaggems_vllm.top_k_per_row_prefill(
                        logits, starts, ends, idx, rows, stride0, 1, k
                    )

                idx.fill_(-9)
                call()
                torch.cuda.synchronize()
                want = torch.topk(logits, k, dim=1).values.sort(dim=1).values
                got = (
                    logits.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
                )
                ok = ok and torch.allclose(got, want) and bool((idx >= 0).all())
                calls.append(call)
            ratios = []
            for _ in range(ROUNDS):
                t1 = device_us(calls[0])
                t2 = device_us(calls[1])
                ratios.append((t1, t2))
            rs = sorted(t2 / t1 for t1, t2 in ratios)
            line = "  ".join(f"{t1:.0f}/{t2:.0f}" for t1, t2 in ratios)
            print(
                f"    top_k {k1:>5} -> {k2:<5} median {rs[len(rs) // 2]:>6.3f}, "
                f"spread {rs[0]:.3f}-{rs[-1]:.3f}  [{line}]  "
                f"{'OK' if ok else 'WRONG'}",
                flush=True,
            )
        print(flush=True)
    print(
        "  A median near 1.0 means doubling the hits is nearly free -- the MTT"
        "\n  reading, and the sampled design would be available for these two"
        "\n  shapes. Near 2.0 means the atomics are per-hit and it cannot pay."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
