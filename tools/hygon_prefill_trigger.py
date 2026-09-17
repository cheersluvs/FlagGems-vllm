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

The confounder is the final select, whose work also grows with top_k. On MTT
it was 10 us of 217, so it is reported per shape rather than assumed away: a
cost that grows much faster than the final select can explain is the
compaction.

    tools/vendor_probe.sh tools/hygon_prefill_trigger.py hygon_prefill_trigger
"""

import sys

import torch
from torch.profiler import ProfilerActivity, profile

import flaggems_vllm

# (num_rows, vocab, stride0, top_k values) -- long rows first, since the MTT
# crossover (MIN_SPAN 16384) puts the short ones out of reach anyway
CASES = [
    (64, 129280, 129280, (512, 1024, 2048, 4096, 8192)),
    (4, 16385, 16648, (512, 1024, 2048, 4096)),
    (16383, 4095, 4352, (512, 1024, 2048)),
    (4, 8193, 8456, (512, 1024, 2048)),
]


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
    dev = "cuda"
    print(
        "device us against top_k at a fixed shape: raising top_k loosens the"
        "\nthreshold and raises the collected count in proportion\n"
    )
    for rows, vocab, stride0, topks in CASES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        print(f"  {rows} x {vocab}")
        print(
            f"    {'top_k':>6} {'density':>8} {'device us':>10} "
            f"{'vs top_k/2':>11} {'vs first':>9} {'ans':>5}"
        )
        base = None
        prev = None
        for top_k in topks:
            idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)

            def call(top_k=top_k, idx=idx):
                flaggems_vllm.top_k_per_row_prefill(
                    logits, starts, ends, idx, rows, stride0, 1, top_k
                )

            idx.fill_(-9)
            call()
            torch.cuda.synchronize()
            want = torch.topk(logits, top_k, dim=1).values.sort(dim=1).values
            got = logits.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
            ok = torch.allclose(got, want) and bool((idx >= 0).all())
            t = device_us(call)
            base = base if base is not None else t
            step = f"{t / prev:>11.3f}" if prev else f"{'-':>11}"
            print(
                f"    {top_k:>6} {top_k / vocab:>8.3f} {t:>10.1f} {step} "
                f"{t / base:>9.3f} {'OK' if ok else 'WRONG':>5}",
                flush=True,
            )
            prev = t
        print(flush=True)
    print(
        "  'vs top_k/2' near 1.0 means compaction does not track the hits --"
        "\n  the MTT reading, and a loose threshold would be nearly free here."
        "\n  Near 2.0 means it tracks them, and the sampled design cannot pay."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
