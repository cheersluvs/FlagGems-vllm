"""Would a sampled threshold pay on BW1000? The one number that decides it.

The trigger-rate measurement changed the picture. I had predicted the MTT
sampled design could not work here, on a replica's reading that atomics are
per-hit. On the operator, with the module pinned and k interleaved against 2k,
the cost of doubling the hits DEPENDS ON DENSITY:

    0.008 -> 0.016   1.117        (64,129280) sits at 0.008 with top_k 1024
    0.016 -> 0.032   1.318
    0.032 -> 0.063   1.815

So at the density that shape actually runs at, doubling the collected count
costs 12%, not 100%. The prediction was wrong and the design is back on the
table.

Whether it pays reduces to one unmeasured quantity. Write

    T(k) = H + C(k) + F(k)

with H the histogram pass, which does not depend on top_k. Sampling replaces H
with H/SSTRIDE and widens the collection to m * top_k:

    T_sampled(m) ~ T(m * k) - H * (1 - 1/SSTRIDE)

Measured already: T(1024) = 231, T(2048) = 259, T(4096) = 339 us. So at
m = 2 the design wins if 0.875 * H > 28 us, and at m = 4 if 0.875 * H > 108.

H is what is missing, and T(top_k = 1) is very nearly it: one element collected,
a final select over almost nothing, and the same full histogram pass. This
measures T over a range of top_k down to 1, takes H from the low end, and
prints the predicted sampled time for each (SSTRIDE, m) -- the go/no-go
number, before anything is built.

Round 1 of this probe got two things wrong.

On (64,129280) it came back non-monotonic -- T(2048) = 255.9 BELOW both
T(1024) = 287.5 and T(1) = 256.7 -- which the model forbids. Pinning the
module did not defeat that shape's bimodality because this probe, unlike the
trigger one, did not interleave. Its predicted speedups (9.2x, 18.8x) were
artifacts of H landing on top of T(2048), and are void.

And T(top_k=1) is NOT H. It is the histogram pass PLUS the fixed cost --
launch, per-program floor, final select over nothing -- and sampling shrinks
only the pass. On a four-row shape the fixed part is a large share of 25 us,
so discounting all of it by 1/SSTRIDE overstated the saving.

Both are fixed here. The per-element part is separated by measuring T(top_k=1)
twice on the same shape with row_end at vocab and at vocab/2: the difference,
doubled, is the pass, and what remains is fixed. And every quantity is
measured interleaved, so drift acts on all of them alike.

It is still a PREDICTION: it assumes the sample pass costs pass/SSTRIDE and
ignores the retry for rows whose estimate misses the window, which is what
sank MTT's first attempt at 0.383 against a generic 0.652. Below about 1.2x
there is no margin for that and it should not be built.

    tools/vendor_probe.sh tools/hygon_prefill_sample_payoff.py hygon_prefill_payoff
"""

import sys
from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, profile

import flaggems_vllm

# (num_rows, vocab, stride0, production top_k) -- the two shapes at or above
# MTT's 16384 crossover, the only ones a sampled path could serve
CASES = [
    (64, 129280, 129280, 1024),
    (4, 16385, 16648, 512),
]
SSTRIDES = (8, 16)
ROUNDS = 4
MULTIPLES = (2, 4, 8)


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
    ov._ENABLED = False  # pin the generic module, as the trigger probe did
    dev = "cuda"
    print(f"slot-scan copy OFF; every quantity interleaved, {ROUNDS} rounds\n")
    for rows, vocab, stride0, prod_k in CASES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        full = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        half = torch.full((rows,), vocab // 2, dtype=torch.int32, device=dev)

        def caller(k, ends):
            idx = torch.empty((rows, k), dtype=torch.int32, device=dev)

            def go():
                flaggems_vllm.top_k_per_row_prefill(
                    logits, starts, ends, idx, rows, stride0, 1, k
                )

            return go, idx

        # what we need: the operator today, the same at m*k, and T(1) at the
        # full range and at half of it to split the pass from the fixed cost
        probes = [("base", prod_k, full), ("one", 1, full), ("one_half", 1, half)]
        probes += [(f"m{m}", prod_k * m, full) for m in MULTIPLES]
        gos, idxs = {}, {}
        for name, k, ends in probes:
            gos[name], idxs[name] = caller(k, ends)

        ok = True
        for name, k, ends in probes:
            idxs[name].fill_(-9)
            gos[name]()
            torch.cuda.synchronize()
            n = int(ends[0])
            want = torch.topk(logits[:, :n], k, dim=1).values.sort(dim=1).values
            got = (
                logits[:, :n]
                .gather(1, idxs[name].long().clamp(0, n - 1))
                .sort(dim=1)
                .values
            )
            ok = ok and torch.allclose(got, want) and bool((idxs[name] >= 0).all())

        acc = {name: [] for name, _, _ in probes}
        for _ in range(ROUNDS):
            for name, _, _ in probes:
                acc[name].append(device_us(gos[name]))
        med = {n: sorted(v)[len(v) // 2] for n, v in acc.items()}

        pass_us = 2.0 * (med["one"] - med["one_half"])
        fixed_us = med["one"] - pass_us
        print(
            f"  {rows} x {vocab}, production top_k {prod_k}: "
            f"{'OK' if ok else 'WRONG'}"
        )
        for name, k, ends in probes:
            spread = max(acc[name]) / min(acc[name])
            print(
                f"    {name:>9} top_k {k:>5} range {int(ends[0]):>6}: "
                f"{med[name]:>8.1f} us  (spread {spread:.3f})"
            )
        print(
            f"    histogram pass {pass_us:>8.1f} us, fixed {fixed_us:>8.1f} us"
            f"  -- pass is {pass_us / med['base'] * 100:.0f}% of the operator"
        )
        if pass_us <= 0:
            print("    pass came out non-positive: the split did not hold\n")
            continue
        print(
            f"    {'sstride':>8} {'m':>3} {'at m*k':>9} {'predicted':>10} "
            f"{'speedup':>8}"
        )
        for ss in SSTRIDES:
            for m in MULTIPLES:
                pred = med[f"m{m}"] - pass_us * (1 - 1 / ss)
                print(
                    (
                        f"    {ss:>8} {m:>3} {med[f'm{m}']:>9.1f} {pred:>10.1f} "
                        f"{med['base'] / pred:>8.3f}"
                        if pred > 0
                        else f"    {ss:>8} {m:>3} {med[f'm{m}']:>9.1f} "
                        f"{'<=0':>10} {'-':>8}"
                    ),
                    flush=True,
                )
        print(flush=True)
    print(
        "  Still a prediction: the sample pass is assumed to cost pass/sstride"
        "\n  and the retry for rows whose estimate misses the window is not"
        "\n  modelled -- that is what sank MTT's first attempt. Under about"
        "\n  1.2x there is no margin for it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
