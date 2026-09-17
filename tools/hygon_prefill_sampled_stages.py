"""The sampled prefill path shipped and did nothing. Which stage, and why?

Predicted 1.73x on (64,129280); the benchmark moved it from ~0.258 ms to
0.243-0.255, i.e. 2-5%. Two explanations need separating before anything is
changed:

  the estimate misses     every row whose collected count leaves
                          [top_k, CAP] is redone from a full histogram inside
                          _s_finish, which costs more than the pass the sample
                          saved. If that fires often the design is sound and
                          the parameters are wrong.
  a stage is not what     the model said prepare ~24 us (1/8 of a 189 us
  the model says          pass), collect ~55, finish ~36. If one of them is
                          several times that, the parameters are fine and the
                          implementation is not.

So: read the collected count per row after a real call and report how many
rows fall outside the window, then time the three launches separately against
the generic path with the sampled gate turned off.

Reaches into the override's plan cache rather than rebuilding the pipeline, so
what is measured is what ships.

    tools/vendor_probe.sh tools/hygon_prefill_sampled_stages.py hygon_prefill_sampled_stages
"""

import sys
from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, profile

import flaggems_vllm

ROWS, VOCAB, TOPK, STRIDE0 = 64, 129280, 1024, 129280


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
    dev = "cuda"
    torch.manual_seed(42)
    buf = torch.randn((ROWS - 1) * STRIDE0 + VOCAB, device=dev, dtype=torch.float32)
    logits = torch.as_strided(buf, (ROWS, VOCAB), (STRIDE0, 1))
    starts = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    ends = torch.full((ROWS,), VOCAB, dtype=torch.int32, device=dev)
    idx = torch.empty((ROWS, TOPK), dtype=torch.int32, device=dev)
    want = torch.topk(logits, TOPK, dim=1).values.sort(dim=1).values

    def call():
        flaggems_vllm.top_k_per_row_prefill(
            logits, starts, ends, idx, ROWS, STRIDE0, 1, TOPK
        )

    print(
        f"{ROWS} x {VOCAB}, top_k {TOPK}; gate ratio "
        f"{ov.SAMPLED_MIN_VOCAB_PER_TOPK}, sstride {ov.SSTRIDE}, "
        f"target {TOPK * ov.TARGET_MULT}\n"
    )
    took = ov._can_sample(logits, starts, ends, ROWS, STRIDE0, 1, TOPK)
    print(f"  _can_sample says: {took}")
    if not took:
        print("  the shipped path is NOT the sampled one; nothing else to say")
        return 1

    idx.fill_(-9)
    call()
    torch.cuda.synchronize()
    got = logits.gather(1, idx.long().clamp(0, VOCAB - 1)).sort(dim=1).values
    ok = torch.allclose(got, want) and bool((idx >= 0).all())
    plan = next(iter(ov._SPLANS.values()))
    cap = plan.cap
    # cnt is left holding what the LAST stage saw: after _s_finish a redone row
    # holds its exact count, an accepted row its sampled one.
    cnt = plan.cnt.float()
    outside = int(((plan.cnt < TOPK) | (plan.cnt > cap)).sum())
    print(
        f"  answer {'OK' if ok else 'WRONG'}; buffer {cap}, window " f"[{TOPK}, {cap}]"
    )
    print(
        f"  collected per row: min {cnt.min().item():.0f}, median "
        f"{cnt.median().item():.0f}, max {cnt.max().item():.0f}; "
        f"{outside} of {ROWS} rows outside the window"
    )

    print("\n  stage device us (each launched alone):")
    stages = (
        (
            "prepare",
            lambda: plan.prepare(
                logits, starts, ends, plan.hist, plan.thr, plan.cnt, STRIDE0
            ),
        ),
        (
            "collect",
            lambda: plan.collect(
                logits,
                starts,
                ends,
                plan.thr,
                plan.cnt,
                plan.cand_idx,
                plan.cand_val,
                STRIDE0,
            ),
        ),
        (
            "finish",
            lambda: plan.finish(
                logits,
                starts,
                ends,
                plan.hist,
                plan.cnt,
                plan.cand_idx,
                plan.cand_val,
                idx,
                plan.counts,
                plan.slot,
                STRIDE0,
            ),
        ),
    )
    model = {"prepare": 24, "collect": 55, "finish": 36}
    total = 0.0
    for name, fn in stages:
        call()  # restore the pipeline's state before timing one piece of it
        torch.cuda.synchronize()
        t = device_us(fn)
        total += t
        print(
            f"    {name:>8} {t:>8.1f}   model said {model[name]:>3} us"
            f"   ({t / model[name]:.1f}x)",
            flush=True,
        )
    whole = device_us(call)
    print(f"    {'sum':>8} {total:>8.1f}\n    {'pipeline':>8} {whole:>8.1f}")

    ov.SAMPLED_MIN_VOCAB_PER_TOPK = 0  # gate off -> the previous path
    generic = device_us(call)
    ov.SAMPLED_MIN_VOCAB_PER_TOPK = 64
    print(f"    {'was':>8} {generic:>8.1f}   ({generic / whole:.3f}x)")
    print(
        "\n  If rows outside the window is large, the estimate is wrong and"
        "\n  the retry is paying for a second full histogram. If it is zero"
        "\n  and a stage is far above its model, the implementation is."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
