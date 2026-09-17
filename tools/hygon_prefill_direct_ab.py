"""Did the direct launch make (64,129280) slower, or is that shape bimodal?

The direct-launch checker read 228.2 us of device time with the path off and
361.9 with it on -- a 1.59x regression on prefill's worst shape, caught by the
control column that exists for exactly that. But this shape has measured 228,
232, 238, 311 and 362 us across probes, in two clear clusters, and BOTH
clusters predate the change: the stage-split run read 362 and the interleaved
A/B read 231-233, and neither had this code. So the checker's on/off pair may
simply have landed in different clusters.

One measurement cannot separate those. Alternate the two paths in one process
over several rounds, device time each: a causal regression holds its ratio
across rounds, bimodality does not. The same technique already overturned a
"1.32x" that was a high denominator.

Also reports what each path actually launches with -- module, BLOCK_SIZE,
num_warps -- because the two are supposed to be identical there, and if they
are not, that is the answer and no amount of repetition will show it.

    tools/vendor_probe.sh tools/hygon_prefill_direct_ab.py hygon_prefill_direct_ab
"""

import sys
from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, profile

import flaggems_vllm

SHAPES = [
    (64, 129280, 1024, 129280),
    (4, 8193, 512, 8456),
    (16383, 4095, 512, 4352),
]
ROUNDS = 5


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
    print(f"device us, direct on and off alternating, {ROUNDS} rounds\n")
    for rows, vocab, top_k, stride0 in SHAPES:
        mod = (
            ov._dense
            if ov._ENABLED and vocab <= ov.DENSE_VOCAB_PER_TOPK * top_k
            else ov._generic
        )
        geo = ov._geometry(rows, vocab) if ov._GEOMETRY else None
        if geo is None:
            block, warps_of = ov._GENERIC_DEFAULTS[id(mod)]
            warps = warps_of(block)
        else:
            block, warps = geo
        which = "dense" if mod is ov._dense else "generic"
        per_row = (mod.NUM_BINS + mod.NUM_FILNAL_ITEMS) * 4
        cached = rows * per_row <= ov.SCRATCH_CACHE_BYTES

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

        print(
            f"  {rows} x {vocab}, top_k {top_k}: module {which}, "
            f"BLOCK_SIZE {block}, num_warps {warps}, "
            f"scratch {'cached' if cached else 'per call'}"
        )
        ratios = []
        for _ in range(ROUNDS):
            ov._DIRECT = True
            on = device_us(call)
            ov._DIRECT = False
            off = device_us(call)
            ratios.append(off / on)
            print(f"      on {on:>8.1f}   off {off:>8.1f}   off/on {off / on:>6.3f}")
        ov._DIRECT = True
        rs = sorted(ratios)
        print(
            f"      median {rs[len(rs) // 2]:.3f}, "
            f"spread {rs[0]:.3f} to {rs[-1]:.3f}\n"
        )
    print(
        "  A ratio that holds across rounds is the change; one that bounces is"
        "\n  the shape. Both paths print their launch arguments above: if those"
        "\n  differ, that is the cause and the rounds are beside the point."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
