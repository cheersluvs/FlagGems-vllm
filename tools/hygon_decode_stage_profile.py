"""Which of the split pipeline's five launches holds decode's 16-56 row time?

Geometry is exhausted: sweeping BLOCK x warps against the split factor left the
shipped B512 w8 best or tied at every row count (narrow programs win in prefill
because its programs are short; a decode chunk is 32768 elements and wants the
threads). Rows 16-56 sit at 0.535-0.805 of vLLM, the biggest block left.

So attribute the time. The pipeline is:

    lens     per-chunk lengths            (rows programs, trivial)
    stage1   generic kernel per chunk     (rows*split programs, the data)
    gather   candidate values             (rows x ncand/BLOCK programs)
    merge    generic kernel on candidates (rows programs, split*top_k elements)
    remap    merged positions -> indices  (rows programs)

Profiler device time per kernel per call, with the shipped split factor, for
rows 16..56, next to vLLM's two kernels. Two very different conclusions:
stage1 dominating means per-element work (fewer passes, e.g. a sampled
threshold) is the only lever left; merge dominating means a ~19 us kernel
floor paid per row over 2048-4096 candidates, which is cheap to restructure.

    tools/vendor_probe.sh tools/hygon_decode_stage_profile.py hygon_decode_stage_profile
"""

import sys
from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, profile

import flaggems_vllm

V, K = 262144, 512
ROWS = (16, 24, 32, 40, 56)


def by_kernel(fn, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    out = {}
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0)
        if not t:
            continue
        name = ev.key
        if name.startswith("void "):
            name = name[5:]
        name = name.split("(")[0][:38]
        out[name] = out.get(name, 0.0) + t / iters
    return out


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    import vllm._custom_ops  # noqa: F401

    print(f"vocab {V}, top_k {K}; device us per call, shipped configuration\n")
    for rows in ROWS:
        torch.manual_seed(rows)
        logits = torch.randn(rows, V, dtype=torch.float32, device="cuda")
        lens = torch.full((rows,), V, dtype=torch.int32, device="cuda")
        idx = torch.empty(rows, K, dtype=torch.int32, device="cuda")
        split = ov._split_factor(rows, V, K)
        ours = by_kernel(
            lambda: flaggems_vllm.top_k_per_row_decode(
                logits, 1, lens, idx, rows, V, 1, K
            )
        )
        base = by_kernel(
            lambda: torch.ops._C.top_k_per_row_decode(
                logits, 1, lens, idx, rows, V, 1, K
            )
        )
        tot_o = sum(ours.values())
        tot_b = sum(v for k, v in base.items() if "topKPerRow" in k)
        print(
            f"  rows {rows:>3} (split {split}, {rows * split} chunk programs): "
            f"ours {tot_o:7.1f} us, vLLM {tot_b:7.1f} us, ratio {tot_b / tot_o:.3f}"
        )
        for name, t in sorted(ours.items(), key=lambda kv: -kv[1]):
            if t < 0.5:
                continue
            print(f"      {t:7.1f} us  {100 * t / tot_o:5.1f}%  {name}")
        for name, t in sorted(base.items(), key=lambda kv: -kv[1]):
            if t < 0.5 or "topKPerRow" not in name:
                continue
            print(f"      {t:7.1f} us     vLLM  {name}")
    print("\n  stage1 is the generic kernel over chunks; merge is the same kernel")
    print("  over split*top_k candidates, one program per row.")


if __name__ == "__main__":
    sys.exit(main())
