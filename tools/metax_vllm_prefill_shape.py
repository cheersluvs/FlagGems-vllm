"""What does MetaX's vLLM prefill actually do, and where is our gap?

The baseline is mcoplib -- MetaX's own MACA build of vLLM's CUDA kernels, not
Triton. On decode it profiled as two C++ template instantiations,
`vllm::topKPerRowDecode<512, true, true, false>` and `<512, true, false, true>`,
which is a split-and-merge pair. Prefill has never been looked at.

Two things to find out:

  1. Its kernel structure -- how many launches, which template instantiations,
     and whether the mix changes with shape. That is what a "how does it work"
     answer has to rest on.

  2. Why 4095 and 4100 elements per row cost it 14.84 and 9.78 us per program.
     Ours are 18.15 and 18.50, flat as they should be for near-identical rows,
     so whatever the 52% is, it belongs to their side.

    python tools/metax_vllm_prefill_shape.py
"""

import sys

import torch

import flaggems_vllm

DEV = flaggems_vllm.device
SMS = 104

try:
    import vllm._custom_ops  # noqa: F401
    HAS = hasattr(torch.ops._C, "top_k_per_row_prefill")
except Exception:  # noqa: BLE001
    HAS = False

SHAPES = [
    (64, 129280, 1024, 129280), (4, 8193, 512, 8456),
    (16383, 4095, 512, 4352), (4, 16385, 512, 16648),
    (12961, 4100, 512, 4360), (16380, 5115, 512, 5376),
    (4100, 1025, 512, 1288),
]


def make(rows, vocab, stride0, top_k):
    buf = torch.randn((rows - 1) * stride0 + vocab, device=DEV, dtype=torch.float32)
    lg = torch.as_strided(buf, (rows, vocab), (stride0, 1))
    st = torch.zeros(rows, dtype=torch.int32, device=DEV)
    en = torch.full((rows,), vocab, dtype=torch.int32, device=DEV)
    out = torch.empty((rows, top_k), dtype=torch.int32, device=DEV)
    return buf, lg, st, en, out


def profile(fn, tag):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    evs = [e for e in p.key_averages() if e.self_device_time_total > 0]
    evs.sort(key=lambda e: -e.self_device_time_total)
    # An aten op node's self time is the sum of its children, so drop the
    # wrapper and keep the kernels; otherwise every total is counted twice.
    kern = [e for e in evs if "::" not in e.key or e.key.startswith("void ")]
    tot = sum(e.self_device_time_total for e in kern) / 10 / 1000
    print(f"    {tag}: {tot:.4f} ms in {sum(e.count for e in kern) / 10:.1f} kernels")
    for e in kern[:5]:
        print(f"        {e.self_device_time_total / 10 / 1000:9.4f} ms "
              f"x{e.count / 10:4.1f}  {e.key[:66]}")
    return tot


def main():
    if not HAS:
        print("no vLLM prefill symbol on this box; nothing to compare against")
        return 1
    print(f"device {DEV} | {SMS} SMs\n")
    for rows, vocab, top_k, stride0 in SHAPES:
        _b, lg, st, en, out = make(rows, vocab, stride0, top_k)
        o2 = torch.empty_like(out)
        print("=" * 78)
        print(f"=== ({rows}, {vocab}) top_k={top_k} stride0={stride0} "
              f"| {rows / SMS:.2f} waves")
        print("=" * 78)
        profile(lambda: torch.ops._C.top_k_per_row_prefill(
            lg, st, en, o2, rows, lg.stride(0), lg.stride(1), top_k), "vLLM ")
        profile(lambda: flaggems_vllm.top_k_per_row_prefill(
            lg, st, en, out, rows, lg.stride(0), lg.stride(1), top_k), "gems ")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
