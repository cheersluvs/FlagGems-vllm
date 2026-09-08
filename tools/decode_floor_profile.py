"""Where do decode's ~0.3 ms of fixed cost go?

Measured on MetaX C550: gems decode at vocab=262144 costs 0.341 ms at 1 row and
0.504 ms at 56 rows -- 56x the work for 1.48x the time. A linear fit puts ~0.28 ms
of that beyond any row count, while vLLM's whole call at 1 row is 0.096 ms.

This does NOT assume where the constant lives. It profiles both implementations
side by side and reports, per call: device time per kernel, the number of kernel
launches, host-side allocations, and any device->host sync. Whichever of those
carries the 0.28 ms is the thing to attack; the rest are eliminated.

    python tools/decode_floor_profile.py
"""

import sys

import torch
from torch.profiler import ProfilerActivity, profile

import flaggems_vllm

DEV = flaggems_vllm.device
VOCAB, TOPK = 262144, 512
ROWS = (1, 8, 56)
ACT = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

try:
    import vllm._custom_ops  # noqa: F401
    HAS_VLLM = hasattr(torch.ops._C, "top_k_per_row_decode")
except Exception:  # noqa: BLE001
    HAS_VLLM = False


def make(rows):
    logits = torch.randn(rows, VOCAB, device=DEV, dtype=torch.float32)
    seq_lens = torch.full((rows,), VOCAB, dtype=torch.int32, device=DEV)
    out = torch.empty((rows, TOPK), dtype=torch.int32, device=DEV)
    return logits, seq_lens, out


def gems(logits, seq_lens, out, rows):
    flaggems_vllm.top_k_per_row_decode(
        logits, 1, seq_lens, out, rows,
        logits.stride(0), logits.stride(1), TOPK)


def vllm_op(logits, seq_lens, out, rows):
    torch.ops._C.top_k_per_row_decode(
        logits, 1, seq_lens, out, rows,
        logits.stride(0), logits.stride(1), TOPK)


def summarize(fn, args, label):
    for _ in range(5):          # warm up compile + caches
        fn(*args)
    torch.cuda.synchronize()

    with profile(activities=ACT, record_shapes=False, profile_memory=True) as prof:
        for _ in range(10):
            fn(*args)
        torch.cuda.synchronize()

    evts = prof.key_averages()
    dev_total = sum(e.self_device_time_total for e in evts) / 10 / 1000.0
    launches = sum(e.count for e in evts if e.self_device_time_total > 0) / 10
    allocs = sum(e.count for e in evts if "aten::empty" in e.key
                 or "aten::zeros" in e.key or "Malloc" in e.key) / 10

    print(f"\n  --- {label}")
    print(f"      device time / call   {dev_total:8.4f} ms")
    print(f"      kernel launches      {launches:8.1f}")
    print(f"      alloc-ish calls      {allocs:8.1f}")
    hot = sorted((e for e in evts if e.self_device_time_total > 0),
                 key=lambda e: -e.self_device_time_total)[:6]
    for e in hot:
        print(f"      {e.self_device_time_total/10/1000.0:8.4f} ms  x{e.count/10:>4.1f}  "
              f"{e.key[:58]}")
    return dev_total


def main():
    print(f"decode floor profile | vocab={VOCAB} top_k={TOPK} | device {DEV}")
    print(f"vLLM baseline available: {HAS_VLLM}")
    for rows in ROWS:
        print("\n" + "=" * 74 + f"\n=== num_rows = {rows}\n" + "=" * 74)
        args = make(rows) + (rows,)
        g = summarize(gems, args, "flaggems_vllm (generic)")
        if HAS_VLLM:
            out2 = torch.empty((rows, TOPK), dtype=torch.int32, device=DEV)
            v = summarize(vllm_op, args[:2] + (out2, rows), "vLLM mcoplib")
            print(f"\n      device-time ratio vllm/gems = {v / g:.3f}"
                  if g else "")


if __name__ == "__main__":
    sys.exit(main())
