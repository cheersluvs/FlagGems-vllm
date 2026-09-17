"""Run the shipped prefill op on one shape, N times. A driver for a profiler.

    python tools/hygon_prefill_one.py <rows> <vocab> <top_k> <stride0> [iters]

Deliberately minimal: one op, no timing, no printing inside the loop, so that
whatever wraps it attributes its counters to the operator's kernel and not to
the harness.
"""

import sys

import torch

import flaggems_vllm


def main():
    rows, vocab, top_k, stride0 = (int(a) for a in sys.argv[1:5])
    iters = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    dev = "cuda"
    torch.manual_seed(42)
    buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
    logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
    starts = torch.zeros(rows, dtype=torch.int32, device=dev)
    ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
    idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)
    for _ in range(3):
        flaggems_vllm.top_k_per_row_prefill(
            logits, starts, ends, idx, rows, stride0, 1, top_k
        )
    torch.cuda.synchronize()
    for _ in range(iters):
        flaggems_vllm.top_k_per_row_prefill(
            logits, starts, ends, idx, rows, stride0, 1, top_k
        )
    torch.cuda.synchronize()
    print(f"ran {iters} x prefill {rows}x{vocab} top_k {top_k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
