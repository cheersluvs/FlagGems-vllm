"""Is prefill's top_k term the O(n^2) rank sort, or something else?

The per-program cost decomposes as roughly

    4.9 us  +  17 ns * top_k  +  1.5-1.9 ns * vocab

and 17 ns per unit of k is about what one dependent scalar load costs, which
points at the rank-by-counting sort in the final selection:

    for j in tl.range(0, final_cnt):
        logit_j = tl.load(s_final_logits_ptr + j)   # one scalar, serial
        better = (logit_i < logit_j) | ...

But that loop is bounded by final_cnt -- the population of the threshold bin --
not by TOPK. With 2048 bins over a 4096-element row the threshold bin should
hold a handful of elements, and the loop should be nearly free. Both cannot be
true, so separate them by moving final_cnt while holding k fixed.

Concentrating the values into fewer distinct radix buckets makes the threshold
bin large without changing k, the row length, or anything else. If the cost
tracks concentration, it is the sort. If it tracks only k, it is elsewhere and
the sort is exonerated.

Two attempts at concentrating did nothing, both because the top 11 bits of a
float32 are sign, exponent and only TWO mantissa bits. Scaling by a constant
shifts every exponent equally and left the bucket count at 88; adding a large
offset collapses everything onto one exponent, where only 4 buckets exist, and
went straight to 1. Bucket count is governed by EXPONENT SPAN, so the control
here is 2**U(0,E): E from 16 to 1 walks 64 buckets down to 4 and the largest
bucket from 96 elements to 1287 -- a 13x range on the sort's loop bound, with
k, row length and row count all fixed.
"""

import sys

import torch

import flaggems_vllm

DEV = flaggems_vllm.device
SMS = 104
ROWS = 4160


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def run(logits, top_k):
    rows, vocab = logits.shape
    st = torch.zeros(rows, dtype=torch.int32, device=DEV)
    en = torch.full((rows,), vocab, dtype=torch.int32, device=DEV)
    out = torch.empty((rows, top_k), dtype=torch.int32, device=DEV)
    t = timed(lambda: flaggems_vllm.top_k_per_row_prefill(
        logits, st, en, out, rows, logits.stride(0), logits.stride(1), top_k))
    return t * 1000 / (rows / SMS)


def bucket_spread(logits):
    """Distinct 11-bit radix buckets per row -- the sort's loop bound proxy."""
    b = logits.view(torch.int32)
    key = torch.where(b < 0, b ^ 0x7FFFFFFF, b)
    buckets = (key >> 21) & 0x7FF
    return float(torch.tensor(
        [len(torch.unique(buckets[i])) for i in range(min(8, buckets.shape[0]))],
        dtype=torch.float32).mean())


def main():
    vocab = 4096
    print(f"device {DEV} | rows {ROWS} vocab {vocab}\n")
    print("Exponent span controls the bucket count, and the bucket count sets")
    print("the threshold bin -- which is the sort's inner loop bound. k, row")
    print("length and row count are identical on every line.\n")
    print(f"  {'input':>9} {'buckets/row':>12} {'k=128':>8} {'k=512':>8} "
          f"{'k=1024':>8}   {'ns/k':>7}")

    for spread in (None, 16, 8, 4, 2, 1):
        torch.manual_seed(0)
        if spread is None:      # the ordinary input, for reference
            logits = torch.randn(ROWS, vocab, device=DEV, dtype=torch.float32)
        else:
            logits = torch.exp2(
                torch.rand(ROWS, vocab, device=DEV, dtype=torch.float32) * spread)
        logits = logits.contiguous()
        nb = bucket_spread(logits)
        ts = {k: run(logits, k) for k in (128, 512, 1024)}
        slope = (ts[1024] - ts[128]) / (1024 - 128) * 1000
        print(f"  {('randn' if spread is None else f'2^U(0,{spread})'):>9} {nb:>12.0f} {ts[128]:>8.2f} {ts[512]:>8.2f} "
              f"{ts[1024]:>8.2f}   {slope:>7.1f}")

    print()
    print("  Cost rising as buckets/row falls  -> it IS the rank sort.")
    print("  Cost flat while buckets/row falls -> the sort is exonerated and")
    print("  the k term is somewhere else; do not rewrite that loop.")


if __name__ == "__main__":
    sys.exit(main())
