"""Two questions about the merge pass, before writing a kernel for it.

Stage two currently runs the full decode op over the gathered candidates. At
8 rows that costs about 0.04 ms against vLLM's 0.115 ms for the whole call, so
it is worth knowing exactly what is being paid for.

  1. Is stage one's output already ordered by value? If it is, S ordered lists
     can be merged by touching about k + S elements instead of selecting over
     S*k, and the 2048-bin histogram disappears.

  2. Does the merge cost scale with the candidate count, or is it flat? Flat
     means the bins dominate and the data does not, which decides whether a
     cheaper selection is worth writing at all.

    python tools/metax_merge_headroom.py
"""

import sys

import torch

import flaggems_vllm

DEV = flaggems_vllm.device
TOPK = 512


def timed(fn, iters=30, warmup=10):
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


def decode(logits, rows, top_k=TOPK):
    sl = torch.full((rows,), logits.shape[1], dtype=torch.int32, device=DEV)
    out = torch.empty((rows, top_k), dtype=torch.int32, device=DEV)
    flaggems_vllm.top_k_per_row_decode(
        logits, 1, sl, out, rows, logits.stride(0), logits.stride(1), top_k)
    return out


def q1_is_output_ordered():
    print("=" * 72)
    print("=== 1. is the op's top-k output ordered by value?")
    print("=" * 72)
    for rows, width in ((4, 32768), (8, 65536), (2, 262144)):
        torch.manual_seed(rows)
        logits = torch.randn(rows, width, device=DEV, dtype=torch.float32)
        idx = decode(logits, rows)
        vals = torch.gather(logits, 1, idx.long())
        desc = bool((vals[:, :-1] >= vals[:, 1:]).all())
        # If not exactly ordered, is it ordered by the radix bucket -- i.e. by
        # the top bits? That would still allow a bucket-wise merge.
        bits = logits.view(torch.int32)
        key = torch.gather(bits, 1, idx.long())
        key = torch.where(key < 0, key ^ 0x7FFFFFFF, key)  # order-preserving
        by_bucket = bool(((key[:, :-1] >> 21) >= (key[:, 1:] >> 21)).all())
        # how far out of order, at worst
        drop = (vals[:, 1:] - vals[:, :-1]).max().item()
        print(f"  rows={rows:<3} width={width:<7} exactly ordered: "
              f"{'YES' if desc else 'no ':<4} | ordered by 11-bit bucket: "
              f"{'YES' if by_bucket else 'no':<4} | worst rise {drop:+.4f}")
    print()


def q2_merge_scaling():
    print("=" * 72)
    print("=== 2. does the merge cost scale with candidates, or with the bins?")
    print("=" * 72)
    print(f"  {'ncand':>8} {'ms':>9} {'ms/1k cand':>12}")
    base = None
    for ncand in (1024, 2048, 4096, 8192, 16384, 32768):
        if ncand < TOPK:
            continue
        torch.manual_seed(0)
        cand = torch.randn(1, ncand, device=DEV, dtype=torch.float32)
        t = timed(lambda: decode(cand, 1))
        base = base or t
        print(f"  {ncand:>8} {t:>9.4f} {t / (ncand / 1024):>12.4f}")
    print(f"\n  A flat first column means the 2048-bin pass dominates and the")
    print(f"  candidates are nearly free -- so a selection without those bins")
    print(f"  is where the merge's cost actually is.")
    print()


if __name__ == "__main__":
    print(f"device {DEV} | top_k {TOPK}\n")
    q1_is_output_ordered()
    q2_merge_scaling()
    sys.exit(0)
