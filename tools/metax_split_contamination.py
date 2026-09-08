"""Does running the split path earlier make later un-split calls slower?

Both benchmark arms import and dispatch through the override; the only
difference is whether rows 1/4/8 actually split. Yet rows 24/32/40 come out
11-12% slower in the arm that did, reproducibly across two A/B pairs, on an
identical code path.

So time one un-split shape twice in ONE process: cold, then again after the
split path has run. If the second reading is slower, the contamination is real
and comes from executing the split path, not from importing the override.

    python tools/metax_split_contamination.py
"""

import os
import sys

import torch

import flaggems_vllm

DEV = flaggems_vllm.device
VOCAB, TOPK = 262144, 512
VICTIM_ROWS = 32          # 0.90x in both A/B pairs
SPLIT_ROWS = 1            # what the benchmark runs first


def timed(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def call(rows):
    logits = torch.randn(rows, VOCAB, device=DEV, dtype=torch.float32)
    sl = torch.full((rows,), VOCAB, dtype=torch.int32, device=DEV)
    out = torch.empty((rows, TOPK), dtype=torch.int32, device=DEV)
    fn = lambda: flaggems_vllm.top_k_per_row_decode(  # noqa: E731
        logits, 1, sl, out, rows, logits.stride(0), logits.stride(1), TOPK)
    return fn


def main():
    ov = sys.modules.get(
        "flaggems_vllm.runtime.backend._metax.fused.top_k_per_row_decode")
    print(f"device {DEV} | vocab {VOCAB} top_k {TOPK}")
    print(f"victim = {VICTIM_ROWS} rows (never splits), "
          f"contaminant = {SPLIT_ROWS} row (splits)\n")

    victim = call(VICTIM_ROWS)
    cold = timed(victim)
    print(f"  1. victim, cold                       {cold:8.4f} ms")

    again = timed(victim)
    print(f"  2. victim again, nothing between      {again:8.4f} ms"
          f"   ({cold / again:.3f}x of cold)")

    splitter = call(SPLIT_ROWS)
    sp = timed(splitter)
    print(f"  3. the split path itself              {sp:8.4f} ms")

    after = timed(victim)
    print(f"  4. victim AFTER the split path        {after:8.4f} ms"
          f"   ({after / again:.3f}x of step 2)")

    # The benchmark's order is 1, 496, 512, 16, 32, ... -- the two big shapes
    # allocate about 512 MB each, and in the override arm the split path's
    # temporaries are allocated BEFORE them. Step 4 above never modelled that,
    # so replay it here: the suspicion is now the allocator's layout, not the
    # split path on its own.
    print()
    big = [call(r) for r in (496, 512)]
    for f in big:
        f()
    torch.cuda.synchronize()
    after_big = timed(victim)
    print(f"  5. victim after 496 and 512 rows too  {after_big:8.4f} ms"
          f"   ({after_big / again:.3f}x of step 2)")

    print(f"     allocator: {torch.cuda.memory_allocated() / 2**20:8.1f} MiB live, "
          f"{torch.cuda.memory_reserved() / 2**20:.1f} MiB reserved")

    del big
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    freed = timed(victim)
    print(f"  6. victim after empty_cache()         {freed:8.4f} ms"
          f"   ({freed / again:.3f}x of step 2)")

    print()
    drift = max(abs(again - cold) / cold * 100, 0.1)
    hit_split = (after - again) / again * 100
    hit_big = (after_big - again) / again * 100
    print(f"  repeat-to-repeat drift on the victim: {drift:.1f}%")
    print(f"  after the split path alone:           {hit_split:+.1f}%")
    print(f"  after the split path AND the big rows:{hit_big:+.1f}%")
    print(f"  after releasing them:                 {(freed - again) / again * 100:+.1f}%")
    if hit_big > 3 * drift and hit_split <= 3 * drift:
        print("\n  It is the SEQUENCE, not the split: the regression needs the big")
        print("  allocations that follow. If step 6 recovers, it is allocator layout.")
    elif hit_big <= 3 * drift:
        print("\n  Still not reproduced. The benchmark regression is not modelled by")
        print("  this sequence either -- look at the harness before the operator.")


if __name__ == "__main__":
    main()
