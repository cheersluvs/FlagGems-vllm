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

    # And with splitting disabled, so step 3 runs the generic path instead.
    os.environ["FLAGGEMS_METAX_TOPK_SPLIT"] = "0"
    if ov is not None:
        ov._split_factor.__globals__  # noqa: B018 - _split_disabled reads os.environ
    gen = timed(splitter)
    print(f"  5. same shape, split DISABLED         {gen:8.4f} ms")
    after2 = timed(victim)
    print(f"  6. victim after the un-split version  {after2:8.4f} ms"
          f"   ({after2 / again:.3f}x of step 2)")

    print()
    drift = abs(again - cold) / cold * 100
    hit = (after - again) / again * 100
    print(f"  repeat-to-repeat drift on the victim: {drift:.1f}%")
    print(f"  change after the split path ran:      {hit:+.1f}%")
    if hit > 3 * max(drift, 1.0):
        print("\n  CONFIRMED: executing the split path slows later un-split calls.")
        print("  Compare step 6 -- if that one is fast, it is the split path's own")
        print("  allocations, not merely calling the override.")
    else:
        print("\n  NOT reproduced here. The benchmark regression comes from")
        print("  something this probe does not model -- do not blame the split path.")


if __name__ == "__main__":
    main()
