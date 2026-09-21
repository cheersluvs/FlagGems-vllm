"""Sweep decode's SAMPLE_TILES against SAFETY -- the two constants that were
never measured together.

WHY. The shipped override reads SAMPLE_TILES=8 tiles (4096 elements, 1.6% of a
262144 row) and loosens the admit rank by SAFETY=4. Neither was ever swept:
`git log -S SAMPLE_TILES` shows the value written once, in the commit that
introduced the path, and never touched. It works, but "it works" is not a
measurement, and the two are coupled -- a smaller sample needs a larger SAFETY
to buy back margin, while a larger SAFETY makes `select` append more and the
tail's radix rank more.

A host-side simulation of the estimator (numpy, standard normals, the
operator's own STEP-0 key, 200 rows per point) says the ACCURACY side is:

    tiles  sample   admitted median   rel.sd   margin to the low edge
      2     1024        2150          33.6%        2.2 sd
      4     2048        2272          27.7%        2.8 sd
      8     4096        2274          17.5%        4.6 sd
     16     8192        2279          12.6%        6.3 sd

so 8 is defensible and 16 costs only another 1.6% of a pass. What that
simulation cannot say is what either constant costs on the card, which is what
this measures.

WHAT IT MEASURES, per (tiles, safety), on the real operator:

  admitted   the per-row candidate count, by running ONLY prepare+select and
             reading the counter before `_tail` resets it: median / min / max,
             and the share of rows outside the [top_k, CAP] window -- those
             rows pay the exact redo inside `_tail`.
  device us  the whole operator, interleaved, fastest of ROUNDS rounds, as a
             ratio against the shipped (8, 4).
  answers    every combination checked against torch.topk. All of them should
             be correct: a bad estimate is an efficiency problem, not a
             correctness one, because `_tail` redoes any row outside the
             window. A WRONG here would mean that safety net is broken.

The window is [top_k, CAP] with CAP = _cap(top_k) = 16*top_k, and it does NOT
move with SAFETY -- CAP_FACTOR is separate -- so raising SAFETY walks the
admitted count towards the upper edge while lowering it walks towards the
lower one.

ROUTING. Both constants are module globals read at _Plan construction, and the
plan cache key does NOT include them, so a sweep that forgets to clear _PLANS
silently measures the first combination for every point. Every point here
clears the cache and then asserts the rebuilt plan actually carries the
constexprs it asked for.

    tools/vendor_probe.sh tools/hygon_decode_sample_sweep.py hygon_decode_sample_sweep
"""

import pathlib
import sys
from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, profile

# (num_rows, vocab, top_k) -- starved and full-card points from the benchmark
SHAPES = [
    (1, 262144, 512),
    (8, 262144, 512),
    (24, 262144, 512),
    (56, 262144, 512),
    (512, 262144, 512),
]
TILES = (4, 8, 16, 32)
SAFETIES = (2, 4, 8)
BASE = (8, 4)  # what ships today
ROUNDS = 3


def occupancy(tag):
    """What else is on the card. This box is shared, and one earlier probe came
    back with per-round ratios spread 0.08-4.2 while the same shapes had held
    within 10% that morning."""
    import shutil
    import subprocess

    for cmd in (["hy-smi"], ["rocm-smi", "--showpids"], ["rocm-smi"]):
        exe = shutil.which(cmd[0]) or (
            f"/opt/dtk/bin/{cmd[0]}"
            if pathlib.Path(f"/opt/dtk/bin/{cmd[0]}").exists()
            else None
        )
        if not exe:
            continue
        try:
            out = subprocess.run(
                [exe] + cmd[1:], capture_output=True, text=True, timeout=30
            ).stdout
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            out = repr(exc)
        print(f"--- card occupancy {tag}: {' '.join(cmd)}")
        print("\n".join(out.strip().splitlines()[:25]))
        return
    print(f"--- card occupancy {tag}: no smi tool found")


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


def configure(ov, tiles, safety):
    """Point the override at (tiles, safety) and prove the plan rebuilt."""
    ov.SAMPLE_TILES = tiles
    ov.SAFETY = safety
    ov._PLANS.clear()


def check_routing(ov, tiles, safety, vocab):
    assert len(ov._PLANS) == 1, f"expected one plan, got {len(ov._PLANS)}"
    plan = next(iter(ov._PLANS.values()))
    got = plan.prepare.constexprs
    want_stride = max(1, vocab // (ov.BLOCK * tiles))
    assert got["SAFETY"] == safety, f"SAFETY {got['SAFETY']} != {safety}"
    assert got["STRIDE"] == want_stride, f"STRIDE {got['STRIDE']} != {want_stride}"
    return plan


def admitted(plan, logits, seq_lens, stride0):
    """prepare + select only, so the counter is read before `_tail` resets it."""
    plan.prepare(logits, seq_lens, plan.hist, plan.thr, plan.cnt, stride0)
    plan.select(
        logits, seq_lens, plan.thr, plan.cnt, plan.cand_idx, plan.cand_val, stride0
    )
    torch.cuda.synchronize()
    return plan.cnt.clone()


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    dev = "cuda"
    occupancy("before")
    print(
        f"\ndevice us, interleaved, fastest of {ROUNDS} rounds; ratio is against"
        f"\nthe shipped SAMPLE_TILES={BASE[0]} SAFETY={BASE[1]}, so > 1 means the"
        "\npoint is faster. 'outside' is the share of rows landing outside the"
        "\n[top_k, CAP] window -- those pay the exact redo inside _tail.\n"
    )
    for num_rows, vocab, top_k in SHAPES:
        cap = ov._cap(top_k)
        torch.manual_seed(42)
        logits = torch.randn((num_rows, vocab), device=dev, dtype=torch.float32)
        seq_lens = torch.full((num_rows,), vocab, dtype=torch.int32, device=dev)
        indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
        want = torch.topk(logits, top_k, dim=1).values.sort(dim=1).values
        print(f"  {num_rows} x {vocab}, top_k {top_k}, window [{top_k}, {cap}]")
        print(
            f"    {'tiles':>6} {'sample':>7} {'safety':>7} {'admit med':>10}"
            f" {'min':>7} {'max':>7} {'outside':>8} {'us':>9} {'vs base':>8} {'ans':>6}"
        )
        base_us = None
        # BASE first, so every later row's ratio has its denominator already.
        combos = [BASE] + [(t, sf) for t in TILES for sf in SAFETIES if (t, sf) != BASE]
        for tiles, safety in combos:
            configure(ov, tiles, safety)

            def run():
                ov.top_k_per_row_decode(
                    logits, 1, seq_lens, indices, num_rows, vocab, 1, top_k
                )

            indices.fill_(-9)
            run()
            torch.cuda.synchronize()
            plan = check_routing(ov, tiles, safety, vocab)
            got = (
                logits.gather(1, indices.long().clamp(0, vocab - 1)).sort(dim=1).values
            )
            ok = torch.allclose(got, want) and bool((indices >= 0).all())
            c = admitted(plan, logits, seq_lens, vocab).float()
            out = float(((c < top_k) | (c > cap)).float().mean()) * 100
            us = min(device_us(run) for _ in range(ROUNDS))
            if (tiles, safety) == BASE:
                base_us = us
            sample = min(tiles * ov.BLOCK, vocab)
            print(
                f"    {tiles:6d} {sample:7d} {safety:7d} {int(c.median()):10d}"
                f" {int(c.min()):7d} {int(c.max()):7d} {out:7.1f}%"
                f" {us:9.1f}"
                f" {(base_us / us if base_us else float('nan')):8.3f}"
                f" {'OK' if ok else 'WRONG':>6}",
                flush=True,
            )
        print()
    ov.SAMPLE_TILES, ov.SAFETY = BASE
    ov._PLANS.clear()
    occupancy("after")
    print(
        "  The shipped (8, 4) is measured first in each block and is the"
        "\n  denominator of the ratio column. Read 'outside' first: a point that"
        "\n  is fast because most rows skip the estimate and fall into the redo"
        "\n  is not a faster estimator, it is a worse one."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
