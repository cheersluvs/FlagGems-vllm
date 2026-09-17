"""Correctness of the sampled prefill path where the tests do not reach.

The functional tests do route one shape here -- vocab 129280, top_k 1024,
full rows -- but their offset test uses vocab 50000, below the sampled gate
(vocab >= 64 * top_k), so nothing tests this path with a nonzero row_start,
a truncated row_end, or a row short enough to force the exact retry inside
_s_finish. And they call each shape once, which only exercises the JIT launch;
every later call goes through a cached CompiledKernel.

The collect pass now stores indices only and _s_finish gathers the values back
through row_start + index -- exactly the kind of offset arithmetic that goes
silently wrong. So each case below is called three times with fresh inputs and
every row is checked against torch.topk over its own range, including the -1
padding a row shorter than top_k must produce.

    tools/vendor_probe.sh tools/hygon_prefill_sampled_check.py hygon_prefill_sampled_check
"""

import sys
from importlib import import_module

import torch

import flaggems_vllm

VOCAB, TOPK = 129280, 1024
ROWS = (1, 64)
# (label, how to build starts/ends from num_rows)
CASES = (
    "full",
    "offset",  # row_start 7..19, row_end trimmed by 0..100
    "short",  # span below top_k: undershoots, retry, -1 padding
    "narrow",  # span 3 * top_k: one sample tile, estimate weak, retry likely
)


def bounds(label, rows, dev):
    r = torch.arange(rows, dtype=torch.int32, device=dev)
    starts = torch.zeros(rows, dtype=torch.int32, device=dev)
    ends = torch.full((rows,), VOCAB, dtype=torch.int32, device=dev)
    if label == "offset":
        starts = 7 + r % 13
        ends = VOCAB - r % 101
    elif label == "short":
        starts = 5 + r % 3
        ends = starts + TOPK - 3
    elif label == "narrow":
        starts = 11 + r % 7
        ends = starts + 3 * TOPK
    return starts.contiguous(), ends.contiguous()


def check(logits, starts, ends, idx):
    bad = 0
    for row in range(logits.shape[0]):
        s, e = int(starts[row]), int(ends[row])
        n = e - s
        kk = min(TOPK, n)
        want = torch.topk(logits[row, s:e], kk).values.sort().values
        sel = idx[row]
        live = sel[(sel >= 0) & (sel < n)].long()
        got = logits[row, s + live].sort().values
        pad_ok = int((sel < 0).sum()) == TOPK - kk
        if not (live.numel() == kk and torch.equal(got, want) and pad_ok):
            bad += 1
    return bad


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    dev = "cuda"
    print(f"vocab {VOCAB}, top_k {TOPK}; each case called 3x with fresh inputs\n")
    print(f"  {'rows':>4} {'range':<8} {'sampled':>8}  call1 call2 call3")
    total_bad = 0
    for rows in ROWS:
        for label in CASES:
            cells = []
            routed = None
            for call in range(3):
                torch.manual_seed(1000 * rows + 17 * call + len(label))
                logits = torch.randn(rows, VOCAB, dtype=torch.float32, device=dev)
                starts, ends = bounds(label, rows, dev)
                idx = torch.full((rows, TOPK), -9, dtype=torch.int32, device=dev)
                routed = ov._can_sample(logits, starts, ends, rows, VOCAB, 1, TOPK)
                flaggems_vllm.top_k_per_row_prefill(
                    logits, starts, ends, idx, rows, VOCAB, 1, TOPK
                )
                torch.cuda.synchronize()
                bad = check(logits, starts, ends, idx)
                total_bad += bad
                cells.append(" ok " if bad == 0 else f"{bad:>3}!")
            print(
                f"  {rows:>4} {label:<8} {str(routed):>8}  " + "  ".join(cells),
                flush=True,
            )
    print(f"\n  sampled plans cached: {len(ov._SPLANS)}")
    print(f"  {'ALL CORRECT' if total_bad == 0 else f'{total_bad} WRONG ROWS'}")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
