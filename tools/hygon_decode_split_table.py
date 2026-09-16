"""Fill in the shipped decode split rule, row by row, on the real override.

The rule (rows <= 4 -> 16, <= 16 -> 8, else 4) was fitted to six measured row
counts; 24, 40 and 48 were extrapolated. The benchmark then read 24 rows at
0.453, below both 16 rows (0.553) and 32 (0.665) -- the sign of a wrong factor.

This sweeps the factor on the SHIPPED override rather than a reimplementation:
FLAGGEMS_HYGON_TOPK_DECODE_SPLIT is read per call, so every point here is the
production path, cached direct launches included. Each point is checked against
torch.topk before it is timed, and vLLM is timed in the same process.

    tools/vendor_probe.sh tools/hygon_decode_split_table.py hygon_decode_split_table
"""

import os
import sys
from importlib import import_module

import torch

import flaggems_vllm

V, K = 262144, 512
ROWS = (1, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 79)
SPLITS = (1, 2, 4, 8, 16, 32)


def wall_us(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    import vllm._custom_ops  # noqa: F401

    print(
        f"vocab {V}, top_k {K}; ratio vs vLLM on the shipped override "
        f"(* best, r = the rule's own choice, ! = WRONG)\n"
    )
    print(
        f"  {'rows':>4} {'rule':>5}  "
        + " ".join(f"{'/' + str(s):>8}" for s in SPLITS)
        + f" {'best':>6}"
    )
    for rows in ROWS:
        torch.manual_seed(rows)
        logits = torch.randn(rows, V, dtype=torch.float32, device="cuda")
        want = torch.topk(logits, K, dim=1).values.sort(dim=1).values
        lens = torch.full((rows,), V, dtype=torch.int32, device="cuda")
        idx = torch.empty(rows, K, dtype=torch.int32, device="cuda")
        os.environ.pop("FLAGGEMS_HYGON_TOPK_DECODE_SPLIT", None)
        rule = ov._split_factor(rows, V, K)
        t_vllm = wall_us(
            lambda: torch.ops._C.top_k_per_row_decode(
                logits, 1, lens, idx, rows, V, 1, K
            )
        )
        cells, best = [], None
        for split in SPLITS:
            os.environ["FLAGGEMS_HYGON_TOPK_DECODE_SPLIT"] = str(split)
            try:
                call = lambda: flaggems_vllm.top_k_per_row_decode(  # noqa: E731
                    logits, 1, lens, idx, rows, V, 1, K
                )
                call()
                torch.cuda.synchronize()
                got = logits.gather(1, idx.long().clamp(0, V - 1)).sort(dim=1).values
                ok = torch.allclose(got, want)
                us = wall_us(call)
                ratio = t_vllm / us
                mark = "!" if not ok else ("r" if split == rule else " ")
                if ok and (best is None or ratio > best[0]):
                    best = (ratio, split)
                cells.append(f"{ratio:>7.3f}{mark}")
            except Exception as e:  # noqa: BLE001
                cells.append(f"{type(e).__name__[:7]:>8}")
        os.environ.pop("FLAGGEMS_HYGON_TOPK_DECODE_SPLIT", None)
        tag = f"{best[1]:>6}" if best else "     -"
        print(f"  {rows:>4} {rule:>5}  " + " ".join(cells) + tag)
    print("\n  A row whose best column is not the one marked r is a rule to fix;")
    print("  /1 is the generic path with no split.")


if __name__ == "__main__":
    sys.exit(main())
