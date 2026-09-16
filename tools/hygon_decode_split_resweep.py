"""Re-sweep decode's programs-per-row, now that the pipeline has changed.

The shipped table -- rows <= 4 give 16, <= 24 give 8, else 4 -- was swept on
the OLD two-pass split pipeline, where each program ran the whole radix
algorithm over its chunk and small chunks did not pay. The pipeline now makes
ONE pass that only compares against a threshold and appends, so the cost of a
small chunk is completely different, and the table has not been re-examined.

The 8-row shape says it needs re-examining. It is the only one left below the
0.9 line (0.827), and it is not noise:

    rows   programs   elements per program   ratio
       4   4 x 16 = 64          16384        1.089
       8   8 x  8 = 64          32768        0.827

The same number of programs, each doing twice the work, because the table
steps down from 16 to 8 at five rows. vLLM is flat across both (~82 us).

Forces each split through FLAGGEMS_HYGON_TOPK_DECODE_SPLIT and times the whole
operator with CUDA events, the way the benchmark's kernel mode does, against
vLLM on the same inputs.

    tools/vendor_probe.sh tools/hygon_decode_split_resweep.py hygon_decode_split_resweep
"""

import os
import sys
from importlib import import_module

import torch

import flaggems_vllm

V, K = 262144, 512
ROWS = (1, 4, 8, 16, 24, 32, 40, 48, 56)
SPLITS = (1, 2, 4, 8, 16, 32)


def wall_us(fn, iters=50, warmup=15):
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

    dev = "cuda"
    print(f"vocab {V}, top_k {K}; ratio vs vLLM, kernel-mode timing\n")
    print(
        f"  {'rows':>4} {'shipped':>8} {'vllm us':>8}"
        + "".join(f"{f'split {s}':>10}" for s in SPLITS)
    )
    for rows in ROWS:
        torch.manual_seed(rows)
        logits = torch.randn(rows, V, dtype=torch.float32, device=dev)
        lens = torch.full((rows,), V, dtype=torch.int32, device=dev)
        idx = torch.empty(rows, K, dtype=torch.int32, device=dev)
        want = torch.topk(logits, K, dim=1).values.sort(dim=1).values

        def call():
            flaggems_vllm.top_k_per_row_decode(logits, 1, lens, idx, rows, V, 1, K)

        t_vllm = wall_us(
            lambda: torch.ops._C.top_k_per_row_decode(
                logits, 1, lens, idx, rows, V, 1, K
            )
        )
        os.environ.pop("FLAGGEMS_HYGON_TOPK_DECODE_SPLIT", None)
        shipped = ov._split_factor(rows, V, K)
        cells = []
        for split in SPLITS:
            os.environ["FLAGGEMS_HYGON_TOPK_DECODE_SPLIT"] = str(split)
            if ov._split_factor(rows, V, K) != split:
                cells.append("       -  ")
                continue
            call()
            torch.cuda.synchronize()
            got = logits.gather(1, idx.long().clamp(0, V - 1)).sort(dim=1).values
            ok = torch.allclose(got, want)
            t = wall_us(call)
            cells.append(f"{t_vllm / t:>9.3f}{'' if ok else '!'}" + (" " if ok else ""))
        os.environ.pop("FLAGGEMS_HYGON_TOPK_DECODE_SPLIT", None)
        print(f"  {rows:>4} {shipped:>8} {t_vllm:>8.1f}" + "".join(c for c in cells))
    print(
        "\n  '-' means the split was rejected (chunk below max(MIN_CHUNK, top_k));"
        "\n  '!' would mean a wrong answer at that split."
    )


if __name__ == "__main__":
    sys.exit(main())
