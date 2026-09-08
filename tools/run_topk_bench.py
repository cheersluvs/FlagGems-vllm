"""Benchmark top_k_per_row_{prefill,decode} against the vendor baseline.

    tools/vendor_probe.sh tools/run_topk_bench.py metax_bench
    tools/vendor_probe.sh tools/run_topk_bench.py metax_bench --repeat 3

`--mode kernel` is the ACCEPTANCE basis for every card except Ascend, so it is
the default and you should not need to pass it. Anything else is passed through
to pytest.

What kernel mode cannot see is host-side cost an override adds, so track that
separately rather than assuming it away: on this card the fused decode reaches
device-time parity with vLLM while operator mode reads 0.53, the difference
being Python sitting outside the kernels.

Two passes by default, because one pass has no spread and without a spread a
1.1x and a 0.9x are the same measurement. On this MetaX box the unchanged
shapes drifted 7-14% between runs, so treat anything under about 15% as
unresolved rather than as a result.
"""

import sys

import pytest

BENCHES = [
    "benchmark/test_top_k_per_row_prefill.py",
    "benchmark/test_top_k_per_row_decode.py",
]

if __name__ == "__main__":
    extra = list(sys.argv[1:])

    repeat = 2
    if "--repeat" in extra:
        i = extra.index("--repeat")
        repeat = int(extra[i + 1])
        del extra[i : i + 2]

    if not any(a.startswith("--mode") or a.startswith("--fg_mode") for a in extra):
        extra = ["--mode", "kernel"] + extra

    rc = 0
    for run in range(1, repeat + 1):
        print(f"\n{'=' * 78}\n=== pass {run} of {repeat}\n{'=' * 78}", flush=True)
        rc |= pytest.main(BENCHES + ["-v", "-s", "--no-header"] + extra)
    sys.exit(rc)
