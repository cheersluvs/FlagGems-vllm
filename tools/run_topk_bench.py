"""Benchmark top_k_per_row_{prefill,decode} against the vendor baseline.

    tools/vendor_probe.sh tools/run_topk_bench.py metax_bench
    tools/vendor_probe.sh tools/run_topk_bench.py metax_bench_k --mode kernel

Defaults to `--mode operator`, which times the whole op call including launch
overhead -- the only mode whose number means the same thing on every card. Pass
anything else through as extra arguments.

Run it TWICE before quoting any ratio. A single pass gives no spread, and
without a spread a 1.1x and a 0.9x are the same measurement
([[benchmark-noise-floor-protocol]]: two low-rep measurements agreeing is not
evidence).
"""

import sys

import pytest

BENCHES = [
    "benchmark/test_top_k_per_row_prefill.py",
    "benchmark/test_top_k_per_row_decode.py",
]

if __name__ == "__main__":
    extra = sys.argv[1:]
    if not any(a.startswith("--mode") or a.startswith("--fg_mode") for a in extra):
        extra = ["--mode", "operator"] + extra
    sys.exit(pytest.main(BENCHES + ["-v", "-s", "--no-header"] + extra))
