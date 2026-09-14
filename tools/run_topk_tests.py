"""Run the two top_k_per_row functional test files, as the box's default
backend dispatches them (vendor overrides included), and report.

    tools/vendor_probe.sh tools/run_topk_tests.py <report-name> [pytest args]
"""

import sys

import pytest

sys.exit(
    pytest.main(
        [
            "tests/test_top_k_per_row_prefill.py",
            "tests/test_top_k_per_row_decode.py",
            "-q",
            "-rf",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ]
        + sys.argv[1:]
    )
)
