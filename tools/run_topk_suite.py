"""Run the top_k_per_row suites in-process, so tools/vendor_probe.sh can carry
the vendor environment into them.

tools/run_tests.py imports `distro`, which several of these images do not have;
pytest itself is enough here.

    tools/vendor_probe.sh tools/run_topk_suite.py topk_suite
"""

import sys

import pytest

ARGS = [
    "tests/test_top_k_per_row_prefill.py",
    "tests/test_top_k_per_row_decode.py",
    "-v",
    "-rs",          # say WHY anything was skipped
    "--no-header",
    "-p", "no:cacheprovider",
]

if __name__ == "__main__":
    sys.exit(pytest.main(ARGS + sys.argv[1:]))
