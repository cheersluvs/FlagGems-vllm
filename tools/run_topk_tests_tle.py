"""Run the top_k_per_row functional tests on the SHIMMED TLE path (MetaX).

The TLE path with tools/metax_tle_shim.py (radix final off) measured prefill
at ~1.22 of vLLM against 0.640 without TLE -- but the forced-TLE correctness
check covered two prefill shapes. A speedup on an unchecked path is not a
result, so run the real tests, in-process, with the shims installed first:
HAS_TLE is fixed at import and pytest.main shares this interpreter.

    PY=/data/wuyuqing/workspace/mctle-test/bin/python \
        tools/vendor_probe.sh tools/run_topk_tests_tle.py metax_tests_tle
    ... --radix     keep radix final on (known WRONG; to count the damage)
"""

import os
import sys

import pytest

extra = list(sys.argv[1:])
radix = "--radix" in extra
extra = [a for a in extra if a != "--radix"]

os.environ["FLAGGEMS_FORCE_TLE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flaggems_vllm  # noqa: E402,F401
import metax_tle_shim  # noqa: E402

ok, msg = metax_tle_shim.install(radix_final=radix)
print(f"=== {msg}", flush=True)
if not ok:
    sys.exit(3)

sys.exit(pytest.main([
    "tests/test_top_k_per_row_prefill.py",
    "tests/test_top_k_per_row_decode.py",
    "-q", "-rf", "--no-header", "-p", "no:cacheprovider",
] + extra))
