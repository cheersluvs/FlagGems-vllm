# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check timer diagnostics preserve backend decisions and numerical results."""

import contextlib
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from qsa_timer_diagnostic_worker_910b import install


class DiagnosticTests(unittest.TestCase):
    def test_original_parameters_and_fallback_results_are_preserved(self):
        calls = []

        def mspti(
            funcs, warmup=5, active=30, clear_l2_cache=False, target_kernel_name=None
        ):
            calls.append(("mspti", warmup, active, clear_l2_cache, target_kernel_name))
            return float("inf")

        def profiler(
            funcs,
            warmup=5,
            active=30,
            clear_l2_cache=False,
            prof_dir=None,
            keep_res=False,
            target_kernel_name=None,
            _raise_on_mismatch=False,
        ):
            calls.append(
                (
                    "profiler",
                    warmup,
                    active,
                    clear_l2_cache,
                    target_kernel_name,
                    keep_res,
                    _raise_on_mismatch,
                )
            )
            root = Path(prof_dir) / "logs"
            root.mkdir(parents=True)
            (root / "error.log").write_text("capture failed")
            return 2.5

        timer = SimpleNamespace(
            do_bench_npu_mspti=mspti, do_bench_npu_profiler=profiler
        )
        outer = object()
        timer.do_bench_npu = outer
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(
            io.StringIO()
        ):
            root = Path(tmp)
            collect = install(timer, root, "case")
            self.assertEqual(timer.do_bench_npu, outer)
            self.assertEqual(timer.do_bench_npu_mspti([lambda: None]), float("inf"))
            self.assertEqual(
                timer.do_bench_npu_profiler(
                    [lambda: None], 5, 30, False, None, False, None, True
                ),
                2.5,
            )
            collect()
            self.assertEqual(
                calls,
                [
                    ("mspti", 5, 30, False, None),
                    ("profiler", 5, 30, False, None, True, True),
                ],
            )
            with gzip.open(root / "case-profiler-evidence.json.gz", "rt") as stream:
                result = json.load(stream)
            self.assertEqual(result["files"][0]["head"], "capture failed")
            self.assertTrue(Path(result["raw_directories"][0]).is_dir())

    def test_mspti_exception_is_logged_and_rethrown(self):
        def fail(funcs):
            raise RuntimeError("mspti unavailable")

        timer = SimpleNamespace(do_bench_npu_mspti=fail, do_bench_npu_profiler=fail)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(
            io.StringIO()
        ):
            root = Path(tmp)
            collect = install(timer, root, "case")
            with self.assertRaisesRegex(RuntimeError, "mspti unavailable"):
                timer.do_bench_npu_mspti([])
            self.assertIn(
                "mspti unavailable", (root / "case-timer-events.txt").read_text()
            )
            collect()


if __name__ == "__main__":
    unittest.main()
