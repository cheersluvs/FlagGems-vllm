# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""No-GPU graph-mode propagation, production scope and accurate report labels."""

import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_report as report


class CudaGraphTests(unittest.TestCase):
    def options(self, mode="cudagraph"):
        return SimpleNamespace(
            mode=mode,
            compiler="flagtree",
            suite="fp8-prefill",
            candidate="production",
            accuracy_profile="iteration",
            reference_file="/tmp/reference.py",
            rounds=2,
            kernel_warmup_ms=20,
            kernel_rep_ms=20,
        )

    def test_mode_reaches_preflight_and_timed_workers(self):
        for mode in ("kernel", "cudagraph"):
            for flag in ("--preflight", "--worker"):
                command = bench.child_command(
                    self.options(mode), Path("/tmp/report"), flag
                )
                self.assertEqual(command[command.index("--mode") + 1], mode)
                self.assertIn(flag, command)
                self.assertEqual(
                    command[command.index("--candidate") + 1], "production"
                )

    def test_cli_selects_exact_existing10_fp8_cases_without_gpu(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                str(Path(bench.__file__)),
                "--compiler",
                "flagtree",
                "--mode",
                "cudagraph",
                "--suite",
                "fp8-prefill",
                "--list-cases",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        cases = json.loads(proc.stdout)
        self.assertEqual(cases, bench.workloads.cases("fp8-prefill"))
        self.assertEqual(len(cases), 10)

    def test_graph_mode_does_not_select_an_experimental_or_stock_path(self):
        for flags in (
            ("--compiler", "stock"),
            ("--candidate", "query-joint"),
            ("--suite", "upstream-prefill"),
        ):
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(Path(bench.__file__)),
                    "--compiler",
                    "flagtree",
                    "--mode",
                    "cudagraph",
                    "--list-cases",
                    *flags,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("CUDA Graph mode currently requires", proc.stderr)

    def test_report_does_not_label_graph_measurements_kernel(self):
        record = dict(
            case=dict(name="sample"),
            status="passed",
            timer=dict(mode="cudagraph"),
            stages={},
        )
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            report.show_timing(record)
        # Add a real row so the table title is emitted.
        record["stages"]["score"] = dict(
            status="passed",
            baseline_us=dict(median=2),
            candidate_us=dict(median=1),
            speedup=2,
            measurement_validated=True,
        )
        with contextlib.redirect_stdout(stream):
            report.show_timing(record)
        self.assertIn("cudagraph benchmark", stream.getvalue())
        self.assertNotIn("kernel benchmark", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
