# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU gates for PR byte identity and completeness of the rerun."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_pr_validation as validation


class PRValidationTests(unittest.TestCase):
    def test_exact_pr_checkout_and_ten_cases(self):
        pins = validation.verify_checkout()
        self.assertEqual(len(pins["required_test_functions"]), 24)
        from tools import benchmark_qsa_prefill_triton_stages as timer
        from tools import qsa_prefill_fp8_workloads as fp8

        self.assertEqual(len(fp8.cases()), 10)
        self.assertTrue(all((timer.ROOT / name).is_file() for name in timer.SOURCES))

    def test_new_dependency_bootstrap_reaches_parent_but_results_are_rejected(self):
        import contextlib
        import io
        import sys

        from tools import benchmark_qsa_prefill_triton_stages as timer
        from tools.qsa_prefill_pr_dependencies import WHEEL

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in (
                "runner.txt",
                "compat134.json",
                "compat134-package.json",
                "flagtree134-tools.json",
                "toolchain-downloads.json",
                "flag-gems-dependency.json",
                WHEEL,
            ):
                (folder / name).write_text("bootstrap placeholder; not executed")
            (folder / "python-deps").mkdir()
            args = [
                "timer",
                "--compiler",
                "flagtree",
                "--candidate",
                "production",
                "--accuracy-profile",
                "full",
                "--report-dir",
                directory,
            ]
            with patch.object(sys, "argv", args), patch.object(
                timer, "parent", return_value=0
            ) as parent:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(timer.main(), 0)
                self.assertEqual(len(parent.call_args[0][1]), 10)
            (folder / "stage-summary.json").write_text("{}")
            with patch.object(sys, "argv", args), patch.object(
                timer, "parent"
            ) as parent:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        timer.main()
                self.assertEqual(error.exception.code, 2)
                parent.assert_not_called()

    def test_native_schema_registration_precedes_flaggems_import(self):
        import ast
        import inspect

        from tools import benchmark_qsa_prefill_triton_stages as timer

        tree = ast.parse(inspect.getsource(timer.runtime))
        native_load = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "native" for t in n.targets)
        ]
        gems_import = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Import)
            and any(alias.name == "flaggems_vllm" for alias in n.names)
        ]
        self.assertEqual(len(native_load), 1)
        self.assertEqual(len(gems_import), 1)
        self.assertLess(native_load[0], gems_import[0])
        self.assertIn(
            "faulthandler.enable(all_threads=True)", inspect.getsource(timer.runtime)
        )

    def test_byte_change_is_rejected(self):
        original = validation.PINS
        pins = json.loads(original.read_text())
        pins["files"]["src/flaggems_vllm/ops/persistent_topk.py"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pins.json"
            path.write_text(json.dumps(pins))
            with patch.object(validation, "PINS", path):
                with self.assertRaisesRegex(RuntimeError, "persistent_topk"):
                    validation.verify_checkout()

    def test_missing_or_skipped_direct_comparison_is_rejected(self):
        pins = json.loads(validation.PINS.read_text())
        records = {
            "tests/test_qsa_select_paged_prefill.py::" + name: {"result": "passed"}
            for name in pins["required_test_functions"]
        }
        self.assertIsNone(validation.validate_records(records))
        key = "tests/test_qsa_select_paged_prefill.py::test_against_pinned_vllm"
        records.pop(key)
        self.assertIn("Missing", validation.validate_records(records))
        records[key] = {"result": "skipped"}
        self.assertIn("Nonpassing", validation.validate_records(records))

    def test_only_second_gpu_skip_is_optional(self):
        pins = json.loads(validation.PINS.read_text())
        records = {
            "tests/test_qsa_select_paged_prefill.py::" + name: {"result": "passed"}
            for name in pins["required_test_functions"]
        }
        records[
            "tests/test_qsa_select_paged_prefill.py::test_noncurrent_cuda_device"
        ] = {"result": "skipped", "reason": "a second SM90 device is needed"}
        self.assertIsNone(validation.validate_records(records))

    def test_incomplete_report_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            path.joinpath("stage-summary.json").write_text(
                json.dumps(
                    {"status": "passed", "accuracy": {"status": "passed"}, "cases": []}
                )
            )
            self.assertEqual(validation.summarize(path), 2)
            summary = json.loads(
                path.joinpath("pr-validation-summary.json").read_text()
            )
            self.assertEqual(summary["status"], "incomplete_or_failed")


if __name__ == "__main__":
    unittest.main()
