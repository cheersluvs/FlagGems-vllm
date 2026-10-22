# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Standalone pytest may be skipped;timed checks/guards and honest status stay."""

import ast
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_stage_joint_trial as trial
from tools import test_qsa_prefill_stage_joint as joint_tests


class BenchmarkOnlyTests(unittest.TestCase):
    def options(self, directory, skip):
        argv = [
            "benchmark",
            "--compiler",
            "flagtree",
            "--candidate",
            "query-stage-joint",
            "--suite",
            "scored-prefill",
            "--compare-previous-tuning",
            "--previous-tuning",
            "combined",
            "--report-dir",
            str(directory),
        ]
        if skip:
            argv.append("--skip-accuracy")
        with patch.object(sys, "argv", argv), patch.object(
            bench, "parent", return_value=0
        ) as parent:
            self.assertEqual(bench.main(), 0)
            options, selected, _ = parent.call_args.args
        self.assertEqual(len(selected), 30)
        self.assertEqual(options.skip_accuracy, skip)
        return options

    def test_real_parent_skips_only_standalone_gate_and_preserves_default(self):
        for skip in (False, True):
            with self.subTest(skip=skip), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                options = self.options(directory, skip)
                source = directory / "native-input.py"
                source.write_bytes(b"native")
                options.reference_file = str(source)
                case = bench.workloads.cases("scored-prefill")[0]
                phases = []

                def child(command, **kw):
                    if "--preflight" in command:
                        phases.append("preflight")
                        bench.support.write_json(
                            directory / "preflight.json",
                            dict(status="passed", runtime={}),
                        )
                    else:
                        phases.append("worker")
                        self.assertIn("--worker", command)
                        bench.support.write_json(
                            directory / (case["name"] + ".json"),
                            dict(status="passed", case=case, stages={}, runtime={}),
                        )
                    self.assertNotIn("--skip-accuracy", command)  # Parent-only flag.
                    return types.SimpleNamespace(returncode=0)

                def accuracy(options, directory, summary, *args):
                    phases.append("pytest")
                    summary["accuracy"] = dict(status="passed")

                # Parent work is real;only GPU/subprocess IO and table formatting
                # are mocked. Default and skipped paths both reach preflight/workers.
                with patch.dict(
                    os.environ,
                    {
                        "CUDA_VISIBLE_DEVICES": "GPU-00000000-0000-0000-0000-000000000001"
                    },
                ), patch.object(bench, "configure_environment"), patch.object(
                    bench.support, "gpu_guard_module"
                ), patch.object(
                    bench, "idle_gate", return_value=True
                ) as idle, patch.object(
                    bench, "runtime_identity", return_value="fixed"
                ), patch.object(
                    bench, "run_accuracy_gate", side_effect=accuracy
                ) as gate, patch.object(
                    bench.subprocess, "run", side_effect=child
                ), patch.object(
                    bench.support, "command", return_value={}
                ), patch.object(
                    bench.ab, "REFERENCE_SHA256", hashlib.sha256(b"native").hexdigest()
                ), patch.object(
                    bench, "write_table"
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    self.assertEqual(bench.parent(options, [case], directory), 0)
                self.assertEqual(
                    phases,
                    (
                        ["preflight", "worker"]
                        if skip
                        else ["preflight", "pytest", "worker"]
                    ),
                )
                self.assertEqual(gate.call_count, 0 if skip else 1)
                self.assertEqual(idle.call_count, 2)
                summary = json.loads((directory / "stage-summary.json").read_text())
                self.assertEqual(summary["manifest"]["options"]["skip_accuracy"], skip)
                self.assertEqual(summary["status"], "passed")
                if skip:
                    self.assertEqual(
                        summary["accuracy"]["status"], "skipped_by_request"
                    )
                    self.assertEqual(summary["accuracy"]["counts"], {})
                    self.assertFalse(summary["accuracy"]["full_suite_validated"])
                    self.assertEqual(
                        json.loads((directory / "accuracy-skipped.json").read_text()),
                        summary["accuracy"],
                    )
                    self.assertFalse((directory / "accuracy-full.json").exists())

    def test_skip_never_turns_missing_accuracy_into_acceptance(self):
        summary = joint_tests.StageJointTests().summary()
        self.assertTrue(trial.acceptance(summary)["measurement_valid"])
        summary["manifest"]["options"]["skip_accuracy"] = True
        summary["accuracy"] = dict(status="skipped_by_request", counts={})
        result = trial.acceptance(summary)
        self.assertFalse(result["measurement_valid"])
        self.assertIsNone(result["target_met"])
        self.assertFalse(result["formal_production_acceptance"])

    def test_skip_parser_scope_and_worker_numeric_guards_remain(self):
        for args in (
            ["--skip-accuracy"],
            ["--compiler", "flagtree", "--skip-accuracy", "--worker", "x"],
        ):
            with patch.object(
                sys, "argv", ["benchmark", *args]
            ), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    bench.main()
                self.assertEqual(error.exception.code, 2)
        tree = ast.parse(Path(bench.__file__).read_text())
        worker = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "worker"
        )
        self.assertNotIn("skip_accuracy", ast.unparse(worker))
        self.assertIn("guarded_telemetry", ast.unparse(worker))
        self.assertIn("validate_pipeline", ast.unparse(worker))
        self.assertIn("validate_topk", ast.unparse(worker))


if __name__ == "__main__":
    unittest.main()
