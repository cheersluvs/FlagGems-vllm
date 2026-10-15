# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU-only checks for benchmark units, production routing and failure gates."""

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "qsa_stage_bench", Path(__file__).with_name("benchmark_qsa_prefill_stages.py")
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class FakeDevice:
    def __init__(self):
        self.milliseconds = 0
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1

    def Event(self, enable_timing):
        assert enable_timing
        device = self

        class Event:
            def record(self):
                self.timestamp = device.milliseconds

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return other.timestamp - self.timestamp

        return Event()


class BenchmarkTests(unittest.TestCase):
    def test_documented_upload_failure_keeps_errexit_parent_alive(self):
        text = (bench.ROOT / "docs/qsa_prefill_stage_baseline.md").read_text()
        block = next(
            code
            for code in re.findall(r"```bash\n(.*?)```", text, re.S)
            if code.startswith("if bash <<'QSA_UPLOAD'")
        )
        with tempfile.TemporaryDirectory() as folder:
            # Force nonexistent paths; this test never reaches Git mutations.
            block = block.replace(
                "qsa_code_repo=/tmp/flaggems-qsa-stage.REPLACE",
                f"qsa_code_repo={folder}/missing-repo",
            ).replace(
                "qsa_report=/tmp/qsa-stage-h800.REPLACE",
                f"qsa_report={folder}/missing-report",
            )
            result = subprocess.run(
                ["bash", "-c", "set -euo pipefail\n" + block + "\necho SESSION_ALIVE"],
                text=True,
                capture_output=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SESSION_ALIVE", result.stdout)
        self.assertIn("Replace qsa_code_repo", result.stderr)
        self.assertNotIn("Result branch:", result.stdout)

    def test_timer_units_and_graph_normalization(self):
        device = FakeDevice()

        def one_call():
            device.milliseconds += 0.5

        self.assertEqual(bench.timed_batch(one_call, device, 20, "event_us"), 500)
        self.assertEqual(bench.timed_batch(one_call, device, 20, "graph_us", 8), 62.5)
        with patch.object(bench.time, "perf_counter", side_effect=[10, 10.02]):
            self.assertAlmostEqual(
                bench.timed_batch(one_call, device, 20, "wall_us"), 1000
            )
        self.assertEqual(device.synchronizations, 4)

    def test_case_provenance_and_boundary_coverage(self):
        tree = ast.parse(
            (bench.ROOT / "benchmark/test_qsa_select_paged_prefill_perf.py").read_text()
        )
        core = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "CORE_SHAPES" for t in node.targets
            )
        )
        cases = bench.cases()
        self.assertEqual(
            [tuple(c["shape"]) for c in cases if c["kind"] == "core"], core
        )
        self.assertEqual(len(cases), len({c["name"] for c in cases}))
        self.assertEqual(sum(c.get("budget_mb") == 0 for c in cases), 2)
        self.assertTrue(any(c["kind"] == "zero" for c in cases))
        self.assertTrue(any(c["kind"] == "mixed" for c in cases))

    def test_observed_tail_chunks_and_frozen_topk_routing(self):
        calls = []

        class Logits:
            def clone(self):
                return object()

        def score(*args):
            calls.append(("score", args))
            return Logits()

        def topk(*args, **kwargs):
            calls.append(("topk", args, kwargs))

        op = types.SimpleNamespace(_prefill_logits=score, persistent_topk=topk)

        def full(*args):
            for offset, count in ((0, 3), (3, 1)):
                logits = op._prefill_logits("q", "workspace", 4096, offset, count)
                op.persistent_topk(logits, "lengths", "out", "scratch", k=512)

        op.qsa_select_paged_prefill = full
        chunks = bench.observe_pipeline(op, ())
        self.assertIs(op._prefill_logits, score)
        self.assertIs(op.persistent_topk, topk)
        self.assertEqual([c["score_args"][-2:] for c in chunks], [(0, 3), (3, 1)])
        functions = bench.stage_functions(op, (), chunks)
        calls.clear()
        functions["topk"]()
        self.assertEqual([c[0] for c in calls], ["topk", "topk"])
        for call, chunk in zip(calls, chunks):
            self.assertIs(call[1][0], chunk["frozen"])
            self.assertEqual(call[1][1:], ("lengths", "out", "scratch"))
            self.assertEqual(call[2], {"k": 512})
        calls.clear()
        functions["pipeline_preallocated"]()
        self.assertEqual([c[0] for c in calls], ["score", "topk", "score", "topk"])

    def test_observer_restores_globals_after_failure(self):
        def fail(*args):
            raise RuntimeError("launch failed")

        op = types.SimpleNamespace(_prefill_logits=fail, persistent_topk=object())
        original_topk = op.persistent_topk
        op.qsa_select_paged_prefill = lambda: op._prefill_logits()
        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            bench.observe_pipeline(op, ())
        self.assertIs(op._prefill_logits, fail)
        self.assertIs(op.persistent_topk, original_topk)

    def test_select_all_has_no_artificial_stage_timings(self):
        op = types.SimpleNamespace(
            qsa_select_paged_prefill=lambda: None,
            _prefill_logits=object(),
            persistent_topk=object(),
        )
        chunks = bench.observe_pipeline(op, ())
        self.assertEqual(chunks, [])
        self.assertEqual(list(bench.stage_functions(op, (), chunks)), ["full"])

    def options(self, **kwargs):
        values = dict(
            skip_accuracy=False,
            timeout=10,
            no_graph=True,
            warmup=1,
            iterations=2,
            rounds=3,
            graph_batch=4,
        )
        values.update(kwargs)
        return argparse.Namespace(**values)

    def test_failed_accuracy_blocks_all_timings(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ, {"QSA_TRITON_RUNTIME_PATH": "", "CUDA_LAUNCH_BLOCKING": "0"}
        ), patch.object(bench, "command", return_value={}), patch.object(
            bench.subprocess, "run", return_value=types.SimpleNamespace(returncode=1)
        ) as run, contextlib.redirect_stdout(
            io.StringIO()
        ):
            directory = Path(folder)
            self.assertEqual(
                bench.parent(self.options(), bench.cases()[:1], directory), 1
            )
            self.assertEqual(run.call_count, 1)
            self.assertIn("pytest", run.call_args.args[0])
            summary = json.loads((directory / "stage-summary.json").read_text())
            self.assertEqual(summary["accuracy"]["status"], "failed")
            self.assertEqual(summary["cases"], [])

    def test_skipped_or_missing_accuracy_cannot_pass_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "accuracy.json"
            self.assertIsNotNone(bench.accuracy_results(path)[1])
            for records in (
                {},
                {"a": {"result": "skipped"}},
                {"a": {"result": "passed"}, "b": {"result": "skipped"}},
            ):
                bench.write_json(path, records)
                self.assertIsNotNone(bench.accuracy_results(path)[1])
            bench.write_json(path, {"a": {"result": "passed"}})
            self.assertEqual(bench.accuracy_results(path), ({"passed": 1}, None))
            bench.write_json(
                path,
                {
                    "a": {"result": "passed"},
                    "tests/test_qsa_select_paged_prefill.py::test_noncurrent_cuda_device": {
                        "result": "skipped",
                        "reason": "a second SM90 device is needed for device-guard validation",
                    },
                },
            )
            self.assertEqual(
                bench.accuracy_results(path), ({"passed": 1, "skipped": 1}, None)
            )

    def test_worker_timeout_is_saved_and_later_case_continues(self):
        executions = []

        def execute(command, **kwargs):
            executions.append((command, kwargs))
            if len(executions) == 1:
                raise subprocess.TimeoutExpired(command, 10)
            name = command[command.index("--worker") + 1]
            folder = Path(command[command.index("--report-dir") + 1])
            bench.write_json(
                folder / f"{name}.json", dict(case={"name": name}, status="passed")
            )
            return types.SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ,
            {
                "QSA_TRITON_RUNTIME_PATH": "",
                "CUDA_LAUNCH_BLOCKING": "0",
                "PYTHONPATH": "/wrong/runtime",
            },
        ), patch.object(bench, "command", return_value={}), patch.object(
            bench.subprocess, "run", side_effect=execute
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            directory = Path(folder)
            self.assertEqual(
                bench.parent(
                    self.options(skip_accuracy=True), bench.cases()[:2], directory
                ),
                1,
            )
            summary = json.loads((directory / "stage-summary.json").read_text())
            self.assertEqual(
                [c["status"] for c in summary["cases"]], ["failed", "passed"]
            )
            self.assertIn("timeout", summary["cases"][0]["error"])
            paths = [x[1]["env"]["TRITON_CACHE_DIR"] for x in executions]
            self.assertNotEqual(*paths)
            self.assertNotIn("/wrong/runtime", executions[0][1]["env"]["PYTHONPATH"])


if __name__ == "__main__":
    unittest.main()
