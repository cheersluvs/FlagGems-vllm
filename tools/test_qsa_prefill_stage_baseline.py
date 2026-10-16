# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU-only checks for benchmark units, production routing and failure gates."""

import argparse
import contextlib
import importlib.util
import io
import itertools
import json
import os
import re
import subprocess
import sys
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

    def test_balanced_rounds_equalize_calls_and_rotate_modes(self):
        device = FakeDevice()
        calls = {"full": 0, "score": 0}

        def function(name):
            def fn():
                calls[name] += 1
                device.milliseconds += 0.5

            return fn

        functions = {name: function(name) for name in calls}
        graphs = {
            name: types.SimpleNamespace(replay=lambda fn=fn: [fn() for _ in range(8)])
            for name, fn in functions.items()
        }
        options = self.options(no_graph=False, iterations=100, graph_batch=8, rounds=3)
        samples, observations, order = bench.measure_rounds(
            functions,
            graphs,
            device,
            options,
            lambda label: label,
        )
        self.assertEqual(bench.timing_plan(options)["calls_per_round"], 104)
        self.assertEqual(bench.timing_plan(options)["graph_replays"], 13)
        self.assertEqual(calls, {"full": 104 * 3 * 3, "score": 104 * 3 * 3})
        self.assertEqual(len(observations), 4)
        self.assertEqual(
            [jobs[0][1] for jobs in order], ["wall_us", "event_us", "graph_us"]
        )
        self.assertEqual([jobs[0][0] for jobs in order], ["full", "score", "full"])
        for stage in samples.values():
            self.assertEqual(stage["event_us"], [500, 500, 500])
            self.assertEqual(stage["graph_us"], [500, 500, 500])

    def test_no_graph_needs_no_rounding_and_no_replays(self):
        options = self.options(no_graph=True, iterations=101, rounds=2)
        self.assertEqual(bench.timing_plan(options)["calls_per_round"], 101)
        self.assertEqual(bench.timing_plan(options)["graph_replays"], 0)
        device = FakeDevice()

        def fn():
            device.milliseconds += 1

        samples, _, order = bench.measure_rounds(
            {"full": fn}, {}, device, options, lambda x: x
        )
        self.assertEqual(samples["full"]["graph_us"], [])
        self.assertTrue(
            all(method != "graph_us" for jobs in order for _, method in jobs)
        )

    def test_profiles_report_real_callable_and_call_counts(self):
        def actual_validator():
            return sum(range(3))

        profile = bench.python_profile(actual_validator, FakeDevice(), 17)
        rows = [
            row
            for row in profile["top_functions"]
            if row["function"] == "actual_validator"
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["total_calls"], 17)
        self.assertEqual(profile["invocations"], 17)
        self.assertIn("actual_validator", profile["text"])

    def test_host_diagnostics_use_actual_validator_and_observed_allocations(self):
        calls = {"validation": 0, "entry": 0}
        allocations = []

        def validator(*args):
            calls["validation"] += 1

        def entry(*args):
            calls["entry"] += 1

        def empty(shape, **kwargs):
            allocations.append((shape, kwargs))

        device = FakeDevice()
        device.device = lambda _: contextlib.nullcontext()
        q = types.SimpleNamespace(shape=(3, 4, 128), device="cuda:0")
        logits = types.SimpleNamespace(
            shape=(3 * 1280,), dtype="float32", device="cuda:0"
        )
        scratch = types.SimpleNamespace(
            shape=(1024 * 1024,), dtype="uint8", device="cuda:0"
        )
        chunks = [
            dict(score_args=(None,) * 5 + (logits,), topk_args=(None,) * 3 + (scratch,))
        ]
        op = types.SimpleNamespace(
            _validate_inputs=validator, qsa_select_paged_prefill=entry
        )
        options = self.options(warmup=3, rounds=2, profile_iterations=8)
        # Tiny fake functions can consume zero process-clock ticks on macOS.
        # Real sample validation stays strict; use a clock with known progress.
        with patch.object(
            bench.time, "process_time", side_effect=itertools.count(0, 0.001)
        ):
            result = bench.host_diagnostics(
                op, (q,), chunks, types.SimpleNamespace(empty=empty), device, options
            )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(calls, {"validation": 3 + 8 * 2 + 8, "entry": 8})
        self.assertEqual(len(allocations), (3 + 8 * 2) * 2)
        self.assertEqual(
            allocations[0], (logits.shape, dict(dtype="float32", device="cuda:0"))
        )
        self.assertEqual(
            allocations[1], (scratch.shape, dict(dtype="uint8", device="cuda:0"))
        )
        self.assertEqual(
            set(result["microbench"]),
            {"noop", "validation", "device_guard", "outer_workspace_allocation"},
        )

    def test_telemetry_selects_physical_gpu_by_uuid(self):
        with patch.object(bench, "command", return_value={"returncode": 0}) as command:
            bench.telemetry("198a9496-3d7a-f01d-ee60-190fe0bc5b51", "round")
        self.assertEqual(command.call_count, 2)
        for call in command.call_args_list:
            self.assertIn("GPU-198a9496-3d7a-f01d-ee60-190fe0bc5b51", call.args[0])

    def test_case_provenance_and_boundary_coverage(self):
        # Explicit legacy diagnostics keep their boundary fixtures.
        cases = bench.cases()
        self.assertEqual(len(cases), len({c["name"] for c in cases}))
        self.assertEqual(sum(c.get("budget_mb") == 0 for c in cases), 2)
        self.assertTrue(any(c["kind"] == "zero" for c in cases))
        self.assertTrue(any(c["kind"] == "mixed" for c in cases))
        tail = [c for c in cases if c["kind"] == "tail"]
        self.assertEqual({c["dtype"] for c in tail}, {"bfloat16", "float8_e4m3fn"})
        self.assertTrue(all(c["budget_mb"] == 1 for c in tail))

    def test_observed_tail_chunks_and_frozen_topk_routing(self):
        calls = []

        class Logits:
            def clone(self):
                return object()

        def score(*args, **kwargs):
            calls.append(("score", args, kwargs))
            return Logits()

        def topk(*args, **kwargs):
            calls.append(("topk", args, kwargs))

        op = types.SimpleNamespace(_prefill_logits=score, persistent_topk=topk)

        def full(*args):
            for offset, count in ((0, 3), (3, 1)):
                logits = op._prefill_logits(
                    "q", "workspace", 4096, offset, count, launch_metadata="prepared"
                )
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
        self.assertTrue(
            all(
                c[2] == {"launch_metadata": "prepared"}
                for c in calls
                if c[0] == "score"
            )
        )

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
            suite="comprehensive",
            skip_accuracy=False,
            timeout=10,
            no_graph=True,
            warmup=1,
            iterations=2,
            rounds=3,
            graph_batch=4,
            host_profile=False,
            profile_iterations=200,
            validation_ab=False,
            validation_baseline="original",
            host_ab=False,
            single_row_ab=False,
            require_idle_gpu=False,
            mode="stages",
            kernel_warmup_ms=1000,
            kernel_rep_ms=100,
            resume_from=None,
        )
        values.update(kwargs)
        return argparse.Namespace(**values)

    def test_idle_rejection_prevents_correctness_and_workers_but_saves_report(self):
        guard = types.SimpleNamespace(
            idle=lambda *_: dict(status="rejected", reasons=["occupied"])
        )
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ, {"QSA_TRITON_RUNTIME_PATH": "", "CUDA_LAUNCH_BLOCKING": "0"}
        ), patch.object(bench, "command", return_value={}), patch.object(
            bench, "gpu_guard_module", return_value=guard
        ), patch.object(
            bench.subprocess, "run"
        ) as run, contextlib.redirect_stdout(
            io.StringIO()
        ):
            directory = Path(folder)
            self.assertEqual(
                bench.parent(
                    self.options(require_idle_gpu=True), bench.cases()[:1], directory
                ),
                2,
            )
            run.assert_not_called()
            summary = json.loads((directory / "stage-summary.json").read_text())
            self.assertTrue(summary["environment_rejected"])
            self.assertEqual(
                summary["accuracy"]["status"], "not_run_environment_rejected"
            )
            self.assertEqual(summary["cases"], [])

    def test_real_cli_rejects_numeric_device_without_gpu_imports(self):
        with tempfile.TemporaryDirectory() as folder:
            env = os.environ.copy()
            env.update(
                QSA_TRITON_RUNTIME_PATH="",
                CUDA_LAUNCH_BLOCKING="0",
                CUDA_VISIBLE_DEVICES="0",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(bench.ROOT / "tools/benchmark_qsa_prefill_stages.py"),
                    "--single-row-ab",
                    "--require-idle-gpu",
                    "--report-dir",
                    folder,
                ],
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("complete GPU UUID", result.stdout)
            summary = json.loads((Path(folder) / "stage-summary.json").read_text())
            self.assertTrue(summary["environment_rejected"])
            self.assertEqual(summary["cases"], [])
            self.assertEqual(len(summary["manifest"]["cases"]), 22)

    def test_trailing_utilization_retries_are_bounded_and_saved(self):
        for eventually_idle in (True, False):
            verdicts = iter(
                [dict(status="rejected", reasons=["util"], retryable_utilization=True)]
                + [
                    dict(
                        status="passed" if eventually_idle else "rejected",
                        reasons=[] if eventually_idle else ["util"],
                        retryable_utilization=not eventually_idle,
                    )
                    for _ in range(2)
                ]
            )
            guard = types.SimpleNamespace(idle=lambda *_: next(verdicts))
            with tempfile.TemporaryDirectory() as folder, patch.object(
                bench.time, "sleep"
            ) as sleep, contextlib.redirect_stdout(io.StringIO()):
                summary = {}
                self.assertEqual(
                    bench.idle_gpu_gate(summary, Path(folder), {}, "start", guard),
                    eventually_idle,
                )
                self.assertEqual(sleep.call_count, 1 if eventually_idle else 2)
                saved = json.loads((Path(folder) / "stage-summary.json").read_text())
                self.assertEqual(
                    len(saved["idle_gpu_checks"]), 2 if eventually_idle else 3
                )
                self.assertEqual(
                    bool(saved.get("environment_rejected")), not eventually_idle
                )

    def test_gpu_is_rechecked_before_each_worker_and_guard_flag_is_forwarded(self):
        checks = iter(
            [
                dict(status="passed", reasons=[]),
                dict(status="passed", reasons=[]),
                dict(status="rejected", reasons=["new job"]),
            ]
        )
        guard = types.SimpleNamespace(idle=lambda *_: next(checks))

        def worker(command, **kwargs):
            self.assertIn("--require-idle-gpu", command)
            name = command[command.index("--worker") + 1]
            directory = Path(command[command.index("--report-dir") + 1])
            bench.write_json(
                directory / f"{name}.json",
                dict(case=dict(name=name), status="passed", stages={}),
            )
            return types.SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ, {"QSA_TRITON_RUNTIME_PATH": "", "CUDA_LAUNCH_BLOCKING": "0"}
        ), patch.object(bench, "command", return_value={}), patch.object(
            bench, "gpu_guard_module", return_value=guard
        ), patch.object(
            bench.subprocess, "run", side_effect=worker
        ) as run, contextlib.redirect_stdout(
            io.StringIO()
        ):
            directory = Path(folder)
            self.assertEqual(
                bench.parent(
                    self.options(require_idle_gpu=True, skip_accuracy=True),
                    bench.cases()[:2],
                    directory,
                ),
                2,
            )
            self.assertEqual(run.call_count, 1)
            summary = json.loads((directory / "stage-summary.json").read_text())
            self.assertEqual(len(summary["cases"]), 1)
            self.assertEqual(summary["stopped_before_case"], "core_1")

    def test_mid_timing_rejection_stops_all_later_cases(self):
        guard = types.SimpleNamespace(idle=lambda *_: dict(status="passed", reasons=[]))

        def worker(command, **kwargs):
            name = command[command.index("--worker") + 1]
            directory = Path(command[command.index("--report-dir") + 1])
            bench.write_json(
                directory / f"{name}.json",
                dict(
                    case=dict(name=name),
                    status="failed",
                    stages={},
                    performance_guard=dict(status="rejected", reasons=["thermal"]),
                ),
            )
            return types.SimpleNamespace(returncode=1)

        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ, {"QSA_TRITON_RUNTIME_PATH": "", "CUDA_LAUNCH_BLOCKING": "0"}
        ), patch.object(bench, "command", return_value={}), patch.object(
            bench, "gpu_guard_module", return_value=guard
        ), patch.object(
            bench.subprocess, "run", side_effect=worker
        ) as run, contextlib.redirect_stdout(
            io.StringIO()
        ):
            directory = Path(folder)
            self.assertEqual(
                bench.parent(
                    self.options(require_idle_gpu=True, skip_accuracy=True),
                    bench.cases()[:2],
                    directory,
                ),
                2,
            )
            self.assertEqual(run.call_count, 1)
            summary = json.loads((directory / "stage-summary.json").read_text())
            self.assertTrue(summary["environment_rejected"])
            self.assertEqual(
                summary["cases"][0]["performance_guard"]["status"], "rejected"
            )

    def test_guarded_monitor_saves_rejection_evidence_before_raising(self):
        point = dict(label="after_round_1")
        verdict = dict(status="rejected", reasons=["thermal"])
        guard = types.SimpleNamespace(
            timing=lambda *_: verdict,
            process_query=lambda *_: {},
            self_pids=lambda: [123],
        )
        with tempfile.TemporaryDirectory() as folder, patch.object(
            bench, "telemetry", return_value=point
        ):
            report = Path(folder) / "case.json"
            result = dict(status="running")
            with self.assertRaisesRegex(RuntimeError, "GPU environment rejected"):
                bench.guarded_telemetry("uuid", "after_round_1", result, report, guard)
            saved = json.loads(report.read_text())
            self.assertEqual(saved["telemetry"][0]["gpu_guard"], verdict)
            self.assertEqual(saved["performance_guard"]["status"], "rejected")

    def test_guard_rejection_records_process_identity_and_nvml_names(self):
        verdict = dict(
            status="rejected",
            reasons=["unattributed PID"],
            processes=[dict(pid=123), dict(pid=456)],
        )
        guard = types.SimpleNamespace(
            timing=lambda *_: verdict,
            process_query=lambda *_: {},
            self_pids=lambda: [123],
            process_details=lambda pids, _: [
                dict(pid=p, program="python", script="job.py") for p in pids
            ],
            process_names=lambda *_: dict(processes=[dict(pid=456, program="python")]),
        )
        with tempfile.TemporaryDirectory() as folder, patch.object(
            bench, "telemetry", return_value={}
        ), patch.object(
            bench,
            "command",
            return_value=dict(returncode=0, stdout="GPU,456,python job.py,610"),
        ):
            report = Path(folder) / "rejected.json"
            with self.assertRaises(RuntimeError):
                bench.guarded_telemetry("uuid", "before_timing", {}, report, guard)
            saved = json.loads(report.read_text())
            check = saved["telemetry"][0]["gpu_guard"]
            self.assertEqual(check["status"], "rejected")
            self.assertEqual(len(check["process_details"]), 2)
            self.assertEqual(
                check["nvml_process_names"]["processes"][0]["program"], "python"
            )

    def test_kernel_mode_no_score_is_na_without_invented_latency(self):
        with tempfile.TemporaryDirectory() as folder:
            result = {}
            report = Path(folder) / "zero.json"
            bench.run_score_kernel(
                self.options(mode="kernel"),
                None,
                None,
                (),
                [],
                result,
                report,
                None,
                None,
                None,
            )
            saved = json.loads(report.read_text())
            self.assertEqual(saved["status"], "passed")
            self.assertEqual(saved["score_kernel"]["status"], "not_applicable_no_score")
            self.assertNotIn("speedup", saved["score_kernel"])
            self.assertFalse(saved["score_kernel"]["includes_topk"])

    def test_resume_skips_completed_cases_and_labels_retained_metrics(self):
        retained = dict(
            case=bench.cases()[0],
            status="passed",
            score_kernel=dict(
                status="passed",
                baseline_us=dict(median=10),
                candidate_us=dict(median=8),
                speedup=1.25,
                paired_ratios=dict(median=1.25),
            ),
            measurement_origin=dict(checkpoint_sha256="old"),
        )
        metadata = dict(
            completed_names=["core_0"],
            prior_accuracy=dict(status="passed", counts=dict(passed=161)),
        )
        module = types.SimpleNamespace(load=lambda *_: (metadata, [retained]))

        def worker(command, **kwargs):
            self.assertIn("--resume-from", command)
            name = command[command.index("--worker") + 1]
            self.assertEqual(name, "core_1")
            directory = Path(command[command.index("--report-dir") + 1])
            bench.write_json(
                directory / f"{name}.json",
                dict(
                    case=dict(name=name),
                    status="passed",
                    stages={},
                    score_kernel=dict(
                        status="passed",
                        baseline_us=dict(median=10),
                        candidate_us=dict(median=9),
                        speedup=10 / 9,
                        paired_ratios=dict(median=10 / 9),
                    ),
                ),
            )
            return types.SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ, {"QSA_TRITON_RUNTIME_PATH": "", "CUDA_LAUNCH_BLOCKING": "0"}
        ), patch.object(bench, "command", return_value={}), patch.object(
            bench, "score_resume_module", return_value=module
        ), patch.object(
            bench.subprocess, "run", side_effect=worker
        ) as run, contextlib.redirect_stdout(
            io.StringIO()
        ):
            directory = Path(folder)
            self.assertEqual(
                bench.parent(
                    self.options(
                        mode="kernel",
                        single_row_ab=True,
                        skip_accuracy=True,
                        resume_from="prior.json",
                    ),
                    bench.cases()[:2],
                    directory,
                ),
                0,
            )
            self.assertEqual(run.call_count, 1)
            summary = json.loads((directory / "stage-summary.json").read_text())
            self.assertEqual(len(summary["retained_cases"]), 1)
            text = (directory / "score-kernel-ab.tsv").read_text()
            self.assertIn("1.25\t1.25\tretained", text)
            self.assertIn("\tcurrent", text)

    def test_kernel_mode_forwarding_and_separate_score_speedup_tsv(self):
        def worker(command, **kwargs):
            self.assertEqual(command[command.index("--mode") + 1], "kernel")
            self.assertEqual(command[command.index("--kernel-warmup-ms") + 1], "1000")
            self.assertEqual(command[command.index("--kernel-rep-ms") + 1], "100")
            name = command[command.index("--worker") + 1]
            directory = Path(command[command.index("--report-dir") + 1])
            bench.write_json(
                directory / f"{name}.json",
                dict(
                    case=dict(name=name),
                    status="passed",
                    stages={},
                    score_kernel=dict(
                        status="passed",
                        baseline_us=dict(median=10),
                        candidate_us=dict(median=8),
                        speedup=1.25,
                        paired_ratios=dict(median=1.25),
                    ),
                ),
            )
            return types.SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ, {"QSA_TRITON_RUNTIME_PATH": "", "CUDA_LAUNCH_BLOCKING": "0"}
        ), patch.object(bench, "command", return_value={}), patch.object(
            bench.subprocess, "run", side_effect=worker
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            directory = Path(folder)
            self.assertEqual(
                bench.parent(
                    self.options(mode="kernel", single_row_ab=True, skip_accuracy=True),
                    bench.cases()[:1],
                    directory,
                ),
                0,
            )
            text = (directory / "score-kernel-ab.tsv").read_text()
            self.assertIn("core_0\tpassed\t10\t8\t1.25\t1.25", text)
            self.assertFalse((directory / "single-row-ab.tsv").exists())
            summary = json.loads((directory / "stage-summary.json").read_text())
            self.assertEqual(
                summary["manifest"]["timing_plan"]["scope"], "score_only_no_topk"
            )
            self.assertNotIn("calls_per_round", summary["manifest"]["timing_plan"])

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
                    self.options(
                        skip_accuracy=True,
                        validation_ab=True,
                        validation_baseline="overlap-retained",
                    ),
                    bench.cases()[:2],
                    directory,
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
            for command, _ in executions:
                self.assertIn("--validation-ab", command)
                self.assertEqual(
                    command[command.index("--validation-baseline") + 1],
                    "overlap-retained",
                )

    def test_host_ab_worker_selection_and_tsv_use_host_baseline(self):
        commands = []

        def execute(command, **kwargs):
            commands.append(command)
            directory = Path(command[command.index("--report-dir") + 1])
            name = command[command.index("--worker") + 1]
            bench.write_json(
                directory / f"{name}.json",
                dict(
                    case=dict(name=name),
                    status="passed",
                    stages=dict(
                        full=dict(wall_us=dict(median=10)),
                        full_host_baseline=dict(wall_us=dict(median=12)),
                    ),
                    host_ab=dict(
                        speedup=dict(
                            wall_us=dict(
                                ratio_of_medians=1.2,
                                paired_round_ratios=dict(median=1.1),
                            )
                        )
                    ),
                ),
            )
            return types.SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ, {"QSA_TRITON_RUNTIME_PATH": "", "CUDA_LAUNCH_BLOCKING": "0"}
        ), patch.object(bench, "command", return_value={}), patch.object(
            bench.subprocess, "run", side_effect=execute
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            directory = Path(folder)
            options = self.options(skip_accuracy=True, host_ab=True)
            self.assertEqual(bench.parent(options, bench.cases()[:1], directory), 0)
            self.assertIn("--host-ab", commands[0])
            self.assertNotIn("--validation-ab", commands[0])
            self.assertIn(
                "core_0\tpassed\twall_us\t12\t10\t1.2\t1.1",
                (directory / "host-ab.tsv").read_text(),
            )
            self.assertFalse((directory / "validation-ab.tsv").exists())
            self.assertEqual(
                bench.ab_configuration(options)[2], "tools.qsa_prefill_chunk_baseline"
            )

    def test_single_row_mode_runs_extra_accuracy_and_writes_its_comparison(self):
        calls = []

        def execute(command, **kwargs):
            calls.append(command)
            if "pytest" in command:
                output = Path(command[command.index("--output") + 1])
                bench.write_json(output, {"fixture": {"result": "passed"}})
            else:
                directory = Path(command[command.index("--report-dir") + 1])
                name = command[command.index("--worker") + 1]
                bench.write_json(
                    directory / f"{name}.json",
                    dict(
                        case=dict(name=name),
                        status="passed",
                        stages=dict(
                            full=dict(wall_us=dict(median=10)),
                            full_matmul_baseline=dict(wall_us=dict(median=12)),
                        ),
                        single_row_ab=dict(
                            speedup=dict(
                                wall_us=dict(
                                    ratio_of_medians=1.2,
                                    paired_round_ratios=dict(median=1.1),
                                )
                            )
                        ),
                    ),
                )
            return types.SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ, {"QSA_TRITON_RUNTIME_PATH": "", "CUDA_LAUNCH_BLOCKING": "0"}
        ), patch.object(bench, "command", return_value={}), patch.object(
            bench.subprocess, "run", side_effect=execute
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            options = self.options(single_row_ab=True)
            directory = Path(folder)
            self.assertEqual(bench.parent(options, bench.cases(True)[:1], directory), 0)
            self.assertIn("tests/test_qsa_prefill_single_row_trial.py", calls[0])
            self.assertIn("tests/test_qsa_select_paged_prefill.py", calls[0])
            self.assertIn("--single-row-ab", calls[1])
            self.assertNotIn("--host-ab", calls[1])
            self.assertIn(
                "core_0\tpassed\twall_us\t12\t10\t1.2\t1.1",
                (directory / "single-row-ab.tsv").read_text(),
            )
        extra = bench.cases(True)[len(bench.cases()) :]
        self.assertEqual(len(extra), 14)
        self.assertEqual(sum(c["kind"] == "tail_one" for c in extra), 2)
        self.assertEqual(bench.ab_configuration(options)[0], "single_row_ab")


if __name__ == "__main__":
    unittest.main()
