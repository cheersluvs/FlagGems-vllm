# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Diagnostic orchestration: never pass profiler samples as benchmark results."""

import contextlib
import copy
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_profiling as profiling


class ProfilingTests(unittest.TestCase):
    def options(self, root):
        return types.SimpleNamespace(
            cases=list(profiling.CASES),
            report_dir=str(root),
            flagtree="/opt/flagtree",
            timeout=1800,
            ncu=None,
            worker=profiling.CASES[0],
            rounds=2,
        )

    def record(self):
        trace = dict(
            offset=0,
            rows=17,
            width=100,
            route="rectangular",
            constants=dict(TILE_R=32),
            compiled=dict(cubin_sha256="abc"),
        )
        return dict(
            status="passed",
            runtime={},
            inputs=["same"],
            chunks=[dict(rows=17)],
            kernel_resources=dict(
                native=dict(status="collected", kernels=[dict(cubin_sha256="native")])
            ),
            cache_granularity=dict(production=[trace], production_baseline=[trace]),
        )

    def test_representatives_are_existing_scored_cases(self):
        cases = {x["name"] for x in bench.workloads.cases("scored-prefill")}
        self.assertEqual(len(profiling.CASES), 5)
        self.assertTrue(set(profiling.CASES) <= cases)

    def test_capture_does_not_wrap_timing_autotune_or_topk(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            options = self.options(root)
            measured = self.record()
            (root / (options.worker + ".json")).write_text(json.dumps(measured))
            calls = []
            cudart = types.SimpleNamespace(
                cudaProfilerStart=lambda: calls.append("start") or 0,
                cudaProfilerStop=lambda: calls.append("stop") or 0,
            )
            torch = types.SimpleNamespace(
                cuda=types.SimpleNamespace(
                    cudart=lambda: cudart,
                    nvtx=types.SimpleNamespace(
                        range_push=lambda x: calls.append(x), range_pop=lambda: None
                    ),
                )
            )
            device = types.SimpleNamespace(synchronize=lambda: None)

            def fn(side):
                return lambda: calls.append(side)

            functions = dict(
                score=dict(baseline=fn("native"), candidate=fn("candidate")),
                score_vs_production=dict(baseline=fn("production")),
            )
            result = copy.deepcopy(measured)
            with patch.object(bench, "runtime_identity", return_value={}):
                profiling.capture(
                    options,
                    result,
                    root / "profile.json",
                    functions,
                    lambda n: dict(passed=n),
                    torch,
                    device,
                    lambda _: None,
                )
            inside = calls[calls.index("start") + 1 : calls.index("stop")]
            self.assertEqual(
                inside,
                [
                    "qsa_score/native",
                    "native",
                    "qsa_score/production",
                    "production",
                    "qsa_score/candidate",
                    "candidate",
                ],
            )
            self.assertEqual(result["profiling"]["expected_profiled_launches"], 3)
            self.assertFalse(result["profiling"]["acceptance_eligible"])
            self.assertIn("validation_after", result["profiling"])

    def test_config_drift_rejects_before_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            options = self.options(root)
            record = self.record()
            (root / (options.worker + ".json")).write_text(json.dumps(record))
            changed = copy.deepcopy(record)
            changed["cache_granularity"]["production"][0]["constants"]["TILE_R"] = 64
            with self.assertRaisesRegex(RuntimeError, "differ"):
                profiling.capture(
                    options, changed, root / "p.json", {}, None, None, None, None
                )

    def test_csv_requires_metrics_not_error_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "c.csv"
            path.write_text("==ERROR== ERR_NVGPUCTRPERM\n")
            self.assertEqual(profiling.exported_metrics(path)["metric_rows"], 0)
            path.write_text(
                '==PROF== start\n"ID","Metric Name","Metric Value"\n'
                '"0","dram__bytes_read.sum","1024"\n'
                '"0","dram__bytes_write.sum","512"\n'
            )
            self.assertEqual(
                profiling.exported_metrics(path), dict(metric_rows=2, kernel_ids=["0"])
            )

    def test_ncu_captures_warmed_ranges_without_clock_changes(self):
        cmd = profiling.ncu_command("/ncu", Path("c.csv"), ["python", "worker"])
        self.assertEqual(cmd[-2:], ["python", "worker"])
        self.assertEqual(cmd[cmd.index("--profile-from-start") + 1], "off")
        self.assertEqual(cmd[cmd.index("--clock-control") + 1], "none")
        self.assertNotIn("--set", cmd)
        self.assertEqual(cmd.count("--section"), len(profiling.SECTIONS))

    def test_self_control_uses_identical_callable_and_original_timer(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []

            def fn():
                pass

            pair = dict(baseline=fn, candidate=fn)
            functions = {
                name: pair
                for name in (
                    "score",
                    "pipeline",
                    "score_vs_production",
                    "pipeline_vs_production",
                )
            }
            options = self.options(Path(folder))

            def timer(actual):
                self.assertIs(actual, fn)
                calls.append(actual)
                return 10.0

            result = {}
            profiling.self_controls(
                options,
                result,
                Path(folder) / "record.json",
                functions,
                timer,
                lambda _: {},
            )
            self.assertEqual(len(calls), 24)
            self.assertEqual(len(result["diagnostics"]["self_controls"]), 6)
            self.assertNotIn("stages", result)

    def test_capture_child_only_and_exclusive(self):
        base = [
            "bench",
            "--compiler",
            "flagtree",
            "--candidate",
            "query-stage-joint",
            "--compare-previous-tuning",
            "--previous-tuning",
            "combined",
            "--suite",
            "scored-prefill",
            "--profile-capture",
        ]
        with patch.object(os.sys, "argv", base), contextlib.redirect_stderr(
            io.StringIO()
        ):
            with self.assertRaises(SystemExit) as error:
                bench.main()
        self.assertEqual(error.exception.code, 2)

    def test_coordinator_keeps_unprofiled_results_and_requires_counter_rows(self):
        from tools import qsa_prefill_flagtree_ab as profile
        from tools import validate_qsa_ptxas_h800 as validation

        for mode in ("passed", "missing", "permission"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                options = self.options(root)
                options.cases = [profiling.CASES[0]]
                commands = []

                def run(command, **kwargs):
                    commands.append(command)
                    name = options.cases[0]
                    if len(commands) == 1:
                        self.assertNotIn("--profile-capture", command)
                        self.assertNotIn("--profile-from-start", command)
                        row = dict(
                            baseline_us=dict(median=12),
                            candidate_us=dict(median=10),
                            speedup=1.2,
                        )
                        summary = dict(
                            status="passed",
                            cases=[
                                dict(
                                    case=dict(name=name),
                                    stages={
                                        stage: row
                                        for stage in (
                                            "score",
                                            "topk",
                                            "pipeline",
                                            "score_vs_production",
                                            "pipeline_vs_production",
                                        )
                                    },
                                )
                            ],
                        )
                        (root / "stage-summary.json").write_text(json.dumps(summary))
                        return 0
                    self.assertIn("--profile-capture", command)
                    (root / ("profile-" + name + ".json")).write_text(
                        json.dumps(
                            dict(
                                status="passed",
                                profiling=dict(expected_profiled_launches=3),
                            )
                        )
                    )
                    csv_path = root / ("profile-" + name + ".csv")
                    csv_path.write_text(
                        '"ID","Metric Name","Metric Value"\n0,x,1\n1,x,2\n2,x,3\n'
                        if mode == "passed"
                        else "==ERROR== ERR_NVGPUCTRPERM\n"
                    )
                    return 0 if mode == "passed" else 1

                def command(args):
                    return dict(
                        returncode=0, stdout=" ".join(profiling.SECTIONS), stderr=""
                    )

                guard = types.SimpleNamespace(
                    idle=lambda *a, **k: dict(status="passed")
                )
                with patch.object(
                    profile, "select_gpu", return_value=True
                ), patch.object(
                    validation,
                    "toolchain",
                    return_value=dict(
                        ptxas=dict(path="/ptxas", sha256=profile.PTXAS_SHA256)
                    ),
                ), patch.object(
                    profiling, "run_process", side_effect=run
                ), patch.object(
                    profiling.shutil,
                    "which",
                    return_value=None if mode == "missing" else "/ncu",
                ), patch.object(
                    bench.support, "command", side_effect=command
                ), patch.object(
                    bench.support, "gpu_guard_module", return_value=guard
                ), patch.dict(
                    os.environ, CUDA_VISIBLE_DEVICES="GPU-test"
                ), contextlib.redirect_stderr(
                    io.StringIO()
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    rc = profiling.coordinate(options)
                self.assertEqual(rc, {"passed": 0, "missing": 2, "permission": 1}[mode])
                saved = json.loads((root / "profiling-summary.json").read_text())
                self.assertEqual(saved["hardware_counters_complete"], mode == "passed")
                self.assertEqual(
                    json.loads((root / "stage-summary.json").read_text())["status"],
                    "passed",
                )
                self.assertTrue((root / "diagnostic-timings.tsv").is_file())

    def test_timeout_kills_only_own_process_group(self):
        proc = types.SimpleNamespace(pid=1234)
        import subprocess

        proc.wait = unittest.mock.Mock(
            side_effect=[subprocess.TimeoutExpired("worker", 1), 0]
        )
        with patch.object(
            profiling.subprocess, "Popen", return_value=proc
        ) as launch, patch.object(profiling.os, "killpg") as kill:
            self.assertEqual(profiling.run_process(["worker"], timeout=1), 124)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        kill.assert_called_once_with(1234, profiling.signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
