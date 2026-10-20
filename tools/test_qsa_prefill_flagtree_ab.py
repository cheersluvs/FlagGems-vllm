# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU coverage for compiler/driver isolation and the mandatory accuracy gate."""

import argparse
import contextlib
import ctypes
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_flagtree_ab as profile
from tools import qsa_prefill_gpu_guard as gpu_guard
from tools import qsa_prefill_score_resume as resume
from tools import validate_qsa_ptxas_h800 as validation


class EnvironmentTests(unittest.TestCase):
    def test_selected_package_and_override_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "triton").mkdir()
            (root / "triton/__init__.py").write_text("")
            with patch.object(sys, "path", list(sys.path)), patch.dict(
                os.environ, {}, clear=True
            ):
                profile.configure(root)
                self.assertEqual(sys.path[0], str(root.resolve()))
                self.assertEqual(
                    os.environ["QSA_TRITON_RUNTIME_PATH"], str(root.resolve())
                )
                os.environ["TRITON_MOCK_PTX_VERSION"] = "87"
                with self.assertRaisesRegex(
                    RuntimeError, "Unexpected compiler override"
                ):
                    profile.configure(root)

    def test_driver_requires_the_actual_verified_mapping_and_api(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            library = root / "libcuda.so.615"
            library.write_bytes(b"verified compat")
            (root / "compat134.json").write_text(
                json.dumps(
                    dict(libcuda=str(library), libcuda_sha256=profile.sha(library))
                )
            )
            maps = Path("/proc/self/maps")
            original = Path.read_text

            class Function:
                def __call__(self, pointer):
                    ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = version
                    return 0

            for version, mapped, succeeds in (
                (13040, library, True),
                (12090, library, False),
                (13040, root / "other", False),
            ):
                with self.subTest(version=version, mapped=mapped), patch.object(
                    profile, "LIBCUDA_SHA256", profile.sha(library)
                ), patch.object(
                    Path,
                    "read_text",
                    lambda p, *a, **kw: (
                        f"mapping {mapped}\n" if p == maps else original(p, *a, **kw)
                    ),
                ), patch.object(
                    profile.ctypes,
                    "CDLL",
                    return_value=types.SimpleNamespace(cuDriverGetVersion=Function()),
                ):
                    if succeeds:
                        self.assertEqual(profile.loaded_cuda(root)["driver_api"], 13040)
                    else:
                        with self.assertRaises(RuntimeError):
                            profile.loaded_cuda(root)

    def test_compiler_rejects_mixed_tle_and_wrong_assembler(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / "triton"
            package.mkdir()
            module = package / "module.py"
            module.write_text("compiler")
            tool = root / "ptxas"
            tool.write_text("verified assembler")
            compiler = types.SimpleNamespace(
                __file__=str(module), get_ptxas=lambda _: tool
            )
            torch = types.SimpleNamespace(
                __file__=str(Path(sys.prefix) / "torch.py"),
                cuda=types.SimpleNamespace(init=lambda: None),
            )
            triton = types.SimpleNamespace(__version__="3.6.0")

            def imported(name):
                if name == "triton.backends.nvidia.compiler":
                    return compiler
                return types.SimpleNamespace(
                    __file__=(
                        str(root / "foreign.py")
                        if foreign and name.endswith("tle.language")
                        else str(module)
                    )
                )

            for foreign, digest, succeeds in (
                (False, profile.sha(tool), True),
                (True, profile.sha(tool), False),
                (False, "wrong", False),
            ):
                with self.subTest(foreign=foreign, digest=digest), patch.dict(
                    os.environ, TRITON_PTXAS_PATH=str(tool)
                ), patch.object(
                    profile.importlib, "import_module", side_effect=imported
                ), patch.object(
                    profile, "PTXAS_SHA256", digest
                ), patch.object(
                    profile, "loaded_cuda", return_value={"driver_api": 13040}
                ):
                    if succeeds:
                        self.assertEqual(
                            profile.compiler_runtime(torch, triton, root, root)[
                                "profile"
                            ],
                            "flagtree-cuda134",
                        )
                    else:
                        with self.assertRaises(RuntimeError):
                            profile.compiler_runtime(torch, triton, root, root)

    def test_bootstrap_executes_fresh_engine_with_verified_tool_and_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            options = argparse.Namespace(report_dir=folder, flagtree="/opt/flagtree")
            options.max_initial_temp_c = 80
            tool = dict(
                ptxas=dict(path="/tmp/verified-ptxas", sha256=profile.PTXAS_SHA256)
            )
            with patch.object(profile, "select_gpu", return_value=True), patch.object(
                validation, "toolchain", return_value=tool
            ), patch.object(profile.os, "execve") as execute, patch.dict(
                os.environ, TRITON_MOCK_PTX_VERSION="87", CUDA_LAUNCH_BLOCKING="1"
            ):
                self.assertEqual(profile.prepare(options, ["--rounds", "7"]), 0)
                python, args, env = execute.call_args.args
                self.assertEqual(args[:2], [python, "-I"])
                self.assertEqual(args[args.index("--compiler") + 1], "flagtree")
                self.assertEqual(args[args.index("--max-initial-temp-c") + 1], "80")
                self.assertEqual(env["TRITON_PTXAS_PATH"], "/tmp/verified-ptxas")
                self.assertNotIn("TRITON_MOCK_PTX_VERSION", env)
                self.assertNotIn("CUDA_LAUNCH_BLOCKING", env)


class GPUSelectionTests(unittest.TestCase):
    def test_prefers_cool_inventory_then_rechecks_processes_and_original_limits(self):
        cold = "GPU-00000000-0000-0000-0000-000000000001"
        cool = "GPU-00000000-0000-0000-0000-000000000002"
        hot = "GPU-00000000-0000-0000-0000-000000000003"
        query = dict(
            returncode=0,
            stdout=f"{hot}, NVIDIA H800, 64\n{cool}, NVIDIA H800, 29\n{cold}, NVIDIA H800, 28\n",
        )
        checks = [
            dict(status="rejected", processes=[dict(pid=123)]),
            dict(status="passed", processes=[]),
        ]
        with patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "auto"}, clear=True
        ), patch.object(bench.support, "command", return_value=query), patch.object(
            gpu_guard, "idle", side_effect=checks
        ) as idle, contextlib.redirect_stdout(
            io.StringIO()
        ):
            summary = {}
            self.assertTrue(profile.select_gpu(summary))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], cool)
            self.assertEqual(
                [call.args[0] for call in idle.call_args_list], [cold, cool]
            )
            self.assertEqual(summary["gpu_uuid"], cool)
            self.assertEqual(
                summary["gpu_selection_checks"][0]["inventory_temperature_c"], 28
            )

    def test_explicit_uuid_is_not_silently_replaced(self):
        uuid = "GPU-00000000-0000-0000-0000-000000000003"
        with patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": uuid}, clear=True
        ), patch.object(bench.support, "command") as command:
            summary = {}
            self.assertTrue(profile.select_gpu(summary))
            self.assertEqual(summary["gpu_uuid"], uuid)
            command.assert_not_called()

    def test_unavailable_or_unreadable_inventory_never_selects_a_gpu(self):
        uuid = "GPU-00000000-0000-0000-0000-000000000001"
        for text, malformed in (
            (f"{uuid}, NVIDIA H800, 80\n", False),
            (f"{uuid}, NVIDIA H800, N/A\n", True),
        ):
            with self.subTest(malformed=malformed), patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "auto"}, clear=True
            ), patch.object(
                bench.support, "command", return_value=dict(returncode=0, stdout=text)
            ), patch.object(
                gpu_guard,
                "idle",
                return_value=dict(
                    status="rejected",
                    reasons=["Initial temperature C 80 exceeds limit 75"],
                ),
            ):
                summary = {}
                if malformed:
                    with self.assertRaises(ValueError):
                        profile.select_gpu(summary)
                else:
                    self.assertFalse(profile.select_gpu(summary))
                    self.assertTrue(summary["environment_rejected"])
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "auto")


class AccuracyTests(unittest.TestCase):
    def test_shorter_defaults_keep_full_suite_stages_and_propagate_temperature(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            sys, "argv", ["benchmark", "--compiler", "flagtree", "--report-dir", folder]
        ), patch.object(
            bench, "parent", return_value=0
        ) as parent, contextlib.redirect_stdout(
            io.StringIO()
        ):
            self.assertEqual(bench.main(), 0)
            options = parent.call_args.args[0]
            self.assertEqual(
                (options.rounds, options.kernel_warmup_ms, options.kernel_rep_ms),
                (3, 200, 100),
            )
            self.assertEqual(options.max_initial_temp_c, 80)
            self.assertEqual(len(parent.call_args.args[1]), 12)
            child = bench.child_command(options, Path(folder), "--accuracy")
            self.assertEqual(child[child.index("--max-initial-temp-c") + 1], "80")
            self.assertEqual(child[child.index("--rounds") + 1], "3")

    def test_gate_accepts_full_native_coverage_and_rejects_failures_or_changed_identity(
        self,
    ):
        for failure in (
            None,
            "pytest",
            "missing",
            "missing_trial",
            "skipped",
            "identity",
            "key_major",
            "missing_key_major",
            "reuse",
            "missing_reuse",
            "missing_reuse_key_major",
        ):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                options = argparse.Namespace(
                    compiler="flagtree",
                    flagtree="/opt/flagtree",
                    suite="upstream-prefill",
                    reference_file="native.py",
                    rounds=7,
                    kernel_warmup_ms=1000,
                    kernel_rep_ms=100,
                    accuracy_timeout=10,
                )
                info = {key: "fixed" for key in resume.RUNTIME_KEYS}
                if failure in ("key_major", "missing_key_major"):
                    options.candidate = "key-major"
                if failure in ("reuse", "missing_reuse", "missing_reuse_key_major"):
                    options.candidate = "key-major-reuse"
                records = {
                    "tests/test.py::" + name: dict(result="passed")
                    for name in (
                        "test_parameterized_against_vllm",
                        "test_native_vllm_topk_dispatch",
                        "test_native_vllm_empty_and_select_all",
                        "test_against_pinned_vllm",
                        "test_upstream_prefill_against_vllm",
                        "test_upstream_split_prefill",
                        "test_upstream_no_complete_blocks_dim8",
                        "test_observed_envelopes_against_native",
                        "test_compact_every_config_and_chunk_boundary",
                        "test_compact_full_entry",
                    )
                }
                if failure == "missing":
                    records.pop("tests/test.py::test_parameterized_against_vllm")
                if failure == "missing_trial":
                    records.pop("tests/test.py::test_compact_full_entry")
                if failure in (
                    "key_major",
                    "missing_key_major",
                    "reuse",
                    "missing_reuse",
                    "missing_reuse_key_major",
                ):
                    records.update(
                        {
                            "tests/test.py::" + name: dict(result="passed")
                            for name in (
                                "test_key_major_every_config_and_chunk_boundary",
                                "test_key_major_observed_against_native",
                                "test_key_major_full_entry",
                            )
                        }
                    )
                if failure == "missing_key_major":
                    records.pop("tests/test.py::test_key_major_full_entry")
                if failure in ("reuse", "missing_reuse", "missing_reuse_key_major"):
                    records.update(
                        {
                            "tests/test.py::" + name: dict(result="passed")
                            for name in (
                                "test_reuse_new_configs_and_chunk_boundary",
                                "test_reuse_observed_against_native",
                                "test_reuse_full_entry",
                            )
                        }
                    )
                if failure == "missing_reuse":
                    records.pop("tests/test.py::test_reuse_full_entry")
                if failure == "missing_reuse_key_major":
                    records.pop("tests/test.py::test_key_major_full_entry")
                if failure == "skipped":
                    records["tests/test.py::test_parameterized_against_vllm"][
                        "result"
                    ] = "skipped"
                (root / "accuracy-full.json").write_text(json.dumps(records))
                observed = (
                    dict(info, triton_source="changed")
                    if failure == "identity"
                    else info
                )
                (root / "accuracy-runtime.json").write_text(
                    json.dumps(dict(status="passed", runtime=observed))
                )
                summary = {}

                def run(command, **kwargs):
                    self.assertEqual(
                        command[command.index("--compiler") + 1], "flagtree"
                    )
                    self.assertEqual(command[-1], "--accuracy")
                    self.assertEqual(kwargs["timeout"], 10)
                    return types.SimpleNamespace(returncode=int(failure == "pytest"))

                with patch.object(bench, "idle_gate", return_value=True), patch.object(
                    bench.subprocess, "run", side_effect=run
                ), contextlib.redirect_stdout(io.StringIO()):
                    if failure and failure not in ("key_major", "reuse"):
                        with self.assertRaisesRegex(
                            RuntimeError, "Accuracy gate failed"
                        ):
                            bench.run_accuracy_gate(
                                options, root, summary, dict(runtime=info), None
                            )
                        self.assertEqual(summary["accuracy"]["status"], "failed")
                    else:
                        bench.run_accuracy_gate(
                            options, root, summary, dict(runtime=info), None
                        )
                        self.assertEqual(summary["accuracy"]["status"], "passed")

    def test_accuracy_failure_blocks_all_benchmark_workers(self):
        from tools.test_qsa_prefill_triton_ab import ParentTests

        original = ParentTests.options

        def options(test, reference):
            value = original(test, reference)
            value.compiler = "flagtree"
            return value

        for environment_rejected in (False, True):

            def reject(options, directory, summary, *args):
                if environment_rejected:
                    summary["environment_rejected"] = True
                raise RuntimeError("accuracy failed")

            with self.subTest(
                environment_rejected=environment_rejected
            ), tempfile.TemporaryDirectory() as folder, patch.object(
                ParentTests, "options", options
            ), patch.object(
                bench, "run_accuracy_gate", side_effect=reject
            ) as gate:
                code, summary, table = ParentTests().exercise(Path(folder))
                self.assertEqual(code, 2 if environment_rejected else 1)
                self.assertEqual(
                    summary["status"],
                    "environment_rejected" if environment_rejected else "failed",
                )
                self.assertEqual(summary["cases"], [])
                self.assertEqual(len(table.splitlines()), 1)
                gate.assert_called_once()


class CooldownTests(unittest.TestCase):
    def check(self, temperature=80, utilization=0, reasons=None, processes=None):
        return dict(
            status="rejected" if reasons is None or reasons else "passed",
            temperature_c=temperature,
            utilization_percent=utilization,
            memory_mib=1,
            processes=[] if processes is None else processes,
            reasons=(
                [f"Initial temperature C {temperature} exceeds limit 75"]
                if reasons is None
                else reasons
            ),
        )

    def run_gate(self, checks, timeout=30, max_initial_temp_c=75):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            summary = {}
            clock = [0]
            sleeps = []

            def sleep(seconds):
                sleeps.append(seconds)
                clock[0] += seconds

            guard = types.SimpleNamespace(
                idle=Mock(side_effect=checks), process_details=Mock(return_value=[])
            )
            with patch.object(
                profile.time, "monotonic", side_effect=lambda: clock[0]
            ), patch.object(
                profile.time, "sleep", side_effect=sleep
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                accepted = profile.cooldown_gate(
                    summary,
                    root,
                    {},
                    "before_case",
                    guard,
                    timeout,
                    10,
                    max_initial_temp_c=max_initial_temp_c,
                )
            self.assertTrue(
                all(
                    c.kwargs["max_initial_temp_c"] == max_initial_temp_c
                    for c in guard.idle.call_args_list
                )
            )
            self.assertEqual(
                json.loads((root / "stage-summary.json").read_text()), summary
            )
            return accepted, summary, sleeps

    def test_custom_temperature_is_retryable_without_waiving_thermal_slowdown(self):
        accepted, summary, sleeps = self.run_gate(
            [
                self.check(82, reasons=["Initial temperature C 82 exceeds limit 80"]),
                self.check(79, reasons=[]),
            ],
            max_initial_temp_c=80,
        )
        self.assertTrue(accepted)
        self.assertEqual(sleeps, [10])
        self.assertEqual(summary["idle_gpu_checks"][-1]["temperature_c"], 79)
        accepted, _, sleeps = self.run_gate(
            [self.check(79, reasons=["SW Thermal Slowdown: Active"])],
            max_initial_temp_c=80,
        )
        self.assertFalse(accepted)
        self.assertEqual(sleeps, [])

    def test_waits_for_original_temperature_and_utilization_limits(self):
        accepted, summary, sleeps = self.run_gate(
            [
                self.check(),
                self.check(
                    76,
                    9,
                    [
                        "Initial temperature C 76 exceeds limit 75",
                        "Initial utilization percent 9 exceeds limit 5",
                    ],
                ),
                self.check(75, reasons=[]),
            ]
        )
        self.assertTrue(accepted)
        self.assertEqual(sleeps, [10, 10])
        self.assertNotIn("environment_rejected", summary)
        self.assertEqual(summary["idle_gpu_checks"][-1]["temperature_c"], 75)

    def test_timeout_retains_rejection_and_never_marks_accuracy_passed(self):
        accepted, summary, sleeps = self.run_gate(
            [self.check(), self.check(), self.check()], 15
        )
        self.assertFalse(accepted)
        self.assertEqual(sleeps, [10, 5])
        self.assertTrue(summary["environment_rejected"])
        self.assertTrue(summary["idle_gpu_checks"][-1]["cooldown_timed_out"])

    def test_process_thermal_memory_and_unknown_failures_do_not_wait(self):
        for reason in (
            "GPU already has compute processes; no jobs will be stopped",
            "HW Thermal Slowdown: Active",
            "Initial memory MiB 300 exceeds limit 256",
            "Cannot read thermal slowdown state",
        ):
            with self.subTest(reason=reason):
                accepted, summary, sleeps = self.run_gate(
                    [self.check(reasons=[reason])]
                )
                self.assertFalse(accepted)
                self.assertEqual(sleeps, [])


if __name__ == "__main__":
    unittest.main()
