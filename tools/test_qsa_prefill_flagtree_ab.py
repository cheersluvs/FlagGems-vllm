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
from unittest.mock import patch

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_flagtree_ab as profile
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
            tool = dict(
                ptxas=dict(path="/tmp/verified-ptxas", sha256=profile.PTXAS_SHA256)
            )
            with patch.object(
                validation, "select_gpu", return_value=True
            ), patch.object(validation, "toolchain", return_value=tool), patch.object(
                profile.os, "execve"
            ) as execute, patch.dict(
                os.environ, TRITON_MOCK_PTX_VERSION="87", CUDA_LAUNCH_BLOCKING="1"
            ):
                self.assertEqual(profile.prepare(options, ["--rounds", "7"]), 0)
                python, args, env = execute.call_args.args
                self.assertEqual(args[:2], [python, "-I"])
                self.assertEqual(args[args.index("--compiler") + 1], "flagtree")
                self.assertEqual(env["TRITON_PTXAS_PATH"], "/tmp/verified-ptxas")
                self.assertNotIn("TRITON_MOCK_PTX_VERSION", env)
                self.assertNotIn("CUDA_LAUNCH_BLOCKING", env)


class AccuracyTests(unittest.TestCase):
    def test_gate_accepts_full_native_coverage_and_rejects_failures_or_changed_identity(
        self,
    ):
        for failure in (None, "pytest", "missing", "skipped", "identity"):
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
                    )
                }
                if failure == "missing":
                    records.pop("tests/test.py::test_parameterized_against_vllm")
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

                with patch.object(
                    bench.support, "idle_gpu_gate", return_value=True
                ), patch.object(
                    bench.subprocess, "run", side_effect=run
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    if failure:
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

        with tempfile.TemporaryDirectory() as folder, patch.object(
            ParentTests, "options", options
        ), patch.object(
            bench, "run_accuracy_gate", side_effect=RuntimeError("accuracy failed")
        ) as gate:
            code, summary, table = ParentTests().exercise(Path(folder))
            self.assertEqual(code, 1)
            self.assertEqual(summary["cases"], [])
            self.assertEqual(len(table.splitlines()), 1)
            gate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
