# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU checks for compiler isolation, original-source binding and stage scopes."""

import argparse
import base64
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_score_resume as resume
from tools import qsa_prefill_triton_ab as ab


class AdapterTests(unittest.TestCase):
    def test_adapter_prepares_once_and_preserves_operator_arguments(self):
        allocations, calls = [], []
        torch = types.SimpleNamespace(
            int32="i32", zeros=lambda *a, **kw: allocations.append((a, kw)) or object()
        )
        adapter = ab.PrefillTopK(lambda *a: calls.append(a), torch)
        logits = types.SimpleNamespace(
            shape=(17, 4096), device="cuda:0", stride=lambda i: (4096, 1)[i]
        )
        start = adapter.prepare(logits)
        adapter.frozen = True
        output, visible = object(), object()
        for _ in range(2):
            adapter(logits, visible, output, object(), k=512, max_seq_len=4096)
        self.assertEqual(len(allocations), 1)
        self.assertEqual(
            calls, [(logits, start, visible, output, 17, 4096, 1, 512)] * 2
        )
        with self.assertRaisesRegex(RuntimeError, "Unexpected row count"):
            adapter.prepare(types.SimpleNamespace(shape=(18, 4096), device="cuda:0"))

    def test_native_output_preserves_nonzero_inference_slice_offset(self):
        calls = []

        class Storage:
            def __getitem__(self, index):
                calls.append(index.start)
                return self

            def view(self, shape):
                calls.append(shape)
                return "native output"

        torch = types.SimpleNamespace(
            full=lambda *a, **kw: calls.append((a, kw)) or Storage()
        )
        output = types.SimpleNamespace(
            storage_offset=lambda: 512,
            numel=lambda: 4096 * 512,
            shape=(4096, 512),
            dtype="i32",
            device="cuda:0",
        )
        self.assertEqual(bench.native_output(output, torch), "native output")
        self.assertEqual(calls[0][0][0], (4097 * 512,))
        self.assertEqual(calls[1:], [512, (4096, 512)])

    def test_private_module_only_substitutes_topk_and_leaves_production_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "production.py"
            source = "def qsa_select_paged_prefill(*args):\n    return persistent_topk(*args)\n"
            path.write_text(source)
            production = types.ModuleType("production")
            production.__file__ = str(path)
            original = object()
            production.persistent_topk = original
            for name in (
                "_prefill_logits",
                "_qsa_prefill_score_kernel",
                "_validate_inputs",
                "_prefill_score_metadata",
            ):
                setattr(production, name, object())
            exec(source, production.__dict__)
            adapter = lambda *args: ("replacement", args)
            with patch.object(
                ab, "PRODUCTION_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
            ):
                candidate = ab.candidate_module(production, adapter)
                self.assertIs(
                    candidate.qsa_select_paged_prefill.__code__,
                    production.qsa_select_paged_prefill.__code__,
                )
                self.assertEqual(
                    candidate.qsa_select_paged_prefill(1, 2), ("replacement", (1, 2))
                )
                self.assertIs(production.persistent_topk, original)
                self.assertIs(candidate._prefill_logits, production._prefill_logits)
                path.write_text(source + "# changed\n")
                with self.assertRaisesRegex(RuntimeError, "Production source changed"):
                    ab.candidate_module(production, adapter)

    def test_observation_normalizes_native_keyword_offsets_and_restores_globals(self):
        class Buffer:
            def fill_(self, value):
                self.value = value

        class Logits:
            def clone(self):
                return "common snapshot"

        native = types.ModuleType("native")
        calls = []
        native._prefill_logits = lambda *a, **kw: calls.append((a, kw)) or Logits()
        native._topk = lambda *a, **kw: None
        exec(
            "def qsa_select_paged_prefill(*args):\n"
            "    logits=_prefill_logits(*args, query_offset=3, num_queries=7)\n"
            "    _topk(logits, 'visible', 2048, 4, 'output', 'scratch')\n",
            native.__dict__,
        )
        original = native._prefill_logits
        buffer = Buffer()
        chunks = ab.observe(native, "_topk", (0, 1, 2, 3, 4, buffer, 7, 4096))
        self.assertIs(native._prefill_logits, original)
        self.assertEqual(ab.signatures(chunks), [(3, 7, 4096)])
        self.assertEqual(chunks[0]["score_args"][-2:], (3, 7))
        self.assertEqual(chunks[0]["score_kwargs"], {})
        self.assertEqual(chunks[0]["common_logits"], "common snapshot")
        self.assertEqual(buffer.value, -float("inf"))

    def test_scores_exclude_topk_common_topk_inputs_match_pipeline_executes_both(self):
        calls = []

        def make(side, name):
            def score(*args, **kw):
                calls.append((side, "score", args[0]))
                return (side, "live", args[0])

            def topk(logits, *args, **kw):
                calls.append((side, "topk", logits))

            return types.SimpleNamespace(_prefill_logits=score, **{name: topk})

        native = make("vllm", "_topk")
        candidate = make("gems", "persistent_topk")

        def chunks():
            return [
                dict(
                    offset=i * 5,
                    rows=5,
                    width=64,
                    score_args=(i,),
                    score_kwargs={},
                    topk_args=(None, "metadata"),
                    topk_kwargs={},
                    common_logits=object(),
                )
                for i in range(2)
            ]

        left, right = chunks(), chunks()
        stages = ab.bind_stages(native, candidate, left, right)
        stages["score"]["baseline"]()
        stages["score"]["candidate"]()
        self.assertEqual([x[1] for x in calls], ["score"] * 4)
        calls.clear()
        stages["topk"]["baseline"]()
        stages["topk"]["candidate"]()
        self.assertIs(calls[0][2], calls[2][2])
        self.assertIs(calls[1][2], calls[3][2])
        self.assertIs(calls[0][2], left[0]["common_logits"])
        calls.clear()
        stages["pipeline"]["candidate"]()
        self.assertEqual(
            calls,
            [
                ("gems", "score", 0),
                ("gems", "topk", ("gems", "live", 0)),
                ("gems", "score", 1),
                ("gems", "topk", ("gems", "live", 1)),
            ],
        )
        right[-1]["rows"] = 4
        with self.assertRaisesRegex(RuntimeError, "chunk geometry differs"):
            ab.bind_stages(native, candidate, left, right)


class StockRuntimeTests(unittest.TestCase):
    def fixture(self, root):
        files = []

        class Entry(str):
            pass

        for name in (
            "triton/__init__.py",
            "triton/testing.py",
            "triton/backends/nvidia/compiler.py",
            "triton/_C/libtriton.so",
        ):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            entry = Entry(name)
            entry.hash = types.SimpleNamespace(
                mode="sha256",
                value=base64.urlsafe_b64encode(
                    hashlib.sha256(path.read_bytes()).digest()
                )
                .decode()
                .rstrip("="),
            )
            files.append(entry)
        dist = types.SimpleNamespace(
            metadata={"Name": "triton", "Summary": "Triton compiler"},
            version="3.6.0",
            files=files,
            locate_file=lambda p: root / str(p),
            read_text=lambda p: None,
        )
        torch = types.SimpleNamespace(__file__=str(root / "torch/__init__.py"))
        triton = types.SimpleNamespace(
            __file__=str(root / "triton/__init__.py"), __version__="3.6.0"
        )
        return torch, triton, dist

    def test_stock_record_verifies_loaded_compiler_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch, triton, dist = self.fixture(root)
            result = ab.stock_runtime(torch, triton, root, dist)
            self.assertEqual(len(result["files"]), 4)
            (root / "triton/testing.py").write_text("modified")
            with self.assertRaisesRegex(RuntimeError, "differs from installed wheel"):
                ab.stock_runtime(torch, triton, root, dist)

    def test_environment_uses_verified_installed_version_and_keeps_failure_details(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch, triton, dist = self.fixture(root)
            torch.__version__ = "2.13.0+cu129"
            # Regression: a genuine wheel other than 3.6.0 must reach the
            # compilation/correctness smoke checks, rather than a version gate.
            triton.__version__ = dist.version = "3.7.0"
            path = root / "preflight.json"
            with (
                patch.object(bench.sys, "prefix", str(root)),
                patch.object(
                    bench.importlib.metadata, "version", return_value=torch.__version__
                ),
                patch.object(ab.importlib.metadata, "distribution", return_value=dist),
                contextlib.redirect_stdout(io.StringIO()) as log,
            ):
                stock = bench.check_stock_environment(torch, triton, path)
                self.assertEqual(stock["version"], "3.7.0")
                self.assertIn("3.7.0", log.getvalue())
                self.assertEqual(
                    json.loads(path.read_text())["environment"]["triton_source"],
                    triton.__file__,
                )
                dist.metadata["Summary"] = "FlagTree"
                with self.assertRaisesRegex(RuntimeError, "FlagTree"):
                    bench.check_stock_environment(torch, triton, path)
                saved = json.loads(path.read_text())
                self.assertEqual(saved["environment"]["triton_version"], "3.7.0")
                self.assertEqual(saved["environment"]["prefix"], str(root))

    def test_shadowed_flagtree_tle_and_mismatched_versions_are_rejected(self):
        for mode in ("outside", "metadata", "tle", "version", "missing_record"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                torch, triton, dist = self.fixture(root)
                if mode == "outside":
                    triton.__file__ = "/opt/flagtree/triton/__init__.py"
                if mode == "metadata":
                    dist.metadata["Summary"] = "FlagTree"
                if mode == "tle":
                    (root / "triton/experimental/tle").mkdir(parents=True)
                if mode == "version":
                    triton.__version__ = "9.0"
                if mode == "missing_record":
                    dist.files = []
                with self.assertRaises(RuntimeError):
                    ab.stock_runtime(torch, triton, root, dist)


class ParentTests(unittest.TestCase):
    def options(self, reference):
        return argparse.Namespace(
            suite="inference-prefill",
            reference_file=str(reference),
            rounds=2,
            kernel_warmup_ms=1,
            kernel_rep_ms=1,
            timeout=10,
            mode="kernel",
            cases=None,
        )

    def exercise(self, root, failure=None):
        content = b"pinned test source"
        reference = root / "reference.py"
        reference.write_bytes(content)
        directory = root / "report"
        directory.mkdir()
        options = self.options(reference)
        selected = bench.workloads.cases()[:1]
        runtime = {k: "fixed" for k in resume.RUNTIME_KEYS}
        runtime["uuid"] = "example"

        def child(command, stdout, **kwargs):
            self.assertEqual(command[1], "-I")
            self.assertEqual(command[command.index("--suite") + 1], "inference-prefill")
            if command[-1] == "--preflight":
                data = (
                    dict(status="failed", error="native extension missing")
                    if failure == "preflight"
                    else dict(status="passed", runtime=runtime)
                )
                bench.support.write_json(directory / "preflight.json", data)
                return types.SimpleNamespace(returncode=int(failure == "preflight"))
            self.assertEqual(command[-2], "--worker")
            observed = dict(runtime)
            if failure == "compiler":
                observed["triton_source"] = "different compiler"
            data = dict(
                case=selected[0],
                status="failed" if failure == "worker" else "passed",
                runtime=observed,
                stages={
                    name: dict(
                        status="passed",
                        accepted=True,
                        baseline_us=dict(median=12),
                        candidate_us=dict(median=8),
                        speedup=1.5,
                        paired_ratios=dict(median=1.5),
                    )
                    for name in ("score", "topk", "pipeline")
                },
            )
            bench.support.write_json(directory / (command[-1] + ".json"), data)
            return types.SimpleNamespace(returncode=int(failure == "worker"))

        with patch.object(bench, "configure_environment"), patch.object(
            bench.support, "idle_gpu_gate", return_value=True
        ), patch.object(bench.support, "command", return_value={}), patch.object(
            bench.subprocess, "run", side_effect=child
        ), patch.object(
            ab, "REFERENCE_SHA256", hashlib.sha256(content).hexdigest()
        ), patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "GPU-2d17872f-4327-2beb-8049-604ecf6a3d66"},
        ), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(
            io.StringIO()
        ):
            code = bench.parent(options, selected, directory)
        summary = json.loads((directory / "stage-summary.json").read_text())
        table = (directory / "triton-stage-ab.tsv").read_text()
        return code, summary, table

    def test_parent_forwards_isolation_and_retains_three_measured_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            code, summary, table = self.exercise(Path(folder))
            self.assertEqual(code, 0)
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(len(table.splitlines()), 4)
            self.assertEqual(table.count("\t1.5\t1.5"), 3)

    def test_preflight_worker_and_compiler_failures_never_publish_accepted_ratios(self):
        for failure in ("preflight", "worker", "compiler"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                code, summary, table = self.exercise(Path(folder), failure)
                self.assertEqual(code, 1)
                self.assertEqual(summary["status"], "failed")
                self.assertNotIn("\t1.5\t1.5", table)

    def test_isolated_cli_lists_twenty_two_without_gpu_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(
                [sys.executable, "-I", str(Path(bench.__file__)), "--list-cases"],
                cwd=folder,
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertEqual(len(json.loads(result.stdout)), 22)


if __name__ == "__main__":
    unittest.main()
