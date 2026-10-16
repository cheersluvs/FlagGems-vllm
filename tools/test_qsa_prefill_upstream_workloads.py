# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU checks for original-token prefix semantics and isolated suite transport."""

import ast
import contextlib
import importlib.util
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

import test_qsa_prefill_stage_baseline as stage_tests

bench = stage_tests.bench

SPEC = importlib.util.spec_from_file_location(
    "upstream", Path(__file__).with_name("qsa_prefill_upstream_workloads.py")
)
workloads = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workloads)


class UpstreamWorkloadTests(unittest.TestCase):
    def test_grid_matches_published_layouts_and_page_size(self):
        cases = workloads.cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({c["name"] for c in cases}), 12)
        self.assertEqual(
            {(c["requests"], c["query_length"], c["prefix_tokens"]) for c in cases},
            {
                (n, q, p)
                for n, q in ((1, 8192), (1, 16384), (2, 8192), (4, 4096))
                for p in (8192, 51200, 102400)
            },
        )
        self.assertEqual({c["page_size"] for c in cases}, {196})
        self.assertEqual({c["dtype"] for c in cases}, {"bfloat16"})

    def test_longest_case_has_causal_extent_and_bounded_real_chunks(self):
        case = next(
            c
            for c in workloads.cases()
            if c["requests"] == 1
            and c["query_length"] == 16384
            and c["prefix_tokens"] == 102400
        )
        geometry = workloads.geometry(case)
        self.assertEqual(geometry["sequence_lengths"], [118784])
        self.assertEqual(geometry["compressed_lengths"], [29696])
        self.assertEqual(geometry["q_shape"], [16384, 4, 128])
        self.assertEqual(geometry["cache_shape"], [152, 196, 1, 128])
        self.assertEqual(geometry["page_table_shape"], [1, 152])
        self.assertEqual(geometry["visible_first"], 25600)
        self.assertEqual(geometry["visible_last"], 29696)
        self.assertEqual(geometry["expected_chunk_rows"], [4519, 4519, 4519, 2827])
        self.assertLessEqual(geometry["score_workspace_bytes"], 512 * 1024**2)
        self.assertGreater(geometry["full_logits_bytes"], 512 * 1024**2)

    def test_prefix_visibility_resets_per_request_and_preserves_partial_block(self):
        case = dict(workloads.cases()[0], requests=2, query_length=6, prefix_tokens=7)
        captured = {}

        def make_inputs(query_lengths, compressed_lengths, **kwargs):
            captured.update(q=query_lengths, contexts=compressed_lengths, **kwargs)
            return tuple(range(10))

        args = workloads.make_inputs(
            case,
            types.SimpleNamespace(make_inputs=make_inputs),
            types.SimpleNamespace(bfloat16="bf16"),
        )
        self.assertEqual(captured["q"], [6, 6])
        self.assertEqual(captured["contexts"], [4, 4])
        self.assertEqual(captured["visible"], [2, 2, 2, 2, 3, 3] * 2)
        self.assertEqual(args[:9], tuple(range(9)))
        self.assertEqual(args[9], 13)  # original length, not rounded to 16
        self.assertEqual(captured["page_size"], 196)
        self.assertEqual(captured["num_heads"], 4)
        self.assertEqual(captured["head_dim"], 128)

    def test_all_cases_are_multi_row_fallback_controls(self):
        for case in workloads.cases():
            geometry = workloads.geometry(case)
            self.assertEqual(
                sum(geometry["expected_chunk_rows"]), geometry["q_shape"][0]
            )
            self.assertTrue(all(n > 1 for n in geometry["expected_chunk_rows"]))
            self.assertLessEqual(geometry["score_workspace_bytes"], 512 * 1024**2)
            self.assertGreaterEqual(geometry["visible_first"], case["k"])

    def test_pytest_benchmark_uses_only_shared_twelve_and_causal_inputs(self):
        # Execute the real shape/input definitions without importing GPU modules.
        source = bench.ROOT / "benchmark/test_qsa_select_paged_prefill_perf.py"
        tree = ast.parse(source.read_text())
        definitions = [
            node
            for node in tree.body
            if (isinstance(node, ast.ClassDef) and node.name == "QSAPrefillBenchmark")
            or (isinstance(node, ast.FunctionDef) and node.name == "_inputs")
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "UPSTREAM_CASES"
                    for target in node.targets
                )
            )
        ]
        captured = {}

        def make_inputs(query_lengths, contexts, **kwargs):
            captured.update(q=query_lengths, contexts=contexts, **kwargs)
            return tuple(range(10))

        namespace = dict(
            base=types.SimpleNamespace(GenericBenchmark=object),
            upstream_workloads=workloads,
            input_helpers=types.SimpleNamespace(make_inputs=make_inputs),
            torch=types.SimpleNamespace(bfloat16="bf16"),
        )
        exec(
            compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"),
            namespace,
        )
        benchmark = namespace["QSAPrefillBenchmark"]()
        benchmark.set_shapes()
        self.assertEqual(benchmark.set_more_shapes(), [])
        self.assertEqual(
            benchmark.shapes,
            [
                (c["requests"], c["query_length"], c["prefix_tokens"])
                for c in workloads.cases()
            ],
        )
        for shape, case in zip(benchmark.shapes, workloads.cases()):
            args = next(namespace["_inputs"](shape, "bf16", "cuda:3"))
            info = workloads.geometry(case)
            self.assertEqual(captured["device"], "cuda:3")
            self.assertEqual(captured["q"], info["query_lengths"])
            self.assertEqual(captured["contexts"], info["compressed_lengths"])
            self.assertEqual(captured["visible"][0], info["visible_first"])
            self.assertEqual(captured["visible"][-1], info["visible_last"])
            self.assertEqual(args[9], max(info["sequence_lengths"]))

    def test_cli_defaults_to_twelve_with_explicit_legacy_diagnostics(self):
        for suite, count in (
            (None, 12),
            ("upstream-prefill", 12),
            ("comprehensive", 38),
        ):
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(bench.__file__)),
                    "--single-row-ab",
                    "--list-cases",
                ]
                + (["--suite", suite] if suite else []),
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(len(json.loads(result.stdout)), count)

    def test_parent_forwards_new_suite_to_each_worker(self):
        options = stage_tests.BenchmarkTests.options(
            self,
            suite="upstream-prefill",
            single_row_ab=True,
            mode="kernel",
            skip_accuracy=True,
        )
        first = workloads.cases()[0]

        def worker(command, **kwargs):
            self.assertEqual(command[command.index("--suite") + 1], "upstream-prefill")
            self.assertEqual(command[command.index("--worker") + 1], first["name"])
            output = Path(command[command.index("--report-dir") + 1])
            bench.write_json(
                output / (first["name"] + ".json"),
                dict(
                    case=first,
                    status="passed",
                    stages={},
                    score_kernel=dict(
                        status="passed",
                        baseline_us=dict(median=2),
                        candidate_us=dict(median=2),
                        speedup=1,
                        paired_ratios=dict(median=1),
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
            self.assertEqual(bench.parent(options, [first], directory), 0)
            summary = json.loads((directory / "stage-summary.json").read_text())
            self.assertEqual(
                summary["manifest"]["options"]["suite"], "upstream-prefill"
            )
            self.assertIn(
                first["name"], (directory / "score-kernel-ab.tsv").read_text()
            )
            self.assertIn(
                first["name"], (directory / "upstream-prefill-shapes.tsv").read_text()
            )
            info = summary["manifest"]["workloads"][first["name"]]
            self.assertEqual(info["sequence_lengths"], [16384])
            self.assertEqual(info["compressed_lengths"], [4096])
            self.assertEqual(
                summary["score_case_progress"],
                dict(
                    completed=1,
                    active=1,
                    pending=0,
                ),
            )

    def test_worker_resume_revalidates_the_new_suite_definitions(self):
        from tools import qsa_prefill_score_resume as resume

        options = stage_tests.BenchmarkTests.options(
            self,
            suite="upstream-prefill",
            single_row_ab=True,
            mode="kernel",
            resume_from="prior.json",
        )

        def reject_after_definition_check(path, root, definitions, opts, uuid):
            self.assertEqual(definitions, workloads.cases())
            self.assertEqual(opts.suite, "upstream-prefill")
            raise ValueError("stop_after_definition_check")

        with patch.object(resume, "load", side_effect=reject_after_definition_check):
            with self.assertRaisesRegex(ValueError, "stop_after_definition_check"):
                bench.run_score_kernel(
                    options,
                    None,
                    None,
                    (),
                    [],
                    {},
                    None,
                    None,
                    None,
                    None,
                )


if __name__ == "__main__":
    unittest.main()
