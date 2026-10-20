# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU checks for inference geometry, view offsets and physical page remapping."""

import ast
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_qsa_prefill_stage_baseline as stage_tests

from tools import qsa_prefill_inference_workloads as workloads
from tools import qsa_prefill_observed_workloads as observed

bench = stage_tests.bench


class InferenceWorkloadTests(unittest.TestCase):
    def test_twenty_two_cases_preserve_exact_upstream_twelve(self):
        cases = workloads.cases()
        self.assertEqual(len(cases), 22)
        self.assertEqual(len({c["name"] for c in cases}), 22)
        self.assertEqual(cases[:12], workloads.upstream.cases())
        for case in cases[:12]:
            self.assertEqual(
                workloads.geometry(case), workloads.upstream.geometry(case)
            )
        self.assertEqual(sum("physical_pages" in c for c in cases), 2)

    def test_inference_lengths_context_and_chunk_contract(self):
        expected = [
            ([128], [8192], [128]),
            ([512], [102400], [512]),
            ([2048], [49152], [2048]),
            ([8192], [0], [8192]),
            ([4096, 2048, 1024, 1024], [8192, 51200, 102400, 0], [5190, 3002]),
            ([256] * 32, [8192] * 32, [8192]),
            ([7168] + [64] * 16, [51200] + [8192] * 16, [8192]),
            ([4096], [4096], [2621, 1475]),
        ]
        for case, (queries, prefixes, chunks) in zip(
            workloads.inference_cases(), expected
        ):
            info = workloads.geometry(case)
            self.assertEqual(info["query_lengths"], queries)
            self.assertEqual(info["prefix_tokens"], prefixes)
            self.assertEqual(info["expected_chunk_rows"], chunks)
            self.assertEqual(sum(chunks), sum(queries))
            self.assertLessEqual(info["score_workspace_bytes"], 512 * 1024**2)
            self.assertGreater(info["logits_width"], 512)  # real score path
        mixed = workloads.geometry(workloads.inference_cases()[7])
        self.assertEqual(mixed["sequence_lengths"], [8192])
        self.assertEqual(mixed["batch_max_seq_len"], 204800)
        self.assertEqual(mixed["query_start_loc"], [1, 4097])
        self.assertEqual(mixed["logits_width"], 51200)
        for case in workloads.inference_cases()[-2:]:
            info = workloads.geometry(case)
            self.assertEqual(info["cache_shape"], [4096, 196, 1, 128])
            self.assertEqual(info["page_table_shape"], [1, 335])
            self.assertEqual(
                info["input_layout"]["1"],
                dict(stride=[50176, 128, 128, 1], storage_offset=25088),
            )
            self.assertIn("not a deployment trace", info["layout_provenance"])

    def test_geometry_rejects_truncated_capacity(self):
        original = workloads.inference_cases()[7]
        for overrides in (
            dict(table_columns=1),
            dict(physical_pages=1),
            dict(batch_max_seq_len=1),
        ):
            with self.assertRaisesRegex(ValueError, "cannot truncate"):
                workloads.geometry(dict(original, **overrides))

    def test_pytest_uses_thirty_two_and_thirty_scored_inputs(self):
        source = bench.ROOT / "benchmark/test_qsa_select_paged_prefill_perf.py"
        tree = ast.parse(source.read_text())
        definitions = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.ClassDef)
                and node.name in ("QSAPrefillBenchmark", "QSAScoredBenchmark")
            )
            or (isinstance(node, ast.FunctionDef) and node.name == "_inputs")
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "BENCHMARK_CASES"
                    for t in node.targets
                )
            )
        ]
        namespace = dict(
            base=types.SimpleNamespace(GenericBenchmark=object),
            workloads=observed,
            input_helpers=object(),
            torch=types.SimpleNamespace(bfloat16="bf16", float8_e4m3fn="fp8"),
        )
        exec(
            compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"),
            namespace,
        )
        benchmark = namespace["QSAPrefillBenchmark"]()
        benchmark.set_shapes()
        self.assertEqual(
            benchmark.shapes, [(c["name"],) for c in observed.cases("combined-prefill")]
        )
        self.assertEqual(benchmark.set_more_shapes(), [])
        self.assertEqual(
            bench.suite_cases(types.SimpleNamespace(suite="upstream-prefill")),
            workloads.upstream.cases(),
        )
        with patch.object(
            observed, "make_inputs", return_value=tuple(range(10))
        ) as builder:
            for shape, case in zip(
                benchmark.shapes, observed.cases("combined-prefill")
            ):
                self.assertEqual(
                    next(
                        namespace["_inputs"](
                            shape,
                            "fp8" if case["dtype"] == "float8_e4m3fn" else "bf16",
                            "cuda:3",
                        )
                    ),
                    tuple(range(10)),
                )
                self.assertEqual(builder.call_args.args[0], case)
                self.assertEqual(builder.call_args.kwargs["device"], "cuda:3")
        scored = namespace["QSAScoredBenchmark"]()
        scored.set_shapes()
        self.assertEqual(len(scored.shapes), 30)
        for shape in scored.shapes:
            self.assertFalse(
                observed.geometry(namespace["BENCHMARK_CASES"][shape]).get("select_all")
            )

    def test_parent_writes_full_inference_manifest_and_forwards_suite(self):
        options = stage_tests.BenchmarkTests.options(
            self,
            suite="inference-prefill",
            single_row_ab=True,
            mode="kernel",
            skip_accuracy=True,
        )
        selected = workloads.inference_cases()[-2:]

        def worker(command, **kwargs):
            self.assertEqual(command[command.index("--suite") + 1], "inference-prefill")
            name = command[command.index("--worker") + 1]
            case = next(c for c in selected if c["name"] == name)
            output = Path(command[command.index("--report-dir") + 1])
            bench.write_json(
                output / (name + ".json"),
                dict(
                    case=case,
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
            path = Path(folder)
            self.assertEqual(bench.parent(options, selected, path), 0)
            report = json.loads((path / "stage-summary.json").read_text())
            self.assertEqual(
                report["manifest"]["workloads"],
                {c["name"]: workloads.geometry(c) for c in selected},
            )
            self.assertEqual(
                len((path / "inference-prefill-shapes.tsv").read_text().splitlines()), 3
            )
            self.assertEqual(
                report["score_case_progress"], dict(completed=2, active=2, pending=0)
            )


@unittest.skipUnless(
    importlib.util.find_spec("numpy"), "NumPy needed for CPU tensor-builder check"
)
class TensorBuilderTests(unittest.TestCase):
    def test_strided_global_pages_preserve_keys_causality_and_decode_view_offsets(self):
        import numpy as np

        class Tensor(np.ndarray):
            def copy_(self, other):
                self[...] = other
                return self

            def long(self):
                return self.astype(np.int64).view(Tensor)

            def to(self, dtype):
                return self.astype(dtype).view(Tensor)

            def stride(self):
                return tuple(s // self.itemsize for s in self.strides)

            def storage_offset(self):
                root = self
                while isinstance(root.base, np.ndarray):
                    root = root.base
                return (self.ctypes.data - root.ctypes.data) // self.itemsize

        def tensor(value, dtype=np.float32):
            return np.array(value, dtype=dtype).view(Tensor)

        fake_torch = types.SimpleNamespace(
            bfloat16=np.float32,
            zeros=lambda shape, dtype, device: np.zeros(shape, dtype=dtype).view(
                Tensor
            ),
            full=lambda shape, value, dtype, device: np.full(
                shape, value, dtype=dtype
            ).view(Tensor),
            randperm=lambda n, device: tensor(
                np.random.default_rng(13).permutation(n), np.int64
            ),
        )
        recorded = {}

        def base_inputs(queries, contexts, **kwargs):
            page, dim, heads = (
                kwargs["page_size"],
                kwargs["head_dim"],
                kwargs["num_heads"],
            )
            pages = [(c + page - 1) // page for c in contexts]
            cache = tensor(
                np.arange(sum(pages) * page * dim).reshape(sum(pages), page, 1, dim)
            )
            table = tensor(np.full((len(queries), max(pages)), -1), np.int32)
            cursor = 0
            for r, n in enumerate(pages):
                table[r, :n] = np.arange(cursor, cursor + n)
                cursor += n
            offsets = [kwargs["query_base"]]
            for q in queries:
                offsets.append(offsets[-1] + q)
            args = [
                tensor(
                    np.arange(sum(queries) * heads * dim).reshape(
                        sum(queries), heads, dim
                    )
                ),
                cache,
                table,
                tensor(offsets, np.int32),
                tensor(kwargs["visible"], np.int32),
                kwargs["k"] * kwargs["ratio"],
                kwargs["ratio"],
                max(queries),
                tensor(np.full((sum(queries), kwargs["k"]), -17), np.int32),
                max(contexts) * kwargs["ratio"],
            ]
            recorded.update(
                cache=cache.copy(), table=table.copy(), visible=kwargs["visible"]
            )
            return args

        for base in (0, 1):
            case = dict(
                workloads.inference_cases()[-1],
                query_lengths=[3, 5],
                prefix_tokens=[13, 29],
                page_size=4,
                dim=2,
                heads=1,
                k=2,
                table_columns=9,
                physical_pages=17,
                batch_max_seq_len=80,
                query_base=base,
            )
            args = workloads.make_inputs(
                case,
                types.SimpleNamespace(make_inputs=base_inputs),
                fake_torch,
                device="cpu",
            )
            info = workloads.geometry(case)
            for index, layout in info["input_layout"].items():
                value = args[int(index)]
                self.assertEqual(list(value.stride()), layout["stride"])
                self.assertEqual(value.storage_offset(), layout["storage_offset"])
            self.assertEqual(args[3].tolist(), info["query_start_loc"])
            self.assertEqual(args[4].tolist(), [3, 3, 4, 7, 7, 8, 8, 8])
            self.assertEqual(args[9], 80)
            self.assertTrue(np.all(args[2][:, 3:] == -1))
            # Every logical live key is identical after physical-page remapping.
            for r, old_row in enumerate(recorded["table"]):
                for column, old_page in enumerate(old_row):
                    if old_page >= 0:
                        np.testing.assert_array_equal(
                            args[1][args[2][r, column]], recorded["cache"][old_page]
                        )
            live = set(args[2][args[2] >= 0].tolist())
            self.assertEqual(len(live), info["active_physical_pages"])
            self.assertGreater(max(live), info["active_physical_pages"] - 1)
            self.assertTrue(
                all(np.all(args[1][i] == 0) for i in range(17) if i not in live)
            )


if __name__ == "__main__":
    unittest.main()
