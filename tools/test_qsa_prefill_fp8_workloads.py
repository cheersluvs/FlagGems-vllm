# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""FP8 matrix/provenance, dtype routing, validation and report contracts."""

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_fp8_workloads as fp8
from tools import qsa_prefill_inference_workloads as inference
from tools import qsa_prefill_observed_workloads as workloads

ROOT = Path(__file__).resolve().parents[1]


class FP8WorkloadsTests(unittest.TestCase):
    def test_original_bf16_matrix_is_exactly_preserved(self):
        cases = workloads.cases("combined-prefill")
        self.assertEqual(len(cases), 32)
        self.assertEqual(len({c["name"] for c in cases}), 32)
        self.assertEqual(
            hashlib.sha256(json.dumps(cases[:22], sort_keys=True).encode()).hexdigest(),
            "c61aceba0a4fc439fbbf63db32b571bb93c50d564c4e0d1c211113a0cf39c1a3",
        )
        self.assertEqual(cases[22:], fp8.cases())
        self.assertEqual(
            [len(workloads.cases(s)) for s in workloads.SUITES], [12, 10, 10, 32, 30]
        )

    def test_selected_envelopes_match_uploaded_manifest(self):
        manifest = json.loads(
            (ROOT / "docs/qsa_prefill_fp8_shape_manifest.json").read_text()
        )
        self.assertEqual(manifest["upload_commit"], fp8.UPLOAD_COMMIT)
        self.assertEqual(manifest["archive_sha256"], fp8.ARCHIVE_SHA256)
        self.assertFalse(manifest["actual_indexer_dtype_observed"])
        self.assertEqual(len(manifest["tuples"]), 128)
        self.assertEqual(sum(r["count"] for r in manifest["tuples"]), 6528)
        captured = {tuple(r["shape"]): r for r in manifest["tuples"]}
        selected = fp8.cases()[:8]
        self.assertEqual(
            [c["recorded_shape"] for c in selected],
            [r["shape"] for r in manifest["selected"]],
        )
        for c in selected:
            row = captured[tuple(c["recorded_shape"])]
            self.assertEqual(c["captured_parameters"], row["parameters"])
            self.assertEqual(c["physical_pages"], 2712)
            self.assertEqual(c["table_columns"], 88)
            self.assertEqual(c["recorded_count"], row["count"])
            self.assertEqual(c["capture_line"], row["line"])
            self.assertEqual(
                c["recorded_shape"],
                [
                    sum(c["query_lengths"]),
                    len(c["query_lengths"]),
                    4,
                    128,
                    max(c["query_lengths"]),
                    c["batch_max_seq_len"],
                ],
            )
            g = workloads.geometry(c)
            self.assertEqual(g["source"], fp8.SOURCE)
            self.assertFalse(g["actual_indexer_dtype_observed"])
            self.assertIn("dtype", g["synthetic_fields"])
            for field in ("compress_ratio", "token_topk", "page_size"):
                self.assertNotIn(field, g["synthetic_fields"])
            self.assertEqual(g["cache_shape"], [2712, 196, 1, 128])
            self.assertEqual(g["page_table_shape"], [len(c["query_lengths"]), 88])
        old = {c["name"]: c for c in workloads.cases("combined-prefill")[:22]}
        for c in fp8.cases()[8:]:
            g = workloads.geometry(c)
            self.assertEqual(g["origin"], "dtype_variant")
            self.assertNotIn("recorded_shape", g)
            self.assertNotIn("recorded_count", g)
            a = workloads.geometry(old[c["variant_of"]])
            for key in (
                "q_shape",
                "cache_shape",
                "page_table_shape",
                "logits_width",
                "expected_chunk_rows",
                "score_workspace_bytes",
            ):
                self.assertEqual(g[key], a[key], key)

    def test_exact_chunks_and_table_capacity_dependency(self):
        expected = [
            [16384],
            [16384],
            [15768, 563],
            [16256, 112],
            [16256, 113],
            [16256, 73],
            [16256, 128],
            [2316],
            [8896],
            [4519, 4519, 4519, 2827],
        ]
        for c, chunks in zip(fp8.cases(), expected):
            g = workloads.geometry(c)
            self.assertEqual(g["expected_chunk_rows"], chunks)
            self.assertEqual(c["dtype"], "float8_e4m3fn")
            self.assertFalse(g["select_all"])
            self.assertLessEqual(g["score_workspace_bytes"], 512 * 1024**2)
        self.assertEqual(
            workloads.geometry(fp8.cases()[1])["score_workspace_bytes"], 512 * 1024**2
        )
        for case, tail in zip(fp8.cases()[3:5], (112, 113)):
            tight = inference.geometry(dict(case, table_columns=42))
            self.assertEqual(tight["expected_chunk_rows"], [16304, tail - 48])
            self.assertEqual(tight["logits_width"], 8232)
            self.assertEqual(inference.geometry(case)["logits_width"], 8256)

    def test_builder_passes_fp8_dtype_and_exact_visibility(self):
        # Unchanged synthetic variants keep the original direct helper route.
        for c in fp8.cases()[8:]:
            captured = {}

            def build(*args, **kwargs):
                captured.update(args=args, kwargs=kwargs)
                return tuple(range(10))

            args = workloads.make_inputs(
                c,
                types.SimpleNamespace(make_inputs=build),
                types.SimpleNamespace(float8_e4m3fn="FP8"),
                device="cuda:3",
            )
            self.assertEqual(captured["kwargs"]["dtype"], "FP8")
            self.assertEqual(captured["kwargs"]["device"], "cuda:3")
            visible = [
                (p + i + 1) // 4
                for n, p in zip(c["query_lengths"], c["prefix_tokens"])
                for i in range(n)
            ]
            self.assertEqual(captured["kwargs"]["visible"], visible)
            self.assertEqual(args[9], c["batch_max_seq_len"])

    def test_captured_builder_routes_to_existing_full_cache_remap(self):
        helper, torch = object(), object()
        for case in fp8.cases()[:8] + fp8.functional_cases():
            with patch.object(inference, "make_inputs", return_value="inputs") as build:
                self.assertEqual(
                    workloads.make_inputs(case, helper, torch, device="cuda:3"),
                    "inputs",
                )
                build.assert_called_once_with(case, helper, torch, device="cuda:3")

    def test_many_request_captures_are_functional_only(self):
        manifest = json.loads(
            (ROOT / "docs/qsa_prefill_fp8_shape_manifest.json").read_text()
        )
        cases = fp8.functional_cases()
        self.assertEqual(len(cases), 2)
        for c, record in zip(cases, manifest["functional"]):
            self.assertEqual(c["captured_parameters"], record["parameters"])
            self.assertEqual(c["recorded_count"], record["count"])
            g = workloads.geometry(c)
            self.assertTrue(g["select_all"])
            self.assertEqual(g["q_shape"], [sum(c["query_lengths"]), 4, 128])
            for suite in workloads.SUITES:
                self.assertNotIn(c["name"], [v["name"] for v in workloads.cases(suite)])
        self.assertEqual([len(c["query_lengths"]) for c in cases], [2048, 451])
        self.assertEqual([sum(c["query_lengths"]) for c in cases], [16384, 902])

    def test_pytest_benchmark_uses_one_dtype_per_named_case(self):
        path = ROOT / "benchmark/test_qsa_select_paged_prefill_perf.py"
        body = [
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef)
            and n.name in ("QSAPrefillBenchmark", "QSAScoredBenchmark")
            or isinstance(n, ast.FunctionDef)
            and n.name == "_inputs"
            or isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "BENCHMARK_CASES" for t in n.targets
            )
        ]
        torch = types.SimpleNamespace(bfloat16="BF16", float8_e4m3fn="FP8")
        ns = dict(
            base=types.SimpleNamespace(GenericBenchmark=object),
            workloads=workloads,
            torch=torch,
            input_helpers=object(),
        )
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
        for cls, counts in (
            ("QSAPrefillBenchmark", (20, 10)),
            ("QSAScoredBenchmark", (20, 10)),
        ):
            obj = ns[cls]()
            obj.set_shapes()
            obj.device = "cuda:0"
            obj.input_fn = lambda shape, dtype, device: iter([(shape, dtype)])
            for dtype, expected in zip(("BF16", "FP8"), counts):
                items = list(obj.get_input_iter(dtype))
                self.assertEqual(len(items), expected)
                self.assertTrue(
                    all(
                        getattr(torch, ns["BENCHMARK_CASES"][shape]["dtype"]) == d
                        for shape, d in items
                    )
                )
        wrong = next(iter(ns["BENCHMARK_CASES"]))
        with self.assertRaises(AssertionError):
            next(ns["_inputs"](wrong, "FP8", "cuda:0"))

    def test_actual_fp8_reference_contract_and_bf16_behavior(self):
        helpers = types.SimpleNamespace(
            assert_selection=Mock(), assert_upstream_selection=Mock()
        )
        torch = types.SimpleNamespace(float8_e4m3fn="FP8")
        args = (object(), object(), object())
        for dtype, dimension in (("BF16", 128), ("FP8", 96)):
            q = types.SimpleNamespace(dtype=dtype, shape=(7, 5, dimension))
            helpers.assert_selection.reset_mock()
            helpers.assert_upstream_selection.reset_mock()
            bench.validate_reference_selection(*args, q, torch, helpers)
            if dtype == "BF16":
                helpers.assert_selection.assert_called_once_with(*args)
                helpers.assert_upstream_selection.assert_not_called()
            else:
                helpers.assert_upstream_selection.assert_called_once_with(
                    *args, dtype="FP8", head_dim=96
                )
                helpers.assert_selection.assert_not_called()

    def test_topk_common_logits_stays_strict_and_uses_actual_dtype_dimension(self):
        class Tensor:
            def __getitem__(self, key):
                return self

            def __sub__(self, other):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [0, 1]

        q = types.SimpleNamespace(dtype="FP8", shape=(1, 5, 96))
        common = Tensor()
        starts = Tensor()
        visible = Tensor()
        left = Tensor()
        right = Tensor()
        a = dict(
            common_logits=common,
            score_args=(q, None, None, starts),
            offset=0,
            rows=1,
            topk_args=(None, visible, None, None, left),
            topk_kwargs={},
        )
        b = dict(topk_args=(None, visible, right), topk_kwargs={})
        helpers = types.SimpleNamespace(
            assert_selection=Mock(), assert_native_selection=Mock()
        )
        native = types.SimpleNamespace(_topk=Mock())
        candidate = types.SimpleNamespace(persistent_topk=Mock())
        with patch.object(bench.scoring, "request_sample_rows", return_value=[0]):
            self.assertEqual(
                bench.validate_topk(native, candidate, [a], [b], object(), helpers), 1
            )
        self.assertEqual(helpers.assert_selection.call_count, 2)
        self.assertEqual(
            helpers.assert_native_selection.call_args.kwargs,
            dict(dtype="FP8", head_dim=96),
        )
        self.assertIs(
            native._topk.call_args.args[0], candidate.persistent_topk.call_args.args[0]
        )

    def test_fp8_cli_and_separate_incomplete_group_reporting(self):
        command = [
            sys.executable,
            "-I",
            str(Path(bench.__file__)),
            "--compiler",
            "flagtree",
            "--suite",
            "fp8-prefill",
            "--candidate",
            "production",
            "--list-cases",
        ]
        proc = subprocess.run(command, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(proc.stdout), fp8.cases())
        cases = workloads.cases("combined-prefill")

        def entry(case):
            return dict(
                case=case,
                status="passed",
                stages={
                    name: dict(
                        status="passed",
                        accepted=True,
                        measurement_validated=True,
                        baseline_us=dict(median=4),
                        candidate_us=dict(median=2),
                        speedup=2,
                        paired_ratios=dict(median=2),
                    )
                    for name in ("score", "topk", "pipeline")
                },
            )

        summary = dict(
            manifest=dict(options=dict(compiler="flagtree"), cases=cases),
            cases=[entry(c) for c in cases],
        )
        for c in summary["cases"]:
            if workloads.geometry(c["case"]).get("select_all"):
                for stage in ("score", "topk"):
                    c["stages"][stage] = dict(status="not_applicable", accepted=False)
        with tempfile.TemporaryDirectory() as directory:
            bench.write_table(summary, Path(directory))
            for group, count in (
                ("upstream_perf", 12),
                ("observed_perf", 8),
                ("fp8_perf", 10),
            ):
                self.assertEqual(
                    summary["grouped_speedups"][group + ":score"]["count"], count
                )
            self.assertEqual(
                summary["grouped_speedups"]["observed_perf:pipeline"]["count"], 10
            )
            summary["cases"].pop()
            bench.write_table(summary, Path(directory))
            self.assertNotIn("fp8_perf:score", summary["grouped_speedups"])
            self.assertIn("upstream_perf:score", summary["grouped_speedups"])


@unittest.skipUnless(
    importlib.util.find_spec("numpy"), "NumPy CPU tensor adapter required"
)
class CapturedCapacityBuilderTests(unittest.TestCase):
    def test_real_builder_preserves_live_keys_and_allocates_captured_capacities(self):
        import numpy as np

        class Tensor(np.ndarray):
            def copy_(self, value):
                self[...] = value
                return self

            def long(self):
                return self.astype(np.int64).view(Tensor)

            def to(self, dtype):
                return self.astype(dtype).view(Tensor)

            def stride(self):
                return tuple(s // self.itemsize for s in self.strides)

        def tensor(value, dtype):
            return np.asarray(value, dtype=dtype).view(Tensor)

        torch = types.SimpleNamespace(
            float8_e4m3fn=np.uint8,
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
        original = {}

        def base_inputs(queries, contexts, **kw):
            page, dim, heads = kw["page_size"], kw["head_dim"], kw["num_heads"]
            pages = [(c + page - 1) // page for c in contexts]
            cache = tensor(
                (np.arange(sum(pages) * page * dim) % 251).reshape(
                    sum(pages), page, 1, dim
                ),
                np.uint8,
            )
            table = tensor(np.full((len(queries), max(pages)), -1), np.int32)
            cursor = 0
            for r, count in enumerate(pages):
                table[r, :count] = np.arange(cursor, cursor + count)
                cursor += count
            starts = [0]
            for q in queries:
                starts.append(starts[-1] + q)
            original.update(
                cache=cache.copy(), table=table.copy(), visible=kw["visible"]
            )
            return (
                tensor(np.zeros((sum(queries), heads, dim)), np.uint8),
                cache,
                table,
                tensor(starts, np.int32),
                tensor(kw["visible"], np.int32),
                kw["k"] * kw["ratio"],
                kw["ratio"],
                max(queries),
                tensor(np.full((sum(queries), kw["k"]), -17), np.int32),
                max(contexts) * kw["ratio"],
            )

        # Run the actual shared remap/padding path at recorded capacities;uint8
        # stands for opaque FP8 bytes,not a substitute numerical score reference.
        for case in (fp8.cases()[0], fp8.cases()[5]):
            args = workloads.make_inputs(
                case,
                types.SimpleNamespace(make_inputs=base_inputs),
                torch,
                device="cpu",
            )
            info = workloads.geometry(case)
            for i, key in ((0, "q_shape"), (1, "cache_shape"), (2, "page_table_shape")):
                self.assertEqual(list(args[i].shape), info[key])
                self.assertEqual(
                    list(args[i].stride()), info["input_layout"][str(i)]["stride"]
                )
            self.assertEqual(args[3].tolist(), info["query_start_loc"])
            self.assertEqual(args[4].tolist(), original["visible"])
            self.assertEqual(args[9], case["batch_max_seq_len"])
            self.assertTrue(np.all(args[2][:, original["table"].shape[1] :] == -1))
            for request, row in enumerate(original["table"]):
                for column, old_id in enumerate(row):
                    if old_id >= 0:
                        np.testing.assert_array_equal(
                            args[1][args[2][request, column]], original["cache"][old_id]
                        )
                    else:
                        self.assertEqual(args[2][request, column], -1)
            ids = set(args[2][args[2] >= 0].tolist())
            self.assertEqual(len(ids), info["active_physical_pages"])
            self.assertGreater(max(ids), info["active_physical_pages"] - 1)
            unused = next(i for i in range(2712) if i not in ids)
            self.assertTrue(np.all(args[1][unused] == 0))


if __name__ == "__main__":
    unittest.main()
