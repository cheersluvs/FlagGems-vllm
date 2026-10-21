# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Integration contracts: retained kernels, real dispatch, public identity."""

import ast
import copy
import hashlib
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_qsa_prefill_chunk_setup as chunk

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_observed_workloads as workloads
from tools import qsa_prefill_triton_ab as ab

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
FROZEN = ROOT / ab.FROZEN_PRODUCTION_PATH


def functions(path):
    return {
        n.name: n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
    }


def retained_host():
    # Execute the actual retained trial launcher over the old public host body,
    # with capture-only kernels. No implementation of routing is mirrored here.
    ns, calls, topks, allocations = chunk.load(FROZEN)
    ns["_production_prefill_logits"] = ns["_prefill_logits"]
    tree = functions(ROOT / "tools/qsa_prefill_compact_trial.py")
    exec(
        compile(
            ast.Module(
                body=[tree["_compact_eligible"], tree["_prefill_logits"]],
                type_ignores=[],
            ),
            "retained_compact_host",
            "exec",
        ),
        ns,
    )
    return ns, calls, topks, allocations


class ProductionIntegrationTests(unittest.TestCase):
    def test_kernels_and_gate_are_exact_transfers(self):
        current = functions(PRODUCTION)
        old = functions(FROZEN)
        rect = copy.deepcopy(
            functions(ROOT / "tools/qsa_prefill_key_major_trial.py")[
                "_qsa_prefill_key_major_score_kernel"
            ]
        )
        rect.name = "_qsa_prefill_score_kernel"
        # Config selection and the operator-local outer cache decorator differ;
        # kernel arithmetic, signature, keys and JIT specialization are exact.
        rect.decorator_list[0] = ast.Name(id="_QSAScoreEntry", ctx=ast.Load())
        for node in ast.walk(rect.decorator_list[1]):
            if (
                isinstance(node, ast.Constant)
                and node.value == "qsa_select_paged_prefill_score"
            ):
                node.value = "qsa_select_paged_prefill_tuning_round1_score"
        self.assertEqual(ast.dump(rect), ast.dump(current[rect.name]))
        compact = functions(ROOT / "tools/qsa_prefill_compact_trial.py")
        for name in ("_qsa_prefill_compact_score_kernel", "_compact_eligible"):
            self.assertEqual(ast.dump(current[name]), ast.dump(compact[name]))
        for name in old.keys() - {"_qsa_prefill_score_kernel", "_prefill_logits"}:
            self.assertEqual(ast.dump(current[name]), ast.dump(old[name]), name)
        self.assertNotIn("eviction_policy", PRODUCTION.read_text())

    def test_pins_exports_and_no_production_dependency_on_trials(self):
        self.assertEqual(
            hashlib.sha256(FROZEN.read_bytes()).hexdigest(), ab.PRODUCTION_SHA256
        )
        self.assertEqual(
            hashlib.sha256(PRODUCTION.read_bytes()).hexdigest(),
            ab.INTEGRATED_PRODUCTION_SHA256,
        )
        tree = ast.parse(PRODUCTION.read_text())
        imports = [
            n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
        ]
        self.assertTrue(all(not s.startswith(("tools", "tests")) for s in imports))
        for path in (
            ROOT / "src/flaggems_vllm/ops/__init__.py",
            ROOT / "src/flaggems_vllm/__init__.py",
        ):
            self.assertIn("qsa_select_paged_prefill", path.read_text())
        # All changed device arithmetic exactly matches reviewed trial ASTs;
        # host/public validation and allocation bodies are unchanged above.
        self.assertEqual(
            {
                n.func.attr
                for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "torch"
            },
            {"empty"},
        )

    def test_actual_production_dispatch_matches_retained_host_on_all32(self):
        routed = []
        for case in workloads.cases("combined-prefill"):
            info = workloads.geometry(case)
            with self.subTest(case=case["name"]):
                outputs = []
                for loader in (chunk.load, retained_host):
                    ns, calls, topks, allocations = loader()
                    rows = info["q_shape"][0]
                    shapes = [
                        info["q_shape"],
                        info["cache_shape"],
                        info["page_table_shape"],
                        [len(info["query_lengths"]) + 1],
                        [rows],
                        [rows, 512],
                    ]
                    tensors = [
                        chunk.Array(
                            s,
                            (
                                ("fp8" if case["dtype"] == "float8_e4m3fn" else "bf16")
                                if i < 2
                                else "i32"
                            ),
                            (i + 1) * 10000000,
                            label=f"input_{i}",
                        )
                        for i, s in enumerate(shapes)
                    ]
                    args = [
                        *tensors[:5],
                        2048,
                        4,
                        max(info["query_lengths"]),
                        tensors[5],
                        max(info["sequence_lengths"]),
                    ]
                    with patch.dict(
                        os.environ, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="512"
                    ):
                        self.assertIsNone(ns["qsa_select_paged_prefill"](*args))
                    outputs.append(
                        (
                            calls,
                            [(tuple(map(chunk.normalize, a)), kw) for a, kw in topks],
                            [chunk.normalize(a) for a in allocations],
                        )
                    )
                    if (
                        loader is chunk.load
                        and any(
                            ns["_compact_eligible"](
                                rows, len(info["query_lengths"]), args[7], n
                            )
                            for n in info["expected_chunk_rows"]
                        )
                        and not info.get("select_all", False)
                    ):
                        routed.append(case["name"])
                self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(
            routed, ["observed_ragged54_bfloat16", "fp8_compact_float8_e4m3fn"]
        )

    def test_chunked_and_strided_host_transfer(self):
        for rows, requests, maximum in ((73, 2, 39), (200, 8, 100), (50, 8, 33)):
            for dtype in ("bf16", "fp8"):
                for padded in (False, True):
                    for budget in ("512", "1", "0"):
                        with self.subTest(
                            rows=rows,
                            requests=requests,
                            dtype=dtype,
                            padded=padded,
                            budget=budget,
                        ):
                            outputs = []
                            for loader in (chunk.load, retained_host):
                                ns, calls, topks, allocations = loader()
                                args = chunk.inputs(
                                    rows=rows, dtype=dtype, padded=padded
                                )
                                args[2] = chunk.Array(
                                    (requests, 512),
                                    "i32",
                                    30000000,
                                    (2048, 2),
                                    label="input_2",
                                )
                                args[3] = chunk.Array(
                                    (requests + 1,), "i32", 40000000, label="input_3"
                                )
                                args[7] = maximum
                                with patch.dict(
                                    os.environ, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=budget
                                ):
                                    ns["qsa_select_paged_prefill"](*args)
                                outputs.append(
                                    (
                                        calls,
                                        [
                                            (tuple(map(chunk.normalize, a)), kw)
                                            for a, kw in topks
                                        ],
                                        [chunk.normalize(a) for a in allocations],
                                    )
                                )
                            self.assertEqual(outputs[0], outputs[1])

    def test_public_binding_rejects_clone_wrong_export_or_changed_source(self):
        op = types.ModuleType("flaggems_vllm.ops.qsa_select_paged_prefill")
        op.__file__ = str(PRODUCTION)
        exec("def qsa_select_paged_prefill(): pass", vars(op))
        binding = bench.operator_binding(
            op, op, op.qsa_select_paged_prefill, require_production=True
        )
        self.assertTrue(binding["candidate_is_canonical_module"])
        clone = types.ModuleType("trial")
        clone.__dict__.update(vars(op))
        for candidate, exported in (
            (clone, op.qsa_select_paged_prefill),
            (op, lambda: None),
        ):
            with self.assertRaisesRegex(RuntimeError, "actual exported entry"):
                bench.operator_binding(op, candidate, exported, require_production=True)
        with patch.object(ab, "INTEGRATED_PRODUCTION_SHA256", "changed"):
            with self.assertRaisesRegex(RuntimeError, "actual exported entry"):
                bench.operator_binding(
                    op, op, op.qsa_select_paged_prefill, require_production=True
                )

    def test_production_accuracy_worker_uses_official_file_and_identity_checks(self):
        fn = functions(Path(bench.__file__))["accuracy_worker"]
        ns = dict(bench.__dict__)
        calls = []
        from tools.qsa_prefill_score_resume import RUNTIME_KEYS

        info = {k: "fixed" for k in RUNTIME_KEYS}
        info["operator_binding"] = {"candidate_is_canonical_module": True}
        ns["runtime"] = lambda *a, **kw: (info,)
        ns["support"] = types.SimpleNamespace(write_json=lambda *a: None)
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), "accuracy_worker", "exec"),
            ns,
        )
        pytest = types.SimpleNamespace(main=lambda args: calls.append(args) or 0)
        for candidate in ("production", "packed-grid", "page-table", "page-table-bf16"):
            calls.clear()
            with patch.dict("sys.modules", pytest=pytest):
                options = types.SimpleNamespace(
                    compiler="flagtree",
                    candidate=candidate,
                    reference_file="reference.py",
                    report_dir="report",
                )
                self.assertEqual(
                    ns["accuracy_worker"](options, Path("runtime.json")), 0
                )
            paths = [p for p in calls[0] if p.endswith(".py")]
            expected = [str(ROOT / "tests/test_qsa_select_paged_prefill.py")]
            if candidate == "packed-grid":
                expected.insert(0, str(ROOT / "tests/test_qsa_prefill_packed_grid.py"))
            if candidate == "page-table":
                expected.insert(0, str(ROOT / "tests/test_qsa_prefill_page_table.py"))
            if candidate == "page-table-bf16":
                expected.insert(
                    0, str(ROOT / "tests/test_qsa_prefill_page_table_bf16.py")
                )
            self.assertEqual(paths, expected)
        changed = dict(info, operator_binding={"candidate_is_canonical_module": False})
        self.assertNotEqual(
            bench.runtime_identity(info), bench.runtime_identity(changed)
        )


if __name__ == "__main__":
    unittest.main()
