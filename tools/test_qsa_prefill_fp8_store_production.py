# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU transfer/dispatch contracts; GPU numerical coverage uses public entry."""

import ast
import copy
import hashlib
import types
import unittest
from pathlib import Path

import test_qsa_prefill_chunk_setup as chunk
import yaml

from tools import qsa_prefill_triton_ab as ab

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
FROZEN = ROOT / ab.PRE_STORE_PRODUCTION_PATH
GROUP = "qsa_select_paged_prefill_fp8_store_score"


def functions(path):
    return {
        n.name: n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
    }


class FP8StoreProductionTests(unittest.TestCase):
    def test_exact_transfer_and_original_paths(self):
        self.assertEqual(
            hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
            ab.PRE_STORE_PRODUCTION_SHA256,
        )
        current, old = functions(PRODUCTION), functions(FROZEN)
        for name in old.keys() - {"_prefill_logits"}:
            self.assertEqual(ast.dump(current[name]), ast.dump(old[name]), name)
        private = functions(ROOT / "tools/qsa_prefill_store_probe_kernel.py")
        for old_name, new_name in (
            ("_bulk_tail_store", "_fp8_bulk_tail_store"),
            ("_qsa_prefill_score_kernel", "_qsa_prefill_fp8_store_score_kernel"),
        ):
            node = copy.deepcopy(private[old_name])
            node.name = new_name
            for part in ast.walk(node):
                if isinstance(part, ast.Name):
                    part.id = {
                        "_bulk_tail_store": "_fp8_bulk_tail_store",
                        "prune_configs": "_prune_fp8_store_configs",
                    }.get(part.id, part.id)
                if (
                    isinstance(part, ast.Constant)
                    and part.value == "qsa_select_paged_prefill_query_joint_score"
                ):
                    part.value = GROUP
            self.assertEqual(ast.dump(node), ast.dump(current[new_name]))
        groups = yaml.safe_load(
            (
                ROOT / "src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml"
            ).read_text()
        )
        self.assertEqual(
            groups[GROUP], groups["qsa_select_paged_prefill_stage_joint_score"]
        )
        self.assertEqual(len(groups[GROUP]), 34)

    def test_gpu_mode_selector_uses_full_yaml_config_including_warps(self):
        groups = yaml.safe_load(
            (
                ROOT / "src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml"
            ).read_text()
        )
        configs = [
            types.SimpleNamespace(
                kwargs=c["META"], num_warps=c["num_warps"], num_stages=c["num_stages"]
            )
            for c in groups[GROUP]
        ]
        function = functions(ROOT / "tests/test_qsa_select_paged_prefill.py")[
            "test_fp8_store_production_modes"
        ]
        # Execute the real GPU test's selection and count assertion, rather
        # than a copied predicate. No GPU imports or kernels are needed.
        body = [
            n
            for n in function.body
            if (
                isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "selected" for t in n.targets
                )
            )
            or isinstance(n, ast.Assert)
        ]
        self.assertEqual(len(body), 2)
        ns = dict(configs=configs)
        exec(
            compile(
                ast.Module(body=body, type_ignores=[]), "gpu_mode_selector", "exec"
            ),
            ns,
        )
        selected = ns["selected"]
        self.assertEqual([c.num_warps for c in selected], [4, 4])
        self.assertEqual([c.num_stages for c in selected], [2, 1])
        self.assertEqual([c.kwargs["QUERY_PARTS"] for c in selected], [1, 2])
        # The original failure was a legitimate extra8-warp choice in YAML.
        self.assertTrue(
            any(
                c.kwargs == selected[1].kwargs
                and c.num_stages == 1
                and c.num_warps == 8
                for c in configs
            )
        )

    def test_actual_host_dispatch_and_alignment(self):
        ns, _, _, _ = chunk.load()
        routes = []

        class Kernel:
            def __init__(self, route):
                self.route = route

            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    routes.append(self.route)

                return launch

        for name, route in (
            ("_qsa_prefill_score_kernel", "original"),
            ("_qsa_prefill_fp8_store_score_kernel", "fp8"),
            ("_qsa_prefill_compact_score_kernel", "compact"),
        ):
            ns[name] = Kernel(route)
        for dtype, width, address, lengths, expected in (
            ("fp8", 1088, 4096, (17,), "fp8"),
            ("bf16", 1088, 4096, (17,), "original"),
            ("fp8", 1088, 4100, (17,), "original"),
            ("fp8", 1089, 4096, (17,), "original"),
            ("fp8", 1088, 4096, (65, 1, 1, 1, 1), "compact"),
        ):
            rows, requests = sum(lengths), len(lengths)
            q = chunk.Array((rows, 4, 128), dtype)
            cache = chunk.Array((128, 16, 1, 128), dtype)
            table = chunk.Array((requests, 128), "i32")
            starts = chunk.Array((requests + 1,), "i32")
            visible = chunk.Array((rows,), "i32")
            workspace = chunk.Array((rows, width), "f32", address=address)
            ns["_prefill_logits"](
                q,
                cache,
                table,
                starts,
                visible,
                workspace,
                max(lengths),
                width,
                0,
                rows,
            )
            self.assertEqual(routes[-1], expected)
        self.assertFalse(
            ns["_fp8_store_eligible"](
                q, chunk.Array((rows, 1088), "f32", strides=(1089, 1)), 1088
            )
        )
        self.assertFalse(
            ns["_fp8_store_eligible"](q, chunk.Array((rows, 1088), "f32"), 1084)
        )

    def test_actual_pruning(self):
        node = functions(PRODUCTION)["_prune_fp8_store_configs"]
        ns = {}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(PRODUCTION), "exec"),
            ns,
        )
        configs = [
            types.SimpleNamespace(kwargs=dict(QUERY_PARTS=p, TILE_R=r))
            for p, r in ((1, 8), (2, 16), (2, 32))
        ]
        self.assertEqual(
            ns[node.name](configs, {"IS_FP8": True}), [configs[0], configs[2]]
        )
        self.assertEqual(ns[node.name](configs, {"IS_FP8": False}), [configs[0]])


if __name__ == "__main__":
    unittest.main()
