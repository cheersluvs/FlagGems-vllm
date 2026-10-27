# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Production transfer, metadata pruning and retained fallback contracts."""

import ast
import copy
import hashlib
import types
import unittest
from pathlib import Path

import yaml

from tools import qsa_prefill_triton_ab as ab

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
GROUP = "qsa_select_paged_prefill_head_group_score"


def functions(path):
    return {
        n.name: n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
    }


def normalize(body, grouped):
    class Branch(ast.NodeTransformer):
        def visit_AnnAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == "GROUP_HEADS":
                return None
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "HEADS_PADDED"
                and isinstance(node.value, ast.IfExp)
            ):
                node.value = node.value.body if grouped else node.value.orelse
            return self.generic_visit(node)

        def visit_If(self, node):
            if isinstance(node.test, ast.Name) and node.test.id == "GROUP_HEADS":
                return [self.visit(n) for n in (node.body if grouped else node.orelse)]
            return self.generic_visit(node)

    return Branch().visit(ast.Module(body=copy.deepcopy(body), type_ignores=[]))


class HeadGroupProductionTests(unittest.TestCase):
    def test_original_branch_and_other_paths_unchanged(self):
        frozen = ROOT / ab.PRE_HEAD_GROUP_PRODUCTION_PATH
        self.assertEqual(
            hashlib.sha256(frozen.read_bytes()).hexdigest(),
            ab.PRE_HEAD_GROUP_PRODUCTION_SHA256,
        )
        current, old = functions(PRODUCTION), functions(frozen)
        for name, node in old.items():
            if name not in {
                "_qsa_prefill_score_kernel",
                "_prune_fp8_store_configs",
                "_qsa_prefill_fp8_store_score_kernel",
            }:
                self.assertEqual(ast.dump(current[name]), ast.dump(node), name)
        self.assertEqual(
            ast.dump(normalize(current["_qsa_prefill_score_kernel"].body, False)),
            ast.dump(
                ast.Module(body=old["_qsa_prefill_score_kernel"].body, type_ignores=[])
            ),
        )

    def test_grouped_body_is_exact_proven_private_transfer(self):
        current = functions(PRODUCTION)["_qsa_prefill_score_kernel"]
        private = functions(ROOT / "tools/qsa_prefill_head_group_kernel.py")[
            "_head_group_score_kernel"
        ]
        body = copy.deepcopy(private.body)
        body = [
            n
            for n in body
            if not (
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == "static_assert"
            )
        ]

        class Transfer(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == "query_low":
                    node.id = "query"
                return node

            def visit_If(self, node):
                if isinstance(node.test, ast.Name) and node.test.id == "IS_FP8":
                    return node.orelse
                return self.generic_visit(node)

        self.assertEqual(
            ast.dump(normalize(current.body, True)),
            ast.dump(Transfer().visit(ast.Module(body=body, type_ignores=[]))),
        )

    def test_yaml_keeps_ten_original_choices_and_one_grouped_choice(self):
        groups = yaml.safe_load(
            (
                ROOT / "src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml"
            ).read_text()
        )
        current = groups[GROUP]
        self.assertEqual(len(current), 11)
        original = copy.deepcopy(current[:10])
        for c in original:
            self.assertIs(c["META"].pop("HEAD_GROUP"), False)
        self.assertEqual(
            original, groups["qsa_select_paged_prefill_tuning_round1_score"]
        )
        self.assertEqual(
            current[-1],
            dict(
                META=dict(TILE_R=64, BLOCK_N=64, K_TILES=16, HEAD_GROUP=True),
                num_warps=4,
                num_stages=2,
            ),
        )

    def test_actual_pruning_and_compile_time_gate(self):
        groups = yaml.safe_load(
            (
                ROOT / "src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml"
            ).read_text()
        )
        configs = [
            types.SimpleNamespace(kwargs=c["META"], num_warps=c["num_warps"])
            for c in groups[GROUP]
        ]
        fn = functions(PRODUCTION)["_prune_head_group_configs"]
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "prune", "exec"), ns)
        for fp8, h, d, count in (
            (False, 4, 128, 11),
            (True, 4, 128, 10),
            (False, 3, 24, 10),
            (False, 8, 128, 10),
            (False, 4, 256, 10),
        ):
            got = ns[fn.name](configs, dict(IS_FP8=fp8, NUM_HEADS=h, HEAD_DIM=d))
            self.assertEqual(got, configs[:count])
        grouped = configs[-1]
        for tile, warps in ((32, 4), (64, 8)):
            invalid = types.SimpleNamespace(
                kwargs=dict(grouped.kwargs, TILE_R=tile), num_warps=warps
            )
            self.assertEqual(
                ns["_prune_head_group_configs"](
                    configs[:10] + [invalid],
                    dict(IS_FP8=False, NUM_HEADS=4, HEAD_DIM=128),
                ),
                configs[:10],
            )
        fn = functions(PRODUCTION)["_qsa_prefill_score_kernel"]
        gate = next(
            n.value
            for n in fn.body
            if isinstance(n, ast.AnnAssign) and n.target.id == "GROUP_HEADS"
        )
        for flag, fp8, h, d, tile, wanted in (
            (True, False, 4, 128, 64, True),
            (False, False, 4, 128, 64, False),
            (True, True, 4, 128, 64, False),
            (True, False, 8, 128, 64, False),
            (True, False, 4, 256, 64, False),
            (True, False, 4, 128, 32, False),
        ):
            self.assertEqual(
                eval(
                    compile(ast.Expression(gate), "gate", "eval"),
                    dict(
                        HEAD_GROUP=flag,
                        IS_FP8=fp8,
                        NUM_HEADS=h,
                        HEAD_DIM=d,
                        TILE_R=tile,
                    ),
                ),
                wanted,
            )


if __name__ == "__main__":
    unittest.main()
