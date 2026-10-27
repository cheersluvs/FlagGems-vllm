# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Paired branch transfer,retained fallbacks and source/config cache isolation."""

import ast
import copy
import hashlib
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from tools import qsa_prefill_fp8_dot_issue as proven
from tools import qsa_prefill_triton_ab as ab

ROOT = Path(__file__).resolve().parents[1]
OP = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
OLD = ROOT / ab.PRE_PAIRED_PRODUCTION_PATH
GROUP = "qsa_select_paged_prefill_fp8_store_score"


def functions(path):
    return {
        n.name: n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
    }


def normalize(node, paired):
    class Branch(ast.NodeTransformer):
        def visit_AnnAssign(self, n):
            if isinstance(n.target, ast.Name) and n.target.id == "PAIR_DOT":
                return None
            return self.generic_visit(n)

        def visit_If(self, n):
            if isinstance(n.test, ast.Name) and n.test.id == "PAIR_DOT":
                return [self.visit(v) for v in (n.body if paired else n.orelse)]
            if (
                isinstance(n.test, ast.UnaryOp)
                and isinstance(n.test.operand, ast.Name)
                and n.test.operand.id == "PAIR_DOT"
            ):
                return [self.visit(v) for v in (n.orelse if paired else n.body)]
            return self.generic_visit(n)

    return Branch().visit(copy.deepcopy(node))


class PairedProductionTests(unittest.TestCase):
    def test_all_other_paths_and_original_branch_are_unchanged(self):
        self.assertEqual(
            hashlib.sha256(OLD.read_bytes()).hexdigest(),
            ab.PRE_PAIRED_PRODUCTION_SHA256,
        )
        current, old = functions(OP), functions(OLD)
        for name in old.keys() - {
            "_qsa_prefill_fp8_store_score_kernel",
            "_prune_fp8_store_configs",
        }:
            self.assertEqual(ast.dump(current[name]), ast.dump(old[name]), name)
        name = "_qsa_prefill_fp8_store_score_kernel"
        self.assertEqual(ast.dump(normalize(current[name], False)), ast.dump(old[name]))
        with patch.object(proven.compiler, "OP", ab.PRE_PAIRED_PRODUCTION_PATH):
            reference = next(
                n
                for n in ast.parse(proven.source("paired")).body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
        reference.decorator_list = copy.deepcopy(current[name].decorator_list)
        self.assertEqual(ast.dump(normalize(current[name], True)), ast.dump(reference))

    def test_pruning_preserves_original_domains_and_enables_only_validated_split_choices(
        self,
    ):
        groups = yaml.safe_load(
            (
                ROOT / "src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml"
            ).read_text()
        )
        configs = [
            types.SimpleNamespace(
                kwargs=v["META"], num_warps=v["num_warps"], num_stages=v["num_stages"]
            )
            for v in groups[GROUP]
        ]
        self.assertEqual(len(configs), 35)
        self.assertEqual(
            groups[GROUP][:-1], groups["qsa_select_paged_prefill_stage_joint_score"]
        )
        funcs = []
        for source in (OLD, OP):
            node = functions(source)["_prune_fp8_store_configs"]
            ns = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "prune", "exec"), ns)
            funcs.append(ns[node.name])
        for dtype, h, d in (
            (True, 3, 24),
            (True, 8, 128),
            (True, 4, 256),
            (False, 4, 128),
        ):
            args = dict(IS_FP8=dtype, NUM_HEADS=h, HEAD_DIM=d)
            self.assertEqual(funcs[0](configs[:-1], args), funcs[1](configs, args))
        selected = funcs[1](configs, dict(IS_FP8=True, NUM_HEADS=4, HEAD_DIM=128))
        split = [c for c in selected if c.kwargs["QUERY_PARTS"] == 2]
        self.assertEqual(
            [(c.kwargs["K_TILES"], c.num_stages) for c in split],
            [(4, 2), (4, 1), (16, 2)],
        )
        self.assertTrue(
            all(
                c.kwargs["TILE_R"] == 32
                and c.kwargs["BLOCK_N"] == 64
                and c.num_warps == 4
                for c in split
            )
        )
        self.assertEqual(
            [c for c in selected if c.kwargs["QUERY_PARTS"] == 1],
            [c for c in configs[:-1] if c.kwargs["QUERY_PARTS"] == 1],
        )


if __name__ == "__main__":
    unittest.main()
