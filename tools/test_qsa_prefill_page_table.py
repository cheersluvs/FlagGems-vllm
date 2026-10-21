# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Execute actual scalar page-load/mask AST with checked fake pointer memory."""

import ast
import copy
import hashlib
import random
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_page_table_trial as trial
from tools import test_qsa_prefill_tuning_round1 as goals

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
KERNEL = ROOT / "tools/qsa_prefill_page_table_kernel.py"


def score(tree):
    return next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_qsa_prefill_score_kernel"
    )


class Pointer:
    def __init__(self, memory, index=0):
        self.memory, self.index = memory, index

    def __add__(self, index):
        return Pointer(self.memory, self.index + index)


class PageTableTests(unittest.TestCase):
    kernel = KERNEL
    bf16_only = False
    fp8 = False

    def test_only_physical_page_load_changes_and_source_is_pinned(self):
        self.assertEqual(
            hashlib.sha256(PRODUCTION.read_bytes()).hexdigest(), trial.PRODUCTION_SHA256
        )
        old, new = score(ast.parse(PRODUCTION.read_text())), score(
            ast.parse(self.kernel.read_text())
        )
        old_loop = next(n for n in old.body if isinstance(n, ast.For))
        new_loop = next(n for n in new.body if isinstance(n, ast.For))
        load_index = next(
            i
            for i, n in enumerate(old_loop.body)
            if isinstance(n, ast.Assign) and n.targets[0].id == "physical_page"
        )
        self.assertIsInstance(new_loop.body[load_index], ast.If)
        new_loop.body[load_index] = copy.deepcopy(old_loop.body[load_index])
        self.assertEqual(ast.dump(old), ast.dump(new))

    def test_actual_load_branch_matches_vector_lookup_at_all_live_columns(self):
        node = score(ast.parse(self.kernel.read_text()))
        loop = next(n for n in node.body if isinstance(n, ast.For))
        branch = next(n for n in loop.body if isinstance(n, ast.If))
        tree = ast.Module(body=[copy.deepcopy(branch)], type_ignores=[])
        code = compile(ast.fix_missing_locations(tree), str(self.kernel), "exec")
        reads = []

        def load(ptr, mask, other):
            if not mask:
                return other
            reads.append(ptr.index)
            self.assertIn(ptr.index, ptr.memory, "unmasked out-of-bounds table read")
            return ptr.memory[ptr.index]

        tl = types.SimpleNamespace(
            load=load, minimum=min, where=lambda cond, a, b: a if cond else b
        )
        rng = random.Random(20261021)
        for page in (1, 7, 16, 64, 96, 196, 256):
            for block in (64, 128):
                for width in (1, 2, 7):
                    for stride in (1, 2, 3):
                        request = 1
                        stride_req = width * stride + 5
                        values = [rng.randrange(0, 30) for _ in range(width)]
                        if width > 2:
                            values[1] = -1
                            values[-1] = 33
                        memory = {
                            request * stride_req + i * stride: v
                            for i, v in enumerate(values)
                        }
                        for visible in sorted(
                            {
                                0,
                                1,
                                max(0, page - 1),
                                page,
                                page + 1,
                                width * page - 1,
                                width * page,
                            }
                        ):
                            for tile in range((width * page + block - 1) // block + 1):
                                for lane in range(block):
                                    column = tile * block + lane
                                    logical = column // page
                                    live = column < visible and logical < width
                                    env = dict(
                                        IS_FP8=self.fp8,
                                        BLOCK_N=block,
                                        PAGE_SIZE=page,
                                        tile=tile,
                                        table_ptr=Pointer(memory),
                                        request=request,
                                        stride_table_req=stride_req,
                                        stride_table_page=stride,
                                        table_width=width,
                                        max_visible=visible,
                                        logical_page=logical,
                                        live=live,
                                        tl=tl,
                                    )
                                    exec(code, env)
                                    if live:
                                        self.assertEqual(
                                            env["physical_page"], values[logical]
                                        )
        self.assertTrue(reads)

    def test_factory_preserves_host_topk_and_isolates_aligned_entry_classes(self):
        class Entry:
            def __init__(self, fn):
                self.fn, self.kernel_cache = fn, ({},)

        production = types.ModuleType("production")
        production.__file__ = str(PRODUCTION)
        production.persistent_topk = object()
        for name in ("_qsa_prefill_score_kernel", "_qsa_prefill_compact_score_kernel"):
            setattr(
                production,
                name,
                Entry(
                    types.SimpleNamespace(configs=[1], keys=["num_rows"], fn=object())
                ),
            )
        exec(
            "def _prefill_logits():\n"
            " return _qsa_prefill_score_kernel,_qsa_prefill_compact_score_kernel\n"
            "def qsa_select_paged_prefill():\n"
            " return _prefill_logits(),persistent_topk\n",
            vars(production),
        )
        kernel = types.ModuleType("tools." + self.kernel.stem)
        kernel.__file__ = str(self.kernel)
        kernel._qsa_prefill_score_kernel = Entry(
            types.SimpleNamespace(configs=[1], keys=["num_rows"], fn=object())
        )
        utils = types.ModuleType("flaggems_vllm.utils")
        utils.libtuner = lambda **kw: lambda jit: types.SimpleNamespace(
            fn=jit, configs=kw["configs"], keys=kw["key"]
        )
        with patch.dict(
            sys.modules,
            {
                "flaggems_vllm.utils": utils,
                kernel.__name__: kernel,
            },
        ):
            candidate = trial.candidate_module(production, bf16_only=self.bf16_only)
        for name in ("_prefill_logits", "qsa_select_paged_prefill"):
            self.assertIs(
                getattr(candidate, name).__code__, getattr(production, name).__code__
            )
        self.assertIs(candidate.persistent_topk, production.persistent_topk)
        self.assertIs(candidate.production_baseline, production)
        for name in ("_qsa_prefill_score_kernel", "_qsa_prefill_compact_score_kernel"):
            self.assertIs(
                type(getattr(candidate, name)), type(getattr(production, name))
            )
            self.assertIsNot(
                getattr(candidate, name).kernel_cache[0],
                getattr(production, name).kernel_cache[0],
            )
        self.assertIs(
            candidate._prefill_logits()[0], candidate._qsa_prefill_score_kernel
        )
        self.assertIs(
            candidate._qsa_prefill_compact_score_kernel.fn.fn,
            production._qsa_prefill_compact_score_kernel.fn.fn,
        )

    def test_experimental_same_production_pipeline_goal_is_retained(self):
        s = goals.AcceptanceTests.summary()
        s["manifest"]["options"].update(
            candidate="page-table-bf16" if self.bf16_only else "page-table",
            compare_previous_tuning=False,
        )
        for c in s["cases"]:
            c["runtime"] = dict(
                operator_binding=dict(canonical_sha256=trial.PRODUCTION_SHA256)
            )
            for k in ("score_vs_production", "pipeline_vs_production"):
                c["stages"][k] = copy.deepcopy(c["stages"]["pipeline"])
        result = trial.acceptance(s, bf16_only=self.bf16_only)
        self.assertTrue(result["measurement_valid"] and result["target_met"])
        self.assertTrue(result["experimental"])
        self.assertFalse(result["formal_production_acceptance"])
        s["cases"][0]["stages"].pop("pipeline_vs_production")
        self.assertFalse(
            trial.acceptance(s, bf16_only=self.bf16_only)["measurement_valid"]
        )


if __name__ == "__main__":
    unittest.main()
