# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU host routing/source checks; vector numerics require the on-card gate."""

import ast
import hashlib
import inspect
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_qsa_prefill_chunk_setup as chunk

TRIAL = Path(__file__).with_name("qsa_prefill_single_row_trial.py")


def load():
    frozen = Path(__file__).with_name("qsa_prefill_frozen_production.py")
    original, dot_calls, topks, allocations = chunk.load(frozen)
    op = types.ModuleType("qsa_production")
    op.__dict__.update(original)
    op.__file__ = str(frozen)
    vector_calls = []

    class Vector:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                vector_calls.append(
                    (
                        grid(dict(TILE_R=1, BLOCK_N=32, K_TILES=1)),
                        tuple(map(chunk.normalize, args)),
                        kwargs,
                    )
                )

            return launch

    tree = ast.parse(TRIAL.read_text())
    constants = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
    }
    namespace = dict(
        __file__=str(TRIAL),
        hashlib=hashlib,
        inspect=inspect,
        types=types,
        Path=Path,
        _qsa_prefill_single_row_score_kernel=Vector(),
        **constants,
    )
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_prefill_logits", "candidate_module"}
    ]
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(TRIAL), "exec"),
        namespace,
    )
    return op, namespace, dot_calls, vector_calls, topks, allocations


class SingleRowTrialTests(unittest.TestCase):
    def test_only_eligible_chunks_use_vector_and_arguments_are_preserved(self):
        for rows in (1, 65, 73):
            for budget in ("512", "1", "0"):
                for heads, dim in ((1, 7), (4, 128), (8, 256), (17, 33), (1, 257)):
                    op, trial, dot, vector, topks, alloc = load()
                    args = chunk.inputs(rows=rows, padded=True, heads=heads, dim=dim)
                    originals = dict(vars(op))
                    candidate = trial["candidate_module"](op)
                    with patch.dict(
                        os.environ, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=budget
                    ):
                        op.qsa_select_paged_prefill(*args)
                        old_calls = list(dot)
                        old_bytes = [x.numel() * x.element_size() for x in alloc]
                        old_topks = [
                            (tuple(map(chunk.normalize, a)), kw) for a, kw in topks
                        ]
                        dot.clear()
                        topks.clear()
                        alloc.clear()
                        candidate.qsa_select_paged_prefill(*args)
                    eligible = [
                        x
                        for x in old_calls
                        if x[1][12] == 1 and heads <= 8 and dim <= 256
                    ]
                    expected_dot = [
                        x
                        for x in old_calls
                        if not (x[1][12] == 1 and heads <= 8 and dim <= 256)
                    ]
                    self.assertEqual(dot, expected_dot)
                    self.assertEqual(len(vector), len(eligible))
                    for new, old in zip(vector, eligible):
                        self.assertEqual(new[1:], old[1:])
                        self.assertEqual(new[0], (2, 1, 256))
                    self.assertEqual(
                        [x.numel() * x.element_size() for x in alloc], old_bytes
                    )
                    self.assertEqual(
                        [(tuple(map(chunk.normalize, a)), kw) for a, kw in topks],
                        old_topks,
                    )
                    self.assertTrue(
                        all(
                            getattr(op, name) is value
                            for name, value in originals.items()
                        )
                    )

    def test_private_module_shares_public_bytecode_and_frozen_callables(self):
        op, trial, _, _, _, _ = load()
        candidate = trial["candidate_module"](op)
        self.assertIs(
            candidate.qsa_select_paged_prefill.__code__,
            op.qsa_select_paged_prefill.__code__,
        )
        self.assertIs(candidate.qsa_select_paged_prefill.__globals__, vars(candidate))
        self.assertIs(candidate._prefill_logits.__globals__, vars(candidate))
        self.assertIsNot(candidate._prefill_logits, op._prefill_logits)
        for name in (
            "_validate_inputs",
            "_prefill_score_metadata",
            "_qsa_prefill_score_kernel",
            "persistent_topk",
            "_select_all_kernel",
        ):
            self.assertIs(getattr(candidate, name), getattr(op, name))
        self.assertEqual(
            candidate._prefill_logits.__kwdefaults__, {"launch_metadata": None}
        )

    def test_changed_production_or_runtime_function_is_rejected(self):
        op, trial, _, _, _, _ = load()
        op._prefill_logits = lambda *a: None
        with self.assertRaisesRegex(RuntimeError, "pinned source"):
            trial["candidate_module"](op)
        op, trial, _, _, _, _ = load()
        op.__file__ = str(TRIAL)
        with self.assertRaisesRegex(RuntimeError, "production source file"):
            trial["candidate_module"](op)

    def test_trial_uses_no_vendor_intrinsic_or_torch_compute(self):
        tree = ast.parse(TRIAL.read_text())
        kernel = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_qsa_prefill_single_row_score_kernel"
        )
        calls = [
            ast.unparse(n.func) for n in ast.walk(kernel) if isinstance(n, ast.Call)
        ]
        self.assertFalse(
            any(x.startswith("torch.") or "asm" in x or "cuda" in x for x in calls)
        )
        self.assertIn("tl.sum", calls)
        self.assertNotIn("tl.dot", calls)


if __name__ == "__main__":
    unittest.main()
