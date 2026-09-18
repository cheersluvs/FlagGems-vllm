"""CPU-only construction checks: python3 tools/hygon_prefill_audit_check.py.

These checks do not claim Triton compilation or GPU correctness.
"""

import ast
import random
import unittest
from pathlib import Path

from hygon_prefill_audit_source import build, function_text, replace_once

ROOT = Path(__file__).resolve().parents[1]


def definitions(source):
    return {
        n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef)
    }


class ConstructionChecks(unittest.TestCase):
    def test_all_generated_variants_compile_as_python(self):
        for dense in (False, True):
            for arm in ("control", "rank8", "rank16", "carry"):
                for diagnostic in (False, True):
                    with self.subTest(dense=dense, arm=arm, diagnostic=diagnostic):
                        compile(build(ROOT, dense, arm, diagnostic), arm, "exec")

    def test_rank_changes_only_final_job(self):
        for dense in (False, True):
            before = definitions(build(ROOT, dense))
            for arm in ("rank8", "rank16"):
                after = definitions(build(ROOT, dense, arm))
                changed = {n for n in before if before[n] != after[n]}
                self.assertEqual(changed, {"_top_k_per_row_job"})
                job = function_text(build(ROOT, dense, arm), "_top_k_per_row_job")
                self.assertIn("if final_cnt <= 256:", job)
                self.assertIn("for j in tl.range(0, final_cnt):", job)

    def test_sparse_carry_is_identity(self):
        self.assertEqual(build(ROOT, False), build(ROOT, False, "carry"))

    def test_dense_carry_patches_every_collection_site(self):
        original = build(ROOT, True)
        candidate = build(ROOT, True, "carry")
        before, after = definitions(original), definitions(candidate)
        self.assertEqual(
            {n for n in before if before[n] != after[n]},
            {"_process_histogram_step", "_process_bins_slotscan"},
        )
        step = ast.parse(function_text(candidate, "_process_histogram_step"))
        calls = [
            n
            for n in ast.walk(step)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_process_bins"
        ]
        self.assertEqual(len(calls), 7)
        assignments = [
            n
            for n in ast.walk(step)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and n.value in calls
        ]
        self.assertEqual(len(assignments), 7)
        for call in calls:
            self.assertEqual(call.args[-1].id, "slot_base")
        helper = function_text(candidate, "_process_bins_slotscan")
        self.assertNotIn("_alloc_slots(", helper)
        self.assertIn("return slot_base", helper)
        self.assertIn("final_pos = tl.atomic_add", helper)
        self.assertIn("out_pos_eq = tl.atomic_add", helper)

    def test_diagnostics_are_not_in_timed_sources(self):
        for dense in (False, True):
            control = build(ROOT, dense)
            diagnostic = build(ROOT, dense, diagnostic=True)
            marker = "tl.store(s_threshold_bin_idx_ptr, STEP)"
            self.assertNotIn(marker, control)
            self.assertIn(marker, diagnostic)
            a, b = definitions(control), definitions(diagnostic)
            self.assertEqual(
                {n for n in a if a[n] != b[n]}, {"_process_histogram_step"}
            )

    def test_source_drift_fails_closed(self):
        for src in ("missing", "same same"):
            with self.assertRaises(ValueError):
                replace_once(src, "same", "replacement")

    def test_tiled_rank_mask_and_tie_rule(self):
        # Check tile tails, infinities, equal values and signed zeros. The remote
        # test exercises these through the actual compiled operator as well.
        rng = random.Random(42)
        for n in (1, 7, 8, 9, 15, 16, 17, 27, 177, 255, 256):
            values = [
                rng.choice([-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf")])
                for _ in range(n)
            ]
            exact = [
                sum(x < y or (x == y and i < j) for j, y in enumerate(values))
                for i, x in enumerate(values)
            ]
            self.assertEqual(sorted(exact), list(range(n)))
            for width in (8, 16):
                padded = values + [0.0] * ((-n) % width)
                tiled = [
                    sum(
                        j < n and (x < padded[j] or (x == padded[j] and i < j))
                        for block in range(0, len(padded), width)
                        for j in range(block, block + width)
                    )
                    for i, x in enumerate(values)
                ]
                self.assertEqual(tiled, exact)


if __name__ == "__main__":
    unittest.main(verbosity=2)
