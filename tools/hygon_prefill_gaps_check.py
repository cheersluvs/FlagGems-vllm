"""CPU construction/isolation and selection-model checks, not GPU validation."""

import ast
import random
import unittest
from argparse import Namespace
from pathlib import Path

from hygon_prefill_audit_source import build, function_text
from hygon_prefill_gaps import jobs
from hygon_prefill_gaps_source import atomic_to_scan, variant

ROOT = Path(__file__).resolve().parents[1]


def definitions(source):
    return {
        n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef)
    }


class Checks(unittest.TestCase):
    def test_complete_matrix_and_independent_workers(self):
        targets = list(jobs(Namespace(stage="all", shape_ids=None)))
        self.assertEqual(len(targets), 182)
        self.assertEqual(len(targets), len(set(targets)))
        self.assertEqual(targets[0], ("preflight", 0, "control", 0))
        for stage in ("radix", "counters", "scratch", "compression", "tle"):
            self.assertEqual(
                {sid for st, sid, _, _ in targets if st == stage}, set(range(7))
            )

    def test_control_matches_current_carry_and_sparse(self):
        for dense in (False, True):
            self.assertEqual(
                variant(ROOT, dense, "control"),
                build(ROOT, dense, "carry" if dense else "control"),
            )

    def test_radix_isolation_and_scratch_stride(self):
        for dense in (False, True):
            before = definitions(variant(ROOT, dense, "control"))
            for gate in (0, 64, 256):
                source = variant(ROOT, dense, f"radix{gate}")
                after = definitions(source)
                self.assertEqual(
                    {key for key in before if before[key] != after[key]},
                    {
                        "_final_select_radix",
                        "_top_k_per_row_job",
                        "non_tle_top_k_per_row_prefill",
                    },
                )
                final = function_text(source, "_final_select_radix")
                self.assertNotIn("tle.", final)
                self.assertIn("s_histogram_ptr + 2048", final)
                self.assertIn(
                    "row_id * 2304",
                    function_text(source, "non_tle_top_k_per_row_prefill"),
                )
                compile(source, "candidate", "exec")

    def test_counter_rewrite_scope(self):
        for dense, arms in (
            (False, ("found_scan", "final_scan", "both_scan")),
            (True, ("final_scan",)),
        ):
            base = variant(ROOT, dense, "control")
            before = definitions(base)
            function = "_process_bins_slotscan" if dense else "_process_bins"
            for arm in arms:
                source = variant(ROOT, dense, arm)
                after = definitions(source)
                self.assertEqual(
                    {key for key in before if before[key] != after[key]}, {function}
                )
                self.assertEqual(
                    before["_process_histogram_step"], after["_process_histogram_step"]
                )
                self.assertIn(
                    "s_histogram_ptr + bin_idx", function_text(source, function)
                )
                compile(source, "candidate", "exec")

    def test_source_drift_fails_closed(self):
        source = variant(ROOT, False, "control")
        with self.assertRaises(ValueError):
            atomic_to_scan(source, "_process_bins", "not_a_counter", "take_lt")
        with self.assertRaises(ValueError):
            atomic_to_scan(source, "_process_bins", "final_cnt_ptrs", "wrong_mask")

    def test_radix_threshold_model(self):
        # Mirrors the existing remain+1 threshold convention and early stop.
        # Compare all selected key multisets to sorted keys, including ties.
        rng = random.Random(42)
        cases = [
            [0] * 1025,
            [0, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000],
            [rng.getrandbits(32) for _ in range(2048)],
            [rng.randrange(8) << 24 for _ in range(257)],
        ]
        for keys in cases:
            for remain in sorted({1, len(keys) // 2, len(keys) - 1, len(keys)}):
                desired, mask, k = 0, 0, remain + 1
                for pos in (24, 16, 8, 0):
                    if k > 1:
                        counts = [0] * 256
                        for key in keys:
                            if key & mask == desired:
                                counts[(key >> pos) & 255] += 1
                        prefix, digit = 0, 255
                        for i, count in enumerate(counts):
                            if prefix < k <= prefix + count:
                                digit = i
                                break
                            prefix += count
                        below = sum(counts[:digit])
                        desired |= digit << pos
                        mask |= 255 << pos
                        k -= below
                chosen = [key for key in keys if key < desired][:remain]
                chosen += [key for key in keys if key == desired][
                    : remain - len(chosen)
                ]
                self.assertEqual(sorted(chosen), sorted(keys)[:remain])


if __name__ == "__main__":
    unittest.main()
