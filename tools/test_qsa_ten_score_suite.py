# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prove frozen workload coverage and unchanged runtime-bound formulas."""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsa_ten_score_suite_910b import (  # noqa: E402
    cases,
    logical_source,
    query_source,
    runtime_source,
    schedule,
    selected_plans,
)


class TenScoreTests(unittest.TestCase):
    def test_all_ten_inputs_and_every_request_boundary_are_covered(self):
        plans = cases()
        self.assertEqual(len(plans), 10)
        self.assertEqual(len({p["name"] for p in plans}), 10)
        mixed = False
        tails = set()
        for plan in plans:
            covered = []
            for chunk in schedule(plan):
                self.assertEqual(
                    chunk["count"],
                    min(plan["chunk_rows"], plan["q_shape"][0] - chunk["first"]),
                )
                parts = []
                for piece in chunk["pieces"]:
                    r = piece["request"]
                    a = piece["first"]
                    b = a + piece["count"]
                    self.assertGreater(piece["count"], 0)
                    self.assertLessEqual(plan["starts"][r], a)
                    self.assertLessEqual(b, plan["starts"][r + 1])
                    self.assertEqual(piece["offset"], a - chunk["first"])
                    parts.extend(range(a, b))
                    tails.add(piece["count"])
                self.assertEqual(
                    parts, list(range(chunk["first"], chunk["first"] + chunk["count"]))
                )
                mixed |= len(chunk["pieces"]) > 1
                covered.extend(parts)
            self.assertEqual(covered, list(range(plan["q_shape"][0])))
        self.assertTrue(mixed)
        self.assertIn(112, tails)
        self.assertIn(113, tails)

    def test_ten_finish_mode_preserves_all_ten_frozen_plans(self):
        args = SimpleNamespace(finish_routing_probe=True, finish_empty_ten=True)
        self.assertEqual(selected_plans(args), cases())
        args.finish_empty_ten = False
        self.assertEqual(
            [p["name"] for p in selected_plans(args)],
            ["int8_first_chunk", "int8_budget_exact", "int8_tail113", "int8_long100k"],
        )

    def test_runtime_source_changes_annotations_only(self):
        old = ast.parse(logical_source() + "\n" + query_source(16))
        new = ast.parse(runtime_source())
        for fn in (n for n in ast.walk(old) if isinstance(n, ast.FunctionDef)):
            for arg in fn.args.args:
                if arg.arg in ("TILES", "RAW_TILES", "MAX_VISIBLE"):
                    arg.annotation = None
        self.assertEqual(ast.dump(old), ast.dump(new))
        for fn in (n for n in ast.walk(new) if isinstance(n, ast.FunctionDef)):
            for arg in fn.args.args:
                if arg.arg in ("TILES", "RAW_TILES", "MAX_VISIBLE"):
                    self.assertIsNone(arg.annotation)

    def test_single_chunk_can_split_many_requests_without_aliasing_rows(self):
        plan = dict(starts=[0, 1, 3, 7, 8], q_shape=[8, 4, 128], chunk_rows=5)
        work = schedule(plan)
        self.assertEqual([p["count"] for p in work[0]["pieces"]], [1, 2, 2])
        self.assertEqual([p["count"] for p in work[1]["pieces"]], [2, 1])
        self.assertEqual([p["offset"] for p in work[1]["pieces"]], [0, 2])

    def test_invalid_boundaries_are_rejected(self):
        for starts, rows, chunk in [
            ([1, 4], 4, 2),
            ([0, 0, 4], 4, 2),
            ([0, 3], 4, 2),
            ([0, 4], 4, 0),
        ]:
            with self.assertRaises(ValueError):
                schedule(dict(starts=starts, q_shape=[rows], chunk_rows=chunk))


if __name__ == "__main__":
    unittest.main()
