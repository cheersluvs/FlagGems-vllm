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

"""Check runtime visibility, standalone stage and complete mixed aggregation."""

import ast
import json
import tempfile
import unittest
from pathlib import Path

import test_qsa_raw_serial_selection as serial_tests
from qsa_raw_mixed_online_910b import harness_source
from qsa_raw_mixed_selection_910b_trial import PreparedRawMixedSelection, piece_geometry
from qsa_ten_score_suite_910b import schedule
from run_qsa_raw_serial_harness_910b import MIXED_OP_NAME, OP_NAME, cases, summarize


class MixedSelectionTests(unittest.TestCase):
    def test_visibility_rounding_covers_all_and_only_required_blocks(self):
        for width in (1088, 4096, 8256, 8512, 29696):
            for rows in (1, 15, 16, 17, 112, 128):
                for maximum in (0, 1, 512, 513, width):
                    g = piece_geometry(rows, width, maximum)
                    self.assertGreaterEqual(g["blocks"] * 512, maximum)
                    self.assertLess(g["blocks"] * 512 - maximum, 513)
                    self.assertEqual(g["active_tiles"], g["blocks"] * 8)
        for maximum in (-1, 1089):
            with self.assertRaises(ValueError):
                piece_geometry(1, 1088, maximum)

    def test_all_ten_request_splits_cover_each_row_once(self):
        for plan in cases():
            covered = []
            for entry in schedule(plan):
                for piece in entry["pieces"]:
                    first, count, req = piece["first"], piece["count"], piece["request"]
                    self.assertGreaterEqual(first, plan["starts"][req])
                    self.assertLessEqual(first + count, plan["starts"][req + 1])
                    maximum = (
                        plan["prefix_tokens"][req] + first + count - plan["starts"][req]
                    ) // 4
                    maximum = min(plan["logits_width"], maximum)
                    g = piece_geometry(count, plan["logits_width"], maximum)
                    self.assertLessEqual(g["groups"], 20)
                    covered.extend(range(first, first + count))
            self.assertEqual(covered, list(range(plan["q_shape"][0])))

    def test_runtime_bounds_and_vector_diagnostic_have_no_ring_dependencies(self):
        tree = ast.parse(harness_source())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name, parameters in (
            ("raw_mixed_prepared", ("BLOCKS",)),
            ("raw_pack_k_ks_padded", ("MAX_VISIBLE", "TILES")),
        ):
            args = {a.arg: a.annotation for a in functions[name].args.args}
            for parameter in parameters:
                self.assertIn(parameter, args)
                self.assertIsNone(args[parameter])
        fn = functions["raw_online_topk_from_scores"]
        text = ast.unparse(fn)
        for forbidden in (
            "SLOTS",
            "Ring",
            "sub_vec_id",
            "sync_block",
            "mixed_chain_op",
        ):
            self.assertNotIn(forbidden, text)
        self.assertIn("tl.program_id(0) * 8 + update_query", text)
        self.assertIn("sort_1d_pack", text)
        self.assertIn("_update_online_topk_ub_snapshot", text)
        self.assertIn("unpack_sort", text)
        loops = [
            n
            for n in ast.walk(functions["raw_mixed_prepared"])
            if isinstance(n, ast.For) and n.target.id in ("cube_block", "vector_block")
        ]
        self.assertEqual(len(loops), 2)
        self.assertTrue(all(ast.unparse(n.iter) == "range(BLOCKS)" for n in loops))

    def test_whole_call_runs_every_piece_in_schedule_order(self):
        mixed = object.__new__(PreparedRawMixedSelection)
        log = []
        mixed.jobs = [dict(execute=lambda i=i: log.append(i)) for i in range(5)]
        mixed.output = object()
        self.assertIs(mixed(), mixed.output)
        self.assertEqual(log, list(range(5)))

    def populate(self, root):
        outcomes = serial_tests.SerialSelectionTests().populate(root)
        for plan in cases():
            name = plan["name"]
            path = root / (name + "-all-score.json")
            data = json.loads(path.read_text())
            checks = data["serial_harness"]
            checks["mixed_identity"] = dict(slots=2)
            checks["stage_ms"].update(serial_combined=1.2, online_topk=0.2)
            for phase in ("precheck", "postcheck"):
                checks[phase].update(
                    extra_candidates=["mixed"],
                    extra_score_trace_rows=plan["q_shape"][0],
                    extra_current_changed_index_sets=dict(mixed=3),
                )
            path.write_text(json.dumps(data))
            path = root / ("harness-" + name + ".json")
            data = json.loads(path.read_text())
            data[MIXED_OP_NAME] = data.pop(OP_NAME)
            data[MIXED_OP_NAME]["details"][0]["result"][0]["shape_detail"][
                1
            ] = "online512_slots2"
            path.write_text(json.dumps(data))
        return outcomes

    def test_complete_mean_and_missing_mixed_trace_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcomes = self.populate(root)
            result = summarize(root, outcomes, mixed=True)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["arithmetic_mean_speedup"], 1.5)
            self.assertAlmostEqual(result["arithmetic_mean_serial_over_mixed"], 1.2)
            self.assertEqual(summarize(root, outcomes)["status"], "incomplete")
            name = cases()[0]["name"]
            path = root / (name + "-all-score.json")
            data = json.loads(path.read_text())
            data["serial_harness"]["postcheck"]["extra_score_trace_rows"] -= 1
            path.write_text(json.dumps(data))
            result = summarize(root, outcomes, mixed=True)
            self.assertEqual(result["status"], "incomplete")
            self.assertNotIn("arithmetic_mean_speedup", result)


if __name__ == "__main__":
    unittest.main()
