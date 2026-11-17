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

"""Check runtime-bound kernels and fail-closed complete ten-case fixed-sort reports."""

import ast
import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import test_qsa_raw_mixed_selection as mixed_tests
from qsa_raw_tle_sort512_910b import harness_source, source
from run_qsa_raw_serial_harness_910b import (
    FIXED_OP_NAME,
    MIXED_OP_NAME,
    cases,
    summarize,
)


def functions(text):
    return {
        fn.name: fn for fn in ast.parse(text).body if isinstance(fn, ast.FunctionDef)
    }


class FixedHarnessTests(unittest.TestCase):
    def test_runtime_generic_preserves_proven_block512_and_fixed_primitive_network(
        self,
    ):
        fs = functions(harness_source())
        old = functions(source())["raw_tle_sort_generic"]
        new = fs["raw_tle_generic_prepared"]
        normal = copy.deepcopy(new)
        for n in ast.walk(normal):
            if isinstance(n, ast.For) and n.target.id in ("cube_block", "vector_block"):
                n.iter = ast.parse("range(triton.cdiv(WIDTH,512))", mode="eval").body
        self.assertEqual(
            ast.dump(ast.Module(body=normal.body, type_ignores=[])),
            ast.dump(ast.Module(body=old.body, type_ignores=[])),
        )
        for name in (
            "raw_tle_generic_prepared",
            "raw_tle_fixed_prepared",
            "raw_tle_generic_topk",
            "raw_tle_fixed_topk",
        ):
            fn = fs[name]
            self.assertIsNone(
                next(a.annotation for a in fn.args.args if a.arg == "BLOCKS")
            )
            calls = [
                n.args[0].value
                for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "tle.dsa.ascend.raw"
            ]
            if "fixed" in name:
                self.assertEqual(calls.count("sort32"), 1)
                self.assertEqual(calls.count("mrgsort"), 2)
                self.assertNotIn("sort_1d_pack", calls)
            else:
                self.assertEqual(calls.count("sort_1d_pack"), 1)
            self.assertEqual(calls.count("unpack_sort"), 1)
            cpu = copy.deepcopy(fn)
            cpu.decorator_list = []
            cpu.body = [ast.Pass()]
            for arg in cpu.args.args:
                arg.annotation = None
            env = {}
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[cpu], type_ignores=[])),
                    "sig",
                    "exec",
                ),
                env,
            )
            kwargs = dict(ROWS=17, WIDTH=8256, BLOCKS=17)
            count = 4 if name.endswith("topk") else 9
            if count == 9:
                kwargs.update(SLOTS=2, TRACE=False)
            inspect.signature(env[name]).bind(
                *[object() for _ in range(count)], **kwargs
            )

    def populate(self, root):
        outcomes = mixed_tests.MixedSelectionTests().populate(root)
        for plan in cases():
            name = plan["name"]
            path = root / (name + "-all-score.json")
            data = json.loads(path.read_text())
            checks = data["serial_harness"]
            checks["mixed_identity"].update(
                kernel="raw_tle_fixed_prepared",
                native="block512_query16_cube",
                local_sort="sort32_mrgsort_fixed512",
            )
            checks["generic_identity"] = dict(
                local_sort="sort_1d_pack",
                native="block512_query16_cube",
                same_packing_and_schedule=True,
            )
            checks["generic_routes"] = checks["mixed_routes"] = [
                dict(first=0, count=17, maximum=8256)
            ]
            checks["stage_ms"].update(generic_combined=1.04, generic_online_topk=0.25)
            for key, calls in (
                ("mixed_compiled_artifacts", dict(sort_1d_pack=0, sort32=1, mrgsort=2)),
                (
                    "generic_compiled_artifacts",
                    dict(sort_1d_pack=1, sort32=0, mrgsort=0),
                ),
            ):
                checks[key] = [dict(primitive_ir=dict(calls=calls))]
            for phase in ("precheck", "postcheck"):
                checks[phase]["extra_candidates"].append("generic")
            path.write_text(json.dumps(data))
            path = root / ("harness-" + name + ".json")
            data = json.loads(path.read_text())
            data[FIXED_OP_NAME] = data.pop(MIXED_OP_NAME)
            data[FIXED_OP_NAME]["details"][0]["result"][0]["shape_detail"][
                1
            ] = "tle_sort512_slots2"
            path.write_text(json.dumps(data))
        return outcomes

    def test_mean_requires_all_ten_and_correct_control_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcomes = self.populate(root)
            summary = summarize(root, outcomes, mixed=True, fixed=True)
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["arithmetic_mean_speedup"], 1.5)
            self.assertAlmostEqual(summary["arithmetic_mean_generic_over_fixed"], 1.04)
            self.assertEqual(
                summarize(root, outcomes, mixed=True)["status"], "incomplete"
            )
            path = root / (cases()[0]["name"] + "-all-score.json")
            data = json.loads(path.read_text())
            data["serial_harness"]["postcheck"]["extra_candidates"].remove("generic")
            path.write_text(json.dumps(data))
            failed = summarize(root, outcomes, mixed=True, fixed=True)
            self.assertEqual(failed["status"], "incomplete")
            self.assertNotIn("arithmetic_mean_speedup", failed)

    def test_rejects_changed_control_score_chain_or_missing_ir(self):
        for field in ("native", "ir"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                outcomes = self.populate(root)
                path = root / (cases()[0]["name"] + "-all-score.json")
                data = json.loads(path.read_text())
                checks = data["serial_harness"]
                if field == "native":
                    checks["generic_identity"]["native"] = "old_eight_call_chain"
                else:
                    checks["mixed_compiled_artifacts"] = []
                path.write_text(json.dumps(data))
                self.assertEqual(
                    summarize(root, outcomes, mixed=True, fixed=True)["status"],
                    "incomplete",
                )


if __name__ == "__main__":
    unittest.main()
