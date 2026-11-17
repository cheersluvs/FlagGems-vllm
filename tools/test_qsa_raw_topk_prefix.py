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

"""Check prefix boundaries, fixed history and complete positional binding."""

import ast
import copy
import inspect
import unittest

from qsa_raw_critical_path_910b import source as critical_source
from qsa_raw_topk_prefix_910b import source


def functions(text):
    return {f.name: f for f in ast.parse(text).body if isinstance(f, ast.FunctionDef)}


def events(fn):
    return [
        ast.unparse(n)
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and ast.unparse(n.func).startswith("al.sync_block_")
    ]


class PrefixTests(unittest.TestCase):
    def test_full_body_cube_and_events_are_frozen(self):
        fs = functions(source())
        old = functions(critical_source())["raw_critical_full"]
        full = fs["raw_topk_full"]
        self.assertEqual(
            ast.dump(ast.Module(body=full.body, type_ignores=[])),
            ast.dump(ast.Module(body=old.body, type_ignores=[])),
        )
        for f in fs.values():
            self.assertEqual(ast.dump(f.body[0]), ast.dump(full.body[0]))
            self.assertEqual(events(f), events(full))

    def test_actual_signatures_bind_twelve_pointers_and_four_configs(self):
        payload = [object() for _ in range(12)]
        for original in functions(source()).values():
            f = copy.deepcopy(original)
            f.decorator_list = []
            f.body = [ast.Pass()]
            for a in f.args.args:
                a.annotation = None
            env = {}
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[f], type_ignores=[])),
                    "prefix-signature",
                    "exec",
                ),
                env,
            )
            bound = inspect.signature(env[f.name]).bind(
                *payload, WIDTH=8256, ROWS=17, SLOTS=2, TRACE=False
            )
            for pos, name in ((9, "Replay"), (10, "Audit"), (11, "Seed")):
                self.assertIs(bound.arguments[name], payload[pos])

    def test_sort_stops_before_history_and_merge_uses_seed_without_commit(self):
        fs = functions(source())
        sort = ast.unparse(fs["raw_topk_sort"].body[1])
        merge = ast.unparse(fs["raw_topk_merge"].body[1])
        self.assertIn("sort_1d_pack", sort)
        for removed in (
            "state_storage",
            "_update_online_topk_ub_snapshot",
            "unpack_sort",
        ):
            self.assertNotIn(removed, sort)
        self.assertIn("_update_online_topk_ub_snapshot", merge)
        self.assertIn("tl.load(Seed", merge)
        self.assertNotIn("unpack_sort", merge)
        self.assertNotIn("bind_buffer=update_state_slot", merge)
        self.assertIn("bind_buffer=init_state_slot", merge)
        for name in ("raw_topk_sort", "raw_topk_merge"):
            fn = fs[name]
            audit = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "tl.store"
                and "Audit" in ast.unparse(n.args[0])
            ]
            self.assertEqual(len(audit), 1)
            for branch in [n for n in ast.walk(fn) if isinstance(n, ast.If)]:
                self.assertNotIn("Audit", ast.unparse(branch))
            self.assertIn("prefix_word // 2 + 1", ast.unparse(fn))


if __name__ == "__main__":
    unittest.main()
