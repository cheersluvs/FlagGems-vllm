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

"""Check opaque Cube ownership, ring addressing and unchanged UB merge helper."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from qsa_raw_mixed_online_910b import geometry, source


class Index(int):
    def to(self, *args):
        return self

    def __add__(self, other):
        return Index(int(self) + int(other))

    def __mul__(self, other):
        return Index(int(self) * int(other))


class RawMixedTests(unittest.TestCase):
    def tree(self):
        return ast.parse(source())

    def test_frozen_ub_snapshot_helper_is_retained_exactly(self):
        original = ast.parse(
            Path(__file__)
            .with_name("probe_qsa_int8_mixed_pipeline_910b.py")
            .read_text()
        )
        name = "_update_online_topk_ub_snapshot"
        old = next(
            n
            for n in original.body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        new = next(
            n
            for n in self.tree().body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        self.assertEqual(ast.dump(old), ast.dump(new))

    def test_cube_has_only_scalar_addresses_custom_call_and_handshakes(self):
        fn = next(
            n
            for n in self.tree().body
            if isinstance(n, ast.FunctionDef) and n.name == "raw_mixed_online"
        )
        cube = fn.body[0]
        self.assertIsInstance(cube, ast.With)
        calls = [ast.unparse(n.func) for n in ast.walk(cube) if isinstance(n, ast.Call)]
        self.assertNotIn("tl.load", calls)
        self.assertNotIn("tl.dot", calls)
        self.assertNotIn("tle.dsa.alloc", calls)
        self.assertEqual(calls.count("raw.call"), 1)
        self.assertEqual(calls.count("tl.cast"), 4)
        self.assertEqual(calls.count("al.sync_block_wait"), 2)
        self.assertEqual(calls.count("al.sync_block_set"), 1)
        self.assertIn("SLOTS", [a.arg for a in fn.args.args])

    def test_actual_raw_pointer_expressions_cover_disjoint_ring_tiles(self):
        fn = next(
            n
            for n in self.tree().body
            if isinstance(n, ast.FunctionDef) and n.name == "raw_mixed_online"
        )
        assigns = {
            n.targets[0].id: n.value
            for n in ast.walk(fn.body[0])
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        }
        for width in (512, 1024, 4096):
            for slots in (1, 2):
                g = geometry(width, slots)
                for block in range(g["steps"]):
                    destinations = []
                    for inner in range(8):
                        scope = dict(
                            block=Index(block),
                            inner=Index(inner),
                            slot=Index(block % slots),
                            kbase=0,
                            rbase=0,
                            tl=SimpleNamespace(uint64=None),
                        )
                        key = eval(
                            compile(ast.Expression(assigns["key"]), "key", "eval"),
                            {},
                            scope,
                        )
                        dest = eval(
                            compile(ast.Expression(assigns["dest"]), "dest", "eval"),
                            {},
                            scope,
                        )
                        self.assertEqual(key, (block * 8 + inner) * 8192)
                        self.assertEqual(dest, ((block % slots) * 8 + inner) * 4096)
                        self.assertLessEqual(dest + 4096, g["ring_elements"] * 4)
                        destinations.append(dest)
                    self.assertEqual(len(set(destinations)), 8)
                # A slot cannot be reused until consumer releases it.
                state = ["free"] * slots
                for first in range(0, g["steps"], slots):
                    active = min(slots, g["steps"] - first)
                    for i in range(active):
                        slot = (first + i) % slots
                        self.assertEqual(state[slot], "free")
                        state[slot] = "ready"
                    for i in range(active):
                        slot = (first + i) % slots
                        self.assertEqual(state[slot], "ready")
                        state[slot] = "free"
                self.assertEqual(state, ["free"] * slots)

    def test_geometry_rejects_unsupported_updates_and_slot_counts(self):
        for width, slots in ((0, 1), (513, 1), (1024, 0), (1024, 3)):
            with self.assertRaises(ValueError):
                geometry(width, slots)
        self.assertEqual(
            {sub * 8 + query for sub in (0, 1) for query in range(8)}, set(range(16))
        )


if __name__ == "__main__":
    unittest.main()
