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

"""Check grouped raw addresses, exact-table padding and row/column masks."""

import ast
import unittest
from types import SimpleNamespace

from qsa_raw_mixed_online_910b import (
    boundary_cases,
    boundary_geometry,
    boundary_source,
    padded_pack_source,
)
from test_qsa_raw_mixed_online import Index


class BoundaryTests(unittest.TestCase):
    def function(self):
        return next(
            n
            for n in ast.parse(boundary_source()).body
            if isinstance(n, ast.FunctionDef) and n.name == "raw_mixed_boundaries"
        )

    def test_plans_cover_tail_groups_and_real_column_widths(self):
        self.assertEqual(len(boundary_cases()), 6)
        for rows, width in boundary_cases():
            for slots in (1, 2):
                g = boundary_geometry(rows, width, slots)
                self.assertGreaterEqual(g["padded_rows"], rows)
                self.assertLess(g["padded_rows"] - rows, 16)
                self.assertGreaterEqual(g["padded_width"], width)
                self.assertLess(g["padded_width"] - width, 512)
                self.assertEqual(g["ring_elements"], g["groups"] * slots * 8192)
        for rows, width, slots in (
            (0, 1088, 1),
            (321, 1088, 1),
            (17, 511, 1),
            (17, 1088, 3),
        ):
            with self.assertRaises(ValueError):
                boundary_geometry(rows, width, slots)

    def test_padded_table_load_never_addresses_outside_exact_table(self):
        tree = ast.parse(padded_pack_source())
        loads = [
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "physical"
        ]
        self.assertEqual(len(loads), 2)
        for _, width in boundary_cases():
            g = boundary_geometry(1, width, 1)
            for tile in range(g["tiles"]):

                def load(index, mask=True, other=0):
                    self.assertGreaterEqual(index, 0)
                    self.assertLess(index, g["table_columns"])
                    return 100 + index if mask else other

                tl = SimpleNamespace(
                    load=load, where=lambda cond, yes, no: yes if cond else no
                )
                for expr in loads:
                    actual = eval(
                        compile(ast.Expression(expr), "table", "eval"),
                        {},
                        dict(tl=tl, Table=0, tile=tile, MAX_VISIBLE=width),
                    )
                    self.assertEqual(
                        actual, 100 + tile // 3 if tile * 64 < width else 0
                    )

    def test_actual_group_offsets_do_not_alias_query_or_ring_groups(self):
        fn = self.function()
        assigns = {
            n.targets[0].id: n.value
            for n in ast.walk(fn.body[0])
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        }

        def evaluate(name, scope):
            return eval(compile(ast.Expression(assigns[name]), name, "eval"), {}, scope)

        for rows, width in boundary_cases():
            for slots in (1, 2):
                g = boundary_geometry(rows, width, slots)
                for group in range(g["groups"]):
                    scope = dict(
                        qbase=0,
                        kbase=0,
                        wbase=0,
                        rbase=0,
                        SLOTS=slots,
                        tl=SimpleNamespace(
                            program_id=lambda _: Index(group), uint64=None
                        ),
                    )
                    self.assertEqual(evaluate("group_q", scope), group * 8192)
                    self.assertEqual(evaluate("group_w", scope), group * 2048)
                    for block in (0, g["steps"] - 1):
                        for inner in (0, 7):
                            scope.update(
                                cube_block=Index(block),
                                cube_inner=Index(inner),
                                cube_slot=Index(block % slots),
                            )
                            dest = evaluate("dest", scope)
                            self.assertEqual(
                                dest,
                                ((group * slots + block % slots) * 8 + inner) * 4096,
                            )
                            self.assertLessEqual(dest + 4096, g["ring_elements"] * 4)
                            self.assertEqual(
                                evaluate("key", scope), (block * 8 + inner) * 8192
                            )

    def test_actual_output_masks_cover_only_logical_rows_and_columns(self):
        fn = self.function()
        stores = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.store"
        ]
        self.assertEqual(len(stores), 3)
        for rows, width in boundary_cases():
            g = boundary_geometry(rows, width, 1)
            for row in range(g["padded_rows"]):
                for col in (0, width - 1, width, g["padded_width"] - 1):
                    scope = dict(
                        ROWS=rows,
                        WIDTH=width,
                        update_row=row,
                        final_row=row,
                        vector_block=col // 512,
                        lane=col % 512,
                    )
                    for call in stores:
                        mask = next(k.value for k in call.keywords if k.arg == "mask")
                        actual = eval(
                            compile(ast.Expression(mask), "mask", "eval"), {}, scope
                        )
                        is_trace = "Trace" in ast.unparse(call.args[0])
                        self.assertEqual(
                            bool(actual),
                            row < rows and (col < width if is_trace else True),
                        )
        # Initial/update/final phases retain disjoint induction variables.
        names = [n.target.id for n in ast.walk(fn) if isinstance(n, ast.For)]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
