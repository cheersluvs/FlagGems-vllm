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

"""Check leaf bounds, balanced order and single-store coverage before card runs."""

import ast
import copy
import unittest

from qsa_finish_merge_probe_910b import WIDTHS, source, store_evidence
from test_qsa_finish_prefix_probe import Pointer, ScalarTL, fixture


def functions(tl):
    nodes = []
    for original in ast.parse(source()).body:
        if not isinstance(original, ast.FunctionDef):
            continue
        node = copy.deepcopy(original)
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
        nodes.append(node)
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = dict(
        tl=tl,
        triton=type(
            "Triton", (), dict(cdiv=staticmethod(lambda a, b: (a + b - 1) // b))
        ),
    )
    exec(compile(tree, "generated_finish_merge", "exec"), namespace)
    return namespace


class BoundedTL(ScalarTL):
    def load(self, pointer, mask=True, other=0):
        # Scalar masked loads may speculate in lowering; even inactive bases
        # must be valid, unlike a check that only tests active dereferences.
        if not 0 <= pointer.offset < len(self.buffers[pointer.name]):
            raise AssertionError(f"Unsafe pointer: {pointer.name}/{pointer.offset}")
        return super().load(pointer, mask, other)


class FinishMergeTests(unittest.TestCase):
    def test_leaf_preserves_score_and_inactive_pointer_bounds(self):
        rows, width, tiles, _, buffers = fixture()
        tl = BoundedTL(buffers)
        fn = functions(tl)["raw_finish_join1"]
        for row in range(rows):
            visible = buffers["Visible"][row]
            for tile in range(32):
                for c in range(64):
                    tl.column = c
                    value = fn(
                        *(Pointer(k) for k in ("Raw", "KS", "Table")),
                        row,
                        tile,
                        visible,
                        width,
                        tiles,
                    )
                    col = tile * 64 + c
                    expected = -float("inf")
                    if col < min(width, visible):
                        raw = buffers["Raw"][
                            ((row // 16) * tiles + tile) * 1024 + row % 16 * 64 + c
                        ]
                        ks = buffers["KS"][
                            buffers["Table"][tile // 3] * 192 + tile % 3 * 64 + c
                        ]
                        expected = raw * ks
                    self.assertEqual(value, expected, (row, tile, c))

    def test_leaf_crosses_physical_page_boundary_without_reordering(self):
        width, tiles = 577, 9
        buffers = dict(
            Raw=[float(i % 251 - 125) for i in range(tiles * 1024)],
            KS=[float(i + 1) / 1024 for i in range(4 * 192)],
            Table=[2, 0, 3, 1],
            Visible=[513],
            Out=[0] * width,
        )
        tl = BoundedTL(buffers)
        fn = functions(tl)["raw_finish_join1"]
        for tile in range(16):
            for c in range(64):
                tl.column = c
                col = tile * 64 + c
                actual = fn(
                    *(Pointer(k) for k in ("Raw", "KS", "Table")),
                    0,
                    tile,
                    513,
                    width,
                    tiles,
                )
                expected = -float("inf")
                if col < 513:
                    expected = (
                        buffers["Raw"][tile * 1024 + c]
                        * buffers["KS"][buffers["Table"][col // 192] * 192 + col % 192]
                    )
                self.assertEqual(actual, expected)

    def test_balanced_concatenation_preserves_all_key_positions(self):
        class TL:
            @staticmethod
            def cat(left, right, can_reorder):
                if can_reorder is not False:
                    raise AssertionError("Reordering disallowed")
                return left + right

        ns = functions(TL)
        leaves = []

        def leaf(raw, ks, table, row, tile, visible, width, tiles):
            leaves.append(tile)
            return list(range(tile * 64, (tile + 1) * 64))

        ns["raw_finish_join1"] = leaf
        for width in WIDTHS:
            count = width // 64
            for first in (0, 3, 16):
                leaves.clear()
                actual = ns[f"raw_finish_join{count}"](
                    None, None, None, 0, first, 0, 1, 1
                )
                self.assertEqual(actual, list(range(first * 64, first * 64 + width)))
                self.assertEqual(leaves, list(range(first, first + count)))

    def test_larger_store_loops_cover_each_output_once(self):
        for block in WIDTHS:
            for rows, width in ((1, 1), (3, 65), (17, 257), (3, 1088)):
                tl = ScalarTL(dict(Visible=[65] * rows, Out=[None] * (rows * width)))
                ns = functions(tl)
                ns[f"raw_finish_join{block//64}"] = lambda *unused: 0.0
                fn = ns[f"raw_finish_merge{block}"]
                cores = min(20, rows * ((width + block - 1) // block))
                for pid in range(cores):
                    for c in range(block):
                        tl.pid = pid
                        tl.column = c
                        fn(
                            None,
                            None,
                            None,
                            Pointer("Visible"),
                            Pointer("Out"),
                            rows,
                            width,
                            1,
                            cores,
                        )
                self.assertEqual(set(tl.writes), set(range(rows * width)))

    def test_store_audit_follows_only_output_aliases(self):
        text = """func.func @test(%arg0: memref<?xf32> loc("Raw"), %arg1: memref<?xf32> loc("Out")) {
%r = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64], strides: [1]
%o = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256], strides: [1]
%s = memref.subview %o[0] [256] [1]
bufferization.materialize_in_destination %x in writable %s
}"""
        result = store_evidence(text)
        self.assertEqual(len(result["output_views"]), 2)
        self.assertEqual(len(result["output_write_sites"]), 1)
        self.assertIn("sizes: [256]", result["output_views"][0])
        self.assertNotIn("sizes: [64]", str(result))

    def test_source_has_only_regular_leaf_reads_and_one_store_per_wrapper(self):
        tree = ast.parse(source())
        fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        leaf = fns["raw_finish_join1"]
        loads = [
            n
            for n in ast.walk(leaf)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.load"
        ]
        self.assertEqual(len(loads), 3)
        self.assertFalse(
            any(
                isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.store"
                for n in ast.walk(leaf)
            )
        )
        for width in WIDTHS:
            wrapper = fns[f"raw_finish_merge{width}"]
            stores = [
                n
                for n in ast.walk(wrapper)
                if isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.store"
            ]
            self.assertEqual(len(stores), 1)
            self.assertEqual(
                len(
                    [
                        n
                        for n in ast.walk(wrapper)
                        if isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.load"
                    ]
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
