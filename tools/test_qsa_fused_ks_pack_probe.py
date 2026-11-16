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

"""Verify logical scale order, padding, K-leaf stability and output paths."""

import ast
import unittest
from unittest.mock import patch

import test_qsa_finish_blocked_probe as blocked_test
from qsa_finish_prefix_probe_910b import ir_liveness
from qsa_fused_ks_pack_probe_910b import source
from qsa_k_pack_2d_probe_910b import source as key_source
from test_qsa_finish_prefix_probe import Pointer, ScalarTL


class FusedKSTests(unittest.TestCase):
    def test_fused_scale_store_order_padding_and_single_writer(self):
        maximum, tiles, pages = 2049, 33, 11
        values = [float(i + 1) / 4096 for i in range(pages * 192)]
        table = list(reversed(range(pages)))
        tl = ScalarTL(dict(KS=values, Table=table, LogicalKS=[None] * (tiles * 64)))
        with patch.object(blocked_test, "source", source):
            ns = blocked_test.functions(tl)
        ns["raw_k_2d_part"] = lambda *unused: None
        for pid in range(20):
            for c in range(64):
                tl.pid = pid
                tl.column = c
                ns["raw_pack_k_ks"](
                    None,
                    Pointer("KS"),
                    Pointer("Table"),
                    None,
                    Pointer("LogicalKS"),
                    maximum,
                    tiles,
                    20,
                )
        self.assertEqual(set(tl.writes), set(range(tiles * 64)))
        for col in range(tiles * 64):
            expected = (
                values[table[col // 192] * 192 + col % 192] if col < maximum else 0
            )
            self.assertEqual(tl.writes[col], expected)

    def test_k_leaf_is_unchanged_and_fast_body_has_no_page_reads(self):
        def fns(s):
            return {
                n.name: n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef)
            }

        old, new = fns(key_source()), fns(source())
        self.assertEqual(ast.dump(old["raw_k_2d_part"]), ast.dump(new["raw_k_2d_part"]))
        body = new["raw_logical_body"]
        names = {n.id for n in ast.walk(body) if isinstance(n, ast.Name)}
        self.assertFalse(names & {"Table", "KS", "Visible"})
        calls = [
            n
            for n in ast.walk(body)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.load"
        ]
        self.assertEqual(len(calls), 2)
        self.assertTrue(
            any(ast.unparse(n.args[0]).startswith("LogicalKS +") for n in calls)
        )

    def test_actual_output_uses_packed_full_block_and_physical_tail(self):
        rows, width, tiles, pages = 17, 2057, 33, 11
        visible = ([0, 1, 1023, 1024, 1025, 2048, 2049] * 3)[:rows]
        ks = [float(i + 1) / 4096 for i in range(pages * 192)]
        table = list(reversed(range(pages)))
        logical = [
            ks[table[c // 192] * 192 + c % 192] if c < 2049 else 0
            for c in range(tiles * 64)
        ]
        buffers = dict(
            Raw=[float(i % 257 - 128) / 16 for i in range(2 * tiles * 1024)],
            KS=ks,
            LogicalKS=logical,
            Table=table,
            Visible=visible,
            Out=[None] * (rows * width),
        )
        tl = blocked_test.TL(buffers)
        with patch.object(blocked_test, "source", source):
            ns = blocked_test.functions(tl)
        for pid in range(20):
            for tile in range(16):
                for c in range(64):
                    tl.pid = pid
                    tl.tile = tile
                    tl.column = c
                    ns["raw_finish_logical1024"](
                        *(
                            Pointer(k)
                            for k in (
                                "Raw",
                                "KS",
                                "Table",
                                "Visible",
                                "Out",
                                "LogicalKS",
                            )
                        ),
                        rows,
                        width,
                        tiles,
                        20,
                    )
        self.assertEqual(set(tl.writes), set(range(rows * width)))
        for row in range(rows):
            for col in range(width):
                expected = -float("inf")
                if col < visible[row]:
                    expected = (
                        buffers["Raw"][
                            ((row // 16) * tiles + col // 64) * 1024
                            + row % 16 * 64
                            + col % 64
                        ]
                        * logical[col]
                    )
                self.assertEqual(tl.writes[row * width + col], expected)

    def test_liveness_requires_new_buffer_read(self):
        inputs = ["Raw", "KS", "Table", "Visible", "LogicalKS", "Out"]
        text = (
            "func.func @test("
            + ",".join(
                f'%arg{i}: memref<?xf32> loc("{name}")' for i, name in enumerate(inputs)
            )
            + ") {\n"
        )
        text += "\n".join(f"%x{i} = memref.load %arg{i}[0]" for i in range(5))
        text += "\nbufferization.materialize_in_destination %value in writable %arg5"
        self.assertTrue(ir_liveness(text, "logicalfull")["passed"])
        with self.assertRaises(RuntimeError):
            ir_liveness(text.replace("%x4 = memref.load %arg4[0]", ""), "logicalfull")


if __name__ == "__main__":
    unittest.main()
