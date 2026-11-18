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

"""Check that the single-block probe preserves scheduling and output masks."""

import ast
import unittest
from types import SimpleNamespace

from qsa_raw_history_merge_910b import source as history_source
from qsa_single_block_probe_910b import ir_evidence, source


class SingleBlockTests(unittest.TestCase):
    def test_single_block_uses_current_query_and_preserves_handshake(self):
        control = next(
            fn
            for fn in ast.parse(history_source()).body
            if isinstance(fn, ast.FunctionDef) and fn.name == "raw_history_snapshot"
        )
        candidate = ast.parse(source()).body[0]
        self.assertEqual(
            ast.unparse(candidate.body[0]), "tl.static_assert(WIDTH == 512)"
        )
        self.assertEqual(ast.dump(candidate.body[1]), ast.dump(control.body[0]))

        def sync_calls(fn):
            return [
                ast.dump(node)
                for node in ast.walk(fn)
                if isinstance(node, ast.Call)
                and ast.unparse(node.func).startswith("al.sync_block_")
            ]

        self.assertEqual(sync_calls(candidate), sync_calls(control))
        control_stores = [
            ast.unparse(node).replace("final_row", "update_row")
            for node in ast.walk(control)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "tl.store"
            and isinstance(node.args[0], ast.BinOp)
        ]
        candidate_stores = [
            ast.unparse(node)
            for node in ast.walk(candidate)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "tl.store"
        ]
        self.assertEqual(set(candidate_stores), set(control_stores))
        generated = ast.unparse(candidate)
        self.assertIn("raw('unpack_sort', pairs, 512", generated)
        self.assertIn(
            "valid = rank < tl.load(Visible + tl.where(update_row < ROWS", generated
        )
        self.assertNotIn("state_storage", generated)

    def test_ir_rejects_retained_merge_or_history(self):
        text = "\n".join(
            'hivm.hir.custom "' + name + '"'
            for name in ("sort32", "mrgsort", "mrgsort", "unpack_sort")
        )
        self.assertFalse(
            ir_evidence(SimpleNamespace(asm={"bcmlir": text}))["history_ub"]
        )
        for unexpected in (
            'hivm.hir.custom "merge_exhaust_sort4"',
            "memref<8192xf32, #hivm.address_space<ub>>",
        ):
            with self.assertRaises(RuntimeError):
                ir_evidence(SimpleNamespace(asm={"bcmlir": text + "\n" + unexpected}))


if __name__ == "__main__":
    unittest.main()
