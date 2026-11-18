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

"""Verify output capacity and reject a temporary destination or retained copy."""

import ast
import unittest
from types import SimpleNamespace

from qsa_history_direct_probe_910b import ir_evidence, source
from qsa_raw_history_merge_910b import source as control_source


def fixture(direct):
    size = 16384 if direct else 8192
    destination = "%old" if direct else "%temp"
    update_copy = (
        ""
        if direct
        else "hivm.hir.copy ins(%head : memref<1024xf32>) outs(%slot : memref<1024xf32>)"
    )
    return f"""
%state = memref.alloc() : memref<{size}xf32, #hivm.address_space<ub>>
%init = memref.subview %state[%i] [1024] [1]
hivm.hir.copy ins(%value : memref<1024xf32>) outs(%init : memref<1024xf32>)
%slot = memref.subview %state[%j] [2048] [1]
%old = bufferization.to_tensor %slot restrict writable
%previous = tensor.extract_slice %old[0] [1024] [1]
%p0 = tensor.insert_slice %previous into %zeros[0] [1024] [1]
%p1 = tensor.insert_slice %pairs into %p0[1024] [1024] [1]
%out:2 = hivm.hir.custom "merge_exhaust_sort4" ins(%p1 : tensor<4096xf32>) outs({destination} : tensor<2048xf32>)
%head = tensor.extract_slice %out#0[0] [1024] [1]
{update_copy}
%last = memref.subview %state[%k] [1024] [1]
%last_tensor = bufferization.to_tensor %last restrict writable
hivm.hir.custom "unpack_sort" ins(%last_tensor : tensor<1024xf32>)
hivm.hir.custom "sort32"
hivm.hir.custom "mrgsort"
hivm.hir.custom "mrgsort"
"""


def proof(ir, direct):
    return ir_evidence(SimpleNamespace(asm=dict(bcmlir=ir)), direct)


class DirectHistoryTests(unittest.TestCase):
    def test_destination_has_full_capacity_and_final_read_uses_its_prefix(self):
        fn = ast.parse(source()).body[0]
        original = next(
            n
            for n in ast.parse(control_source()).body
            if isinstance(n, ast.FunctionDef) and n.name == "raw_history_prefix"
        )
        self.assertEqual(ast.dump(fn.body[0]), ast.dump(original.body[0]))
        vec = ast.unparse(fn.body[1])
        self.assertIn("subview(state_storage, [update_query * 2048], [2048], [1])", vec)
        self.assertIn("subview(state_storage, [final_query * 2048], [1024], [1])", vec)
        self.assertIn("insert_slice(packed, old_prefix, (0,), (1024,), (1,))", vec)
        self.assertIn("out=[destination, tl.zeros((4,), tl.int32)]", vec)
        self.assertEqual(vec.count("bind_buffer="), 1)

    def test_ir_distinguishes_initialization_and_history_update_copies(self):
        self.assertEqual(len(proof(fixture(False), False)["copies_into_history"]), 2)
        direct = fixture(True)
        self.assertEqual(len(proof(direct, True)["copies_into_history"]), 1)
        self.assertTrue(proof(direct, True)["direct_history_destination"])

        with self.assertRaisesRegex(RuntimeError, "binding"):
            proof(direct.replace("outs(%old", "outs(%temporary"), True)
        with self.assertRaisesRegex(RuntimeError, "copies into history"):
            proof(
                direct
                + "hivm.hir.copy ins(%head : memref<1024xf32>) outs(%slot : memref<1024xf32>)",
                True,
            )
        with self.assertRaisesRegex(RuntimeError, "snapshot old history"):
            proof(
                direct.replace("%previous into %zeros", "%unrelated into %zeros"), True
            )


if __name__ == "__main__":
    unittest.main()
