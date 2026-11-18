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

"""Check the peeled schedule and the merge evidence required before timing."""

import ast
import copy
import unittest
from types import SimpleNamespace

from qsa_first_block_seed_probe_910b import ir_evidence, source
from qsa_raw_history_merge_910b import source as history_source


class FirstBlockSeedTests(unittest.TestCase):
    def test_only_first_vector_block_seeds_history(self):
        control = next(
            node
            for node in ast.parse(history_source()).body
            if isinstance(node, ast.FunctionDef) and node.name == "raw_history_snapshot"
        )
        candidate = ast.parse(source()).body[0]
        self.assertEqual(ast.dump(candidate.body[1]), ast.dump(control.body[0]))
        base_vector = control.body[1].body
        candidate_vector = candidate.body[2].body
        self.assertEqual(ast.dump(candidate_vector[-1]), ast.dump(base_vector[-1]))
        base_loop = next(
            n
            for n in base_vector
            if isinstance(n, ast.For) and n.target.id == "vector_block"
        )
        tail = next(
            n
            for n in candidate_vector
            if isinstance(n, ast.For) and n.target.id == "vector_block"
        )
        self.assertEqual(ast.unparse(tail.iter), "range(1, triton.cdiv(WIDTH, 512))")
        self.assertEqual(
            [ast.dump(n) for n in tail.body], [ast.dump(n) for n in base_loop.body]
        )

        seed_start = candidate_vector.index(tail) - len(base_loop.body)
        self.assertEqual(
            ast.unparse(candidate_vector[seed_start - 1]),
            "vector_block = tl.full((), 0, tl.int32)",
        )
        first = candidate_vector[seed_start : candidate_vector.index(tail)]
        first_row = next(n for n in first if isinstance(n, ast.For))
        first_other = [ast.dump(n) for n in first if n is not first_row]
        self.assertEqual(
            first_other,
            [ast.dump(n) for n in base_loop.body if not isinstance(n, ast.For)],
        )
        control_row = copy.deepcopy(
            next(n for n in base_loop.body if isinstance(n, ast.For))
        )
        load = next(
            i
            for i, n in enumerate(control_row.body)
            if isinstance(n, ast.Assign) and n.targets[0].id == "update_state"
        )
        control_row.body[load + 2].value.args[0] = ast.Name(id="pairs", ctx=ast.Load())
        del control_row.body[load : load + 2]
        self.assertEqual(ast.dump(first_row), ast.dump(control_row))
        self.assertNotIn("init_query", ast.unparse(candidate))

    def test_ir_rejects_extra_first_block_merge(self):
        names = ["sort32", "mrgsort", "mrgsort"] * 2 + [
            "merge_exhaust_sort4",
            "unpack_sort",
        ]
        text = "\n".join(f'hivm.hir.custom "{name}"' for name in names)
        self.assertTrue(
            ir_evidence(SimpleNamespace(asm={"bcmlir": text}))[
                "retained_snapshot_merge"
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "Unexpected first-block"):
            ir_evidence(
                SimpleNamespace(
                    asm={"bcmlir": text + '\nhivm.hir.custom "merge_exhaust_sort4"'}
                )
            )


if __name__ == "__main__":
    unittest.main()
