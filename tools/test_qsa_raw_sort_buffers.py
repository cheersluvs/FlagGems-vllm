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

"""Verify buffer-only mutation, poisoning coverage and retained sort evidence."""

import ast
import copy
import inspect
import unittest
from types import SimpleNamespace

from qsa_raw_cube_block512_910b import source as full_source
from qsa_raw_sort_buffers_910b import sort_ir_evidence, source


def functions(text):
    return {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}


def sort_call(fn):
    return next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "tle.dsa.ascend.raw"
        and n.args[0].value == "sort_1d_pack"
    )


class SortBufferTests(unittest.TestCase):
    def test_only_buffer_operands_change_and_cube_events_stay_identical(self):
        fs = functions(source())
        original = functions(full_source())["raw_mixed_block512"]
        zero = fs["raw_sort_zero"]
        self.assertEqual(
            ast.dump(ast.Module(body=zero.body, type_ignores=[])),
            ast.dump(ast.Module(body=original.body, type_ignores=[])),
        )
        base = sort_call(zero)
        for fn in fs.values():
            self.assertEqual(ast.dump(fn.body[0]), ast.dump(zero.body[0]))
            call = sort_call(fn)
            for i in (0, 1, 3, 4, 5, 6):
                self.assertEqual(ast.dump(call.args[i]), ast.dump(base.args[i]))
        uninit = fs["raw_sort_uninit"]
        allocs = uninit.body[1].body[:2]
        self.assertEqual(
            [n.targets[0].id for n in allocs], ["sort_scratch", "sort_output"]
        )
        self.assertEqual([n.value.args[0].elts[0].value for n in allocs], [2048, 1024])
        call = sort_call(uninit)
        self.assertEqual(ast.unparse(call.args[2]), "tle.dsa.to_tensor(sort_scratch)")
        self.assertEqual(
            ast.unparse(next(k.value for k in call.keywords if k.arg == "out")),
            "tle.dsa.to_tensor(sort_output)",
        )

    def test_poison_is_inside_every_row_update_before_the_sort(self):
        poison = functions(source())["raw_sort_poison"]
        update = next(
            n
            for n in ast.walk(poison)
            if isinstance(n, ast.For) and n.target.id == "update_query"
        )
        pair = next(
            i
            for i, n in enumerate(update.body)
            if isinstance(n, ast.Assign) and n.targets[0].id == "pairs"
        )
        before = update.body[pair - 3 : pair]
        self.assertEqual(len(before), 3)
        text = "\n".join(ast.unparse(n) for n in before)
        for term in (
            "float('nan')",
            "bind_buffer=sort_scratch",
            "bind_buffer=sort_output",
            "tl.debug_barrier()",
        ):
            self.assertIn(term, text)
        uninit = ast.unparse(functions(source())["raw_sort_uninit"])
        self.assertNotIn("bind_buffer=sort_scratch", uninit)
        self.assertNotIn("bind_buffer=sort_output", uninit)

    def test_actual_signature_binds_nine_payloads_without_config_collision(self):
        for fn in functions(source()).values():
            fn = copy.deepcopy(fn)
            fn.decorator_list = []
            fn.body = [ast.Pass()]
            for a in fn.args.args:
                a.annotation = None
            env = {}
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                    "sig",
                    "exec",
                ),
                env,
            )
            bound = inspect.signature(env[fn.name]).bind(
                *[object() for _ in range(9)], WIDTH=8256, ROWS=17, SLOTS=2, TRACE=False
            )
            self.assertEqual(bound.arguments["WIDTH"], 8256)

    def test_ir_gate_rejects_missing_sort_or_wrong_shape(self):
        line = (
            'hivm.hir.custom {symbol="custom_sort_1d_pack_float"} "sort_1d_pack" '
            "ins(tensor<512xf32>, tensor<2048xf32>) outs(tensor<1024xf32>)"
        )
        self.assertIn(
            "sort_call", sort_ir_evidence(SimpleNamespace(asm=dict(bcmlir=line)))
        )
        for bad in ("", line.replace("512xf32", "256xf32"), line + "\n" + line):
            with self.assertRaises(RuntimeError):
                sort_ir_evidence(SimpleNamespace(asm=dict(bcmlir=bad)))


if __name__ == "__main__":
    unittest.main()
