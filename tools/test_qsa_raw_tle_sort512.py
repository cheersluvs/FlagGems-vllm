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

"""Check primitive composition against independent full-sort values and indices."""

import ast
import copy
import inspect
import random
import unittest
from types import SimpleNamespace

from qsa_raw_cube_block512_910b import source as original_source
from qsa_raw_tle_sort512_910b import ir_evidence, source


def functions(text):
    return {
        fn.name: fn for fn in ast.parse(text).body if isinstance(fn, ast.FunctionDef)
    }


def update(fn):
    return next(
        n
        for n in ast.walk(fn.body[1])
        if isinstance(n, ast.For) and n.target.id == "update_query"
    )


class Indices(list):
    def __radd__(self, x):
        return Indices(x + i for i in self)

    def to(self, dtype):
        return self


class Sort512Tests(unittest.TestCase):
    def test_only_local_sort_changes_with_frozen_cube_and_history(self):
        fs = functions(source())
        old = functions(original_source())["raw_mixed_block512"]
        base, fixed = fs["raw_tle_sort_generic"], fs["raw_tle_sort_fixed"]
        self.assertEqual(
            ast.dump(ast.Module(body=base.body, type_ignores=[])),
            ast.dump(ast.Module(body=old.body, type_ignores=[])),
        )
        self.assertEqual(ast.dump(fixed.body[0]), ast.dump(base.body[0]))
        candidate = copy.deepcopy(fixed)
        body = update(candidate).body
        start = next(
            i
            for i, n in enumerate(body)
            if isinstance(n, ast.Assign) and n.targets[0].id == "local_indices"
        )
        end = (
            next(
                i
                for i, n in enumerate(body)
                if isinstance(n, ast.Assign) and n.targets[0].id == "pairs"
            )
            + 2
        )
        oldpair = next(
            n
            for n in update(base).body
            if isinstance(n, ast.Assign) and n.targets[0].id == "pairs"
        )
        body[start:end] = [copy.deepcopy(oldpair)]
        self.assertEqual(ast.dump(candidate.body[1]), ast.dump(base.body[1]))

    def test_actual_fixed_network_sorts_all512_and_keeps_offset_indices(self):
        fn = functions(source())["raw_tle_sort_fixed"]
        body = update(fn).body
        start = next(
            i
            for i, n in enumerate(body)
            if isinstance(n, ast.Assign) and n.targets[0].id == "local_indices"
        )
        end = (
            next(
                i
                for i, n in enumerate(body)
                if isinstance(n, ast.Assign) and n.targets[0].id == "pairs"
            )
            + 2
        )
        code = compile(
            ast.fix_missing_locations(
                ast.Module(body=copy.deepcopy(body[start:end]), type_ignores=[])
            ),
            "network",
            "exec",
        )
        rng = random.Random(17)
        fixtures = [
            [-float("inf")] * 512,
            [0.0] * 512,
            list(range(512)),
            list(range(511, -1, -1)),
            [(-0.0 if i % 2 else 0.0) for i in range(512)],
            [rng.choice((-3.0, 0.0, 1.0, 9.0, 65536.0)) for _ in range(512)],
        ]
        fixtures += [
            [rng.random() if i < n else -float("inf") for i in range(512)]
            for n in (1, 31, 32, 33, 127, 128, 129, 511)
        ]
        for data in fixtures:
            for offset in (0, 512, 29696):
                calls = []

                def raw(name, *args, out=None):
                    calls.append((name, args[2:] if name == "sort32" else args[1:]))
                    if name == "sort32":
                        values, indices, repeats = args
                        self.assertEqual(repeats, 16)
                        proposals = []
                        for first in range(0, 512, 32):
                            pairs = sorted(
                                zip(
                                    values[first : first + 32],
                                    indices[first : first + 32],
                                ),
                                reverse=True,
                            )
                            proposals.extend(x for pair in pairs for x in pair)
                        return proposals
                    packed, *config = args
                    offs, lens = config[:4], config[4:8]
                    exhausted, valid, repeats = config[8:]
                    self.assertFalse(exhausted)
                    self.assertEqual(valid, 15)
                    stride = sum(lens)
                    result = []
                    for repeat in range(repeats):
                        pairs = []
                        for off, length in zip(offs, lens):
                            begin = repeat * stride + off
                            end = begin + length
                            self.assertLessEqual(end, len(packed) // 2)
                            pairs.extend(
                                zip(
                                    packed[begin * 2 : end * 2 : 2],
                                    packed[begin * 2 + 1 : end * 2 : 2],
                                )
                            )
                        result.extend(
                            x for pair in sorted(pairs, reverse=True) for x in pair
                        )
                    self.assertEqual(len(result), 1024)
                    return result

                env = dict(
                    score=data,
                    vector_block=offset // 512,
                    lane=Indices(range(512)),
                    tl=SimpleNamespace(
                        uint32=None,
                        float32=None,
                        zeros=lambda *a: None,
                        debug_barrier=lambda: None,
                    ),
                    tle=SimpleNamespace(
                        dsa=SimpleNamespace(ascend=SimpleNamespace(raw=raw))
                    ),
                )
                exec(code, env)
                packed = env["pairs"]
                self.assertEqual(packed[::2], sorted(data, reverse=True))
                self.assertEqual(
                    sorted(packed[1::2]), list(range(offset, offset + 512))
                )
                for value, index in zip(packed[::2], packed[1::2]):
                    self.assertEqual(value, data[index - offset])
                self.assertEqual(
                    [name for name, _ in calls], ["sort32", "mrgsort", "mrgsort"]
                )

    def test_real_signatures_bind_nine_pointers_and_configs(self):
        for fn in functions(source()).values():
            fn = copy.deepcopy(fn)
            fn.body = [ast.Pass()]
            fn.decorator_list = []
            for arg in fn.args.args:
                arg.annotation = None
            env = {}
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                    "signature",
                    "exec",
                ),
                env,
            )
            inspect.signature(env[fn.name]).bind(
                *[object() for _ in range(9)], WIDTH=8256, ROWS=17, SLOTS=2, TRACE=False
            )

    def test_ir_gate_requires_fixed_and_history_primitives(self):
        def ir(names):
            return "\n".join('hivm.hir.custom "' + n + '"' for n in names)

        good = ir(
            ["sort32", "mrgsort", "mrgsort", "merge_exhaust_sort4", "unpack_sort"]
        )
        self.assertEqual(
            ir_evidence(SimpleNamespace(asm=dict(bcmlir=good)), True)["calls"][
                "mrgsort"
            ],
            2,
        )
        for bad in (
            good.replace('"sort32"', '"sort_1d_pack"'),
            good.replace('"unpack_sort"', '"missing"'),
            "",
        ):
            with self.assertRaises(RuntimeError):
                ir_evidence(SimpleNamespace(asm=dict(bcmlir=bad)), True)


if __name__ == "__main__":
    unittest.main()
