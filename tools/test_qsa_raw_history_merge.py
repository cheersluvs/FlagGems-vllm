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

"""Check complete-merge math,real history evolution and frozen mixed-kernel work."""

import ast
import copy
import inspect
import random
import unittest
from types import SimpleNamespace

from qsa_raw_history_merge_910b import HELPER, PREFIX, ir_evidence, source
from qsa_raw_tle_sort512_910b import source as fixed_source


def functions(text):
    return {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}


class HistoryTests(unittest.TestCase):
    def test_only_history_helper_changes_and_outer_commit_stays(self):
        fs = functions(source())
        baseline = fs["raw_history_snapshot"]
        old = ast.parse(fixed_source()).body[1]
        self.assertEqual(
            ast.dump(ast.Module(body=baseline.body, type_ignores=[])),
            ast.dump(ast.Module(body=old.body, type_ignores=[])),
        )
        candidate = copy.deepcopy(fs["raw_history_complete"])
        changed = 0
        for n in ast.walk(candidate):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "_history_complete"
            ):
                n.func.id = "_update_online_topk_ub_snapshot"
                changed += 1
        self.assertEqual(changed, 1)
        self.assertEqual(
            ast.dump(ast.Module(body=candidate.body, type_ignores=[])),
            ast.dump(ast.Module(body=baseline.body, type_ignores=[])),
        )
        helper = ast.unparse(functions(HELPER)["_history_complete"])
        for removed in ("saved_state", "tl.gather", "consumed", "tle.dsa.alloc"):
            self.assertNotIn(removed, helper)

    def test_actual_helper_on_four_adversarial_evolving_updates(self):
        helper = functions(HELPER)["_history_complete"]
        helper.decorator_list = []
        for arg in helper.args.args:
            arg.annotation = None

        def insert(dst, src, offset, shape, stride):
            result = dst[:]
            result[offset[0] : offset[0] + shape[0]] = src[: shape[0]]
            return result

        def raw(name, packed, *args, out=None):
            self.assertEqual(name, "mrgsort")
            off0, off1, off2, off3, len0, len1, len2, len3, exhaust, valid, repeats = (
                args
            )
            self.assertEqual((off0, off1, off2, off3), (0, 512, 0, 0))
            self.assertEqual(
                (len0, len1, len2, len3, exhaust, valid, repeats),
                (512, 512, 0, 0, False, 3, 1),
            )
            pairs = list(zip(packed[:2048:2], packed[1:2048:2]))
            merged = sorted(pairs, reverse=True)
            return [word for pair in merged for word in pair]

        def static_assert(condition):
            self.assertTrue(condition)

        env = dict(
            tl=SimpleNamespace(
                zeros=lambda shape, dtype: [0] * shape[0],
                float32=None,
                static_assert=static_assert,
            ),
            tle=SimpleNamespace(
                dsa=SimpleNamespace(
                    insert_slice=insert, ascend=SimpleNamespace(raw=raw)
                )
            ),
            al=SimpleNamespace(
                extract_slice=lambda src, offset, shape, stride: src[
                    offset[0] : offset[0] + shape[0]
                ]
            ),
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])),
                "helper",
                "exec",
            ),
            env,
        )
        rng = random.Random(17)
        for scenario in ("low", "high", "mixed", "ties", "infinity"):
            pool = [
                (0.0 if scenario == "ties" else float(512 - i), i) for i in range(512)
            ]
            state = sorted(pool, reverse=True)
            for step in range(4):
                values = [
                    (
                        float(-10000 - step * 1000 - i)
                        if scenario == "low"
                        else (
                            float(10000 + step * 1000 - i)
                            if scenario == "high"
                            else (
                                0.0
                                if scenario == "ties"
                                else (
                                    -float("inf")
                                    if scenario == "infinity" and step == 1
                                    else float(rng.randint(-1000, 1000))
                                )
                            )
                        )
                    )
                    for i in range(512)
                ]
                tile = [(value, (step + 1) * 512 + i) for i, value in enumerate(values)]
                pool.extend(tile)
                packed_state = [word for pair in state for word in pair]
                packed_tile = [
                    word for pair in sorted(tile, reverse=True) for word in pair
                ]
                merged = env["_history_complete"](packed_state, packed_tile, 512, 512)
                state = list(zip(merged[::2], merged[1::2]))
                self.assertEqual(
                    [v for v, i in state],
                    sorted((v for v, i in pool), reverse=True)[:512],
                )
                self.assertEqual(len(set(i for v, i in state)), 512)
                mapping = dict((i, v) for v, i in pool)
                self.assertTrue(all(mapping[i] == v for v, i in state))

    def test_actual_exhausted_prefix_is_sufficient_on_evolving_history(self):
        helper = functions(PREFIX)["_history_prefix"]
        helper.decorator_list = []
        for arg in helper.args.args:
            arg.annotation = None

        def insert(dst, src, offset, shape, stride):
            result = dst[:]
            result[offset[0] : offset[0] + shape[0]] = src[: shape[0]]
            return result

        def raw(name, packed, ways, *args, out=None):
            self.assertEqual(name, "merge_exhaust_sort4")
            self.assertEqual(ways, 2)
            self.assertEqual(args, (0, 512, 1024, 1536, 512, 512, 0, 0))
            a = list(zip(packed[:1024:2], packed[1:1024:2]))
            b = list(zip(packed[1024:2048:2], packed[1025:2048:2]))
            emitted = []
            i = j = 0
            while i < 512 and j < 512:
                if a[i][0] >= b[j][0]:
                    emitted.append(a[i])
                    i += 1
                else:
                    emitted.append(b[j])
                    j += 1
            self.assertGreaterEqual(len(emitted), 512)
            words = [word for pair in emitted for word in pair]
            return words + [float("nan") for _ in range(2048 - len(words))], [
                i,
                j,
                0,
                0,
            ]

        def static_assert(condition):
            self.assertTrue(condition)

        env = dict(
            tl=SimpleNamespace(
                zeros=lambda shape, dtype: [0] * shape[0],
                float32=None,
                int32=None,
                static_assert=static_assert,
            ),
            tle=SimpleNamespace(
                dsa=SimpleNamespace(
                    insert_slice=insert, ascend=SimpleNamespace(raw=raw)
                )
            ),
            al=SimpleNamespace(
                extract_slice=lambda src, off, shape, stride: src[
                    off[0] : off[0] + shape[0]
                ]
            ),
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])),
                "prefix",
                "exec",
            ),
            env,
        )
        rng = random.Random(19)
        for scenario in ("low", "high", "mixed", "ties", "invisible"):
            pool = [
                (0.0 if scenario == "ties" else float(512 - i), i) for i in range(512)
            ]
            state = sorted(pool, reverse=True)
            for step in range(4):
                tile = [
                    (
                        (
                            float(-10000 - i)
                            if scenario == "low"
                            else (
                                float(10000 + step * 1000 - i)
                                if scenario == "high"
                                else (
                                    0.0
                                    if scenario == "ties"
                                    else (
                                        -float("inf")
                                        if scenario == "invisible"
                                        else float(rng.randint(-1000, 1000))
                                    )
                                )
                            )
                        ),
                        (step + 1) * 512 + i,
                    )
                    for i in range(512)
                ]
                pool.extend(tile)
                packed_state = [w for pair in state for w in pair]
                packed_tile = [w for pair in sorted(tile, reverse=True) for w in pair]
                result = env["_history_prefix"](packed_state, packed_tile, 512, 512)
                state = list(zip(result[::2], result[1::2]))
                self.assertEqual(
                    [v for v, i in state],
                    sorted((v for v, i in pool), reverse=True)[:512],
                )
                self.assertEqual(len(set(i for v, i in state)), 512)

    def test_actual_signatures_bind_transition_and_mixed_payloads(self):
        for name, original in functions(source()).items():
            if name.startswith("_history_"):
                continue
            fn = copy.deepcopy(original)
            fn.body = [ast.Pass()]
            fn.decorator_list = []
            for arg in fn.args.args:
                arg.annotation = None
            env = {}
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                    "sig",
                    "exec",
                ),
                env,
            )
            kwargs = (
                {}
                if name.startswith("history_check_")
                else dict(ROWS=17, WIDTH=8256, SLOTS=2, TRACE=False)
            )
            count = 3 if name.startswith("history_check_") else 9
            inspect.signature(env[name]).bind(
                *[object() for _ in range(count)], **kwargs
            )

    def test_ir_gate_for_complete_and_snapshot_paths(self):
        for complete in (False, True):
            for local in (False, True):
                names = ["sort32", "mrgsort", "mrgsort", "unpack_sort"] if local else []
                names += ["mrgsort" if complete else "merge_exhaust_sort4"]
                text = "\n".join('hivm.hir.custom "' + n + '"' for n in names)
                ir_evidence(SimpleNamespace(asm=dict(bcmlir=text)), complete, local)
                with self.assertRaises(RuntimeError):
                    ir_evidence(SimpleNamespace(asm=dict(bcmlir="")), complete, local)


if __name__ == "__main__":
    unittest.main()
