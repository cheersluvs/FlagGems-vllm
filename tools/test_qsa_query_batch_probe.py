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

"""Check generated repeat layouts keep every query's sorted list independent."""

import ast
import copy
import random
import unittest
from types import SimpleNamespace

from qsa_query_batch_probe_910b import ir_evidence, source
from qsa_raw_history_merge_910b import source as baseline_source


class Lanes(list):
    def __mod__(self, value):
        return Lanes(v % value for v in self)

    def __radd__(self, value):
        return Lanes(value + v for v in self)

    def to(self, dtype):
        return self


def functions(text):
    return {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}


class BatchTests(unittest.TestCase):
    def test_cube_events_final_output_and_state_commit_are_preserved(self):
        old = functions(baseline_source())["raw_history_prefix"]
        for batch in (2, 4):
            new = functions(source())["raw_history_batch" + str(batch)]
            self.assertEqual(ast.dump(old.body[0]), ast.dump(new.body[0]))
            self.assertEqual(
                ast.dump(old.body[1].body[-1]), ast.dump(new.body[1].body[-1])
            )

            def events(fn):
                return [
                    ast.dump(n)
                    for n in ast.walk(fn)
                    if isinstance(n, ast.Call)
                    and ast.unparse(n.func).startswith("al.sync_block_")
                ]

            self.assertEqual(events(old), events(new))
            text = ast.unparse(new)
            self.assertIn("bind_buffer=update_state_slot", text)
            self.assertIn("_history_prefix(update_state, pairs, 512, 512)", text)
            self.assertIn(f"tl.static_range({batch})", text)

    def test_generated_network_does_not_mix_queries(self):
        rng = random.Random(18)
        for batch in (2, 4):
            fn = functions(source())["raw_history_batch" + str(batch)]
            loop = next(
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.For) and n.target.id == "query_group"
            )
            start = next(
                i
                for i, n in enumerate(loop.body)
                if isinstance(n, ast.Assign) and n.targets[0].id == "flat_lane"
            )
            end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.For))
            code = compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=copy.deepcopy(loop.body[start:end]), type_ignores=[]
                    )
                ),
                "batch",
                "exec",
            )
            for fixture in range(4):
                rows = []
                for row in range(batch):
                    values = [
                        rng.choice((-1.0, 0.0, 1.0, float(row * 100 + i)))
                        for i in range(512)
                    ]
                    if fixture == 0:
                        values = [float(row * 1000)] * 512
                    elif fixture == 1:
                        values = [-float("inf")] * 512
                    elif fixture == 2:
                        values[31 + row :] = [-float("inf")] * (481 - row)
                    rows.append(values)
                calls = []

                def raw(name, *args, out=None):
                    calls.append(name)
                    if name == "sort32":
                        values, indices, repeats = args
                        self.assertEqual(repeats, 16 * batch)
                        packed = []
                        for off in range(0, len(values), 32):
                            packed.extend(
                                x
                                for pair in sorted(
                                    zip(
                                        values[off : off + 32], indices[off : off + 32]
                                    ),
                                    reverse=True,
                                )
                                for x in pair
                            )
                        return packed
                    packed, *cfg = args
                    offsets, lengths = cfg[:4], cfg[4:8]
                    exhausted, valid, repeats = cfg[8:]
                    self.assertFalse(exhausted)
                    self.assertEqual(valid, 15)
                    stride = sum(lengths)
                    self.assertEqual(stride * repeats, batch * 512)
                    result = []
                    for repeat in range(repeats):
                        pairs = []
                        for offset, length in zip(offsets, lengths):
                            begin = repeat * stride + offset
                            pairs += list(
                                zip(
                                    packed[begin * 2 : (begin + length) * 2 : 2],
                                    packed[begin * 2 + 1 : (begin + length) * 2 : 2],
                                )
                            )
                        result.extend(
                            x for pair in sorted(pairs, reverse=True) for x in pair
                        )
                    return result

                env = dict(
                    score=[v for row in rows for v in row],
                    vector_block=7,
                    tle=SimpleNamespace(
                        dsa=SimpleNamespace(ascend=SimpleNamespace(raw=raw))
                    ),
                    tl=SimpleNamespace(
                        arange=lambda a, b: Lanes(range(a, b)),
                        uint32="u32",
                        float32="fp32",
                        reshape=lambda x, shape: x,
                        zeros=lambda *a: None,
                        debug_barrier=lambda: None,
                    ),
                )
                exec(code, env)
                packed = env["sorted_rows"]
                self.assertEqual(calls, ["sort32", "mrgsort", "mrgsort"])
                for row in range(batch):
                    part = packed[row * 1024 : (row + 1) * 1024]
                    self.assertEqual(part[::2], sorted(rows[row], reverse=True))
                    indices = part[1::2]
                    self.assertEqual(set(indices), set(range(7 * 512, 8 * 512)))
                    self.assertEqual(
                        [rows[row][i - 7 * 512] for i in indices], part[::2]
                    )

    def test_ir_gate_rejects_wrong_counts_and_gather(self):
        for batch in (2, 4):
            lines = [
                f'hivm.hir.custom "{name}"'
                for name in ["sort32", "mrgsort", "mrgsort"]
                + ["merge_exhaust_sort4"] * batch
                + ["unpack_sort"]
            ]
            kernel = SimpleNamespace(asm=dict(bcmlir="\n".join(lines)))
            self.assertEqual(ir_evidence(kernel, batch)["gather_references"], 0)
            kernel.asm["bcmlir"] += "triton_gather"
            with self.assertRaises(RuntimeError):
                ir_evidence(kernel, batch)


if __name__ == "__main__":
    unittest.main()
