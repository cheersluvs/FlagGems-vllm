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

"""Exercise actual 2D source bounds, order, coverage and missing-trace retry."""

import ast
import contextlib
import copy
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qsa_finish_blocked_probe_910b import source
from qsa_finish_merge_probe_910b import WIDTHS
from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan
from qsa_int8_trace_timer_910b import IncompleteScoreTrace
from test_qsa_finish_merge_probe import BoundedTL
from test_qsa_finish_prefix_probe import Integer, Pointer, fixture


class VectorIndex(Integer):
    def __getitem__(self, index):
        return self

    def __add__(self, other):
        return VectorIndex(int(self) + int(other))

    __radd__ = __add__

    def __mul__(self, other):
        return VectorIndex(int(self) * int(other))

    __rmul__ = __mul__

    def __mod__(self, other):
        return VectorIndex(int(self) % int(other))

    def __floordiv__(self, other):
        return VectorIndex(int(self) // int(other))


class TL(BoundedTL):
    @staticmethod
    def reshape(value, shape):
        return value

    def arange(self, begin, end):
        return VectorIndex(
            self.column
            if end == 64
            else self.tile * 64 + self.column if end in WIDTHS else self.tile
        )

    def where(self, condition, yes, no):
        value = yes if condition else no
        return VectorIndex(value) if isinstance(value, int) else value

    def load(self, *args, **kwargs):
        value = super().load(*args, **kwargs)
        return VectorIndex(value) if isinstance(value, int) else value


def functions(tl):
    nodes = []
    for fn in ast.parse(source()).body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        fn = copy.deepcopy(fn)
        fn.decorator_list = []
        for arg in fn.args.args:
            arg.annotation = None
        nodes.append(fn)
    ns = dict(tl=tl, triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b))
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            "blocked",
            "exec",
        ),
        ns,
    )
    return ns


class BlockedTests(unittest.TestCase):
    def test_actual_2d_source_matches_scores_and_tail_masks(self):
        rows, width, tiles, _, buffers = fixture()
        for block in WIDTHS:
            tl = TL(buffers)
            fn = functions(tl)[f"raw_finish_merge{block}"]
            cores = min(20, rows * ((width + block - 1) // block))
            for pid in range(cores):
                for tile in range(block // 64):
                    for col in range(64):
                        tl.pid = pid
                        tl.tile = tile
                        tl.column = col
                        fn(
                            *(
                                Pointer(k)
                                for k in ("Raw", "KS", "Table", "Visible", "Out")
                            ),
                            rows,
                            width,
                            tiles,
                            cores,
                        )
            self.assertEqual(set(tl.writes), set(range(rows * width)))
            for row in range(rows):
                for col in range(width):
                    expected = -float("inf")
                    if col < buffers["Visible"][row]:
                        expected = (
                            buffers["Raw"][
                                ((row // 16) * tiles + col // 64) * 1024
                                + row % 16 * 64
                                + col % 64
                            ]
                            * buffers["KS"][
                                buffers["Table"][col // 192] * 192 + col % 192
                            ]
                        )
                    self.assertEqual(tl.writes[row * width + col], expected)

    def test_shuffled_pages_and_nonzero_query_group(self):
        rows, width, tiles = 17, 577, 9
        buffers = dict(
            Raw=[float(i % 251 - 125) for i in range(2 * tiles * 1024)],
            KS=[float(i + 1) / 1024 for i in range(4 * 192)],
            Table=[2, 0, 3, 1],
            Visible=[513] * rows,
            Out=[0] * (rows * width),
        )
        tl = TL(buffers)
        fn = functions(tl)["raw_finish_merge1024"]
        for pid in range(17):
            for tile in range(16):
                for c in range(64):
                    tl.pid = pid
                    tl.tile = tile
                    tl.column = c
                    fn(
                        *(Pointer(k) for k in ("Raw", "KS", "Table", "Visible", "Out")),
                        rows,
                        width,
                        tiles,
                        17,
                    )
        for row in (0, 16):
            for col in range(width):
                expected = -float("inf")
                if col < 513:
                    expected = (
                        buffers["Raw"][
                            ((row // 16) * tiles + col // 64) * 1024
                            + row % 16 * 64
                            + col % 64
                        ]
                        * buffers["KS"][buffers["Table"][col // 192] * 192 + col % 192]
                    )
                self.assertEqual(tl.writes[row * width + col], expected)

    def test_no_concat_transpose_or_reshape_and_one_store(self):
        tree = ast.parse(source())
        for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
            calls = [
                ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)
            ]
            self.assertEqual(calls.count("tl.store"), 1)
            self.assertEqual(calls.count("tl.reshape"), 1)
            self.assertEqual(calls.count("tl.load"), 4)
            self.assertFalse(set(calls) & {"tl.cat", "tl.join", "tl.trans"})

    def test_missing_profiler_export_retries_exactly_three_times(self):
        class Prof:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def export_chrome_trace(self, path):
                pass

        api = SimpleNamespace(
            profile=lambda **kw: Prof(), ProfilerActivity=SimpleNamespace(CPU=0, NPU=1)
        )
        torch = SimpleNamespace(
            npu=SimpleNamespace(synchronize=lambda: None),
            profiler=SimpleNamespace(
                record_function=lambda _: contextlib.nullcontext()
            ),
        )
        result = dict(kernel_timer={})
        with tempfile.TemporaryDirectory() as root, patch.dict(
            "sys.modules", torch_npu=SimpleNamespace(profiler=api)
        ):
            args = SimpleNamespace(
                rounds=1, kernel_active=1, kernel_warmup=1, report=root, case="missing"
            )
            with self.assertRaises(IncompleteScoreTrace):
                measure_prefix_plan(
                    args, ["a"], lambda *args: None, {"a": None}, torch, result
                )
        self.assertEqual(
            [r["attempt"] for r in result["rejected_prefix_plans"]], [1, 2, 3]
        )


if __name__ == "__main__":
    unittest.main()
