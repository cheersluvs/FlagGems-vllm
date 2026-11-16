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

"""Check full/tail routing, exact outputs and frozen tail source."""

import ast
import unittest
from unittest.mock import patch

import test_qsa_finish_blocked_probe as blocked_test
from qsa_finish_blocked_probe_910b import source as baseline_source
from qsa_finish_full_block_probe_910b import source
from test_qsa_finish_prefix_probe import Pointer


class FullBlockTests(unittest.TestCase):
    def test_full_tail_and_page_boundaries_match_frozen_kernel(self):
        rows, width, tiles, pages = 17, 2057, 33, 11
        visible = ([0, 1, 1023, 1024, 1025, 2048, 2049] * 3)[:rows]
        buffers = dict(
            Raw=[float(i % 257 - 128) / 16 for i in range(2 * tiles * 1024)],
            KS=[float(i + 1) / 4096 for i in range(pages * 192)],
            Table=list(reversed(range(pages))),
            Visible=visible,
            Out=[None] * (rows * width),
        )
        outputs = []
        for label in ("raw_finish_merge1024", "raw_finish_full1024"):
            tl = blocked_test.TL(buffers)
            with patch.object(blocked_test, "source", source):
                ns = blocked_test.functions(tl)
            calls = dict(full=0, tail=0)
            for kind, fn_name in (("full", "raw_full_body"), ("tail", "raw_full_tail")):
                fn = ns[fn_name]

                def counted(*args, fn=fn, kind=kind):
                    calls[kind] += 1
                    return fn(*args)

                ns[fn_name] = counted
            cores = 20
            for pid in range(cores):
                for tile in range(16):
                    for col in range(64):
                        tl.pid = pid
                        tl.tile = tile
                        tl.column = col
                        ns[label](
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
            outputs.append(tl.writes)
            if label.endswith("full1024"):
                self.assertGreater(calls["full"], 0)
                self.assertGreater(calls["tail"], 0)
        self.assertEqual(outputs[0], outputs[1])
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
                        * buffers["KS"][buffers["Table"][col // 192] * 192 + col % 192]
                    )
                self.assertEqual(outputs[1][row * width + col], expected)

    def test_fast_path_is_unmasked_and_has_no_conditional_tile_selection(self):
        tree = ast.parse(source())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "raw_full_body"
        )
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        self.assertFalse(any(ast.unparse(n.func) == "tl.where" for n in calls))
        loads = [n for n in calls if ast.unparse(n.func) == "tl.load"]
        self.assertEqual(len(loads), 3)
        self.assertFalse(any(k.arg == "mask" for n in calls for k in n.keywords))
        self.assertFalse(any(isinstance(n, ast.For) for n in ast.walk(fn)))

    def test_frozen_source_and_tail_expressions_are_retained(self):
        self.assertTrue(source().startswith(baseline_source()))
        tree = ast.parse(source())
        fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        baseline = fns["raw_finish_merge1024"].body[0].body
        start = next(
            i
            for i, n in enumerate(baseline)
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "tiles"
        )
        kept = [
            n
            for n in baseline[start:]
            if not (
                isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "visible"
            )
        ]
        self.assertEqual(
            [ast.dump(n) for n in kept],
            [ast.dump(n) for n in fns["raw_full_tail"].body],
        )


if __name__ == "__main__":
    unittest.main()
