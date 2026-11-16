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

"""Exercise generated routes, exact masks/bounds and shared trace accounting."""

import ast
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_qsa_finish_blocked_probe as blocked  # noqa: E402
from qsa_finish_routing_probe_910b import VARIANTS, source, stage_totals  # noqa: E402
from qsa_ten_score_suite_910b import runtime_source  # noqa: E402
from test_qsa_finish_prefix_probe import Pointer  # noqa: E402


class TL(blocked.TL):
    float32 = float

    def full(self, shape, value, dtype):
        return value


class FinishRoutingTests(unittest.TestCase):
    def test_actual_generated_routes_are_exact_and_bounded(self):
        for width in (65, 1024, 2057):
            rows = 17
            tiles = (width + 63) // 64
            pages = (width + 191) // 192
            visible = ([0, 1, 63, 64, 65, 1023, 1024, 1025, 2048, 2049, 2057] * 2)[
                :rows
            ]
            visible = [min(width, n) for n in visible]
            table = list(reversed(range(pages)))
            ks = [float(i + 1) / 4096 for i in range(pages * 192)]
            logical = [ks[table[i // 192] * 192 + i % 192] for i in range(tiles * 64)]
            buffers = dict(
                Raw=[float(i % 257 - 128) / 16 for i in range(2 * tiles * 1024)],
                KS=ks,
                Table=table,
                Visible=visible,
                LogicalKS=logical,
                Out=[None] * (rows * width),
            )
            baseline = None
            for arm in ("logical1024",) + VARIANTS:
                tl = TL(buffers)
                with patch.object(
                    blocked, "source", lambda: runtime_source() + source()
                ):
                    ns = blocked.functions(tl)
                for pid in range(20):
                    for tile in range(16):
                        for col in range(64):
                            tl.pid, tl.tile, tl.column = pid, tile, col
                            ns["raw_finish_" + arm](
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
                if baseline is None:
                    baseline = tl.writes
                self.assertEqual(tl.writes, baseline, arm)
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

    def test_empty_helper_has_no_reads_and_tail_has_no_page_table(self):
        fns = {
            n.name: n
            for n in ast.parse(source()).body
            if isinstance(n, ast.FunctionDef)
        }
        calls = [
            ast.unparse(n.func)
            for n in ast.walk(fns["raw_empty_block"])
            if isinstance(n, ast.Call)
        ]
        self.assertNotIn("tl.load", calls)
        names = {
            n.id for n in ast.walk(fns["raw_logical_tail"]) if isinstance(n, ast.Name)
        }
        self.assertNotIn("Table", names)
        self.assertNotIn("KS", names)
        compile(ast.parse(runtime_source() + source()), "routing", "exec")

    def test_stage_sums_use_same_windows_and_reject_wrong_total(self):
        events = [
            dict(ph="M", name="process_name", pid=1, args=dict(name="Ascend Hardware"))
        ]
        for i in range(2):
            events.append(dict(ph="X", name=f"qsa/timed/{i}", pid=2, ts=i * 20, dur=10))
        for ts, dur, name in [
            (1, 2, "raw_finish_combined1024"),
            (4, 3, "raw_query16_cube"),
            (21, 7, "raw_finish_logical1024"),
        ]:
            events.append(dict(ph="X", name=name, pid=1, ts=ts, dur=dur))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "trace.json.gz").write_bytes(
                gzip.compress(json.dumps(events).encode())
            )
            samples = {
                "combined1024": [
                    dict(artifact="trace.json.gz", window_indices=[0], kernel_us=5)
                ],
                "logical1024": [
                    dict(artifact="trace.json.gz", window_indices=[1], kernel_us=7)
                ],
            }
            result = stage_totals(path, samples)
            self.assertEqual(result["combined1024"], dict(finish=2, cube=3))
            self.assertEqual(result["logical1024"], dict(finish=7))
            samples["combined1024"][0]["kernel_us"] = 6
            with self.assertRaises(AssertionError):
                stage_totals(path, samples)


if __name__ == "__main__":
    unittest.main()
