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

"""Check repeated prefix accounting without depending on NPU imports."""

import ast
import unittest
from pathlib import Path

from qsa_packing_cost_probe_910b import traffic


class PackingTests(unittest.TestCase):
    def test_growth_tail_and_requests_are_counted_separately(self):
        pieces = [
            dict(request=0, maximum=192, blocks=1, pages=[4]),
            dict(request=0, maximum=513, blocks=2, pages=[4, 5, 6]),
            dict(request=1, maximum=100, blocks=1, pages=[4]),
        ]
        d = traffic(pieces, 128, 4)
        self.assertEqual(d["input_read_bytes"], 805 * 132)
        self.assertEqual(d["repeated_input_bytes"], 192 * 132)
        self.assertEqual(d["repeated_packed_write_bytes"], 512 * 130)
        self.assertEqual(d["all_request_max_packed_bytes"], 1536 * 130)

    def test_zero_visibility_and_changed_mapping(self):
        d = traffic([dict(request=0, maximum=0, blocks=1, pages=[])], 128, 4)
        self.assertEqual(d["input_read_bytes"], 0)
        self.assertEqual(d["packed_write_bytes"], 512 * 130)
        with self.assertRaisesRegex(ValueError, "prefix changed"):
            traffic(
                [
                    dict(request=0, maximum=192, blocks=1, pages=[4]),
                    dict(request=0, maximum=192, blocks=1, pages=[5]),
                ],
                128,
                4,
            )
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            traffic([dict(request=0, maximum=193, blocks=1, pages=[4])], 128, 4)

    def test_execute_preserves_order_and_binds_each_piece(self):
        tree = ast.parse(
            Path(__file__)
            .with_name("qsa_raw_mixed_selection_910b_trial.py")
            .read_text()
        )
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "execute"
        )
        self.assertEqual(
            [ast.unparse(n) for n in fn.body],
            ["pack_qw()", "pack_k()", "return mixed_only(traced)"],
        )
        self.assertEqual(
            [ast.unparse(d) for d in fn.args.defaults],
            ["False", "pack_qw", "pack_k", "mixed_only"],
        )


if __name__ == "__main__":
    unittest.main()
