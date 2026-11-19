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

import ast
import unittest
from types import SimpleNamespace

import torch
from qsa_history_compact32_910b import COMPACT
from qsa_history_sort_census_910b import SortCensus, replay


class SortCensusTest(unittest.TestCase):
    def test_two_blocks_share_cutoff_and_odd_tail_uses_updated_cutoff(self):
        scores = torch.tensor(
            [[4, 3, 2, 1, -1, -2, -3, -4, 10, 11, 12, 13, 1, 0, -2, -3, 10, 9]]
        ).float()
        first, second, tail = replay(torch, scores, torch.tensor([18]), 5, size=4)

        self.assertEqual(first["retained"].tolist(), [[4, 4]])
        self.assertEqual(second["retained"].tolist(), [[4, 1]])
        self.assertEqual(second["ties"].tolist(), [[0, 1]])
        self.assertEqual(tail["retained"].tolist(), [[1]])
        self.assertEqual(tail["visible"].tolist(), [[2]])
        self.assertEqual(tail["ties"].tolist(), [[1]])

    def test_compacted_scores_and_original_indices(self):
        # Execute the generated packing expressions up to the hardware sort.
        body = ast.parse(COMPACT).body
        pos = next(
            i
            for i, n in enumerate(body)
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "short_pairs"
        )
        program = compile(
            ast.Module(body[:pos], type_ignores=[]), "<compact32>", "exec"
        )
        def gather(x, idx, axis):
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(idx.dtype, torch.int32)
            return x.gather(axis, idx.long())

        tl = SimpleNamespace(
            cumsum=torch.cumsum,
            arange=lambda start, stop: torch.arange(start, stop, dtype=torch.int32),
            where=torch.where,
            static_range=range,
            int32=torch.int32,
            float32=torch.float32,
            uint32=torch.uint32,
            full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
            gather=gather,
        )
        gen = torch.Generator().manual_seed(19)
        for count in (1, 2, 17, 31, 32):
            lanes = (
                torch.cat(
                    (
                        torch.tensor([511]),
                        torch.randperm(511, generator=gen)[: count - 1],
                    )
                )
                .sort()
                .values
            )
            score = torch.full((512,), -9.0)
            score[lanes] = torch.arange(count).float() - 3
            variables = dict(
                tl=tl, score=score, vector_block=3, keep=score >= -3, count=count
            )
            exec(program, variables)
            ids = variables["indices32"].view(torch.int32)
            torch.testing.assert_close(ids[:count].long(), lanes + 1536)
            torch.testing.assert_close(variables["scores32"][:count], score[lanes])
            self.assertTrue((ids[count:] == -1).all())
            self.assertTrue(variables["scores32"][count:].isneginf().all())

    def test_empty_blocks_and_ties_are_counted_separately(self):
        scores = torch.zeros((3, 1536))
        scores[:2, :1024] = 1
        scores[0, 1024] = 1
        scores[0, 1056] = 1
        scores[2] = -float("inf")
        census = SortCensus(torch)
        census.observe(scores, torch.tensor([1536, 1536, 0]), 3)
        result = census.result()
        counts = result["counts"]

        self.assertEqual(counts["fixed_sort_lanes"], 9 * 512)
        self.assertEqual(counts["fully_masked_blocks"], 3)
        self.assertEqual(counts["cutoff_unfilled_blocks"], 4)
        self.assertEqual(counts["threshold_skippable_blocks"], 1)
        self.assertEqual(counts["equal_cutoff_elements"], 2)
        self.assertEqual(counts["nonempty_sort32_groups"], 66)
        self.assertEqual(counts["compacted_sort32_groups"], 65)
        self.assertEqual(result["shares"]["skip_by_cutoff_of_steady_visible"], 0.5)
        self.assertEqual(result["steady_nonempty_at_most"][16], 1)
        self.assertEqual(result["survivor_percentiles"]["steady_visible"]["p50"], 0)


if __name__ == "__main__":
    unittest.main()
