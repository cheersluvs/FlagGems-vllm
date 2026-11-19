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

import unittest

import torch
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
