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

"""Replay paired TopK updates on the exact score trace of the mixed kernel."""

import json
from collections import Counter

from qsa_raw_serial_selection_910b_trial import check_rank


def replay(torch, scores, lengths, blocks, size=512):
    rows, width = scores.shape
    columns = torch.arange(size)
    history = scores.new_full((rows, size), -float("inf"))
    for start_block in range(0, blocks, 2):
        steps = min(2, blocks - start_block)
        cutoff = history[:, -1, None, None]
        start = start_block * size
        stop = (start_block + steps) * size
        tile = scores[:, start:stop]
        if stop > width:
            tile = torch.nn.functional.pad(tile, (0, stop - width), value=-float("inf"))
        tile = tile.reshape(rows, steps, size)
        offsets = columns + torch.arange(start, stop, size)[:, None]
        visible = offsets < lengths[:, None, None]
        if not torch.equal(tile.isfinite(), visible) or not bool(
            (tile.masked_select(~visible) == -float("inf")).all()
        ):
            raise AssertionError(
                f"Invalid scored values or visibility at block {start_block}"
            )
        keeps = visible & (tile >= cutoff)
        yield dict(
            visible=visible.sum(dim=-1),
            retained=keeps.sum(dim=-1),
            ties=(visible & (tile == cutoff)).sum(dim=-1),
            full=cutoff[:, 0, 0].isfinite(),
            nonempty_sort32_groups=keeps.reshape(rows, steps, -1, min(32, size))
            .any(dim=-1)
            .sum(dim=-1),
        )
        # Publish the cutoff only after both blocks, or after the odd tail.
        pruned = tile.masked_fill(~keeps, -float("inf")).reshape(rows, -1)
        history = torch.cat((history, pruned), dim=1).topk(size, dim=1).values
    torch.testing.assert_close(history, scores.topk(size, dim=1).values, rtol=0, atol=0)


def percentiles(hist):
    total = sum(hist)
    if not total:
        return {}
    result = {}
    for percent in (50, 90, 95, 99):
        cumulative = 0
        for count, frequency in enumerate(hist):
            cumulative += frequency
            if cumulative * 100 >= percent * total:
                result[f"p{percent}"] = count
                break
    return result


class SortCensus:
    def __init__(self, torch):
        self.torch = torch
        self.counts = Counter()
        self.histograms = {
            name: torch.zeros(513, dtype=torch.int64)
            for name in ("all", "steady_visible", "steady_nonempty", "odd_tail")
        }
        self.by_pair = {}

    def observe(self, scores, lengths, blocks):
        torch = self.torch
        self.counts.update(rows=scores.size(0), pieces=1, odd_tail_pieces=blocks % 2)
        for pair, data in enumerate(replay(torch, scores, lengths, blocks)):
            visible, retained, ties = (data[k] for k in ("visible", "retained", "ties"))
            full = data["full"][:, None]
            steady = full & (visible > 0)
            empty = retained == 0
            steps = visible.size(1)
            counts = dict(
                row_blocks=visible.numel(),
                visible_elements=int(visible.sum()),
                retained_elements=int(retained.sum()),
                equal_cutoff_elements=int(ties.sum()),
                fully_masked_blocks=int((visible == 0).sum()),
                cutoff_unfilled_blocks=int((~full & (visible > 0)).sum()),
                steady_visible_blocks=int(steady.sum()),
                threshold_skippable_blocks=int((steady & empty).sum()),
                nonempty_sort32_groups=int(data["nonempty_sort32_groups"].sum()),
                compacted_sort32_groups=int(((retained + 31) // 32).sum()),
            )
            if steps == 2:
                counts.update(
                    row_pairs=visible.size(0),
                    row_pairs_both_empty=int(empty.all(dim=1).sum()),
                    row_pairs_one_empty=int((empty.sum(dim=1) == 1).sum()),
                    row_pairs_both_pruned_by_cutoff=int(
                        ((steady & empty).all(dim=1)).sum()
                    ),
                )
            else:
                counts["odd_tail_row_blocks"] = visible.numel()
            self.counts.update(counts)
            for name, values in (
                ("all", retained),
                ("steady_visible", retained[steady]),
                ("steady_nonempty", retained[steady & ~empty]),
                ("odd_tail", retained if steps == 1 else retained[:0]),
            ):
                self.histograms[name] += torch.bincount(values.flatten(), minlength=513)
            self.by_pair.setdefault(pair, Counter()).update(counts)

    def result(self):
        counts = self.counts
        hist = {name: values.tolist() for name, values in self.histograms.items()}
        padded = counts["row_blocks"] * 512
        steady_blocks = counts["steady_visible_blocks"]
        return dict(
            status="passed",
            counts=dict(counts, fixed_sort_lanes=padded),
            shares=dict(
                retained_of_fixed_lanes=counts["retained_elements"] / padded,
                retained_of_visible=counts["retained_elements"]
                / counts["visible_elements"],
                nonempty_sort32_group_share=counts["nonempty_sort32_groups"]
                / (padded // 32),
                compacted_sort32_group_share=counts["compacted_sort32_groups"]
                / (padded // 32),
                skip_by_cutoff_of_steady_visible=(
                    counts["threshold_skippable_blocks"] / steady_blocks
                    if steady_blocks
                    else None
                ),
            ),
            survivor_percentiles={
                name: percentiles(values) for name, values in hist.items()
            },
            steady_nonempty_at_most={
                width: sum(hist["steady_nonempty"][1 : width + 1])
                for width in (16, 32, 64, 128, 256, 512)
            },
            histograms=hist,
            by_pair=self.by_pair,
            ties="Keep scores equal to the cutoff. Compare exact ranked values.",
            scope="One complete synthetic long100k fixture using NPU mixed scores, no timing.",
        )


def collect(torch, context, raw, mixed, path):
    lengths = context["cpu"]["visible"]
    if any(route["mode"] != "paired_reuse_local32" for route in mixed.dispatch_routes):
        raise AssertionError("Expected the paired out=local32 route for all pieces")
    census = SortCensus(torch)
    mixed.reset_guards()
    selected = mixed().cpu()
    mixed.check_guards()

    def observe(item, scores):
        first, count = item["first"], item["count"]
        rows = slice(first, first + count)
        census.observe(scores, lengths[rows], item["blocks"])
        check_rank(torch, selected[rows], scores, lengths[rows], rtol=0, atol=0)

    for index, (entry, pieces) in enumerate(context["jobs"], 1):
        raw.score_chunk(pieces)
        torch.npu.synchronize()
        expected = context["scores"][: entry["count"]].cpu()
        mixed.check_chunk_scores(torch, entry, expected, on_trace=observe)
        if index % 16 == 0 or index == len(context["jobs"]):
            print(
                f"HISTORY_SORT_CENSUS_PROGRESS {index}/{len(context['jobs'])}",
                flush=True,
            )
    if census.counts["rows"] != mixed.rows or census.counts["pieces"] != len(
        mixed.routes
    ):
        raise AssertionError("Incomplete traced coverage")
    result = census.result()
    path.write_text(json.dumps(result, indent=2) + "\n")
    return {
        key: value
        for key, value in result.items()
        if key not in ("histograms", "by_pair")
    }
