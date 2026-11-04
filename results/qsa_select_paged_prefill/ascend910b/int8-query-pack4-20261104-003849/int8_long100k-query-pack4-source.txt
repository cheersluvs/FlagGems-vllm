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

"""Four-query shared-cache score diagnostic, restricted to validated row maps."""

import bisect

import triton
import triton.language as tl


def validate_shared_groups(table_rows, starts, first):
    """Validate CPU fixture metadata; no NPU tensor contents are inspected."""
    if (
        not table_rows
        or not table_rows[0]
        or any(len(row) != len(table_rows[0]) for row in table_rows)
    ):
        raise ValueError("Nonempty rectangular CPU page table required")
    if (
        len(starts) < 2
        or starts[0] != 0
        or any(b < a for a, b in zip(starts, starts[1:]))
        or first < 0
        or first + len(table_rows) > starts[-1]
    ):
        raise ValueError("Invalid CPU request boundaries")
    for base in range(0, len(table_rows), 4):
        anchor = table_rows[base]
        owner = bisect.bisect_right(starts, first + base) - 1
        for row in range(base, min(base + 4, len(table_rows))):
            if (
                table_rows[row] != anchor
                or bisect.bisect_right(starts, first + row) - 1 != owner
            ):
                raise NotImplementedError(
                    f"Query group crosses a request or page mapping at row {first+row}"
                )
    return dict(
        groups=triton.cdiv(len(table_rows), 4),
        all_group_maps_equal=True,
        same_request_per_group=True,
    )


@triton.jit
def _qsa_query_pack4_score(
    Q,
    K,
    QS,
    KS,
    Weights,
    Table,
    Visible,
    Out,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_PRODUCT: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(WIDTH, BLOCK_N)
    groups: tl.constexpr = triton.cdiv(ROWS, 4)
    dims = tl.arange(0, 128)
    lanes = tl.arange(0, 16)
    queries = tl.arange(0, 4)
    for work in range(tl.program_id(0), groups * tiles, tl.num_programs(0)):
        base = (work // tiles) * 4
        rows = base + queries
        columns = (work % tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
        visible = tl.load(Visible + rows, mask=rows < ROWS, other=0)
        max_visible = tl.max(visible, axis=0)
        key_live = (columns < WIDTH) & (columns < max_visible)
        physical = tl.load(
            Table + base * COLUMNS + columns // PAGE, mask=key_live, other=0
        )
        token = physical.to(tl.int64) * PAGE + columns % PAGE
        keys = tl.load(
            K + token[:, None] * 128 + dims[None, :], mask=key_live[:, None], other=0
        )
        key_scale = tl.load(KS + token, mask=key_live, other=0).to(tl.float32)
        query_rows = base + lanes // 4
        heads = lanes % 4
        query = tl.load(
            Q + (query_rows[None, :] * 4 + heads[None, :]) * 128 + dims[:, None],
            mask=query_rows[None, :] < ROWS,
            other=0,
        )
        dot = tl.dot(keys, query, out_dtype=tl.int32)
        query_scale = tl.load(
            QS + query_rows * 4 + heads, mask=query_rows < ROWS, other=0
        ).to(tl.float32)
        weights = tl.load(
            Weights + query_rows * 4 + heads, mask=query_rows < ROWS, other=0
        ).to(tl.float32)
        if SCALE_PRODUCT:
            dequant = dot.to(tl.float32) * (query_scale[None, :] * key_scale[:, None])
        else:
            dequant = dot.to(tl.float32) * query_scale[None, :] * key_scale[:, None]
        weighted = tl.maximum(dequant, 0.0) * weights[None, :]
        scores = tl.sum(tl.reshape(weighted, (BLOCK_N, 4, 4)), axis=2)
        live = (columns[:, None] < WIDTH) & (columns[:, None] < visible[None, :])
        tl.store(
            Out + rows[None, :] * WIDTH + columns[:, None],
            tl.where(live, scores, -float("inf")),
            mask=(columns[:, None] < WIDTH) & (rows[None, :] < ROWS),
        )
