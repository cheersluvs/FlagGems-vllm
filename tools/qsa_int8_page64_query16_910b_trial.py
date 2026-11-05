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

"""Restricted sixteen-query N64 scoring candidate and guarded adapter."""

import bisect
import time

import torch
import triton
import triton.language as tl
from qsa_int8_shared_map_selection_910b_trial import PreparedSharedMapInt8Selection


@triton.jit
def _qsa_page64_query16_score(
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
    KEY_TILES: tl.constexpr,
):
    tl.static_assert(BLOCK_N == 64 and PAGE == 192)
    groups: tl.constexpr = triton.cdiv(ROWS, 16)
    tiles: tl.constexpr = triton.cdiv(WIDTH, BLOCK_N)
    dims = tl.arange(0, 128)
    lanes = tl.arange(0, 64)
    queries = tl.arange(0, 16)
    blocks: tl.constexpr = triton.cdiv(tiles, KEY_TILES)
    for unit in range(tl.program_id(0), groups * blocks, tl.num_programs(0)):
        group = unit // blocks
        first_tile = unit % blocks * KEY_TILES
        base = group * 16
        rows = base + queries
        flat_heads = base * 4 + lanes
        query = tl.load(
            Q + flat_heads[:, None] * 128 + dims[None, :],
            mask=flat_heads[:, None] < ROWS * 4,
            other=0,
        )
        query_scale = tl.load(QS + flat_heads, mask=flat_heads < ROWS * 4, other=0).to(
            tl.float32
        )
        weights = tl.load(Weights + flat_heads, mask=flat_heads < ROWS * 4, other=0).to(
            tl.float32
        )
        visible = tl.load(Visible + rows, mask=rows < ROWS, other=0)
        max_visible = tl.max(visible, axis=0)
        for local_tile in range(KEY_TILES):
            tile = first_tile + local_tile
            columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
            key_live = (columns < WIDTH) & (columns < max_visible)
            start = tile * BLOCK_N
            page = start // PAGE
            offset = start % PAGE
            physical = tl.load(
                Table + base * COLUMNS + tl.minimum(page, COLUMNS - 1),
                mask=(start < WIDTH) & (start < max_visible) & (page < COLUMNS),
                other=0,
            )
            token = physical.to(tl.int64) * PAGE + offset + tl.arange(0, BLOCK_N)
            keys = tl.load(
                K + token[:, None] * 128 + dims[None, :],
                mask=key_live[:, None],
                other=0,
            )
            key_scale = tl.load(KS + token, mask=key_live, other=0).to(tl.float32)
            dot = tl.dot(query, tl.trans(keys), out_dtype=tl.int32)
            dequant = dot.to(tl.float32) * query_scale[:, None] * key_scale[None, :]
            weighted = tl.maximum(dequant, 0.0) * weights[:, None]
            scores = tl.sum(tl.reshape(weighted, (16, 4, BLOCK_N)), axis=1)
            live = (columns[None, :] < WIDTH) & (columns[None, :] < visible[:, None])
            tl.store(
                Out + rows[:, None] * WIDTH + columns[None, :],
                tl.where(live, scores, -float("inf")),
                mask=(columns[None, :] < WIDTH) & (rows[:, None] < ROWS),
            )


class PreparedQuery16Int8Selection(PreparedSharedMapInt8Selection):
    """Use M64 only for proven shared-map N64 chunks; retain current fallback."""

    def __init__(self, *args, **kwargs):
        started = time.perf_counter()
        super().__init__(*args, **kwargs)
        starts = list(kwargs.get("request_starts", args[5] if len(args) > 5 else []))
        table = self.values["row_table"].cpu().tolist()
        self.query16_routes = {}
        for entry in self.route_checks:
            first, count = entry["first"], entry["count"]
            eligible = self.single_page64 and entry["shared"]
            reason = "Current route retained"
            if eligible:
                for base in range(first, first + count, 16):
                    owner = bisect.bisect_right(starts, base) - 1
                    if any(
                        table[row] != table[base]
                        or bisect.bisect_right(starts, row) - 1 != owner
                        for row in range(base, min(base + 16, first + count))
                    ):
                        eligible = False
                        reason = "Sixteen-query group crosses request or page mapping"
                        break
            self.query16_routes[first, count] = eligible
            entry["query_group"] = 16 if eligible else 4
            entry["query16_reason"] = (
                "Shared mapping and request owner proven" if eligible else reason
            )
        self.preparation_ms = (time.perf_counter() - started) * 1000

    def score_into(self, first, count, target):
        if not self.query16_routes.get((first, count), False):
            return super().score_into(first, count, target)
        if (
            target.shape != (count, self.width)
            or target.dtype != torch.float32
            or target.device != self.scratch.device
            or not target.is_contiguous()
        ):
            raise ValueError("Contiguous FP32 score target required")
        v = self.values
        blocks = triton.cdiv(triton.cdiv(self.width, 64), 8)
        return _qsa_page64_query16_score[(min(20, triton.cdiv(count, 16) * blocks),)](
            v["q"][first : first + count],
            v["cache"],
            v["qs"][first : first + count],
            v["ks"],
            v["weights"][first : first + count],
            v["row_table"][first : first + count],
            v["visible"][first : first + count],
            target,
            ROWS=count,
            COLUMNS=v["row_table"].shape[1],
            WIDTH=self.width,
            PAGE=192,
            BLOCK_N=64,
            KEY_TILES=8,
            num_warps=4,
        )
