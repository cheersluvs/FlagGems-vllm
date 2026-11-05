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

"""Prepared immutable-input QK scoring with per-chunk shared-map fallback."""

import bisect
import time

import torch
import triton
from qsa_int8_query_pack4_910b import (
    _qsa_page64_query16_score,
    _qsa_page64_skip_invisible_score,
    _qsa_qk_key_group_score,
    _qsa_qk_single_page_score,
    _qsa_query_pack4_score,
    validate_shared_groups,
)
from qsa_int8_selection_910b_trial import PreparedInt8Selection


def visibility_tile_policy(visible, width, policy):
    """Count group16/N64 issued tiles and choose a conservative trial route."""
    if policy not in ("all", "conservative"):
        raise ValueError("Unknown visibility policy")
    if not visible or width < 1 or any(v < 0 for v in visible):
        raise ValueError("Nonempty nonnegative visibility and positive width required")
    tiles = ((width + 63) // 64 + 7) // 8 * 8
    issued = ((len(visible) + 15) // 16) * tiles
    useful = sum(
        (min(width, max(visible[b : b + 16])) + 63) // 64
        for b in range(0, len(visible), 16)
    )
    skipped = issued - useful
    selected = policy == "all" or (issued >= 512 and skipped * 8 >= issued)
    return dict(
        issued_tiles=issued,
        useful_tiles=useful,
        skipped_tiles=skipped,
        skipped_fraction=skipped / issued,
        selected=selected,
        policy=policy,
    )


class PreparedSharedMapInt8Selection(PreparedInt8Selection):
    """Prepare guards from actual device tables; inputs must remain immutable.

    This is an unregistered INT8 trial. Preparation copies metadata to CPU and
    is separate from kernel timing. Rebuild the adapter after input mutation.
    """

    def __init__(
        self,
        values,
        output,
        scratch,
        workspace,
        topk,
        request_starts,
        key_group4=False,
        single_page64=False,
        query16=True,
        skip_invisible=False,
        visibility_policy="all",
    ):
        started = time.perf_counter()
        super().__init__(values, output, scratch, workspace, topk)
        if visibility_policy not in ("all", "conservative"):
            raise ValueError("Unknown visibility policy")
        self.skip_invisible = bool(skip_invisible)
        self.key_group4 = bool(key_group4) and triton.cdiv(self.width, 128) >= 16
        self.single_page64 = bool(single_page64) and self.width >= 8192
        starts = list(request_starts)
        if (
            len(starts) < 2
            or starts[0] != 0
            or starts[-1] != self.rows
            or any(not isinstance(x, int) for x in starts)
            or any(b < a for a, b in zip(starts, starts[1:]))
        ):
            raise ValueError("Invalid CPU request boundaries")
        table = values["row_table"].cpu().tolist()
        self.shared_routes, self.route_checks = {}, []
        for first, count in self.chunks:
            try:
                proof = validate_shared_groups(
                    table[first : first + count], starts, first
                )
            except NotImplementedError as error:
                shared, proof = False, {"reason": str(error)}
            else:
                shared = True
            self.shared_routes[first, count] = shared
            self.route_checks.append(
                dict(
                    first=first,
                    count=count,
                    shared=shared,
                    proof=proof,
                    route=(
                        "legacy"
                        if not shared
                        else (
                            "single_page64"
                            if self.single_page64
                            else "group4" if self.key_group4 else "qk"
                        )
                    ),
                )
            )
        self.query16_routes = {}
        self.skip_invisible_routes = {}
        visible = values["visible"].cpu().tolist() if self.skip_invisible else None
        for entry in self.route_checks:
            (first, count) = (entry["first"], entry["count"])
            eligible = query16 and self.single_page64 and entry["shared"]
            reason = "Current route retained"
            if eligible:
                for base in range(first, first + count, 16):
                    owner = bisect.bisect_right(starts, base) - 1
                    if any(
                        (
                            table[row] != table[base]
                            or bisect.bisect_right(starts, row) - 1 != owner
                            for row in range(base, min(base + 16, first + count))
                        )
                    ):
                        eligible = False
                        reason = "Sixteen-query group crosses request or page mapping"
                        break
            self.query16_routes[first, count] = eligible
            selected = False
            if eligible and self.skip_invisible:
                stats = visibility_tile_policy(
                    visible[first : first + count], self.width, visibility_policy
                )
                entry["visibility_tiles"] = stats
                selected = stats["selected"]
            self.skip_invisible_routes[first, count] = selected
            entry["skip_invisible"] = selected
            entry["query_group"] = 16 if eligible else 4
            entry["query16_reason"] = (
                "Shared mapping and request owner proven" if eligible else reason
            )

        self.preparation_ms = (time.perf_counter() - started) * 1000

    def score_into(self, first, count, target):
        if self.query16_routes.get((first, count), False):
            return self._score_query16(first, count, target)
        if not self.shared_routes.get((first, count), False):
            return super().score_into(first, count, target)
        if (
            target.shape != (count, self.width)
            or target.dtype != torch.float32
            or target.device != self.scratch.device
            or not target.is_contiguous()
        ):
            raise ValueError("Contiguous FP32 score target required")
        v = self.values
        if self.single_page64 or self.key_group4:
            block_n = 64 if self.single_page64 else 128
            key_tiles = 8 if self.single_page64 else 4
            kernel = (
                _qsa_qk_single_page_score
                if self.single_page64
                else _qsa_qk_key_group_score
            )
            blocks = triton.cdiv(triton.cdiv(self.width, block_n), key_tiles)
            return kernel[(min(20, triton.cdiv(count, 4) * blocks),)](
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
                BLOCK_N=block_n,
                KEY_TILES=key_tiles,
                num_warps=4,
            )
        return _qsa_query_pack4_score[
            (min(20, triton.cdiv(count, 4) * triton.cdiv(self.width, 128)),)
        ](
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
            BLOCK_N=128,
            SCALE_PRODUCT=False,
            STORE_MODE=1,
            FLAT_QUERY=True,
            QUERY_LEFT=True,
            num_warps=4,
        )

    def _score_query16(self, first, count, target):
        if (
            target.shape != (count, self.width)
            or target.dtype != torch.float32
            or target.device != self.scratch.device
            or (not target.is_contiguous())
        ):
            raise ValueError("Contiguous FP32 score target required")
        v = self.values
        blocks = triton.cdiv(triton.cdiv(self.width, 64), 8)
        kernel = (
            _qsa_page64_skip_invisible_score
            if self.skip_invisible_routes.get((first, count), False)
            else _qsa_page64_query16_score
        )
        return kernel[min(20, triton.cdiv(count, 16) * blocks),](
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
