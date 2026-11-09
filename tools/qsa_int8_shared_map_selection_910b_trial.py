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
from functools import partial

import torch
import triton
from qsa_int8_query_pack4_910b import (
    _qsa_page64_page_reuse_full_score,
    _qsa_page64_page_reuse_score,
    _qsa_page64_query16_score,
    _qsa_page64_skip_invisible_score,
    _qsa_qk_key_group_score,
    _qsa_qk_single_page_score,
    _qsa_query_pack4_score,
    validate_shared_groups,
)
from qsa_int8_selection_910b_trial import PreparedInt8Selection


def visibility_tile_policy(visible, width, policy, *, maxima_out=None):
    """Count issued tiles and optionally capture maxima in an empty list."""
    if policy not in ("all", "conservative"):
        raise ValueError("Unknown visibility policy")
    if not visible or width < 1 or any(v < 0 for v in visible):
        raise ValueError("Nonempty nonnegative visibility and positive width required")
    tiles = ((width + 63) // 64 + 7) // 8 * 8
    issued = ((len(visible) + 15) // 16) * tiles
    maxima = (max(visible[b : b + 16]) for b in range(0, len(visible), 16))
    if maxima_out is not None:
        if maxima_out:
            raise ValueError("maxima_out must be empty")
        maxima_out.extend(maxima)
        maxima = maxima_out
    useful = sum((min(width, maximum) + 63) // 64 for maximum in maxima)
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


def first_invalid_page(mapping, physical_pages):
    return next(
        (i for i, page in enumerate(mapping) if not 0 <= page < physical_pages),
        len(mapping),
    )


def direct_page_route(
    table,
    first,
    count,
    width,
    physical_pages,
    *,
    maxima,
    valid_prefixes=None,
    proven_prefixes=None,
):
    """Prove all scalar page reads, including those with no live keys."""
    if count % 16:
        return "partial_query_group"
    columns = len(table[first])
    # Full-dot loops round up to eight N64 tiles and clamp the table index.
    full_columns = min(columns, triton.cdiv(triton.cdiv(width, 512) * 512, 192))
    if proven_prefixes is not None:
        invalid = (
            min(proven_prefixes) < full_columns
            if maxima is None
            else any(
                min(width, maximum) > prefix * 192
                for maximum, prefix in zip(maxima, proven_prefixes, strict=True)
            )
        )
        if invalid:
            return "invalid_page_id"
        return "full" if maxima is None else "bounded"
    for group, row in enumerate(range(first, first + count, 16)):
        # The bounded loop stops at the last visible tile of each query group.
        required = (
            full_columns
            if maxima is None
            else triton.cdiv(min(width, maxima[group]), 192)
        )
        if valid_prefixes is None:
            pages = table[row][:required]
            if pages and (min(pages) < 0 or max(pages) >= physical_pages):
                return "invalid_page_id"
        else:
            mapping = tuple(table[row])
            prefix = valid_prefixes.get(mapping)
            if prefix is None:
                prefix = first_invalid_page(mapping, physical_pages)
                valid_prefixes[mapping] = prefix
            if required > prefix:
                return "invalid_page_id"
    return "full" if maxima is None else "bounded"


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
        skip_invisible=True,
        visibility_policy="conservative",
        page_reuse=True,
        page_reuse_full=True,
        group_visible_kernels=None,
        *,
        allow_short_query16=False,
        direct_page_kernels=None,
        cache_direct_page_guard=False,
        shared_direct_page_proof=False,
        time_page_proof=False,
        reuse_topk_scratch=False,
    ):
        started = time.perf_counter()
        super().__init__(
            values,
            output,
            scratch,
            workspace,
            topk,
            reuse_topk_scratch=reuse_topk_scratch,
        )
        if visibility_policy not in ("all", "conservative"):
            raise ValueError("Unknown visibility policy")
        self.skip_invisible = bool(skip_invisible)
        self.page_reuse = bool(page_reuse)
        self.page_reuse_full = self.page_reuse and bool(page_reuse_full)
        self.visibility_kernel = (
            _qsa_page64_page_reuse_score
            if self.page_reuse
            else _qsa_page64_skip_invisible_score
        )
        self.key_group4 = bool(key_group4) and triton.cdiv(self.width, 128) >= 16
        self.single_page64 = bool(single_page64) and self.width >= 8192
        query16_page64 = self.single_page64 or (single_page64 and allow_short_query16)
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
        self.group_visible_kernels = group_visible_kernels
        self.group_visible_offsets = {}
        self.direct_page_kernels = direct_page_kernels
        self.direct_page_routes = {}
        direct_page_guard_ns = 0
        query16_proof_ns = 0
        valid_prefixes = (
            {} if cache_direct_page_guard or shared_direct_page_proof else None
        )
        if shared_direct_page_proof:
            physical_pages = values["cache"].shape[0]
            last_mapping, last_prefix = None, None
        packed = []
        metadata_pack_ns = 0
        groups_from_policy = 0
        for entry in self.route_checks:
            first, count = (entry["first"], entry["count"])
            eligible = query16 and query16_page64 and entry["shared"]
            reason = "Current route retained"
            prefixes = (
                []
                if eligible
                and shared_direct_page_proof
                and direct_page_kernels is not None
                and self.page_reuse
                and count % 16 == 0
                else None
            )
            if eligible:
                if time_page_proof:
                    proof_start = time.perf_counter_ns()
                for base in range(first, first + count, 16):
                    owner = bisect.bisect_right(starts, base) - 1
                    end = min(base + 16, first + count)
                    anchor = table[base]
                    if shared_direct_page_proof:
                        # The four-row proof already established equality within each block.
                        mismatch = end > starts[owner + 1] or any(
                            table[row] != anchor for row in range(base + 4, end, 4)
                        )
                    else:
                        mismatch = any(
                            table[row] != anchor
                            or bisect.bisect_right(starts, row) - 1 != owner
                            for row in range(base, end)
                        )
                    if mismatch:
                        eligible = False
                        reason = "Sixteen-query group crosses request or page mapping"
                        break
                    if prefixes is not None:
                        if anchor != last_mapping:
                            mapping = tuple(anchor)
                            last_prefix = valid_prefixes.get(mapping)
                            if last_prefix is None:
                                last_prefix = first_invalid_page(anchor, physical_pages)
                                valid_prefixes[mapping] = last_prefix
                            last_mapping = anchor
                        prefixes.append(last_prefix)
                if time_page_proof:
                    query16_proof_ns += time.perf_counter_ns() - proof_start
            self.query16_routes[first, count] = eligible
            selected = False
            want_metadata = (
                group_visible_kernels is not None
                and eligible
                and self.page_reuse
                and count % 16 == 0
            )
            want_page_check = (
                direct_page_kernels is not None and eligible and self.page_reuse
            )
            maxima = (
                [] if want_metadata or (want_page_check and count % 16 == 0) else None
            )
            if eligible and self.skip_invisible:
                stats = visibility_tile_policy(
                    visible[first : first + count],
                    self.width,
                    visibility_policy,
                    maxima_out=maxima,
                )
                entry["visibility_tiles"] = stats
                selected = stats["selected"]
            self.skip_invisible_routes[first, count] = selected
            entry["skip_invisible"] = selected
            entry["page_reuse"] = self.page_reuse and (
                selected or (eligible and self.page_reuse_full)
            )
            entry["query_group"] = 16 if eligible else 4
            entry["query16_reason"] = (
                "Shared mapping and request owner proven" if eligible else reason
            )
            if direct_page_kernels is not None:
                route = "ineligible_route"
                if want_page_check and entry["page_reuse"]:
                    guard_start = time.perf_counter_ns()
                    route = direct_page_route(
                        table,
                        first,
                        count,
                        self.width,
                        (
                            physical_pages
                            if shared_direct_page_proof
                            else values["cache"].shape[0]
                        ),
                        maxima=maxima if selected else None,
                        valid_prefixes=valid_prefixes,
                        proven_prefixes=prefixes,
                    )
                    direct_page_guard_ns += time.perf_counter_ns() - guard_start
                self.direct_page_routes[first, count] = route
            if want_metadata and entry["page_reuse"]:
                metadata_start = time.perf_counter_ns()
                self.group_visible_offsets[first, count] = len(packed)
                if maxima:
                    packed.extend(maxima)
                    groups_from_policy += len(maxima)
                else:
                    if visible is None:
                        visible = values["visible"].cpu().tolist()
                    packed.extend(
                        max(visible[row : row + 16])
                        for row in range(first, first + count, 16)
                    )
                metadata_pack_ns += time.perf_counter_ns() - metadata_start

        self.group_visible_metadata = None
        copy_ns = 0
        if packed:
            copy_start = time.perf_counter_ns()
            self.group_visible_metadata = torch.tensor(
                packed, dtype=torch.int32, device="cpu"
            ).to(self.scratch.device)
            copy_ns = time.perf_counter_ns() - copy_start
        self.group_visible_stats = dict(
            hit_chunks=len(self.group_visible_offsets),
            bytes=len(packed) * 4,
            groups_from_policy=groups_from_policy,
            copies=int(bool(packed)),
            pack_us=metadata_pack_ns / 1000,
            alloc_copy_us=copy_ns / 1000,
            prepare_us=(metadata_pack_ns + copy_ns) / 1000,
        )
        self.direct_page_guard_us = direct_page_guard_ns / 1000
        self.page_proof_us = (query16_proof_ns + direct_page_guard_ns) / 1000
        self.direct_page_unique_mappings = (
            None if valid_prefixes is None else len(valid_prefixes)
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

    def prepare_bound_execution(self):
        """Prepare a callable for sequential reuse with this adapter's tensors."""
        if not any(self.query16_routes.values()):
            return self
        launches = []
        for first, count in self.chunks:
            if self.query16_routes.get((first, count), False):
                target = self.scratch[:count]
                launch = self.bind_query16_score(first, count, target)
            else:
                target, launch = None, None
            launches.append((first, count, target, launch))

        def execute():
            persistent_scratch = self._new_topk_scratch()
            for first, count, target, launch in launches:
                if launch is None:
                    target = self.scratch[:count]
                    self.score_into(first, count, target)
                else:
                    launch()
                self.select_chunk(first, target, persistent_scratch)
            return self.output

        return execute

    def bind_query16_score(self, first, count, target):
        """Bind one eligible chunk's launch for the host-dispatch comparison."""
        if not self.query16_routes.get((first, count), False):
            raise ValueError("Binding requires an eligible query16 chunk")
        return self._score_query16(first, count, target, bind=True)

    def _score_query16(self, first, count, target, *, bind=False):
        if (
            target.shape != (count, self.width)
            or target.dtype != torch.float32
            or target.device != self.scratch.device
            or (not target.is_contiguous())
        ):
            raise ValueError("Contiguous FP32 score target required")
        v = self.values
        blocks = triton.cdiv(triton.cdiv(self.width, 64), 8)
        if self.skip_invisible_routes.get((first, count), False):
            kernel = self.visibility_kernel
        elif self.page_reuse_full:
            kernel = _qsa_page64_page_reuse_full_score
        else:
            kernel = _qsa_page64_query16_score
        group_offset = (
            self.group_visible_offsets.get((first, count))
            if self.group_visible_offsets
            else None
        )
        if group_offset is not None:
            kernel = self.group_visible_kernels[
                self.skip_invisible_routes[(first, count)]
            ]
        elif self.direct_page_routes.get((first, count)) in ("bounded", "full"):
            kernel = self.direct_page_kernels[self.skip_invisible_routes[first, count]]
        launch = kernel[min(20, triton.cdiv(count, 16) * blocks),]
        args = (
            v["q"][first : first + count],
            v["cache"],
            v["qs"][first : first + count],
            v["ks"],
            v["weights"][first : first + count],
            v["row_table"][first : first + count],
            v["visible"][first : first + count],
            target,
        )
        if group_offset is not None:
            args += (self.group_visible_metadata, group_offset)
        kwargs = dict(
            ROWS=count,
            COLUMNS=v["row_table"].shape[1],
            WIDTH=self.width,
            PAGE=192,
            BLOCK_N=64,
            KEY_TILES=8,
            num_warps=4,
        )
        return partial(launch, *args, **kwargs) if bind else launch(*args, **kwargs)
