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


"""Compare one- and two-slot Cube-to-Vector score and TopK scheduling."""
"""Prepared mixed Cube/Vector candidate for the full INT8 harness inputs."""

from functools import partial

import torch
import triton

from probe_qsa_int8_mixed_pipeline_910b import (
    CUBE_TILE,
    PAGE,
    TOPK,
    _qsa_int8_handoff_copy,
    _qsa_int8_merge_sorted_tiles,
    _qsa_int8_merge_two_levels,
)


class PreparedFusedBinarySelection:
    def __init__(self, values, host_visible, starts, chunk_rows):
        device = values["q"].device
        rows = values["q"].shape[0]
        self.output = torch.empty((rows, TOPK), device=device, dtype=torch.int32)
        self.chunks = []
        rank = torch.arange(TOPK, device=device, dtype=torch.int32)
        key = values["cache"].view(-1, 128)
        key_scale = values["ks"].view(-1)
        scratch = {}

        def buffer(shape, dtype):
            return torch.empty(shape, device=device, dtype=dtype)

        for begin, end in zip(starts[:-1], starts[1:], strict=True):
            # Both rows in each Cube program use the same request page table.
            # Start a new chunk at every request boundary.
            lists = max(1, triton.cdiv(int(host_visible[end - 1]), CUBE_TILE))
            for first in range(begin, end, chunk_rows):
                last = min(first + chunk_rows, end)
                count = last - first
                cores = triton.cdiv(count, 2)
                packed_q = torch.zeros((cores, 16, 128), dtype=torch.int8, device=device)
                # Use a padded row dimension so an odd final row never reads the next request.
                q = values["q"][first:last]
                pair_count = count // 2
                if pair_count:
                    packed_q[:pair_count, :8].copy_(q[:2 * pair_count].view(pair_count, 8, 128))
                if count % 2:
                    packed_q[-1, :4].copy_(q[-1])

                layout = count, lists
                if layout not in scratch:
                    ring = buffer((cores, 2, 16, CUBE_TILE), torch.int32)
                    proposals = buffer((count, lists, 2 * CUBE_TILE), torch.float32)
                    levels = []
                    groups, span = lists, CUBE_TILE
                    while groups > 2 or groups * span < TOPK:
                        groups = triton.cdiv(groups, 2)
                        span = min(2 * span, TOPK)
                        levels.append(buffer((count, groups, 2 * span), torch.float32))
                    scratch[layout] = (
                        ring, proposals, levels, buffer((count, TOPK), torch.float32),
                        buffer((count, TOPK), torch.int64),
                    )
                ring, proposals, levels, scores, positions = scratch[layout]
                visible = values["visible"][first:last]
                producer = partial(
                    _qsa_int8_handoff_copy[(cores,)], packed_q, key,
                    values["row_table"][begin], ring, proposals,
                    values["qs"][first:last], key_scale, values["weights"][first:last], visible,
                    lists, 2, PAGE, CUBE_TILE, count, True, True,
                    use_bytecode=True, disable_auto_cv_work_space_manage=True,
                    disable_auto_inject_block_sync=True, enable_auto_bind_sub_block=False,
                    multibuffer=False, sync_solver=False, unit_flag=False,
                )
                merges = []
                source = proposals
                # With fewer than three lists the two-level prefix does not exist.
                fused = len(levels) >= 2 and lists > 4
                for level, target in enumerate(levels):
                    if fused and level == 0:
                        continue
                    if fused and level == 1:
                        kernel = _qsa_int8_merge_two_levels
                        widths = (CUBE_TILE,)
                    else:
                        kernel = _qsa_int8_merge_sorted_tiles
                        widths = (source.shape[2] // 2, target.shape[2] // 2)
                    merges.append(partial(
                        kernel[(count * target.shape[1],)], source, target,
                        source.shape[1], target.shape[1], *widths,
                        use_bytecode=True, enable_ubuf_saving=True,
                        enable_auto_bind_sub_block=False, multibuffer=False,
                    ))
                    source = target
                self.chunks.append((
                    first, count, lists, fused, producer, merges, source.view(count, -1, 2),
                    scores, positions, rank >= visible[:, None], self.output[first:last],
                ))

    def __call__(self, *, synchronize=False):
        for first, count, lists, fused, producer, merges, pairs, scores, positions, padding, output in self.chunks:
            if synchronize:
                print(f"mixed first={first} rows={count} tiles={lists} fused={fused}", flush=True)
            producer()
            if synchronize:
                torch.npu.synchronize()
            for level, merge in enumerate(merges):
                if synchronize:
                    print(f"mixed first={first} merge={level}", flush=True)
                merge()
                if synchronize:
                    torch.npu.synchronize()
            candidates = pairs[..., 0].contiguous()
            indices = pairs[..., 1].contiguous().view(torch.int32)
            torch.topk(candidates, TOPK, dim=-1, sorted=True, out=(scores, positions))
            torch.gather(indices, 1, positions, out=output)
            output.masked_fill_(padding, -1)
            if synchronize:
                torch.npu.synchronize()
        return self.output
