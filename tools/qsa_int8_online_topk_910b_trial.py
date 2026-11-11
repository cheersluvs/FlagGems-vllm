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


"""Prepare full-input online TopK using the validated scoped mixed kernel."""

from functools import partial

import torch
import triton

from probe_qsa_int8_mixed_pipeline_910b import (
    CUBE_TILE, PAGE, TOPK, _qsa_online_scoped, _qsa_online_grouped, _qsa_online_handoff_grouped, _qsa_online_query4, _qsa_online_prefix, _qsa_score_handoff_prefix,
)
from qsa_int8_fused_binary_910b_trial import pack_mixed_queries


class PreparedOnlineTopKSelection:
    def __init__(
        self, values, host_visible, starts, chunk_rows, *,
        scan_width=None, update_keys=64, handoff_keys=64, queries_per_program=2, prefix_stage=0, handoff_stage=None,
    ):
        if update_keys not in (64, 256, 512):
            raise ValueError("TopK update size must be 64, 256, or 512")
        kernel = _qsa_online_scoped if update_keys == 64 else _qsa_online_grouped
        kernel_args = {} if update_keys == 64 else {"UPDATE_KEYS": update_keys}
        if handoff_keys not in (64, 256, 512):
            raise ValueError("Cube/Vector handoff size must be 64, 256, or 512")
        if handoff_keys != 64:
            if update_keys != 64:
                raise ValueError("Grouped handoff requires 64-key TopK updates")
            kernel = _qsa_online_handoff_grouped
            kernel_args = {"HANDOFF_KEYS": handoff_keys}
        if queries_per_program not in (2, 4):
            raise ValueError("Queries per Cube program must be two or four")
        if queries_per_program == 4:
            if update_keys != 64 or handoff_keys != 64:
                raise ValueError("Four-query probe requires 64-key updates and handoffs")
            kernel = _qsa_online_query4
        if prefix_stage not in (0, 1, 2, 3):
            raise ValueError("Prefix stage must be zero through three")
        if prefix_stage:
            if (update_keys, handoff_keys, queries_per_program) != (64, 64, 2):
                raise ValueError("Prefix probe requires the two-query 64-key kernel")
            kernel = _qsa_online_prefix
            kernel_args = {"PREFIX_STAGE": prefix_stage}
        if handoff_stage is not None:
            if handoff_stage not in (0, 1, 2):
                raise ValueError("Score handoff stage must be zero through two")
            if prefix_stage or (update_keys, handoff_keys, queries_per_program) != (64, 64, 2):
                raise ValueError("Score handoff probe requires the two-query 64-key kernel")
            kernel = _qsa_score_handoff_prefix
            kernel_args = {"HANDOFF_STAGE": handoff_stage}
        self.prefix_tiles = []
        self.handoff_tiles = []
        device = values["q"].device
        properties = triton.runtime.driver.active.utils.get_device_properties(device.index)
        cube_cores = int(properties["num_aicore"])
        assert cube_cores > 0
        self.chunk_rows = min(chunk_rows, queries_per_program * cube_cores)
        rows = values["q"].shape[0]
        self.output = torch.empty((rows, TOPK), device=device, dtype=torch.int32)
        self.chunks = []
        self.launches = []
        key = values["cache"].view(-1, 128)
        key_scale = values["ks"].view(-1)
        buffers = {}

        for begin, end in zip(starts[:-1], starts[1:], strict=True):
            request_width = int(host_visible[end - 1])
            if scan_width is not None:
                assert request_width <= scan_width <= values["row_table"].shape[1] * PAGE
                request_width = scan_width
            subtiles = max(1, triton.cdiv(request_width, CUBE_TILE))
            for first in range(begin, end, self.chunk_rows):
                last = min(first + self.chunk_rows, end)
                count = last - first
                cores = triton.cdiv(count, queries_per_program)
                packed_q = pack_mixed_queries(values["q"][first:last], queries_per_program)
                score_shape = (count, subtiles, 2 * CUBE_TILE) if prefix_stage else (count, TOPK)
                if handoff_stage is not None:
                    score_shape = (cores, subtiles, 8, CUBE_TILE)
                score_dtype = torch.int32 if handoff_stage in (0, 1) else torch.float32
                if count not in buffers:
                    buffers[count] = (
                        torch.empty((cores, 2, 16, handoff_keys), device=device, dtype=torch.int32),
                        torch.empty(score_shape, device=device, dtype=score_dtype),
                    )
                ring, scores = buffers[count]
                if prefix_stage:
                    self.prefix_tiles.append(scores)
                if handoff_stage is not None:
                    self.handoff_tiles.append(scores)
                self.chunks.append((first, count))
                self.launches.append(partial(
                    kernel[(cores,)],
                    packed_q, key, values["row_table"][begin], ring, scores,
                    values["qs"][first:last], key_scale, values["weights"][first:last],
                    values["visible"][first:last],
                    subtiles, 2, PAGE, CUBE_TILE, count, True, True,
                    ONLINE_TOP=TOPK, OnlineIndices=self.output[first:last],
                    PACKED_OUTPUT=False, KEEP_TILE_STORE=bool(prefix_stage), UB_SNAPSHOT=True, **kernel_args,
                    use_bytecode=True, disable_auto_cv_work_space_manage=True,
                    disable_auto_inject_block_sync=True, enable_auto_bind_sub_block=False,
                    multibuffer=False, sync_solver=False, unit_flag=False,
                ))

    def __call__(self, *, synchronize=False):
        for (first, count), launch in zip(self.chunks, self.launches, strict=True):
            if synchronize:
                print(f"online TopK first={first} rows={count}", flush=True)
            launch()
            if synchronize:
                torch.npu.synchronize()
        return self.output
