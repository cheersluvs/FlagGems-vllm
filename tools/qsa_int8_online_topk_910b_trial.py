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

from probe_qsa_int8_mixed_pipeline_910b import CUBE_TILE, PAGE, TOPK, _qsa_online_scoped
from qsa_int8_fused_binary_910b_trial import pack_mixed_queries


class PreparedOnlineTopKSelection:
    def __init__(self, values, host_visible, starts, chunk_rows, *, scan_width=None):
        device = values["q"].device
        properties = triton.runtime.driver.active.utils.get_device_properties(device.index)
        cube_cores = int(properties["num_aicore"])
        assert cube_cores > 0
        self.chunk_rows = min(chunk_rows, 2 * cube_cores)
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
                cores = triton.cdiv(count, 2)
                packed_q = pack_mixed_queries(values["q"][first:last])
                if count not in buffers:
                    buffers[count] = (
                        torch.empty((cores, 2, 16, CUBE_TILE), device=device, dtype=torch.int32),
                        torch.empty((count, TOPK), device=device, dtype=torch.float32),
                    )
                ring, scores = buffers[count]
                self.chunks.append((first, count))
                self.launches.append(partial(
                    _qsa_online_scoped[(cores,)],
                    packed_q, key, values["row_table"][begin], ring, scores,
                    values["qs"][first:last], key_scale, values["weights"][first:last],
                    values["visible"][first:last],
                    subtiles, 2, PAGE, CUBE_TILE, count, True, True,
                    ONLINE_TOP=TOPK, OnlineIndices=self.output[first:last],
                    PACKED_OUTPUT=False, KEEP_TILE_STORE=False, UB_SNAPSHOT=True,
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
