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

"""Duplicate FP16 key scales into 16 columns for layout diagnostics."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_pack_key_scale(KS, Expanded, TOKENS: tl.constexpr, BLOCK: tl.constexpr):
    for start in range(
        tl.program_id(0) * BLOCK, TOKENS * 16, tl.num_programs(0) * BLOCK
    ):
        offsets = start + tl.arange(0, BLOCK)
        values = tl.load(KS + offsets // 16, mask=offsets < TOKENS * 16, other=0)
        tl.store(Expanded + offsets, values, mask=offsets < TOKENS * 16)


def pack_key_scale(source, target):
    if (
        source.ndim != 3
        or source.shape[2] != 1
        or source.dtype != torch.float16
        or source.device.type != "npu"
        or target.shape != (*source.shape[:2], 16)
        or not source.is_contiguous()
        or not target.is_contiguous()
        or source.device != target.device
        or source.dtype != target.dtype
    ):
        raise ValueError("Matching contiguous FP16 scale buffers required")
    tokens = source.numel()
    if tokens == 0:
        return None
    return _qsa_pack_key_scale[(min(20, triton.cdiv(tokens * 16, 1024)),)](
        source, target, TOKENS=tokens, BLOCK=1024, num_warps=4
    )
