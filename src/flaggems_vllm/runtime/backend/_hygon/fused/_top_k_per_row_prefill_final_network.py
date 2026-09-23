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

"""Exact, small candidate final selection for the Hygon prefill dense path."""

import triton
import triton.language as tl


@triton.jit
def _ordered_key(x):
    # Numeric equality treats both signs of zero as tied.
    x = tl.where(x == 0, 0.0, x)
    bits = x.to(tl.uint32, bitcast=True)
    return bits ^ tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)


@triton.jit
def final_network(Values, Indices, Output, count, base, remain, CAP: tl.constexpr):
    p = tl.arange(0, CAP)
    valid = p < count
    x = tl.load(Values + p, mask=valid, other=0.0)
    # Preserve the original selector's later-position tie rule.
    code = (_ordered_key(x).to(tl.uint64) << 32) | p.to(tl.uint64)
    code = tl.where(valid, code, 0).to(tl.uint64)
    if remain == count:
        direct_idx = tl.load(Indices + p, mask=valid, other=0)
        tl.store(Output + base + p, direct_idx, mask=valid)
    elif remain == 1:
        best = tl.max(code, 0)
        best_pos = (best & 0xFFFFFFFF).to(tl.int32)
        best_idx = tl.load(Indices + best_pos)
        tl.store(Output + base, best_idx)
    elif remain > 0:
        ranked = tl.sort(code, descending=True)
        sorted_pos = (ranked & 0xFFFFFFFF).to(tl.int32)
        sorted_idx = tl.load(Indices + sorted_pos, mask=p < remain, other=0)
        tl.store(Output + base + p, sorted_idx, mask=p < remain)
