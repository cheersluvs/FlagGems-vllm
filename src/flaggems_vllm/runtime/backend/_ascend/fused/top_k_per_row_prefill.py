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

"""Ascend rewrites for top_k_per_row_prefill, kept out of the generic operator.

Every change here exists because of a defect in this backend, so none of it
belongs in code that NVIDIA and Moore Threads compile. The generic operator is
therefore left untouched and the pieces that need to differ are REBOUND onto its
module at import.

Rebinding is necessary rather than stylistic: a Triton jit function resolves the
jit functions it calls through its OWN module's globals. Defining a replacement
here would be ignored -- `_distribute_to_bins` lives in the generic module and
would keep seeing the generic `_extract_bin_idx`. Assigning to the module
attribute before anything is traced is what makes the substitution take.

This file is the feasibility slice: one function, `_extract_bin_idx`. If the
histogram comes out right on Ascend with the generic operator reverted, the
mechanism carries the rest of the chain (`_process_bins`, `_process_histogram_step`,
`_top_k_per_row_job`), which is another ~500 lines per operator.

The defect this one covers: **`tl.uint16 >>` is lowered as an ARITHMETIC shift**.
Every value with the top bit set -- under this mapping, every negative input --
comes out with a negative bin index, so its histogram store addresses outside the
buffer and is dropped. Measured: positives 9986/9986 correct, negatives 0/10014,
kernel bin range [-894, 922] against [487, 1570], histogram totals 9986 of 20000
with every missing bin above 1024, which is exactly where negatives land.
"""

from importlib import import_module

import triton
import triton.language as tl

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")

_convert_to_uint32 = _generic._convert_to_uint32


@triton.jit
def _extract_bin_idx(x, in_range, pattern, STEP: tl.constexpr):
    """The generic version, with the uint16 shift made explicitly logical.

    `mapped` is uint16, so `>> 5` must be a logical shift. Widening to int32 and
    masking to 16 bits before shifting forces that. The bit patterns are
    identical to the generic version's on a backend that shifts uint16 logically
    -- this is a repair, not a different algorithm.

    The int32 result type also matters here: the value is used as
    `s_histogram_ptr + bin_idx`, and pointer arithmetic promotes the offset to
    64 bits. An unsigned 32-bit source makes that a uint32_t_to_uint64_t cast,
    which this backend cannot lower ('hivm.hir.vcast' op currently don't support
    cast uint32_t_to_uint64_t_rintmode). The masks leave 10 or 11 bits, so
    0..2047 in every branch and int32 holds it exactly.
    """
    is_partial_match = in_range
    if STEP == 0:
        h = x.to(tl.float16)
        bits = h.to(tl.uint16, bitcast=True)
        sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
        sign_set = (bits & sign_mask) != 0
        inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
        mapped = tl.where(sign_set, bits, inv)
        bin_idx = (mapped.to(tl.int32) & 0xFFFF) >> 5
    else:
        bits = _convert_to_uint32(x)
        if STEP == 1:
            bin_idx = (bits >> 21).to(tl.int32)
        elif STEP == 2:
            bin_idx = ((bits >> 10) & 0x7FF).to(tl.int32)
            is_partial_match &= ((bits ^ pattern) >> 21) == 0
        elif STEP == 3:
            bin_idx = (bits & 0x3FF).to(tl.int32)
            is_partial_match &= ((bits ^ pattern) >> 10) == 0
    return bin_idx, is_partial_match


# Rebind BEFORE anything is traced. Every caller in the generic module resolves
# this name through that module's globals, so the assignment is what makes the
# replacement reach _distribute_to_bins and _process_bins.
_generic._extract_bin_idx = _extract_bin_idx


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """The generic host, unchanged. Only the rebound internals differ.

    Registering a function under this name is also what makes the suite's
    _OVERRIDE_ACTIVE guard run instead of skip on this backend.
    """
    return _generic.top_k_per_row_prefill(
        logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
    )
