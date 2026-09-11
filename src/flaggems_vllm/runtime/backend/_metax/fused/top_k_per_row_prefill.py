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

"""top_k_per_row_prefill on MetaX: the generic operator, on its TLE path when
the installed FlagTree passes top_k_per_row_tle's self-test (0.640 -> 1.194 of
vLLM, kernel mode). Otherwise exactly the generic non-TLE path."""

from importlib import import_module

from flaggems_vllm.runtime.backend._metax.fused import top_k_per_row_tle as _tle

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    _tle.ensure_tle(logits.device)
    return _generic.top_k_per_row_prefill(
        logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
    )
