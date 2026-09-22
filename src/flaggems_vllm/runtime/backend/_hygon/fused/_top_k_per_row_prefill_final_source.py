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

"""Construct the Hygon-only final-network VEC2 module; fail on source drift."""

import ast


def _function_text(source, name):
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "\n".join(source.splitlines()[first - 1 : node.end_lineno])


def build_final_source(source):
    """Replace only the non-TLE final rank loop; retain exact scalar overflow."""
    job = _function_text(source, "_top_k_per_row_job")
    start = job.index("            sort_chunks = tl.cdiv(final_cnt, BLOCK_SIZE)")
    end = job.index("            tl.debug_barrier()", start)
    original = job[start:end]
    branch = (
        "            remain = tl.minimum(tl.maximum(TOPK - base_idx, 0), final_cnt)\n"
    )
    for i, cap in enumerate((64, 128, 256)):
        branch += f"            {'if' if i == 0 else 'elif'} final_cnt <= {cap}:\n"
        branch += (
            "                _hygon_final_network(\n"
            "                    s_final_logits_ptr, s_histogram_ptr,\n"
            "                    s_out_indices_ptr, final_cnt, base_idx, remain,\n"
            f"                    CAP={cap},\n"
            "                )\n"
        )
    branch += "            else:\n"
    branch += "".join("    " + line + "\n" for line in original.splitlines())
    changed = job[:start] + branch + job[end:]
    if source.count(job) != 1:
        raise ValueError("Final selection source drift")
    source = source.replace(job, changed, 1)
    source += (
        "\nfrom flaggems_vllm.runtime.backend._hygon.fused."
        "_top_k_per_row_prefill_final_network import "
        "final_network as _hygon_final_network\n"
    )
    compile(source, "<hygon-prefill-final-network>", "exec")
    return source
