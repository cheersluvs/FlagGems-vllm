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

"""Build the Hygon dense prefill register-carry experiment.

The generated module owns both its collection helper and the histogram step.
Triton resolves JIT globals at compile time, so patching only a helper in an
already loaded module would not select the intended implementation.
"""

import ast
import re


def _replace_once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected one source block, got {count}: {old[:80]!r}")
    return source.replace(old, new, 1)


def _function_text(source, name):
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "\n".join(source.splitlines()[first - 1 : node.end_lineno])


def build_carry_source(generic_source, override_source):
    """Return the exact audited dense source with a carried slot counter.

    Raises ValueError on source drift; callers must keep the shipped dense
    module as a fallback. No production kernel is changed by this function.
    """
    source = generic_source
    for name in ("_alloc_slots", "_process_bins_slotscan"):
        source += "\n\n" + _function_text(override_source, name) + "\n"
    source += "\n_process_bins = _process_bins_slotscan\n"

    old_helper = _function_text(source, "_process_bins_slotscan")
    helper = _replace_once(
        old_helper, "    STEP: tl.constexpr,", "    slot_base,\n    STEP: tl.constexpr,"
    )
    helper = _replace_once(
        helper,
        "    out_pos_lt = _alloc_slots(found_topk_values_ptrs, take_lt)",
        """    take_int = take_lt.to(tl.int32)
    flat_take = tl.reshape(take_int, (take_int.numel,))
    offsets = tl.cumsum(flat_take, axis=0) - flat_take
    out_pos_lt = slot_base + tl.reshape(offsets, take_int.shape)
    slot_base += tl.sum(flat_take, axis=0)""",
    )
    helper += "\n    return slot_base"
    source = _replace_once(source, old_helper, helper)

    old_step = _function_text(source, "_process_histogram_step")
    step = _replace_once(
        old_step,
        "    final_cnt_ptrs = s_final_cnt_ptr + zeros\n",
        "    final_cnt_ptrs = s_final_cnt_ptr + zeros\n"
        "    slot_base = tl.load(s_found_topk_values_ptr)\n",
    )
    calls = [
        n
        for n in ast.walk(ast.parse(step))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_process_bins"
    ]
    if len(calls) != 7:
        raise ValueError(f"Expected seven collection sites, got {len(calls)}")
    replacements = []
    lines = step.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    for node in calls:
        old = ast.get_source_segment(step, node)
        new, count = re.subn(
            r"(?m)^(\s*)STEP=STEP,", r"\1slot_base,\n\1STEP=STEP,", old
        )
        if count != 1:
            raise ValueError("Missing STEP keyword in collection call")
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        replacements.append((start, end, "slot_base = " + new))
    for start, end, new in sorted(replacements, reverse=True):
        step = step[:start] + new + step[end:]
    step = _replace_once(
        step,
        "    tl.debug_barrier()\n    return final_bin_size > NUM_FINAL_ITEMS, logit_pattern, threshold_bin_idx",
        "    tl.store(s_found_topk_values_ptr, slot_base)\n"
        "    tl.debug_barrier()\n"
        "    return final_bin_size > NUM_FINAL_ITEMS, logit_pattern, threshold_bin_idx",
    )
    source = _replace_once(source, old_step, step)
    compile(source, "<hygon-prefill-carry>", "exec")
    return source
