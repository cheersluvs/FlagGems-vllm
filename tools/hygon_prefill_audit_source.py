"""Pure-stdlib source construction for the Hygon prefill audit.

No accelerator imports or execution. Each arm changes one mechanism only.
"""

import ast
import hashlib
import re
from pathlib import Path

GENERIC = "src/flaggems_vllm/ops/top_k_per_row_prefill.py"
OVERRIDE = "src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py"


def replace_once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected one source block, got {count}: {old[:80]!r}")
    return source.replace(old, new, 1)


def function_text(source, name):
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "\n".join(source.splitlines()[first - 1 : node.end_lineno])


def control_source(root, dense):
    generic = (Path(root) / GENERIC).read_text()
    override = (Path(root) / OVERRIDE).read_text()
    constants = {}
    for node in ast.parse(override).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.startswith("_ONESCAN_"):
                if isinstance(node.value, ast.Constant) and isinstance(
                    node.value.value, str
                ):
                    constants[target.id] = node.value.value
    for part in ("CLEAR", "SCAN"):
        generic = replace_once(
            generic,
            constants[f"_ONESCAN_{part}_OLD"],
            constants[f"_ONESCAN_{part}_NEW"],
        )
    if dense:
        for name in ("_alloc_slots", "_process_bins_slotscan"):
            generic += "\n\n" + function_text(override, name) + "\n"
        generic += "\n_process_bins = _process_bins_slotscan\n"
    return generic


SCALAR_RANK = """                for j in tl.range(0, final_cnt):
                    logit_j = tl.load(s_final_logits_ptr + j)
                    better = (logit_i < logit_j) | ((logit_i == logit_j) & (pos < j))
                    out_rank = out_rank + (valid & better).to(tl.int32)
"""


def tiled_rank(source, width):
    if width not in (8, 16):
        raise ValueError("Only the two frozen rank tiles are supported")
    new = f"""                if final_cnt <= 256:
                    j_lane = tl.arange(0, {width})
                    for j_block in tl.range(0, tl.cdiv(final_cnt, {width})):
                        js = j_block * {width} + j_lane
                        j_valid = js < final_cnt
                        xj = tl.load(s_final_logits_ptr + js, mask=j_valid, other=0.0)
                        better = (logit_i[:, None] < xj[None, :]) | (
                            (logit_i[:, None] == xj[None, :]) & (pos[:, None] < js[None, :])
                        )
                        pairs = valid[:, None] & j_valid[None, :] & better
                        out_rank += tl.sum(pairs.to(tl.int32), axis=1)
                else:
""" + "".join(
        "    " + line + "\n" for line in SCALAR_RANK.splitlines()
    )
    return replace_once(source, SCALAR_RANK, new)


def register_carry(source):
    old_helper = function_text(source, "_process_bins_slotscan")
    helper = replace_once(
        old_helper, "    STEP: tl.constexpr,", "    slot_base,\n    STEP: tl.constexpr,"
    )
    helper = replace_once(
        helper,
        "    out_pos_lt = _alloc_slots(found_topk_values_ptrs, take_lt)",
        """    take_int = take_lt.to(tl.int32)
    flat_take = tl.reshape(take_int, (take_int.numel,))
    offsets = tl.cumsum(flat_take, axis=0) - flat_take
    out_pos_lt = slot_base + tl.reshape(offsets, take_int.shape)
    slot_base += tl.sum(flat_take, axis=0)""",
    )
    helper += "\n    return slot_base"
    source = replace_once(source, old_helper, helper)

    old_step = function_text(source, "_process_histogram_step")
    step = replace_once(
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
        raise ValueError(
            f"Expected seven aligned/head/tail/strided collection sites: {len(calls)}"
        )
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
    step = replace_once(
        step,
        "    tl.debug_barrier()\n    return final_bin_size > NUM_FINAL_ITEMS, logit_pattern, threshold_bin_idx",
        "    tl.store(s_found_topk_values_ptr, slot_base)\n"
        "    tl.debug_barrier()\n"
        "    return final_bin_size > NUM_FINAL_ITEMS, logit_pattern, threshold_bin_idx",
    )
    return replace_once(source, old_step, step)


def diagnostic_source(source):
    # These scalar outputs are unused by the shipped one-scan step. Keep this
    # instrumentation in a SEPARATE module and never time it.
    return replace_once(
        source,
        "    use_final = final_bin_size <= NUM_FINAL_ITEMS\n",
        "    tl.store(s_threshold_bin_idx_ptr, STEP)\n"
        "    tl.store(s_final_bin_size_ptr, final_bin_size)\n"
        "    use_final = final_bin_size <= NUM_FINAL_ITEMS\n",
    )


def build(root, dense, arm="control", diagnostic=False):
    source = control_source(root, dense)
    if arm in ("rank8", "rank16", "rank8_carry"):
        source = tiled_rank(source, 8 if arm == "rank8_carry" else int(arm[4:]))
    if arm in ("carry", "rank8_carry"):
        if dense:
            source = register_carry(source)
    elif arm not in ("control", "rank8", "rank16"):
        raise ValueError(f"Unknown arm {arm}")
    if diagnostic:
        source = diagnostic_source(source)
    compile(source, f"<hygon-prefill-{arm}>", "exec")
    return source


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()
