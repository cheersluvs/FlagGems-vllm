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

"""Probe write-only UB output for the last local mrgsort."""

import ast
import copy
import re
import struct
from unittest.mock import patch

from audit_qsa_local_handoff_910b import replay_compilation
from diagnose_qsa_structural_mixed_compile_910b import CaptureCompiler
from qsa_paired_history_910b import direct_history_pack_source, sort_barrier_source

POISON_VALUE = -1234567.0


def _use_local32(call):
    assert ast.literal_eval(call.args[0]) == "mrgsort"
    assert ast.unparse(call.args[1]) == "local128"
    out = next(keyword for keyword in call.keywords if keyword.arg == "out")
    assert ast.unparse(out.value) == "tl.zeros((1024,), tl.float32)"
    out.value = ast.Name(id="local32", ctx=ast.Load())


def prepared_reuse_source(source_text):
    kernel = next(
        node for node in ast.parse(source_text).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "raw_history_prepared_paired_no_sort_barriers"
    )
    kernel.name = "raw_history_prepared_paired_reuse_local32"
    assignment = next(
        node for node in ast.walk(kernel)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "pairs"
    )
    _use_local32(assignment.value)
    return "\n" + ast.unparse(ast.fix_missing_locations(kernel)) + "\n"


def prepared_reuse_ir(kernel):
    from probe_qsa_raw_cube_chain_910b import record_launch_mode
    from qsa_paired_history_910b import sort_barrier_ir

    evidence = sort_barrier_ir(kernel, "no_sort32_barrier", sort32_boundary=True)
    if record_launch_mode(kernel, require_cube=False)["mix_mode"] == "aiv":
        return evidence
    sort32 = evidence["sort32_interval"][0].split(" = hivm.hir.custom", 1)[0]
    last_merge = evidence["merge_interval"][-1]
    if f"outs({sort32} : tensor<1024xf32>)" not in last_merge:
        raise RuntimeError("Final mrgsort destination differs from the sort32 output")
    return dict(**evidence, final_mrgsort_reuses_local32=True)


def source(*, reuse_local32=False, drop_handoff_barrier=False, direct_history_pack=False):
    common = sort_barrier_source(sort32_boundary=True)
    full = ast.parse(common).body[-1]
    assert full.name == "raw_history_prepared_paired_no_sort_barriers"
    first_constexpr = next(i for i, arg in enumerate(full.args.args) if arg.annotation)
    full.args.args.insert(first_constexpr, ast.arg(arg="PairTrace"))
    variants = []
    if direct_history_pack:
        common += direct_history_pack_source()
        labels = ("reuse_local32", "direct_history_pack")
    elif drop_handoff_barrier:
        labels = ("reuse_local32", "no_handoff_barrier")
    elif reuse_local32:
        labels = ("full", "reuse_local32")
    else:
        labels = ("full", "poison", "uninit")
    for label in labels:
        kernel = copy.deepcopy(full)
        kernel.name = "raw_final_mrgsort_" + label
        vector = kernel.body[1].body
        block = next(
            n for n in vector if isinstance(n, ast.For) and n.target.id == "vector_block"
        )
        query = next(
            n for n in block.body if isinstance(n, ast.For) and n.target.id == "update_query"
        )
        index = next(
            i for i, n in enumerate(query.body)
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "pairs"
        )
        call = query.body[index].value
        assert ast.literal_eval(call.args[0]) == "mrgsort"
        assert ast.unparse(query.body[index + 1]) == "tl.debug_barrier()"
        if label in ("reuse_local32", "no_handoff_barrier", "direct_history_pack"):
            _use_local32(call)
        elif label != "full":
            out = next(keyword for keyword in call.keywords if keyword.arg == "out")
            if label == "poison":
                expression = f"tl.full((1024,), {POISON_VALUE}, tl.float32)"
            else:
                vector.insert(
                    vector.index(block),
                    ast.parse(
                        "sort_out = tle.dsa.alloc([1024], dtype=tl.float32, "
                        "mem_addr_space=tle.dsa.ascend.UB)"
                    ).body[0],
                )
                expression = "tle.dsa.to_tensor(sort_out)"
            out.value = ast.parse(expression, mode="eval").body
        if label == "direct_history_pack":
            history_merge = next(
                node for node in ast.walk(query)
                if isinstance(node, ast.Call) and ast.unparse(node.func) == "_history_three_way"
            )
            history_merge.func.id = "_history_three_way_direct_pack"
        trace_index = index + 2
        if label == "no_handoff_barrier":
            del query.body[index + 1]
            trace_index -= 1
        query.body[trace_index:trace_index] = ast.parse(
            """
if TRACE:
    pair_word = tl.arange(0, 1024)
    trace_base = (update_row.to(tl.int64) * BLOCKS + vector_block) * 1024
    tl.store(PairTrace + trace_base + pair_word, pairs.to(tl.int32, bitcast=True),
             mask=update_row < ROWS)
"""
        ).body
        variants.append(ast.unparse(ast.fix_missing_locations(kernel)))
    return common + "\n" + "\n".join(variants) + "\n"


def check_output_init(text, *, inspect_history_input=False):
    active = False
    merge_lines = []
    for line in text.splitlines():
        if "IR Dump" in line:
            active = "IR Dump After" in line and "graphsyncsolver" in re.sub(
                r"[^a-z]", "", line.lower()
            )
        if active:
            merge_lines.append(line)
    first, last = [
        i for i, line in enumerate(merge_lines)
        if "hivm.hir.custom" in line and '"mrgsort"' in line
    ]
    sort32, = [
        i for i, line in enumerate(merge_lines[:first])
        if "hivm.hir.custom" in line and '"sort32"' in line
    ]
    original = re.search(r"outs\((%[\w.]+) :", merge_lines[sort32])
    dest = re.search(r"outs\((%[\w.]+) :", merge_lines[last])
    final_input = re.search(r"ins\((%[\w.]+),", merge_lines[last])
    if original is None or dest is None or final_input is None:
        raise ValueError("Cannot identify local-sort buffers in post-solver IR")
    out = dest[1]
    pointer_bases = {}
    for line in merge_lines[:last]:
        if pointer := re.search(r"(%[\w.]+) = hivm.hir.pointer_cast\((%[\w.]+)\)", line):
            pointer_bases[pointer[1]] = pointer[2]

    def shares_buffer(left, right):
        return left == right or (
            left in pointer_bases
            and right in pointer_bases
            and pointer_bases[left] == pointer_bases[right]
        )

    interval = merge_lines[first + 1 : last]
    writes = []
    for line in interval:
        destination = re.search(r"\bouts\((%[\w.]+)\s*:", line)
        if destination and shares_buffer(destination[1], out):
            writes.append(line.strip())
    copies = [line for line in writes if "hivm.hir.copy" in line or "memref.copy" in line]
    fills = [line for line in writes if "hivm.hir.vbrc" in line]
    copy_sources = {
        m[1] for line in copies
        if (m := re.search(r"ins\((%[\w.]+) :", line))
    }
    templates = [
        line.strip() for line in merge_lines[:last]
        if "hivm.hir.vbrc" in line
        and any(f"outs({item} :" in line for item in copy_sources)
    ]
    constant_defs = {}
    for line in merge_lines[:last]:
        if match := re.search(r"(%[\w.]+) = arith.constant (\S+) : f32\b", line):
            constant_defs[match[1]] = match[2]
    fill_values = []
    for line in [*fills, *templates]:
        match = re.search(r"vbrc ins\((%[\w.]+) : f32", line)
        if match and match[1] in constant_defs:
            fill_values.append(constant_defs[match[1]])
    handoff = {}
    for i, line in enumerate(merge_lines[last + 1 : last + 65], last + 1):
        if '"merge_exhaust_sort4"' in line:
            break
        operand = re.search(r"hivm\.hir\.copy ins\((%[\w.]+) :", line)
        if operand and shares_buffer(operand[1], out):
            sync = [
                item.strip() for item in merge_lines[last + 1 : i]
                if "hivm.hir.pipe_barrier" in item
                or "hivm.hir.set_flag" in item or "hivm.hir.wait_flag" in item
            ]
            handoff = dict(
                first_output_copy=line.strip(),
                pipe_all=sum("PIPE_ALL" in item for item in sync),
                pipe_v=sum("pipe_barrier[<PIPE_V>]" in item for item in sync),
                sync=sync,
            )
            break
    evidence = dict(
        destination=out,
        handoff=handoff,
        sort32_destination=original[1],
        final_input=final_input[1],
        reuses_sort32_buffer=shares_buffer(original[1], out),
        aliases_final_input=shares_buffer(final_input[1], out),
        output_call=merge_lines[last].strip(),
        copies=copies,
        fills=fills,
        other_writes=[line for line in writes if line not in copies and line not in fills],
        templates=templates,
        fill_values=fill_values,
        interval=[line.strip() for line in interval],
    )
    if inspect_history_input:
        evidence["history_input"] = _history_input_ir(merge_lines, last)
    return evidence


def _history_input_ir(lines, last_sort):
    merge = next(
        i for i in range(last_sort + 1, len(lines))
        if "hivm.hir.custom" in lines[i] and '"merge_exhaust_sort4"' in lines[i]
    )
    match = re.search(r'\bins\((%[\w.]+), (%[\w.]+),', lines[merge])
    if match is None or not re.search(
        rf"{re.escape(match[2])} = arith.constant 3 : i32\b",
        "\n".join(lines[:merge]),
    ):
        raise ValueError("Cannot identify the three-way merge input in post-solver IR")
    packed = match[1]
    interval = lines[last_sort + 1 : merge]
    addresses = {
        m[1]: m[2] for line in lines[:merge]
        if (m := re.search(r"(%[\w.]+) = hivm.hir.pointer_cast\((%[\w.]+)\)", line))
    }
    aliases = {packed}
    if packed in addresses:
        aliases.update(name for name, address in addresses.items() if address == addresses[packed])
    targets = "|".join(re.escape(alias) for alias in sorted(aliases))
    full_write = re.compile(
        rf"\bouts\((?:{targets})\s*:\s*memref<4096xf32"
        rf"|\bmemref.copy\s+%[\w.]+,\s*(?:{targets})\s*:"
    )
    whole_writes = [line.strip() for line in interval if full_write.search(line)]
    subview = re.compile(rf"(%[\w.]+) = memref.subview (?:{targets})\[")
    subviews = [m[1] for line in interval if (m := subview.search(line))]
    return dict(
        input_buffer=packed,
        full_width_writes=whole_writes,
        subview_writes=[
            line.strip() for line in interval
            if any(f"outs({name} :" in line for name in subviews)
        ],
        merge_call=lines[merge].strip(),
        interval=[line.strip() for line in interval],
    )


def compile_capture(compiler, report, label, warmup):
    capture = CaptureCompiler(report, label)
    with patch.object(compiler, "subprocess", capture):
        compiled = warmup()
    if len(capture.calls) != 1 or capture.calls[0].get("returncode") != 0:
        raise RuntimeError(f"Expected one successful final-mrgsort compile for {label}")
    return compiled, capture.calls[0]


def replay_output_init(compiler, report, report_label, call, *, inspect_history_input=False):
    _, env = compiler._get_npucompiler_path()
    replay = replay_compilation(report, report, call, report_label, 180, env=env)
    if replay["status"] != "captured":
        raise RuntimeError(f"Post-solver IR unavailable for {report_label}")
    streams = [
        (report / replay[name]["artifact"]).read_text(errors="replace")
        for name in ("stdout", "stderr")
        if replay[name]["summary"]["after_solver_sections"]
    ]
    evidence = check_output_init(
        "\n".join(streams), inspect_history_input=inspect_history_input
    )
    return dict(replay=replay, **evidence)


def gate_output_init(evidence, label):
    copies, fills = evidence["copies"], evidence["fills"]
    if evidence["other_writes"]:
        raise RuntimeError(f"Unexpected writes to final mrgsort output in {label}")
    if label in ("uninit", "reuse_local32", "no_handoff_barrier", "direct_history_pack"):
        if copies or fills:
            raise RuntimeError("Final mrgsort candidate still initializes its output")
        if evidence["aliases_final_input"]:
            raise RuntimeError("Final mrgsort output overlaps the live input buffer")
        return
    if label == "full":
        if len(copies) != 1 or "memref<1024xf32" not in copies[0] or fills:
            raise RuntimeError("Expected one 4 KiB final mrgsort initialization copy")
        return
    # Verify that the validation binary actually initializes the destination.
    poison_ir_values = []
    for value in evidence["fill_values"]:
        if value.startswith("0x"):
            poison_ir_values.append(struct.unpack("!f", bytes.fromhex(value[2:]))[0])
        else:
            poison_ir_values.append(float(value))
    if POISON_VALUE not in poison_ir_values:
        raise RuntimeError(f"Cannot confirm poison initialization: {evidence['fill_values']}")



def gate_handoff_barrier(evidence, label):
    if not evidence["reuses_sort32_buffer"] or not evidence["handoff"]:
        raise RuntimeError(f"Missing local32 handoff evidence for {label}")
    return evidence["handoff"]["pipe_all"] == int(label == "reuse_local32")
