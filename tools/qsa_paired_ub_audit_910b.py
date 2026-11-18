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

"""Inspect the installed backend lowering of the paired Top512 handoff."""

import json
import re
from collections import Counter
from unittest.mock import patch

from audit_qsa_local_handoff_910b import replay_compilation
from diagnose_qsa_structural_mixed_compile_910b import CaptureCompiler
from probe_qsa_raw_cube_chain_910b import record_launch_mode, save_asm
from qsa_paired_history_910b import sort_barrier_ir
from qsa_raw_score_chunks_910b import audit_binary, decoded_instructions

LOCAL_SORT = re.compile(r"\b(?:custom_)?mrgsort\b")
HISTORY = re.compile(r"\b(?:custom_)?merge_exhaust_sort4\b")
BRANCH = re.compile(r"\bscf\.(?:if|for|yield)\b")
LOCAL_OP = re.compile(
    r"\b(?:hivm\.hir\.(?:[\w.]*copy[\w.]*|ub2ub|[\w.]*flag[\w.]*|"
    r"[\w.]*barrier[\w.]*|sync_block[\w.]*)|"
    r"memref\.(?:alloc|copy|subview|reinterpret_cast))\b"
)


def trace_handoff(text, *, post_solver=False):
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    lines = text.splitlines()
    active = not post_solver
    sort, merge = [], []
    for i, line in enumerate(lines):
        if post_solver and "IR Dump" in line:
            active = "IR Dump After" in line and "graphsyncsolver" in re.sub(
                r"[^a-z]", "", line.lower()
            )
        if active and LOCAL_SORT.search(line):
            sort.append(i)
        if active and HISTORY.search(line):
            merge.append(i)
    # Require both anchors when naming a region. The full dump remains
    # available when the backend changes or inlines the call names.
    for first_merge in merge:
        preceding = [i for i in sort if i < first_merge]
        if len(preceding) < 2:
            continue
        last_sort = preceding[-1]
        if first_merge - last_sort > 180:
            continue
        end = next((i for i in sort if i > first_merge), len(lines))
        end = min(end, first_merge + 48)
        matches = []
        counts = Counter()
        for i in range(last_sort, end):
            line = lines[i]
            items = LOCAL_OP.findall(line)
            counts.update(items)
            if (
                i == last_sort
                or HISTORY.search(line)
                or items
                or BRANCH.search(line)
                or "tensor.insert_slice" in line
                or "tensor.extract_slice" in line
            ):
                matches.append(dict(line=i + 1, text=line.strip()))
        return dict(
            status="anchors_found",
            final_local_sort_line=last_sort + 1,
            first_history_merge_line=first_merge + 1,
            operations=dict(counts),
            lines=matches,
        )
    return dict(
        status="anchors_unresolved", local_sort_matches=len(sort), merge_matches=len(merge)
    )


def run_audit(torch, device, module, compiler, report):
    rows, width, blocks = 112, 29696, 58
    groups = (rows + 15) // 16
    slots = 2
    tiles = blocks * 8
    kernel_name = "raw_history_prepared_paired_no_sort_barriers"
    kernel = getattr(module, kernel_name)
    layout = (
        ((groups, 8192), torch.int8),
        ((tiles, 8192), torch.int8),
        ((groups, 1024), torch.float16),
        ((blocks * 512,), torch.float16),
        ((rows,), torch.int32),
        ((groups * slots * 8192,), torch.float32),
        ((rows, 512), torch.int32),
        ((rows, 512), torch.float32),
        ((rows, width), torch.float32),
    )
    operands = [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in layout]
    label = "paired-ub-long"
    capture = CaptureCompiler(report, label)
    path = report / "paired-ub-audit.json"
    entry = dict(
        status="running",
        rows=rows,
        width=width,
        blocks=blocks,
        slots=slots,
        trace=False,
        kernel=kernel_name,
        device_execution=False,
    )

    def save():
        path.write_text(json.dumps(entry, indent=2))

    save()
    try:
        with patch.object(compiler, "subprocess", capture):
            compiled = kernel.warmup(
                *operands,
                WIDTH=width,
                ROWS=rows,
                BLOCKS=blocks,
                SLOTS=slots,
                TRACE=False,
                grid=(groups,),
                num_warps=4,
            )
        entry["launch"] = record_launch_mode(compiled)
        if entry["launch"]["mix_mode"] != "mix":
            raise RuntimeError(f"Expected the mixed kernel: {entry['launch']}")
        entry["artifacts"] = save_asm(compiled, report, label)
        entry["pre_solver"] = trace_handoff(compiled.asm["bcmlir"])
        entry["sort_barriers"] = sort_barrier_ir(
            compiled, "no_sort32_barrier", sort32_boundary=True
        )["explicit_sort_barriers"]
        if len(capture.calls) != 1 or capture.calls[0].get("returncode") != 0:
            raise RuntimeError(
                f"Expected one successful compiler call, got {len(capture.calls)}"
            )
        _, compiler_env = compiler._get_npucompiler_path()
        entry["replay"] = replay_compilation(
            report, report, capture.calls[0], label, 180, env=compiler_env
        )
        post = entry["replay"]
        entry["post_solver"] = {}
        for stream in ("stdout", "stderr"):
            artifact = post[stream]
            if artifact["summary"]["after_solver_sections"]:
                content = (report / artifact["artifact"]).read_text(errors="replace")
                entry["post_solver"][stream] = trace_handoff(content, post_solver=True)
        save()
        try:
            binary = audit_binary(compiled, report, label)
            if "file" in binary:
                disassembly = (report / binary["file"]).read_text(errors="replace")
                binary["sort_copy_sync_instructions"] = [
                    line
                    for mnemonic, line in decoded_instructions(disassembly)
                    if re.search(r"sort|copy|mov|barrier|flag", mnemonic)
                ][:160]
            entry["device_binary"] = binary
        except (OSError, KeyError, ValueError) as exc:
            entry["device_binary"] = dict(status="unavailable", error=repr(exc))
        entry["status"] = "captured" if post["status"] == "captured" else "failed"
    except Exception:
        entry["status"] = "failed"
        raise
    finally:
        entry["compiler_calls"] = capture.calls
        save()

    summary = dict(
        status=entry["status"],
        pre_solver=entry["pre_solver"].get("operations"),
        post_solver={
            name: dict(status=stage["status"], operations=stage.get("operations"))
            for name, stage in entry["post_solver"].items()
        },
        device_binary=entry["device_binary"]["status"],
    )
    print("PAIRED_UB_AUDIT", json.dumps(summary), flush=True)
    return entry
