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

"""Capture the exact mixed-kernel compiler inputs for four fixed variants."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch


class CaptureCompiler:
    def __init__(self, report, label):
        self.report, self.label, self.calls = Path(report), label, []

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def run(self, command, *args, **kwargs):
        if not isinstance(command, (list, tuple)) or not command or Path(command[0]).name != "bishengir-compile":
            return subprocess.run(command, *args, **kwargs)
        record = dict(command=[str(item) for item in command])
        self.calls.append(record)
        stem = f"{self.label}-compiler-{len(self.calls)}"
        for item in command[1:]:
            source = Path(str(item))
            if source.suffix == ".mlir" and source.is_file():
                name = stem + "-input.mlir.txt"
                (self.report / name).write_bytes(source.read_bytes())
                record["input_ir"] = name
        try:
            result = subprocess.run(command, *args, **kwargs)
        except subprocess.CalledProcessError as exc:
            self.save(stem, record, exc)
            raise
        self.save(stem, record, result)
        return result

    def save(self, stem, record, result):
        record["returncode"] = result.returncode
        for field in ("stdout", "stderr"):
            value = getattr(result, field, None)
            if value:
                data = value.encode() if isinstance(value, str) else value
                name = f"{stem}-{field}.txt"
                limit = 8 * 1024 * 1024
                (self.report / name).write_bytes(data[:limit])
                record[field] = dict(file=name, bytes=len(data), truncated=len(data) > limit)


def _capture_variant(adapter, report, label, trace, stage):
    from triton.backends.ascend import compiler

    capture = CaptureCompiler(report, label)
    entry = dict(slots=adapter.slots, trace=trace, stage=stage,
                 structure=adapter.structure, compile_only=True, status="failed")
    try:
        with patch.object(compiler, "subprocess", capture):
            kernels = adapter.compile_variant(trace, stage)
        entry.update(status="compiled", metadata=[str(kernel.metadata) for kernel in kernels])
        for index, kernel in enumerate(kernels):
            for kind, text in kernel.asm.items():
                if isinstance(text, str):
                    (report / f"{label}-{index}-{kind}.txt").write_text(text)
    except Exception as exc:
        entry["error"] = repr(exc)
    entry["compiler_calls"] = capture.calls
    print(f"QSA MIXED COMPILE {label} status={entry['status']}", flush=True)
    return entry


def compile_matrix(values, width, owners, report, case):
    from qsa_cann_mixed_910b_trial import PreparedMixedOuter

    report = Path(report)
    records = []
    prefix_probe = os.environ.get("QSA_MIXED_COMPILE_PREFIX") == "1"
    for slots in ((1,) if prefix_probe else (1, 2)):
        adapter = PreparedMixedOuter(values, width, slots=slots, request_ids=owners)
        variants = [(True, stage) for stage in (-1, 0, 1, 2)] if prefix_probe else [(trace, 2) for trace in (False, True)]
        for trace, stage in variants:
            label = f"{case}-slots{slots}-trace{int(trace)}-stage{stage}"
            records.append(_capture_variant(adapter, report, label, trace, stage))
            (report / f"{case}-mixed-compile-matrix.json").write_text(json.dumps(records, indent=2))
    return records


def compile_structure_matrix(values, width, owners, report, case):
    """Isolate handoff structure; execute no reduced prefix kernels."""
    from qsa_cann_mixed_910b_trial import PreparedMixedOuter

    report = Path(report)
    path = report / f"{case}-mixed-structure-matrix.json"
    frozen_path = report / "mixed-structure-selection.json"
    result = dict(status="running", raw=[], full=[], selected=None,
                  selection_rule="first candidate compiling all four full variants; no timing selection")

    def save():
        path.write_text(json.dumps(result, indent=2, allow_nan=False))

    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        candidates = [frozen["structure"]]
        result["frozen_from_case"] = frozen["case"]
    else:
        candidates = []
        for plain, tiled in ((False, False), (True, False), (False, True), (True, True)):
            structure = dict(TABLE_PLAIN=plain, RING_TILE_MAJOR=tiled)
            adapter = PreparedMixedOuter(values, width, slots=1, request_ids=owners, structure=structure)
            label = f"{case}-plain{int(plain)}-tiled{int(tiled)}-raw"
            entry = _capture_variant(adapter, report, label, True, -1)
            result["raw"].append(entry)
            save()
            if entry["status"] == "compiled":
                candidates.append(structure)
    for structure in candidates:
        complete = []
        for slots in (1, 2):
            adapter = PreparedMixedOuter(values, width, slots=slots, request_ids=owners, structure=structure)
            for trace in (False, True):
                label = (f"{case}-plain{int(structure['TABLE_PLAIN'])}"
                         f"-tiled{int(structure['RING_TILE_MAJOR'])}-slots{slots}-trace{int(trace)}-full")
                complete.append(_capture_variant(adapter, report, label, trace, 2))
                result["full"].append(complete[-1])
                save()
        if all(entry["status"] == "compiled" for entry in complete):
            result.update(status="compiled", selected=structure)
            save()
            if not frozen_path.exists():
                frozen_path.write_text(json.dumps(dict(case=case, structure=structure), indent=2))
            print("QSA MIXED STRUCTURE SELECTED " + json.dumps(structure), flush=True)
            return result
    result["status"] = "failed"
    save()
    raise RuntimeError("No mixed handoff structure compiled all four full variants; see structure matrix and compiler IR")


def compile_workspace_matrix(values, width, owners, report, case):
    """Compile the fixed compact-workspace algorithm before device execution."""
    from qsa_cann_mixed_910b_trial import PreparedMixedOuter

    report = Path(report)
    structure = dict(TABLE_PLAIN=True, RING_TILE_MAJOR=False)
    result = dict(status="running", structure=structure, full=[], prefixes=[],
                  scratch_words=8192, active_lists=2, state_capacity=2048)
    path = report / f"{case}-mixed-workspace-matrix.json"
    for slots in (1, 2):
        adapter = PreparedMixedOuter(values, width, slots=slots, request_ids=owners, structure=structure)
        for trace in (False, True):
            label = f"{case}-workspace-slots{slots}-trace{int(trace)}-full"
            result["full"].append(_capture_variant(adapter, report, label, trace, 2))
            path.write_text(json.dumps(result, indent=2))
    if not all(entry["status"] == "compiled" for entry in result["full"]):
        adapter = PreparedMixedOuter(values, width, slots=1, request_ids=owners, structure=structure)
        for stage in (0, 1):
            label = f"{case}-workspace-slots1-trace1-stage{stage}"
            result["prefixes"].append(_capture_variant(adapter, report, label, True, stage))
            path.write_text(json.dumps(result, indent=2))
        result["status"] = "failed"
        path.write_text(json.dumps(result, indent=2))
        raise RuntimeError("Mixed compact workspace compilation failed; see full/prefix IR and resource logs")
    result["status"] = "compiled"
    path.write_text(json.dumps(result, indent=2))
    return result
