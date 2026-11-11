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


def compile_matrix(values, width, owners, report, case):
    from triton.backends.ascend import compiler
    from qsa_cann_mixed_910b_trial import PreparedMixedOuter

    report = Path(report)
    records = []
    prefix_probe = os.environ.get("QSA_MIXED_COMPILE_PREFIX") == "1"
    for slots in ((1,) if prefix_probe else (1, 2)):
        adapter = PreparedMixedOuter(values, width, slots=slots, request_ids=owners)
        variants = [(True, stage) for stage in (-1, 0, 1, 2)] if prefix_probe else [(trace, 2) for trace in (False, True)]
        for trace, stage in variants:
            label = f"{case}-slots{slots}-trace{int(trace)}-stage{stage}"
            capture = CaptureCompiler(report, label)
            entry = dict(slots=slots, trace=trace, stage=stage, compile_only=True, status="failed")
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
            records.append(entry)
            (report / f"{case}-mixed-compile-matrix.json").write_text(json.dumps(records, indent=2))
            print(f"QSA MIXED COMPILE MATRIX slots={slots} trace={trace} stage={stage} status={entry['status']}", flush=True)
    return records
