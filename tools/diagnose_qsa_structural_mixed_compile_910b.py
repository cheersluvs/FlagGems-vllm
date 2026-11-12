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

import hashlib
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
        command = record["command"]
        if os.environ.get("QSA_CAPTURE_COMPILER_OUTPUTS") == "1" and "-o" in command:
            root = Path(command[command.index("-o") + 1]).parent
            remaining = 2 * 1024 * 1024
            artifacts = []
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.name == "kernel.mlir":
                    continue
                if path.suffix not in (".ll", ".mlir", ".s", ".cce", ".json", ".o", ".so", ".bin"):
                    continue
                data = path.read_bytes()
                entry = dict(path=str(path.relative_to(root)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                if path.suffix in (".ll", ".mlir", ".s", ".cce", ".json"):
                    limit = min(remaining, 512 * 1024)
                    if limit:
                        name = stem + "-output-" + str(path.relative_to(root)).replace("/", "_") + ".txt"
                        (self.report / name).write_bytes(data[:limit])
                        entry.update(file=name, truncated=len(data) > limit)
                        remaining -= min(len(data), limit)
                    else:
                        entry["not_exported"] = "per-call text budget exhausted"
                artifacts.append(entry)
            record["output_artifacts"] = artifacts
            record["output_capture_limit_bytes"] = 2 * 1024 * 1024
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


def probe_merge_representations(values, report, case):
    """Compile and validate the same two-list merge with three input forms."""
    import torch
    from triton.backends.ascend import compiler
    from qsa_cann_mixed_910b_trial import _merge_representation_probe

    report = Path(report)
    records = []
    path = report / f"{case}-merge-representations.json"
    device = values["q"].device
    generator = torch.Generator().manual_seed(113)
    fixtures = []
    for name, finite, tied in (("random", 4096, False), ("ties", 4096, True),
                                ("padded", 513, False), ("invisible", 0, False)):
        scores = torch.randn(4096, generator=generator)
        if tied:
            scores = scores.round()
        scores[finite:] = -float("inf")
        pairs = []
        for start in (0, 2048):
            order = scores[start:start + 2048].argsort(descending=True) + start
            packed = torch.empty(4096, dtype=torch.float32)
            packed[0::2] = scores[order]
            packed.view(torch.int32)[1::2] = order.int()
            pairs.append(packed.to(device))
        fixtures.append((name, scores, pairs))
    output = torch.empty(4096, device=device, dtype=torch.float32)
    counts = torch.empty(4, device=device, dtype=torch.int32)
    for kind, name in ((0, "tensor8192"), (1, "buffer8192"), (2, "tensor16384")):
        label = f"{case}-merge-{name}"
        capture = CaptureCompiler(report, label)
        entry = dict(kind=kind, representation=name, status="failed", checks=[])
        # A device failure must stop the run; only compile failures are isolated.
        try:
            with patch.object(compiler, "subprocess", capture):
                kernel = _merge_representation_probe.warmup(
                    *fixtures[0][2], output, counts, MERGE_KIND=kind,
                    grid=(1,), use_bytecode=True,
                )
            entry["compile_status"] = "compiled"
            for asm_kind, text in kernel.asm.items():
                if isinstance(text, str):
                    (report / f"{label}-{asm_kind}.txt").write_text(text)
        except Exception as exc:
            entry.update(compile_status="failed", error=repr(exc), compiler_calls=capture.calls)
            records.append(entry)
            path.write_text(json.dumps(records, indent=2))
            print(f"QSA MERGE REPRESENTATION {name}: compile failed", flush=True)
            continue
        entry["compiler_calls"] = capture.calls
        records.append(entry)
        path.write_text(json.dumps(records, indent=2))
        try:
            for fixture, scores, pairs in fixtures:
                _merge_representation_probe[(1,)](*pairs, output, counts, MERGE_KIND=kind, use_bytecode=True)
                actual = output.cpu()
                consumed = counts.cpu()
                indices = actual.view(torch.int32)[1::2].long()
                assert bool(((indices >= 0) & (indices < 4096)).all())
                assert indices.unique().numel() == 2048
                torch.testing.assert_close(actual[0::2], scores[indices], rtol=0, atol=0)
                torch.testing.assert_close(actual[0::2], scores.topk(2048).values, rtol=0, atol=0)
                assert bool(((consumed[:2] >= 0) & (consumed[:2] <= 2048)).all())
                assert int(consumed[:2].sum()) >= 2048 and bool((consumed[2:] == 0).all())
                entry["checks"].append(dict(fixture=fixture, status="passed", consumed=consumed.tolist()))
                path.write_text(json.dumps(records, indent=2))
            entry["status"] = "passed"
            print(f"QSA MERGE REPRESENTATION {name}: four fixtures passed", flush=True)
        except Exception as exc:
            entry["error"] = repr(exc)
            raise
        finally:
            path.write_text(json.dumps(records, indent=2))
    return records


def compile_representation_matrix(values, width, owners, report, case):
    """Gate full execution on isolated tensor merge and complete compilation."""
    from qsa_cann_mixed_910b_trial import PreparedMixedOuter

    report = Path(report)
    result = dict(status="running", merge_input="tensor8192", query_loop="range(2)",
                  structure=dict(TABLE_PLAIN=True, RING_TILE_MAJOR=False), prefixes=[], full=[])
    path = report / f"{case}-mixed-representation-matrix.json"
    result["merge_checks"] = probe_merge_representations(values, report, case)
    path.write_text(json.dumps(result, indent=2))
    if result["merge_checks"][0]["status"] != "passed":
        result["status"] = "failed"
        path.write_text(json.dumps(result, indent=2))
        raise RuntimeError("Compact tensor merge did not pass isolated checks; refusing full execution")
    adapter = PreparedMixedOuter(values, width, slots=1, request_ids=owners, structure=result["structure"])
    for stage in (0, 1):
        for trace in (False, True):
            label = f"{case}-representation-slots1-trace{int(trace)}-stage{stage}"
            result["prefixes"].append(_capture_variant(adapter, report, label, trace, stage))
            path.write_text(json.dumps(result, indent=2))
    for slots in (1, 2):
        adapter = PreparedMixedOuter(values, width, slots=slots, request_ids=owners, structure=result["structure"])
        for trace in (False, True):
            label = f"{case}-representation-slots{slots}-trace{int(trace)}-full"
            result["full"].append(_capture_variant(adapter, report, label, trace, 2))
            path.write_text(json.dumps(result, indent=2))
    result["status"] = "compiled" if all(e["status"] == "compiled" for e in result["full"]) else "failed"
    path.write_text(json.dumps(result, indent=2))
    if result["status"] != "compiled":
        raise RuntimeError("Full mixed representation matrix incomplete; see separate scoring/sort/full IR and errors")
    return result


def compile_final_matrix(values, width, owners, report, case):
    """Require both exact timing kernels to compile; no diagnostic stores."""
    from qsa_cann_mixed_910b_trial import PreparedMixedOuter

    report = Path(report)
    result = dict(status="running", structure=dict(TABLE_PLAIN=True, RING_TILE_MAJOR=False),
                  validation_scope="noninstrumented final/prefix Top512; no internal Top2048 snapshots",
                  merge_checks=probe_merge_representations(values, report, case), full=[])
    path = report / f"{case}-mixed-final-matrix.json"
    path.write_text(json.dumps(result, indent=2))
    if result["merge_checks"][0]["status"] != "passed":
        raise RuntimeError("Compact merge check failed; refusing mixed execution")
    for slots in (1, 2):
        adapter = PreparedMixedOuter(values, width, slots=slots, request_ids=owners, structure=result["structure"])
        result["full"].append(_capture_variant(adapter, report, f"{case}-final-slots{slots}", False, 2))
        path.write_text(json.dumps(result, indent=2))
    result["status"] = "compiled" if all(e["status"] == "compiled" for e in result["full"]) else "failed"
    path.write_text(json.dumps(result, indent=2))
    if result["status"] != "compiled":
        raise RuntimeError("Noninstrumented mixed compilation incomplete")
    return result


def verify_final_prefixes(values, inputs, refs, width, owners, slots, structure, *, boundaries_only=False):
    """Check the exact noninstrumented kernel over page and outer-block tails."""
    import torch
    from qsa_cann_mixed_910b_trial import PreparedMixedOuter
    from probe_qsa_int8_pipeline_910b import check_selection
    from probe_qsa_int8_ten_shapes_910b import verify_structure

    widths = [width] if boundaries_only else sorted({min(width, w) for w in (1, 191, 192, 193, 2047, 2048, 2049, width)})
    checks = []
    for prefix_width in widths:
        visible = inputs["visible"].clamp(max=prefix_width)
        target = dict(values)
        target["visible"] = visible.to(values["q"].device)
        adapter = PreparedMixedOuter(target, prefix_width, slots=slots, request_ids=owners, structure=structure)
        adapter()
        indices = adapter.output.cpu()
        scores = adapter.output_values.cpu()
        prefix_refs = [ref[:prefix_width] for ref in refs]
        verify_structure(indices, visible)
        selected, _ = check_selection(indices, prefix_refs)
        assert selected["checks_passed"], (prefix_width, slots, selected)
        for row, ref in enumerate(prefix_refs):
            valid = indices[row] >= 0
            torch.testing.assert_close(scores[row, valid], ref[indices[row, valid].long()], rtol=1e-3, atol=1e-3)
            assert bool(torch.isneginf(scores[row, ~valid]).all())
        entry = dict(width=prefix_width, slots=slots, status="passed", trace=False, output_k=512)
        checks.append(entry)
        print("QSA MIXED FINAL PREFIX " + json.dumps(entry), flush=True)
    return checks
