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

"""Inspect installed local-memory APIs and replay saved IR with a print-only flag."""

import argparse
import collections
import hashlib
import importlib
import inspect
import json
import re
import subprocess
import traceback
from pathlib import Path

SOURCE_COMMIT = "889121172dbe56da4cba2bc97520f72b0f913e0b"
DUMP_FLAG = "--bishengir-print-ir-after=hivm-graph-sync-solver"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def find_source(explicit):
    candidates = (
        [Path(explicit)]
        if explicit
        else sorted(
            Path("/tmp").glob("qsa-ascend910b-int8-harness.*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )
    for root in candidates:
        marker = root / "code-commit.txt"
        if not marker.is_file() or marker.read_text().strip() != SOURCE_COMMIT:
            continue
        path = root / "int8_budget_exact-fused-second-cube.json"
        if path.is_file():
            record = json.loads(path.read_text())
            if (
                record.get("status") == "completed"
                and record.get("reduction_diagnostics", {}).get("status") == "completed"
            ):
                return root, record
    raise RuntimeError(
        "No completed report for pinned diagnostic commit; pass --source-report explicitly"
    )


def inventory(report):
    import triton

    tle = importlib.import_module("triton.experimental.tle")
    result = dict(
        triton_version=triton.__version__,
        triton_source=triton.__file__,
        tle_source=tle.__file__,
        symbols={},
        source_files=[],
        limitation="Symbol/source presence only; no claim of supported transfers, compilation or device execution",
    )
    for name in (
        "alloc",
        "copy",
        "to_buffer",
        "to_tensor",
        "local_ptr",
        "subview",
        "matmul",
    ):
        value = getattr(tle.dsa, name, None)
        entry = dict(present=value is not None)
        if value is not None:
            try:
                entry.update(
                    signature=str(inspect.signature(value)),
                    source=inspect.getsourcefile(value),
                )
            except (TypeError, ValueError):
                entry["signature"] = "unavailable"
        result["symbols"]["dsa." + name] = entry
    for name in ("UB", "L1", "L0A", "L0B", "L0C", "raw"):
        value = getattr(tle.dsa.ascend, name, None)
        result["symbols"]["ascend." + name] = dict(
            present=value is not None, representation=str(value)
        )
    try:
        custom = importlib.import_module(
            "triton.experimental.tle.language.dsa.ascend.custom_ops"
        )
        result["custom_ops"] = dict(
            source=custom.__file__,
            exported_names=[name for name in dir(custom) if not name.startswith("_")],
        )
    except ImportError as exc:
        result["custom_ops"] = dict(import_error=repr(exc))
    root = Path(triton.__file__).resolve().parent
    roots = [root / "experimental" / "tle", root / "language" / "extra" / "cann"]
    pattern = re.compile(
        r"fixpipe|fix_pipe|DEQF16|mmad|matmul|dequant|CUSTOM_OPS_BITCODE|L0C|\bL1\b|"
        r"def (?:to_buffer|to_tensor|matmul|copy|alloc)\b",
        re.I,
    )
    budget = 1024 * 1024
    for directory in roots:
        for path in sorted(directory.rglob("*.py")):
            data = path.read_bytes()
            lines = data.decode(errors="replace").splitlines()
            hits = [i for i, line in enumerate(lines) if pattern.search(line)]
            if not hits:
                continue
            entry = dict(
                source=str(path),
                sha256=sha(data),
                bytes=len(data),
                hits=[dict(line=i + 1, text=lines[i][:500]) for i in hits[:80]],
            )
            result["source_files"].append(entry)
            if budget > 0:
                saved = data[: min(65536, budget)]
                name = "api-" + str(path.relative_to(root)).replace("/", "_") + ".txt"
                (report / name).write_bytes(saved)
                entry.update(artifact=name, truncated=len(saved) != len(data))
                budget -= len(saved)
            else:
                entry["not_exported"] = "source capture budget exhausted"
    (report / "installed-api.json").write_text(json.dumps(result, indent=2))
    print("INSTALLED_API " + json.dumps(result["symbols"]), flush=True)
    return result


def summarize_ir(text):
    active = False
    sections = 0
    function = None
    counts = collections.Counter()
    evidence = []
    for number, line in enumerate(text.splitlines(), 1):
        if "IR Dump" in line:
            active = "IR Dump After" in line and "graphsyncsolver" in re.sub(
                r"[^a-z]", "", line.lower()
            )
            sections += int(active)
        if not active:
            continue
        match = re.search(r"func\.func\s+(?:private\s+)?@([^\s(]+)", line)
        if match:
            function = match[1]
        tokens = re.findall(
            r"\b(?:hivm\.hir\.[\w.]+|memref_ext\.alloc_workspace|memref\.(?:alloc|copy|load|store))\b",
            line,
        )
        if tokens:
            counts.update(tokens)
            if len(evidence) < 160 and any(
                re.search(r"fixpipe|matmul|mmad|alloc|copy|load|store|sync|flag", t)
                for t in tokens
            ):
                evidence.append(
                    dict(line=number, function=function, text=line.strip()[:2200])
                )
    return dict(
        after_solver_sections=sections,
        operations=dict(counts),
        examples=evidence,
        interpretation=(
            "Static operations, not dynamic counts or timing. "
            "Inspect SSA producer/consumer buffers in full IR before asserting a GM or L1 handoff."
        ),
    )


def replay(report, source, record, arm, timeout):
    original = record["arms"][arm]["checks"][0]["compiler_calls"]
    if len(original) != 1 or original[0].get("returncode") != 0:
        raise ValueError("Expected one successful saved full-fixture compilation")
    call = original[0]
    saved_input = source / call["input_ir"]
    work = report / arm
    work.mkdir()
    input_path = work / "kernel.mlir"
    input_path.write_bytes(saved_input.read_bytes())
    command = list(call["command"])
    if DUMP_FLAG in command:
        raise ValueError("Source invocation already has diagnostic option")
    inputs = [i for i, value in enumerate(command) if value.endswith(".mlir")]
    if len(inputs) != 1:
        raise ValueError("Ambiguous compiler input")
    command[inputs[0]] = str(input_path)
    command[command.index("-o") + 1] = str(work / "kernel")
    command.append(DUMP_FLAG)
    entry = dict(
        status="running",
        input_sha256=sha(input_path.read_bytes()),
        original_command=call["command"],
        diagnostic_command=command,
        changed_options=[DUMP_FLAG],
        device_execution=False,
        ordinary_score_us=record["arms"][arm]["median_us"],
        binary_comparison={},
    )
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        result = subprocess.CompletedProcess(
            command, 124, exc.stdout or b"", exc.stderr or b""
        )
        entry["timeout"] = True
    entry["returncode"] = result.returncode
    sections = 0
    for name, data in (("stdout", result.stdout), ("stderr", result.stderr)):
        saved = data[: 24 * 1024**2]
        filename = arm + "-post-sync-" + name + ".txt"
        (report / filename).write_bytes(saved)
        summary = summarize_ir(saved.decode(errors="replace"))
        sections += summary["after_solver_sections"]
        entry[name] = dict(
            artifact=filename,
            bytes=len(data),
            truncated=len(saved) != len(data),
            summary=summary,
        )
    for artifact in call.get("output_artifacts", []):
        path = work / artifact["path"]
        if path.is_file():
            digest = sha(path.read_bytes())
            entry["binary_comparison"][artifact["path"]] = dict(
                original_sha256=artifact["sha256"],
                diagnostic_sha256=digest,
                identical=digest == artifact["sha256"],
            )
    entry["status"] = "captured" if result.returncode == 0 and sections else "failed"
    print(
        "LOCAL_HANDOFF_IR "
        + json.dumps(
            dict(
                arm=arm,
                status=entry["status"],
                sections=sections,
                binary_comparison=entry["binary_comparison"],
            )
        ),
        flush=True,
    )
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-report")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="running",
        arms={},
        scope="Installed API and saved-IR audit; no new performance or correctness claim",
    )
    try:
        source, record = find_source(args.source_report)
        result.update(
            source_report=str(source),
            source_commit=SOURCE_COMMIT,
            source_record_sha256=sha(
                (source / "int8_budget_exact-fused-second-cube.json").read_bytes()
            ),
        )
        (report / "prior-budget-exact.json").write_text(json.dumps(record, indent=2))
        result["api"] = inventory(report)
        for arm in ("current_int8_integrated", "query16_second_cube"):
            result["arms"][arm] = replay(report, source, record, arm, args.timeout)
        result["status"] = (
            "completed"
            if all(e["status"] == "captured" for e in result["arms"].values())
            else "failed"
        )
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        print(result["error"], flush=True)
    (report / "summary.json").write_text(json.dumps(result, indent=2))
    print(
        "LOCAL_HANDOFF_AUDIT "
        + json.dumps(
            dict(status=result["status"], source_report=result.get("source_report"))
        ),
        flush=True,
    )
    return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
