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

"""Prepare an independent FlagTree copy with final native core-mode propagation."""

import argparse
import ast
import difflib
import hashlib
import inspect
import json
import shutil
import stat
import traceback
from pathlib import Path

COMPILER = Path("triton/backends/ascend/compiler.py")
SOURCE_SHA256 = "d11660971dc400be7cb26cc47eb308609464d8fb69ecee1c9191fe32a366e0e3"
ROUTES = {
    "linalg_to_bin_enable_npu_compile_910_95",
    "linalg_to_bin_enable_npu_compile_A2_A3",
}
HELPER = "_reconcile_mix_mode_from_task_type"


def _reconcile_mix_mode_from_task_type(metadata):
    """Use final native task kind for loading; retain the task/ratio encoding."""
    task_type = metadata.get("bs_task_type")
    if task_type is None:
        return
    task_kind = int(task_type) // 10
    mode = {1: "aiv", 2: "aic", 3: "mix", 4: "mix"}.get(task_kind)
    if mode is not None:
        metadata["mix_mode"] = mode


def patch_source(source):
    tree = ast.parse(source)
    if any(isinstance(n, ast.FunctionDef) and n.name == HELPER for n in ast.walk(tree)):
        raise ValueError("Compiler already contains the candidate helper")
    anchor = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_parse_linalg_metadata"
        ),
        None,
    )
    if anchor is None:
        raise ValueError("Compiler metadata parser anchor missing")
    callbacks = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if (
                not isinstance(call.func, ast.Name)
                or call.func.id != "__get_metadata_attr_by_callback"
            ):
                continue
            if (
                len(call.args) != 4
                or not isinstance(call.args[1], ast.Constant)
                or call.args[1].value != "_infer_task_type_function"
            ):
                continue
            if (
                not isinstance(call.args[2], ast.Name)
                or call.args[2].id != "metadata"
                or not isinstance(call.args[3], ast.Constant)
                or call.args[3].value != "bs_task_type"
            ):
                raise ValueError("Unexpected task callback ABI")
            callbacks.append((fn.name, node))
    if len(callbacks) != 2 or {name for name, _ in callbacks} != ROUTES:
        raise ValueError(
            f"Expected exactly the two known binary compilation routes, got {[n for n, _ in callbacks]}"
        )
    lines = source.splitlines(keepends=True)
    edits = [
        (
            anchor.lineno - 1,
            inspect.getsource(_reconcile_mix_mode_from_task_type) + "\n\n",
        )
    ]
    for _, node in callbacks:
        line = lines[node.lineno - 1]
        indent = line[: len(line) - len(line.lstrip())]
        edits.append((node.end_lineno, f"{indent}{HELPER}(metadata)\n"))
    for position, text in sorted(edits, reverse=True):
        lines.insert(position, text)
    patched = "".join(lines)
    compile(patched, "candidate-compiler.py", "exec")
    changed = ast.parse(patched)
    calls = [
        n
        for n in ast.walk(changed)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == HELPER
    ]
    if len(calls) != 2:
        raise ValueError("Candidate callback insertion count mismatch")
    return patched, sorted(name for name, _ in callbacks)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source_root, destination, report, expected_source_sha256=SOURCE_SHA256):
    source_root, destination, report = (
        Path(p).resolve() for p in (source_root, destination, report)
    )
    if (
        destination == source_root
        or destination.is_relative_to(source_root)
        or source_root.is_relative_to(destination)
        or report.is_relative_to(source_root)
    ):
        raise ValueError(
            "Candidate/report paths must be independent of the original package"
        )
    if destination.exists():
        raise ValueError("Candidate destination must not exist")
    report.mkdir(parents=True, exist_ok=True)
    source_compiler = source_root / COMPILER
    before = source_compiler.read_bytes()
    if sha256(before) != expected_source_sha256:
        raise ValueError(
            f"Original compiler hash differs from verified snapshot: {sha256(before)}; "
            f"expected {expected_source_sha256}"
        )
    patched, routes = patch_source(before.decode())
    diff = "".join(
        difflib.unified_diff(
            before.decode().splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile="a/third_party/ascend/backend/compiler.py",
            tofile="b/third_party/ascend/backend/compiler.py",
            n=0,
        )
    )
    print(f"Copying FlagTree from {source_root} to {destination}", flush=True)
    shutil.copytree(
        source_root,
        destination,
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    candidate = destination / COMPILER
    if (
        not candidate.resolve().is_relative_to(destination)
        or candidate.read_bytes() != before
    ):
        raise ValueError(
            "Candidate compiler did not materialize as an independent copy"
        )
    candidate.parent.chmod(candidate.parent.stat().st_mode | stat.S_IWUSR)
    temporary = candidate.with_suffix(".candidate")
    temporary.write_text(patched)
    temporary.replace(candidate)
    if source_compiler.read_bytes() != before:
        raise ValueError("Original compiler changed during preparation")
    (report / "flagtree-core-mode.patch.txt").write_text(diff)
    (report / "compiler-before.py.txt").write_bytes(before)
    (report / "compiler-after.py.txt").write_bytes(candidate.read_bytes())
    manifest = dict(
        status="prepared",
        source_root=str(source_root),
        candidate_root=str(destination),
        source_compiler=str(source_compiler),
        candidate_compiler=str(candidate),
        source_compiler_sha256=sha256(before),
        patched_compiler_sha256=sha256(candidate.read_bytes()),
        patch_sha256=sha256(diff.encode()),
        callback_routes=routes,
        helper=HELPER,
        original_unchanged=True,
        scope="Experimental compiler source candidate in a materialized copy; no installed package or operator changes",
    )
    (report / "compiler-core-mode-fix.json").write_text(json.dumps(manifest, indent=2))
    print("COMPILER_CANDIDATE " + json.dumps(manifest), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.resolve().is_relative_to(args.source_root.resolve()):
        parser.error("Report directory cannot be inside the original package")
    try:
        prepare(args.source_root, args.destination, args.report)
    except Exception as exc:
        args.report.mkdir(parents=True, exist_ok=True)
        (args.report / "compiler-core-mode-fix.json").write_text(
            json.dumps(dict(status="preparation_failed", error=repr(exc)), indent=2)
        )
        (args.report / "corefix-preparation-error.txt").write_text(
            traceback.format_exc()
        )
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
