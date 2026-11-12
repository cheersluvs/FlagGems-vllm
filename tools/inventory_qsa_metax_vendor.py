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

"""Read-only inventory of vendor wheels, image archives and QSA sources."""

import argparse
import ast
import email.parser
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

LIMIT = 512 * 1024
SOURCE_NAMES = {"qsa_indexer.py", "indexer_qsa.py"}
SYMBOLS = (
    "qsa_select_paged_prefill",
    "qsa_select_paged_decode",
    "expand_qsa_block_indices",
)


def source_info(data, origin):
    text = data.decode(errors="replace")
    result = dict(origin=origin, sha256=hashlib.sha256(data).hexdigest(), source=text)
    result["mentions"] = [n for n in SYMBOLS if n in text]
    try:
        tree = ast.parse(text)
        result["functions"] = [
            n.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        result["target_definitions"] = [n for n in SYMBOLS if n in result["functions"]]
    except SyntaxError as exc:
        result["parse_error"] = str(exc)
    return result


def wheel_info(handle, origin):
    result = dict(origin=origin, kind="wheel", packages=[], sources=[], extensions=[])
    with zipfile.ZipFile(handle) as archive:
        for member in archive.infolist():
            name = member.filename
            if name.endswith(".dist-info/METADATA") and member.file_size <= LIMIT:
                meta = email.parser.Parser().parsestr(
                    archive.read(member).decode(errors="replace")
                )
                result["packages"].append(
                    dict(
                        name=meta.get("Name"),
                        version=meta.get("Version"),
                        requires_python=meta.get("Requires-Python"),
                        requires_dist=meta.get_all("Requires-Dist", []),
                    )
                )
            if (
                Path(name).name in SOURCE_NAMES
                and member.file_size <= LIMIT
                and len(result["sources"]) < 12
            ):
                result["sources"].append(
                    source_info(archive.read(member), origin + "::" + name)
                )
            if name.endswith(".so") and (
                "vllm" in name.lower() or "topk" in name.lower()
            ):
                result["extensions"].append(name)
    return result


def scan_tar(archive, origin, deadline, allow_layers=False):
    result = dict(
        origin=origin, kind="tar", wheels=[], sources=[], manifests=[], layers=[]
    )
    members = []
    for count, member in enumerate(archive):
        if count >= 30000 or time.monotonic() >= deadline:
            result["incomplete"] = "member/time limit"
            break
        if not member.isfile():
            continue
        name = member.name
        if allow_layers:
            members.append(member)
        if Path(name).name in ("manifest.json", "index.json") and member.size <= LIMIT:
            with archive.extractfile(member) as stream:
                value = json.load(stream)
            result["manifests"].append(dict(path=name, data=value))
        if name.endswith(".whl") and any(k in name.lower() for k in ("vllm", "metax")):
            entry = dict(path=name, bytes=member.size)
            # Seekable uncompressed layers permit ZIP metadata reads without extraction.
            with archive.extractfile(member) as stream:
                entry["inspection"] = wheel_info(stream, origin + "::" + name)
            result["wheels"].append(entry)
        if (
            Path(name).name in SOURCE_NAMES
            and member.size <= LIMIT
            and len(result["sources"]) < 12
        ):
            with archive.extractfile(member) as stream:
                result["sources"].append(
                    source_info(stream.read(), origin + "::" + name)
                )
    if allow_layers:
        lookup = {m.name: m for m in members}
        layers = []
        for manifest in result["manifests"]:
            if isinstance(manifest["data"], list):
                for image in manifest["data"]:
                    layers.extend(image.get("Layers", []))
        for name in list(dict.fromkeys(layers))[::-1][:16]:
            if time.monotonic() >= deadline:
                result["incomplete"] = "layer/time limit"
                break
            member = lookup.get(name)
            if member is None:
                result["layers"].append(dict(path=name, status="not enumerated"))
                continue
            try:
                with archive.extractfile(member) as stream:
                    with tarfile.open(fileobj=stream, mode="r:") as nested:
                        result["layers"].append(
                            scan_tar(nested, origin + "::" + name, deadline)
                        )
            except (tarfile.TarError, OSError, zipfile.BadZipFile) as exc:
                result["layers"].append(
                    dict(path=name, status="not inspected", error=str(exc))
                )
        if len(set(layers)) > 16:
            result["incomplete"] = "layer count limit"
    return result


def inspect_path(path):
    if path.suffix == ".whl":
        return wheel_info(path, str(path))
    if path.suffix == ".py":
        if path.stat().st_size > LIMIT:
            return dict(status="source too large")
        return source_info(path.read_bytes(), str(path))
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            matches = [
                info.filename
                for info in archive.infolist()
                if any(k in info.filename.lower() for k in ("vllm", "metax", ".whl"))
            ]
        return dict(
            kind="zip_archive",
            matched_paths=matches[:1000],
            incomplete=len(matches) > 1000,
            scope="Names only; nested archive contents not inspected",
        )
    if not path.name.endswith(".tar") and path.stat().st_size > 256 * 1024**2:
        return dict(
            status="deferred",
            reason="Large compressed archive: avoid multi-GB decompression; filename alone proves no content",
        )
    with tarfile.open(path, "r:*") as archive:
        return scan_tar(
            archive,
            str(path),
            time.monotonic() + 12,
            allow_layers=path.name.endswith(".tar"),
        )


def command(args):
    tool = shutil.which(args[0])
    if not tool:
        return dict(status="unavailable")
    try:
        run = subprocess.run(
            [tool, *args[1:]], capture_output=True, text=True, timeout=10
        )
        return dict(
            returncode=run.returncode,
            stdout=run.stdout[:64000],
            stderr=run.stderr[:8000],
        )
    except subprocess.TimeoutExpired:
        return dict(status="timeout")


def installed_inventory():
    result = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name", "")
        if not any(
            k in name.lower()
            for k in ("vllm", "triton", "flagtree", "torch", "maca", "metax")
        ):
            continue
        entry = dict(
            name=name,
            version=dist.version,
            location=str(dist.locate_file("")),
            sources=[],
        )
        if "vllm" in name.lower():
            for file in dist.files or []:
                if Path(str(file)).name in SOURCE_NAMES:
                    path = Path(dist.locate_file(file))
                    if path.is_file() and path.stat().st_size <= LIMIT:
                        entry["sources"].append(
                            source_info(path.read_bytes(), str(path))
                        )
        result.append(entry)
    return result


def collect(args):
    started = time.monotonic()
    root = args.root.resolve()
    result = dict(
        status="running",
        root=str(root),
        python=sys.executable,
        platform=platform.platform(),
        artifacts=[],
        scan_errors=[],
        scope=(
            "Bounded read-only inventory; no runtime package imports, installation, image load/run "
            "or GPU kernel execution. Absence in a bounded scan is not proof of absence from the machine."
        ),
    )
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    result["installed"] = installed_inventory()
    result["docker_images"] = command(
        ["docker", "image", "ls", "--format", "{{json .}}"]
    )
    result["gpu_inventory"] = command(["mx-smi"])
    candidates = []
    examined = 0
    if not root.is_dir():
        result.update(
            status="root_unavailable",
            error="Not a readable directory in this shell/container namespace",
        )
    else:

        def onerror(exc):
            if len(result["scan_errors"]) < 30:
                result["scan_errors"].append(str(exc))

        for directory, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
            if len(Path(directory).relative_to(root).parts) >= args.max_depth:
                if dirs:
                    result["scan_incomplete"] = "depth limit"
                dirs[:] = []
            for name in files:
                examined += 1
                if examined > 50000 or time.monotonic() - started > 120:
                    break
                path = Path(directory) / name
                low = name.lower()
                if (
                    low.endswith(".whl")
                    or low.endswith(
                        (".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.zst", ".zip")
                    )
                    or name in SOURCE_NAMES
                ):
                    candidates.append(path)
            if examined > 50000 or time.monotonic() - started > 120:
                result["scan_incomplete"] = "file/time limit"
                break
        candidates.sort(
            key=lambda p: (
                not (p.suffix == ".whl" and "vllm" in p.name.lower()),
                str(p),
            )
        )
        result["candidate_paths"] = [str(p) for p in candidates]
        for path in candidates[:80]:
            remaining = 120 - (time.monotonic() - started)
            if remaining < 1:
                result["scan_incomplete"] = "inspection time limit"
                break
            entry = dict(path=str(path))
            try:
                entry["bytes"] = path.stat().st_size
                run = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        str(path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=min(20, remaining),
                )
                entry["returncode"] = run.returncode
                if run.stdout:
                    entry["content"] = json.loads(run.stdout)
                if run.stderr:
                    entry["stderr"] = run.stderr[:8000]
            except subprocess.TimeoutExpired:
                entry["status"] = "inspection timeout"
            except (OSError, ValueError) as exc:
                entry["error"] = str(exc)
            result["artifacts"].append(entry)
            print("VENDOR_ARTIFACT", str(path), entry.get("bytes"), "bytes", flush=True)
        if len(candidates) > 80:
            result["scan_incomplete"] = "artifact count limit"
        result["status"] = "completed"
    result["examined_files"] = examined
    result["elapsed_seconds"] = time.monotonic() - started

    # Export bounded captured code as text for review, never execute it.
    def export(node):
        if isinstance(node, dict):
            if "source" in node and isinstance(node["source"], str):
                name = (
                    "qsa-source-"
                    + hashlib.sha256(node.get("origin", "").encode()).hexdigest()[:16]
                    + ".txt"
                )
                (report / name).write_text(node.pop("source"))
                node["source_artifact"] = name
                print(
                    "QSA_SOURCE",
                    node.get("origin"),
                    node.get("target_definitions"),
                    flush=True,
                )
            for child in list(node.values()):
                export(child)
        elif isinstance(node, list):
            for child in node:
                export(child)

    export(result)
    (report / "summary.json").write_text(json.dumps(result, indent=2))
    print(
        "ROOT",
        root,
        "STATUS",
        result["status"],
        "SCANNED",
        examined,
        "ARTIFACTS",
        len(result["artifacts"]),
        flush=True,
    )
    print(
        "INSTALLED",
        json.dumps(
            [dict(name=d["name"], version=d["version"]) for d in result["installed"]]
        ),
        flush=True,
    )
    print(
        "INVENTORY_ONLY: package/source presence does not prove import, "
        "TopK extension availability or device correctness",
        flush=True,
    )
    return int(result["status"] != "completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/zhouxt"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            print(json.dumps(inspect_path(args.worker)))
            return 0
        except Exception as exc:
            print(json.dumps(dict(status="inspection_failed", error=repr(exc))))
            return 1
    if args.report is None:
        parser.error("--report is required")
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
