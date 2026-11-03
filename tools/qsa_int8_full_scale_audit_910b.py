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

"""Compare compiler artifacts and retain bounded profiler diagnostics."""

import gzip
import hashlib
import json
import re
import sqlite3
from pathlib import Path


def strip_locations(text):
    """Remove MLIR source locations; preserve SSA and operation ordering."""
    text = "\n".join(
        line for line in text.splitlines() if not re.match(r"\s*#loc[\w]*\s*=", line)
    )
    output = []
    index = 0
    while index < len(text):
        if text.startswith("loc(", index):
            depth = 1
            position = index + 4
            quoted = False
            escape = False
            while position < len(text) and depth:
                char = text[position]
                if quoted:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        quoted = False
                elif char == '"':
                    quoted = True
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                position += 1
            if depth:
                raise ValueError("Unbalanced MLIR location")
            index = position
        else:
            output.append(text[index])
            index += 1
    return " ".join("".join(output).split())


def audit_full_scales(report, case, arms, compiler):
    result = {}
    for name in arms:
        if compiler[name]["load_count"] != 7:
            raise AssertionError(f"Full score load count changed: {name}")
        hashes = {}
        for key in ("ttir", "ttadapter", "bcmlir"):
            path = Path(report) / f"{case}-{name}-{key}.txt"
            if path.is_file():
                hashes[key] = hashlib.sha256(
                    strip_locations(path.read_text()).encode()
                ).hexdigest()
        result[name] = dict(
            location_stripped_ir_sha256=hashes,
            npubin_sha256=compiler[name].get("npubin_sha256"),
        )
    reference = result["integrated_full"]
    for name in arms:
        result[name]["same_ir_as_integrated"] = {
            key: digest == reference["location_stripped_ir_sha256"].get(key)
            for key, digest in result[name]["location_stripped_ir_sha256"].items()
        }
        binary = result[name]["npubin_sha256"]
        result[name]["same_binary_as_integrated"] = (
            binary == reference["npubin_sha256"]
            if binary and reference["npubin_sha256"]
            else None
        )
    return result


def collect_profiler_logs(report, case):
    """Copy logs/schema names from this worker's named export directories only."""
    report = Path(report)
    try:
        text = (report / f"{case}-console.txt").read_text(errors="replace")
        paths = list(
            dict.fromkeys(
                re.findall(
                    r"Start parsing profiling data in sync mode at: ([^\r\n]+)", text
                )
            )
        )[-5:]
        records = []
        budget = 2 * 1024 * 1024
        db_count = 0
        for value in paths:
            path = Path(value.strip())
            entry = dict(path=str(path), exists=path.is_dir(), files=[])
            records.append(entry)
            if (
                not path.is_absolute()
                or not path.name.endswith("_ascend_pt")
                or path.parent.name != "export_only_prof_dir"
                or path.is_symlink()
                or not path.is_dir()
            ):
                entry["collection"] = "missing_or_outside_expected_export_directory"
                continue
            for file in sorted(path.rglob("*")):
                if len(entry["files"]) >= 48 or budget <= 0:
                    break
                if (
                    not file.is_file()
                    or file.is_symlink()
                    or not file.resolve().is_relative_to(path.resolve())
                ):
                    continue
                relative = file.relative_to(path)
                if file.suffix == ".db":
                    if db_count >= 8:
                        continue
                    db_count += 1
                    item = dict(path=str(relative), bytes=file.stat().st_size)
                    try:
                        with sqlite3.connect(
                            file.resolve().as_uri() + "?mode=ro", uri=True, timeout=1
                        ) as db:
                            item["tables"] = [
                                row[0]
                                for row in db.execute(
                                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 64"
                                )
                            ]
                    except Exception as exc:
                        item["schema_error"] = str(exc)
                    entry["files"].append(item)
                elif "log" in str(relative).lower() and file.suffix in (
                    ".log",
                    ".txt",
                    ".json",
                ):
                    size = min(128 * 1024, budget)
                    with file.open("rb") as stream:
                        payload = stream.read(size)
                    budget -= len(payload)
                    entry["files"].append(
                        dict(
                            path=str(relative),
                            bytes=file.stat().st_size,
                            truncated=file.stat().st_size > len(payload),
                            text=payload.decode(errors="replace"),
                        )
                    )
        data = dict(
            case=case,
            directories=records,
            scope="Own worker export logs and SQLite table names only; no tensor data; absent paths explicit",
        )
        archive = report / f"{case}-profiler-diagnostics.json.gz"
        with gzip.open(archive, "wt") as stream:
            json.dump(data, stream, indent=2)
        return dict(
            artifact=archive.name,
            export_directories=len(records),
            existing=sum(bool(x["exists"]) for x in records),
        )
    except Exception as exc:
        return dict(status="diagnostic_collection_failed", error=str(exc))
