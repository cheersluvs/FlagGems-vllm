# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Isolated, digest-pinned FlagGems import dependency; never installs runtimes."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

VERSION = "5.0.2"
WHEEL = "flag_gems-5.0.2-py3-none-any.whl"
SHA256 = "d7c5a6121bf2d3f7daddde43bfa7cae739de6cb5b294dcfe7ae6173045c35552"
URL = (
    "https://files.pythonhosted.org/packages/08/50/944776a89835961de5d03c942efc58237492df9c584765e7044ed617d099/"
    + WHEEL
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(report, wheel=None):
    report = Path(report).resolve()
    report.mkdir(parents=True, exist_ok=True)
    archive = report / WHEEL
    if wheel:
        content = Path(wheel).read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=90) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise RuntimeError("FlagGems wheel SHA256 mismatch")
    archive.write_bytes(content)
    target = report / "python-deps"
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            name = PurePosixPath(info.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or name.parts[0] not in ("flag_gems", "flag_gems-5.0.2.dist-info")
            ):
                raise RuntimeError("Unexpected wheel member: " + info.filename)
        source.extractall(target)
    hashes = {
        str(path.relative_to(target)): digest(path)
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }
    state = {
        "status": "prepared",
        "package": "flag_gems",
        "version": VERSION,
        "url": URL,
        "wheel_sha256": SHA256,
        "target": str(target),
        "files": hashes,
        "policy": "Only FlagGems extracted under the report; no pip, dependencies, Torch or Triton installation",
    }
    (report / "flag-gems-dependency.json").write_text(
        json.dumps(state, indent=2) + "\n"
    )
    print("Prepared isolated FlagGems:", target, flush=True)
    return target


def configure():
    value = os.environ.get("QSA_PR_DEPENDENCY_REPORT")
    if not value:
        raise RuntimeError("Missing PR dependency preparation report")
    state = json.loads((Path(value) / "flag-gems-dependency.json").read_text())
    if state["wheel_sha256"] != SHA256 or state["version"] != VERSION:
        raise RuntimeError("Unexpected FlagGems dependency identity")
    target = Path(state["target"]).resolve()
    # Only allow the two verified top-level directories; never shadow runtimes.
    if {p.name for p in target.iterdir()} != {"flag_gems", "flag_gems-5.0.2.dist-info"}:
        raise RuntimeError("Unexpected top-level dependency overlay")
    for name, expected in state["files"].items():
        if digest(target / name) != expected:
            raise RuntimeError("FlagGems dependency file changed: " + name)
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
    spec = importlib.util.find_spec("flag_gems")
    expected = target / "flag_gems/__init__.py"
    if spec is None or Path(spec.origin).resolve() != expected:
        raise RuntimeError("FlagGems import resolves outside isolated dependency")
    return {
        "version": VERSION,
        "source": str(expected),
        "wheel_sha256": SHA256,
        "source_sha256": state["files"]["flag_gems/__init__.py"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--wheel")
    args = parser.parse_args()
    prepare(args.report, args.wheel)
