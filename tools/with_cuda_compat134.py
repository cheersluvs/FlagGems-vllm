# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Run a diagnostic with verified CUDA 13.4 user-space libraries in /tmp."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path

PACKAGE_URL = (
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/"
    "cuda-compat-13-4_615.71.09-2ubuntu1_amd64.deb"
)
PACKAGE_SHA256 = "bfacf8a0c2bc5cac76880d443f5bd542b209166b14f90dbe2384f962b379d00a"


def download(path):
    with urllib.request.urlopen(PACKAGE_URL, timeout=180) as src, path.open(
        "wb"
    ) as dst:
        shutil.copyfileobj(src, dst)


def run(script, report, extra):
    try:
        prefix = Path(tempfile.mkdtemp(prefix="qsa-cuda134-compat."))
        package = prefix / "cuda-compat-13-4.deb"
        print("Preparing isolated CUDA 13.4 compat:", prefix, flush=True)
        download(package)
        digest = hashlib.sha256(package.read_bytes()).hexdigest()
        if digest != PACKAGE_SHA256:
            raise RuntimeError("CUDA compatibility package SHA256 mismatch")
        root = prefix / "root"
        proc = subprocess.run(
            ["dpkg-deb", "--extract", str(package), str(root)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode:
            raise RuntimeError("dpkg-deb extraction failed: " + proc.stderr)
        compat = root / "usr/local/cuda-13.4/compat"
        library = (compat / "libcuda.so.1").resolve()
        if not library.is_file() or not library.is_relative_to(root.resolve()):
            raise RuntimeError("Expected extracted libcuda.so.1 in isolated prefix")
        env = os.environ.copy()
        previous = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(compat) + (
            os.pathsep + previous if previous else ""
        )
        metadata = dict(
            package_url=PACKAGE_URL,
            package_sha256=digest,
            compat_directory=str(compat),
            libcuda=str(library),
            libcuda_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            previous_ld_library_path=previous,
            child_ld_library_path=env["LD_LIBRARY_PATH"],
        )
        (report / "compat134.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("Starting diagnostic with isolated libcuda:", library, flush=True)
        # A fresh process is necessary: changing LD_LIBRARY_PATH after loading
        # libcuda in a Python process does not replace the already-loaded library.
        os.execve(
            sys.executable,
            [sys.executable, "-I", str(script), "--report-dir", str(report), *extra],
            env,
        )
        return 0  # Only reached by CPU tests mocking execve.
    except BaseException:
        result = dict(
            status="compat_preparation_failed",
            error=traceback.format_exc(),
            package_url=PACKAGE_URL,
        )
        (report / "ptxas-summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result["error"], file=sys.stderr, flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    options, extra = parser.parse_known_args()
    return run(options.script.resolve(), options.report_dir.resolve(), extra)


if __name__ == "__main__":
    sys.exit(main())
