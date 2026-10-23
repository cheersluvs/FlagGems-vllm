# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""SHA-pinned standalone NVIDIA Nsight Compute; no system/package changes."""

import hashlib
import json
import os
import platform
import subprocess
import tarfile
import tempfile
import traceback
import urllib.request
from pathlib import Path

MANIFEST_URL = (
    "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.3.json"
)
VERSION = "2026.3.1.2"
URL = (
    "https://developer.download.nvidia.com/compute/cuda/redist/"
    "nsight_compute/linux-x86_64/"
    "nsight_compute-linux-x86_64-2026.3.1.2-archive.tar.xz"
)
SHA256 = "a0039be47cfe551f89d82087256080a9f33c6257d99a72c1164a87466ab072aa"
SIZE = 397165560


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verified(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size == SIZE and digest(path) == SHA256


def download(path):
    partial = Path(str(path) + ".partial")
    with urllib.request.urlopen(URL, timeout=60) as source, partial.open(
        "wb"
    ) as target:
        total = 0
        while True:
            chunk = source.read(4 * 1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            total += len(chunk)
            if total > SIZE:
                raise RuntimeError("Nsight Compute download exceeds pinned size")
            if total % (64 * 1024 * 1024) == 0:
                print(f"Nsight Compute download: {total}/{SIZE} bytes", flush=True)
    if not verified(partial):
        raise RuntimeError("Nsight Compute archive size/SHA256 mismatch")
    partial.replace(path)


def extract(archive, directory):
    if not verified(archive):
        raise RuntimeError("Refusing an unverified Nsight Compute archive")
    # Python3.12 data filter rejects traversal, unsafe links and special devices.
    with tarfile.open(archive, "r:xz") as package:
        package.extractall(directory, filter="data")
    matches = [
        p for p in directory.rglob("ncu") if p.is_file() and os.access(p, os.X_OK)
    ]
    if not matches:
        raise RuntimeError("Verified archive contains no executable ncu")
    return min(matches, key=lambda p: (len(p.parts), str(p)))


def prepare(report, archive=None):
    report = Path(report)
    state = dict(
        status="preparing",
        manifest_url=MANIFEST_URL,
        url=URL,
        version=VERSION,
        expected_sha256=SHA256,
        expected_bytes=SIZE,
    )
    path = report / "ncu-preparation.json"
    try:
        if platform.system() != "Linux" or platform.machine() not in (
            "x86_64",
            "AMD64",
        ):
            raise RuntimeError("Pinned Nsight Compute package requires Linux x86_64")
        cached = (
            Path(archive).resolve()
            if archive
            else Path(tempfile.gettempdir()) / f"qsa-ncu-{VERSION}.tar.xz"
        )
        state["archive"] = str(cached)
        path.write_text(json.dumps(state, indent=2) + "\n")
        if archive:
            if not verified(cached):
                raise RuntimeError(
                    "Supplied Nsight Compute archive failed verification"
                )
        elif not verified(cached):
            print(
                f"Downloading isolated Nsight Compute {VERSION}: {SIZE} bytes",
                flush=True,
            )
            download(cached)
        state.update(status="verified", archive_sha256=digest(cached))
        path.write_text(json.dumps(state, indent=2) + "\n")
        directory = Path(tempfile.mkdtemp(prefix="qsa-ncu-portable."))
        executable = extract(cached, directory)
        result = subprocess.run(
            [str(executable), "--version"], capture_output=True, text=True, timeout=30
        )
        state.update(
            directory=str(directory),
            executable=str(executable),
            executable_sha256=digest(executable),
            version_check=dict(
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
            ),
        )
        if result.returncode:
            raise RuntimeError("Extracted ncu --version failed: " + result.stderr)
        state["status"] = "ready"
        print("Isolated Nsight Compute:", executable, flush=True)
        return str(executable)
    except BaseException:
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        path.write_text(json.dumps(state, indent=2) + "\n")
