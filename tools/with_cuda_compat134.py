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
        return dict(
            response_url=src.geturl(),
            http_status=getattr(src, "status", None),
            content_type=src.headers.get("Content-Type"),
            content_length=src.headers.get("Content-Length"),
        )


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_candidates():
    # Only previous outputs of this wrapper; never search arbitrary packages.
    return sorted(
        Path(tempfile.gettempdir()).glob("qsa-cuda134-compat.*/cuda-compat-13-4.deb"),
        reverse=True,
    )


def prepare_package(package, report):
    state = dict(
        status="preparing",
        package_url=PACKAGE_URL,
        expected_sha256=PACKAGE_SHA256,
        bootstrap_source=str(Path(__file__).resolve()),
        bootstrap_sha256=file_sha256(Path(__file__)),
        destination=str(package),
        cached_candidates=[],
        download_attempts=[],
    )
    record = report / "compat134-package.json"

    def save():
        record.write_text(json.dumps(state, indent=2) + "\n")

    def inspect(path):
        return dict(bytes=path.stat().st_size, actual_sha256=file_sha256(path))

    save()
    for source in cache_candidates():
        if source.resolve() == package.resolve():
            continue
        entry = dict(path=str(source), status="rejected")
        state["cached_candidates"].append(entry)
        try:
            entry.update(inspect(source))
            if entry["actual_sha256"] != PACKAGE_SHA256:
                entry["error"] = "cached SHA256 mismatch"
                continue
            shutil.copyfile(source, package)
            copied = inspect(package)
            if copied["actual_sha256"] != PACKAGE_SHA256:
                raise RuntimeError("Copied compatibility package SHA256 mismatch")
            entry["status"] = "verified"
            state.update(
                status="verified",
                source_kind="verified_local_cache",
                source_path=str(source),
                **copied,
            )
            print(
                "Reusing SHA256-verified CUDA compatibility package:",
                source,
                flush=True,
            )
            return state
        except Exception as error:
            entry["error"] = str(error)
        finally:
            save()

    for attempt in range(1, 4):
        entry = dict(attempt=attempt, status="failed")
        state["download_attempts"].append(entry)
        try:
            package.unlink(missing_ok=True)
            response = download(package)
            if isinstance(response, dict):
                entry.update(response)
            entry.update(inspect(package))
            if entry["actual_sha256"] != PACKAGE_SHA256:
                raise RuntimeError(
                    "CUDA compatibility package SHA256 mismatch: "
                    f"expected={PACKAGE_SHA256}, actual={entry['actual_sha256']}, "
                    f"bytes={entry['bytes']}"
                )
            entry["status"] = "verified"
            state.update(
                status="verified",
                source_kind="download",
                source_path=PACKAGE_URL,
                bytes=entry["bytes"],
                actual_sha256=entry["actual_sha256"],
            )
            return state
        except Exception as error:
            entry["error"] = str(error)
            if package.is_file():
                entry.update(inspect(package))
            print(
                f"CUDA compatibility download attempt {attempt}/3 failed: {error}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            save()
    state["status"] = "failed"
    save()
    raise RuntimeError(
        "CUDA compatibility package preparation failed after3 attempts: "
        + state["download_attempts"][-1]["error"]
        + "; see compat134-package.json"
    )


def run(script, report, extra):
    try:
        prefix = Path(tempfile.mkdtemp(prefix="qsa-cuda134-compat."))
        package = prefix / "cuda-compat-13-4.deb"
        print("Preparing isolated CUDA 13.4 compat:", prefix, flush=True)
        verification = prepare_package(package, report)
        digest = file_sha256(package)
        if digest != PACKAGE_SHA256:
            raise RuntimeError("Compatibility package changed after verification")
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
            package_size_bytes=package.stat().st_size,
            package_source_kind=verification["source_kind"],
            package_source_path=verification["source_path"],
            package_verification_file=str(report / "compat134-package.json"),
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
