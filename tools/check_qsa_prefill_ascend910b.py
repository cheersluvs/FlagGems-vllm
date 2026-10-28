#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Collect 910B prerequisites; isolate native imports and optional tiny probe.

No packages are installed, no services are stopped and no compiler is modified.
--probe launches one tiny BF16 scoring tile on the selected logical NPU.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

PACKAGES = ("torch", "torch-npu", "triton", "triton-ascend", "flagtree", "vllm")


def worker(args):
    result = {"status": "not_ready", "python": sys.executable, "modules": {}}
    try:
        # Explicit NPU initialization precedes Triton to avoid circular imports.
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        import torch
        import torch_npu
        import triton

        for name, module in (
            ("torch", torch),
            ("torch_npu", torch_npu),
            ("triton", triton),
        ):
            result["modules"][name] = {
                "version": str(getattr(module, "__version__", "unknown")),
                "file": str(Path(module.__file__).resolve()),
            }
        result["npu_available"] = torch.npu.is_available()
        result["device_count"] = torch.npu.device_count()
        if not result["npu_available"] or args.device >= result["device_count"]:
            raise RuntimeError("Requested logical NPU is unavailable")
        result["device_name"] = torch.npu.get_device_name(args.device)
        try:
            import triton.experimental.tle.language as tle

            result["tle"] = {
                "importable": True,
                "file": getattr(tle, "__file__", None),
            }
        except ImportError:
            result["tle"] = {"importable": False}
        result["lightning_indexer_schema_present"] = hasattr(
            torch.ops.npu, "npu_lightning_indexer"
        )
        result["e3_schema_present"] = hasattr(torch.ops._C_ascend, "qsa_expand_e3_out")
        result["schema_note"] = (
            "Presence does not establish 910B support; E3 extension is not loaded here"
        )
        if args.probe:
            path = Path(__file__).with_name("qsa_prefill_ascend910b_probe.py")
            spec = importlib.util.spec_from_file_location("_qsa_910b_probe", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            result["probe"] = module.run(args.device)
        result["status"] = "probe_passed" if args.probe else "imports_passed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    Path(args.worker).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] != "not_ready" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="qsa-ascend910b-env")
    parser.add_argument("--device", type=int, default=0, help="logical NPU index")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device < 0 or args.timeout <= 0:
        parser.error("device must be nonnegative and timeout must be positive")
    if args.worker:
        return worker(args)
    report = Path(args.report).resolve()
    report.mkdir(parents=True, exist_ok=True)
    result = {
        "python": sys.executable,
        "python_version": sys.version,
        "packages": {},
        "files": {},
        "environment": {
            name: os.environ.get(name)
            for name in (
                "ASCEND_HOME_PATH",
                "ASCEND_TOOLKIT_HOME",
                "ASCEND_RT_VISIBLE_DEVICES",
            )
        },
    }
    for name in PACKAGES:
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    for filename in (
        "/etc/os-release",
        "/usr/local/Ascend/driver/version.info",
        "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
    ):
        path = Path(filename)
        try:
            result["files"][filename] = (
                path.read_text()[:16384] if path.is_file() else None
            )
        except OSError as exc:
            result["files"][filename] = str(exc)
    smi = shutil.which("npu-smi")
    if smi:
        try:
            run = subprocess.run(
                [smi, "info"], capture_output=True, text=True, timeout=15
            )
            result["npu_smi"] = {
                "exit_code": run.returncode,
                "stdout": run.stdout,
                "stderr": run.stderr,
            }
        except subprocess.TimeoutExpired:
            result["npu_smi"] = {"error": "15-second timeout"}
    else:
        result["npu_smi"] = {"error": "npu-smi not found"}
    worker_json = report / "worker.json"
    if worker_json.exists():
        parser.error("report already contains worker.json; use a new report directory")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(worker_json),
        "--device",
        str(args.device),
    ]
    if args.probe:
        command.append("--probe")
    try:
        run = subprocess.run(
            command, capture_output=True, text=True, timeout=args.timeout
        )
        result["worker_exit_code"] = run.returncode
        (report / "worker.stdout.txt").write_text(run.stdout)
        (report / "worker.stderr.txt").write_text(run.stderr)
        result["worker"] = (
            json.loads(worker_json.read_text())
            if worker_json.is_file()
            else {"error": "worker exited without JSON"}
        )
    except subprocess.TimeoutExpired as exc:
        result["worker"] = {"error": f"worker exceeded {args.timeout} seconds"}
        (report / "worker.stdout.txt").write_bytes(exc.stdout or b"")
        (report / "worker.stderr.txt").write_bytes(exc.stderr or b"")
    passed = result["worker"].get("status") in ("imports_passed", "probe_passed")
    result["status"] = "passed" if passed else "not_ready"
    (report / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Report: {report}", flush=True)
    print(
        "No environment changes were made. Imports/probe do not validate the production operator.",
        flush=True,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
