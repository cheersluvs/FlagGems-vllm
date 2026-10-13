# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Check the current runtime's TLE and actual FlagGems persistent Top-K.

No packages/environment are changed. The parent bounds compilation/execution
time; a successful import alone does not establish kernel usability.
"""

import argparse
import importlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path


def check():
    print("python:", sys.executable, flush=True)
    print("prefix:", sys.prefix)
    print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
    print("sys.path:", sys.path)
    runtime = os.environ.get("QSA_TRITON_RUNTIME_PATH")
    if runtime:
        runtime_path = Path(runtime)
        if not runtime_path.is_absolute() or not (runtime_path / "triton").is_dir():
            print(
                "FAIL: explicit runtime must be an absolute directory containing triton/"
            )
            return 2
        sys.path.insert(0, str(runtime_path))
        print("Explicitly selected runtime:", runtime)
    try:
        import torch
        import triton
    except ImportError as exc:
        print("FAIL: current Python cannot import Torch/Triton:", exc, flush=True)
        return 2

    print("torch:", torch.__version__, torch.__file__)
    print("triton:", triton.__version__, triton.__file__)
    for name in ("torch", "triton", "flagtree"):
        try:
            print(name, "distribution:", importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            print(name, "distribution: absent")
    try:
        tle = importlib.import_module("triton.experimental.tle.language")
    except ImportError as exc:
        print("FAIL: current Triton has no importable TLE:", exc, flush=True)
        return 2
    print("TLE source:", tle.__file__)
    if runtime:
        assert Path(triton.__file__).resolve().is_relative_to(Path(runtime).resolve())
    assert (
        Path(tle.__file__)
        .resolve()
        .is_relative_to(Path(triton.__file__).resolve().parent)
    )
    print(
        "For the QSA runner, explicit runtime directory:",
        Path(triton.__file__).resolve().parents[1],
    )
    for name in ("alloc", "local_ptr", "smem"):
        if not hasattr(tle.gpu, name):
            print("FAIL: required tle.gpu API absent:", name, flush=True)
            return 2
    if not torch.cuda.is_available():
        print("FAIL: TLE imports, but CUDA execution is unavailable", flush=True)
        return 2
    print("device:", torch.cuda.get_device_name())
    if torch.cuda.get_device_capability() != (9, 0):
        print("FAIL: this QSA check requires SM90 H800/H100", flush=True)
        return 2
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    module = importlib.import_module("flaggems_vllm.ops.persistent_topk")
    print("persistent_topk source:", module.__file__)
    if not module.HAS_TLE:
        print("FAIL: persistent_topk's TLE >=3.6 gate is not satisfied", flush=True)
        return 2
    # Exercise both the persistent small-batch path and per-row CTA path.
    # Distinct exactly representable scores give an exact expected index set.
    for rows in (1, 33):
        columns, k = 1055, 512
        logits = torch.arange(columns, device="cuda", dtype=torch.float32)
        logits = logits.repeat(rows, 1)
        lengths = torch.full((rows,), columns, device="cuda", dtype=torch.int32)
        output = torch.empty((rows, k), device="cuda", dtype=torch.int32)
        workspace = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)
        for repeat in range(2):
            print("Testing persistent_topk rows:", rows, "repeat:", repeat, flush=True)
            module.persistent_topk(logits, lengths, output, workspace, k, columns)
            torch.cuda.synchronize()
            expected = torch.arange(
                columns - k, columns, device="cuda", dtype=torch.int32
            )
            torch.testing.assert_close(
                output.sort(dim=1).values, expected.expand(rows, -1), rtol=0, atol=0
            )
    print(
        "PASS: TLE imports and both persistent_topk dispatch paths execute", flush=True
    )
    print("This is readiness validation, not QSA accuracy/performance acceptance.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return check()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    print("Checking current environment; timeout:", args.timeout, "seconds", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            env=env,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired:
        print("FAIL: TLE/persistent_topk check timed out", flush=True)
        return 2
    if result.returncode:
        print("FAIL: TLE/persistent_topk worker exit:", result.returncode, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
