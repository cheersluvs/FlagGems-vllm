# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Isolate QSA failures in fresh processes; this is not a performance run."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def fingerprint(stage):
    import torch
    import triton

    print("stage:", stage, flush=True)
    print("python:", sys.executable)
    print("prefix:", sys.prefix)
    print("sys.path:", sys.path)
    for name in (
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONNOUSERSITE",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_LAUNCH_BLOCKING",
        "TRITON_CACHE_DIR",
    ):
        print(name + ":", os.environ.get(name))
    for name, module in (("torch", torch), ("triton", triton)):
        path = Path(module.__file__).resolve()
        print(name, "module version:", module.__version__)
        print(name, "distribution version:", importlib.metadata.version(name))
        print(name, "source:", path)
        print(name, "source SHA256:", hashlib.sha256(path.read_bytes()).hexdigest())
    print("torch CUDA:", torch.version.cuda, flush=True)


def child_environment():
    import torch

    fingerprint("before FlagGems import")
    import flaggems_vllm

    print("FlagGems source:", flaggems_vllm.__file__)
    fingerprint("after FlagGems import")
    import vllm
    import vllm._custom_ops  # noqa: F401

    fingerprint("after vLLM import")
    print("vLLM source:", vllm.__file__)
    print("vLLM distribution version:", importlib.metadata.version("vllm"))
    reference = (
        Path(vllm.__file__).parent / "models/qwen4_exp/nvidia/ops/qsa_indexer.py"
    )
    print("installed QSA source:", reference)
    print("installed QSA SHA256:", hashlib.sha256(reference.read_bytes()).hexdigest())
    torch.cuda.init()
    print("device:", torch.cuda.get_device_name())
    print("capability:", torch.cuda.get_device_capability())
    maps = Path("/proc/self/maps")
    if maps.exists():
        libraries = sorted(
            {
                line.split()[-1]
                for line in maps.read_text().splitlines()
                if any(
                    name in line
                    for name in ("libcuda.", "libnvidia-ptxjit", "libnvvm.")
                )
            }
        )
        print("loaded CUDA driver libraries:", libraries)


def child_topk(args):
    import torch

    fingerprint("before imports")
    if args.import_gems:
        import flaggems_vllm  # noqa: F401

        fingerprint("after FlagGems import")
    import vllm._custom_ops  # noqa: F401

    fingerprint("after vLLM import")
    assert torch.cuda.get_device_capability() == (9, 0), "SM90 device required"
    # Exactly representable, distinct, nonnegative scores. No QSA score kernel
    # runs here, so its asynchronous errors cannot be attributed to this Top-K.
    logits = torch.arange(args.cols, device="cuda", dtype=torch.float32)
    logits = logits.repeat(args.rows, 1)
    lengths = torch.full((args.rows,), args.cols, device="cuda", dtype=torch.int32)
    output = torch.empty((args.rows, args.topk), device="cuda", dtype=torch.int32)
    workspace = torch.empty(16 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    torch.cuda.synchronize()
    print("input preparation synchronized", flush=True)
    print("launch:", args.op, "shape:", logits.shape, "K:", args.topk, flush=True)
    getattr(torch.ops._C, args.op)(
        logits, lengths, output, workspace, args.topk, args.cols
    )
    torch.cuda.synchronize()
    print("Top-K synchronized", flush=True)
    actual = output.cpu().sort(dim=1).values
    expected = torch.arange(args.cols - args.topk, args.cols, dtype=torch.int32)
    torch.testing.assert_close(actual, expected.expand(args.rows, -1), rtol=0, atol=0)
    print("PASS: exact selected index sets", flush=True)


def pytest_collection_finish(session):
    # Loaded only via pytest -p below. Observe the actual test process after
    # conftest/collection imports; do not synchronize or change execution.
    fingerprint("pytest collection finished")


def diagnose(mode):
    root = Path(__file__).resolve().parents[1]
    report = Path(tempfile.mkdtemp(prefix="qsa-prefill-h800."))
    print("Reports:", report, flush=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True)
    (report / "commit.txt").write_text(commit)
    status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True)
    (report / "worktree-status.txt").write_text(status)
    env = os.environ.copy()
    env.update(
        PYTHONPATH=f"{root / 'src'}:{root}",
        PYTHONUNBUFFERED="1",
        DNN_VENDOR="nvidia",
        VLLM_PLUGINS="",
        QSA_REQUIRE_VLLM_REFERENCE="1",
    )
    probe = [sys.executable, str(Path(__file__).resolve())]
    pytest = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-ra",
        "-x",
        "-p",
        "tools.diagnose_qsa_prefill_h800",
    ]
    tests = "tests/test_qsa_select_paged_prefill.py"
    jobs = [
        ("environment", probe + ["--child", "environment"]),
        (
            "environment-stdin-path",
            [
                sys.executable,
                "-c",
                "from tools.diagnose_qsa_prefill_h800 import child_environment; child_environment()",
            ],
        ),
        (
            "environment-runner-path",
            [
                sys.executable,
                "-c",
                "from tools.diagnose_qsa_prefill_h800 import child_environment; child_environment()",
            ],
        ),
        ("cooperative-topk", probe + ["--child", "topk"]),
        ("persistent-topk", probe + ["--child", "topk", "--op", "persistent_topk"]),
        ("cooperative-topk-with-gems", probe + ["--child", "topk", "--import-gems"]),
        (
            "accuracy-full-torch",
            pytest
            + [
                tests,
                "-k",
                "not vllm",
                "--record",
                "json",
                "--output",
                str(report / "accuracy-full-torch.json"),
            ],
        ),
        (
            "accuracy-first-vllm",
            pytest
            + [
                tests + "::test_against_pinned_vllm",
                "--quick",
                "--record",
                "json",
                "--output",
                str(report / "accuracy-first-vllm.json"),
            ],
        ),
        (
            "accuracy-combined-quick",
            pytest + [tests, "--quick", "--record", "json"],
        ),
    ]
    outcomes = []
    modes = ("async", "sync") if mode == "matrix" else (mode,)
    (report / "launch-environment.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "python": sys.executable,
                "inherited_python": os.environ.get("PYTHON"),
                "inherited_pythonpath": os.environ.get("PYTHONPATH"),
                "inherited_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
            },
            indent=2,
        )
        + "\n"
    )
    scheduled = []
    for launch_mode in modes:
        for name, command in jobs:
            if (
                mode == "matrix"
                and launch_mode == "sync"
                and name
                not in (
                    "environment",
                    "environment-stdin-path",
                    "environment-runner-path",
                    "accuracy-combined-quick",
                )
            ):
                continue  # Prior report already covered isolated sync checks.
            command = list(command)
            name = f"{launch_mode}-{name}"
            if "--output" in command:
                command[command.index("--output") + 1] = str(report / f"{name}.json")
            elif "--record" in command:
                command += ["--output", str(report / f"{name}.json")]
            scheduled.append((launch_mode, name, command))
    for launch_mode, name, command in scheduled:
        if launch_mode == "sync":
            env["CUDA_LAUNCH_BLOCKING"] = "1"
        else:
            env.pop("CUDA_LAUNCH_BLOCKING", None)
        job_env = env.copy()
        if name.endswith("environment-runner-path"):
            inherited_path = os.environ.get("PYTHONPATH")
            if inherited_path:
                job_env["PYTHONPATH"] += ":" + inherited_path
        print("Running in fresh process:", name, flush=True)
        with (report / f"{name}.txt").open("w") as log:
            log.write(
                "command: "
                + repr(command)
                + "\nCUDA_LAUNCH_BLOCKING="
                + str(env.get("CUDA_LAUNCH_BLOCKING"))
                + "\n"
            )
            log.flush()
            try:
                result = subprocess.run(
                    command,
                    cwd=root,
                    env=job_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=900,
                )
                code = result.returncode
            except subprocess.TimeoutExpired:
                log.write("\nTIMEOUT after 900 seconds\n")
                code = 124
        outcomes.append({"name": name, "returncode": code})
        print(name, "returncode:", code, flush=True)
    (report / "diagnostic-summary.json").write_text(
        json.dumps(outcomes, indent=2) + "\n"
    )
    print("Diagnostics finished. Return all text/JSON files in:", report, flush=True)
    print(
        "No performance measurements; upstream dispatch was not modified.", flush=True
    )
    return int(any(item["returncode"] != 0 for item in outcomes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sync", "async", "matrix"), default="sync")
    parser.add_argument(
        "--child", choices=("environment", "topk"), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--op",
        choices=("cooperative_topk", "persistent_topk"),
        default="cooperative_topk",
    )
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=8192)
    parser.add_argument("--topk", type=int, choices=(512, 1024, 2048), default=512)
    parser.add_argument("--import-gems", action="store_true")
    args = parser.parse_args()
    if args.cols < args.topk or args.cols % 4 or not 1 <= args.rows <= 64:
        parser.error("require cols >= K, cols divisible by 4, and 1 <= rows <= 64")
    if args.child == "environment":
        child_environment()
        return 0
    if args.child == "topk":
        child_topk(args)
        return 0
    return diagnose(args.mode)


if __name__ == "__main__":
    sys.exit(main())
