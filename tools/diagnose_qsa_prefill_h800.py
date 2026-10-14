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
    print("pid:", os.getpid())
    print("prefix:", sys.prefix)
    print("sys.path:", sys.path)
    runtime = os.environ.get("QSA_TRITON_RUNTIME_PATH")
    print("explicit TLE runtime:", runtime)
    if runtime:
        assert Path(triton.__file__).resolve().is_relative_to(Path(runtime).resolve())
    elif os.environ.get("QSA_DIAGNOSTIC_STOCK_RUNTIME") == "1":
        assert (
            Path(triton.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        )
        assert triton.__version__ == importlib.metadata.version("triton")
    if os.environ.get("QSA_DIAGNOSTIC_COMPAT") == "1":
        assert Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert torch.__version__ == importlib.metadata.version("torch")
        if runtime:
            from triton.experimental.tle import language as tle

            assert (
                Path(tle.__file__)
                .resolve()
                .is_relative_to(Path(triton.__file__).resolve().parent)
            )
            print("TLE source:", tle.__file__)
        if "vllm" in sys.modules:
            import vllm
            import vllm._C_stable_libtorch as native_extension

            assert importlib.metadata.version("vllm").split("+")[0] == "0.31.0"
            for module in (vllm, native_extension):
                source = Path(module.__file__).resolve()
                assert source.is_relative_to(Path(sys.prefix).resolve()), source
                print("native vLLM source:", source)
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


def child_qsa(args):
    import torch

    from tests.qsa_select_prefill_utils import (
        assert_selection,
        make_inputs,
        torch_logits,
        vllm_reference,
    )

    fingerprint("QSA before launches")
    # Reproduce the first failed native comparison: 1 MiB gives 32/32/9 rows.
    os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = "1"
    reference = vllm_reference()
    source = Path(reference.__file__)
    print("pinned upstream source:", source, flush=True)
    print(
        "pinned upstream SHA256:",
        hashlib.sha256(source.read_bytes()).hexdigest(),
        flush=True,
    )
    reference.envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB = 1
    inputs = make_inputs((39, 34), (4096, 8192), query_base=113)
    expected = torch_logits(*inputs[:5])
    torch.cuda.synchronize()
    print("input and Torch reference synchronized", flush=True)
    if args.warm_gems:
        import flaggems_vllm

        print("launch FlagGems QSA before upstream", flush=True)
        flaggems_vllm.qsa_select_paged_prefill(*inputs)
        torch.cuda.synchronize()
        assert_selection(expected, inputs[4], inputs[8])
        print("FlagGems QSA synchronized and validated", flush=True)

    original_topk = reference._topk
    chunk = 0

    def synchronized_topk(*topk_args, **kwargs):
        nonlocal chunk
        print("upstream chunk:", chunk, "score synchronization begins", flush=True)
        torch.cuda.synchronize()
        print("upstream score synchronized; native Top-K begins", flush=True)
        original_topk(*topk_args, **kwargs)
        torch.cuda.synchronize()
        print("upstream native Top-K synchronized", flush=True)
        chunk += 1

    # Add observation boundaries only in this diagnostic process; dispatch,
    # kernels and the hash-pinned source remain unchanged. Never time this run.
    reference._topk = synchronized_topk
    print("launch upstream QSA", flush=True)
    reference.qsa_select_paged_prefill(*inputs)
    torch.cuda.synchronize()
    assert_selection(expected, inputs[4], inputs[8])
    print("PASS: upstream QSA synchronized and validated", flush=True)


def pytest_collection_finish(session):
    # Loaded only via pytest -p below. Observe the actual test process after
    # conftest/collection imports; do not synchronize or change execution.
    fingerprint("pytest collection finished")


def child_compat_qsa(args):
    """Alternate both implementations in one CUDA context, without patching vLLM."""
    import torch

    import flaggems_vllm
    from tests.qsa_select_prefill_utils import (
        assert_native_selection,
        assert_selection,
        assert_upstream_selection,
        make_inputs,
        torch_logits,
        vllm_reference,
    )

    fingerprint("same-process QSA compatibility")
    assert torch.cuda.get_device_capability() == (9, 0)
    assert importlib.metadata.version("vllm").split("+")[0] == "0.31.0"
    # Set envs after import too: vLLM may cache its environment lookups.
    reference = vllm_reference()
    fingerprint("after original native QSA import")
    torch.backends.cuda.matmul.allow_tf32 = False
    print("original pinned QSA:", reference.__file__, flush=True)
    print("original Top-K dispatch:", reference._topk, flush=True)
    order = ("native", "gems") if args.native_first else ("gems", "native")
    calls = {
        "gems": flaggems_vllm.qsa_select_paged_prefill,
        "native": reference.qsa_select_paged_prefill,
    }
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        for rows in (3, 65):
            for k in (512, 1024, 2048):
                for budget in (512, 1):
                    os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = str(budget)
                    reference.envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB = budget
                    inputs = make_inputs((rows,), (8192,), dtype=dtype, k=k)
                    expected = torch_logits(*inputs[:5])
                    outputs = {name: torch.empty_like(inputs[8]) for name in order}
                    torch.cuda.synchronize()
                    for repeat in range(2):
                        for name in order:
                            launch_args = list(inputs)
                            launch_args[8] = outputs[name]
                            print(
                                "launch:",
                                name,
                                "dtype:",
                                dtype,
                                "rows:",
                                rows,
                                "K:",
                                k,
                                "budget MiB:",
                                budget,
                                "repeat:",
                                repeat,
                                flush=True,
                            )
                            calls[name](*launch_args)
                            torch.cuda.synchronize()
                            if dtype == torch.float8_e4m3fn:
                                assert_upstream_selection(
                                    expected, inputs[4], outputs[name], dtype=dtype
                                )
                            else:
                                assert_selection(expected, inputs[4], outputs[name])
                            print(name, "synchronized and validated", flush=True)
                        assert_native_selection(
                            expected,
                            inputs[4],
                            outputs["gems"],
                            outputs["native"],
                            dtype=dtype,
                            head_dim=128,
                        )
    print("PASS: repeated same-process QSA in order", order, flush=True)


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
        PYTHONNOUSERSITE="1",
    )
    env.pop("PYTHONHOME", None)
    if mode == "compat":
        # Reject subprocess adapters rather than claiming compatibility from
        # comparisons that actually execute native vLLM in another process.
        if os.environ.get("QSA_NATIVE_REFERENCE_PYTHON"):
            raise ValueError("--mode compat cannot use QSA_NATIVE_REFERENCE_PYTHON")
        for name in ("QSA_NATIVE_REPORT_DIR", "QSA_DIAGNOSTIC_STOCK_RUNTIME"):
            env.pop(name, None)
        env["QSA_DIAGNOSTIC_COMPAT"] = "1"
    runtime = os.environ.get("QSA_TRITON_RUNTIME_PATH")
    if runtime:
        runtime = str(Path(runtime).resolve())
        if not Path(runtime, "triton").is_dir():
            raise ValueError("QSA_TRITON_RUNTIME_PATH must contain triton/")
        env["PYTHONPATH"] += ":" + runtime
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
    if mode == "tle":
        if not runtime:
            raise ValueError("--mode tle needs explicit QSA_TRITON_RUNTIME_PATH")
        jobs = [
            ("stock-cooperative-topk", probe + ["--child", "topk"]),
            ("stock-upstream-qsa", probe + ["--child", "qsa"]),
            ("tle-cooperative-topk", probe + ["--child", "topk"]),
            (
                "tle-persistent-topk-native",
                probe + ["--child", "topk", "--op", "persistent_topk"],
            ),
            ("tle-upstream-qsa", probe + ["--child", "qsa"]),
            (
                "tle-gems-then-upstream-qsa",
                probe + ["--child", "qsa", "--warm-gems"],
            ),
            (
                "tle-combined-first-vllm",
                pytest
                + [
                    tests,
                    "-k",
                    "not upstream and not dim8 and not noncurrent",
                    "--quick",
                ],
            ),
        ]
    if mode == "compat":
        if not runtime:
            raise ValueError("--mode compat needs explicit QSA_TRITON_RUNTIME_PATH")
        jobs = [
            ("stock-cooperative-topk", probe + ["--child", "topk"]),
            (
                "stock-persistent-topk",
                probe + ["--child", "topk", "--op", "persistent_topk", "--rows", "65"],
            ),
            ("stock-upstream-qsa", probe + ["--child", "qsa"]),
            ("tle-environment", probe + ["--child", "environment"]),
            (
                "tle-cooperative-topk-with-gems",
                probe + ["--child", "topk", "--import-gems"],
            ),
            (
                "tle-persistent-topk-with-gems",
                probe
                + [
                    "--child",
                    "topk",
                    "--op",
                    "persistent_topk",
                    "--rows",
                    "65",
                    "--import-gems",
                ],
            ),
            ("tle-gems-then-native-qsa", probe + ["--child", "compat-qsa"]),
            (
                "tle-native-then-gems-qsa",
                probe + ["--child", "compat-qsa", "--native-first"],
            ),
            (
                "tle-accuracy-combined-full",
                pytest + [tests, "--record", "json"],
            ),
        ]
    outcomes = []
    modes = (
        ("async", "sync")
        if mode in ("matrix", "compat")
        else ("async" if mode == "tle" else mode,)
    )
    (report / "launch-environment.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "python": sys.executable,
                "inherited_python": os.environ.get("PYTHON"),
                "inherited_pythonpath": os.environ.get("PYTHONPATH"),
                "inherited_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
                "explicit_tle_runtime": runtime,
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
        if "-stock-" in name:
            job_env["PYTHONPATH"] = f"{root / 'src'}:{root}"
            job_env.pop("QSA_TRITON_RUNTIME_PATH", None)
            job_env["QSA_DIAGNOSTIC_STOCK_RUNTIME"] = "1"
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
                + "\nPYTHONPATH="
                + job_env["PYTHONPATH"]
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
    if mode == "compat":
        stock = [item for item in outcomes if "-stock-" in item["name"]]
        tle = [item for item in outcomes if "-tle-" in item["name"]]
        (report / "compatibility-summary.json").write_text(
            json.dumps(
                {
                    "scope": "same-process FlagTree/native-vLLM functional compatibility; no timing",
                    "stock_controls_passed": all(
                        item["returncode"] == 0 for item in stock
                    ),
                    "async_tle_passed": all(
                        item["returncode"] == 0
                        for item in tle
                        if item["name"].startswith("async-")
                    ),
                    "sync_tle_passed": all(
                        item["returncode"] == 0
                        for item in tle
                        if item["name"].startswith("sync-")
                    ),
                    "cases": outcomes,
                },
                indent=2,
            )
            + "\n"
        )
    print("Diagnostics finished. Return all text/JSON files in:", report, flush=True)
    print(
        "No performance measurements; upstream dispatch was not modified.", flush=True
    )
    return int(any(item["returncode"] != 0 for item in outcomes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("sync", "async", "matrix", "tle", "compat"), default="sync"
    )
    parser.add_argument(
        "--child",
        choices=("environment", "topk", "qsa", "compat-qsa"),
        help=argparse.SUPPRESS,
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
    parser.add_argument("--warm-gems", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--native-first", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    max_rows = 64 if args.op == "cooperative_topk" else 2048
    if args.cols < args.topk or args.cols % 4 or not 1 <= args.rows <= max_rows:
        parser.error(
            f"require cols >= K, cols divisible by 4, and 1 <= rows <= {max_rows}"
        )
    if args.child == "environment":
        child_environment()
        return 0
    if args.child == "topk":
        child_topk(args)
        return 0
    if args.child == "qsa":
        child_qsa(args)
        return 0
    if args.child == "compat-qsa":
        child_compat_qsa(args)
        return 0
    return diagnose(args.mode)


if __name__ == "__main__":
    sys.exit(main())
