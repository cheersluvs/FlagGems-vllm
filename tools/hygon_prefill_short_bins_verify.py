"""Verify the public Hygon short-bins route against the preceding VEC2 path."""

import os
import pathlib
import platform
import subprocess
import sys

from hygon_prefill_audit import emit, occupancy

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(label, command, enabled, timeout):
    env = dict(
        os.environ,
        FLAGGEMS_HYGON_TOPK_SHORT_BINS="1" if enabled else "0",
        FLAGGEMS_HYGON_TOPK_VEC2="1",
    )
    emit("command", label=label, enabled=enabled, command=command)
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        timeout=timeout,
        text=True,
        capture_output=True,
    )
    print(proc.stdout, end="", flush=True)
    print(proc.stderr, end="", flush=True)
    emit("command_exit", label=label, enabled=enabled, code=proc.returncode)
    return proc.returncode


def preflight(enabled):
    import flaggems_vllm
    import vllm._custom_ops  # noqa: F401 - register the compiled baseline

    from importlib import import_module

    ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
    if flaggems_vllm.top_k_per_row_prefill is not ov.top_k_per_row_prefill:
        raise RuntimeError("public prefill operator is not the Hygon override")
    if enabled:
        if ov._dense_short_bins is None or ov._SHORT_BINS_PATH is None:
            raise RuntimeError("short-bins route did not load")
        source = pathlib.Path(ov._SHORT_BINS_PATH).read_text()
        if "bin_idx = (mapped >> 7).to(tl.uint32)" not in source:
            raise RuntimeError("short-bins source has no 512-bin STEP0 key")
        if "(512 if STEP == 0 else RADIX11_SIZE)" not in source:
            raise RuntimeError("short-bins source has no 512-bin radix width")
        emit(
            "preflight",
            enabled=True,
            ok=True,
            source=ov._SHORT_BINS_PATH,
            dense_module=ov._dense_short_bins.__file__,
        )
    else:
        if ov._dense_short_bins is not None or ov._SHORT_BINS_PATH is not None:
            raise RuntimeError("short-bins route remained loaded when disabled")
        emit("preflight", enabled=False, ok=True)


def main():
    import torch  # noqa: F401 - fail early outside the accelerator host

    emit(
        "probe",
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        host=platform.node(),
        env={k: os.environ.get(k) for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")},
    )
    occupancy("before")
    py = sys.executable
    self_test = [py, "-u", __file__, "--preflight"]
    for enabled in (False, True):
        if run("preflight", self_test, enabled, 1200):
            return 1
    if run(
        "functional",
        [py, "-m", "pytest", "-q", "tests/test_top_k_per_row_prefill.py"],
        True,
        1800,
    ):
        return 1

    benchmark = [
        py,
        "-m",
        "pytest",
        "-q",
        "-s",
        "benchmark/test_top_k_per_row_prefill.py",
        "--mode",
        "kernel",
    ]
    failures = []
    for index, enabled in enumerate((False, True, True, False)):
        code = run(f"benchmark_{index}", benchmark, enabled, 2400)
        if code:
            failures.append([index, code])
            break
    occupancy("after")
    emit("probe_complete", ok=not failures, failures=failures)
    return int(bool(failures))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--preflight":
        preflight(os.environ.get("FLAGGEMS_HYGON_TOPK_SHORT_BINS") == "1")
        raise SystemExit(0)
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise