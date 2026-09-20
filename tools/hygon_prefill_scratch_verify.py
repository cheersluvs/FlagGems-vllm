"""Verify the Hygon production scratch-reuse route.

The standalone scratch probe established the wall-clock opportunity. This
probe checks the shipped dispatch with the cache disabled and enabled, runs
the full functional test, and repeats the public kernel benchmark in an
interleaved order. The benchmark is a device-time sanity check; the
allocation-inclusive numbers remain in hygon_prefill_scratch_reuse_v2.txt.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

from hygon_prefill_audit import emit, occupancy

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(label, command, enabled, timeout):
    env = dict(
        os.environ,
        FLAGGEMS_HYGON_TOPK_SCRATCH_REUSE="1" if enabled else "0",
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
    from importlib import import_module

    import flaggems_vllm

    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    if flaggems_vllm.top_k_per_row_prefill is not ov.top_k_per_row_prefill:
        raise RuntimeError("public prefill operator is not the Hygon override")
    if ov._scratch_reuse_enabled() != enabled:
        raise RuntimeError("scratch reuse environment switch was not honored")
    emit(
        "preflight",
        enabled=enabled,
        cache_entries=len(ov._SCRATCH_CACHE),
        cache_bytes=ov._SCRATCH_CACHE_BYTES,
        ok=True,
    )


def main():
    import torch  # noqa: F401

    emit(
        "probe",
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
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
    try:
        if len(sys.argv) == 2 and sys.argv[1] == "--preflight":
            preflight(
                os.environ.get("FLAGGEMS_HYGON_TOPK_SCRATCH_REUSE") == "1"
            )
            raise SystemExit(0)
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise