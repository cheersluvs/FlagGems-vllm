"""Confirm the gated dense VEC=2 path with the existing vLLM benchmark."""

import os
import subprocess
import sys

from hygon_prefill_audit import emit


def preflight(mode):
    from importlib import import_module
    from pathlib import Path

    import flaggems_vllm
    import torch
    import vllm._custom_ops  # noqa: F401 - registers the compiled baseline

    from flaggems_vllm.runtime.backend._hygon.fused._top_k_per_row_prefill_carry_source import (
        set_vector_width,
    )

    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    if flaggems_vllm.vendor_name != "hygon":
        raise RuntimeError("Expected Hygon override")
    if flaggems_vllm.top_k_per_row_prefill is not ov.top_k_per_row_prefill:
        raise RuntimeError("Public prefill operator is not the Hygon override")
    if ov._dense_carry is None or ov._CARRY_PATH is None:
        raise RuntimeError("Expected carried VEC=4 control")
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "_C::top_k_per_row_prefill", "CUDA"
    ):
        raise RuntimeError("Compiled vLLM prefill baseline absent")
    if mode == "candidate":
        if ov._dense_vec2 is None or ov._VEC2_PATH is None:
            raise RuntimeError("Gated VEC=2 candidate was not loaded")
        control = Path(ov._CARRY_PATH).read_text()
        candidate = Path(ov._VEC2_PATH).read_text()
        if candidate != set_vector_width(control, 2):
            raise RuntimeError("Loaded VEC=2 source differs from isolated variant")
    elif ov._dense_vec2 is not None:
        raise RuntimeError("VEC=2 unexpectedly enabled in control")
    emit("vec2_preflight", mode=mode, ok=True)


def run(label, command, candidate=False):
    env = dict(os.environ, FLAGGEMS_HYGON_TOPK_VEC2="1" if candidate else "0")
    emit("vec2_command", label=label, candidate=candidate, command=command)
    try:
        code = subprocess.run(command, env=env, timeout=1200).returncode
    except subprocess.TimeoutExpired:
        code = 124
    emit("vec2_command_exit", label=label, code=code)
    return code


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--preflight":
        preflight("candidate" if os.environ.get("FLAGGEMS_HYGON_TOPK_VEC2") == "1"
                  else "control")
        return 0
    if len(sys.argv) != 1:
        raise SystemExit("Usage: python tools/hygon_prefill_vec2_trial.py [--preflight]")
    py = sys.executable
    self_cmd = [py, "-u", __file__, "--preflight"]
    for candidate in (False, True):
        if run("preflight", self_cmd, candidate):
            return 1
    test = [py, "-m", "pytest", "-q", "tests/test_top_k_per_row_prefill.py"]
    if run("functional", test, candidate=True):
        return 1
    bench = [py, "-m", "pytest", "-q", "-s",
             "benchmark/test_top_k_per_row_prefill.py", "--mode", "kernel"]
    failures = []
    for i, candidate in enumerate((False, True, True, False)):
        code = run(f"benchmark_{i}_{'vec2' if candidate else 'vec4'}", bench, candidate)
        if code:
            failures.append([i, code])
            break
    emit("vec2_suite_summary", failures=failures, order="VEC4-VEC2-VEC2-VEC4")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
