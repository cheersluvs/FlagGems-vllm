"""Validate the public Hygon final-network route and benchmark it against off.

The report runner invokes this on BW1000 and uploads the complete output.
"""

import hashlib
import os
import platform
import subprocess
import sys
import time
import traceback
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

from hygon_prefill_audit import SHAPES, check_output, emit, inputs, occupancy, oracle

ROOT = Path(__file__).resolve().parents[1]
OVERRIDE = "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"


def run(label, command, enabled, timeout):
    env = dict(
        os.environ,
        FLAGGEMS_FORCE_TLE="0",
        FLAGGEMS_HYGON_TOPK_FINAL_NETWORK="1" if enabled else "0",
        FLAGGEMS_HYGON_TOPK_VEC2="1",
        FLAGGEMS_HYGON_TOPK_SHORT_BINS="1",
        FLAGGEMS_HYGON_TOPK_SLOTSCAN="1",
        FLAGGEMS_HYGON_TOPK_GEOMETRY="1",
        FLAGGEMS_HYGON_TOPK_SCRATCH_REUSE="1",
    )
    emit("production_command", label=label, enabled=enabled, command=command)
    started = time.monotonic()
    try:
        code = subprocess.run(command, cwd=ROOT, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        code = 124
    emit(
        "production_command_exit",
        label=label,
        enabled=enabled,
        code=code,
        elapsed_s=round(time.monotonic() - started, 1),
    )
    return code


def preflight(enabled):
    import torch

    flaggems_vllm = import_module("flaggems_vllm")
    import_module("vllm._custom_ops")  # loads the C++ benchmark baseline

    if flaggems_vllm.vendor_name != "hygon":
        raise RuntimeError(f"Expected Hygon, got {flaggems_vllm.vendor_name}")
    if not hasattr(torch.ops._C, "top_k_per_row_prefill"):
        raise RuntimeError("vLLM C++ baseline is unavailable")
    ov = import_module(OVERRIDE)
    if flaggems_vllm.top_k_per_row_prefill is not ov.top_k_per_row_prefill:
        raise RuntimeError("Public prefill entry is not the Hygon override")
    if not ov._scratch_reuse_enabled():
        raise RuntimeError("Public scratch reuse is disabled")
    if enabled:
        if ov._dense_vec2_final is None or ov._VEC2_FINAL_PATH is None:
            raise RuntimeError("Final-network production module did not load")
        if ov._dense_vec2_final.HAS_TLE:
            raise RuntimeError("Expected non-TLE final-network module")
        source = Path(ov._VEC2_FINAL_PATH).read_text()
        if "_hygon_final_network(" not in source:
            raise RuntimeError("Loaded module has no final-network selector")
        sha = hashlib.sha256(source.encode()).hexdigest()
        for shape_id in (2, 4, 5):
            rows, vocab, k, _ = SHAPES[shape_id]
            meta = SimpleNamespace(shape=(rows, vocab), dtype=torch.float32)
            if ov._select_module(meta, rows, k) is not ov._dense_vec2_final:
                raise RuntimeError(f"Shape {shape_id} missed the final-network route")
        rows, vocab, k, _ = SHAPES[6]
        meta = SimpleNamespace(shape=(rows, vocab), dtype=torch.float32)
        if ov._select_module(meta, rows, k) is ov._dense_vec2_final:
            raise RuntimeError("Short-bin shape incorrectly selected final network")
    else:
        if ov._dense_vec2_final is not None or ov._VEC2_FINAL_PATH is not None:
            raise RuntimeError("Final-network module loaded while disabled")
        for shape_id in (2, 4, 5):
            rows, vocab, k, _ = SHAPES[shape_id]
            meta = SimpleNamespace(shape=(rows, vocab), dtype=torch.float32)
            if ov._select_module(meta, rows, k) is not ov._dense_vec2:
                raise RuntimeError(f"Shape {shape_id} missed the VEC2 control route")
        sha = None
    emit("production_preflight", enabled=enabled, ok=True, source_sha256=sha)


def validate_public():
    import torch

    import flaggems_vllm

    # Each call reaches the production row-count route and repeats with the
    # same buffers, including the public scratch-cache hot path.
    cases = [(s, "normal", 42) for s in (2, 4, 5)]
    cases += [
        (4, name, 123)
        for name in ("tied", "constant", "partial", "short", "special", "strided")
    ]
    for shape_id, case, seed in cases:
        rows, vocab, k, stride = SHAPES[shape_id]
        tensors = inputs(rows, vocab, stride, k, seed, case)
        x, starts, ends = tensors
        want = oracle(tensors, k)
        guard = torch.empty((rows * k + 32,), device=x.device, dtype=torch.int32)
        out = guard[16:-16].view(rows, k)
        guard.fill_(-123456)
        for repeat in range(2):
            out.fill_(-9)
            flaggems_vllm.top_k_per_row_prefill(
                x, starts, ends, out, rows, x.stride(0), x.stride(1), k
            )
            torch.cuda.synchronize()
            if not bool((guard[:16] == -123456).all() & (guard[-16:] == -123456).all()):
                raise AssertionError("Public output guard overwritten")
            check_output(out, tensors, k, want)
            emit(
                "production_validation",
                shape_id=shape_id,
                case=case,
                repeat=repeat,
                ok=True,
            )


def main():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    emit("production_probe", commit=commit, host=platform.node())
    occupancy("before")
    py = sys.executable
    for enabled in (False, True):
        command = [py, "-u", __file__, "--preflight"]
        if run("preflight", command, enabled, 300):
            return 1
    if run("public_validation", [py, "-u", __file__, "--validate"], True, 1800):
        return 1
    if run(
        "functional_tests",
        [py, "-m", "pytest", "-q", "tests/test_top_k_per_row_prefill.py", "--quick"],
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
    for index, enabled in enumerate((False, True, True, False)):
        if run(f"official_benchmark_{index}", benchmark, enabled, 1800):
            return 1
    occupancy("after")
    emit("production_probe_complete", ok=True)
    return 0


if __name__ == "__main__":
    try:
        if len(sys.argv) == 2 and sys.argv[1] == "--preflight":
            preflight(os.environ.get("FLAGGEMS_HYGON_TOPK_FINAL_NETWORK") == "1")
        elif len(sys.argv) == 2 and sys.argv[1] == "--validate":
            validate_public()
        else:
            raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
