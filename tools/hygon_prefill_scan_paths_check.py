"""Stdlib-only source-drift check for the two BW1000 scan-path variants."""

import ast
import importlib.util
from pathlib import Path

from hygon_prefill_scan_paths_source import (
    fullrow_variant,
    function_text,
    wave64_variant,
)

ROOT = Path(__file__).resolve().parents[1]
FUSED = ROOT / "src/flaggems_vllm/runtime/backend/_hygon/fused"
GENERIC = ROOT / "src/flaggems_vllm/ops/top_k_per_row_prefill.py"
OVERRIDE = FUSED / "top_k_per_row_prefill.py"


def module_from_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    carry = module_from_file(FUSED / "_top_k_per_row_prefill_carry_source.py", "carry")
    final = module_from_file(FUSED / "_top_k_per_row_prefill_final_source.py", "final")
    generic = GENERIC.read_text()
    override = OVERRIDE.read_text()
    names = {
        "_ONESCAN_CLEAR_OLD",
        "_ONESCAN_CLEAR_NEW",
        "_ONESCAN_SCAN_OLD",
        "_ONESCAN_SCAN_NEW",
    }
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(override).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    }
    if set(values) != names:
        raise ValueError("One-scan source constants drifted")
    for old, new in (
        ("_ONESCAN_CLEAR_OLD", "_ONESCAN_CLEAR_NEW"),
        ("_ONESCAN_SCAN_OLD", "_ONESCAN_SCAN_NEW"),
    ):
        if generic.count(values[old]) != 1:
            raise ValueError("Generic one-scan source drifted")
        generic = generic.replace(values[old], values[new], 1)
    dense = carry.set_vector_width(carry.build_carry_source(generic, override), 2)
    if dense.count("bin_idx = (mapped >> 5).to(tl.uint32)") != 1:
        raise ValueError("Short-bin key source drifted")
    if (
        dense.count(
            "RADIX_SIZE: tl.constexpr = RADIX10_SIZE if STEP == 3 else RADIX11_SIZE"
        )
        != 1
    ):
        raise ValueError("Short-bin radix source drifted")
    short = dense.replace(
        "bin_idx = (mapped >> 5).to(tl.uint32)",
        "bin_idx = (mapped >> 7).to(tl.uint32)",
        1,
    ).replace(
        "RADIX_SIZE: tl.constexpr = RADIX10_SIZE if STEP == 3 else RADIX11_SIZE",
        "RADIX_SIZE: tl.constexpr = (RADIX10_SIZE if STEP == 3 else "
        "(512 if STEP == 0 else RADIX11_SIZE))",
        1,
    )
    sources = {
        "sparse": generic,
        "dense": dense,
        "final": final.build_final_source(dense),
        "short": short,
    }
    for name, source in sources.items():
        result = fullrow_variant(source)
        if result == source or result.count("tl.assume(skip_elems == 0)") != 1:
            raise AssertionError(f"Full-row variant was not applied: {name}")
        if name != "sparse":
            step = function_text(result, "_process_histogram_step")
            if step.count("slot_base = _process_bins(") != 8:
                raise AssertionError(f"Dense tail slot carry was not applied: {name}")
        print(f"fullrow {name}: source parses, {len(result)} bytes")
    for name in ("dense", "final", "short"):
        source = sources[name]
        result = wave64_variant(source)
        if result == source or result.count("N_WAVES: tl.constexpr") != 1:
            raise AssertionError(f"Wave64 variant was not applied: {name}")
        print(f"wave64 {name}: source parses, {len(result)} bytes")


if __name__ == "__main__":
    main()
