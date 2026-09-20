"""Find the short-row STEP-0 bin crossover for the current Hygon dense path.

The previous short-special run found a large win around row_len=1025 but a
loss at 4095 and 5115.  This probe fills the missing interval with the same
production dense VEC=2 source, the same geometry, and the same four isolated
bin widths.  It does not modify the operator or reduce scratch allocation.
"""

import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
from importlib import import_module

import torch

from hygon_prefill_audit import emit, occupancy
from hygon_prefill_short_special import (
    BIN_CANDIDATES,
    check_values,
    configure,
    device_time,
    load,
    make_inputs,
    patch_bins,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROWS = 4100
VOCAB = 4095
TOPK = 512
STRIDE0 = 4352
LENGTHS = (513, 576, 640, 704, 768, 832, 896, 960, 1024, 1025, 1152, 1280, 1536, 1792, 2048)


def main():
    import flaggems_vllm
    import vllm._custom_ops  # noqa: F401 - register the compiled baseline

    ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
    if flaggems_vllm.top_k_per_row_prefill is not ov.top_k_per_row_prefill:
        raise RuntimeError("public prefill operator is not the Hygon override")
    if not ov._ONESCAN_PATH or not ov._ENABLED or not ov._GEOMETRY:
        raise RuntimeError("expected shipped Hygon one-scan, slot-scan and geometry")
    source_path = ov._VEC2_PATH or ov._CARRY_PATH
    if not source_path:
        raise RuntimeError("current dense source is unavailable")

    emit(
        "probe",
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        host=platform.node(),
        env={k: os.environ.get(k) for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "FLAGGEMS_FORCE_TLE")},
        rows=ROWS,
        vocab=VOCAB,
        top_k=TOPK,
        lengths=LENGTHS,
        bins=BIN_CANDIDATES,
        source=source_path,
    )
    occupancy("before")

    base = pathlib.Path(source_path).read_text()
    geo = ov._geometry(ROWS, VOCAB)
    with tempfile.TemporaryDirectory(prefix="hygon_short_bins_crossover_") as folder:
        folder = pathlib.Path(folder)
        modules = {}
        for bins in BIN_CANDIDATES:
            path = folder / f"bins_{bins}.py"
            path.write_text(patch_bins(base, bins))
            modules[bins] = load(
                f"flaggems_vllm.ops._hygon_short_bins_crossover_{bins}", path
            )
        emit("setup", geometry=geo, source=source_path)

        for length in LENGTHS:
            tensors = make_inputs(ROWS, VOCAB, TOPK, STRIDE0, length, 42)
            tied = make_inputs(ROWS, VOCAB, TOPK, STRIDE0, length, 43, tied=True)
            output = torch.empty((ROWS, TOPK), device="cuda", dtype=torch.int32)

            for bins, mod in modules.items():
                configure(mod, ov, ROWS, VOCAB)
                for label, data in (("normal", tensors), ("tied", tied)):
                    mod.top_k_per_row_prefill(
                        data[0], data[1], data[2], output, ROWS, STRIDE0, 1, TOPK
                    )
                    torch.cuda.synchronize()
                    check_values(output, data, TOPK)
                    emit(
                        "validation",
                        length=length,
                        bins=bins,
                        case=label,
                        ok=True,
                    )

            def control(data=tensors):
                flaggems_vllm.top_k_per_row_prefill(
                    data[0], data[1], data[2], output, ROWS, STRIDE0, 1, TOPK
                )

            summaries = {}
            for bins, mod in modules.items():
                def candidate(mod=mod, data=tensors):
                    mod.top_k_per_row_prefill(
                        data[0], data[1], data[2], output, ROWS, STRIDE0, 1, TOPK
                    )

                pairs = []
                for round_id in range(3):
                    order = ("control", "candidate", "candidate", "control")
                    fns = {"control": control, "candidate": candidate}
                    readings = [
                        (arm, device_time(fns[arm], "non_tle_top_k_per_row_prefill"))
                        for arm in order
                    ]
                    pair = dict(readings)
                    ratio = pair["control"] / pair["candidate"]
                    pairs.append((pair["control"], pair["candidate"], ratio))
                    emit(
                        "round",
                        length=length,
                        bins=bins,
                        round=round_id,
                        readings=readings,
                        ratio=ratio,
                    )
                summaries[str(bins)] = {
                    "control_us": statistics.median(x[0] for x in pairs),
                    "candidate_us": statistics.median(x[1] for x in pairs),
                    "ratio": statistics.median(x[2] for x in pairs),
                    "min_ratio": min(x[2] for x in pairs),
                    "max_ratio": max(x[2] for x in pairs),
                }
                emit("summary", length=length, bins=bins, **summaries[str(bins)])

            best = max(summaries, key=lambda bins: summaries[bins]["ratio"])
            emit(
                "length_summary",
                length=length,
                adaptive_hint="row_len/8 heuristic is intentionally not applied",
                best_bins=int(best),
                summaries=summaries,
            )
            del tensors, tied, output

    occupancy("after")
    emit("probe_complete", ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise