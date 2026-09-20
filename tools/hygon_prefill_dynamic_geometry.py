"""Probe total-work-driven Hygon prefill geometry.

The shipped route uses rows/SM plus row length. This isolated candidate computes
a work estimate from rows * ceil(row_len / 256) and selects a wider or narrower
launch from that estimate. It changes no kernel source and is measured against
the current dispatch on benchmark and crossover shapes.
"""

from __future__ import annotations

import gc
import pathlib
import statistics
import subprocess
from importlib import import_module

import torch

from hygon_prefill_audit import emit, occupancy

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = (
    ("long_64", 64, 129280, 1024, 129280),
    ("long_16", 16, 129280, 1024, 129280),
    ("long_32_16385", 32, 16385, 512, 16648),
    ("sparse_4_8193", 4, 8193, 512, 8456),
    ("sparse_4_16385", 4, 16385, 512, 16648),
    ("dense_many", 16383, 4095, 512, 4352),
    ("dense_4100", 12961, 4100, 512, 4352),
    ("dense_short", 4100, 1025, 512, 1288),
)


def make_inputs(rows, vocab, stride0, seed):
    torch.manual_seed(seed)
    buf = torch.randn(
        (rows - 1) * stride0 + vocab,
        device="cuda",
        dtype=torch.float32,
    )
    logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), vocab, dtype=torch.int32, device="cuda")
    return logits, starts, ends


def check_values(out, data, top_k):
    logits, starts, ends = data
    rows, vocab = logits.shape
    cols = torch.arange(vocab, device=logits.device)[None, :]
    live = (cols >= starts[:, None]) & (cols < ends[:, None])
    want = torch.topk(torch.where(live, logits, float("-inf")), top_k, dim=1).values
    want = want.sort(dim=1).values
    valid = torch.arange(top_k, device=logits.device)[None, :] < (ends - starts)[:, None]
    absolute = torch.where(valid, out + starts[:, None], 0).long()
    got = torch.where(valid, logits.gather(1, absolute), float("-inf")).sort(dim=1).values
    if not torch.equal(got, want):
        raise AssertionError("dynamic geometry output differs from torch.topk")


def candidate_geometry(ov, rows, row_len):
    sms = ov._sm_count()
    total_tiles = rows * ((row_len + 255) // 256)
    # Use enough programs for the estimated work, but preserve the proven
    # high-row geometry once the grid itself fills the card.
    if rows >= 32 * sms:
        return (256, 2) if row_len <= ov.SHORT_ROW_MAX else (256, 4)
    if total_tiles >= 8 * sms:
        return 256, 4
    if total_tiles >= 2 * sms:
        return 512, 4
    return 512, 8


def event_us(fn, iters=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return end.elapsed_time(begin) * 1000.0 / iters


def main():
    import flaggems_vllm

    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    original_geometry = ov._geometry
    emit(
        "probe",
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        sm_count=ov._sm_count(),
    )
    occupancy("before")
    for name, rows, vocab, top_k, stride0 in CASES:
        data = make_inputs(rows, vocab, stride0, 81)
        control_out = torch.empty((rows, top_k), device="cuda", dtype=torch.int32)
        candidate_out = torch.empty_like(control_out)
        control_geo = original_geometry(rows, vocab)
        candidate_geo = candidate_geometry(ov, rows, vocab)

        def control():
            ov._geometry = original_geometry
            flaggems_vllm.top_k_per_row_prefill(
                data[0], data[1], data[2], control_out, rows, stride0, 1, top_k
            )

        def candidate():
            ov._geometry = lambda r, n: candidate_geometry(ov, r, n)
            flaggems_vllm.top_k_per_row_prefill(
                data[0], data[1], data[2], candidate_out, rows, stride0, 1, top_k
            )

        control()
        torch.cuda.synchronize()
        check_values(control_out, data, top_k)
        candidate()
        torch.cuda.synchronize()
        check_values(candidate_out, data, top_k)
        emit(
            "validation",
            shape=name,
            control_geometry=control_geo,
            candidate_geometry=candidate_geo,
            ok=True,
        )

        ratios = []
        control_readings = []
        candidate_readings = []
        for round_id in range(3):
            if round_id % 2 == 0:
                control_us = event_us(control)
                candidate_us = event_us(candidate)
            else:
                candidate_us = event_us(candidate)
                control_us = event_us(control)
            control_readings.append(control_us)
            candidate_readings.append(candidate_us)
            ratios.append(control_us / candidate_us if candidate_us else None)
            emit(
                "paired_round",
                shape=name,
                round=round_id,
                control_us=control_us,
                candidate_us=candidate_us,
                ratio=ratios[-1],
            )
        emit(
            "benchmark",
            shape=name,
            rows=rows,
            vocab=vocab,
            top_k=top_k,
            control_geometry=control_geo,
            candidate_geometry=candidate_geo,
            control_us=statistics.median(control_readings),
            candidate_us=statistics.median(candidate_readings),
            ratio=statistics.median(ratios),
        )
        ov._geometry = original_geometry
        del data, control_out, candidate_out
        gc.collect()
        torch.cuda.empty_cache()
    ov._geometry = original_geometry
    occupancy("after")
    emit("probe_complete", ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise