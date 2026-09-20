"""Probe the complete decode-style sampled single-tail pipeline on prefill rows.

This reuses the Hygon decode implementation as an isolated candidate:
prepare/sample, one candidate select, then one device-side fallback/exact
tail. It is intentionally measured only on long-row prefill shapes and does
not alter the prefill dispatch.
"""

from __future__ import annotations

import gc
import pathlib
import subprocess
import sys
import time
from importlib import import_module

import torch

from hygon_prefill_audit import emit, occupancy

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = (
    ("long_64", 64, 129280, 1024, 129280),
    ("long_16", 16, 129280, 1024, 129280),
    ("long_4_8193", 4, 8193, 512, 8456),
    ("long_4_16385", 4, 16385, 512, 16648),
)


def make_inputs(rows, vocab, stride0, seed, tied=False):
    torch.manual_seed(seed)
    buf = torch.randn(
        (rows - 1) * stride0 + vocab,
        device="cuda",
        dtype=torch.float32,
    )
    if tied:
        buf = (buf * 4).round() / 4
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
    if not bool(torch.all((out >= 0) & (out < (ends - starts)[:, None]))):
        raise AssertionError("decode-tail candidate produced invalid relative indices")
    absolute = torch.where(valid, out + starts[:, None], 0).long()
    got = torch.where(valid, logits.gather(1, absolute), float("-inf")).sort(dim=1).values
    if not torch.equal(got, want):
        raise AssertionError("decode-tail candidate values differ from torch.topk")


def event_us(fn, iters=6):
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

    decode = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    prefill = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    emit(
        "probe",
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        decode_module=decode.__file__,
        prefill_module=prefill.__file__,
    )
    occupancy("before")
    for name, rows, vocab, top_k, stride0 in CASES:
        normal = make_inputs(rows, vocab, stride0, 61)
        tied = make_inputs(rows, vocab, stride0, 62, tied=True)
        prefill_out = torch.empty((rows, top_k), device="cuda", dtype=torch.int32)
        decode_out = torch.empty_like(prefill_out)

        def control(data=normal):
            flaggems_vllm.top_k_per_row_prefill(
                data[0], data[1], data[2], prefill_out, rows, stride0, 1, top_k
            )

        def candidate(data=normal):
            decode.top_k_per_row_decode(
                data[0], 1, data[2], decode_out, rows, stride0, 1, top_k
            )

        for label, data in (("normal", normal), ("tied", tied)):
            control(data)
            torch.cuda.synchronize()
            check_values(prefill_out, data, top_k)
            candidate(data)
            torch.cuda.synchronize()
            check_values(decode_out, data, top_k)
            emit("validation", shape=name, case=label, ok=True)

        control_us = event_us(control)
        candidate_us = event_us(candidate)
        emit(
            "benchmark",
            shape=name,
            rows=rows,
            vocab=vocab,
            top_k=top_k,
            prefill_us=control_us,
            decode_tail_us=candidate_us,
            ratio=control_us / candidate_us if candidate_us else None,
        )
        del normal, tied, prefill_out, decode_out
        gc.collect()
        torch.cuda.empty_cache()
    occupancy("after")
    emit("probe_complete", ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise