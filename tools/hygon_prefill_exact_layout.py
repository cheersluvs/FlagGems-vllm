"""Probe a non-bitonic exact-layout specialization just above top_k.

For row_len = TOPK + DROP, DROP is small. One 1024-lane CTA loads the row
once, repeatedly removes the exact smallest value with an index tie-break,
then compacts the remaining TOPK relative indices. This is deliberately
limited to small DROP values and is not enabled in production.
"""

from __future__ import annotations

import gc
import pathlib
import statistics
import subprocess

import torch
import triton
import triton.language as tl

from hygon_prefill_audit import emit, occupancy

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOPK = 512
VOCAB = 1024
ROWS = 4100
STRIDE0 = 1024
DROPS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@triton.jit
def _ordered_key(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x80000000, tl.uint32)
    sign_set = (bits & sign_mask) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFFFFFF, tl.uint32)
    return tl.where(sign_set, bits, inv)


@triton.jit
def _drop_smallest(
    logits_ptr,
    row_starts,
    row_ends,
    out_ptr,
    stride0,
    stride1,
    TOPK: tl.constexpr,
    ROW_LEN: tl.constexpr,
    DROP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    row_start = tl.load(row_starts + row)
    row_end = tl.load(row_ends + row)
    pos = lane
    valid = (pos < ROW_LEN) & (pos < (row_end - row_start))
    x = tl.load(
        logits_ptr + row * stride0 + row_start + pos * stride1,
        mask=valid,
        other=float("-inf"),
    )
    key = _ordered_key(x)
    active = valid
    for _ in tl.static_range(0, DROP):
        worst = tl.max(
            tl.where(active, key, tl.full([BLOCK], 0, tl.uint32)),
            axis=0,
        )
        winner = tl.min(
            tl.where(
                active & (key == worst),
                pos,
                tl.full([BLOCK], BLOCK, tl.int32),
            ),
            axis=0,
        )
        active = active & (pos != winner)
    keep = active
    rank = tl.cumsum(keep.to(tl.int32), axis=0) - keep
    tl.store(out_ptr + row * TOPK + rank, pos.to(tl.int32), mask=keep)


def make_inputs(seed, tied=False):
    torch.manual_seed(seed)
    buf = torch.randn(
        (ROWS - 1) * STRIDE0 + VOCAB,
        device="cuda",
        dtype=torch.float32,
    )
    if tied:
        buf = (buf * 4).round() / 4
    logits = torch.as_strided(buf, (ROWS, VOCAB), (STRIDE0, 1))
    starts = torch.zeros(ROWS, dtype=torch.int32, device="cuda")
    return logits, starts


def check_values(out, data, row_len):
    logits, starts = data
    ends = starts + row_len
    want = torch.topk(logits[:, :row_len], TOPK, dim=1).values.sort(dim=1).values
    absolute = out.long()
    got = logits.gather(1, absolute).sort(dim=1).values
    if not bool(torch.all((out >= 0) & (out < row_len))):
        raise AssertionError("exact-layout output index out of bounds")
    if not torch.equal(got, want):
        raise AssertionError("exact-layout values differ from torch.topk")


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
    return begin.elapsed_time(end) * 1000.0 / iters


def main():
    import flaggems_vllm

    emit(
        "probe",
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        rows=ROWS,
        top_k=TOPK,
        block=1024,
        warps=16,
    )
    occupancy("before")
    for drop in DROPS:
        row_len = TOPK + drop
        normal = make_inputs(91)
        tied = make_inputs(92, tied=True)
        starts = normal[1]
        ends = starts + row_len
        candidate_out = torch.empty((ROWS, TOPK), device="cuda", dtype=torch.int32)
        control_out = torch.empty_like(candidate_out)

        def candidate(data=normal):
            _drop_smallest[(ROWS,)](
                data[0],
                data[1],
                ends,
                candidate_out,
                STRIDE0,
                1,
                TOPK=TOPK,
                ROW_LEN=row_len,
                DROP=drop,
                BLOCK=1024,
                num_warps=16,
            )

        def control(data=normal):
            flaggems_vllm.top_k_per_row_prefill(
                data[0],
                data[1],
                ends,
                control_out,
                ROWS,
                STRIDE0,
                1,
                TOPK,
            )

        for label, data in (("normal", normal), ("tied", tied)):
            candidate(data)
            torch.cuda.synchronize()
            check_values(candidate_out, data, row_len)
            control(data)
            torch.cuda.synchronize()
            check_values(control_out, data, row_len)
            emit("validation", drop=drop, row_len=row_len, case=label, ok=True)

        candidate_us = event_us(candidate)
        control_us = event_us(control)
        emit(
            "benchmark",
            drop=drop,
            row_len=row_len,
            control_us=control_us,
            candidate_us=candidate_us,
            ratio=control_us / candidate_us if candidate_us else None,
        )
        del normal, tied, candidate_out, control_out
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