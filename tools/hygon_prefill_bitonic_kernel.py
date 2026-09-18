"""Isolated BW1000 feasibility kernel; never imported by production code."""

import triton
import triton.language as tl


@triton.jit
def bitonic_topk_indices(
    logits,
    starts,
    ends,
    output,
    stride0,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(starts + row)
    end = tl.load(ends + row)
    length = end - start
    out = output + row * TOPK

    if length <= TOPK:
        kpos = tl.arange(0, TOPK)
        tl.store(out + kpos, tl.where(kpos < length, kpos, -1).to(tl.int32))
        return

    pos = tl.arange(0, BLOCK)
    valid = pos < length
    values = tl.load(
        logits + row * stride0 + start + pos,
        mask=valid,
        other=float("-inf"),
    )
    # Bitonic top-k keeps the kth value in registers. The original row values
    # remain live, so an exact tie quota and one scan produce relative indices
    # without a global histogram or a second read of logits.
    kth = tl.min(tl.topk(values, TOPK))
    better = valid & (values > kth)
    equal = valid & (values == kth)
    quota = TOPK - tl.sum(better.to(tl.int32), axis=0)
    equal_rank = tl.cumsum(equal.to(tl.int32), axis=0) - 1
    selected = better | (equal & (equal_rank < quota))
    out_pos = tl.cumsum(selected.to(tl.int32), axis=0) - 1
    tl.store(out + out_pos, pos.to(tl.int32), mask=selected)
