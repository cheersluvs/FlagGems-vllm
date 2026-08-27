#!/usr/bin/env python3
"""Count how many refinement passes prefill actually makes, per row.

This exists because an earlier measurement of mine was wrong in a way that
inverted a conclusion. I timed the inner loop with SCALAR loads and got 252 GB/s,
concluded the load dominated a single pass, and therefore that the operator was
essentially one pass and refinement depth was not a lever.

The kernel does not load like that. Its scan builds `base = t*BLOCK*VEC +
lane*VEC` with VEC=4 and loads a [BLOCK, 4] tile. Measured with that pattern the
load runs at 645-661 GB/s at 60-64 rows -- 2.5x what I reported -- so one full
histogram pass costs about 69 us at (60, 131072), not 177. Against an operator
time of 209 us that is roughly THREE passes, and vLLM's 129 us is roughly two.

If that holds, pass count is where the 1.63x lives, and the rewrite worth trying
is making step 0 selective enough to finish sooner -- not touching the loop body,
which is already vectorised, with near-free bit extraction and an atomic worth
16%.

But "roughly three" is inferred from ratios. This counts it directly: the same
four-step loop, with a counter incremented on each step that actually executes.

    VLLM_PLUGINS=musa PYTHONPATH=src python tools/mtt_pass_count.py

Measurement only.
"""

import sys

import torch
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.ops.top_k_per_row_prefill import (
    NUM_THREADS_PER_BLOCK,
    _num_warps,
    _process_histogram_step,
)

DEV = flaggems_vllm.device

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = None


@triton.jit
def _count_steps(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    STEPS,
    stride0,
    stride1,
    vocab_size,
    TOPK: tl.constexpr,
    TOPKP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """The real four-step loop, instrumented. Mirrors tle_top_k_per_row_prefill's
    prologue exactly so the step decisions are the ones the operator makes."""
    NUM_FILNAL_ITEMS: tl.constexpr = 2048
    NUM_BINS: tl.constexpr = 2048
    VEC: tl.constexpr = 4

    row_id = tl.program_id(0)
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    logits_ptr += row_id * stride0
    x_off_mod = (row_id * stride0 + row_start) % VEC
    skip_elems = 0 if x_off_mod == 0 else VEC - x_off_mod
    out_indices_ptr += row_id * TOPK

    s_histogram = tle.gpu.alloc(
        [NUM_BINS], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_logits = tle.gpu.alloc(
        [NUM_FILNAL_ITEMS], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_out_indices = tle.gpu.alloc(
        [TOPKP], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_cnt = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_thr = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_bin = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_found = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    hp = tle.gpu.local_ptr(s_histogram, (0,))
    flp = tle.gpu.local_ptr(s_final_logits, (0,))
    oip = tle.gpu.local_ptr(s_out_indices, (0,))
    fcp = tle.gpu.local_ptr(s_final_cnt, (0,))
    tbp = tle.gpu.local_ptr(s_thr, (0,))
    fbp = tle.gpu.local_ptr(s_bin, (0,))
    fvp = tle.gpu.local_ptr(s_found, (0,))

    tl.store(fcp, 0)
    tl.store(fvp, 0)
    tl.debug_barrier()

    assume_aligned = (
        (row_start == 0) & (row_end == vocab_size) & (stride1 == 1)
        & ((vocab_size % BLOCK_SIZE) == 0)
    )
    pattern = tl.zeros((), dtype=tl.uint32)
    go = tl.full((), True, dtype=tl.int1)
    thr_bin = tl.full((), -1, dtype=tl.int32)
    n = tl.zeros((), dtype=tl.int32)
    for step_idx in tl.static_range(0, 4):
        if go:
            n += 1
            (go, pattern, thr_bin) = _process_histogram_step(
                logits_ptr, row_start, row_end, stride1, vocab_size, skip_elems,
                None, pattern, thr_bin, assume_aligned,
                hp, flp, fcp, tbp, fbp, fvp, oip, None,
                STEP=step_idx, TOPK=TOPK, BLOCK_SIZE=BLOCK_SIZE, HAS_TLE=True,
                MULTIPLE_BLOCKS_PER_ROW=False, MERGE_BLOCKS=False,
            )
    tl.store(STEPS + row_id, n)


def main():
    print("=" * 78)
    print("  每行实际执行的精化遍数")
    print("=" * 78)
    if tle is None:
        print("  !! 无 TLE, 退出")
        return 1

    # (num_rows, vocab, top_k, one-pass cost in us, measured op cost in us)
    CASES = [
        (60, 131072, 512, 69.2, 209.4, "sweep 1 的形状"),
        (64, 129280, 1024, None, 217.8, "最差形状"),
        (4100, 1025, 512, None, 387.5, "已经赢的形状"),
        (16383, 4095, 512, None, 2333.2, "赢得最多的形状"),
    ]
    print(f"  {'shape':>16}{'平均遍数':>10}{'最少':>7}{'最多':>7}   备注")
    for rows, vocab, top_k, one_pass, op_us, note in CASES:
        torch.manual_seed(42)
        logits = torch.randn((rows, vocab), dtype=torch.float32, device=DEV)
        starts = torch.zeros((rows,), dtype=torch.int32, device=DEV)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=DEV)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=DEV)
        steps = torch.zeros((rows,), dtype=torch.int32, device=DEV)
        try:
            _count_steps[(rows,)](
                logits, idx, starts, ends, steps,
                logits.stride(0), logits.stride(1), vocab,
                TOPK=top_k, TOPKP=triton.next_power_of_2(top_k),
                BLOCK_SIZE=NUM_THREADS_PER_BLOCK,
                num_warps=_num_warps(NUM_THREADS_PER_BLOCK),
            )
            flaggems_vllm.runtime.torch_device_fn.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  {f'({rows},{vocab})':>16}   失败 {type(e).__name__}: {str(e)[:36]}")
            continue
        f = steps.float()
        extra = ""
        if one_pass:
            extra = f"   -> {op_us/one_pass:.1f} 遍的等效耗时"
        print(f"  {f'({rows},{vocab})':>16}{f.mean().item():>10.2f}"
              f"{int(steps.min()):>7}{int(steps.max()):>7}   {note}{extra}", flush=True)

    print("\n  平均遍数接近 3 => 与按耗时推算的一致, 减少遍数是真杠杆")
    print("  平均遍数接近 1 => 时间花在别处(final select), 遍数不是杠杆")
    return 0


if __name__ == "__main__":
    sys.exit(main())
