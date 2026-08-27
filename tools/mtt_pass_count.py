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

    # (num_rows, vocab, top_k, note)
    CASES = [
        (60, 131072, 512, "sweep 1 的形状"),
        (64, 129280, 1024, "最差形状"),
        (4100, 1025, 512, "已经赢的形状"),
        (16383, 4095, 512, "赢得最多的形状"),
    ]
    print(f"  {'shape':>16}{'平均遍数':>10}{'最少':>7}{'最多':>7}   备注")
    for rows, vocab, top_k, note in CASES:
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
        print(f"  {f'({rows},{vocab})':>16}{f.mean().item():>10.2f}"
              f"{int(steps.min()):>7}{int(steps.max()):>7}   {note}", flush=True)

    # --- 直接测量：直方图阶段 vs 整算子 ---
    print("\n" + "=" * 78)
    print("  直方图阶段 vs 整算子 -- 两者都实测, final select 由差值得出")
    print("=" * 78)
    print(f"  {'shape':>16}{'直方图 us':>11}{'整算子 us':>11}{'之后 us':>10}"
          f"{'之后占比':>10}{'vLLM us':>10}")
    for rows, vocab, top_k, _ in CASES:
        torch.manual_seed(42)
        logits = torch.randn((rows, vocab), dtype=torch.float32, device=DEV)
        starts = torch.zeros((rows,), dtype=torch.int32, device=DEV)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=DEV)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=DEV)
        steps = torch.zeros((rows,), dtype=torch.int32, device=DEV)
        nw = _num_warps(NUM_THREADS_PER_BLOCK)
        try:
            hist = triton.testing.do_bench(
                lambda: _count_steps[(rows,)](
                    logits, idx, starts, ends, steps,
                    logits.stride(0), logits.stride(1), vocab,
                    TOPK=top_k, TOPKP=triton.next_power_of_2(top_k),
                    BLOCK_SIZE=NUM_THREADS_PER_BLOCK, num_warps=nw,
                ), warmup=25, rep=100, return_mode="median")
            full = triton.testing.do_bench(
                lambda: flaggems_vllm.top_k_per_row_prefill(
                    logits, starts, ends, idx, rows,
                    logits.stride(0), logits.stride(1), top_k,
                ), warmup=25, rep=100, return_mode="median")
        except Exception as e:  # noqa: BLE001
            print(f"  {f'({rows},{vocab})':>16}   失败 {type(e).__name__}")
            continue
        v = float("nan")
        try:
            import vllm._custom_ops  # noqa: F401

            if hasattr(torch.ops._C, "top_k_per_row_prefill"):
                v = triton.testing.do_bench(
                    lambda: torch.ops._C.top_k_per_row_prefill(
                        logits, starts, ends, idx, rows,
                        logits.stride(0), logits.stride(1), top_k,
                    ), warmup=25, rep=100, return_mode="median") * 1000
        except Exception:  # noqa: BLE001
            pass
        after = (full - hist) * 1000
        print(f"  {f'({rows},{vocab})':>16}{hist*1000:>11.1f}{full*1000:>11.1f}"
              f"{after:>10.1f}{after/(full*1000)*100:>9.0f}%{v:>10.1f}", flush=True)
    print("\n  '之后' 就是阈值扫描 + final select + 输出, 这次是实测差值不是推算")
    print("  若它超过 vLLM 整个算子的耗时, 那重写它是唯一有意义的方向")
    return 0


if __name__ == "__main__":
    sys.exit(main())
