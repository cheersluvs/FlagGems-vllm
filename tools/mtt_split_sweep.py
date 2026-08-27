#!/usr/bin/env python3
"""Sweep the row-split factor on MTT's three sub-1.0 prefill shapes.

The earlier multi-block attempt used SPLIT=10 and lost 34%. That test answered
"does splitting ten ways help", not "does splitting help" -- it multiplied the
per-program fixed cost (a 2048-bin histogram clear, a threshold scan, a final
select) tenfold to buy parallelism the shape did not need that much of.

Re-reading the occupancy for (64, 129280) says how much it actually needs.
BLOCK=512 is 16 warps, the SM tops out at 32, so an SM holds two programs and the
device holds 120. At 64 rows, four SMs run two programs (32 warps) and fifty-six
run one (16 warps): **half the machine's warp capacity is idle**, and no BLOCK
change fixes it -- widening to 1024 drops capacity to 60 and makes 64 rows two
waves, which is exactly why it measured 0.82x.

Reaching 32 warps/SM needs ~120 programs. From 64 rows that is SPLIT=2, not 10.
SPLIT 2/3/4 have never been measured.

The kernel here is the one from a77d417, which passed the full suite on MTT
(20/20, including the non-zero row_start cases that would catch the index-basis
error this decomposition invites). Only the split factor is new.

    VLLM_PLUGINS=musa PYTHONPATH=src python tools/mtt_split_sweep.py

Measurement only. Registers nothing.
"""

import sys

import torch
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.ops.top_k_per_row_prefill import (
    NUM_THREADS_PER_BLOCK,
    NUM_THREADS_PER_BLOCK_MERGE,
    _num_warps,
    _top_k_per_row_job,
    _use_radix_final_for_prefill,
)
from flaggems_vllm.ops.top_k_per_row_prefill import (
    top_k_per_row_prefill as _generic_prefill,
)

DEV = flaggems_vllm.device

HAS_VLLM = False
try:
    import vllm._custom_ops  # noqa: F401

    if hasattr(torch.ops._C, "top_k_per_row_prefill"):
        HAS_VLLM = True
except (ImportError, AttributeError, RuntimeError):
    pass

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = None

# The three MTT shapes below 1.0, plus one that already wins as a control.
SHAPES = [
    (64, 129280, 1024, 129280, "worst; 64 rows on 60 SMs = half the warps idle"),
    (4, 8193, 512, 8456, "launch-bound; splitting should NOT help"),
    (4, 16385, 512, 16648, "launch-bound; splitting should NOT help"),
    (4100, 1025, 512, 1288, "control: already 1.46, must not regress"),
]

@triton.jit
def _multi_block_prefill(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    stride0,
    stride1,
    vocab_size,
    out_logits_ptr,
    indices_ptr,
    TOPK: tl.constexpr,
    TOPKP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
    MULTIPLE_BLOCKS_NUM: tl.constexpr,
    MERGE_BLOCKS: tl.constexpr,
):
    """Outer kernel for the two-pass split. Handles ONLY split and merge.

    The single-block case is never routed here -- the dispatcher returns the
    generic function for it -- which keeps this kernel to two modes.
    """
    NUM_FILNAL_ITEMS: tl.constexpr = 2048
    NUM_BINS: tl.constexpr = 2048
    VEC: tl.constexpr = 4

    row_id = tl.program_id(0)

    if MERGE_BLOCKS:
        # Pass 2. `logits_ptr` is the aux logits [num_rows, SPLIT*TOPK] and
        # `indices_ptr` the aux indices, which already carry indices relative to
        # row_starts[row_id] (see the pointer shift below). So the merge emits
        # the caller's convention directly, with no offset to undo.
        row_start = 0
        row_end = MULTIPLE_BLOCKS_NUM * TOPK
        indices_ptr += row_id * MULTIPLE_BLOCKS_NUM * TOPK
        out_indices_ptr += row_id * TOPK
        logits_ptr += row_id * stride0
        skip_elems = 0
    else:
        # Pass 1. In multi-block mode `_top_k_per_row_job` emits `pos +
        # row_start`, i.e. indices relative to whatever origin the caller uses.
        # Decode can pass an absolute offset because its rows start at 0; prefill
        # rows start at row_starts[i], so we shift the base pointer to that start
        # and give the job *span-relative* bounds. Passing the absolute start
        # instead would skew every index by row_starts[i].
        rs = tl.load(row_starts + row_id)
        re = tl.load(row_ends + row_id)
        span = re - rs
        logits_ptr += row_id * stride0 + rs * stride1

        blk_id = tl.program_id(1)
        blk = span // MULTIPLE_BLOCKS_NUM
        row_start = blk * blk_id
        # Last block absorbs the remainder. When span < MULTIPLE_BLOCKS_NUM this
        # degenerates to empty ranges for blocks 0..N-2 and the whole span for the
        # last -- wasteful but correct, and the gate keeps us out of that regime.
        row_end = span if blk_id == MULTIPLE_BLOCKS_NUM - 1 else row_start + blk

        out_indices_ptr += row_id * MULTIPLE_BLOCKS_NUM * TOPK + blk_id * TOPK
        out_logits_ptr += row_id * MULTIPLE_BLOCKS_NUM * TOPK + blk_id * TOPK

        # float4 alignment is a property of the true memory offset, so it must be
        # computed from the absolute position, not the span-relative one.
        x_off_mod = (row_id * stride0 + rs + row_start) % VEC
        skip_elems = 0 if x_off_mod == 0 else VEC - x_off_mod

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
    s_threshold_bin_idx = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_bin_size = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_found_topk_values = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_histogram_ptr = tle.gpu.local_ptr(s_histogram, (0,))
    s_final_logits_ptr = tle.gpu.local_ptr(s_final_logits, (0,))
    s_out_indices_ptr = tle.gpu.local_ptr(s_out_indices, (0,))
    s_final_cnt_ptr = tle.gpu.local_ptr(s_final_cnt, (0,))
    s_threshold_bin_idx_ptr = tle.gpu.local_ptr(s_threshold_bin_idx, (0,))
    s_final_bin_size_ptr = tle.gpu.local_ptr(s_final_bin_size, (0,))
    s_found_topk_values_ptr = tle.gpu.local_ptr(s_found_topk_values, (0,))

    if MERGE_BLOCKS:
        s_out_logits_ptr = None
    else:
        s_out_logits = tle.gpu.alloc(
            [TOPKP], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        s_out_logits_ptr = tle.gpu.local_ptr(s_out_logits, (0,))

    _top_k_per_row_job(
        logits_ptr,
        out_indices_ptr,
        row_start,
        row_end,
        stride1,
        vocab_size,
        skip_elems,
        out_logits_ptr,
        indices_ptr,
        s_histogram_ptr,
        s_final_logits_ptr,
        s_final_cnt_ptr,
        s_threshold_bin_idx_ptr,
        s_final_bin_size_ptr,
        s_found_topk_values_ptr,
        s_out_indices_ptr,
        s_out_logits_ptr,
        TOPK=TOPK,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_RADIX_FINAL=USE_RADIX_FINAL,
        HAS_TLE=True,
        MULTIPLE_BLOCKS_PER_ROW=not MERGE_BLOCKS,
        MERGE_BLOCKS=MERGE_BLOCKS,
    )


def _run_split(logits, starts, ends, idx, num_rows, s0, s1, top_k, split):
    """One split pass plus the merge. split=1 means call the generic op."""
    if split == 1:
        return _generic_prefill(logits, starts, ends, idx, num_rows, s0, s1, top_k)

    vocab = logits.shape[1]
    topkp = triton.next_power_of_2(top_k)
    urf = _use_radix_final_for_prefill(vocab)
    dev = logits.device
    ia = torch.empty((num_rows, split, top_k), device=dev, dtype=torch.int32)
    la = torch.empty((num_rows, split, top_k), device=dev, dtype=torch.float32)

    _multi_block_prefill[(num_rows, split)](
        logits, ia, starts, ends, s0, s1, vocab, la, None,
        TOPK=top_k, TOPKP=topkp, BLOCK_SIZE=NUM_THREADS_PER_BLOCK,
        USE_RADIX_FINAL=urf, MULTIPLE_BLOCKS_NUM=split, MERGE_BLOCKS=False,
        num_warps=_num_warps(NUM_THREADS_PER_BLOCK),
    )
    _multi_block_prefill[(num_rows,)](
        la, idx, starts, ends, split * top_k, 1, split * top_k, None, ia,
        TOPK=top_k, TOPKP=topkp, BLOCK_SIZE=NUM_THREADS_PER_BLOCK_MERGE,
        USE_RADIX_FINAL=urf, MULTIPLE_BLOCKS_NUM=split, MERGE_BLOCKS=True,
        num_warps=_num_warps(NUM_THREADS_PER_BLOCK_MERGE),
    )


def _correct(logits, idx, top_k):
    ref = torch.topk(logits[0], top_k, largest=True, sorted=False).indices
    got = idx[0].to(torch.int64)
    got = got[got >= 0]
    if got.numel() != min(top_k, logits.shape[1]):
        return False
    return torch.allclose(
        torch.sort(logits[0][got]).values,
        torch.sort(logits[0][ref]).values,
        atol=1e-6, rtol=1e-6,
    )


def main():
    print("=" * 78)
    print("  行内切分 SPLIT 扫描 -- MTT 三个低于 1.0 的形状")
    print("=" * 78)
    if tle is None:
        print("  !! 无 TLE, 多块路径依赖 tle smem, 退出")
        return 1
    if not HAS_VLLM:
        print("  !! 无 vLLM 基线, 用 VLLM_PLUGINS=musa 跑")

    for num_rows, vocab, top_k, s0, note in SHAPES:
        print(f"\n  ({num_rows}, {vocab})  top_k={top_k}   {note}")
        torch.manual_seed(42)
        buf = torch.randn((num_rows - 1) * s0 + vocab, device=DEV)
        logits = torch.as_strided(buf, (num_rows, vocab), (s0, 1))
        starts = torch.zeros((num_rows,), dtype=torch.int32, device=DEV)
        ends = torch.full((num_rows,), vocab, dtype=torch.int32, device=DEV)
        idx = torch.empty((num_rows, top_k), dtype=torch.int32, device=DEV)

        v = None
        if HAS_VLLM:
            v = triton.testing.do_bench(
                lambda: torch.ops._C.top_k_per_row_prefill(
                    logits, starts, ends, idx, num_rows, s0, 1, top_k
                ), warmup=25, rep=100, return_mode="median")

        print(f"    {'split':>6}{'programs':>10}{'prog/SM':>9}{'us':>9}"
              f"{'speedup':>9}{'vs split=1':>12}  正确")
        base = None
        for split in (1, 2, 3, 4):
            try:
                idx.fill_(-1)
                _run_split(logits, starts, ends, idx, num_rows, s0, 1, top_k, split)
                flaggems_vllm.runtime.torch_device_fn.synchronize()
                ok = _correct(logits, idx, top_k)
                t = triton.testing.do_bench(
                    lambda k=split: _run_split(
                        logits, starts, ends, idx, num_rows, s0, 1, top_k, k
                    ), warmup=25, rep=100, return_mode="median")
            except Exception as e:  # noqa: BLE001
                print(f"    {split:>6}   失败 {type(e).__name__}: {str(e)[:44]}")
                continue
            if base is None:
                base = t
            progs = num_rows * split
            sp = v / t if v else float("nan")
            mark = "  <-- 更好" if base / t > 1.02 else ""
            print(f"    {split:>6}{progs:>10}{progs/60:>9.2f}{t*1000:>9.1f}"
                  f"{sp:>9.3f}{base/t:>11.2f}x  {'OK' if ok else '!! 错误'}{mark}",
                  flush=True)
    print("\n  目标: (64,129280) 在 split=2 附近应接近满载 (128 program, 2.13/SM)")
    print("  控制组 (4100,1025) 必须不退化, 否则任何门控都要把它排除")
    return 0


if __name__ == "__main__":
    sys.exit(main())
