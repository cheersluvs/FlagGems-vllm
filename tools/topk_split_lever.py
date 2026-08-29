#!/usr/bin/env python3
"""Does splitting a prefill row across programs beat the 60-row wide-block cliff?

MTT prefill is stuck at 0.902 on (64, 129280). 64 rows is four past the wide
block's 60-row capacity cliff, so widening can never help it, and grid=64 on a
card that holds ~120 concurrent programs leaves half the machine idle.

Row splitting is the obvious fix and it is ALREADY IMPLEMENTED. Every shared
kernel honours MULTIPLE_BLOCKS_PER_ROW -- _process_bins, _process_histogram_step,
_top_k_per_row_job and _final_select_radix all take it, and those five functions
are byte-identical between the prefill and decode files. Only
tle_top_k_per_row_prefill hardcodes the flag to False and its host never reads
SPLIT_WORK_THRESHOLD, while decode wires all of it.

So this probe needs NO new kernel. tle_top_k_per_row_decode is a drop-in for
prefill's semantics whenever row_start is 0:

    decode:   row_start = 0
              row_end   = seq_len - next_n + batch_offset + 1
    with next_n=1 -> batch_offset=0, batch_id=row_id -> row_end = seq_lens[row_id]

    prefill:  row_start = row_starts[row_id]   (all 0 in the benchmark)
              row_end   = row_ends[row_id]     (all vocab)

Past that both kernels are the same lines and call the same job. So the probe
drives the decode entry over PREFILL shapes, once unsplit and once per split
factor, and the difference is the split and nothing else.

    VLLM_PLUGINS=musa PYTHONPATH=src python tools/topk_split_lever.py

Correctness is checked on every configuration, not just timed: the split path has
never run on these shapes, and a faster wrong answer is the risk.

Measurement only. Registers nothing, changes no shipped file.
"""

import sys

import torch
import triton

import flaggems_vllm
from flaggems_vllm.ops.top_k_per_row_decode import (
    NUM_THREADS_PER_BLOCK,
    NUM_THREADS_PER_BLOCK_MERGE,
    SORTING_ALGORITHM_THRESHOLD,
    _num_warps,
    tle_top_k_per_row_decode,
)

DEV = flaggems_vllm.device

try:
    import vllm._custom_ops  # noqa: F401

    HAS_VLLM = hasattr(torch.ops._C, "top_k_per_row_prefill")
except (ImportError, AttributeError, RuntimeError):
    HAS_VLLM = False

# prefill's own benchmark shapes: num_rows, vocab, top_k, stride0, stride1
SHAPES = [
    (64, 129280, 1024, 129280, 1),   # the one stuck at 0.902
    (4, 16385, 512, 16648, 1),
    (16383, 4095, 512, 4352, 1),     # control: grid already huge, split must lose
    (12961, 4100, 512, 4360, 1),     # control
]
# grid becomes num_rows * SPLIT. MTT holds ~120 programs at BLOCK=512, so 64 rows
# wants 2-4; 10 (the shipped constant for decode) is likely already too many here.
SPLITS = (2, 3, 4, 6, 10)


def make_inputs(num_rows, vocab, top_k, stride0, stride1):
    """Exactly the prefill benchmark's construction: a strided view."""
    torch.manual_seed(42)
    buf = torch.randn(
        (num_rows - 1) * stride0 + (vocab - 1) * stride1 + 1,
        device=DEV, dtype=torch.float32,
    )
    logits = torch.as_strided(buf, (num_rows, vocab), (stride0, stride1))
    starts = torch.zeros(num_rows, dtype=torch.int32, device=DEV)
    ends = torch.full((num_rows,), vocab, dtype=torch.int32, device=DEV)
    # next_n MUST be a plain int: a tensor makes it pointer<int32> and the kernel
    # dies at compile time on `row_id // next_n`.
    seq_lens = torch.full((num_rows,), vocab, dtype=torch.int32, device=DEV)
    idx = torch.empty((num_rows, top_k), dtype=torch.int32, device=DEV)
    return logits, starts, ends, seq_lens, idx


def run_decode_entry(logits, seq_lens, idx, num_rows, vocab, top_k, stride0,
                     stride1, split, block=NUM_THREADS_PER_BLOCK):
    """decode's host dispatch, with the split factor as a parameter.

    split=1 is the single unsplit launch; anything larger is decode's two-launch
    path with that many blocks per row.
    """
    topkp = triton.next_power_of_2(top_k)
    use_radix_final = vocab >= SORTING_ALGORITHM_THRESHOLD
    if split == 1:
        tle_top_k_per_row_decode[(num_rows,)](
            logits, idx, seq_lens, 1, stride0, stride1, vocab, None, None,
            TOPK=top_k, TOPKP=topkp, BLOCK_SIZE=block,
            USE_RADIX_FINAL=use_radix_final, MULTIPLE_BLOCKS_PER_ROW=False,
            MULTIPLE_BLOCKS_NUM=1, MERGE_BLOCKS=False,
            num_warps=_num_warps(block),
        )
        return
    ai = torch.empty((num_rows, split, top_k), device=DEV, dtype=torch.int32)
    al = torch.empty((num_rows, split, top_k), device=DEV, dtype=torch.float32)
    tle_top_k_per_row_decode[(num_rows, split)](
        logits, ai, seq_lens, 1, stride0, stride1, vocab, al, None,
        TOPK=top_k, TOPKP=topkp, BLOCK_SIZE=block,
        USE_RADIX_FINAL=use_radix_final, MULTIPLE_BLOCKS_PER_ROW=True,
        MULTIPLE_BLOCKS_NUM=split, MERGE_BLOCKS=False,
        num_warps=_num_warps(block),
    )
    tle_top_k_per_row_decode[(num_rows,)](
        al, idx, seq_lens, 1, split * top_k, 1, split * top_k, None, ai,
        TOPK=top_k, TOPKP=topkp, BLOCK_SIZE=NUM_THREADS_PER_BLOCK_MERGE,
        USE_RADIX_FINAL=use_radix_final, MULTIPLE_BLOCKS_PER_ROW=False,
        MULTIPLE_BLOCKS_NUM=split, MERGE_BLOCKS=True,
        num_warps=_num_warps(NUM_THREADS_PER_BLOCK_MERGE),
    )


def correct(logits, idx, vocab, top_k):
    """Selected VALUES must match torch.topk's as a multiset (ties reorder)."""
    got = idx.to(torch.int64)
    if int(got.min()) < 0 or int(got.max()) >= vocab:
        return "索引越界"
    a = torch.sort(torch.gather(logits, 1, got), dim=1).values
    b = torch.sort(torch.topk(logits, top_k, dim=1, largest=True,
                              sorted=False).values, dim=1).values
    return "OK" if torch.equal(a, b) else "不符"


def bench(fn, reps=100):
    return triton.testing.do_bench(fn, warmup=25, rep=reps,
                                   return_mode="median") * 1000


# Splitting was refuted at 64 rows, where grid=64 was already half the card's
# ~120-program capacity so the extra parallelism had little left to buy. At 4
# rows grid is 4, and even SPLIT=15 lands at 60 -- still inside capacity, and
# inside the HALVED capacity a wide block leaves. That case was never measured,
# and vLLM cannot reach it at all: topKPerRowPrefill in sampler.cu takes no
# gridDim.y and no merge, so its prefill is structurally one block per row
# (4 blocks x 512 threads at 4 rows, ~16 GB/s on a 60-SM part).
SMALL_SHAPES = [
    (4, 129280, 1024, 129280, 1),
    (16, 129280, 1024, 129280, 1),
    (32, 129280, 1024, 129280, 1),
    (4, 16385, 512, 16648, 1),
]
SMALL_SPLITS = (1, 2, 4, 8, 15)
SMALL_BLOCKS = (512, 1024)


def small_sweep():
    """Does splitting pay where the grid is genuinely tiny, and does it stack
    with a wide block?

    Capacity is ~120 programs at BLOCK=512 and ~60 at 1024, so the two levers
    compete for the same budget: rows x split must stay under it. If splitting
    wins anywhere it is here, and if it does not, the 0.902 ceiling is not an
    occupancy problem at all.
    """
    print("  小 num_rows 上的拆分 x 块宽。grid = rows x split\n")
    for num_rows, vocab, top_k, s0, s1 in SMALL_SHAPES:
        logits, starts, ends, seq_lens, idx = make_inputs(
            num_rows, vocab, top_k, s0, s1)
        reps = 400 if num_rows <= 8 else 100
        v = None
        if HAS_VLLM:
            vidx = torch.empty_like(idx)
            v = bench(lambda: torch.ops._C.top_k_per_row_prefill(
                logits, starts, ends, vidx, num_rows, s0, s1, top_k), reps)

        idx.fill_(-1)
        flaggems_vllm.top_k_per_row_prefill(
            logits, starts, ends, idx, num_rows, s0, s1, top_k)
        flaggems_vllm.runtime.torch_device_fn.synchronize()
        ship_ok = correct(logits, idx, vocab, top_k)
        ship = bench(lambda: flaggems_vllm.top_k_per_row_prefill(
            logits, starts, ends, idx, num_rows, s0, s1, top_k), reps)

        print(f"  ({num_rows},{vocab},k={top_k})"
              + (f"   vLLM {v:.1f} µs" if v else ""))
        print(f"    {'配置':<26}{'grid':>7}{'µs':>9}{'vs 出货':>9}"
              f"{'speedup':>9}   正确性")
        print(f"    {'出货 prefill(采样+宽块)':<26}{num_rows:>7}{ship:>9.1f}"
              f"{1.0:>9.3f}{(v / ship if v else 0):>9.3f}   {ship_ok}")
        for block in SMALL_BLOCKS:
            for split in SMALL_SPLITS:
                name = f"decode BLOCK={block} SPLIT={split}"
                try:
                    idx.fill_(-1)
                    run_decode_entry(logits, seq_lens, idx, num_rows, vocab,
                                     top_k, s0, s1, split, block)
                    flaggems_vllm.runtime.torch_device_fn.synchronize()
                    ok = correct(logits, idx, vocab, top_k)
                    t = bench(lambda sp=split, b=block: run_decode_entry(
                        logits, seq_lens, idx, num_rows, vocab, top_k, s0, s1,
                        sp, b), reps)
                except Exception as e:  # noqa: BLE001
                    print(f"    {name:<26}   失败 {type(e).__name__}: "
                          f"{str(e).splitlines()[-1][:40]}")
                    continue
                print(f"    {name:<26}{num_rows * split:>7}{t:>9.1f}"
                      f"{t / ship:>9.3f}{(v / t if v else 0):>9.3f}   {ok}")
        print()
    print("  读法")
    print("    某个 (BLOCK,SPLIT) 的 speedup 高于出货 => 拆分在小 grid 上成立,")
    print("      而且能和采样叠加的话还会更高 -- 出货版是采样+宽块, 这里是通用+拆分")
    print("    全都不如出货 => 0.902 的天花板不是占用率问题, 拆分整条路关掉")
    print("    最优点的 grid 落在容量线附近(512 档约 120, 1024 档约 60) => 模型成立")
    return 0

def main():
    print("=" * 82)
    print("  prefill 行拆分：机制已存在，只是入口 kernel 写死关闭")
    print("=" * 82)
    print(f"  vLLM 基线: {'有' if HAS_VLLM else '无'}")
    bound = flaggems_vllm.top_k_per_row_prefill.__module__
    print(f"  出货算子绑定: {bound}\n")

    if "--small" in sys.argv:
        return small_sweep()

    for num_rows, vocab, top_k, s0, s1 in SHAPES:
        tag = f"({num_rows},{vocab},k={top_k})"
        logits, starts, ends, seq_lens, idx = make_inputs(
            num_rows, vocab, top_k, s0, s1)
        reps = 400 if num_rows <= 8 else 100

        v = None
        if HAS_VLLM:
            vidx = torch.empty_like(idx)
            try:
                v = bench(lambda: torch.ops._C.top_k_per_row_prefill(
                    logits, starts, ends, vidx, num_rows, s0, s1, top_k), reps)
            except Exception as e:  # noqa: BLE001
                print(f"  {tag} vLLM 失败: {type(e).__name__}")

        print(f"  {tag}   grid = num_rows x SPLIT"
              + (f"   vLLM {v:.1f} µs" if v else ""))
        print(f"    {'配置':<22}{'grid':>8}{'µs':>10}{'vs 出货':>9}"
              f"{'speedup':>9}   正确性")

        # 1. what ships today (MTT sampled override), for reference
        idx.fill_(-1)
        flaggems_vllm.top_k_per_row_prefill(
            logits, starts, ends, idx, num_rows, s0, s1, top_k)
        flaggems_vllm.runtime.torch_device_fn.synchronize()
        ship_ok = correct(logits, idx, vocab, top_k)
        ship = bench(lambda: flaggems_vllm.top_k_per_row_prefill(
            logits, starts, ends, idx, num_rows, s0, s1, top_k), reps)
        print(f"    {'出货 prefill':<22}{num_rows:>8}{ship:>10.1f}{1.0:>9.3f}"
              f"{(v / ship if v else 0):>9.3f}   {ship_ok}")

        # 2. the decode entry over the same shape, unsplit then split
        for split in (1,) + SPLITS:
            name = "decode 入口 不拆" if split == 1 else f"decode 入口 SPLIT={split}"
            try:
                idx.fill_(-1)
                run_decode_entry(logits, seq_lens, idx, num_rows, vocab, top_k,
                                 s0, s1, split)
                flaggems_vllm.runtime.torch_device_fn.synchronize()
                ok = correct(logits, idx, vocab, top_k)
                t = bench(lambda sp=split: run_decode_entry(
                    logits, seq_lens, idx, num_rows, vocab, top_k, s0, s1, sp),
                    reps)
            except Exception as e:  # noqa: BLE001
                print(f"    {name:<22}   失败 {type(e).__name__}: "
                      f"{str(e).splitlines()[-1][:44]}")
                continue
            print(f"    {name:<22}{num_rows * split:>8}{t:>10.1f}"
                  f"{t / ship:>9.3f}{(v / t if v else 0):>9.3f}   {ok}")
        print()

    print("  读法")
    print("    「decode 入口 不拆」正确性 OK  => 两个入口在 row_start=0 时确实等价")
    print("    某个 SPLIT 的 speedup > 出货    => 杠杆成立, 值得接进 prefill 入口")
    print("    (64,129280) 越过 0.9 且明显高于 0.902 => 那个悬崖是可以绕开的")
    print("    两个大行数形状上 SPLIT 必须变慢  => 门控要看 num_rows, 不能只看 vocab")
    print()
    print("  注意: 「出货 prefill」在 (64,129280)/(4,16385) 上是 MTT 采样 override,")
    print("  在两个大行数形状上是通用实现。拆分与采样是两条独立的路, 都赢则可叠加。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
