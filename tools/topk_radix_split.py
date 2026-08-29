#!/usr/bin/env python3
"""Is prefill's row-count radix split worth its cost?

Found while running tools/topk_split_lever.py: on the two big shapes the DECODE
entry beat the shipped generic prefill, same shared kernels, same BLOCK=512.

    (16383, 4095)   prefill 2332.4 us / 1.590   decode entry 2153.0 / 1.723
    (12961, 4100)   prefill 1775.4   / 1.075    decode entry 1706.8 / 1.118

The difference is the host, and it looks like an oddity rather than a decision:

    decode    use_radix_final = vocab_size >= SORTING_ALGORITHM_THRESHOLD
              a VOCAB test. 4095 -> False -> ONE launch, insert sort everywhere.

    prefill   n_insert = 0 if use_radix_final
                         else min(num_rows, SORTING_ALGORITHM_THRESHOLD)
              the same constant 12288 used as a ROW COUNT. Rows [0, 12288) get
              insert sort and rows [12288, num_rows) get radix-final, in a
              SECOND launch.

So at vocab 4095 the rows past 12288 are switched to a path that appears to be
slower there, and pay an extra launch for it. This probe drives prefill's OWN
entry kernel three ways so the launch structure is the only variable:

    C  two launches, exactly as the generic host does it today
    A  one launch, USE_RADIX_FINAL=False for every row
    B  one launch, USE_RADIX_FINAL=True for every row

A vs C answers the question. B says whether radix-final is slower per se at this
vocabulary or only as a second launch.

Only shapes with num_rows > 12288 AND vocab < 65536 can differ; the others are
controls that must come out identical, and if they do not, the probe is wrong
rather than the operator.

    VLLM_PLUGINS=musa PYTHONPATH=src python tools/topk_radix_split.py

Measurement only. Registers nothing, changes no shipped file.
"""

import sys

import torch
import triton

import flaggems_vllm
from flaggems_vllm.ops.top_k_per_row_prefill import (
    SORTING_ALGORITHM_THRESHOLD,
    _num_warps,
    _prefill_block_size,
    _use_radix_final_for_prefill,
    tle_top_k_per_row_prefill,
)
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

try:
    import vllm._custom_ops  # noqa: F401

    HAS_VLLM = hasattr(torch.ops._C, "top_k_per_row_prefill")
except (ImportError, AttributeError, RuntimeError):
    HAS_VLLM = False

DEV = flaggems_vllm.device

# prefill's own benchmark shapes: num_rows, vocab, top_k, stride0, stride1
SHAPES = [
    (16383, 4095, 512, 4352, 1),      # 4095 rows switch to radix-final
    (16380, 5115, 512, 5376, 1),      # 4092 rows switch
    (12961, 4100, 512, 4360, 1),      # 673 rows switch
    (4100, 1025, 512, 1288, 1),       # control: num_rows < 12288, nothing switches
    (64, 129280, 1024, 129280, 1),    # control: vocab >= 65536, ALL rows radix
]


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
    idx = torch.empty((num_rows, top_k), dtype=torch.int32, device=DEV)
    return logits, starts, ends, idx


def launch(logits, starts, ends, idx, num_rows, vocab, top_k, s0, s1, mode):
    """mode: 'C' as the host does it, 'A' all insert sort, 'B' all radix-final."""
    block = _prefill_block_size(num_rows)
    nw = _num_warps(block)
    topkp = triton.next_power_of_2(top_k)
    common = dict(TOPK=top_k, TOPKP=topkp, BLOCK_SIZE=block, num_warps=nw)

    if mode == "A" or mode == "B":
        tle_top_k_per_row_prefill[(num_rows,)](
            logits, idx, starts, ends, s0, s1, vocab,
            USE_RADIX_FINAL=(mode == "B"), ROW_OFFSET=0, **common,
        )
        return

    # mode C: reproduce the generic host exactly, including its two launches
    use_radix_final = _use_radix_final_for_prefill(vocab)
    n_insert = (
        0 if use_radix_final else min(num_rows, SORTING_ALGORITHM_THRESHOLD)
    )
    if n_insert > 0:
        tle_top_k_per_row_prefill[(n_insert,)](
            logits, idx, starts, ends, s0, s1, vocab,
            USE_RADIX_FINAL=False, ROW_OFFSET=0, **common,
        )
    if num_rows > n_insert:
        tle_top_k_per_row_prefill[(num_rows - n_insert,)](
            logits, idx, starts, ends, s0, s1, vocab,
            USE_RADIX_FINAL=True, ROW_OFFSET=n_insert, **common,
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


NAMES = {
    "C": "C 现状 两次启动",
    "A": "A 单次 全插入排序",
    "B": "B 单次 全 radix",
}


def main():
    print("=" * 84)
    print("  prefill 的行数 radix 切分值不值：SORTING_ALGORITHM_THRESHOLD 当行数用")
    print("=" * 84)
    if not has_triton_tle(3, 6, 0):
        print("  !! 无 TLE, 该入口不可用, 退出")
        return 1
    print(f"  vLLM 基线: {'有' if HAS_VLLM else '无'}\n")

    for num_rows, vocab, top_k, s0, s1 in SHAPES:
        n_switch = 0
        if not _use_radix_final_for_prefill(vocab):
            n_switch = max(0, num_rows - min(num_rows, SORTING_ALGORITHM_THRESHOLD))
        tag = f"({num_rows},{vocab},k={top_k})"
        note = (f"切到 radix 的行数 {n_switch}" if n_switch
                else ("全部 radix (vocab>=65536)"
                      if _use_radix_final_for_prefill(vocab)
                      else "无行切换 (num_rows<12288)"))
        logits, starts, ends, idx = make_inputs(num_rows, vocab, top_k, s0, s1)

        v = None
        if HAS_VLLM:
            vidx = torch.empty_like(idx)
            try:
                v = bench(lambda: torch.ops._C.top_k_per_row_prefill(
                    logits, starts, ends, vidx, num_rows, s0, s1, top_k))
            except Exception as e:  # noqa: BLE001
                print(f"  {tag} vLLM 失败: {type(e).__name__}")

        print(f"  {tag}   {note}   BLOCK={_prefill_block_size(num_rows)}"
              + (f"   vLLM {v:.1f} µs" if v else ""))
        print(f"    {'配置':<22}{'µs':>11}{'vs C':>9}{'speedup':>9}   正确性")

        base = None
        for mode in ("C", "A", "B"):
            try:
                idx.fill_(-1)
                launch(logits, starts, ends, idx, num_rows, vocab, top_k, s0,
                       s1, mode)
                flaggems_vllm.runtime.torch_device_fn.synchronize()
                ok = correct(logits, idx, vocab, top_k)
                t = bench(lambda m=mode: launch(
                    logits, starts, ends, idx, num_rows, vocab, top_k, s0, s1, m))
            except Exception as e:  # noqa: BLE001
                print(f"    {NAMES[mode]:<22}   失败 {type(e).__name__}: "
                      f"{str(e).splitlines()[-1][:44]}")
                continue
            if mode == "C":
                base = t
            rel = f"{t / base:.3f}" if base else ""
            sp = f"{v / t:.3f}" if v else ""
            print(f"    {NAMES[mode]:<22}{t:>11.1f}{rel:>9}{sp:>9}   {ok}")
        print()

    print("  读法")
    print("    A 明显快于 C, 且只在「切到 radix 的行数>0」的形状上  => 行切分是负收益,")
    print("      改法是把 n_insert 设为 num_rows, 一行改动, 通用算子层面")
    print("    A ≈ C  => 8% 的差异不在这里, 得回到 decode 入口再找别的差异")
    print("    B 也快于 C => 快的是「单次启动」而不是「插入排序」, 结论不同")
    print("    两个对照形状上 A/B/C 必须几乎相同; 不同则是本探针的问题, 不是算子的")
    return 0


if __name__ == "__main__":
    sys.exit(main())
