#!/usr/bin/env python3
"""Which stage of the operator first disagrees with a host reference?

The atomic defect was found by noticing that the histogram came out complete
while the output did not -- one stage right, the next wrong. With the scan
compaction in place that asymmetry is gone and the result is still wrong, so the
same method is applied stage by stage, each against a reference computed on the
host from the operator's own bit trick.

    A. histogram      _distribute_to_bins over the row, STEP 0
    B. counters       the whole operator, then s_found_topk_values / s_final_cnt
    C. output         how many of top_k slots were written, and are they right

The first stage that disagrees is where to look. Run it on a known-good card
too: every number here has a host reference, so a stage that fails on BOTH is a
bug in this probe, not in the backend.

    VLLM_PLUGINS=musa PYTHONPATH=src python tools/stage_compare.py          # 对照
    source ...set_env.sh; PYTHONPATH=src:$PYTHONPATH python tools/stage_compare.py
"""

import os
import sys
from importlib import import_module

import torch
import triton
import triton.language as tl

import flaggems_vllm

# import_module: ops/__init__ re-exports the FUNCTION under this name.
M = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
DEV = flaggems_vllm.device

VOCAB, TOPK = 20000, 1024
BLOCK = M._compaction_block_size()


def host_bins(x):
    """_extract_bin_idx STEP 0, in torch.

    f32 -> f16 -> raw bits; negatives keep their pattern, positives are
    inverted, then the top 11 bits are the bin. Lower bin means larger value.
    """
    h = x.to(torch.float16)
    bits = h.view(torch.int16).to(torch.int32) & 0xFFFF
    sign_set = (bits & 0x8000) != 0
    inv = (~bits) & 0x7FFF
    mapped = torch.where(sign_set, bits, inv)
    return mapped >> 5


@triton.jit
def _hist_only(logits_ptr, hist_ptr, N, BLOCK_SIZE: tl.constexpr):
    lane = tl.arange(0, BLOCK_SIZE)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    for t in tl.range(0, tl.cdiv(N, BLOCK_SIZE)):
        offs = t * BLOCK_SIZE + lane
        in_range = offs < N
        x = tl.load(logits_ptr + offs, mask=in_range, other=float("-inf"))
        M_distribute(x, in_range, ones, 0, hist_ptr, STEP=0)


M_distribute = M._distribute_to_bins
M_extract = M._extract_bin_idx


@triton.jit
def _bins_only(logits_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Store bin_idx itself, before any atomic touches it.

    Separates 'the bin is computed wrong' from 'the histogram write is
    dropped'. Both look identical in a histogram comparison.
    """
    lane = tl.arange(0, BLOCK_SIZE)
    for t in tl.range(0, tl.cdiv(N, BLOCK_SIZE)):
        offs = t * BLOCK_SIZE + lane
        in_range = offs < N
        x = tl.load(logits_ptr + offs, mask=in_range, other=float("-inf"))
        b, _ = M_extract(x, in_range, 0, STEP=0)
        tl.store(out_ptr + offs, b.to(tl.int32), mask=in_range)


def main():
    print("=" * 84)
    print("  逐阶段比对：第一处和主机参照不符的地方")
    print("=" * 84)
    print(f"  vendor={flaggems_vllm.vendor_name}  BLOCK={BLOCK}  "
          f"HAS_ATOMIC_RETURN={M.HAS_ATOMIC_RETURN}")
    print(f"  vocab={VOCAB}  top_k={TOPK}\n")

    torch.manual_seed(0)
    logits = torch.randn((1, VOCAB), dtype=torch.float32, device=DEV)

    # ---- 主机参照 ----
    b = host_bins(logits[0])
    ref_hist = torch.bincount(b, minlength=2048)
    csum = torch.cumsum(ref_hist, 0)
    thr_bin = int((csum >= TOPK).nonzero()[0, 0])
    n_below = int(csum[thr_bin - 1]) if thr_bin > 0 else 0
    n_in_bin = int(ref_hist[thr_bin])
    print(f"  主机参照: 阈值 bin={thr_bin}  bin 之前有 {n_below} 个  "
          f"bin 内 {n_in_bin} 个")

    # ---- A0. bin_idx 本身 ----
    bins_dev = torch.full((VOCAB,), -12345, dtype=torch.int32, device=DEV)
    _bins_only[(1,)](logits, bins_dev, VOCAB, BLOCK_SIZE=BLOCK,
                     num_warps=M._num_warps(BLOCK))
    flaggems_vllm.runtime.torch_device_fn.synchronize()
    ref_b = b.to(torch.int32)
    eq = int((bins_dev == ref_b).sum())
    neg_mask = logits[0] < 0
    eq_pos = int((bins_dev[~neg_mask] == ref_b[~neg_mask]).sum())
    eq_neg = int((bins_dev[neg_mask] == ref_b[neg_mask]).sum())
    n_pos, n_neg = int((~neg_mask).sum()), int(neg_mask.sum())
    print(f"\n  A0 bin_idx: 一致 {eq}/{VOCAB}   "
          f"正值 {eq_pos}/{n_pos}   负值 {eq_neg}/{n_neg}")
    if eq != VOCAB:
        bad = (bins_dev != ref_b).nonzero().flatten()[:5].tolist()
        for i in bad:
            print(f"      元素 {i}: 值 {float(logits[0][i]):+.4f}  "
                  f"内核 bin {int(bins_dev[i])}  参照 {int(ref_b[i])}")
        print("      范围: 内核 [" + str(int(bins_dev.min())) + ","
              + str(int(bins_dev.max())) + "]  参照 ["
              + str(int(ref_b.min())) + "," + str(int(ref_b.max())) + "]")

    # ---- A. 直方图 ----
    hist = torch.zeros(2048, dtype=torch.int32, device=DEV)
    _hist_only[(1,)](logits, hist, VOCAB, BLOCK_SIZE=BLOCK,
                     num_warps=M._num_warps(BLOCK))
    flaggems_vllm.runtime.torch_device_fn.synchronize()
    same = int((hist == ref_hist.to(torch.int32)).sum())
    total = int(hist.sum())
    a_ok = same == 2048 and total == VOCAB
    print(f"\n  A 直方图: {'一致 ✓' if a_ok else '不符 ✗'}  "
          f"相同 bin {same}/2048  总计数 {total}/{VOCAB}")
    if not a_ok:
        d = (hist != ref_hist.to(torch.int32)).nonzero().flatten()[:6].tolist()
        for i in d:
            print(f"      bin {i}: 内核 {int(hist[i])}  参照 {int(ref_hist[i])}")
        print("\n  A 不符。对照 A0：A0 也错 => 分 bin 本身错；")
        print("  A0 对而 A 错 => bin 对但直方图写入被丢。")
        return 0

    # ---- B/C. 整算子 ----
    SENT = -987654321
    idx = torch.full((1, TOPK), SENT, dtype=torch.int32, device=DEV)
    starts = torch.zeros(1, dtype=torch.int32, device=DEV)
    ends = torch.full((1,), VOCAB, dtype=torch.int32, device=DEV)
    M.top_k_per_row_prefill(logits, starts, ends, idx, 1,
                            logits.stride(0), logits.stride(1), TOPK)
    flaggems_vllm.runtime.torch_device_fn.synchronize()

    written = int((idx != SENT).sum())
    got = idx[0][idx[0] != SENT].to(torch.int64)
    in_range = int(((got >= 0) & (got < VOCAB)).sum())
    print(f"\n  C 输出: 写入 {written}/{TOPK}  其中下标合法 {in_range}/{written}")
    if written:
        vals = logits[0][got[(got >= 0) & (got < VOCAB)]]
        ref_vals = torch.topk(logits[0], TOPK, largest=True, sorted=False).values
        thr_val = float(ref_vals.min())
        n_true = int((vals >= thr_val).sum())
        print(f"      写进去的值里真正属于 top-{TOPK} 的: {n_true}/{int(vals.numel())}")
    # ---- D. 成本拆分 ----
    # The scan turned out 6x FASTER than the atomic on this backend, the
    # opposite of Moore Threads, because without TLE the scratch lives in
    # GLOBAL memory and the compaction's per-lane atomics were global ones.
    # What the scan does NOT touch is the histogram: _distribute_to_bins still
    # issues one global atomic per input element per refinement step. Time that
    # alone against the whole operator and see how much of it that is.
    def ms(fn):
        return triton.testing.do_bench(fn, warmup=10, rep=200,
                                       return_mode="median")

    big_rows, big_vocab = 4096, 4100
    lg = torch.randn((big_rows, big_vocab), dtype=torch.float32, device=DEV)
    st = torch.zeros(big_rows, dtype=torch.int32, device=DEV)
    en = torch.full((big_rows,), big_vocab, dtype=torch.int32, device=DEV)
    oi = torch.empty((big_rows, 512), dtype=torch.int32, device=DEV)
    hh = torch.zeros((big_rows, 2048), dtype=torch.int32, device=DEV)

    t_hist = ms(lambda: _hist_only[(big_rows,)](
        lg, hh, big_vocab, BLOCK_SIZE=BLOCK, num_warps=M._num_warps(BLOCK)))
    t_full = ms(lambda: M.top_k_per_row_prefill(
        lg, st, en, oi, big_rows, lg.stride(0), lg.stride(1), 512))
    t_torch = ms(lambda: torch.topk(lg, 512, dim=1, largest=True, sorted=False))

    print(f"\n  D 成本拆分  形状 ({big_rows}, {big_vocab})  top_k=512")
    print(f"      只建直方图（一遍）  {t_hist:9.2f} ms")
    print(f"      整算子              {t_full:9.2f} ms   "
          f"直方图占 {100 * t_hist / t_full:.0f}%")
    print(f"      torch.topk          {t_torch:9.2f} ms")

    print("\n  读法")
    print("    D 里直方图占大头 => 瓶颈是每元素一次全局原子，非 TLE 路径的结构问题，")
    print("      和我们的绕法无关；要提速得让 scratch 上片，而这块卡没有 TLE")
    print("    D 里直方图是零头 => 瓶颈在别处（最终定序那段是 O(n^2) 的插入排序）")
    print("    A 不符            => 分 bin / 直方图，最早")
    print("    A 对、C 写入数不足 => 压缩仍在丢，scan 也没救回来")
    print("    A 对、写满但值错   => 选择或定序那一段")
    return 0


if __name__ == "__main__":
    sys.exit(main())
