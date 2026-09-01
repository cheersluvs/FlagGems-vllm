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
        print("\n  A 不符 => 问题在分 bin 或直方图累加，比压缩更早。后面两级不必看。")
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
    print("\n  读法")
    print("    A 不符            => 分 bin / 直方图，最早")
    print("    A 对、C 写入数不足 => 压缩仍在丢，scan 也没救回来")
    print("    A 对、写满但值错   => 选择或定序那一段")
    return 0


if __name__ == "__main__":
    sys.exit(main())
