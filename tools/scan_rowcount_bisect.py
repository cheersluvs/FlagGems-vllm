#!/usr/bin/env python3
"""Where does the scan compaction start disagreeing with torch?

With FLAGGEMS_ATOMIC_RETURN=0 the suite is 18/20 on Moore Threads. The two
failures are test_top_k_per_row_prefill_variable_lengths at num_rows=16383, for
both vocabularies; the same test at num_rows=4 passes, and the whole suite
passes on the atomic path. So the scan is wrong somewhere only large,
variable-length batches reach.

Two candidates, and a sweep of num_rows separates them:

  * the host splits at SORTING_ALGORITHM_THRESHOLD = 12288 -- above it there is
    a SECOND launch with USE_RADIX_FINAL=True for the remaining rows
  * row_ends is drawn from [top_k, vocab], so a row of length exactly top_k --
    which takes the short-row branch -- is near-certain at 16383 rows and about
    0.1% likely at 4

If the boundary is at 12288 it is the split; if failures appear as soon as the
batch is large enough to contain a top_k-length row, it is the short-row branch.

    VLLM_PLUGINS=musa PYTHONPATH=src python tools/scan_rowcount_bisect.py

Runs both paths for every size, so a size that fails on scan and passes on
atomic is attributable, and one that fails on both is not the scan's fault.
"""

import os
import sys

import torch

import flaggems_vllm

DEV = flaggems_vllm.device
VOCAB, TOPK = 4095, 512
SIZES = [4, 64, 1024, 8192, 12287, 12288, 12289, 13000, 16383]


def reference(logits, row_starts, row_ends, top_k):
    out = torch.full((logits.shape[0], top_k), -1, dtype=torch.int32,
                     device=logits.device)
    for i in range(logits.shape[0]):
        s, e = int(row_starts[i]), int(row_ends[i])
        k = min(top_k, e - s)
        idx = torch.topk(logits[i, s:e], k, largest=True, sorted=False).indices
        out[i, :k] = idx.to(torch.int32)
    return out


def values_match(logits, got, ref, row_starts, top_k):
    """Compare the SELECTED VALUES as multisets; index ties are allowed."""
    n = logits.shape[0]
    for i in range(n):
        s = int(row_starts[i])
        g = got[i][got[i] >= 0].to(torch.int64)
        r = ref[i][ref[i] >= 0].to(torch.int64)
        if g.numel() != r.numel():
            return False, i, f"数量 {g.numel()} vs {r.numel()}"
        a = torch.sort(logits[i, s:][g]).values
        b = torch.sort(logits[i, s:][r]).values
        if not torch.equal(a, b):
            return False, i, "值不符"
    return True, -1, ""


def run(num_rows, use_atomic):
    os.environ["FLAGGEMS_ATOMIC_RETURN"] = "1" if use_atomic else "0"
    # Re-import so the module-level gate is re-evaluated.
    import importlib
    import flaggems_vllm.ops.top_k_per_row_prefill as M
    importlib.reload(M)

    torch.manual_seed(123)
    logits = torch.randn(num_rows, VOCAB, device=DEV, dtype=torch.float32)
    starts = torch.zeros(num_rows, dtype=torch.int32, device=DEV)
    ends = torch.randint(TOPK, VOCAB + 1, (num_rows,), dtype=torch.int32,
                         device=DEV)
    got = torch.empty((num_rows, TOPK), dtype=torch.int32, device=DEV)
    M.top_k_per_row_prefill(logits, starts, ends, got, num_rows,
                            logits.stride(0), logits.stride(1), TOPK)
    flaggems_vllm.runtime.torch_device_fn.synchronize()
    ref = reference(logits, starts, ends, TOPK)
    ok, row, why = values_match(logits, got, ref, starts, TOPK)
    n_eq_topk = int((ends - starts == TOPK).sum())
    return ok, row, why, n_eq_topk


def main():
    print("=" * 84)
    print("  scan 压缩：行数从哪里开始出错")
    print("=" * 84)
    print(f"  vocab={VOCAB}  top_k={TOPK}  行长随机取 [{TOPK}, {VOCAB}]")
    print(f"  拆分阈值 SORTING_ALGORITHM_THRESHOLD = 12288\n")
    print(f"  {'num_rows':>9}{'恰好=top_k 的行数':>18}{'原子':>8}{'scan':>8}   首个出错行")
    for n in SIZES:
        a_ok, _, _, n_eq = run(n, True)
        s_ok, s_row, s_why, _ = run(n, False)
        note = "" if s_ok else f"第 {s_row} 行 {s_why}"
        print(f"  {n:>9}{n_eq:>18}{'OK' if a_ok else 'FAIL':>8}"
              f"{'OK' if s_ok else 'FAIL':>8}   {note}", flush=True)
    print("\n  读法")
    print("    scan 在 12289 开始错而 12287 对  => 是两次启动/radix-final 那条路")
    print("    scan 从「恰好=top_k 的行数」首次非零起错 => 是短行分支")
    print("    两者都不对齐                    => 另有原因，继续二分行数")
    print("    原子那一列必须全 OK；不全 OK 就说明这个探针本身有问题")
    return 0


if __name__ == "__main__":
    sys.exit(main())
