"""Upper bound on what per-program scratch could buy -- results deliberately wrong.

A persistent rewrite of top_k_per_row would change three things: fewer launches
(measured at 10-37 ns per program, i.e. nothing), a scratch footprint of
O(cores) instead of O(rows), and the same scratch staying hot across the rows a
program handles.  Only the last is still worth anything, and testing it properly
means writing the persistent kernel first.

So bound it cheaply instead: keep the grid at one program per row, but fold the
scratch offsets onto WIDTH slots.  Concurrent programs then collide and the
output is WRONG -- that is the point.  The timing is still valid as an upper
bound on the locality effect, because the memory traffic pattern is exactly what
a persistent kernel would produce (a small resident set instead of a 270 MB
stream) while everything else is unchanged.

If even this is not faster, the locality argument is dead and the persistent
rewrite should not be written.

Correctness is checked and reported as FAIL on purpose, so no one mistakes a
run of this for a working configuration.
"""

import os
import sys
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton
from importlib import import_module

sys.path.insert(0, "src")
import flaggems_vllm  # noqa: F401

_gen = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
_asc = import_module(
    "flaggems_vllm.runtime.backend._ascend.fused.top_k_per_row_prefill"
)

NUM_BINS = _gen.NUM_BINS
NUM_FINAL = _gen.NUM_FILNAL_ITEMS
BLOCK = _asc.SCAN_BLOCK_SIZE
DEV = "npu"


def run(num_rows, vocab, top_k, width, reps=5, warm=2):
    """width=None keeps the real per-row scratch; an int folds it onto `width`."""
    logits = torch.randn((num_rows, vocab), dtype=torch.float32, device=DEV)
    row_starts = torch.zeros((num_rows,), dtype=torch.int32, device=DEV)
    row_ends = torch.full((num_rows,), vocab, dtype=torch.int32, device=DEV)
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=DEV)

    slots = num_rows if width is None else min(width, num_rows)
    hist = torch.empty((slots, NUM_BINS), device=DEV, dtype=torch.int32)
    fin = torch.empty((slots, NUM_FINAL), device=DEV, dtype=torch.float32)
    cnt = torch.empty((slots,), device=DEV, dtype=torch.int32)
    thr = torch.empty((slots,), device=DEV, dtype=torch.int32)
    bsz = torch.empty((slots,), device=DEV, dtype=torch.int32)
    fnd = torch.empty((slots,), device=DEV, dtype=torch.int32)

    # ROW_OFFSET is a constexpr the kernel adds to program_id; leaving it at 0
    # and shrinking the scratch is what makes rows alias -- the kernel still
    # indexes scratch by row_id, so with fewer slots it wraps into other rows'.
    # That is the collision we want for the timing, and the reason the answer
    # is wrong.
    def once():
        _asc.non_tle_top_k_per_row_prefill[(num_rows,)](
            logits, indices, row_starts, row_ends,
            logits.stride(0), logits.stride(1), vocab,
            hist, fin, cnt, thr, bsz, fnd,
            TOPK=top_k, BLOCK_SIZE=BLOCK, ROW_OFFSET=0,
            num_warps=_asc._num_warps(BLOCK),
        )

    for _ in range(warm):
        once()
    torch.npu.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        once()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()

    ref = torch.topk(logits.float(), top_k, dim=-1).indices
    got = indices.to(torch.int64)
    ok = torch.equal(got.sort(dim=-1).values, ref.sort(dim=-1).values)
    mb = (slots * NUM_BINS * 4 + slots * NUM_FINAL * 4) / 2**20
    return ts[len(ts) // 2], ok, mb


SHAPES = [(16383, 4095, 64), (12961, 4100, 64), (4100, 1025, 64)]
print(f"triton {triton.__version__} | BLOCK={BLOCK}")
print(f"\n{'shape':<20} {'scratch':>10} {'暂存 MB':>9} {'median ms':>10} {'正确':>6} {'相对':>7}")
for rows, vocab, k in SHAPES:
    base = None
    for width in (None, 40, 8):
        t, ok, mb = run(rows, vocab, k, width)
        if base is None:
            base = t
        tag = "每行" if width is None else f"折叠 {width}"
        print(f"{f'({rows}, {vocab})':<20} {tag:>10} {mb:>9.1f} {t:>10.2f} "
              f"{'OK' if ok else 'FAIL':>6} {t / base:>6.2f}x")

print("\n折叠版结果必然为 FAIL —— 并发程序共用暂存。只看时间：")
print("若折叠后并不更快，则 persistent 化的局部性收益不存在，不必写那个内核。")
