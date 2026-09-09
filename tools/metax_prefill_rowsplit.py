"""Is splitting prefill's launch by row worth anything on its own?

vLLM splits its prefill at 12288 rows into two kernels, and the measurement
says that split COSTS it time rather than saving it: its cheap path runs at
75.5 ns/row and its expensive one at 344, so (16383,4095) takes 2.34 ms where
all-cheap would be 1.24. The threshold is the same constant this repo calls
SORTING_ALGORITHM_THRESHOLD, and what it caps is how many rows can have the
2048-wide scratch buffer -- 16383 rows of it is 134 MB. A capacity compromise,
not an optimisation.

We give every row that buffer and run one launch, so copying their split whole
would mean adopting their slower path for the tail. But the split has a second,
separable effect worth pricing: two launches instead of one, over the same code
path. If THAT is worth something, it is worth having; if it is neutral, then
row splitting per se buys nothing and only the algorithm change does -- which
the non-TLE kernel cannot do, having no USE_RADIX_FINAL.

Both halves here run the identical kernel. Only the launch is split.

    python tools/metax_prefill_rowsplit.py
"""

import sys
from importlib import import_module
from types import ModuleType

import torch
import triton

import flaggems_vllm

# NOT `from flaggems_vllm.ops import top_k_per_row_prefill as G`: ops/__init__
# re-exports the FUNCTION under that name, which shadows the module. Third time
# in this campaign; the assert makes the next one fail at the import instead of
# at the first attribute.
G = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
assert isinstance(G, ModuleType), "got the function, not the module"

DEV = flaggems_vllm.device
SMS = 104
SHAPES = [(16383, 4095, 512, 4352), (12961, 4100, 512, 4360),
          (16380, 5115, 512, 5376), (4100, 1025, 512, 1288)]


def scratch(num_rows, device):
    return (
        torch.empty((num_rows, G.NUM_BINS), device=device, dtype=torch.int32),
        torch.empty((num_rows, G.NUM_FILNAL_ITEMS), device=device, dtype=torch.float32),
        torch.empty((num_rows,), device=device, dtype=torch.int32),
        torch.empty((num_rows,), device=device, dtype=torch.int32),
        torch.empty((num_rows,), device=device, dtype=torch.int32),
        torch.empty((num_rows,), device=device, dtype=torch.int32),
    )


def launch(logits, starts, ends, out, num_rows, s, top_k, splits):
    """splits: row boundaries, e.g. [0, 12288, num_rows]."""
    for a, b in zip(splits, splits[1:]):
        if b <= a:
            continue
        G.non_tle_top_k_per_row_prefill[(b - a,)](
            logits, out, starts, ends,
            logits.stride(0), logits.stride(1), logits.shape[1],
            *s,
            TOPK=top_k,
            BLOCK_SIZE=G.NUM_THREADS_PER_BLOCK,
            ROW_OFFSET=a,
            num_warps=G._num_warps(G.NUM_THREADS_PER_BLOCK),
        )


def timed(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main():
    print(f"device {DEV} | {SMS} SMs | both halves run the SAME kernel\n")
    print(f"  {'shape':>16} {'one launch':>11} {'split@12288':>12} {'even split':>11} "
          f"{'best vs one':>12}")
    for rows, vocab, top_k, stride0 in SHAPES:
        buf = torch.randn((rows - 1) * stride0 + vocab, device=DEV, dtype=torch.float32)
        lg = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        st = torch.zeros(rows, dtype=torch.int32, device=DEV)
        en = torch.full((rows,), vocab, dtype=torch.int32, device=DEV)
        out = torch.empty((rows, top_k), dtype=torch.int32, device=DEV)
        s = scratch(rows, DEV)

        ref = torch.topk(lg, top_k, dim=-1).values.sort(-1, descending=True).values
        plans = {
            "one": [0, rows],
            "12288": [0, min(G.SORTING_ALGORITHM_THRESHOLD, rows), rows],
            "even": [0, rows // 2, rows],
        }
        res = {}
        for name, sp in plans.items():
            t = timed(lambda: launch(lg, st, en, out, rows, s, top_k, sp))
            got = torch.gather(lg, 1, out.long()).sort(-1, descending=True).values
            res[name] = (t, torch.allclose(got, ref, atol=1e-6, rtol=1e-6))
        one = res["one"][0]
        best = min(v[0] for v in res.values())
        ok = all(v[1] for v in res.values())
        print(f"  {f'({rows},{vocab})':>16} {one:>11.4f} {res['12288'][0]:>12.4f} "
              f"{res['even'][0]:>11.4f} {one / best:>11.3f}x"
              f"{'' if ok else '   CORRECTNESS FAILED'}")
    print("\n  Near 1.000x means the split buys nothing by itself, and vLLM's")
    print("  advantage is entirely in what its second kernel DOES differently.")


if __name__ == "__main__":
    sys.exit(main())
