"""Two host-side prefill ideas, both free of correctness risk.

A. A wider block. The non-TLE entry launches at BLOCK_SIZE=512, which on this
   64-lane part is 8 warps. `(4, 8193)` runs 4 programs on 104 SMs, so the card
   is empty and the only parallelism available is inside one program -- 1024
   threads would double it. The wide-block heuristic that exists for TLE
   (`_wide_max_rows`) returns 0 on any part whose warp is not 32 lanes, so its
   crossover has never been measured here at all.

B. Cached scratch. The entry allocates six tensors per call. The caching
   allocator makes that cheap in absolute terms, which is why it cannot matter
   for a 2.8 ms shape -- but `(4, 8193)` is a 43 us call, so a few microseconds
   of allocation is around a tenth of it.

Both are measured against the same launch, so the deltas are attributable.
Correctness is checked on every variant; neither should change an answer.

    python tools/metax_prefill_host.py
"""

import sys
from importlib import import_module
from types import ModuleType

import torch

import flaggems_vllm

# ops/__init__ re-exports the function under the module's own name.
G = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
assert isinstance(G, ModuleType), "got the function, not the module"

DEV = flaggems_vllm.device
SMS = 104
SHAPES = [(4, 8193, 512, 8456), (4, 16385, 512, 16648),
          (64, 129280, 1024, 129280), (4100, 1025, 512, 1288),
          (16383, 4095, 512, 4352)]


def scratch(num_rows, device):
    return (
        torch.empty((num_rows, G.NUM_BINS), device=device, dtype=torch.int32),
        torch.empty((num_rows, G.NUM_FILNAL_ITEMS), device=device, dtype=torch.float32),
        torch.empty((num_rows,), device=device, dtype=torch.int32),
        torch.empty((num_rows,), device=device, dtype=torch.int32),
        torch.empty((num_rows,), device=device, dtype=torch.int32),
        torch.empty((num_rows,), device=device, dtype=torch.int32),
    )


def launch(lg, st, en, out, rows, s, top_k, block):
    G.non_tle_top_k_per_row_prefill[(rows,)](
        lg, out, st, en, lg.stride(0), lg.stride(1), lg.shape[1], *s,
        TOPK=top_k, BLOCK_SIZE=block, ROW_OFFSET=0,
        num_warps=G._num_warps(block),
    )


def timed(fn, iters=20, warmup=5):
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
    print(f"device {DEV} | {SMS} SMs | BLOCK_SIZE=512 is 8 warps here, 1024 is 16\n")
    print(f"  {'shape':>16} {'waves':>6} {'shipped':>9} {'cached':>9} {'blk=1024':>9} "
          f"{'both':>9} {'best':>7}  ok")
    for rows, vocab, top_k, stride0 in SHAPES:
        buf = torch.randn((rows - 1) * stride0 + vocab, device=DEV, dtype=torch.float32)
        lg = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        st = torch.zeros(rows, dtype=torch.int32, device=DEV)
        en = torch.full((rows,), vocab, dtype=torch.int32, device=DEV)
        out = torch.empty((rows, top_k), dtype=torch.int32, device=DEV)
        ref = torch.topk(lg, top_k, dim=-1).values.sort(-1, descending=True).values
        held = scratch(rows, DEV)          # allocated once, reused

        def run(block, cache):
            s = held if cache else scratch(rows, DEV)
            launch(lg, st, en, out, rows, s, top_k, block)

        res, ok = {}, True
        for name, block, cache in (("shipped", 512, False), ("cached", 512, True),
                                   ("blk1024", 1024, False), ("both", 1024, True)):
            try:
                res[name] = timed(lambda: run(block, cache))
                got = torch.gather(lg, 1, out.long()).sort(-1, descending=True).values
                ok &= bool(torch.allclose(got, ref, atol=1e-6, rtol=1e-6))
            except Exception as exc:  # noqa: BLE001 - a failing variant is a result
                res[name] = float("nan")
                print(f"      {name} failed: {type(exc).__name__}: {str(exc)[:70]}")
        base = res["shipped"]
        best = min(v for v in res.values() if v == v)
        print(f"  {f'({rows},{vocab})':>16} {rows / SMS:>6.2f} {base:>9.4f} "
              f"{res['cached']:>9.4f} {res['blk1024']:>9.4f} {res['both']:>9.4f} "
              f"{base / best:>6.2f}x  {'yes' if ok else 'WRONG'}")
    print("\n  Allocation shows up only where the call is short enough for a few")
    print("  microseconds to matter; a wide block only where the card is empty.")


if __name__ == "__main__":
    sys.exit(main())
