"""Is one-row decode's ~83 us of unexplained fixed cost a code path, or the launch?

After ablating the final select (0.4 us at vocab 4096 -- refuted), one-row
decode still sits at ~113 us flat from vocab 4096 to 65536 against a ~30 us
launch floor. Meanwhile a 4-row PREFILL call at vocab 8193 costs ~32 us in
total, and both enter the same _top_k_per_row_job.

One difference between those two calls: decode at vocab 4096 (and 262144) is
divisible by BLOCK_SIZE with row_start == 0 and row_end == vocab, so it takes
the `assume_aligned` branch; the prefill shape at 8193 takes the stride1 one.

Hold the compiled kernel fixed (vocab 4096, same constexprs) and move only
seq_len, which steers the branch at run time:

    4096           assume_aligned
    4095 4000 1024 stride1 branch, same kernel
    512  1         row_len <= TOPK: the kernel returns almost immediately

If seq_len=1 still costs ~110 us, the cost is launch or register setup of a
large kernel, not work executed. If only 4096 is slow, it is the aligned path.
Also prints each compiled kernel's regs/spills as a direct size measure, and
the same seq_len sweep for prefill (row_ends) for contrast.

    tools/vendor_probe.sh tools/hygon_decode_path_cost.py hygon_decode_path_cost
"""

import sys
from importlib import import_module

import torch

import flaggems_vllm  # noqa: F401  (runtime init)

VOCAB = 4096
K = 512
SEQS = (4096, 4095, 4000, 1024, 512, 1)


def timed(fn, iters=40, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000  # us


def size_of(jit_fn):
    out = []
    for dev_cache in getattr(jit_fn, "device_caches", {}).values():
        c = dev_cache[0] if isinstance(dev_cache, tuple) else dev_cache
        for ck in (c.values() if isinstance(c, dict) else []):
            md = getattr(ck, "metadata", None)
            out.append(
                f"regs={getattr(ck, 'n_regs', '?')} spills={getattr(ck, 'n_spills', '?')} "
                f"shared={getattr(md, 'shared', '?')} warps={getattr(md, 'num_warps', '?')}"
            )
    return out or ["(no compiled variant found)"]


def main():
    dec = import_module("flaggems_vllm.ops.top_k_per_row_decode")
    pre = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
    torch.manual_seed(0)
    logits = torch.randn(1, VOCAB, dtype=torch.float32, device="cuda")
    idx = torch.empty(1, K, dtype=torch.int32, device="cuda")

    print(f"=== decode, 1 row, vocab {VOCAB}, one compiled kernel; only seq_len moves")
    print(f"  {'seq_len':>7} {'us/call':>9}  path")
    for s in SEQS:
        lens = torch.full((1,), s, dtype=torch.int32, device="cuda")
        us = timed(
            lambda ln=lens: dec.top_k_per_row_decode(logits, 1, ln, idx, 1, VOCAB, 1, K)
        )
        path = (
            "assume_aligned"
            if s == VOCAB and VOCAB % 512 == 0
            else "row_len <= TOPK (early return)" if s <= K else "stride1"
        )
        print(f"  {s:>7} {us:>9.1f}  {path}")
    print("  compiled decode kernel(s):")
    for ln in size_of(dec.non_tle_top_k_per_row_decode):
        print(f"    {ln}")

    print(f"\n=== prefill, 1 row, vocab {VOCAB}; only row_end moves")
    print(f"  {'row_end':>7} {'us/call':>9}")
    starts = torch.zeros(1, dtype=torch.int32, device="cuda")
    for s in SEQS:
        ends = torch.full((1,), s, dtype=torch.int32, device="cuda")
        us = timed(
            lambda e=ends: pre.top_k_per_row_prefill(
                logits, starts, e, idx, 1, VOCAB, 1, K
            )
        )
        print(f"  {s:>7} {us:>9.1f}")
    print("  compiled prefill kernel(s):")
    for ln in size_of(pre.non_tle_top_k_per_row_prefill):
        print(f"    {ln}")

    print("\n  Launch floor of a trivial kernel on this card: ~30 us.")


if __name__ == "__main__":
    sys.exit(main())
