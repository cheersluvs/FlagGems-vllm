"""Decode row split on Hygon, with every launch cached and direct.

Why re-test a split that "did not pay" here: that ceiling was WALL time, and
one-row decode is host-bound on this card -- ~110 us of Triton dispatch per
launch (device ~2 us when idle), which a split pays three extra times. In
device time the decode kernel is ~19 us + 1.16 ns/element, so 8 chunks model
at ~85 us of device work against 322. hygon_decode_direct_launch.py then
measured a cached CompiledKernel launched directly at ~12 us of host submit
(from 112), with the hit path's answer correct.

This builds the MetaX two-pass split (removed there in 3aec975; its kernels
are copied verbatim below) with all four launches direct and all buffers
preallocated, and sweeps rows x split at vocab 262144:

    stage 1  the generic non-TLE kernel over a chunked strided view
    gather   _gather_candidates: chunk-local indices -> candidate values
    merge    the generic kernel again, over split * top_k candidates
    remap    _remap_indices: merged positions -> row indices

Reported per point: wall us (CUDA events), host submit us, and the hit path's
answer against torch.topk; the shipped op and vLLM for reference.

    tools/vendor_probe.sh tools/hygon_decode_split_direct.py hygon_decode_split_direct
"""

import sys
import time
from importlib import import_module

import torch
import triton
import triton.language as tl

import flaggems_vllm  # noqa: F401  (runtime init)

V, K = 262144, 512
ROWS = (1, 4, 8, 16, 32, 56)
SPLITS = (1, 2, 4, 8, 16, 32)


@triton.jit
def _gather_candidates(
    logits_ptr,
    cand_ptr,
    out_ptr,
    stride0,
    stride1,
    floor,
    SPLIT: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    NCAND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = p < NCAND
    chunk_id = p // TOPK
    local = tl.load(
        cand_ptr + (row * SPLIT + chunk_id) * TOPK + (p % TOPK), mask=m, other=-1
    )
    ok = m & (local >= 0)
    val = tl.load(
        logits_ptr + row * stride0 + (chunk_id * CHUNK + local) * stride1,
        mask=ok,
        other=floor,
    )
    tl.store(out_ptr + row * NCAND + p, tl.where(ok, val, floor), mask=m)


@triton.jit
def _remap_indices(
    cand_ptr,
    merged_ptr,
    out_ptr,
    SPLIT: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    m = j < TOPK
    pos = tl.load(merged_ptr + row * TOPK + j, mask=m, other=0)
    live = m & (pos >= 0)
    chunk_id = pos // TOPK
    local = tl.load(
        cand_ptr + (row * SPLIT + chunk_id) * TOPK + (pos % TOPK), mask=live, other=-1
    )
    tl.store(
        out_ptr + row * TOPK + j,
        tl.where(live & (local >= 0), chunk_id * CHUNK + local, -1),
        mask=m,
    )


def direct(jit_fn, grid, args, constexprs, num_warps):
    """Compile once through JIT, then return a zero-dispatch launcher."""
    ck = jit_fn.run(*args, **constexprs, num_warps=num_warps, grid=grid, warmup=False)
    if ck is None:
        raise RuntimeError("kernel.run returned None on this Triton")
    g3 = tuple(grid) + (1,) * (3 - len(grid))
    runner = ck[g3]
    cvals = tuple(constexprs.values())
    return lambda: runner(*args, *cvals)


class Split:
    def __init__(self, dec, logits, rows, split):
        dev = logits.device
        self.logits = logits
        block, warps = dec.NUM_THREADS_PER_BLOCK, dec._num_warps(
            dec.NUM_THREADS_PER_BLOCK
        )
        kern = dec.non_tle_top_k_per_row_decode
        self.out = torch.empty(rows, K, dtype=torch.int32, device=dev)

        def scratch(n):
            return [
                torch.empty(s, dtype=d, device=dev)
                for s, d in (
                    ((n, dec.NUM_BINS), torch.int32),
                    ((n, dec.NUM_FILNAL_ITEMS), torch.float32),
                    ((n,), torch.int32),
                    ((n,), torch.int32),
                    ((n,), torch.int32),
                    ((n,), torch.int32),
                )
            ]

        if split == 1:
            lens = torch.full((rows,), V, dtype=torch.int32, device=dev)
            self.launches = [
                direct(
                    kern,
                    (rows,),
                    (logits, self.out, lens, 1, V, 1, V, *scratch(rows)),
                    {"TOPK": K, "BLOCK_SIZE": block},
                    warps,
                )
            ]
            return
        chunk = V // split
        nv = rows * split
        ncand = split * K
        view = logits.as_strided((nv, chunk), (chunk, 1))
        self.cand = torch.empty(nv, K, dtype=torch.int32, device=dev)
        sub_lens = torch.full((nv,), chunk, dtype=torch.int32, device=dev)
        self.vals = torch.empty(rows, ncand, dtype=logits.dtype, device=dev)
        self.merged = torch.empty(rows, K, dtype=torch.int32, device=dev)
        mlens = torch.full((rows,), ncand, dtype=torch.int32, device=dev)
        gblock = min(1024, triton.next_power_of_2(ncand))
        self.keep = (view, sub_lens, mlens)
        self.launches = [
            direct(
                kern,
                (nv,),
                (view, self.cand, sub_lens, 1, chunk, 1, chunk, *scratch(nv)),
                {"TOPK": K, "BLOCK_SIZE": block},
                warps,
            ),
            direct(
                _gather_candidates,
                (rows, triton.cdiv(ncand, gblock)),
                (logits, self.cand, self.vals, V, 1, torch.finfo(logits.dtype).min),
                {
                    "SPLIT": split,
                    "TOPK": K,
                    "CHUNK": chunk,
                    "NCAND": ncand,
                    "BLOCK": gblock,
                },
                4,
            ),
            direct(
                kern,
                (rows,),
                (self.vals, self.merged, mlens, 1, ncand, 1, ncand, *scratch(rows)),
                {"TOPK": K, "BLOCK_SIZE": block},
                warps,
            ),
            direct(
                _remap_indices,
                (rows,),
                (self.cand, self.merged, self.out),
                {
                    "SPLIT": split,
                    "TOPK": K,
                    "CHUNK": chunk,
                    "BLOCK": triton.next_power_of_2(K),
                },
                4,
            ),
        ]

    def __call__(self):
        for launch in self.launches:
            launch()


def wall_us(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000


def submit_us(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / iters * 1e6


def main():
    dec = import_module("flaggems_vllm.ops.top_k_per_row_decode")
    import vllm._custom_ops  # noqa: F401

    print(
        f"vocab {V}, top_k {K}; wall us / host-submit us per call; ratio = vLLM/ours;"
        f" ! = WRONG\n"
    )
    for rows in ROWS:
        torch.manual_seed(rows)
        logits = torch.randn(rows, V, dtype=torch.float32, device="cuda")
        want = torch.topk(logits, K, dim=1).values.sort(dim=1).values
        lens = torch.full((rows,), V, dtype=torch.int32, device="cuda")
        idx = torch.empty(rows, K, dtype=torch.int32, device="cuda")
        t_vllm = wall_us(
            lambda: torch.ops._C.top_k_per_row_decode(
                logits, 1, lens, idx, rows, V, 1, K
            )
        )
        t_op = wall_us(
            lambda: dec.top_k_per_row_decode(logits, 1, lens, idx, rows, V, 1, K)
        )
        print(
            f"rows {rows:>3}: vLLM {t_vllm:7.1f}  shipped op {t_op:7.1f} "
            f"(ratio {t_vllm / t_op:.3f})"
        )
        best = None
        for split in SPLITS:
            try:
                sp = Split(dec, logits, rows, split)
                sp()
                torch.cuda.synchronize()
                sp.out.fill_(-3)
                sp()  # hit path
                torch.cuda.synchronize()
                got = logits.gather(1, sp.out.long().clamp(0, V - 1)).sort(dim=1).values
                ok = torch.allclose(got, want) and bool((sp.out >= 0).all())
                w, s = wall_us(sp), submit_us(sp)
                mark = "" if ok else " !"
                print(
                    f"    split {split:>2} ({len(sp.launches)} launches): wall {w:7.1f}"
                    f"  submit {s:6.1f}  ratio {t_vllm / w:.3f}{mark}"
                )
                if ok and (best is None or w < best[0]):
                    best = (w, split)
                del sp
            except Exception as e:  # noqa: BLE001
                print(
                    f"    split {split:>2}: FAILED {type(e).__name__}: {str(e)[:100]}"
                )
            torch.cuda.empty_cache()
        if best:
            print(
                f"  -> best split {best[1]}: ratio {t_vllm / best[0]:.3f} "
                f"(shipped {t_vllm / t_op:.3f})\n"
            )


if __name__ == "__main__":
    sys.exit(main())
