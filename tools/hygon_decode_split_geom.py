"""Decode 16-56 rows on Hygon: split factor x launch geometry, together.

After the split override, rows 16-56 read 0.535-0.805 of vLLM -- the biggest
remaining block. At 16 rows with split 8, stage one launches 128 programs on
80 SMs: 1.6 waves, the second half empty. prefill met the same shape and
narrow programs won up to 2.1x there (many rows means the GRID is already the
parallelism, so wide programs only crowd each other out). The decode split
still runs the generic geometry, BLOCK_SIZE=512 on 8 warps.

Split factor and geometry are coupled -- the factor sets how many programs,
the geometry how wide each is -- and a one-variable sweep of launch parameters
has misled twice on this card, so both axes move here.

The override builds a plan from the generic module's NUM_THREADS_PER_BLOCK and
_num_warps, so setting those and clearing the plan cache measures the
production path itself. Every point is checked against torch.topk first.

    tools/vendor_probe.sh tools/hygon_decode_split_geom.py hygon_decode_split_geom
"""

import os
import sys
from importlib import import_module

import torch

import flaggems_vllm

V, K = 262144, 512
ROWS = (16, 24, 32, 40, 56)
SPLITS = (4, 8, 16)
GEOMS = ((512, 8), (512, 4), (256, 4), (256, 2), (1024, 16))


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


def main():
    gen = import_module("flaggems_vllm.ops.top_k_per_row_decode")
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    import vllm._custom_ops  # noqa: F401

    base_block, base_warps = gen.NUM_THREADS_PER_BLOCK, gen._num_warps
    sms = ov._sm_count()
    print(
        f"vocab {V}, top_k {K}, {sms} SMs; ratio vs vLLM; "
        f"(r) = what ships today; ! = WRONG\n"
    )
    for rows in ROWS:
        torch.manual_seed(rows)
        logits = torch.randn(rows, V, dtype=torch.float32, device="cuda")
        want = torch.topk(logits, K, dim=1).values.sort(dim=1).values
        lens = torch.full((rows,), V, dtype=torch.int32, device="cuda")
        idx = torch.empty(rows, K, dtype=torch.int32, device="cuda")
        os.environ.pop("FLAGGEMS_HYGON_TOPK_DECODE_SPLIT", None)
        shipped_split = ov._split_factor(rows, V, K)
        t_vllm = wall_us(
            lambda: torch.ops._C.top_k_per_row_decode(
                logits, 1, lens, idx, rows, V, 1, K
            )
        )
        print(f"  rows {rows:>3} (ships split {shipped_split}, vLLM {t_vllm:6.1f} us)")
        print(
            f"    {'geometry':<14} "
            + " ".join(f"{'split ' + str(s):>10}" for s in SPLITS)
            + f"  {'programs (stage 1)':>20}"
        )
        best = None
        for block, warps in GEOMS:
            cells = []
            for split in SPLITS:
                os.environ["FLAGGEMS_HYGON_TOPK_DECODE_SPLIT"] = str(split)
                gen.NUM_THREADS_PER_BLOCK = block
                gen._num_warps = lambda b, w=warps: w
                ov._PLANS.clear()
                try:
                    call = lambda: flaggems_vllm.top_k_per_row_decode(  # noqa: E731
                        logits, 1, lens, idx, rows, V, 1, K
                    )
                    call()
                    torch.cuda.synchronize()
                    got = (
                        logits.gather(1, idx.long().clamp(0, V - 1)).sort(dim=1).values
                    )
                    ok = torch.allclose(got, want)
                    ratio = t_vllm / wall_us(call)
                    ships = (block, warps) == (
                        base_block,
                        base_warps,
                    ) and split == shipped_split
                    cells.append(
                        f"{ratio:>9.3f}{'!' if not ok else ('r' if ships else ' ')}"
                    )
                    if ok and (best is None or ratio > best[0]):
                        best = (ratio, block, warps, split)
                except Exception as e:  # noqa: BLE001
                    cells.append(f"{type(e).__name__[:9]:>10}")
            progs = " ".join(f"{rows * s:>10}" for s in SPLITS)
            print(f"    B{block:<5} w{warps:<6} " + " ".join(cells) + f"  {progs}")
        gen.NUM_THREADS_PER_BLOCK, gen._num_warps = base_block, base_warps
        ov._PLANS.clear()
        os.environ.pop("FLAGGEMS_HYGON_TOPK_DECODE_SPLIT", None)
        if best:
            print(
                f"    -> best {best[0]:.3f} at B{best[1]} w{best[2]} split {best[3]}\n"
            )
    print("  Stage one launches rows*split programs; the merge launches rows.")


if __name__ == "__main__":
    sys.exit(main())
