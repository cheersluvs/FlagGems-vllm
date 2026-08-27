#!/usr/bin/env python3
"""Run the REAL prefill kernel at BLOCK_SIZE 512 vs 1024, on the real shapes.

Ten hypotheses died before this one. The survivor: bandwidth on this card tracks
warps per SM almost linearly (16 warps 252 GB/s, 32 warps 476, 64 warps 815), and
at small num_rows the program count is fixed by the shape, so the only way to add
warps is threads per program. The isolated load loop confirms it -- BLOCK 512 ->
1024 measured 1.73x (252 -> 436 GB/s).

That was a microbenchmark of one loop. This drives the actual
`tle_top_k_per_row_prefill` at both block sizes across the seven benchmark
shapes, against vLLM, and checks correctness at each -- because BLOCK_SIZE is a
constexpr threaded through the histogram clear, the tile loops and the final
select, so 1024 is not obviously safe.

Expected, stated up front so it can be falsified:
  - small num_rows / large vocab  (64,129280) improves, perhaps 0.61 -> ~0.82
  - large num_rows / small vocab  (16383,4095) does NOT, and may regress: those
    already have 100+ warps/SM, and 1024 threads over a 4095-element row is
    4 elements per thread

If that is the split, the answer is a shape-dependent BLOCK, not a global one.

    VLLM_PLUGINS=musa PYTHONPATH=src python tools/mtt_block_size.py

Measurement only.
"""

import sys

import torch
import triton

import flaggems_vllm
from flaggems_vllm.ops.top_k_per_row_prefill import (
    SORTING_ALGORITHM_THRESHOLD,
    _use_radix_final_for_prefill,
    tle_top_k_per_row_prefill,
)

DEV = flaggems_vllm.device

HAS_VLLM = False
try:
    import vllm._custom_ops  # noqa: F401

    if hasattr(torch.ops._C, "top_k_per_row_prefill"):
        HAS_VLLM = True
except (ImportError, AttributeError, RuntimeError):
    pass

# The seven benchmark shapes: (num_rows, vocab, top_k, stride0)
SHAPES = [
    (64, 129280, 1024, 129280),
    (4, 8193, 512, 8456),
    (16383, 4095, 512, 4352),
    (4, 16385, 512, 16648),
    (12961, 4100, 512, 4360),
    (16380, 5115, 512, 5376),
    (4100, 1025, 512, 1288),
]


def _run(logits, starts, ends, idx, num_rows, s0, s1, top_k, block):
    """Replicates the generic dispatcher's TLE path with BLOCK_SIZE as a knob."""
    vocab = logits.shape[1]
    topkp = triton.next_power_of_2(top_k)
    use_radix_final = _use_radix_final_for_prefill(vocab)
    n_insert = 0 if use_radix_final else min(num_rows, SORTING_ALGORITHM_THRESHOLD)
    nw = block // 32

    if n_insert > 0:
        tle_top_k_per_row_prefill[(n_insert,)](
            logits, idx, starts, ends, s0, s1, vocab,
            TOPK=top_k, TOPKP=topkp, BLOCK_SIZE=block,
            USE_RADIX_FINAL=False, ROW_OFFSET=0, num_warps=nw,
        )
    if num_rows > n_insert:
        tle_top_k_per_row_prefill[(num_rows - n_insert,)](
            logits, idx, starts, ends, s0, s1, vocab,
            TOPK=top_k, TOPKP=topkp, BLOCK_SIZE=block,
            USE_RADIX_FINAL=True, ROW_OFFSET=n_insert, num_warps=nw,
        )


def _correct(logits, idx, top_k):
    """Selected values must match torch.topk, order-independent."""
    ref = torch.topk(logits[0], top_k, largest=True, sorted=False).indices
    got = idx[0].to(torch.int64)
    got = got[got >= 0]
    if got.numel() != min(top_k, logits.shape[1]):
        return False
    a = torch.sort(logits[0][got]).values
    b = torch.sort(logits[0][ref]).values
    return torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def sweep_crossover():
    """Where does the wide block stop paying? Measure it, do not guess it.

    The gate now shipping in the generic op uses num_rows <= 8, which was picked
    from two data points at num_rows=4. This walks num_rows across two vocabs so
    the constant is measured.

    It also probes the open contradiction: BLOCK 512 -> 1024 made the ISOLATED
    load loop 1.73x faster, yet made the real operator at (64, 129280) 0.82x
    slower. Something outside the load degrades sharply at 1024, and where the
    crossover sits says how sharply.
    """
    print("\n" + "=" * 78)
    print("  交叉点扫描: BLOCK=1024 从哪个 num_rows 开始不划算")
    print("=" * 78)
    for vocab, top_k, s0 in ((8193, 512, 8456), (129280, 1024, 129280)):
        print(f"\n  vocab={vocab}, top_k={top_k}")
        print(f"    {'rows':>6}{'512 us':>10}{'1024 us':>10}{'1024/512':>10}")
        for rows in (2, 4, 8, 16, 32, 64, 128):
            torch.manual_seed(42)
            buf = torch.randn((rows - 1) * s0 + vocab, device=DEV)
            logits = torch.as_strided(buf, (rows, vocab), (s0, 1))
            starts = torch.zeros((rows,), dtype=torch.int32, device=DEV)
            ends = torch.full((rows,), vocab, dtype=torch.int32, device=DEV)
            idx = torch.empty((rows, top_k), dtype=torch.int32, device=DEV)
            r = {}
            for blk in (512, 1024):
                try:
                    r[blk] = triton.testing.do_bench(
                        lambda b=blk: _run(
                            logits, starts, ends, idx, rows, s0, 1, top_k, b
                        ),
                        warmup=25, rep=100, return_mode="median",
                    )
                except Exception:  # noqa: BLE001
                    r[blk] = None
            if r[512] is None or r[1024] is None:
                print(f"    {rows:>6}   失败")
                continue
            g = r[512] / r[1024]
            mark = "  <-- 1024 更好" if g > 1.02 else ""
            print(
                f"    {rows:>6}{r[512]*1000:>10.1f}{r[1024]*1000:>10.1f}{g:>9.2f}x{mark}",
                flush=True,
            )
    print("\n  最后一个带标记的 rows = 门控阈值应该设的位置")


def main():
    print("=" * 78)
    print("  真算子 BLOCK_SIZE 512 vs 1024 -- 七个 benchmark 形状")
    print("=" * 78)
    if not HAS_VLLM:
        print("  !! 无 vLLM 基线, 用 VLLM_PLUGINS=musa 跑")
    print(f"  {'shape':>16}{'vLLM us':>10}{'512 us':>9}{'1024 us':>10}"
          f"{'sp512':>8}{'sp1024':>8}{'增益':>8}  正确")
    for num_rows, vocab, top_k, s0 in SHAPES:
        torch.manual_seed(42)
        buf = torch.randn((num_rows - 1) * s0 + vocab, device=DEV)
        logits = torch.as_strided(buf, (num_rows, vocab), (s0, 1))
        starts = torch.zeros((num_rows,), dtype=torch.int32, device=DEV)
        ends = torch.full((num_rows,), vocab, dtype=torch.int32, device=DEV)
        idx = torch.empty((num_rows, top_k), dtype=torch.int32, device=DEV)

        res = {}
        ok = {}
        for blk in (512, 1024):
            try:
                idx.fill_(-1)
                _run(logits, starts, ends, idx, num_rows, s0, 1, top_k, blk)
                flaggems_vllm.runtime.torch_device_fn.synchronize()
                ok[blk] = _correct(logits, idx, top_k)
                res[blk] = triton.testing.do_bench(
                    lambda b=blk: _run(
                        logits, starts, ends, idx, num_rows, s0, 1, top_k, b
                    ),
                    warmup=25, rep=100, return_mode="median",
                )
            except Exception as e:  # noqa: BLE001
                res[blk] = None
                ok[blk] = False
                print(f"  {f'({num_rows},{vocab})':>16}  BLOCK={blk} 失败: "
                      f"{type(e).__name__}: {str(e)[:40]}")

        v = None
        if HAS_VLLM:
            v = triton.testing.do_bench(
                lambda: torch.ops._C.top_k_per_row_prefill(
                    logits, starts, ends, idx, num_rows, s0, 1, top_k
                ),
                warmup=25, rep=100, return_mode="median",
            )

        t5, t10 = res.get(512), res.get(1024)
        if t5 is None or t10 is None:
            continue
        sp5 = v / t5 if v else float("nan")
        sp10 = v / t10 if v else float("nan")
        flag = "OK" if (ok[512] and ok[1024]) else "!! 错误"
        gain = t5 / t10
        mark = " <-- 更好" if gain > 1.02 else (" <-- 更差" if gain < 0.98 else "")
        print(
            f"  {f'({num_rows},{vocab})':>16}"
            f"{(v*1000 if v else float('nan')):>10.1f}{t5*1000:>9.1f}{t10*1000:>10.1f}"
            f"{sp5:>8.3f}{sp10:>8.3f}{gain:>7.2f}x  {flag}{mark}",
            flush=True,
        )
    print("\n  只有部分形状受益 => 按 shape 选 BLOCK; 全部受益 => 直接改默认值")
    sweep_crossover()
    return 0


if __name__ == "__main__":
    sys.exit(main())
