#!/usr/bin/env python3
"""Does MUSA graph capture shrink the run-to-run spread on the tiny shapes?

The benchmark's `kernel` mode is already the most robust of the three it offers
-- triton.testing.do_bench, median, L2 flush -- and `--iter` there is a duration
in milliseconds, so a 34 us shape already gets ~1300 reps per number. The spread
we actually see is therefore NOT within-run: it is between runs of the same
build, in separate processes.

Reading the raw millisecond columns rather than the speedups makes that
concrete. Four runs of identical code at (4, 8193):

    gems   0.029680   0.034240   0.034120   0.034080     <- three within 0.5%
    torch  0.026640   0.025800   0.026340   0.025920

So it is not a steady 22% jitter -- that figure came from comparing speedups and
including one outlying run. Per side it is 2-4%, the ratio compounds both sides
to ~5%, and occasionally a whole run lands 13% off.

Graph capture removes host-side dispatch from the measurement, which is a
plausible source of that. It also changes what is measured, so both sides must
be captured or the ratio is meaningless.

This probe therefore reports FOUR things per shape: eager and graph, each over
several independent measurement rounds, with the spread of each. The spread is
the answer; a single faster number would not be.

    VLLM_PLUGINS=musa PYTHONPATH=src python tools/mtt_graph_noise.py
    VLLM_PLUGINS=musa PYTHONPATH=src python tools/mtt_graph_noise.py --rounds 9

Measurement only. Registers nothing, changes no shipped file.
"""

import argparse
import statistics
import sys

import torch
import triton

import flaggems_vllm

DEV = flaggems_vllm.device

try:
    import vllm._custom_ops  # noqa: F401

    HAS_VLLM = hasattr(torch.ops._C, "top_k_per_row_prefill")
except (ImportError, AttributeError, RuntimeError):
    HAS_VLLM = False

# num_rows, vocab, top_k, stride0, stride1 -- the two shapes whose numbers are
# in doubt, plus the decisive one as a control that should stay put either way.
SHAPES = [
    (4, 8193, 512, 8456, 1),
    (4, 16385, 512, 16648, 1),
    (64, 129280, 1024, 129280, 1),
]


def make_inputs(num_rows, vocab, top_k, stride0, stride1):
    """Exactly the benchmark's construction: a strided view, not a fresh 2-D."""
    torch.manual_seed(42)
    buf = torch.randn(
        (num_rows - 1) * stride0 + (vocab - 1) * stride1 + 1,
        device=DEV, dtype=torch.float32,
    )
    logits = torch.as_strided(buf, (num_rows, vocab), (stride0, stride1))
    starts = torch.zeros(num_rows, dtype=torch.int32, device=DEV)
    ends = torch.full((num_rows,), vocab, dtype=torch.int32, device=DEV)
    idx = torch.empty((num_rows, top_k), dtype=torch.int32, device=DEV)
    return logits, starts, ends, idx


def capture(fn):
    """Capture fn into a MUSA graph, warming up on a side stream first.

    Returns the graph, or a string explaining why it could not be captured --
    a failure here is a datapoint, not a reason to abandon the run.
    """
    try:
        s = torch.musa.Stream()
        s.wait_stream(torch.musa.current_stream())
        with torch.musa.stream(s):
            for _ in range(5):
                fn()
        torch.musa.current_stream().wait_stream(s)
        torch.musa.synchronize()
        g = torch.musa.MUSAGraph()
        ctx = getattr(torch.musa, "graph", None) or getattr(torch.musa, "musa_graph")
        with ctx(g):
            fn()
        torch.musa.synchronize()
        return g
    except Exception as e:  # noqa: BLE001 - the failure IS the result
        return f"{type(e).__name__}: {str(e).splitlines()[0][:60]}"


def bench(fn, reps):
    return triton.testing.do_bench(fn, warmup=25, rep=reps,
                                   return_mode="median") * 1000


def spread(xs):
    return (max(xs) - min(xs)) / statistics.median(xs) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=100,
                    help="do_bench rep in MILLISECONDS, as the benchmark uses it")
    args = ap.parse_args()

    print("=" * 84)
    print("  MUSA graph 捕获能否压住轮间方差")
    print("=" * 84)
    if not HAS_VLLM:
        print("  !! 无 vLLM 基线，比值无意义，退出")
        return 1
    print(f"  每形状 {args.rounds} 轮独立测量，每轮 do_bench rep={args.reps} ms")
    print(f"  出货绑定: {flaggems_vllm.top_k_per_row_prefill.__module__}\n")

    for num_rows, vocab, top_k, s0, s1 in SHAPES:
        logits, starts, ends, idx = make_inputs(num_rows, vocab, top_k, s0, s1)
        vidx = torch.empty_like(idx)

        def gems():
            flaggems_vllm.top_k_per_row_prefill(
                logits, starts, ends, idx, num_rows, s0, s1, top_k)

        def base():
            torch.ops._C.top_k_per_row_prefill(
                logits, starts, ends, vidx, num_rows, s0, s1, top_k)

        g_gems, g_base = capture(gems), capture(base)
        graph_ok = not isinstance(g_gems, str) and not isinstance(g_base, str)

        print(f"  ({num_rows},{vocab},k={top_k})"
              + ("" if graph_ok else "   图捕获失败"))
        if not graph_ok:
            for nm, g in (("gems", g_gems), ("vLLM", g_base)):
                if isinstance(g, str):
                    print(f"    {nm}: {g}")

        rec = {"eager": {"g": [], "v": [], "sp": []},
               "graph": {"g": [], "v": [], "sp": []}}
        for _ in range(args.rounds):
            tg, tv = bench(gems, args.reps), bench(base, args.reps)
            rec["eager"]["g"].append(tg)
            rec["eager"]["v"].append(tv)
            rec["eager"]["sp"].append(tv / tg)
            if graph_ok:
                tg2 = bench(lambda: g_gems.replay(), args.reps)
                tv2 = bench(lambda: g_base.replay(), args.reps)
                rec["graph"]["g"].append(tg2)
                rec["graph"]["v"].append(tv2)
                rec["graph"]["sp"].append(tv2 / tg2)

        print(f"    {'模式':<8}{'gems µs':>22}{'跨度':>8}"
              f"{'vLLM µs':>22}{'跨度':>8}{'speedup 中位':>13}{'跨度':>8}")
        for mode in ("eager", "graph"):
            r = rec[mode]
            if not r["g"]:
                continue
            gs = "/".join(f"{x:.1f}" for x in r["g"][:4])
            vs = "/".join(f"{x:.1f}" for x in r["v"][:4])
            print(f"    {mode:<8}{gs:>22}{spread(r['g']):>7.1f}%"
                  f"{vs:>22}{spread(r['v']):>7.1f}%"
                  f"{statistics.median(r['sp']):>13.3f}{spread(r['sp']):>7.1f}%")
        print()

    print("  读法")
    print("    graph 行的 speedup 跨度明显小于 eager  => 抖动来自主机端派发，")
    print("      值得把这两个形状的数字改用 graph 模式重报（但不能与历史数据混用）")
    print("    两者跨度相当                          => 抖动是设备侧的（时钟/其他进程），")
    print("      graph 解决不了，只能多轮取中位数")
    print("    graph 的绝对耗时明显更低              => 启动开销占比大，这本身就是")
    print("      小形状难测的原因，也说明它们的比值天然不稳")
    return 0


if __name__ == "__main__":
    sys.exit(main())
