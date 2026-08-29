#!/usr/bin/env python3
"""How much of a benchmark number is real?

Measured across this session, the same code read:

    (16380,5115)  2474 us   1.520 / 1.521 / 1.523 / 1.524      0.3%
    (64,129280)    157 us   0.902 / 0.906 / 0.908              0.7%
    (4,16385)       38 us   0.759 / 0.786 / 0.809 / 0.817 ...  9%
    (4,8193)        30 us   0.665 / 0.733 / 0.754 / 0.772 ...  35%

so the crossover sits near 100 us of absolute kernel time. Splitting the worst
one into its parts shows the surprise: the torch baseline moved 3% while OUR
kernel moved 15%, i.e. this is not a small-number-division artifact, it is the
4-program launch itself being unstable.

`do_bench` already takes a median over thousands of iterations, so within a run
it is steady; everything above is BETWEEN-run drift -- clocks, power state, what
ran before. This tool measures that drift instead of guessing at it, and gives
every shape a noise floor to compare a claimed optimisation against. The largest
misjudgement in this operator's optimisation came from not having one: a 2.8%
"regression" was believed because two low-rep measurements agreed.

    VLLM_PLUGINS=musa PYTHONPATH=src python tools/bench_stability.py
    ... --rounds 8 --burn 5 --interleave        # test the two remedies
    ... --json before.json                      # then diff against after.json

Measurement only. Registers nothing, changes no shipped file.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time

import torch
import triton

import flaggems_vllm

DEV = flaggems_vllm.device

try:
    import vllm._custom_ops  # noqa: F401

    HAS_VLLM = hasattr(torch.ops._C, "top_k_per_row_prefill")
except (ImportError, AttributeError, RuntimeError):
    HAS_VLLM = False

# The benchmark's own shapes: num_rows, vocab, top_k, stride0, stride1
SHAPES = [
    (64, 129280, 1024, 129280, 1),
    (4, 8193, 512, 8456, 1),
    (16383, 4095, 512, 4352, 1),
    (4, 16385, 512, 16648, 1),
    (12961, 4100, 512, 4360, 1),
    (16380, 5115, 512, 5376, 1),
    (4100, 1025, 512, 1288, 1),
]


def clock_state():
    """Whatever this vendor will tell us about clocks and power.

    Printed, not parsed: the point is a record attached to the numbers, so a
    later run can be compared against the state this one was taken in.
    """
    for cmd in (["mthreads-gmi"], ["musa-smi"], ["mthreads-gmi", "-q"],
                ["nvidia-smi"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=10)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0 and out.stdout.strip():
            keep = [ln for ln in out.stdout.splitlines()
                    if any(k in ln for k in ("MHz", "Clock", "clock", "Power",
                                             "Temp", "%", "W /", "MiB"))]
            return cmd[0], "\n".join(f"      {ln.strip()}" for ln in keep[:8])
    return None, None


def make_inputs(num_rows, vocab, top_k, stride0, stride1):
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


def bench(fn, ms):
    return triton.testing.do_bench(fn, warmup=10, rep=ms,
                                   return_mode="median") * 1000


def burn(seconds):
    """Hold the device busy so clocks settle before anything is timed.

    A cold device boosts, and the first shape of a run then reads faster than
    the same shape later -- which shows up as between-round drift rather than
    as the warmup it is.
    """
    a = torch.randn((4096, 4096), device=DEV, dtype=torch.float32)
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(20):
            a = torch.nn.functional.relu(a * 1.0000001)
    flaggems_vllm.runtime.torch_device_fn.synchronize()


def stats(xs):
    med = statistics.median(xs)
    rng = (max(xs) - min(xs)) / med * 100 if med else 0.0
    cv = (statistics.stdev(xs) / statistics.mean(xs) * 100
          if len(xs) > 1 and statistics.mean(xs) else 0.0)
    return med, rng, cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--ms", type=int, default=100, help="do_bench rep, ms")
    ap.add_argument("--burn", type=float, default=0.0,
                    help="seconds of device warmup before measuring")
    ap.add_argument("--interleave", action="store_true",
                    help="alternate baseline and gems in short bursts, so "
                         "drift lands on both and partly cancels in the ratio")
    ap.add_argument("--bursts", type=int, default=6)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    print("=" * 92)
    print("  benchmark 稳定性：同一份代码跑多轮，量出每个形状的噪声底")
    print("=" * 92)
    # WHICH device, and is anything else on it? mthreads-gmi shows no clock
    # fields at all on this part, so frequency can be neither locked nor even
    # recorded -- but it does show per-device memory, and a device with another
    # process resident is a far more likely source of drift for a 30 us kernel
    # than anything this tool can control.
    try:
        dfn = flaggems_vllm.runtime.torch_device_fn
        cur = dfn.current_device()
        print(f"  当前设备: {cur}  ({DEV})")
        print("  ↑ 对照下表该设备的显存占用：不是 ~20MiB 就说明有别的进程在上面，")
        print("    小形状的离散度多半来自它。空闲卡可用 MUSA_VISIBLE_DEVICES=<n> 选。")
    except Exception as e:  # noqa: BLE001
        print(f"  当前设备: 读取失败 {type(e).__name__}: {e}")
    tool, state = clock_state()
    if state:
        print(f"  设备状态（{tool}）:\n{state}")
    else:
        print("  设备状态: 没有可用的 gmi/smi 工具 —— 无法记录时钟，"
              "跨轮漂移就只能观测不能解释")
    print(f"  vLLM 基线: {'有' if HAS_VLLM else '无（只报我方内核的离散度）'}")
    print(f"  轮数={args.rounds}  rep={args.ms}ms  "
          f"计时={'交错' if args.interleave else '分段'}  burn={args.burn}s\n")

    if args.burn:
        burn(args.burn)

    inputs = {s: make_inputs(*s) for s in SHAPES}
    raw = {s: {"gems": [], "torch": []} for s in SHAPES}

    for r in range(args.rounds):
        for s in SHAPES:
            num_rows, vocab, top_k, s0, s1 = s
            logits, starts, ends, idx = inputs[s]

            def g():
                flaggems_vllm.top_k_per_row_prefill(
                    logits, starts, ends, idx, num_rows, s0, s1, top_k)

            vidx = torch.empty_like(idx)

            def b():
                torch.ops._C.top_k_per_row_prefill(
                    logits, starts, ends, vidx, num_rows, s0, s1, top_k)

            if args.interleave and HAS_VLLM:
                per = max(1, args.ms // args.bursts)
                gs, bs = [], []
                for _ in range(args.bursts):
                    gs.append(bench(g, per))
                    bs.append(bench(b, per))
                tg, tb = statistics.median(gs), statistics.median(bs)
            else:
                tg = bench(g, args.ms)
                tb = bench(b, args.ms) if HAS_VLLM else float("nan")
            raw[s]["gems"].append(tg)
            raw[s]["torch"].append(tb)
        print(f"  第 {r + 1}/{args.rounds} 轮完成", flush=True)

    print()
    hdr = (f"  {'形状':<20}{'耗时µs':>9}{'我方极差':>10}{'基线极差':>10}"
           f"{'speedup中位':>12}{'极差':>8}{'CV':>7}   可信?")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    out = {}
    for s in SHAPES:
        num_rows, vocab, top_k = s[0], s[1], s[2]
        g = raw[s]["gems"]
        b = raw[s]["torch"]
        gmed, grng, _ = stats(g)
        if HAS_VLLM:
            bmed, brng, _ = stats(b)
            sp = [bb / gg for bb, gg in zip(b, g)]
            smed, srng, scv = stats(sp)
        else:
            bmed = brng = float("nan")
            smed = srng = scv = float("nan")
        verdict = ("可信" if srng < 2 else
                   ("勉强" if srng < 5 else "不可引用"))
        print(f"  {f'({num_rows},{vocab})':<20}{gmed:>9.1f}{grng:>9.1f}%"
              f"{brng:>9.1f}%{smed:>12.3f}{srng:>7.1f}%{scv:>6.1f}%   {verdict}")
        out[f"{num_rows}x{vocab}x{top_k}"] = {
            "gems_us": g, "torch_us": b,
            "speedup_median": smed, "speedup_range_pct": srng,
        }

    print("\n  读法")
    print("    「我方极差」远大于「基线极差」  => 抖的是我们的内核，不是除法放大")
    print("    speedup 极差 < 2%              => 这个形状的数字可以直接引用")
    print("    speedup 极差 > 5%              => 只能报区间，不能报单值；任何小于")
    print("      该极差的「收益」都不是收益")
    print("\n  用法")
    print("    换一张空闲卡再跑一次（MUSA_VISIBLE_DEVICES=<n>），比较极差 ——")
    print("      共享设备对 ms 级内核影响不大，对 30µs 的内核可能就是主因")
    print("    --interleave 与默认各跑一次，比较极差，就知道分段计时占多少")
    print("    --burn 5 与不 burn 各跑一次，看第 1 轮是否系统性偏快")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rounds": args.rounds, "ms": args.ms,
                       "interleave": args.interleave, "burn": args.burn,
                       "shapes": out}, f, indent=2)
        print(f"\n  原始数据已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
