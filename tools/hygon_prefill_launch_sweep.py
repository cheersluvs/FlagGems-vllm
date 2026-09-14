"""Sweep prefill's launch geometry on Hygon: BLOCK_SIZE x num_warps, 2-D.

tools/hygon_histogram_cost.py split the vocab term (~1.5 ns/element): reading
the row is ~0.46 ns/element per pass and the operator makes at least two
passes, so reads are ~60% of it; the global atomic histogram is ~0.60. A read
at 0.46 ns/element is ~8.7 GB/s per program, about half of this card's
~16.75 GB/s per SM ceiling measured for fused_v4.

On this same card fused_v4's bandwidth peaked at 16 elements per lane, and the
operator today runs [512, 4] tiles (2048 elements) on 8 warps x 64 lanes --
4 elements per lane. That recorded work also carries a warning: a
one-variable sweep of launch parameters misled twice, because program width
and per-lane width both matter. So both axes, together.

Both knobs are host-side globals read at the launch site of the generic
module (NUM_THREADS_PER_BLOCK becomes the BLOCK_SIZE constexpr, _num_warps the
launch option), so each config simply rebinds them -- on the generic module
AND on the Hygon override's dense copy, which dispatches to either -- in its
own process. Every shape is checked against torch.topk before it is timed.

    tools/vendor_probe.sh tools/hygon_prefill_launch_sweep.py hygon_prefill_launch
"""

import os
import subprocess
import sys

SHAPES = (
    (64, 129280, 1024),
    (4, 8193, 512),
    (16383, 4095, 512),
    (4, 16385, 512),
    (12961, 4100, 512),
    (16380, 5115, 512),
    (4100, 1025, 512),
)
CONFIGS = [(b, w) for b in (256, 512, 1024) for w in (1, 2, 4, 8, 16) if w * 64 <= 1024]

if len(sys.argv) == 1:
    results = {}
    for block, warps in CONFIGS:
        env = dict(os.environ, SWEEP_BLOCK=str(block), SWEEP_WARPS=str(warps))
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "child"],
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
        for ln in (r.stdout + r.stderr).splitlines():
            if ln.startswith("ROW|"):
                _, shp, ms, ok = ln.split("|")
                results[(block, warps, shp)] = (float(ms), ok)
            elif ln.startswith("BASE|"):
                _, shp, ms = ln.split("|")
                results.setdefault(("vllm", shp), float(ms))
        if r.returncode != 0 and not any(k[:2] == (block, warps) for k in results):
            tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or ["?"]
            print(f"  config B={block} w={warps}: FAILED -- {tail[0][:120]}")

    shapes = [f"{n}x{v}/k{k}" for n, v, k in SHAPES]
    print(
        "ratio vs vLLM per shape (kernel wall time); * = best config for that shape;"
        " ! = WRONG answer\n"
    )
    head = f"  {'B':>5} {'w':>3} {'e/lane':>6}  " + " ".join(
        f"{s[:14]:>14}" for s in shapes
    )
    print(head + f" {'geomean':>8}")
    best = {}
    for s in shapes:
        cands = [
            (results[("vllm", s)] / results[(b, w, s)][0], b, w)
            for b, w in CONFIGS
            if (b, w, s) in results
            and results[(b, w, s)][1] == "OK"
            and ("vllm", s) in results
        ]
        if cands:
            best[s] = max(cands)[1:]
    import math

    for b, w in CONFIGS:
        cells, logs = [], []
        for s in shapes:
            if (b, w, s) not in results or ("vllm", s) not in results:
                cells.append(f"{'-':>14}")
                continue
            ms, ok = results[(b, w, s)]
            ratio = results[("vllm", s)] / ms
            logs.append(math.log(ratio))
            mark = "!" if ok != "OK" else ("*" if best.get(s) == (b, w) else " ")
            cells.append(f"{ratio:>13.3f}{mark}")
        g = math.exp(sum(logs) / len(logs)) if logs else float("nan")
        e_lane = b * 4 / (w * 64)
        print(f"  {b:>5} {w:>3} {e_lane:>6g}  " + " ".join(cells) + f" {g:>8.3f}")
    print("\n  today: B=512 w=8 (4 elements per lane). e/lane counts the [B, 4] tile.")
    sys.exit(0)

# ---------------------------------------------------------------- child
from importlib import import_module  # noqa: E402

import torch  # noqa: E402

import flaggems_vllm  # noqa: E402

BLOCK = int(os.environ["SWEEP_BLOCK"])
WARPS = int(os.environ["SWEEP_WARPS"])
mods = [import_module("flaggems_vllm.ops.top_k_per_row_prefill")]
try:
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    mods.append(ov._dense)
except Exception:  # noqa: BLE001
    pass
for m in mods:
    m.NUM_THREADS_PER_BLOCK = BLOCK
    m._num_warps = lambda block_size, w=WARPS: w

have_vllm = False
try:
    import vllm._custom_ops  # noqa: F401

    have_vllm = hasattr(torch.ops._C, "top_k_per_row_prefill")
except Exception:  # noqa: BLE001
    pass


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


torch.manual_seed(42)
for n, v, k in SHAPES:
    tag = f"{n}x{v}/k{k}"
    logits = torch.randn(n, v, dtype=torch.float32, device="cuda")
    starts = torch.zeros(n, dtype=torch.int32, device="cuda")
    ends = torch.full((n,), v, dtype=torch.int32, device="cuda")
    idx = torch.empty(n, k, dtype=torch.int32, device="cuda")
    s0, s1 = logits.stride(0), logits.stride(1)

    def call():
        flaggems_vllm.top_k_per_row_prefill(logits, starts, ends, idx, n, s0, s1, k)

    try:
        call()
        torch.cuda.synchronize()
        want = torch.topk(logits, k, dim=1).values.sort(dim=1).values
        got = logits.gather(1, idx.long().clamp(0, v - 1)).sort(dim=1).values
        ok = "OK" if torch.allclose(got, want) else "WRONG"
        print(f"ROW|{tag}|{timed(call):.5f}|{ok}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"ROW|{tag}|1e9|FAIL:{type(e).__name__}", flush=True)
    if have_vllm:  # every config, so one failed config cannot lose the baseline
        idx2 = torch.empty(n, k, dtype=torch.int32, device="cuda")
        t = timed(
            lambda: torch.ops._C.top_k_per_row_prefill(
                logits, starts, ends, idx2, n, s0, s1, k
            )
        )
        print(f"BASE|{tag}|{t:.5f}", flush=True)
