"""Where does prefill's best launch geometry switch from wide to narrow?

tools/hygon_prefill_launch_sweep.py (full operator, checked against
torch.topk) found two regimes on BW1000:

    config (threads/program)   16383x4095  4100x1025   64x129280
    B=512  w=8   (512, today)       0.426      0.417       0.432
    B=256  w=2   (128)              0.831      0.744       0.289
    B=1024 w=16  (1024)             0.182      0.172       0.449

Thousands of rows want narrow programs (1.8-1.95x); 64 rows wants wide ones.
The variable is occupancy -- rows against this card's 80 SMs -- not elements
per lane. This places the switch: rows from 16 to 4160 at two row lengths,
each config in its own process (they compile different kernels), every point
checked against torch.topk. num_warps=1 is excluded: it returned WRONG answers.

Ratios are wall time against vLLM's kernel, with at least 64 rows so that
host overhead is negligible (the 4-row shapes in the first sweep were not).

    tools/vendor_probe.sh tools/hygon_prefill_rows_sweep.py hygon_prefill_rows
"""

import os
import subprocess
import sys

ROWS = (64, 160, 320, 640, 1280, 2560, 4160)
LENGTHS = ((4096, 512), (129280, 1024))
CONFIGS = ((512, 8), (512, 4), (256, 4), (256, 2), (1024, 16))

if len(sys.argv) == 1:
    res = {}
    for block, warps in CONFIGS:
        env = dict(os.environ, SWEEP_BLOCK=str(block), SWEEP_WARPS=str(warps))
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "child"],
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
        )
        got = 0
        for ln in (r.stdout + r.stderr).splitlines():
            if ln.startswith("ROW|"):
                _, v, n, ours, base, ok = ln.split("|")
                res[(block, warps, int(v), int(n))] = (float(base) / float(ours), ok)
                got += 1
        if got < len(ROWS) * len(LENGTHS):
            tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or ["?"]
            print(f"  B={block} w={warps}: only {got} points -- {tail[0][:140]}")

    for v, k in LENGTHS:
        print(f"\n=== row length {v}, top_k {k}: ratio vs vLLM (* best, ! WRONG)")
        print(f"  {'B':>5} {'w':>3} {'thr':>5}  " + " ".join(f"{n:>7}" for n in ROWS))
        best = {}
        for n in ROWS:
            c = [
                (res[(b, w, v, n)][0], b, w)
                for b, w in CONFIGS
                if (b, w, v, n) in res and res[(b, w, v, n)][1] == "OK"
            ]
            if c:
                best[n] = max(c)[1:]
        for b, w in CONFIGS:
            cells = []
            for n in ROWS:
                if (b, w, v, n) not in res:
                    cells.append(f"{'-':>7}")
                    continue
                ratio, ok = res[(b, w, v, n)]
                mark = "!" if ok != "OK" else ("*" if best.get(n) == (b, w) else " ")
                cells.append(f"{ratio:>6.3f}{mark}")
            print(f"  {b:>5} {w:>3} {w * 64:>5}  " + " ".join(cells))
        print("  rows/SM: " + " ".join(f"{n / 80:>7.1f}" for n in ROWS))
    sys.exit(0)

# ---------------------------------------------------------------- child
from importlib import import_module  # noqa: E402

import torch  # noqa: E402

import flaggems_vllm  # noqa: E402

BLOCK, WARPS = int(os.environ["SWEEP_BLOCK"]), int(os.environ["SWEEP_WARPS"])
mods = [import_module("flaggems_vllm.ops.top_k_per_row_prefill")]
try:
    mods.append(
        import_module(
            "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
        )._dense
    )
except Exception:  # noqa: BLE001
    pass
for m in mods:
    m.NUM_THREADS_PER_BLOCK = BLOCK
    m._num_warps = lambda block_size, w=WARPS: w

import vllm._custom_ops  # noqa: E402,F401


def timed(fn, iters=8, warmup=3):
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
for v, k in LENGTHS:
    for n in ROWS:
        if n * v > 600_000_000:  # keep the input under ~2.4 GB
            continue
        logits = torch.randn(n, v, dtype=torch.float32, device="cuda")
        st = torch.zeros(n, dtype=torch.int32, device="cuda")
        en = torch.full((n,), v, dtype=torch.int32, device="cuda")
        idx = torch.empty(n, k, dtype=torch.int32, device="cuda")
        idx2 = torch.empty(n, k, dtype=torch.int32, device="cuda")
        s0 = logits.stride(0)
        try:
            flaggems_vllm.top_k_per_row_prefill(logits, st, en, idx, n, s0, 1, k)
            torch.cuda.synchronize()
            want = torch.topk(logits, k, dim=1).values.sort(dim=1).values
            got = logits.gather(1, idx.long().clamp(0, v - 1)).sort(dim=1).values
            ok = "OK" if torch.allclose(got, want) else "WRONG"
            args = (logits, st, en)
            t_ours = timed(
                lambda a=args, o=idx: flaggems_vllm.top_k_per_row_prefill(
                    *a, o, n, s0, 1, k
                )
            )
            t_base = timed(
                lambda a=args, o=idx2: torch.ops._C.top_k_per_row_prefill(
                    *a, o, n, s0, 1, k
                )
            )
            print(f"ROW|{v}|{n}|{t_ours:.5f}|{t_base:.5f}|{ok}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL|{v}|{n}|{type(e).__name__}: {str(e)[:100]}", flush=True)
        del logits, idx, idx2
        torch.cuda.empty_cache()
