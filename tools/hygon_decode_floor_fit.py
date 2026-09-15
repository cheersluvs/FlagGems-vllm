"""Split one-row decode's 0.325 ms into a fixed part and a per-element part.

tools/decode_floor_profile.py on BW1000: gems decode is ONE kernel of 0.325 ms
device time at 1 row and at 8 rows alike (each row runs on its own SM), while
vLLM's two CUDA kernels total 0.071 ms. Combined with the split ceiling
(8 chunks of 32768: 0.159 ms; a 4096-candidate merge: 0.157 ms), a per-program
model F + e * elements gives F ~ 0.14 ms and e ~ 0.72 ns -- half fixed, half
data. Nothing in the algorithm's fixed work (clear, threshold scan, radix final)
comes near 0.14 ms, so the suspect is the per-launch device floor of a
512-thread program. That is a guess until measured, so this measures:

  1. launch floor: a trivial kernel, 1 program, over BLOCK x num_warps
  2. the operator at 1 row over vocab 4096..262144 -> fit F and e (and vLLM's)
  3. the operator at 1 row, vocab 262144, over BLOCK x num_warps

Every operator point is checked against torch.topk. Each geometry runs in its
own process, since BLOCK_SIZE is a constexpr and num_warps a launch option.

    tools/vendor_probe.sh tools/hygon_decode_floor_fit.py hygon_decode_floor_fit
"""

import os
import subprocess
import sys

VOCABS = (4096, 16384, 65536, 262144)
K = 512
GEOMS = ((512, 8), (512, 4), (256, 4), (256, 2), (128, 2), (1024, 16), (512, 2))

if len(sys.argv) == 1:
    lines = []
    for block, warps in GEOMS:
        env = dict(os.environ, FIT_BLOCK=str(block), FIT_WARPS=str(warps))
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "child"],
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
        got = [
            ln
            for ln in (r.stdout + r.stderr).splitlines()
            if ln.startswith(("NOOP|", "OP|", "BASE|", "FAIL|"))
        ]
        if not got:
            tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or ["?"]
            got = [f"FAIL|{block}|{warps}|child: {tail[0][:120]}"]
        lines += got

    print("=== 1. launch floor of a trivial kernel, 1 program (us)")
    for ln in lines:
        if ln.startswith("NOOP|"):
            _, b, w, us = ln.split("|")
            print(f"  B={b:>5} w={w:>3} threads={int(w) * 64:>5}   {float(us):8.1f}")

    print("\n=== 2/3. operator at 1 row (ms), and fitted fixed part + slope")
    base = {
        int(v): float(ms)
        for _, v, ms in (ln.split("|") for ln in lines if ln.startswith("BASE|"))
    }
    if base:
        xs = sorted(base)
        n = len(xs)
        mx = sum(xs) / n
        my = sum(base[x] for x in xs) / n
        e = sum((x - mx) * (base[x] - my) for x in xs) / sum((x - mx) ** 2 for x in xs)
        f = my - e * mx
        print(
            "  vLLM:        "
            + "  ".join(f"{v}:{base[v]:.4f}" for v in xs)
            + f"   -> fixed {f * 1000:6.1f} us, {e * 1e6:.3f} ns/element"
        )
    by_geo = {}
    for ln in lines:
        if ln.startswith("OP|"):
            _, b, w, v, ms, ok = ln.split("|")
            by_geo.setdefault((int(b), int(w)), {})[int(v)] = (float(ms), ok)
    for (b, w), pts in by_geo.items():
        xs = sorted(pts)
        cells = "  ".join(
            f"{v}:{pts[v][0]:.4f}{'' if pts[v][1] == 'OK' else '!'}" for v in xs
        )
        if len(xs) >= 2:
            n = len(xs)
            mx = sum(xs) / n
            my = sum(pts[x][0] for x in xs) / n
            e = sum((x - mx) * (pts[x][0] - my) for x in xs) / sum(
                (x - mx) ** 2 for x in xs
            )
            f = my - e * mx
            fit = f"-> fixed {f * 1000:6.1f} us, {e * 1e6:.3f} ns/element"
        else:
            fit = ""
        print(f"  B={b:>4} w={w:>2}:  {cells}   {fit}")
    for ln in lines:
        if ln.startswith("FAIL|"):
            print(f"  {ln}")
    print("\n  ! = WRONG answer. Today's geometry is B=512 w=8. If the operator's")
    print("  fixed part tracks the trivial kernel's floor across geometries, the")
    print("  0.14 ms is launch cost; if it stays put, it is the kernel's own work.")
    sys.exit(0)

# ---------------------------------------------------------------- child
from importlib import import_module  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import flaggems_vllm  # noqa: E402,F401  (runtime init)

BLOCK, WARPS = int(os.environ["FIT_BLOCK"]), int(os.environ["FIT_WARPS"])
dec = import_module("flaggems_vllm.ops.top_k_per_row_decode")
dec.NUM_THREADS_PER_BLOCK = BLOCK
dec._num_warps = lambda block_size, w=WARPS: w


@triton.jit
def k_noop(x_ptr, out_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    tl.store(out_ptr + lane, tl.load(x_ptr + lane))


def timed(fn, iters=30, warmup=5):
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


x = torch.randn(BLOCK, device="cuda")
y = torch.empty_like(x)
us = timed(lambda: k_noop[(1,)](x, y, BLOCK=BLOCK, num_warps=WARPS)) * 1000
print(f"NOOP|{BLOCK}|{WARPS}|{us:.2f}", flush=True)

try:
    import vllm._custom_ops  # noqa: F401

    have_vllm = hasattr(torch.ops._C, "top_k_per_row_decode")
except Exception:  # noqa: BLE001
    have_vllm = False

torch.manual_seed(0)
for v in VOCABS:
    logits = torch.randn(1, v, dtype=torch.float32, device="cuda")
    lens = torch.full((1,), v, dtype=torch.int32, device="cuda")
    idx = torch.empty(1, K, dtype=torch.int32, device="cuda")
    try:
        dec.top_k_per_row_decode(logits, 1, lens, idx, 1, v, 1, K)
        torch.cuda.synchronize()
        want = torch.topk(logits, K, dim=1).values.sort(dim=1).values
        got = logits.gather(1, idx.long().clamp(0, v - 1)).sort(dim=1).values
        ok = "OK" if torch.allclose(got, want) else "WRONG"
        ms = timed(
            lambda a=logits, s=lens, o=idx, n=v: dec.top_k_per_row_decode(
                a, 1, s, o, 1, n, 1, K
            )
        )
        print(f"OP|{BLOCK}|{WARPS}|{v}|{ms:.5f}|{ok}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL|{BLOCK}|{WARPS}|{v}|{type(e).__name__}: {str(e)[:90]}", flush=True)
    if have_vllm and BLOCK == 512 and WARPS == 8:
        idx2 = torch.empty(1, K, dtype=torch.int32, device="cuda")
        ms = timed(
            lambda a=logits, s=lens, o=idx2, n=v: torch.ops._C.top_k_per_row_decode(
                a, 1, s, o, 1, n, 1, K
            )
        )
        print(f"BASE|{v}|{ms:.5f}", flush=True)
