"""Sweep the TLE path's launch geometry on MetaX: BLOCK_SIZE and, for decode,
blocks per row.

Every TLE constant the generic ops use was set on NVIDIA (32-lane warps):
NUM_THREADS_PER_BLOCK=512 as the tile, MULTIPLE_BLOCKS_PER_ROW_CONFIG=10
blocks per decode row. The C550 has 64-lane warps, 104 SMs and a 512-thread
block limit, and nobody has measured what it wants. Both constants are module
globals read by the host dispatch on every call, so they are rebound here per
config -- no generic diff -- and each config compiles its own kernels.

Constraints the sweep respects:
  BLOCK_SIZE must divide 1024 (threshold_rounds = 1024 // BLOCK_SIZE), so
  128 / 256 / 512 / 1024. num_warps follows the generic rule
  (BLOCK_SIZE // 64, capped at 8 warps), so 1024 means 2 elements per thread
  -- the regime where a masked shared atomic was measured writing wrong byte
  offsets. EVERY config is therefore checked against torch.topk before it is
  timed, and a wrong or failing config is reported, never ranked.

Timing is the benchmark's kernel mode: triton.testing.do_bench, median. The
default config (512 / 10) is timed first and again last; if those two differ
by more than the gaps being ranked, the sweep measured the box.

    PY=/data/wuyuqing/workspace/mctle-v2/bin/python \
        tools/vendor_probe.sh tools/metax_tle_geometry_sweep.py metax_geom_sweep
    ... --op decode | --op prefill        one side only
    ... --blocks 256,512 --bpr 4,8,10     narrower grid
"""

import argparse
import math
import sys
from importlib import import_module

import torch
import triton

import flaggems_vllm

DEV = flaggems_vllm.device

DECODE_SHAPES = [1, 4, 8, 16, 56]  # rows; vocab 262144, k 512
PREFILL_SHAPES = [
    (4, 8193, 512, 8456),
    (4, 16385, 512, 16648),
    (64, 129280, 1024, 129280),
    (4100, 1025, 512, 1288),
]
PREFILL_ALL = PREFILL_SHAPES + [
    (16383, 4095, 512, 4352),
    (12961, 4100, 512, 4360),
    (16380, 5115, 512, 5376),
]
CONTROL = {"decode": 56, "prefill": (4100, 1025, 512, 1288)}

dec = import_module("flaggems_vllm.ops.top_k_per_row_decode")
pre = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
tle = import_module("flaggems_vllm.runtime.backend._metax.fused.top_k_per_row_tle")

try:
    import vllm._custom_ops  # noqa: F401  -- registers torch.ops._C

    HAS_VLLM = hasattr(torch.ops._C, "top_k_per_row_decode")
except Exception:  # noqa: BLE001
    HAS_VLLM = False


def decode_inputs(rows):
    torch.manual_seed(42)
    v, k = 262144, 512
    logits = torch.randn(rows, v, device=DEV, dtype=torch.float32)
    seq = torch.full((rows,), v, dtype=torch.int32, device=DEV)
    idx = torch.zeros((rows, k), dtype=torch.int32, device=DEV)
    return (logits, 1, seq, idx, rows, v, 1, k)


def prefill_inputs(shape):
    rows, v, k, s0 = shape
    torch.manual_seed(42)
    buf = torch.randn((rows - 1) * s0 + v, device=DEV, dtype=torch.float32)
    logits = torch.as_strided(buf, (rows, v), (s0, 1))
    starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    ends = torch.full((rows,), v, dtype=torch.int32, device=DEV)
    idx = torch.empty((rows, k), dtype=torch.int32, device=DEV)
    return (logits, starts, ends, idx, rows, s0, 1, k)


def correct(args):
    logits, idx, k = args[0], args[3], args[7]
    got = logits.gather(1, idx.long().clamp(0, logits.shape[1] - 1)).sort(dim=1).values
    want = torch.topk(logits, k, dim=1).values.sort(dim=1).values
    return bool(torch.equal(got, want))


def bench(fn):
    return triton.testing.do_bench(fn, warmup=100, rep=400, return_mode="median")


def apply(op, block, bpr):
    m = dec if op == "decode" else pre
    m.NUM_THREADS_PER_BLOCK = block
    if op == "decode":
        m.MULTIPLE_BLOCKS_PER_ROW_CONFIG = bpr


def run_one(op, shape, block, bpr):
    apply(op, block, bpr)
    args = decode_inputs(shape) if op == "decode" else prefill_inputs(shape)
    fn_op = (
        flaggems_vllm.top_k_per_row_decode
        if op == "decode"
        else flaggems_vllm.top_k_per_row_prefill
    )
    try:
        fn_op(*args)
        torch.cuda.synchronize()
        if not correct(args):
            return None, "WRONG"
        return bench(lambda: fn_op(*args)), "ok"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:60]}"


def vllm_ms(op, shape):
    if not HAS_VLLM:
        return None
    args = decode_inputs(shape) if op == "decode" else prefill_inputs(shape)
    f = (
        torch.ops._C.top_k_per_row_decode
        if op == "decode"
        else torch.ops._C.top_k_per_row_prefill
    )
    f(*args)
    torch.cuda.synchronize()
    return bench(lambda: f(*args))


def label(shape):
    return (
        f"{shape} rows"
        if isinstance(shape, int)
        else f"({shape[0]},{shape[1]}) k{shape[2]}"
    )


def sweep(op, blocks, bprs, rows=None, prefill_all=False):
    if op == "decode":
        shapes = rows or DECODE_SHAPES
    else:
        shapes = PREFILL_ALL if prefill_all else PREFILL_SHAPES
    configs = [(b, k) for b in blocks for k in (bprs if op == "decode" else [None])]
    default = (512, 10 if op == "decode" else None)
    base = {s: vllm_ms(op, s) for s in shapes}
    print(f"\n######## {op}  vLLM baseline: {'yes' if HAS_VLLM else 'NO (ms only)'}")
    first = {s: run_one(op, s, *default)[0] for s in shapes}
    res = {}
    for cfg in configs:
        res[cfg] = {s: run_one(op, s, *cfg) for s in shapes}
    last = {s: run_one(op, s, *default)[0] for s in shapes}
    apply(op, *default)

    def cname(c):
        return f"B{c[0]}" + (f"/bpr{c[1]}" if c[1] is not None else "")

    print("\n  drift check, default config timed first vs last (ms):")
    for s in shapes:
        a, b = first[s], last[s]
        d = f"{(b / a - 1) * 100:+.1f}%" if a and b else "n/a"
        print(
            f"    {label(s):<22} {a if a is None else round(a, 4)!s:>9} -> {b if b is None else round(b, 4)!s:>9}  {d}"
        )

    print("\n  per shape: ms (ratio vs vLLM) ; * = best correct config for the shape")
    for s in shapes:
        ok = {c: r[s][0] for c, r in res.items() if r[s][0] is not None}
        best = min(ok, key=ok.get) if ok else None
        dref = (
            min(v for v in (first[s], last[s]) if v) if (first[s] or last[s]) else None
        )
        print(
            f"\n  {label(s)}   vLLM {base[s] if base[s] is None else round(base[s], 4)} ms"
            f"   default {dref if dref is None else round(dref, 4)} ms"
        )
        for c in configs:
            ms, st = res[c][s]
            if ms is None:
                print(f"    {cname(c):<12} {st}")
                continue
            ratio = f"{base[s] / ms:.3f}" if base[s] else "-"
            vs = f"{dref / ms:.2f}x vs default" if dref else ""
            print(
                f"    {cname(c):<12} {ms:>8.4f} ms  ratio {ratio:>6}  {vs}{'  *' if c == best else ''}"
            )

    if HAS_VLLM:
        print(
            "\n  geomean ratio vs vLLM over the swept shapes (control excluded / included):"
        )
        rows = []
        for c in configs:
            if any(res[c][s][0] is None for s in shapes):
                rows.append((c, None, None))
                continue
            rs = {s: base[s] / res[c][s][0] for s in shapes}
            g_all = math.exp(sum(map(math.log, rs.values())) / len(rs))
            no_ctl = [v for s, v in rs.items() if s != CONTROL[op]] or list(rs.values())
            g_low = math.exp(sum(map(math.log, no_ctl)) / len(no_ctl))
            rows.append((c, g_low, g_all))
        for c, g_low, g_all in sorted(rows, key=lambda r: -(r[1] or 0)):
            tag = " <- default" if c == default else ""
            if g_low is None:
                print(f"    {cname(c):<12} (a shape failed or was wrong)")
            else:
                print(
                    f"    {cname(c):<12} low-row {g_low:.3f}   with control {g_all:.3f}{tag}"
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["decode", "prefill", "both"], default="both")
    ap.add_argument("--blocks", default="128,256,512,1024")
    ap.add_argument("--bpr", default="1,2,4,8,10,16,32")
    ap.add_argument("--rows", default="", help="decode rows, e.g. 1,4,8,16,24,56,496")
    ap.add_argument("--prefill-all", action="store_true", help="all 7 prefill shapes")
    a = ap.parse_args()
    blocks = [int(x) for x in a.blocks.split(",")]
    bprs = [int(x) for x in a.bpr.split(",")]

    warm = decode_inputs(4)
    flaggems_vllm.top_k_per_row_decode(*warm)  # runs the TLE self-test
    torch.cuda.synchronize()
    st = tle.status()
    print(f"TLE status: {st}")
    if not st.get("on"):
        print(
            "!! TLE path is off -- sweep would measure the non-TLE kernel. Use the mctle venv."
        )
        return 3
    print(
        f"warp geometry (generic _launch_geometry): {dec._launch_geometry()}  "
        f"merge tile: {dec.NUM_THREADS_PER_BLOCK_MERGE}"
    )
    for op in (["decode", "prefill"] if a.op == "both" else [a.op]):
        rows = [int(x) for x in a.rows.split(",")] if a.rows else None
        sweep(op, blocks, bprs, rows, a.prefill_all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
