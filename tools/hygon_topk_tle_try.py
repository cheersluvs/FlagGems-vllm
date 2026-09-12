"""Hygon: is there a vendor kernel anywhere, and does the TLE path just work?

The survey said TLE is fully present on hcu -- all four bindings, including the
two metax lacks (create_exclusive_cumsum, get_memdesc_type) -- and that
tle.gpu.alloc lowers correctly. HAS_TLE is nevertheless False because the hygon
VendorDescriptor never declared tle_enabled. So the question is whether the
generic TLE path runs as-is here, with none of the MetaX shims.

Part 1 hunts for a vendor implementation under every module that could hold
one (the survey only checked torch.ops._C, and vllm._custom_ops has 148 names
plus a vllm_hcu platform plugin).

Part 2 runs the operator with FLAGGEMS_FORCE_TLE=0 and =1 in separate
processes -- HAS_TLE is fixed at import -- checks both against torch.topk and
times them, so the table says both "is it correct" and "is it worth it".

    tools/vendor_probe.sh tools/hygon_topk_tle_try.py hygon_topk_tle
"""

import os
import subprocess
import sys
from importlib import import_module

SHAPES = (
    ("decode", 1, 262144, 512),
    ("decode", 8, 262144, 512),
    ("decode", 64, 262144, 512),
    ("prefill", 64, 129280, 1024),
    ("prefill", 4100, 1025, 512),
)

if len(sys.argv) == 1:
    print("=== part 1: any vendor kernel for this op?")
    import torch  # noqa: F401

    for mod in ("vllm._custom_ops", "vllm_hcu", "lightop", "hgai"):
        try:
            m = import_module(mod)
        except Exception as e:  # noqa: BLE001
            print(f"  {mod:<20} absent ({type(e).__name__})")
            continue
        hits = [n for n in dir(m) if "topk" in n.lower() or "top_k" in n.lower()]
        print(f"  {mod:<20} {len(dir(m))} names, topk-ish: {hits or 'none'}")
    for ns in ("_C", "_C_utils", "vllm_hcu", "hcu"):
        try:
            o = getattr(torch.ops, ns)
            hits = [n for n in dir(o) if "topk" in n.lower() or "top_k" in n.lower()]
            print(f"  torch.ops.{ns:<12} topk-ish: {hits or 'none'}")
        except Exception:  # noqa: BLE001
            print(f"  torch.ops.{ns:<12} absent")

    print("\n=== part 2: generic path vs TLE path (own process each)")
    rows = {}
    for tle in ("0", "1"):
        env = dict(os.environ, FLAGGEMS_FORCE_TLE=tle)
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "child"],
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
        for ln in (r.stdout + r.stderr).splitlines():
            if ln.startswith("ROW "):
                _, key, verdict, ms = ln.split("|")
                rows.setdefault(key, {})[tle] = (verdict, float(ms))
            elif ln.startswith("HAS_TLE"):
                print(f"  FLAGGEMS_FORCE_TLE={tle}: {ln}")
            elif "Traceback" in ln or "Error:" in ln:
                print(f"  FLAGGEMS_FORCE_TLE={tle}: {ln.strip()[:140]}")
    print(f"\n  {'shape':<34} {'generic':>18} {'TLE':>18} {'speedup':>8}")
    for key, got in rows.items():
        g, t = got.get("0"), got.get("1")
        gs = f"{g[0]} {g[1]:.3f}ms" if g else "-"
        ts = f"{t[0]} {t[1]:.3f}ms" if t else "MISSING"
        sp = f"{g[1] / t[1]:.2f}x" if g and t and t[1] > 0 else "-"
        print(f"  {key:<34} {gs:>18} {ts:>18} {sp:>8}")
    print("\n  TLE needs both columns CORRECT before any speedup counts.")
    sys.exit(0)

# ---------------------------------------------------------------- child
import torch  # noqa: E402

import flaggems_vllm  # noqa: E402

dec = import_module("flaggems_vllm.ops.top_k_per_row_decode")
pre = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
print(f"HAS_TLE decode={dec.HAS_TLE} prefill={pre.HAS_TLE}")


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


torch.manual_seed(0)
for op, B, V, K in SHAPES:
    key = f"{op} B={B} V={V} K={K}"
    try:
        logits = torch.randn(B, V, dtype=torch.float32, device="cuda")
        idx = torch.zeros(B, K, dtype=torch.int32, device="cuda")
        s0, s1 = logits.stride(0), logits.stride(1)
        if op == "decode":
            lens = torch.full((B,), V, dtype=torch.int32, device="cuda")
            call = lambda: flaggems_vllm.top_k_per_row_decode(  # noqa: E731
                logits, 1, lens, idx, B, s0, s1, K
            )
        else:
            st = torch.zeros(B, dtype=torch.int32, device="cuda")
            en = torch.full((B,), V, dtype=torch.int32, device="cuda")
            call = lambda: flaggems_vllm.top_k_per_row_prefill(  # noqa: E731
                logits, st, en, idx, B, s0, s1, K
            )
        call()
        torch.cuda.synchronize()
        want = torch.topk(logits, K, dim=1).values.sort(dim=1).values
        got = logits.gather(1, idx.long().clamp(0, V - 1)).sort(dim=1).values
        oob = int(((idx < 0) | (idx >= V)).sum())
        verdict = "OK" if (torch.allclose(got, want) and oob == 0) else "WRONG"
        print(f"ROW |{key}|{verdict}|{timed(call):.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"ROW |{key}|FAIL:{type(e).__name__}|0")
        print(f"Error: {str(e).strip().splitlines()[-1][:160]}")
