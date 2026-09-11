"""Force top_k_per_row decode AND prefill onto the TLE path on MetaX; print the
ROOT cause of any failure, and check the answer when it runs.

Round 1 found the root cause pytest hid: tle.cumsum calls
builder.create_exclusive_cumsum, which metax's libtriton does not have. So
this round swaps a shim in for `tle` in the two generic modules -- a real
ModuleType (Triton only lets @jit code reach module-typed globals), whose
`gpu` is the real tle.gpu and whose `cumsum` is plain tl:

    exclusive prefix = tl.cumsum(x) - x,   total = tl.sum(x)

the same arithmetic the non-TLE branch already does. The shim lives in THIS
tool only; nothing in src/ changes until the path is proven.

Expected next wall: smem tensor loads at < 4 elements per thread hit the
plugin's vec assert (LoadStoreOpToLLVM.cpp:427); BLOCK_SIZE=512 on 8 warps is
exactly 1. Each shape runs in its own process so an abort does not hide the
rest.

    PYTHONPATH=src:$PYTHONPATH /data/wuyuqing/workspace/mctle-test/bin/python \
        tools/metax_tle_force_decode.py            # shim on
    SHIM=0 ...                                     # round-1 behaviour
"""

import os
import subprocess
import sys
import types
from importlib import import_module

CASES = (
    ("decode", 1, 262144, 512),
    ("decode", 64, 129280, 512),
    ("decode", 8, 32768, 2048),
    ("prefill", 4, 32768, 2048),
    ("prefill", 64, 131072, 2048),
)

if len(sys.argv) == 1:
    print(f"SHIM={os.environ.get('SHIM', '1')}  (each case in its own process)")
    for c in CASES:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), *map(str, c)],
                           capture_output=True, text=True, timeout=900)
        out = (r.stdout + r.stderr).rstrip().splitlines()
        if not any(l.startswith("RESULT") for l in out):
            ab = next((l for l in out if "Assertion" in l or "assert" in l.lower()), None)
            out = [f"RESULT {' '.join(map(str, c))}: CRASHED (exit {r.returncode})"
                   + (f" -- {ab.strip()[:170]}" if ab else "")] + out[-6:]
        print("\n".join(out))
    sys.exit(0)

os.environ["FLAGGEMS_FORCE_TLE"] = "1"

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import flaggems_vllm  # noqa: E402

dec = import_module("flaggems_vllm.ops.top_k_per_row_decode")
pre = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
if not (dec.HAS_TLE and pre.HAS_TLE):
    print(f"!! TLE did not switch on (decode={dec.HAS_TLE} prefill={pre.HAS_TLE})")
    sys.exit(3)


@triton.jit
def _cumsum_shim(x, axis: tl.constexpr = 0, reverse: tl.constexpr = False):
    tl.static_assert(not reverse, "cumsum shim: reverse=True not implemented")
    return tl.cumsum(x, axis=axis) - x, tl.sum(x, axis=axis)


if os.environ.get("SHIM", "1") != "0":
    shim = types.ModuleType("tle_metax_shim")
    shim.gpu = dec.tle.gpu
    shim.cumsum = _cumsum_shim
    dec.tle = shim
    pre.tle = shim


def chain(e):
    seen, links = set(), []
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        links.append(e)
        e = e.__cause__ or e.__context__
    return links


op, B, V, K = sys.argv[1], *map(int, sys.argv[2:5])
tag = f"{op} B={B} V={V} K={K}"
torch.manual_seed(0)
logits = torch.randn(B, V, dtype=torch.float32, device="cuda")
idx = torch.zeros(B, K, dtype=torch.int32, device="cuda")
s0, s1 = logits.stride(0), logits.stride(1)
try:
    if op == "decode":
        seq_lens = torch.full((B,), V, dtype=torch.int32, device="cuda")
        flaggems_vllm.top_k_per_row_decode(logits, 1, seq_lens, idx, B, s0, s1, K)
    else:
        starts = torch.zeros(B, dtype=torch.int32, device="cuda")
        ends = torch.full((B,), V, dtype=torch.int32, device="cuda")
        flaggems_vllm.top_k_per_row_prefill(logits, starts, ends, idx, B, s0, s1, K)
    torch.cuda.synchronize()
except Exception as e:  # noqa: BLE001
    links = chain(e)
    inner = links[-1]
    print(f"RESULT {tag}: FAILED -- {type(inner).__name__}: {str(inner).strip()[:200]}")
    for i, x in enumerate(links):
        print(f"  --- [{i}] {type(x).__name__}")
        for ln in str(x).strip().splitlines()[-10:]:
            print(f"      {ln}")
    sys.exit(1)

got = logits.gather(1, idx.long().clamp(0, V - 1)).sort(dim=1).values
want = torch.topk(logits, K, dim=1).values.sort(dim=1).values
dup = sum(int(idx[r].unique().numel()) != K for r in range(B))
oob = int(((idx < 0) | (idx >= V)).sum())
ok = torch.allclose(got, want) and dup == 0 and oob == 0
print(f"RESULT {tag}: {'CORRECT' if ok else 'WRONG'}  rows_with_dup={dup}  "
      f"out_of_range={oob}  max|diff|={float((got - want).abs().max()):.3g}")
