"""Which branch of the TLE decode path is wrong on Hygon?

tools/hygon_topk_tle_try.py: with FLAGGEMS_FORCE_TLE=1 the generic TLE path
compiles and runs on hcu with no shims at all, but decode returns WRONG values
at every shape tried, while prefill is correct. Same bisect that worked on
MetaX, where the answer was the radix final select:

    radix0    the radix final select off (rank select instead)
    mb0       the multi-block + merge path off (one program per row)
    both

Each case in its own process, one shape list per case, checked against
torch.topk. A case that is CORRECT names the branch at fault.

    tools/vendor_probe.sh tools/hygon_tle_decode_bisect.py hygon_tle_bisect
"""

import os
import subprocess
import sys
from importlib import import_module

SHAPES = (
    (1, 262144, 512),
    (8, 262144, 512),
    (64, 262144, 512),
    (8, 32768, 512),
    (64, 129280, 1024),
)
CASES = ("base", "radix0", "mb0", "radix0+mb0")

if len(sys.argv) == 1:
    print("FLAGGEMS_FORCE_TLE=1 for every case; 'base' is the TLE path as it ships\n")
    for case in CASES:
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), case],
            capture_output=True,
            text=True,
            timeout=1800,
            env=dict(os.environ, FLAGGEMS_FORCE_TLE="1"),
        )
        out = (r.stdout + r.stderr).splitlines()
        lines = [x for x in out if x.startswith("RESULT")]
        if not lines:
            hint = next(
                (x for x in out if "Error" in x or "Assertion" in x),
                out[-1] if out else "no output",
            )
            lines = [
                f"RESULT {case}: CRASHED (exit {r.returncode}) -- {hint.strip()[:120]}"
            ]
        for ln in lines:
            print(f"  {ln}")
    print("\n  A case that turns every shape CORRECT is the branch at fault.")
    sys.exit(0)

CASE = sys.argv[1]
import torch  # noqa: E402

import flaggems_vllm  # noqa: E402

dec = import_module("flaggems_vllm.ops.top_k_per_row_decode")
if not dec.HAS_TLE:
    print(f"RESULT {CASE}: SKIP (HAS_TLE is False -- FLAGGEMS_FORCE_TLE not set?)")
    sys.exit(3)
if "radix0" in CASE:
    dec.SORTING_ALGORITHM_THRESHOLD = 1 << 40  # never use the radix final
if "mb0" in CASE:
    dec.SPLIT_WORK_THRESHOLD = 1 << 40  # never take multi-block+merge

torch.manual_seed(0)
verdicts = []
for B, V, K in SHAPES:
    tag = f"{B}x{V}/{K}"
    try:
        logits = torch.randn(B, V, dtype=torch.float32, device="cuda")
        lens = torch.full((B,), V, dtype=torch.int32, device="cuda")
        idx = torch.zeros(B, K, dtype=torch.int32, device="cuda")
        flaggems_vllm.top_k_per_row_decode(
            logits, 1, lens, idx, B, logits.stride(0), logits.stride(1), K
        )
        torch.cuda.synchronize()
        want = torch.topk(logits, K, dim=1).values.sort(dim=1).values
        got = logits.gather(1, idx.long().clamp(0, V - 1)).sort(dim=1).values
        oob = int(((idx < 0) | (idx >= V)).sum())
        dup = sum(int(idx[r].unique().numel()) != K for r in range(B))
        if torch.allclose(got, want) and oob == 0 and dup == 0:
            verdicts.append(f"{tag}:OK")
        else:
            verdicts.append(f"{tag}:WRONG(dup={dup},oob={oob})")
    except Exception as e:  # noqa: BLE001
        verdicts.append(f"{tag}:{type(e).__name__}")
print(f"RESULT {CASE}: " + "  ".join(verdicts))
