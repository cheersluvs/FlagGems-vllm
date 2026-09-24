"""One environment switch per operator, validated on the card.

Also: the module copies are now registered under the override's own name
(<override>._dense, ._vec2, ...) instead of flaggems_vllm.ops._top_k_*_hygon_*.

The Hygon prefill override had twelve switches and decode two; both now have
exactly one: FLAGGEMS_HYGON_TOPK_PREFILL=0 and FLAGGEMS_HYGON_TOPK_DECODE=0
(renamed from FLAGGEMS_HYGON_TOPK_DECODE_SAMPLED). The tuning values that were
env-read at import (SSTRIDE 16, TARGET_MULT 1.25, SSPLIT 4, ratio 64) are
constants. With the switches unset the code takes exactly the branches it took
before, so the numbers should not move.

PART 0 -- the switches, each in its own process:
    default          every module copy loaded, no warning, prefill and decode
                     exact on one sampled, one one-read and one dense shape
    PREFILL=0        no patched copy (onescan/carry/vec2/short-bins/retry all
                     None), calls go to the generic function, answers exact
    DECODE=0         decode's _enabled() False, answer exact
    stale old vars   the removed variables set to garbage (SSPLIT=four ...):
                     import must still succeed -- before, an unparsable value
                     raised at import and took the whole vendor lib down

PART 1 -- tools/hygon_prefill_before_after.py: both suites, then prefill
before (PREFILL=0) / after interleaved twice.

PART 2 -- decode benchmark, two passes, kernel mode.

    tools/vendor_probe.sh tools/hygon_switch_check.py hygon_switch_check
"""

import math
import os
import pathlib
import re
import subprocess
import sys

CHILD = r"""
import logging, os, sys, torch
records = []
class H(logging.Handler):
    def emit(self, r):
        records.append(r.getMessage())
logging.getLogger().addHandler(H())
logging.getLogger().setLevel(logging.WARNING)
from importlib import import_module
pre = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
dec = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode")
gen = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
print("OUT prefill _ENABLED", pre._ENABLED, "| decode _enabled()", dec._enabled())
print("OUT copies", {n: getattr(pre, n) is not None for n in
      ("_dense_carry", "_dense_vec2", "_dense_short_bins", "_dense_retry")},
      "onescan", pre._ONESCAN_PATH is not None, "sparse is generic", pre._sparse is gen)
print("OUT constants SSTRIDE", pre.SSTRIDE, "TARGET_MULT", pre.TARGET_MULT,
      "SSPLIT", pre.SSPLIT, "ratio", pre.SAMPLED_MIN_VOCAB_PER_TOPK)
print("OUT warnings", [r for r in records if "hygon" in r.lower()] or "none")
print("OUT copy names", [m.__name__ for m in (pre._dense, pre._sparse, pre._dense_carry,
      pre._dense_vec2, pre._dense_short_bins, pre._dense_retry) if m is not None])
print("OUT old names in sys.modules",
      [n for n in sys.modules if "_top_k_per_row_prefill_hygon" in n] or "none")

calls = []
orig = gen.top_k_per_row_prefill
def spy(*a, **k):
    calls.append(1)
    return orig(*a, **k)
gen.top_k_per_row_prefill = spy

dev = "cuda"
def check_prefill(num_rows, vocab, top_k, stride0):
    torch.manual_seed(1)
    buf = torch.randn((num_rows - 1) * stride0 + vocab, device=dev)
    x = torch.as_strided(buf, (num_rows, vocab), (stride0, 1))
    st = torch.zeros(num_rows, dtype=torch.int32, device=dev)
    en = torch.full((num_rows,), vocab, dtype=torch.int32, device=dev)
    out = torch.full((num_rows, top_k), -1, dtype=torch.int32, device=dev)
    n0 = len(calls)
    pre.top_k_per_row_prefill(x, st, en, out, num_rows, stride0, 1, top_k)
    torch.cuda.synchronize()
    ref = torch.topk(x, top_k, dim=1).values
    got = torch.gather(x, 1, out.long().clamp(min=0)).sort(dim=1, descending=True)[0]
    ok = float((got - ref).abs().max()) == 0.0 and bool((out >= 0).all())
    print(f"OUT prefill {num_rows}x{vocab} k{top_k}: {'ok' if ok else 'WRONG'}"
          f"  generic called: {len(calls) > n0}")

check_prefill(64, 129280, 1024, 129280)   # sampled route
check_prefill(12961, 4100, 512, 4360)     # one-read route
check_prefill(4100, 1025, 512, 1025)      # dense route

rows, vocab, top_k = 8, 262144, 512
torch.manual_seed(2)
x = torch.randn(rows, vocab, device=dev)
lens = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
out = torch.full((rows, top_k), -1, dtype=torch.int32, device=dev)
dec.top_k_per_row_decode(x, 1, lens, out, rows, x.stride(0), x.stride(1), top_k)
torch.cuda.synchronize()
ref = torch.topk(x, top_k, dim=1).values
got = torch.gather(x, 1, out.long().clamp(min=0)).sort(dim=1, descending=True)[0]
print("OUT decode 8x262144 k512:", "ok" if float((got - ref).abs().max()) == 0.0 else "WRONG")
"""

STALE = {
    "FLAGGEMS_HYGON_PREFILL_SSPLIT": "four",
    "FLAGGEMS_HYGON_PREFILL_SSTRIDE": "x",
    "FLAGGEMS_HYGON_PREFILL_TARGET_MULT": "1,25",
    "FLAGGEMS_HYGON_PREFILL_SAMPLED_RATIO": "?",
    "FLAGGEMS_HYGON_TOPK_DECODE_SPLIT": "?",
}
CASES = [
    ("default", {}),
    ("PREFILL=0", {"FLAGGEMS_HYGON_TOPK_PREFILL": "0"}),
    ("DECODE=0", {"FLAGGEMS_HYGON_TOPK_DECODE": "0"}),
    ("stale old vars", STALE),
]


def parse(out):
    rows = {}
    for m in re.finditer(
        r"SUCCESS\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[torch\.Size\(\[(\d+), (\d+)\]\)"
        r".*?, (\d+), (\d+), 1, (\d+)\]",
        out,
    ):
        rows[(int(m.group(4)), int(m.group(5)), int(m.group(8)))] = float(m.group(3))
    return rows


def main():
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "tests"],
        capture_output=True,
        text=True,
    ).stdout
    if dirty.strip():
        raise SystemExit("the tree is modified:\n" + dirty)
    here = pathlib.Path(__file__).resolve().parent

    print("### part 0: switches\n", flush=True)
    for tag, extra in CASES:
        env = {
            k: v for k, v in os.environ.items() if not k.startswith("FLAGGEMS_HYGON_")
        }
        env.update(extra)
        r = subprocess.run(
            [sys.executable, "-c", CHILD], capture_output=True, text=True, env=env
        )
        print(f"  [{tag}]", flush=True)
        for ln in r.stdout.splitlines():
            if ln.startswith("OUT"):
                print("    " + ln[4:], flush=True)
        if r.returncode:
            print("    ! failed:")
            for ln in r.stderr.strip().splitlines()[-8:]:
                print(f"      | {ln[:200]}")

    print("\n### part 1: before / after\n", flush=True)
    subprocess.run([sys.executable, str(here / "hygon_prefill_before_after.py")])

    print("\n### part 2: decode benchmark, two passes\n", flush=True)
    res = []
    for _ in range(2):
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-s",
                "benchmark/test_top_k_per_row_decode.py",
                "--mode",
                "kernel",
            ],
            capture_output=True,
            text=True,
        )
        rows = parse(r.stdout)
        if not rows:
            print(r.stdout[-2000:])
            raise SystemExit("decode: no SUCCESS rows")
        res.append(rows)
    shapes = sorted(res[0], key=lambda k: k[0] * k[1])
    for k in shapes:
        print(f"  {k[0]:>5} {k[1]:>7} {k[2]:>5}   {res[0][k]:.3f} / {res[1][k]:.3f}")
    g = [math.exp(sum(math.log(p[k]) for k in shapes) / len(shapes)) for p in res]
    print(f"\n  decode geomean {g[0]:.3f} / {g[1]:.3f}  (PR table: 1.529 / 1.560)")


if __name__ == "__main__":
    main()
