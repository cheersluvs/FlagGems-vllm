"""Sweep the TLE launch knobs the geometry sweep left alone, on MetaX.

Baseline is the SHIPPED rule, not the NVIDIA defaults: decode blocks-per-row
from the MetaX entry's _blocks_per_row(rows) on a 512 tile, prefill tile 1024
for vocab >= 16384 else 512. Each config changes ONE thing:

  decode   main-kernel warps (default 8 = 1 element/thread on the 512 tile)
           merge tile 256 / 512 / 1024 and its warps (merge is the second of
           the two launches per call)
  prefill  warps; the launch split at SORTING_ALGORITHM_THRESHOLD (12288):
           rows past it get a second launch that, with radix final replaced by
           the rank select, runs the same work -- is it pure overhead?

Warps are set by rebinding the generic module's _num_warps. Decode's two
launches both ask _num_warps(512) when the merge tile is 512, so they cannot
be told apart by argument; the host calls main then merge on every call, so
the warps are handed out in that order, and a config whose call makes any
other number of launches is reported invalid rather than timed.

Every config is checked against torch.topk before timing; the baseline is
timed first and last to expose drift. Calls go to the GENERIC op, because the
MetaX entry re-applies the shipped rule on every call.

    PY=/data/wuyuqing/workspace/mctle-v2/bin/python \\
        tools/vendor_probe.sh tools/metax_tle_launch_sweep.py metax_launch_sweep
    ... --op decode | --op prefill
"""

import argparse
import itertools
import math
import os
import sys
from importlib import import_module

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metax_tle_geometry_sweep as G  # noqa: E402
import torch  # noqa: E402

import flaggems_vllm  # noqa: E402

dec, pre, tle = G.dec, G.pre, G.tle
ovd = import_module("flaggems_vllm.runtime.backend._metax.fused.top_k_per_row_decode")
ovp = import_module("flaggems_vllm.runtime.backend._metax.fused.top_k_per_row_prefill")

DECODE_ROWS = [1, 4, 8, 16, 24, 32, 40, 48, 56, 496, 512]
INF = 1 << 40

# (name, main_warps, merge_tile, merge_warps); None = the generic rule
DECODE_CONFIGS = [
    ("base: main W8 | merge 512 W8", None, 512, None),
    ("main W4 (2/thr)", 4, 512, None),
    ("main W2 (4/thr)", 2, 512, None),
    ("merge 256 W4 (1/thr)", None, 256, 4),
    ("merge 256 W2 (2/thr)", None, 256, 2),
    ("merge 512 W4 (2/thr)", None, 512, 4),
    ("merge 512 W2 (4/thr)", None, 512, 2),
    ("merge 1024 W8 (2/thr)", None, 1024, None),
]
# (name, warps, threshold)
PREFILL_CONFIGS = [
    ("base: W rule | split 12288", None, 12288),
    ("W4", 4, 12288),
    ("W2", 2, 12288),
    ("no split", None, INF),
    ("no split + W4", 4, INF),
]


class Launches:
    """Hands out warps in launch order and counts launches per op call."""

    def __init__(self, module, per_launch):
        self.orig = module._num_warps
        self.per_launch = per_launch
        self.count = 0
        self.cycle = itertools.cycle(per_launch)

    def __call__(self, block_size):
        self.count += 1
        w = next(self.cycle)
        return w if w else self.orig(block_size)


def apply(op, cfg, shape):
    if op == "decode":
        _, main_w, merge_tile, merge_w = cfg
        dec.NUM_THREADS_PER_BLOCK = 512
        dec.MULTIPLE_BLOCKS_PER_ROW_CONFIG = ovd._blocks_per_row(shape)
        dec.NUM_THREADS_PER_BLOCK_MERGE = merge_tile
        hook = Launches(ORIG["decode"], [main_w, merge_w])
        dec._num_warps = hook
        return hook, 2
    _, w, thresh = cfg
    rows, vocab = shape[0], shape[1]
    pre.NUM_THREADS_PER_BLOCK = 1024 if vocab >= ovp.WIDE_TILE_VOCAB else 512
    pre.SORTING_ALGORITHM_THRESHOLD = thresh
    hook = Launches(ORIG["prefill"], [w])
    pre._num_warps = hook
    # vocab >= 65536 goes straight to the (radix-flagged) launch: one launch.
    # Otherwise rows past the threshold get a second launch.
    if pre._use_radix_final_for_prefill(vocab):
        return hook, 1
    return hook, (2 if rows > thresh else 1)


class _Orig:
    def __init__(self, module):
        self._num_warps = module._num_warps


ORIG = {}


def restore():
    dec._num_warps = ORIG["decode"]._num_warps
    pre._num_warps = ORIG["prefill"]._num_warps
    dec.NUM_THREADS_PER_BLOCK = 512
    dec.NUM_THREADS_PER_BLOCK_MERGE = ORIG["merge"]
    pre.NUM_THREADS_PER_BLOCK = 512
    pre.SORTING_ALGORITHM_THRESHOLD = ORIG["thresh"]


def run_one(op, cfg, shape):
    hook, launches = apply(op, cfg, shape)
    args = G.decode_inputs(shape) if op == "decode" else G.prefill_inputs(shape)
    fn = dec.top_k_per_row_decode if op == "decode" else pre.top_k_per_row_prefill
    try:
        hook.count = 0
        fn(*args)
        torch.cuda.synchronize()
        if hook.count != launches:
            return None, f"INVALID: {hook.count} launches, expected {launches}"
        if not G.correct(args):
            return None, "WRONG"
        return G.bench(lambda: fn(*args)), "ok"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:70]}"
    finally:
        restore()


def sweep(op):
    shapes = DECODE_ROWS if op == "decode" else G.PREFILL_ALL
    configs = DECODE_CONFIGS if op == "decode" else PREFILL_CONFIGS
    base = configs[0]
    vllm = {s: G.vllm_ms(op, s) for s in shapes}
    print(f"\n######## {op}   vLLM baseline: {'yes' if G.HAS_VLLM else 'NO'}")
    first = {s: run_one(op, base, s)[0] for s in shapes}
    res = {c: {s: run_one(op, c, s) for s in shapes} for c in configs[1:]}
    last = {s: run_one(op, base, s)[0] for s in shapes}

    print("\n  drift, baseline first -> last (ms):")
    for s in shapes:
        a, b = first[s], last[s]
        d = f"{(b / a - 1) * 100:+.1f}%" if a and b else "n/a"
        print(f"    {G.label(s):<22} {a!s:>22} -> {b!s:>22}  {d}")

    print("\n  ratio vs vLLM (x = speed vs baseline); * = fastest correct")
    for s in shapes:
        ref = (
            min(v for v in (first[s], last[s]) if v) if (first[s] or last[s]) else None
        )
        row = {c: res[c][s] for c in configs[1:]}
        ok = {c: r[0] for c, r in row.items() if r[0] is not None}
        best = min(ok, key=ok.get) if ok and ref and min(ok.values()) < ref else None
        head = f"base {vllm[s] / ref:.3f}" if (ref and vllm[s]) else f"base {ref} ms"
        print(f"\n  {G.label(s)}   {head}")
        for c in configs[1:]:
            ms, st = row[c]
            if ms is None:
                print(f"    {c[0]:<26} {st}")
                continue
            r = f"{vllm[s] / ms:.3f}" if vllm[s] else "-"
            print(f"    {c[0]:<26} {r:>6}  {ref / ms:.3f}x{'  *' if c == best else ''}")

    if G.HAS_VLLM:
        print("\n  geomean ratio vs vLLM:")

        def gm(vals):
            return math.exp(sum(map(math.log, vals)) / len(vals))

        refs = {s: min(v for v in (first[s], last[s]) if v) for s in shapes}
        print(f"    {base[0]:<26} {gm([vllm[s] / refs[s] for s in shapes]):.3f}")
        for c in configs[1:]:
            if any(res[c][s][0] is None for s in shapes):
                print(f"    {c[0]:<26} (a shape failed / was wrong / invalid)")
                continue
            print(f"    {c[0]:<26} {gm([vllm[s] / res[c][s][0] for s in shapes]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["decode", "prefill", "both"], default="both")
    a = ap.parse_args()

    flaggems_vllm.top_k_per_row_decode(*G.decode_inputs(4))  # TLE self-test
    torch.cuda.synchronize()
    st = tle.status()
    print(f"TLE status: {st}")
    if not st.get("on"):
        print("!! TLE path is off; use the mctle venv")
        return 3
    ORIG["decode"] = _Orig(dec)
    ORIG["prefill"] = _Orig(pre)
    ORIG["merge"] = dec.NUM_THREADS_PER_BLOCK_MERGE
    ORIG["thresh"] = pre.SORTING_ALGORITHM_THRESHOLD
    print(
        f"geometry {dec._launch_geometry()}  merge tile {ORIG['merge']}  "
        f"prefill split {ORIG['thresh']}"
    )
    for op in (["decode", "prefill"] if a.op == "both" else [a.op]):
        sweep(op)
    return 0


if __name__ == "__main__":
    sys.exit(main())
