"""Tighten the sampled path's TARGET_MULT -- and check the narrow band first.

THE REASON THIS PROBE LEADS WITH CORRECTNESS. The sampled prefill path keys
everything off `_key11 = _generic._convert_to_trt_uint16_hi11`: the sample
histogram, the collect, and -- this is the problem -- the exact fallback in
`_s_exact_pass`. There is no 32-bit escape anywhere in it.

Decode shipped a fix for exactly this last week. Its comment:

    The 11-bit fp16 key can COLLAPSE: a row whose values sit in a narrow band
    away from zero (relative spread below about 1%, the key's resolution being
    magnitude/32) maps to one or two bins, and then "an overflow can only drop
    what shares the k-th element's key" is true but vacuous -- everything
    shares it, so the true top-k can be dropped. The generic operator escapes
    through STEP 1-3, which refine over the full 32 bits; this path has no
    STEP 1-3.

The sampled prefill path has no STEP 1-3 either. On a band like 10.0 + 0.2*u
the whole row lands in one bin, `_s_collect` admits all 129280 elements into a
CAP=4096 buffer, `_s_finish` sees the overflow and redoes it -- with `_key11`
again, so the redo produces the same single bin and keeps an ARBITRARY 4096,
and the radix rounds then return the exact top-k OF THE WRONG 4096.

`tests/test_top_k_per_row_prefill.py` has eight tests and **no narrow-band
case**, so the "19 passed, 1 skipped" that every arm of
tools/hygon_prefill_sample_bench.py reported says nothing about this. Decode's
suite only grew its case when the bug was found there.

So phase 1 checks four inputs per setting and phase 2 times every setting
anyway, correct or not, because a fast wrong answer must never read as a win.

PHASE 1, in process, no timing: normal, tied (rounded to 1/4), NARROW BAND
(10.0 + 0.2*u, the decode repro), and constant. Per setting: whether the
answer matches torch.topk, the median and max candidate count, and the share
of rows landing outside the acceptance window [top_k, CAP], read between
collect and finish because `_s_finish` rewrites that buffer for the rows it
redoes.

PHASE 2: the real benchmark, two interleaved passes. TARGET_MULT 1.5 measured
+34% on (64,129280) and 1.25 measured +41%, still improving, so this goes to
1.0. The window's lower edge is top_k itself -- there is no margin under it,
and an undershoot costs a full redo -- which is what the outside column is for.

    tools/vendor_probe.sh tools/hygon_prefill_sample_tight.py hygon_prefill_sample_tight
"""

import importlib.util
import math
import os
import pathlib
import re
import subprocess
import sys

import torch

OVERRIDE = pathlib.Path(
    "src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py"
)
ROWS, VOCAB, TOPK = 64, 129280, 1024
STRIDE0 = VOCAB
MULTS = [1.0, 1.1, 1.25, 1.5]
SSTRIDE = 16
PASSES = 2
BENCH = ["benchmark/test_top_k_per_row_prefill.py", "--mode", "kernel"]
FOCUS = (ROWS, VOCAB, TOPK)


def sh(*a, **k):
    return subprocess.run(a, capture_output=True, text=True, **k)


def occupancy(tag):
    import shutil

    for cmd in (["hy-smi"], ["rocm-smi"]):
        exe = shutil.which(cmd[0]) or (
            f"/opt/dtk/bin/{cmd[0]}"
            if pathlib.Path(f"/opt/dtk/bin/{cmd[0]}").exists()
            else None
        )
        if not exe:
            continue
        print(f"--- card occupancy {tag}: {cmd[0]}")
        print("\n".join(sh(exe, *cmd[1:]).stdout.strip().splitlines()[:25]))
        return
    print(f"--- card occupancy {tag}: no smi tool found")


def load_sample8():
    spec = importlib.util.spec_from_file_location(
        "_s8src", "tools/hygon_prefill_sample8.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_s8src"] = mod
    spec.loader.exec_module(mod)
    return mod


def cases(dev):
    """The four row populations, (64, 129280) each."""
    torch.manual_seed(42)
    u = torch.rand(ROWS, VOCAB, device=dev, dtype=torch.float32)
    out = {
        "normal": torch.randn(ROWS, VOCAB, device=dev, dtype=torch.float32),
        # the decode repro: one binade, ULP 0.25, so the 11-bit key has ~1 bin
        "narrow": 10.0 + 0.2 * u,
        "constant": torch.full((ROWS, VOCAB), 3.5, device=dev, dtype=torch.float32),
    }
    out["tied"] = (out["normal"] * 4).round() / 4
    return out


def phase1(mod, dev):
    starts = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    ends = torch.full((ROWS,), VOCAB, dtype=torch.int32, device=dev)
    idx = torch.empty((ROWS, TOPK), dtype=torch.int32, device=dev)
    data = cases(dev)
    print(
        f"\nphase 1: correctness and window occupancy, SSTRIDE {SSTRIDE},"
        f" {ROWS}x{VOCAB} top_k {TOPK}\n"
    )
    print(
        f"  {'mult':>6} {'case':>9} {'answer':>8} {'cand med':>9} {'cand max':>9}"
        f" {'outside':>8}"
    )
    bad = set()
    for mult in MULTS:
        mod.SSTRIDE = SSTRIDE
        mod.TARGET_MULT = mult
        with mod._SPLAN_LOCK:
            mod._SPLANS.clear()
        for name, src in data.items():
            want = torch.topk(src, TOPK, dim=1).values.sort(dim=1).values
            idx.fill_(-9)
            mod.top_k_per_row_prefill(src, starts, ends, idx, ROWS, STRIDE0, 1, TOPK)
            torch.cuda.synchronize()
            got = src.gather(1, idx.long().clamp(0, VOCAB - 1)).sort(dim=1).values
            ok = torch.allclose(got, want) and bool((idx >= 0).all())
            if not ok:
                bad.add(mult)
            plan = next(iter(mod._SPLANS.values()))
            plan.prepare(src, starts, ends, plan.hist, plan.thr, plan.cnt, STRIDE0)
            plan.collect(
                src,
                starts,
                ends,
                plan.thr,
                plan.cnt,
                plan.cand_idx,
                plan.cand_val,
                STRIDE0,
            )
            torch.cuda.synchronize()
            c = plan.cnt.float()
            out = float(((c < min(TOPK, VOCAB)) | (c > plan.cap)).float().mean()) * 100
            print(
                f"  {mult:>6.2f} {name:>9} {'OK' if ok else 'WRONG':>8}"
                f" {int(c.median()):>9} {int(c.max()):>9} {out:>7.1f}%"
            )
    return bad


def parse_bench(out):
    rows = {}
    for m in re.finditer(
        r"SUCCESS\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[torch\.Size\(\[(\d+), (\d+)\]\)"
        r".*?, (\d+), (\d+), 1, (\d+)\]",
        out,
    ):
        rows[(int(m.group(4)), int(m.group(5)), int(m.group(8)))] = (
            float(m.group(3)),
            float(m.group(1)),
        )
    return rows


def run_bench(env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s"] + BENCH,
        capture_output=True,
        text=True,
        env=env,
    )
    rows = parse_bench(r.stdout)
    if not rows:
        print(r.stdout[-2500:])
        raise SystemExit("the benchmark produced no SUCCESS rows")
    return rows


def main():
    s8 = load_sample8()
    fixed = s8.fixed_source()
    dirty = sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
    if dirty.strip():
        raise SystemExit("the override is already modified:\n" + dirty)
    occupancy("before")
    mod = s8.sampled_module()
    bad = phase1(mod, "cuda")
    del mod

    arms = [("off", {"FLAGGEMS_HYGON_PREFILL_SAMPLED_RATIO": "0"})]
    for m in MULTS:
        arms.append(
            (
                f"m{m}",
                {
                    "FLAGGEMS_HYGON_PREFILL_SSTRIDE": str(SSTRIDE),
                    "FLAGGEMS_HYGON_PREFILL_TARGET_MULT": str(m),
                },
            )
        )
    bench = {tag: [] for tag, _ in arms}
    try:
        OVERRIDE.write_text(fixed)
        for p in range(PASSES):
            for tag, env in arms:
                print(f"### pass {p + 1}, arm {tag}", flush=True)
                bench[tag].append(run_bench(env))
    finally:
        r = sh("git", "checkout", "--", str(OVERRIDE))
        left = sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
        print(f"### restored; git status: {left.strip() or 'clean'} ({r.returncode})")

    print(f"\nphase 2: benchmark SpeedUp on {ROWS}x{VOCAB}, two passes\n")
    print(f"  {'arm':>8} {'pass 1':>9} {'pass 2':>9} {'vs off':>9}  answer")
    offv = sum(bench["off"][p][FOCUS][0] for p in range(PASSES)) / PASSES
    for tag, _ in arms:
        v = [bench[tag][p][FOCUS][0] for p in range(PASSES)]
        mult = None if tag == "off" else float(tag[1:])
        note = "n/a" if mult is None else ("WRONG somewhere" if mult in bad else "OK")
        print(
            f"  {tag:>8} {v[0]:>9.3f} {v[1]:>9.3f}"
            f" {sum(v) / PASSES / offv:>9.3f}  {note}"
        )
    g = lambda x: math.exp(sum(map(math.log, x)) / len(x))  # noqa: E731
    shapes = sorted(bench["off"][0])
    print("\n  whole-suite geomean (the reverted file lacks one-scan, so read")
    print("  these against 'off', never against production):")
    for tag, _ in arms:
        gm = [g([bench[tag][p][k][0] for k in shapes]) for p in range(PASSES)]
        print(f"    {tag:>8} {gm[0]:.3f} / {gm[1]:.3f}")
    vals = [bench[t][p][FOCUS][1] for t, _ in arms for p in range(PASSES)]
    spread = max(vals) / min(vals)
    print(f"\n  vLLM latency on that shape, max/min {spread:.2f}")
    occupancy("after")
    return 0


if __name__ == "__main__":
    sys.exit(main())
