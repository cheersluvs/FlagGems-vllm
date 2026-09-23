"""Before/after for the upstream prefill PR, measured in one session.

`before` switches every part of the Hygon override off, so a call reaches the
generic module's own top_k_per_row_prefill at its own launch geometry -- what
upstream main runs on this card today:

    FLAGGEMS_HYGON_PREFILL_SAMPLED_RATIO=0   no sampled path
    FLAGGEMS_HYGON_TOPK_SLOTSCAN=0           no dense copies
    FLAGGEMS_HYGON_TOPK_ONESCAN=0            the untouched generic module
    FLAGGEMS_HYGON_TOPK_GEOMETRY=0           generic's BLOCK_SIZE / num_warps
    FLAGGEMS_HYGON_TOPK_SCRATCH_REUSE=0      generic's own host wrapper
    FLAGGEMS_HYGON_PREFILL_DENSE_SAMPLED=0   no one-read dense route

`after` is the override as it ships. Interleaved before/after/before/after,
prefill benchmark in kernel mode, vLLM latency printed per run.

    tools/vendor_probe.sh tools/hygon_prefill_before_after.py hygon_prefill_before_after
"""

import os
import re
import subprocess
import sys

BENCH = ["benchmark/test_top_k_per_row_prefill.py", "--mode", "kernel"]
OFF = {
    "FLAGGEMS_HYGON_PREFILL_SAMPLED_RATIO": "0",
    "FLAGGEMS_HYGON_TOPK_SLOTSCAN": "0",
    "FLAGGEMS_HYGON_TOPK_ONESCAN": "0",
    "FLAGGEMS_HYGON_TOPK_GEOMETRY": "0",
    "FLAGGEMS_HYGON_TOPK_SCRATCH_REUSE": "0",
    "FLAGGEMS_HYGON_PREFILL_DENSE_SAMPLED": "0",
}
ORDER = ["before", "after", "before", "after"]


def parse(out):
    rows = {}
    for mm in re.finditer(
        r"SUCCESS\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[torch\.Size\(\[(\d+), (\d+)\]\)"
        r".*?, (\d+), (\d+), 1, (\d+)\]",
        out,
    ):
        rows[(int(mm.group(4)), int(mm.group(5)), int(mm.group(8)))] = (
            float(mm.group(1)),
            float(mm.group(2)),
            float(mm.group(3)),
        )
    return rows


def geo(vals):
    g = 1.0
    for v in vals:
        g *= v
    return g ** (1.0 / len(vals))


def main():
    print("### tests, as shipped", flush=True)
    for suite in ("prefill", "decode"):
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-rf",
                f"tests/test_top_k_per_row_{suite}.py",
            ],
            capture_output=True,
            text=True,
        )
        tail = [ln for ln in r.stdout.splitlines() if "passed" in ln or "failed" in ln]
        print(f"  {suite}: {tail[-1] if tail else 'no result'}", flush=True)
        for ln in [x for x in r.stdout.splitlines() if x.startswith("FAILED")][:5]:
            print(f"    {ln[:200]}", flush=True)
    runs = {"before": [], "after": []}
    for i, arm in enumerate(ORDER):
        env = dict(os.environ)
        if arm == "before":
            env.update(OFF)
        print(f"### run {i + 1}: {arm}", flush=True)
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-s"] + BENCH,
            capture_output=True,
            text=True,
            env=env,
        )
        rows = parse(r.stdout)
        if not rows:
            print("  ! no SUCCESS rows", flush=True)
            for ln in (r.stdout + r.stderr).strip().splitlines()[-10:]:
                print(f"    | {ln[:200]}", flush=True)
            continue
        runs[arm].append(rows)

    if not (runs["before"] and runs["after"]):
        raise SystemExit("a side failed; nothing to compare")
    shapes = sorted(runs["after"][0], key=lambda k: (k[1] < 100000, k))
    print(
        "\n  num_rows  vocab   top_k   vLLM ms   before ms  after ms"
        "   before SpeedUp       after SpeedUp"
    )
    for s in shapes:
        vl = [r[s][0] for side in runs.values() for r in side]
        b = [r[s] for r in runs["before"]]
        a = [r[s] for r in runs["after"]]
        print(
            f"  {s[0]:>8} {s[1]:>7} {s[2]:>6}   {vl[0]:.4f}    {b[0][1]:.4f}    {a[0][1]:.4f}"
            f"   {' / '.join(f'{x[2]:.3f}' for x in b):>15}   "
            f"{' / '.join(f'{x[2]:.3f}' for x in a):>15}"
        )
    for side in ("before", "after"):
        g = [geo([r[s][2] for s in shapes]) for r in runs[side]]
        g5 = [geo([r[s][2] for s in shapes if s[0] != 4]) for r in runs[side]]
        n = [sum(1 for s in shapes if r[s][2] >= 0.9) for r in runs[side]]
        print(
            f"  {side:>6}: geomean {' / '.join(f'{x:.3f}' for x in g)}"
            f"   without the 4-row pair {' / '.join(f'{x:.3f}' for x in g5)}"
            f"   >= 0.9: {' / '.join(str(x) for x in n)} of {len(shapes)}"
        )
    for s in shapes:
        vl = [r[s][0] for side in runs.values() for r in side]
        print(
            f"  vLLM latency {s[0]}x{s[1]} across runs: max/min {max(vl) / min(vl):.2f}"
        )


if __name__ == "__main__":
    main()
