"""Acceptance for the landed prefill routes: tests, then both operators twice.

This is the plain acceptance run for what topk-metax now carries, not an
experiment. Nothing is patched, swapped or loaded from another ref: it runs
the suite and the benchmark exactly as the repo ships them.

    tests/test_top_k_per_row_{prefill,decode}.py
    benchmark/test_top_k_per_row_{prefill,decode}.py --mode kernel, twice

Expected, from tools/hygon_prefill_audit_bench.py and the decode work:

    prefill geomean ~0.787, 3 of 7 shapes at or above the 0.9 bar
    decode  geomean ~1.53, every shape at or above 1.13

The vLLM latency column is printed per pass. It is the same C++ kernel on the
same input every time, so a column that is not flat means the card was busy and
that pass should be discarded rather than quoted -- the failure mode that made
the audit branch's own report unreadable.

    tools/vendor_probe.sh tools/hygon_topk_accept.py hygon_topk_accept
"""

import math
import pathlib
import re
import subprocess
import sys

PASSES = 2
OPS = ("prefill", "decode")


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True)


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


def pytest(args):
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s"] + args,
        capture_output=True,
        text=True,
    )


def parse(out):
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


def main():
    occupancy("before")
    print("\n### tests", flush=True)
    for op in OPS:
        r = pytest([f"tests/test_top_k_per_row_{op}.py"])
        m = re.search(r"\d+ passed(?:, \d+ skipped)?(?:, \d+ failed)?", r.stdout)
        bad = re.search(r"(\d+) (failed|error)", r.stdout)
        print(
            f"  {op:>8}: {m.group(0) if m else r.stdout[-300:]}"
            f"{'   <== FAILURES' if bad else ''}",
            flush=True,
        )

    res = {op: [] for op in OPS}
    for p in range(PASSES):
        for op in OPS:
            print(f"### benchmark pass {p + 1}, {op}", flush=True)
            r = pytest([f"benchmark/test_top_k_per_row_{op}.py", "--mode", "kernel"])
            rows = parse(r.stdout)
            if not rows:
                print(r.stdout[-2000:])
                raise SystemExit(f"{op}: no SUCCESS rows")
            res[op].append(rows)

    g = lambda v: math.exp(sum(map(math.log, v)) / len(v))  # noqa: E731
    for op in OPS:
        shapes = sorted(res[op][0], key=lambda k: k[0] * k[1])
        print(f"\n{op}, --mode kernel, baseline = vLLM's own C++\n")
        print(
            f"  {'num_rows':>9} {'vocab':>7} {'top_k':>6} {'vLLM p1':>9}"
            f" {'Gems p1':>9} {'SpeedUp p1':>11} {'SpeedUp p2':>11}"
        )
        for k in shapes:
            a, b = res[op][0][k], res[op][1][k]
            flag = "" if a[0] >= 0.9 else "   below 0.9"
            print(
                f"  {k[0]:>9} {k[1]:>7} {k[2]:>6} {a[1]:>9.4f}"
                f" {a[1] / a[0]:>9.4f} {a[0]:>11.3f} {b[0]:>11.3f}{flag}"
            )
        gm = [g([res[op][p][k][0] for k in shapes]) for p in range(PASSES)]
        over = sum(res[op][0][k][0] >= 0.9 for k in shapes)
        print(
            f"\n  geomean {gm[0]:.3f} / {gm[1]:.3f}"
            f"   at or above 0.9: {over}/{len(shapes)}"
        )
        spreads = [
            max(res[op][p][k][1] for p in range(PASSES))
            / min(res[op][p][k][1] for p in range(PASSES))
            for k in shapes
        ]
        print(f"  vLLM latency max/min across passes: {max(spreads):.2f}")
    occupancy("after")
    return 0


if __name__ == "__main__":
    sys.exit(main())
