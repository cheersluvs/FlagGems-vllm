"""The audit branch's prefill override, through the real benchmark.

WHY NOT THE PROFILER. tools/hygon_prefill_audit_ab.py put every arm in one
process and interleaved them, which settled the arm-to-arm question: the audit
branch is 1.094x the shipped override, and 1.112x with its scratch cache off.
But its `vllm` column is unusable -- this card's profiler DOUBLE-COUNTS the
vLLM baseline, measured there at 1.69-1.99x the benchmark's own latency
(geomean 1.88) with ROCTracer's "duplicate flow start" warning in the log.
That is a trap already written down in this operator's notes, and the probe
built to fix someone else's measurement error walked straight into it.

So the ratios that go in an acceptance table have to come from the benchmark
itself. This runs `benchmark/test_top_k_per_row_prefill.py --mode kernel`.

WHY A FILE SWAP IS SAFE HERE. Outside reports/ and tools/, topk-metax and
codex/hygon-prefill-audit differ in exactly FOUR files, all of them the prefill
override and its companions:

    top_k_per_row_prefill.py
    _top_k_per_row_prefill_carry_source.py      (audit only)
    _top_k_per_row_prefill_final_network.py     (audit only)
    _top_k_per_row_prefill_final_source.py      (audit only)

The generic operator, the benchmark harness, core_shapes.yaml, the fused
__init__ and even the decode override are IDENTICAL on both branches, so
swapping those four and nothing else makes the comparison apples-to-apples by
construction rather than by hope.

ARMS, each run twice, interleaved pass by pass so any drift is shared:

    ship         the four paths as this branch has them
    audit        the four paths from the audit ref
    audit-nosr   the same, with FLAGGEMS_HYGON_TOPK_SCRATCH_REUSE=0. The
                 interleaved probe put the 512 MB cache at -1.2% overall: it
                 costs 11% on (64,129280) and 4-7% on the other sparse shapes
                 to buy ~3% on the dense ones.

The vLLM latency column is printed per arm and per pass. It is the same C++
kernel on the same input every time, so anything but a flat column means the
card was busy and that pass should be thrown away -- which is exactly how the
audit branch's own report went wrong (2.69x on one shape).

RECOVERY. The swap is undone in a finally block. If the process is killed
between the two, restore by hand:

    git checkout -- src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py
    rm -f src/flaggems_vllm/runtime/backend/_hygon/fused/_top_k_per_row_prefill_*.py

    tools/vendor_probe.sh tools/hygon_prefill_audit_bench.py hygon_prefill_audit_bench
"""

import math
import os
import pathlib
import re
import subprocess
import sys

FUSED = pathlib.Path("src/flaggems_vllm/runtime/backend/_hygon/fused")
OVERRIDE = FUSED / "top_k_per_row_prefill.py"
COMPANIONS = [
    FUSED / "_top_k_per_row_prefill_carry_source.py",
    FUSED / "_top_k_per_row_prefill_final_network.py",
    FUSED / "_top_k_per_row_prefill_final_source.py",
]
PATHS = [OVERRIDE] + COMPANIONS
AUDIT_REFS = ("origin/codex/hygon-prefill-audit", "0b05008")
PASSES = 2
BENCH = ["benchmark/test_top_k_per_row_prefill.py", "--mode", "kernel"]


def sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


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
        out = sh(exe, *cmd[1:]).stdout
        print(f"--- card occupancy {tag}: {cmd[0]}")
        print("\n".join(out.strip().splitlines()[:25]))
        return
    print(f"--- card occupancy {tag}: no smi tool found")


def audit_ref():
    for ref in AUDIT_REFS:
        if sh("git", "show", f"{ref}:{OVERRIDE}").returncode == 0:
            return ref
    raise SystemExit(
        f"cannot read the audit override from any of {AUDIT_REFS}.\n"
        "    run:  git fetch origin codex/hygon-prefill-audit"
    )


def install(ref):
    """`ref` None restores this branch's state; otherwise take the four paths
    from the audit ref. Companions are absent on this branch, so restoring
    means deleting them."""
    if ref is None:
        r = sh("git", "checkout", "--", str(OVERRIDE))
        assert r.returncode == 0, r.stderr
        for c in COMPANIONS:
            if c.exists():
                c.unlink()
        return
    for p in PATHS:
        r = sh("git", "show", f"{ref}:{p}")
        assert r.returncode == 0 and r.stdout, f"cannot read {p} from {ref}"
        p.write_text(r.stdout)


def parse(out):
    """(speedup, vllm_ms) per shape, keyed by (rows, vocab, top_k)."""
    rows = {}
    for m in re.finditer(
        r"SUCCESS\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[torch\.Size\(\[(\d+), (\d+)\]\)"
        r".*?, (\d+), (\d+), 1, (\d+)\]",
        out,
    ):
        key = (int(m.group(4)), int(m.group(5)), int(m.group(8)))
        rows[key] = (float(m.group(3)), float(m.group(1)), float(m.group(2)))
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
    rows = parse(r.stdout)
    if not rows:
        print(r.stdout[-3000:])
        print(r.stderr[-2000:])
        raise SystemExit("the benchmark produced no SUCCESS rows")
    return rows


def main():
    ref = audit_ref()
    dirty = sh("git", "status", "--porcelain", "--", *[str(p) for p in PATHS]).stdout
    if dirty.strip():
        raise SystemExit(
            "these paths are already modified; commit or restore them first:\n" + dirty
        )
    print(f"### audit override taken from {ref}")
    arms = [("ship", None, {}), ("audit", ref, {})]
    arms.append(("audit-nosr", ref, {"FLAGGEMS_HYGON_TOPK_SCRATCH_REUSE": "0"}))
    occupancy("before")
    results = {tag: [] for tag, _, _ in arms}
    try:
        for p in range(PASSES):
            for tag, r, env in arms:
                install(r)
                print(f"### pass {p + 1}, arm {tag}", flush=True)
                results[tag].append(run_bench(env))
    finally:
        install(None)
        left = sh("git", "status", "--porcelain", "--", *[str(x) for x in PATHS]).stdout
        print(f"### restored; git status on those paths: {left.strip() or 'clean'}")

    shapes = sorted(results["ship"][0], key=lambda k: (k[0] * k[1]))
    g = lambda v: math.exp(sum(map(math.log, v)) / len(v))  # noqa: E731
    print("\nbenchmark SpeedUp, --mode kernel, two interleaved passes\n")
    head = f"  {'num_rows':>9} {'vocab':>7} {'top_k':>6}"
    for tag, _, _ in arms:
        head += f"{tag + ' p1':>13}{tag + ' p2':>13}"
    print(head)
    for k in shapes:
        line = f"  {k[0]:>9} {k[1]:>7} {k[2]:>6}"
        for tag, _, _ in arms:
            for p in range(PASSES):
                line += f"{results[tag][p][k][0]:>13.3f}"
        print(line)
    print()
    for tag, _, _ in arms:
        for p in range(PASSES):
            print(
                f"  geomean {tag:>11} pass {p + 1}: "
                f"{g([results[tag][p][k][0] for k in shapes]):.3f}"
            )
    print("\n  vLLM latency (ms) -- the same kernel on the same input every time")
    hd = f"  {'shape':>14}"
    for tag, _, _ in arms:
        hd += f"{tag + ' p1':>13}{tag + ' p2':>13}"
    print(hd + f"{'max/min':>9}")
    for k in shapes:
        vals = [results[t][p][k][1] for t, _, _ in arms for p in range(PASSES)]
        line = f"  {f'{k[0]}x{k[1]}':>14}"
        for v in vals:
            line += f"{v:>13.4f}"
        print(line + f"{max(vals) / min(vals):>9.2f}")
    occupancy("after")
    print(
        "\n  Any row of the vLLM table whose max/min is not ~1.00 was measured"
        "\n  against a moving baseline; throw that shape out rather than quote it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
