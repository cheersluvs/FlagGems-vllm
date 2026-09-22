"""The sampled prefill path's SAFETY, through the real benchmark.

WHY. On (64,129280) the audit branch's own v5 report reads

    sampled-cands 4722   ship-dev 295.4   sample-dev 342.8   vllm/sample 0.612

i.e. about 4.6x top_k admitted, and the sampled path 16% SLOWER than the
shipped one. tools/hygon_prefill_sample8.py measured the same shape against
the same baseline (its vllm arm read 209.4 us against their 209.8, so the two
are the same measurement) with TARGET_MULT at 1.5:

    off 293.5  ->  s16/1.5 184.6  = 1.590x,  0.0% of rows outside the window

So the knob is the collected count, and between 4.6x and 1.5x top_k this path
swings 1.85x. That shape sits at 0.419 in the benchmark and drags the geomean
hardest; taking it to ~0.66 moves the whole operator 0.787 -> ~0.84, which is
larger than anything else currently proposed.

Both numbers above come from the profiler, and this card's profiler
DOUBLE-COUNTS the vLLM baseline by 1.88x geomean. So the ratio that decides
this has to come from `benchmark/test_top_k_per_row_prefill.py --mode kernel`,
which is what this runs.

WHAT IS INSTALLED. The sampled path lives in the reverted commit and needs
five fixes before it runs at all -- two gates that each exclude the dense
shapes, and a tl.range/tl.static_range loop-variable clash in `_s_finish` that
meant it never compiled. Those live in tools/hygon_prefill_sample8.fixed_source
with their assertions, and are imported rather than copied.

Its own default gate, `vocab >= 64 * top_k`, already admits (64,129280) and
nothing else in the benchmark, which is exactly the shape under test. The
other six shapes therefore run that file's non-sampled paths, which is what
the `s-off` arm is for: same binary, gate shut. `ship` is the production
override and gives the absolute position, but note that the reverted file
predates the one-scan patch, so `ship` and `s-off` are NOT expected to agree
on the dense shapes -- read the sampled arms against `s-off`.

ARMS (SSTRIDE / TARGET_MULT), each benchmarked twice, interleaved pass by pass:

    ship        production override
    s-off       fixed reverted file, FLAGGEMS_HYGON_PREFILL_SAMPLED_RATIO=0
    s8/1.5      the reverted file's own defaults
    s16/1.5     sample vocab/16; the interleaved probe put this ahead of s8
    s16/1.25    tighter still
    s8/3.0      near where their v5 sat, as a reference point

Correctness is not optional here: decode has just shipped a fix for an 11-bit
fp16 key that COLLAPSES on a narrow band of values, where "an overflow can
only drop what shares the k-th element's key" is true but vacuous. Every
sampled arm runs tests/test_top_k_per_row_prefill.py once before any timing,
and an arm that fails is reported and still timed, so a fast wrong answer
cannot be mistaken for a result.

RECOVERY. The install is undone in a finally block. If the process is killed:

    git checkout -- src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py

    tools/vendor_probe.sh tools/hygon_prefill_sample_bench.py hygon_prefill_sample_bench
"""

import importlib.util
import math
import os
import pathlib
import re
import subprocess
import sys

OVERRIDE = pathlib.Path(
    "src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py"
)
PASSES = 2
BENCH = ["benchmark/test_top_k_per_row_prefill.py", "--mode", "kernel"]
TESTS = ["tests/test_top_k_per_row_prefill.py"]
FOCUS = (64, 129280, 1024)

# (tag, install_fixed, env)
ARMS = [
    ("ship", False, {}),
    ("s-off", True, {"FLAGGEMS_HYGON_PREFILL_SAMPLED_RATIO": "0"}),
    ("s8/1.5", True, {"FLAGGEMS_HYGON_PREFILL_SSTRIDE": "8"}),
    ("s16/1.5", True, {"FLAGGEMS_HYGON_PREFILL_SSTRIDE": "16"}),
    (
        "s16/1.25",
        True,
        {
            "FLAGGEMS_HYGON_PREFILL_SSTRIDE": "16",
            "FLAGGEMS_HYGON_PREFILL_TARGET_MULT": "1.25",
        },
    ),
    (
        "s8/3.0",
        True,
        {
            "FLAGGEMS_HYGON_PREFILL_SSTRIDE": "8",
            "FLAGGEMS_HYGON_PREFILL_TARGET_MULT": "3.0",
        },
    ),
]


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


def install(fixed_src):
    if fixed_src is None:
        r = sh("git", "checkout", "--", str(OVERRIDE))
        assert r.returncode == 0, r.stderr
    else:
        OVERRIDE.write_text(fixed_src)


def run(cmd, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s"] + cmd,
        capture_output=True,
        text=True,
        env=env,
    )


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


def main():
    s8 = load_sample8()
    fixed = s8.fixed_source()
    print(f"### fixed reverted override: {len(fixed.splitlines())} lines")
    dirty = sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
    if dirty.strip():
        raise SystemExit(
            "the override is already modified; restore it first:\n" + dirty
        )
    occupancy("before")
    tests, bench = {}, {tag: [] for tag, _, _ in ARMS}
    try:
        for tag, use_fixed, env in ARMS:
            if not use_fixed:
                continue
            install(fixed)
            r = run(TESTS, env)
            m = re.search(r"(\d+) passed(?:, (\d+) skipped)?", r.stdout)
            fail = re.search(r"(\d+) failed", r.stdout)
            tests[tag] = (m.group(0) if m else "no summary") + (
                f"  [{fail.group(0)}]" if fail else ""
            )
            print(f"### tests {tag}: {tests[tag]}", flush=True)
        for p in range(PASSES):
            for tag, use_fixed, env in ARMS:
                install(fixed if use_fixed else None)
                print(f"### pass {p + 1}, arm {tag}", flush=True)
                r = run(BENCH, env)
                rows = parse_bench(r.stdout)
                if not rows:
                    print(r.stdout[-2500:])
                    raise SystemExit(f"{tag}: the benchmark produced no SUCCESS rows")
                bench[tag].append(rows)
    finally:
        install(None)
        left = sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
        print(f"### restored; git status on the override: {left.strip() or 'clean'}")

    shapes = sorted(bench["ship"][0], key=lambda k: k[0] * k[1])
    g = lambda v: math.exp(sum(map(math.log, v)) / len(v))  # noqa: E731
    print("\nbenchmark SpeedUp, --mode kernel, two interleaved passes\n")
    head = f"  {'num_rows':>9} {'vocab':>7} {'top_k':>6}"
    for tag, _, _ in ARMS:
        head += f"{tag:>11}"
    print(head + "   (pass 1 / pass 2 below each other)")
    for k in shapes:
        mark = "  <== the shape under test" if k == FOCUS else ""
        for p in range(PASSES):
            line = (
                f"  {k[0]:>9} {k[1]:>7} {k[2]:>6}"
                if p == 0
                else f"  {'':>9} {'':>7} {'':>6}"
            )
            for tag, _, _ in ARMS:
                line += f"{bench[tag][p][k][0]:>11.3f}"
            print(line + (mark if p == 0 else ""))
    print()
    for tag, _, _ in ARMS:
        gm = [g([bench[tag][p][k][0] for k in shapes]) for p in range(PASSES)]
        foc = [bench[tag][p][FOCUS][0] for p in range(PASSES)]
        t = tests.get(tag, "not run (production)")
        print(
            f"  {tag:>9}  geomean {gm[0]:.3f} / {gm[1]:.3f}"
            f"   {FOCUS[0]}x{FOCUS[1]} {foc[0]:.3f} / {foc[1]:.3f}   tests: {t}"
        )
    print("\n  vLLM latency on the shape under test (ms), same kernel every time:")
    vals = [bench[t][p][FOCUS][1] for t, _, _ in ARMS for p in range(PASSES)]
    spread = max(vals) / min(vals)
    print("    " + " ".join(f"{v:.4f}" for v in vals) + f"   max/min {spread:.2f}")
    occupancy("after")
    print(
        "\n  Read the sampled arms against s-off, not against ship: the reverted"
        "\n  file predates the one-scan patch, so the two differ on the dense"
        "\n  shapes for reasons that have nothing to do with sampling."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
