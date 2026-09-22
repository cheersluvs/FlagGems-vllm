"""Can the long-row collect pass hide its memory waits?

WHY THIS SHAPE, AND WHY NOW. (64,129280) is the lowest shape on the board at
0.580 and it is occupancy-starved, not atomic-bound: the counters put it at 64
workgroups = 512 waves against 3200 wave slots, 16% occupancy, arch_vgpr 44 --
registers are not the limit, there simply is not enough work. Its waves wait
3.3x longer each than the dense shape's. When you cannot add waves, the lever
is hiding latency inside one, and splitting it to add waves was already
refuted (each chunk re-ran the whole radix, 249 -> 317 us).

What changed is the target. That shape now takes the sampled route, so the
long-row loop is no longer the generic operator's two passes -- it is
`_s_collect`'s ONE unmasked bulk loop, 63 iterations of 2048 elements, and it
is the only place left where a wait can be hidden.

Not pursued, and why: the codegen report for this loop already shows spills=0
and vectorised loads, so "remove the spills" and "get it to vectorise" have
nothing to act on.

ARMS. Only the bulk loop changes; the remainder loop, the keys, the atomics
and the stores are untouched.

    base    as it ships
    ns2/3/4 tl.range(..., num_stages=N), Triton's own software pipelining
    u2      two tiles per iteration with BOTH loads issued before either is
            used -- the hand-rolled version of the same idea, and the one that
            does not depend on the backend honouring a hint

**Whether this Triton honours num_stages on this backend is itself unknown**,
so an arm that fails to compile is reported as unsupported rather than
crashing the run: that is a result too, and it is the thing to check before
designing around the parameter.

Measured through the benchmark, not the profiler. Every arm makes the same
three launches so the gaps cancel and a device-time sum would be sound here --
but the benchmark is what the acceptance table quotes, the shape is stable in
it (0.580 / 0.581 across passes), and there is no reason to introduce a second
instrument for one question.

RECOVERY, if killed between the swap and the restore:

    git checkout -- src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py

    tools/vendor_probe.sh tools/hygon_prefill_pipeline.py hygon_prefill_pipeline
"""

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
FOCUS = (64, 129280, 1024)
LOOP_HEAD = "    for t in tl.range(0, n_vec):"

UNROLL = """    n_pair = n_vec // 2
    for tp in tl.range(0, n_pair):
        i0 = (2 * tp) * BLOCK * VEC + off
        i1 = (2 * tp + 1) * BLOCK * VEC + off
        x0 = tl.load(base + i0)
        x1 = tl.load(base + i1)
        take0 = _key11(x0).to(tl.int32) < thr
        pos0 = tl.atomic_add(cnt2, ones2, mask=take0, sem="relaxed", scope="cta")
        keep0 = take0 & (pos0 >= 0) & (pos0 < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos0, i0.to(tl.int32), mask=keep0)
        take1 = _key11(x1).to(tl.int32) < thr
        pos1 = tl.atomic_add(cnt2, ones2, mask=take1, sem="relaxed", scope="cta")
        keep1 = take1 & (pos1 >= 0) & (pos1 < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos1, i1.to(tl.int32), mask=keep1)
    for t in tl.range(2 * n_pair, n_vec):
"""

ARMS = ["base", "ns2", "ns3", "ns4", "u2"]


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


def variant(src, tag):
    assert src.count(LOOP_HEAD) == 1, "the bulk loop moved; re-read _s_collect"
    if tag == "base":
        return src
    if tag == "u2":
        return src.replace(LOOP_HEAD, UNROLL.rstrip("\n"), 1)
    n = int(tag[2:])
    return src.replace(
        LOOP_HEAD, f"    for t in tl.range(0, n_vec, num_stages={n}):", 1
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


def run_bench():
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s"] + BENCH,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    return parse(r.stdout), r.stdout


def main():
    dirty = sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
    if dirty.strip():
        raise SystemExit("the override is already modified:\n" + dirty)
    pristine = OVERRIDE.read_text()
    occupancy("before")
    res, unsupported = {t: [] for t in ARMS}, {}
    try:
        for p in range(PASSES):
            for tag in ARMS:
                if tag in unsupported:
                    continue
                OVERRIDE.write_text(variant(pristine, tag))
                print(f"### pass {p + 1}, arm {tag}", flush=True)
                rows, out = run_bench()
                if not rows:
                    why = [
                        ln
                        for ln in out.splitlines()
                        if "Error" in ln or "error" in ln or "num_stages" in ln
                    ]
                    unsupported[tag] = why[-1][:160] if why else "no SUCCESS rows"
                    print(f"      ! {tag} unsupported: {unsupported[tag]}", flush=True)
                    continue
                res[tag].append(rows)
    finally:
        OVERRIDE.write_text(pristine)
        left = sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
        print(f"### restored; git status: {left.strip() or 'clean'}")

    good = [t for t in ARMS if len(res[t]) == PASSES]
    print(f"\nbenchmark SpeedUp on {FOCUS[0]}x{FOCUS[1]}, two passes\n")
    print(f"  {'arm':>6} {'pass 1':>9} {'pass 2':>9} {'vs base':>9}")
    b = sum(res["base"][p][FOCUS][0] for p in range(PASSES)) / PASSES
    for t in good:
        v = [res[t][p][FOCUS][0] for p in range(PASSES)]
        print(f"  {t:>6} {v[0]:>9.3f} {v[1]:>9.3f} {sum(v) / PASSES / b:>9.3f}")
    for t, why in unsupported.items():
        print(f"  {t:>6}      unsupported: {why}")
    g = lambda v: math.exp(sum(map(math.log, v)) / len(v))  # noqa: E731
    shapes = sorted(res["base"][0])
    print("\n  whole suite, to confirm the other six shapes do not move:")
    for t in good:
        gm = [g([res[t][p][k][0] for k in shapes]) for p in range(PASSES)]
        print(f"    {t:>6} {gm[0]:.3f} / {gm[1]:.3f}")
    vals = [res[t][p][FOCUS][1] for t in good for p in range(PASSES)]
    print(f"\n  vLLM latency on that shape, max/min {max(vals) / min(vals):.2f}")
    occupancy("after")
    print(
        "\n  A flat table means the waits are not where this looks, and the"
        "\n  shape's 16% occupancy has to be attacked some other way."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
