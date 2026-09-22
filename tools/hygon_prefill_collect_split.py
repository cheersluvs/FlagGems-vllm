"""Split the sampled collect pass across programs, the way decode splits select.

WHY. Normalising every benchmark shape by its element count says the remaining
gap is one shape and it is ours, not the baseline's:

    shape          vLLM ps/elem   Gems ps/elem
    64x129280          13.2          22.7
    12961x4100         12.8          16.9
    16383x4095         17.2          16.6
    16380x5115         14.9          16.0

Our cost is FLAT at 16.0-16.9 across the three big dense shapes -- and 22.7 on
(64,129280), 40% worse than our own dense figure. (It also says the 0.762 vs
1.042 spread between 12961x4100 and 16383x4095, which looked like our anomaly,
is vLLM varying 34% while we vary 2%.)

The cause is known: 64 workgroups is 512 waves against 3200 slots, 16%
occupancy against roughly 80% on the dense shapes, so nothing covers the
memory latency. num_stages=2 bought 3% of it. The rest needs waves.

WHY THIS IS NOT THE SPLIT THAT WAS REFUTED. Row splitting lost because every
chunk re-ran the whole radix, so the per-program cost did not shrink with the
chunk (249 -> 1029 us as split went 1 -> 32). Stage splitting lost at 0.833
because it split the GENERIC two-pass algorithm's histogram. Here the pipeline
is already the sampled one, prepare and finish stay one program per row, and
only `_s_collect` -- a single pass that compares and appends, with no radix in
it -- is divided. That is exactly what decode's `_select` does, where it was
decisive: 1 row went 0.232 to 1.121 across split 1 to 32.

The cost is the counter: SPLIT programs sharing a row's candidate counter need
a device-scoped atomic where one program needed only a CTA-scoped one. The
`g1` arm is there to price that separately -- one program per row, but
scope="gpu" -- so a win can be attributed to the waves rather than assumed.

ARMS: base (as it ships, CTA scope), then gpu scope at SPLIT 1, 2, 4, 8, 16.
CHUNK is rounded up to a multiple of BLOCK*VEC so the bulk loop stays unmasked
inside a chunk, which is worth 4% on this pass by its own docstring.

Only (64,129280) takes the sampled route, so the other six shapes are a
control: they must not move.

RECOVERY, if killed between the swap and the restore:

    git checkout -- src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py

    tools/vendor_probe.sh tools/hygon_prefill_collect_split.py hygon_prefill_collect_split
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
SPLITS = [1, 2, 4, 8, 16]

LAUNCH_OLD = """        self.collect = _SLaunch(
            _s_collect,
            (num_rows,),
            {"CAP": cap, "BLOCK": SBLOCK, "VEC": 4},
            SWARPS,
        )"""
LAUNCH_NEW = """        schunk = triton.cdiv(triton.cdiv(vocab, SSPLIT), SBLOCK * 4) * SBLOCK * 4
        self.collect = _SLaunch(
            _s_collect,
            (num_rows * SSPLIT,),
            {
                "CAP": cap,
                "BLOCK": SBLOCK,
                "VEC": 4,
                "SPLIT": SSPLIT,
                "CHUNK": schunk,
            },
            SWARPS,
        )"""

COLLECT_NEW = '''@triton.jit
def _s_collect(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    thr_ptr,
    cnt_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    stride0,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
    SPLIT: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """One pass, divided into SPLIT programs per row.

    SPLIT programs share the row's candidate counter, so that atomic is scoped
    to the device; everything else is the single-program version. CHUNK is a
    multiple of BLOCK * VEC so the bulk loop stays unmasked inside a chunk.
    """
    pid = tl.program_id(0)
    row = pid // SPLIT
    part = pid % SPLIT
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    span = e - s
    thr = tl.load(thr_ptr + row)
    base = logits_ptr + row * stride0 + s
    lane = tl.arange(0, BLOCK)
    off = lane[:, None] * VEC + tl.arange(0, VEC)[None, :]
    ones2 = tl.full([BLOCK, VEC], 1, tl.int32)
    ones1 = tl.full([BLOCK], 1, tl.int32)
    cnt2 = cnt_ptr + row + tl.zeros([BLOCK, VEC], tl.int32)
    cnt1 = cnt_ptr + row + tl.zeros([BLOCK], tl.int32)

    start = part * CHUNK
    stop = tl.minimum(start + CHUNK, span)
    have = tl.maximum(stop - start, 0)
    n_vec = have // (BLOCK * VEC)
    for t in tl.range(0, n_vec, num_stages=2):
        i = start + t * BLOCK * VEC + off
        x = tl.load(base + i)
        take = _key11(x).to(tl.int32) < thr
        pos = tl.atomic_add(cnt2, ones2, mask=take, sem="relaxed", scope="gpu")
        keep = take & (pos >= 0) & (pos < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos, i.to(tl.int32), mask=keep)

    tail = start + n_vec * BLOCK * VEC
    for t in tl.range(0, tl.cdiv(tl.maximum(stop - tail, 0), BLOCK)):
        i = tail + t * BLOCK + lane
        m = i < stop
        x = tl.load(base + i, mask=m, other=0.0)
        take = m & (_key11(x).to(tl.int32) < thr)
        pos = tl.atomic_add(cnt1, ones1, mask=take, sem="relaxed", scope="gpu")
        keep = take & (pos >= 0) & (pos < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos, i.to(tl.int32), mask=keep)'''


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


def variant(src, split):
    """split None leaves the file alone."""
    if split is None:
        return src
    a = src.index("@triton.jit\ndef _s_collect(")
    b = src.index("@triton.jit", a + 12)
    src = src[:a] + COLLECT_NEW + "\n\n\n" + src[b:]
    assert src.count(LAUNCH_OLD) == 1, "the collect launch moved"
    src = src.replace(LAUNCH_OLD, LAUNCH_NEW, 1)
    anchor = "SRADIX = 256"
    assert src.count(anchor) == 1
    return src.replace(anchor, f"{anchor}\nSSPLIT = {split}", 1)


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
    dirty = sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
    if dirty.strip():
        raise SystemExit("the override is already modified:\n" + dirty)
    pristine = OVERRIDE.read_text()
    arms = [("base", None)] + [(f"g{n}", n) for n in SPLITS]
    occupancy("before")
    res, broken = {t: [] for t, _ in arms}, {}
    try:
        for p in range(PASSES):
            for tag, sp in arms:
                if tag in broken:
                    continue
                OVERRIDE.write_text(variant(pristine, sp))
                print(f"### pass {p + 1}, arm {tag}", flush=True)
                r = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "-s"] + BENCH,
                    capture_output=True,
                    text=True,
                    env=dict(os.environ),
                )
                rows = parse(r.stdout)
                if not rows:
                    why = [ln for ln in r.stdout.splitlines() if "rror" in ln]
                    broken[tag] = why[-1][:160] if why else "no SUCCESS rows"
                    print(f"      ! {tag}: {broken[tag]}", flush=True)
                    continue
                res[tag].append(rows)
    finally:
        OVERRIDE.write_text(pristine)
        left = sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
        print(f"### restored; git status: {left.strip() or 'clean'}")

    good = [t for t, _ in arms if len(res[t]) == PASSES]
    print(f"\nbenchmark SpeedUp on {FOCUS[0]}x{FOCUS[1]}, two passes\n")
    print(f"  {'arm':>6} {'programs':>9} {'pass 1':>9} {'pass 2':>9} {'vs base':>9}")
    b = sum(res["base"][p][FOCUS][0] for p in range(PASSES)) / PASSES
    for tag, sp in arms:
        if tag not in good:
            continue
        v = [res[tag][p][FOCUS][0] for p in range(PASSES)]
        n = FOCUS[0] * (sp or 1)
        print(
            f"  {tag:>6} {n:>9} {v[0]:>9.3f} {v[1]:>9.3f} {sum(v) / PASSES / b:>9.3f}"
        )
    for t, why in broken.items():
        print(f"  {t:>6}   FAILED: {why}")
    g = lambda v: math.exp(sum(map(math.log, v)) / len(v))  # noqa: E731
    shapes = sorted(res["base"][0])
    print("\n  the other six shapes are a control -- they must not move:")
    for tag in good:
        others = [k for k in shapes if k != FOCUS]
        gm = [g([res[tag][p][k][0] for k in others]) for p in range(PASSES)]
        print(f"    {tag:>6} {gm[0]:.3f} / {gm[1]:.3f}")
    vals = [res[t][p][FOCUS][1] for t in good for p in range(PASSES)]
    print(f"\n  vLLM latency on that shape, max/min {max(vals) / min(vals):.2f}")
    occupancy("after")
    print(
        "\n  g1 is one program per row with a DEVICE-scoped counter: base/g1 is"
        "\n  the price of the scope, and anything beyond it is the waves."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
