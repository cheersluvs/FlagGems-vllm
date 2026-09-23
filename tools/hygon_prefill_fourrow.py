"""Route the two 4-row prefill shapes to the sampled path, with a wider split.

WHY. After bae5a98 the prefill shapes still below 0.9 are 12961x4100 (vLLM
being fast there, per element) and the two 4-row shapes, 4x8193 and
4x16385 at 0.61-0.71. Those two run the generic operator at one program per
row -- 4 programs on an 80-CU card -- so they are a latency problem, and a
latency problem is what decode's recipe answered (sampled threshold + a split
collect: 1 row went 0.232 -> 1.121). The sampled path already has that
recipe; it is kept off these shapes by one gate, vocab >= 64 * top_k, and
their ratios are 16 and 32.

The evidence that the sampled path LOSES on small ratios came from the
many-row dense shapes (ratio 8-10, 0.17-0.33), where the generic operator is
well occupied. It says nothing about four rows. And the reason a wide split
could not be tried before -- one shared counter per row, which saturated as
the split grew -- is gone since bae5a98.

EVERY KNOB IS ALREADY AN ENVIRONMENT VARIABLE, so no source is patched:

    FLAGGEMS_HYGON_PREFILL_SAMPLED_RATIO   64 -> 16 routes EXACTLY the two
        4-row shapes (8193 >= 16 * 512 by one element); every dense shape
        stays below it
    FLAGGEMS_HYGON_PREFILL_SSPLIT          programs per row in collect
    FLAGGEMS_HYGON_PREFILL_SSTRIDE         1 / sample fraction

TWO LIMITS WORTH KNOWING BEFORE READING THE TABLE.

  1. CHUNK is a multiple of BLOCK * VEC = 2048 (it keeps the bulk loop
     unmasked), so a 8193-long row gives work to at most 5 programs whatever
     SPLIT says, and 16385 to at most 9 (at SPLIT 16). Decode's split of 32 is
     not reachable here without changing the collect geometry.
  2. prepare samples the first 512 of every 512 * STRIDE window, so at the
     shipped STRIDE 16 a 8193 row yields 513 samples and a target of 40 --
     roughly 16% relative error on the threshold, against a 25% margin. At
     STRIDE 4 it is 2049 samples and a target of 160.

SSPLIT and SSTRIDE are GLOBAL, so every arm that moves them also moves
(64,129280), which is shipped at 4 / 16. Its numbers are printed and are
collateral, not a verdict: a landing would make both per-shape.

ARMS
    base          shipped: 4-row shapes on the generic operator
    s4st16        sampled, SPLIT 4, STRIDE 16   (64x129280 untouched)
    s4st4         sampled, SPLIT 4, STRIDE 4
    s8st4         sampled, SPLIT 8, STRIDE 4    (SPLIT 8 was bad on 64x129280)
    s16st4        sampled, SPLIT 16, STRIDE 4   (16385 gets 9 busy programs)

THESE SHAPES ARE BIMODAL -- the same binary has swung 10-13% between passes.
So: FOUR benchmark passes, the arm order rotated each pass so no arm always
runs first or last, the median reported with min-max, and vLLM's latency on
each 4-row shape printed per pass, so the bimodality can be assigned to the
baseline or to us. A child also times our op alone with do_bench, which does
not involve vLLM at all.

Correctness per arm: normal, narrow-band (forces the retry) and partial rows
on both 4-row shapes against torch.topk, then the full prefill suite.

    tools/vendor_probe.sh tools/hygon_prefill_fourrow.py hygon_prefill_fourrow
"""

import os
import pathlib
import re
import statistics
import subprocess
import sys

PASSES = 4
BENCH = ["benchmark/test_top_k_per_row_prefill.py", "--mode", "kernel"]
TESTS = ["tests/test_top_k_per_row_prefill.py"]
FOUR = [(4, 8193, 512), (4, 16385, 512)]
LONG = (64, 129280, 1024)

# tag, sampled ratio, SPLIT, STRIDE
ARMS = [
    ("base", 64, 4, 16),
    ("s4st16", 16, 4, 16),
    ("s4st4", 16, 4, 4),
    ("s8st4", 16, 8, 4),
    ("s16st4", 16, 16, 4),
]

CHILD = r"""
import torch, triton
from importlib import import_module

M = "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
m = import_module(M)
dev = "cuda"


def inputs(num_rows, vocab, stride0, kind):
    torch.manual_seed(42)
    buf = torch.randn((num_rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
    if kind == "band":
        buf = 10.0 + 0.2 * torch.rand_like(buf)
    x = torch.as_strided(buf, (num_rows, vocab), (stride0, 1))
    assert x.stride(0) == stride0 and x.stride(1) == 1
    if kind == "partial":
        g = torch.Generator(device="cpu").manual_seed(7)
        st = torch.randint(0, 500, (num_rows,), generator=g).to(torch.int32)
        en = (vocab - torch.randint(0, 500, (num_rows,), generator=g)).to(torch.int32)
    else:
        st = torch.zeros(num_rows, dtype=torch.int32)
        en = torch.full((num_rows,), vocab, dtype=torch.int32)
    return x, st.to(dev), en.to(dev)


def check(num_rows, vocab, top_k, stride0, kind):
    x, st, en = inputs(num_rows, vocab, stride0, kind)
    out = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
    m.top_k_per_row_prefill(x, st, en, out, num_rows, stride0, 1, top_k)
    torch.cuda.synchronize()
    col = torch.arange(vocab, device=dev)[None, :]
    inside = (col >= st[:, None].long()) & (col < en[:, None].long())
    ref = torch.topk(x.masked_fill(~inside, float("-inf")), top_k, dim=1).values
    pads = int((out < 0).sum())
    got = torch.gather(x, 1, st[:, None].long() + out.long().clamp(min=0))
    err = float((got.sort(dim=1, descending=True)[0] - ref).abs().max())
    return "ok" if err == 0.0 and pads == 0 else f"WRONG({err:.1e},{pads})"


def bench(fn):
    ts = [triton.testing.do_bench(fn, warmup=50, rep=200) * 1e3 for _ in range(3)]
    return min(ts)


for num_rows, vocab, top_k, stride0 in ((4, 8193, 512, 8456), (4, 16385, 512, 16648)):
    x, st, en = inputs(num_rows, vocab, stride0, "normal")
    out = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
    routed = m._can_sample(x, st, en, num_rows, stride0, 1, top_k)
    checks = [check(num_rows, vocab, top_k, stride0, k) for k in ("normal", "band", "partial")]
    whole = bench(
        lambda: m.top_k_per_row_prefill(x, st, en, out, num_rows, stride0, 1, top_k)
    )
    extra = "prep=- coll=- fin=- cand=- retry=-"
    if routed:
        plan = m._SPlan(x.device, x.dtype, num_rows, vocab, top_k)

        def prep():
            plan.prepare(x, st, en, plan.hist, plan.thr, plan.cnt, stride0)

        def prep_coll():
            prep()
            plan.collect(
                x, st, en, plan.thr, plan.cnt, plan.cand_idx, plan.cand_val, stride0
            )

        t_p = bench(prep)
        t_pc = bench(prep_coll)
        t_a = bench(lambda: plan.run(x, st, en, out, stride0))
        plan.run(x, st, en, out, stride0)
        torch.cuda.synchronize()
        seg = plan.cnt.view(num_rows, m.SSPLIT).to(torch.int64)
        segcap = plan.cap // m.SSPLIT
        c = seg.clamp(max=segcap).sum(1)
        retry = int(((c < top_k) | (seg > segcap).any(1)).sum())
        extra = (
            f"prep={t_p:.1f} coll={t_pc - t_p:.1f} fin={t_a - t_pc:.1f}"
            f" cand={int(c.min())}-{int(c.max())} retry={retry}/{num_rows}"
        )
    print(
        f"CHILD {num_rows}x{vocab} route={'sampled' if routed else 'generic'}"
        f" whole={whole:.1f} {extra} checks={'/'.join(checks)}"
        f" split={m.SSPLIT} stride={m.SSTRIDE}"
    )
"""


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


def env_for(arm):
    e = dict(os.environ)
    _, ratio, split, stride = arm
    e["FLAGGEMS_HYGON_PREFILL_SAMPLED_RATIO"] = str(ratio)
    e["FLAGGEMS_HYGON_PREFILL_SSPLIT"] = str(split)
    e["FLAGGEMS_HYGON_PREFILL_SSTRIDE"] = str(stride)
    return e


def parse_bench(out):
    rows = {}
    for mm in re.finditer(
        r"SUCCESS\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[torch\.Size\(\[(\d+), (\d+)\]\)"
        r".*?, (\d+), (\d+), 1, (\d+)\]",
        out,
    ):
        rows[(int(mm.group(4)), int(mm.group(5)), int(mm.group(8)))] = (
            float(mm.group(3)),
            float(mm.group(1)),
        )
    return rows


def parse_tests(out):
    p = re.search(r"(\d+) passed", out)
    k = re.search(r"(\d+) skipped", out)
    f = re.search(r"(\d+) failed", out)
    if not (p or f):
        return "no result"
    s = f"{p.group(1) if p else 0} passed"
    if k:
        s += f", {k.group(1)} skipped"
    if f:
        s += f", {f.group(1)} FAILED"
    return s


def why(r):
    text = (r.stdout + r.stderr).splitlines()
    hits = [
        ln.strip()
        for ln in text
        if ("Error" in ln or "error:" in ln) and "error_msg = str(e)" not in ln
    ]
    return (hits[-1] if hits else "no output")[:220], text[-10:]


def geo(vals):
    g = 1.0
    for v in vals:
        g *= v
    return g ** (1.0 / len(vals))


def main():
    dirty = sh("git", "status", "--porcelain", "--", "src").stdout
    if dirty.strip():
        raise SystemExit("the source tree is modified:\n" + dirty)
    occupancy("before")

    tests, bench, broken = {}, {a[0]: [] for a in ARMS}, {}
    for arm in ARMS:
        tag = arm[0]
        print(f"### child (routing, checks, do_bench), arm {tag}", flush=True)
        r = subprocess.run(
            [sys.executable, "-c", CHILD],
            capture_output=True,
            text=True,
            env=env_for(arm),
        )
        lines = [x for x in r.stdout.splitlines() if x.startswith("CHILD")]
        if len(lines) != len(FOUR):
            broken[tag], tail = why(r)
            print(f"      ! {tag}: {broken[tag]}", flush=True)
            for ln in tail:
                print(f"        | {ln[:200]}", flush=True)
            continue
        for ln in lines:
            print(f"      {ln[6:]}", flush=True)

        print(f"### tests, arm {tag}", flush=True)
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-rf"] + TESTS,
            capture_output=True,
            text=True,
            env=env_for(arm),
        )
        tests[tag] = parse_tests(r.stdout)
        print(f"      {tests[tag]}", flush=True)
        for ln in [x for x in r.stdout.splitlines() if x.startswith("FAILED")][:3]:
            print(f"        {ln[:220]}", flush=True)

    live = [a for a in ARMS if a[0] not in broken]
    for p in range(PASSES):
        order = live[p % len(live) :] + live[: p % len(live)]
        for arm in order:
            tag = arm[0]
            if tag in broken:
                continue
            print(f"### benchmark pass {p + 1}, arm {tag}", flush=True)
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-s"] + BENCH,
                capture_output=True,
                text=True,
                env=env_for(arm),
            )
            rows = parse_bench(r.stdout)
            if not rows:
                broken[tag], tail = why(r)
                print(f"      ! {tag}: {broken[tag]}", flush=True)
                for ln in tail:
                    print(f"        | {ln[:200]}", flush=True)
                continue
            bench[tag].append(rows)

    good = [a[0] for a in ARMS if len(bench[a[0]]) == PASSES]
    for shape in FOUR + [LONG]:
        label = f"{shape[0]}x{shape[1]}"
        note = "  (collateral: SPLIT/STRIDE are global)" if shape == LONG else ""
        print(f"\n  {label}, benchmark SpeedUp over {PASSES} passes{note}\n")
        print(f"  {'arm':>7}  {'median':>7}  {'min':>6}  {'max':>6}   per pass")
        base_med = None
        for arm in ARMS:
            tag = arm[0]
            if tag not in good:
                print(f"  {tag:>7}   FAILED: {broken.get(tag, 'incomplete')}")
                continue
            v = [bench[tag][p][shape][0] for p in range(PASSES)]
            med = statistics.median(v)
            if tag == "base":
                base_med = med
            rel = f"  x{med / base_med:.3f}" if base_med else ""
            print(
                f"  {tag:>7}  {med:7.3f}  {min(v):6.3f}  {max(v):6.3f}   "
                + " ".join(f"{x:.3f}" for x in v)
                + rel
            )
        lat = [bench[t][p][shape][1] for t in good for p in range(PASSES)]
        if lat:
            print(
                f"  vLLM latency (ms) on {label}, every run: "
                + " ".join(f"{x:.4f}" for x in lat)
                + f"   max/min {max(lat) / min(lat):.2f}"
            )

    print("\n  suite geomean, median over passes (all 7 / stable 5):")
    for tag in good:
        g7, g5 = [], []
        for p in range(PASSES):
            rows = bench[tag][p]
            g7.append(geo([v[0] for v in rows.values()]))
            g5.append(geo([v[0] for k, v in rows.items() if k[0] != 4]))
        print(
            f"      {tag:>7}: {statistics.median(g7):.3f} / {statistics.median(g5):.3f}"
            f"   tests {tests.get(tag, '-')}"
        )
    occupancy("after")


if __name__ == "__main__":
    main()
