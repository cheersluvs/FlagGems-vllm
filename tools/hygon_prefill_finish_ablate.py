"""Localise finish's 68 us by removing one phase at a time.

WHY AN ABLATION AND NOT ANOTHER HYPOTHESIS. Three in a row have now been
refuted on this shape, the last two decisively:

    collect stores the value too        0.968  -- the gather it removes is
                                               cheap (collect appends in row
                                               order, so consecutive candidate
                                               slots come from nearby row
                                               positions and the "gather" is a
                                               strided stream, not 1362
                                               independent lines); the store it
                                               adds is on collect's critical
                                               path, which is 46% of the time
    finish BLOCK 512 / 1024 / 2048      0.994 / 0.995 / 0.991
    finish warps 8 / 16                 flat
    collect warps 8 / 16                0.994
    prepare BLOCK 1024 / 16 warps       0.284  -- see below, it is not a
                                                 performance knob at all

finish is insensitive to BOTH the lane count and the tile count. That rules
out every throughput explanation -- if it were bound by atomic throughput or
by memory parallelism, more lanes would move it -- and leaves a serial chain:
four radix rounds, each a clear, a barrier, an atomic histogram, a barrier and
a 256-wide cumsum, then two emission passes, ten barriers in all. I do not
know which link costs the 68 us and I am done guessing. This measures it.

ALSO LEARNED, and worth more than the sweep: a program on this card is capped
at 1024 threads, so 16 warps is the maximum and 64 programs can reach 1024 of
3200 wave slots -- 32%, not the 64% I assumed when I proposed 32 warps.

AND: prepare's BLOCK is a STATISTICAL parameter, not a geometry knob. _s_hist
samples the first BLOCK of every BLOCK * STRIDE window, so changing BLOCK
changes WHICH elements are sampled. At BLOCK 1024 the sample is twice as
clustered, the threshold estimate degrades, and one row of 64 fell under
TOPK -- one row, 1.6% -- which took the full-row retry and turned finish from
68.5 us into 506.6. prepare itself got FASTER (25.7 -> 22.3) and the whole
thing went to 0.284. That is the price of a single straggler, measured.

ARMS. Every arm is WRONG BY CONSTRUCTION except `full`; they exist to be
timed, not to be right, so this probe runs do_bench only and never the
benchmark. prepare and collect are printed as a control: they must not move.

    full        as shipped
    seqgather   the gather reads row_base + pos instead of row_base + ci --
                sequential where the real one is scattered. This also drops
                the dependency on the index load, so it is an UPPER bound on
                what the scatter costs.
    r1          one radix round instead of four
    r2          two
    noemit      the emission runs its "strictly better" pass but not its
                "exact ties" pass

RECOVERY, if killed between the swap and the restore:

    git checkout -- src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py

    tools/vendor_probe.sh tools/hygon_prefill_finish_ablate.py hygon_prefill_finish_ablate
"""

import os
import pathlib
import subprocess
import sys

OVERRIDE = pathlib.Path(
    "src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py"
)
REPS = 2

GATHER = (
    "        tl.store(vbase + pos, tl.load(row_base + ci, mask=valid, other=0.0),"
    " mask=valid)"
)
GATHER_SEQ = (
    "        tl.store(vbase + pos, tl.load(row_base + pos, mask=valid, other=0.0),"
    " mask=valid)"
)
ROUNDS = "    for digit_pos in tl.static_range(24, -1, -8):"
EMIT = "    for equal in tl.static_range(2):"

ARMS = {
    "full": [],
    "seqgather": [(GATHER, GATHER_SEQ)],
    "r1": [(ROUNDS, "    for digit_pos in tl.static_range(24, 16, -8):")],
    "r2": [(ROUNDS, "    for digit_pos in tl.static_range(24, 8, -8):")],
    "noemit": [(EMIT, "    for equal in tl.static_range(1):")],
}

BUDGET = r"""
import torch, triton
from importlib import import_module

M = "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
m = import_module(M)
dev = "cuda"
num_rows, vocab, top_k, stride0, stride1 = 64, 129280, 1024, 129280, 1

torch.manual_seed(42)
buf = torch.randn(
    (num_rows - 1) * stride0 + (vocab - 1) * stride1 + 1, device=dev,
    dtype=torch.float32,
)
logits = torch.as_strided(buf, (num_rows, vocab), (stride0, stride1))
assert logits.stride(0) == stride0 and logits.stride(1) == stride1
starts = torch.zeros(num_rows, dtype=torch.int32, device=dev)
ends = torch.full((num_rows,), vocab, dtype=torch.int32, device=dev)
out = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)

assert m._can_sample(
    logits, starts, ends, num_rows, stride0, stride1, top_k
), "this shape is not routed to the sampled path"
plan = m._SPlan(logits.device, logits.dtype, num_rows, vocab, top_k)


def prep():
    plan.prepare(logits, starts, ends, plan.hist, plan.thr, plan.cnt, stride0)


def prep_coll():
    prep()
    plan.collect(
        logits, starts, ends, plan.thr, plan.cnt, plan.cand_idx, plan.cand_val,
        stride0,
    )


def whole():
    plan.run(logits, starts, ends, out, stride0)


b = triton.testing.do_bench
t_p = b(prep, warmup=100, rep=300) * 1e3
t_pc = b(prep_coll, warmup=100, rep=300) * 1e3
t_a = b(whole, warmup=100, rep=300) * 1e3

whole()
torch.cuda.synchronize()
cnt = plan.cnt.to(torch.int64)
print(
    f"BUDGET {t_p:.1f} {t_pc - t_p:.1f} {t_a - t_pc:.1f} {t_a:.1f}"
    f" {int(cnt.min())} {int((cnt < top_k).sum())}"
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


def variant(src, arm):
    for old, new in ARMS[arm]:
        assert src.count(old) == 1, f"{arm}: anchor moved: {old.strip()[:60]!r}"
        src = src.replace(old, new, 1)
    return src


def main():
    dirty = sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
    if dirty.strip():
        raise SystemExit("the override is already modified:\n" + dirty)
    pristine = OVERRIDE.read_text()
    for arm in ARMS:  # fail before touching the card if an anchor moved
        variant(pristine, arm)
    occupancy("before")

    res, broken = {a: [] for a in ARMS}, {}
    try:
        for r in range(REPS):
            for arm in ARMS:
                if arm in broken:
                    continue
                OVERRIDE.write_text(variant(pristine, arm))
                print(f"### rep {r + 1}, arm {arm}", flush=True)
                out = subprocess.run(
                    [sys.executable, "-c", BUDGET],
                    capture_output=True,
                    text=True,
                    env=dict(os.environ),
                )
                line = [x for x in out.stdout.splitlines() if x.startswith("BUDGET")]
                if not line:
                    broken[arm] = "\n".join(out.stderr.strip().splitlines()[-3:])[:300]
                    print(f"      ! {arm}: {broken[arm]}", flush=True)
                    continue
                res[arm].append([float(x) for x in line[0].split()[1:]])
    finally:
        OVERRIDE.write_text(pristine)
        left = sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
        print(f"### restored; git status: {left.strip() or 'clean'}", flush=True)

    print("\n  64x129280, do_bench, us -- EVERY ARM BUT `full` IS WRONG BY")
    print("  CONSTRUCTION; the numbers are for attribution only\n")
    head = ("arm", "prepare", "collect", "finish", "whole", "vs full", "saved")
    print("  " + " ".join(f"{h:>10}" for h in head))
    base = None
    for arm in ARMS:
        if len(res[arm]) != REPS:
            print(f"  {arm:>10}   FAILED: {broken.get(arm, 'incomplete')}")
            continue
        avg = [sum(c) / REPS for c in zip(*res[arm])]
        if base is None:
            base = avg[2]
        print(
            "  "
            + " ".join(f"{x:>10.1f}" for x in avg[:4])
            + f" {avg[2] / base:>10.3f} {base - avg[2]:>10.1f}"
        )
    print("\n  per-rep finish, to show the spread:")
    for arm in ARMS:
        if len(res[arm]) == REPS:
            print(f"      {arm:>10}: " + "  ".join(f"{v[2]:.1f}" for v in res[arm]))
    occupancy("after")


if __name__ == "__main__":
    main()
