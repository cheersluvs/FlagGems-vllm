"""tl.histogram IS faster here. Find out what my masking broke, and fix it.

    arm     finish us   whole    vs base   answer   max |err|
    base         67.0   171.8      1.000   ok       0.000e+00
    flat         28.0   132.7      0.419   WRONG    7.401e+00
    hist         50.6   155.8      0.757   WRONG    5.963e+00

**The primitive wins by 25% on finish in this regime** -- 16.4 us, 9.5% of the
whole operator -- which settles that the earlier "tl.histogram is 8.5x slower"
verdict was about the OTHER regime (a whole 129280-element row into 2048 bins
on the generic operator) and does not carry to a 512-lane tile and 256 bins.

But the arm answers wrong, so the number is not yet a result. The suspect is
the one thing I invented rather than copied:

    cnts += tl.histogram(tl.where(take, digit, RADIX), RADIX)

I assumed a value of RADIX -- one past the last bin -- is DROPPED. Triton
documents tl.histogram over [0, num_bins); what it does with anything outside
that is not something I should be assuming on a vendor backend. If those lanes
are clamped into bin RADIX-1, or wrapped to 0, every round's counts are wrong
by the number of non-matching lanes, which is most of them after round 1.

PART A -- WHAT DOES IT ACTUALLY DO. A ten-line kernel over a known input
containing in-range values, num_bins, num_bins+1, a large value and a negative
one, against numpy's bincount of just the in-range part. Dropped, clamped and
wrapped each leave a different signature, printed side by side.

PART B -- A FORMULATION THAT DOES NOT NEED TO KNOW. Send every non-matching
lane to bin 0, which certainly exists, then subtract exactly how many were
sent there:

    h = tl.histogram(tl.where(take, digit, 0), RADIX)
    nmiss = BLOCK - tl.sum(take.to(tl.int32), axis=0)
    cnts += h - tl.where(bins == 0, nmiss, 0)

`hist` is kept as the reference point so the fix can be attributed, and
`flat` as the floor -- what finish would cost if the scatter were free.

RECOVERY, if killed between the swap and the restore:

    git checkout -- src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py

    tools/vendor_probe.sh tools/hygon_prefill_hist_fix.py hygon_prefill_hist_fix
"""

import os
import pathlib
import subprocess
import sys

OVERRIDE = pathlib.Path(
    "src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py"
)
REPS = 2

ATOMIC = """                tl.atomic_add(
                    cbase + digit,"""
ATOMIC_FLAT = """                tl.atomic_add(
                    cbase + digit * 0,"""

HIST_OLD = """            tl.store(cbase + bins, tl.zeros([RADIX], tl.int32))
            tl.debug_barrier()
            for t in tl.range(0, tiles):
                pos = t * BLOCK + lane
                valid = pos < n
                key = _key32(tl.load(vbase + pos, mask=valid, other=0.0))
                digit = ((key >> digit_pos) & (RADIX - 1)).to(tl.int32)
                tl.atomic_add(
                    cbase + digit,
                    ones,
                    mask=valid & ((key & desired_mask) == desired),
                    sem="relaxed",
                    scope="cta",
                )
            tl.debug_barrier()
            cnts = tl.load(cbase + bins)"""

_HEAD = """            cnts = tl.zeros([RADIX], tl.int32)
            for t in tl.range(0, tiles):
                pos = t * BLOCK + lane
                valid = pos < n
                key = _key32(tl.load(vbase + pos, mask=valid, other=0.0))
                digit = ((key >> digit_pos) & (RADIX - 1)).to(tl.int32)
                take = valid & ((key & desired_mask) == desired)
"""
HIST_SENTINEL = _HEAD + (
    "                cnts += tl.histogram(tl.where(take, digit, RADIX), RADIX)"
)
HIST_BIN0 = (
    _HEAD
    + """                h = tl.histogram(tl.where(take, digit, 0), RADIX)
                nmiss = BLOCK - tl.sum(take.to(tl.int32), axis=0)
                cnts += h - tl.where(bins == 0, nmiss, 0)"""
)

ARMS = {
    "base": [],
    "flat": [(ATOMIC, ATOMIC_FLAT)],
    "hist": [(HIST_OLD, HIST_SENTINEL)],
    "histbin0": [(HIST_OLD, HIST_BIN0)],
}

SEMANTICS = r"""
import numpy as np, torch, triton
import triton.language as tl

N, NB = 128, 8


@triton.jit
def _h(out_ptr, x_ptr, N: tl.constexpr, NB: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, N))
    tl.store(out_ptr + tl.arange(0, NB), tl.histogram(x, NB))


# 0..7 five times each, then NB, NB+1, a big value and a negative one, padded
# with 0s so the padding lands in a bin we can account for exactly
vals = list(range(NB)) * 5 + [NB, NB + 1, 100000, -1, -NB]
vals = vals + [0] * (N - len(vals))
x = torch.tensor(vals, dtype=torch.int32, device="cuda")
out = torch.zeros(NB, dtype=torch.int32, device="cuda")
_h[(1,)](out, x, N=N, NB=NB, num_warps=4)
torch.cuda.synchronize()
got = out.cpu().numpy().tolist()

a = np.array(vals)
inrange = np.bincount(a[(a >= 0) & (a < NB)], minlength=NB).tolist()
clamped = np.bincount(np.clip(a, 0, NB - 1), minlength=NB).tolist()
wrapped = np.bincount(np.mod(a, NB), minlength=NB).tolist()
print("  tl.histogram semantics, N=128 values into 8 bins")
print(f"      got               {got}")
print(f"      if out-of-range DROPPED  {inrange}   <- {got == inrange}")
print(f"      if CLAMPED               {clamped}   <- {got == clamped}")
print(f"      if WRAPPED (mod)         {wrapped}   <- {got == wrapped}")
print(f"      total counted {sum(got)} of {N} lanes")
"""

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
got = torch.gather(logits, 1, out.long().clamp(min=0)).sort(dim=1, descending=True)[0]
ref = torch.topk(logits, top_k, dim=1).values
bad = int((out < 0).sum())
err = float((got - ref).abs().max())
print(
    f"BUDGET {t_p:.1f} {t_pc - t_p:.1f} {t_a - t_pc:.1f} {t_a:.1f}"
    f" {'ok' if err == 0.0 and bad == 0 else 'WRONG'} {err:.3e} {bad}"
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
    for arm in ARMS:
        variant(pristine, arm)
    occupancy("before")

    print("### part A: what tl.histogram does with out-of-range values", flush=True)
    r = subprocess.run(
        [sys.executable, "-c", SEMANTICS],
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    print(r.stdout or "", flush=True)
    if r.returncode:
        print("  ! the semantics child failed:", flush=True)
        print("\n".join(r.stderr.strip().splitlines()[-8:]), flush=True)

    res, broken = {a: [] for a in ARMS}, {}
    try:
        for rep in range(REPS):
            for arm in ARMS:
                if arm in broken:
                    continue
                OVERRIDE.write_text(variant(pristine, arm))
                print(f"### part B rep {rep + 1}, arm {arm}", flush=True)
                out = subprocess.run(
                    [sys.executable, "-c", BUDGET],
                    capture_output=True,
                    text=True,
                    env=dict(os.environ),
                )
                line = [x for x in out.stdout.splitlines() if x.startswith("BUDGET")]
                if not line:
                    broken[arm] = "\n".join(out.stderr.strip().splitlines()[-4:])[:400]
                    print(f"      ! {arm}: {broken[arm]}", flush=True)
                    continue
                res[arm].append(line[0].split()[1:])
    finally:
        OVERRIDE.write_text(pristine)
        left = sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
        print(f"### restored; git status: {left.strip() or 'clean'}", flush=True)

    print("\n  64x129280, do_bench, us\n")
    head = ("arm", "prepare", "collect", "finish", "whole", "vs base", "answer")
    print("  " + " ".join(f"{h:>10}" for h in head) + "   max |err|")
    base = None
    for arm in ARMS:
        if len(res[arm]) != REPS:
            print(f"  {arm:>10}   FAILED: {broken.get(arm, 'incomplete')}")
            continue
        avg = [sum(float(v[i]) for v in res[arm]) / REPS for i in range(4)]
        if base is None:
            base = avg[2]
        last = res[arm][-1]
        print(
            f"  {arm:>10} "
            + " ".join(f"{x:>10.1f}" for x in avg)
            + f" {avg[2] / base:>10.3f} {last[4]:>8}   {last[5]:>9}"
        )
    print("\n  per-rep finish, to show the spread:")
    for arm in ARMS:
        if len(res[arm]) == REPS:
            print(
                f"      {arm:>10}: " + "  ".join(f"{float(v[2]):.1f}" for v in res[arm])
            )
    occupancy("after")


if __name__ == "__main__":
    main()
