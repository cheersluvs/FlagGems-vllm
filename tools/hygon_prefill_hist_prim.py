"""Replace finish's scattered global-atomic histogram with tl.histogram.

WHAT THE LAST PROBE SETTLED, AND WHAT IT KILLED.

    arm          finish us   vs base   answer   max thr bin
    base             67.7      1.000   ok               509
    flat             28.0      0.414   WRONG            509
    noround1         20.4      0.302   WRONG            509
    noround12        21.1      0.312   WRONG            509
    r16              80.7      1.191   ok               509

`flat` -- every round writes bin 0, same atomic count, same shifts and masks,
zero address spread -- is the one controlled number here: **the spread costs
39.7 us, 59% of finish and 23% of the whole operator.**

Everything I built around that is dead. The threshold bin is 509, not under
8, so the candidates do NOT share a top byte: round 1 spreads over ~64 bins
and narrows properly, and both skip-arms answer wrong, as they must. A 4-bit
radix is 19% WORSE -- halving the addresses per wave does not pay for
doubling the rounds, because each round's fixed cost (three tile loads, a
clear, two barriers, a cumsum) is paid again.

I also cannot reconcile the earlier ablation's round-by-round arithmetic with
this: it attributed 42 us to the round at bits 23-16, which fires ~64x fewer
atomics than round 1 once round 1 narrows. Something in that attribution is
wrong -- most likely the compiler folds round 1's mask away (desired_mask is
provably zero there after unrolling) and the arms differ in more than the
round count. **The `flat` control does not depend on that attribution**, so it
is what this probe builds on.

THE ONE THING LEFT TO TRY. Every substitute for the global-atomic histogram
has been refuted ON THE GENERIC OPERATOR, where the histogram covers a whole
129280-element row into 2048 bins: tl.histogram 8.5x slower, shared memory
2.5-3.5x slower, 512 bins slower on six of seven shapes. This is a different
regime and the difference is not small -- here the tile is 512 lanes, n is
about 1362, and there are 256 bins. That is the size tl.histogram exists for,
and it also deletes the per-round clear, the two barriers and the global
`counts` buffer, not just the atomics.

    hist    cnts = sum over tiles of tl.histogram(where(take, digit, RADIX),
            RADIX), accumulated in registers. Lanes that do not match are sent
            to bin RADIX, which is out of range and therefore dropped.

`flat` is carried over as the floor: it is what finish would cost if the
scatter were free.

This is the last idea I have for this shape. If it loses, the answer is that
0.63 is where the sampled path sits on this card and the work goes upstream.

RECOVERY, if killed between the swap and the restore:

    git checkout -- src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py

    tools/vendor_probe.sh tools/hygon_prefill_hist_prim.py hygon_prefill_hist_prim
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
HIST_NEW = """            cnts = tl.zeros([RADIX], tl.int32)
            for t in tl.range(0, tiles):
                pos = t * BLOCK + lane
                valid = pos < n
                key = _key32(tl.load(vbase + pos, mask=valid, other=0.0))
                digit = ((key >> digit_pos) & (RADIX - 1)).to(tl.int32)
                take = valid & ((key & desired_mask) == desired)
                cnts += tl.histogram(tl.where(take, digit, RADIX), RADIX)"""

ARMS = {
    "base": [],
    "flat": [(ATOMIC, ATOMIC_FLAT)],
    "hist": [(HIST_OLD, HIST_NEW)],
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
