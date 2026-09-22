"""finish's cost is ONE radix round. Find out why, and whether it is needed.

THE ABLATION (tools/hygon_prefill_finish_ablate.py, do_bench, two reps):

    arm          finish us   what it removes
    full             68.2    --
    seqgather        59.5    the scattered gather            ->  8.6 us
    r2               64.5    radix rounds at bits 8 and 0    ->  3.7 us
    noemit           65.5    the "exact ties" emission pass  ->  2.7 us
    r1               22.1    ALSO the round at bits 23-16    -> 42.4 us

**One round costs 42 us. The two rounds after it cost 1.85 us each.** That is
24% of the whole operator in a single pass over 87k candidates.

(Caveat on my own instrument: r1 and r2 leave `thr_key` with zeroed low bits,
so their emission also does less work. `noemit` bounds all of emission at
~5.4 us, so it cannot account for 42.)

WHY ROUND 2 AND NOT ROUND 1. Round 1's mask is `(key & 0) == 0`, so both
rounds fire an atomic for every candidate -- the counts are the same and the
cost differs 5x. What differs is WHERE they land. collect selects on the
11-bit key, so every candidate sits in bin [0, tb] of that key; the top 8 bits
of the 32-bit key are the top 8 of those 11, so if tb < 8 every candidate has
the SAME top byte and round 1 is 1362 atomics into ONE address. Round 2 is the
first round whose addresses spread across the 256 bins, and within a wave 64
distinct addresses are 64 transactions where 64 identical ones combine into
roughly one.

That is a hypothesis about the hardware, so it gets a control rather than a
free pass, and a consequence that can be shipped if it holds:

    flat        every round writes bin 0 -- same atomic count, same shifts and
                masks, ZERO address spread. WRONG. This prices the spread.
    r16         a 4-bit radix: 16 bins, 8 rounds. If the spread is what costs,
                16 addresses per wave should beat 256 even at twice the rounds
                -- and rounds after the first two are nearly free.

AND THE FREE ONE. If every candidate really does share the top byte, round 1
narrows nothing (`lt` is 0, `k_to_find` is unchanged) and simply costs ~8 us:

    noround1    start at bit 16, three rounds
    noround12   start at bit 8, two rounds

These are NOT wrong by construction -- they are wrong only if the common
prefix is shorter than assumed. So every arm is checked against torch.topk on
the values, and the table says which arms answered correctly. An arm that is
both correct and faster is shippable as it stands.

RECOVERY, if killed between the swap and the restore:

    git checkout -- src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py

    tools/vendor_probe.sh tools/hygon_prefill_radix_shape.py hygon_prefill_radix_shape
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
ROUNDS = "    for digit_pos in tl.static_range(24, -1, -8):"
RETRY_ROUNDS = "        for rdpos in tl.static_range(24, -1, -8):"
SRADIX = "SRADIX = 256"

ARMS = {
    "base": [],
    "flat": [(ATOMIC, ATOMIC_FLAT)],
    "noround1": [(ROUNDS, "    for digit_pos in tl.static_range(16, -1, -8):")],
    "noround12": [(ROUNDS, "    for digit_pos in tl.static_range(8, -1, -8):")],
    "r16": [
        (SRADIX, "SRADIX = 16"),
        (ROUNDS, "    for digit_pos in tl.static_range(28, -1, -4):"),
        (RETRY_ROUNDS, "        for rdpos in tl.static_range(28, -1, -4):"),
    ],
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

# correctness on the VALUES: ties make the index set ambiguous, the multiset
# of selected values never is
whole()
torch.cuda.synchronize()
got = torch.gather(logits, 1, out.long().clamp(min=0)).sort(dim=1, descending=True)[0]
ref = torch.topk(logits, top_k, dim=1).values
bad = int((out < 0).sum())
err = float((got - ref).abs().max())
print(
    f"BUDGET {t_p:.1f} {t_pc - t_p:.1f} {t_a - t_pc:.1f} {t_a:.1f}"
    f" {'ok' if err == 0.0 and bad == 0 else 'WRONG'} {err:.3e} {bad}"
    f" {int(plan.thr.max())}"
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
                res[arm].append(line[0].split()[1:])
    finally:
        OVERRIDE.write_text(pristine)
        left = sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
        print(f"### restored; git status: {left.strip() or 'clean'}", flush=True)

    print("\n  64x129280, do_bench, us\n")
    head = ("arm", "prepare", "collect", "finish", "whole", "vs base", "answer")
    print(
        "  " + " ".join(f"{h:>10}" for h in head) + "   max |err|   pads   max thr bin"
    )
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
            + f" {avg[2] / base:>10.3f} {last[4]:>8}"
            + f"   {last[5]:>9}   {last[6]:>4}   {last[7]:>11}"
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
