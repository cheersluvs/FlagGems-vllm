"""The three sampled launches share one geometry, and it was never swept.

THE BUDGET SAYS THE MONEY IS NOT WHERE THE BYTES ARE
(tools/hygon_prefill_value_store.py, do_bench on our own kernels):

                us     share   bytes touched        GB/s
    prepare   25.8     14.9%   2.07 MB sampled        80
    collect   79.1     45.7%   33.1 MB sequential    419
    finish    68.2     39.4%   0.35 MB of candidates   -
    whole    173.1             (vLLM does the job in 109.4)

finish spends 39% of the time on 0.7% of the elements, and prepare gets 80
GB/s where collect gets 419 -- neither is bandwidth-bound. Both are bound by
global atomics under barriers at 16% occupancy: prepare fires ~517k atomics
into 2048 bins, finish makes SEVEN passes over the candidate buffer (one
gather, four radix rounds, two emission) with ~349k atomics into 256 bins and
ten barriers between them.

All three launch at BLOCK 512 / 8 warps because all three read SBLOCK and
SWARPS, which were chosen for collect. Nothing about finish's 1362 candidates
or prepare's 2048-bin histogram says 512 lanes is right for them. At BLOCK
512 / 8 warps a program is 8 waves and 64 programs is 512 of 3200 slots, 16%;
the same work at 32 warps would be 2048 waves, 64%, WITHOUT adding programs --
which matters because the serial chain of barriers inside finish is exactly
what cannot be split across programs.

ARMS. Each changes ONE launch's geometry; f1024w8 is there to separate the
tile count from the waves, the way g1 separated atomic scope from waves in the
collect split.

    base       as shipped, all three at 512 / 8
    f1024w8    finish BLOCK 1024, 8 warps   -- tiles 3 -> 2, waves unchanged
    f1024w16   finish BLOCK 1024, 16 warps
    f2048w16   finish BLOCK 2048, 16 warps  -- tiles 3 -> 1
    f2048w32   finish BLOCK 2048, 32 warps
    p1024w16   prepare BLOCK 1024, 16 warps
    c512w16    collect 16 warps, BLOCK unchanged (VEC and CHUNK depend on it)

Every arm reports BOTH the per-launch budget and the real benchmark, so a
change in the end-to-end number can be attributed to the launch it was made
in. The other six shapes do not take the sampled route and are a control.

RECOVERY, if killed between the swap and the restore:

    git checkout -- src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py

    tools/vendor_probe.sh tools/hygon_prefill_geometry.py hygon_prefill_geometry
"""

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

# (tag, PBLOCK, PWARPS, CWARPS, FBLOCK, FWARPS); 0 means "keep the shipped one"
ARMS = [
    ("base", 0, 0, 0, 0, 0),
    ("f1024w8", 0, 0, 0, 1024, 8),
    ("f1024w16", 0, 0, 0, 1024, 16),
    ("f2048w16", 0, 0, 0, 2048, 16),
    ("f2048w32", 0, 0, 0, 2048, 32),
    ("p1024w16", 1024, 16, 0, 0, 0),
    ("c512w16", 0, 0, 16, 0, 0),
]

KNOBS = """SSPLIT = max(1, int(os.environ.get("FLAGGEMS_HYGON_PREFILL_SSPLIT", "2")))"""
KNOBS_NEW = (
    KNOBS
    + """
_PB = int(os.environ.get("FLAGGEMS_HYGON_PREFILL_PBLOCK", "0")) or SBLOCK
_PW = int(os.environ.get("FLAGGEMS_HYGON_PREFILL_PWARPS", "0")) or SWARPS
_CW = int(os.environ.get("FLAGGEMS_HYGON_PREFILL_CWARPS", "0")) or SWARPS
_FB = int(os.environ.get("FLAGGEMS_HYGON_PREFILL_FBLOCK", "0")) or SBLOCK
_FW = int(os.environ.get("FLAGGEMS_HYGON_PREFILL_FWARPS", "0")) or SWARPS"""
)

PREPARE_OLD = """        self.prepare = _SLaunch(
            _s_prepare,
            (num_rows,),
            {
                "TARGET": int(top_k * TARGET_MULT),
                "NB": nb,
                "STRIDE": SSTRIDE,
                "BLOCK": SBLOCK,
            },
            SWARPS,
        )"""
PREPARE_NEW = PREPARE_OLD.replace('"BLOCK": SBLOCK', '"BLOCK": _PB').replace(
    "            SWARPS,", "            _PW,"
)

COLLECT_OLD = """                "CHUNK": schunk,
            },
            SWARPS,
        )"""
COLLECT_NEW = """                "CHUNK": schunk,
            },
            _CW,
        )"""

FINISH_OLD = """            {"TOPK": top_k, "NB": nb, "CAP": cap, "RADIX": SRADIX, "BLOCK": SBLOCK},
            SWARPS,
        )"""
FINISH_NEW = """            {"TOPK": top_k, "NB": nb, "CAP": cap, "RADIX": SRADIX, "BLOCK": _FB},
            _FW,
        )"""

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
    f" {int(cnt.min())} {float(cnt.float().mean()):.0f} {int(cnt.max())}"
    f" {int((cnt < top_k).sum())} geometry"
    f" P{m._PB}/{m._PW} C{m.SBLOCK}/{m._CW} F{m._FB}/{m._FW}"
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


def patched(src):
    """The knobs, applied once; the arms then select through the environment."""
    for old, new in (
        (KNOBS, KNOBS_NEW),
        (PREPARE_OLD, PREPARE_NEW),
        (COLLECT_OLD, COLLECT_NEW),
        (FINISH_OLD, FINISH_NEW),
    ):
        assert src.count(old) == 1, f"anchor moved: {old.splitlines()[0]!r}"
        src = src.replace(old, new, 1)
    return src


def env_for(arm):
    e = dict(os.environ)
    for name, v in zip(("PBLOCK", "PWARPS", "CWARPS", "FBLOCK", "FWARPS"), arm[1:]):
        e[f"FLAGGEMS_HYGON_PREFILL_{name}"] = str(v)
    return e


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
    occupancy("before")

    res = {a[0]: [] for a in ARMS}
    budget, broken = {}, {}
    try:
        OVERRIDE.write_text(patched(pristine))
        for arm in ARMS:
            tag = arm[0]
            print(f"### budget, arm {tag}", flush=True)
            r = subprocess.run(
                [sys.executable, "-c", BUDGET],
                capture_output=True,
                text=True,
                env=env_for(arm),
            )
            line = [ln for ln in r.stdout.splitlines() if ln.startswith("BUDGET")]
            if line:
                budget[tag] = line[0].split()[1:]
            else:
                broken[tag] = "\n".join(r.stderr.strip().splitlines()[-3:])[:300]
                print(f"      ! {tag}: {broken[tag]}", flush=True)

        for p in range(PASSES):
            for arm in ARMS:
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
        print(f"### restored; git status: {left.strip() or 'clean'}", flush=True)

    print("\n  per-launch budget on 64x129280, do_bench, us\n")
    hdr = ("arm", "prepare", "collect", "finish", "whole", "cand min", "mean", "max")
    print("  " + " ".join(f"{h:>9}" for h in hdr) + "   under TOPK   geometry")
    for tag, v in budget.items():
        print(
            "  "
            + " ".join(f"{x:>9}" for x in (tag,) + tuple(v[:7]))
            + f"   {v[7]:>9}   "
            + " ".join(v[9:])
        )

    good = [a[0] for a in ARMS if len(res[a[0]]) == PASSES]
    print(f"\n  benchmark SpeedUp on {FOCUS[0]}x{FOCUS[1]}, two passes\n")
    print(f"  {'arm':>9} {'pass 1':>9} {'pass 2':>9} {'vs base':>9}")
    if "base" in good:
        b0 = sum(res["base"][p][FOCUS][0] for p in range(PASSES)) / PASSES
        for arm in ARMS:
            tag = arm[0]
            if tag not in good:
                print(f"  {tag:>9}   FAILED: {broken.get(tag, 'incomplete')}")
                continue
            v = [res[tag][p][FOCUS][0] for p in range(PASSES)]
            print(f"  {tag:>9} {v[0]:9.3f} {v[1]:9.3f} {sum(v) / 2 / b0:9.3f}")

    print("\n  the other six shapes are a control -- they must not move:")
    for tag in good:
        gm = []
        for p in range(PASSES):
            vals = [v[0] for k, v in res[tag][p].items() if k != FOCUS]
            g = 1.0
            for x in vals:
                g *= x
            gm.append(g ** (1.0 / len(vals)))
        print(
            f"      {tag:>9}: geomean of the six " + " / ".join(f"{x:.3f}" for x in gm)
        )

    if "base" in good:
        lat = [res[a][p][FOCUS][1] for a in good for p in range(PASSES)]
        print(f"\n  vLLM latency on that shape, max/min {max(lat) / min(lat):.2f}")
    occupancy("after")


if __name__ == "__main__":
    main()
