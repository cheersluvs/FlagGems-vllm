"""Where the time on (64,129280) goes, and whether collect should store values.

PART A -- THE BUDGET. The sampled pipeline is three launches and I have never
priced them separately. do_bench on our own kernels only, no baseline, so the
profiler's double-count cannot enter. collect cannot be timed alone (it
atomically grows `cnt`, so a second call runs with the counter past CAP and
stores nothing), and prepare is what zeroes `cnt` -- so the three quantities
are measured as nested prefixes and differenced:

    prepare           = do_bench(prepare)
    prepare+collect   = do_bench(prepare, collect)
    whole             = do_bench(plan.run)

PART B -- THE VALUE STORE. `_s_collect` stores only the index; `_s_finish`
gathers the values back with a scattered load per candidate. That choice came
from the MTT override, where dropping the value store netted 8.6 us AFTER the
re-read, and it has never been A/B'd on this card. The arithmetic here does
not obviously favour it:

    collect reads              64 x 129280 x 4 B      = 33.1 MB, sequential
    finish gathers ~1280/row   64 x 1280 scattered 4B loads

At a 128-byte line and no reuse, that gather pulls ~10.5 MB of lines to
retrieve 0.33 MB of values -- a 32x read amplification, a third of the whole
collect pass, for 0.3% of the values. L2 is 8 MB and the row set is 33 MB, so
the lines are not resident. The store it would replace is one scattered 4-byte
store per hit into a 64 x CAP buffer of 1 MB, which IS L2-sized.

ARMS, both on the real benchmark, two passes:
    idx   as shipped -- index only, finish gathers
    val   collect stores the value too, finish's gather and its barrier go

The other six shapes do not take the sampled route and are printed as a
control: they must not move.

RECOVERY, if killed between the swap and the restore:

    git checkout -- src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py

    tools/vendor_probe.sh tools/hygon_prefill_value_store.py hygon_prefill_value_store
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

IDX_STORE = """        keep = take & (pos >= 0) & (pos < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos, i.to(tl.int32), mask=keep)"""
VAL_STORE = (
    IDX_STORE
    + """
        tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)"""
)

GATHER_OLD = """    # _s_collect stores indices only; gather the candidate values back from
    # global memory. The retry above stores values as well, so this re-read is
    # redundant there -- but correct, and the retry does not fire in practice.
    # The barrier is required: this program loads from vbase right after
    # storing to it.
    row_base = logits_ptr + row * stride0 + s
    for t in tl.range(0, tiles):
        pos = t * BLOCK + lane
        valid = pos < n
        ci = tl.load(ibase + pos, mask=valid, other=0)
        tl.store(vbase + pos, tl.load(row_base + ci, mask=valid, other=0.0), mask=valid)
    tl.debug_barrier()
"""
GATHER_NEW = """    # _s_collect stored the value beside the index, so there is nothing to
    # gather. The kernel boundary orders those stores against these loads.
"""

BUDGET = r"""
import torch, triton
from importlib import import_module

M = "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
m = import_module(M)
dev = "cuda"
num_rows, vocab, top_k, stride0, stride1 = 64, 129280, 1024, 129280, 1

# the benchmark's own recipe, so the geometry and the stride are its geometry
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
cap = plan.cap
gathered = int(cnt.clamp(max=cap).sum())
row_bytes = num_rows * vocab * 4

print("")
print("  budget on 64x129280, do_bench on our own kernels (us)")
print("")
print(f"      {'prepare':>18}  {t_p:8.1f}   {100 * t_p / t_a:5.1f}%"
      f"   {row_bytes / m.SSTRIDE / t_p / 1e3:7.0f} GB/s")
print(f"      {'collect':>18}  {t_pc - t_p:8.1f}   {100 * (t_pc - t_p) / t_a:5.1f}%"
      f"   {row_bytes / (t_pc - t_p) / 1e3:7.0f} GB/s")
print(f"      {'finish':>18}  {t_a - t_pc:8.1f}   {100 * (t_a - t_pc) / t_a:5.1f}%")
print(f"      {'whole':>18}  {t_a:8.1f}")
print("")
print(f"      SSTRIDE {m.SSTRIDE}  TARGET_MULT {m.TARGET_MULT}  SSPLIT {m.SSPLIT}"
      f"  CAP {cap}  TOPK {top_k}")
print(f"      candidates per row  min {int(cnt.min())}  mean {float(cnt.float().mean()):.0f}"
      f"  max {int(cnt.max())}")
print(f"      rows under TOPK {int((cnt < top_k).sum())}"
      f"   rows over CAP {int((cnt > cap).sum())}   (either forces the full retry)")
print(f"      finish gathers {gathered} values = {gathered * 4 / 1e6:.2f} MB of data;"
      f" at a 128 B line with no reuse that is {gathered * 128 / 1e6:.1f} MB of traffic,"
      f" against collect's {row_bytes / 1e6:.1f} MB")
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
    if arm == "idx":
        return src
    assert src.count(IDX_STORE) == 2, "the collect stores moved"
    src = src.replace(IDX_STORE, VAL_STORE)
    assert src.count(GATHER_OLD) == 1, "the finish gather moved"
    return src.replace(GATHER_OLD, GATHER_NEW, 1)


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

    print("### part A: the budget", flush=True)
    r = subprocess.run(
        [sys.executable, "-c", BUDGET],
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    print(r.stdout or "", flush=True)
    if r.returncode:
        print("  ! the budget child failed:", flush=True)
        print("\n".join(r.stderr.strip().splitlines()[-12:]), flush=True)

    arms = ["idx", "val"]
    res, broken = {a: [] for a in arms}, {}
    try:
        for p in range(PASSES):
            for arm in arms:
                if arm in broken:
                    continue
                OVERRIDE.write_text(variant(pristine, arm))
                print(f"### part B pass {p + 1}, arm {arm}", flush=True)
                r = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "-s"] + BENCH,
                    capture_output=True,
                    text=True,
                    env=dict(os.environ),
                )
                rows = parse(r.stdout)
                if not rows:
                    why = [ln for ln in r.stdout.splitlines() if "rror" in ln]
                    broken[arm] = why[-1][:160] if why else "no SUCCESS rows"
                    print(f"      ! {arm}: {broken[arm]}", flush=True)
                    continue
                res[arm].append(rows)
    finally:
        OVERRIDE.write_text(pristine)
        left = sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
        print(f"### restored; git status: {left.strip() or 'clean'}", flush=True)

    good = [a for a in arms if len(res[a]) == PASSES]
    print(f"\nbenchmark SpeedUp on {FOCUS[0]}x{FOCUS[1]}, two passes\n")
    print(f"  {'arm':>5} {'pass 1':>9} {'pass 2':>9} {'vs idx':>9}")
    if "idx" in good:
        base = sum(res["idx"][p][FOCUS][0] for p in range(PASSES)) / PASSES
        for arm in arms:
            if arm not in good:
                print(f"  {arm:>5}   FAILED: {broken.get(arm, 'incomplete')}")
                continue
            v = [res[arm][p][FOCUS][0] for p in range(PASSES)]
            print(f"  {arm:>5} {v[0]:9.3f} {v[1]:9.3f} {sum(v) / 2 / base:9.3f}")

    print("\n  the other six shapes are a control -- they must not move:")
    for arm in good:
        for p in range(PASSES):
            other = [
                f"{k[0]}x{k[1]} {v[0]:.3f}"
                for k, v in sorted(res[arm][p].items())
                if k != FOCUS
            ]
            print(f"      {arm} pass {p + 1}: " + "  ".join(other))

    if "idx" in good:
        lat = [res[a][p][FOCUS][1] for a in good for p in range(PASSES)]
        print(f"\n  vLLM latency on that shape, max/min {max(lat) / min(lat):.2f}")
    occupancy("after")


if __name__ == "__main__":
    main()
