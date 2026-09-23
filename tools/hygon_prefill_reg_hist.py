"""prepare's threshold from register histograms instead of a global one.

WHY. tools/hygon_prefill_fourrow.py timed prepare on its own: 15.2 us for a
4x8193 plan that samples 513 elements a row, against ~24 us for 64x129280 at
8080. Most of prepare is FIXED cost, not sampling: it clears a 2048-bin
histogram in global memory, barriers, fires one global atomic per sample,
barriers again, and scans the 2048 bins in four 512-wide chunks, each a global
load, a cumsum, three reductions and a carried flag. After bae5a98
(64,129280) is at 0.96 and prepare is ~20% of it.

WHAT. Two register-resident tl.histogram passes over the SAME sample, giving
the SAME 11-bit threshold, so collect and finish see identical inputs and any
change is prepare's alone:

    coarse  tl.histogram(key11 >> 3, 256), accumulated across the sample's
            tiles; one 256-wide cumsum finds the coarse bin cb holding rank
            `target` and the count above it
    fine    tl.histogram(key11 & 7 if key11 >> 3 == cb else 8, 16) over the
            sample again (it is 32 KB a row and L2-resident); bins 0-7 are cb's
            eight fine bins, 8 is a sink for everything else
    thr     cb * 8 + fb + 1 -- the lowest 11-bit bin whose inclusive prefix
            reaches `target`, which is exactly what the 2048-bin scan returns

No global histogram, no clear, no barrier, no atomic. tl.histogram at 256
bins is the regime where it was cheap in a replica (0.268 vs the 2048-bin
atomic's 0.605 per element, tools/hygon_histogram_cost2) -- and a replica has
flipped three verdicts here, so the operator decides.

Out-of-row lanes cannot be masked out of tl.histogram (it does NOT drop
out-of-range values on this backend -- established by the hist_prim probes),
so they load -inf, whose key is 2016: the bottom of the order, below every
finite value. That changes the result only when the sample holds fewer than
`target` real elements, where the old scan returned "not found" (2047) and
this returns 2016; both then collect every non-NaN element, and the exact
retry takes over.

ARMS
    base    shipped
    reg     coarse + fine, the same threshold          <- the candidate
    reg1    coarse only, thr = (cb + 1) * 8: one pass, but it takes the whole
            coarse boundary bin, ~8 fine bins, so more candidates reach
            finish. It prices exactness against a second pass.

`reg` must reproduce base's thresholds EXACTLY. The child prints a checksum of
every row's threshold and of every row's candidate count; base and reg must
print the same pair. Correctness is also checked on normal, narrow-band
(forces the retry) and partial rows against torch.topk, then the full suite.
The budget is printed per arm: collect and finish are the control, and for
`reg` they must not move.

RECOVERY, if killed between the swap and the restore:

    git checkout -- src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py

    tools/vendor_probe.sh tools/hygon_prefill_reg_hist.py hygon_prefill_reg_hist
"""

import importlib.util
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_pc", HERE / "hygon_prefill_private_counters.py"
)
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)

OVERRIDE = pc.OVERRIDE
PASSES = 2
BENCH = pc.BENCH
TESTS = pc.TESTS
FOCUS = (64, 129280, 1024)

PREPARE_BODY = '''    """Zero this row's counters, then take the threshold from two register
    histograms over the sample: 256 coarse bins, then the eight fine bins of
    the coarse boundary bin. Same 11-bit threshold as a 2048-bin scan, with no
    global histogram, barrier or atomic."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    tl.store(cnt_ptr + row * SPLIT + tl.arange(0, SPLIT), tl.zeros([SPLIT], tl.int32))
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    rbase = logits_ptr + row * stride0
    ntiles = tl.cdiv(e - s, BLOCK * STRIDE)
    target: tl.constexpr = (TARGET + STRIDE - 1) // STRIDE

    cbins = tl.arange(0, 256)
    hc = tl.zeros([256], tl.int32)
    for t in tl.range(0, ntiles):
        i = s + t * BLOCK * STRIDE + lane
        x = tl.load(rbase + i, mask=i < e, other=float("-inf"))
        hc += tl.histogram(_key11(x).to(tl.int32) >> 3, 256)
    pre = tl.cumsum(hc, axis=0) - hc
    hit = (pre < target) & (pre + hc >= target)
    cb = tl.min(tl.where(hit, cbins, 255), axis=0)
    above = tl.max(tl.where(cbins == cb, pre, 0), axis=0)
'''

FINE = """
    fbins = tl.arange(0, 16)
    hf = tl.zeros([16], tl.int32)
    for t in tl.range(0, ntiles):
        i = s + t * BLOCK * STRIDE + lane
        x = tl.load(rbase + i, mask=i < e, other=float("-inf"))
        k = _key11(x).to(tl.int32)
        hf += tl.histogram(tl.where((k >> 3) == cb, k & 7, 8), 16)
    pre2 = tl.cumsum(hf, axis=0) - hf
    need = target - above
    hit2 = (fbins < 8) & (pre2 < need) & (pre2 + hf >= need)
    fb = tl.min(tl.where(hit2, fbins, 7), axis=0)
    tl.store(thr_ptr + row, cb * 8 + fb + 1)
"""

COARSE_ONLY = """    tl.store(thr_ptr + row, (cb + 1) * 8)
"""


def variant(src, arm):
    if arm == "base":
        return src
    a, b = pc._fn_bounds(src, "_s_prepare")
    fn = src[a:b]
    d = fn.index('    """Zero, sample and threshold')
    new = fn[:d] + PREPARE_BODY + (FINE if arm == "reg" else COARSE_ONLY)
    return src[:a] + new + src[b:]


ARMS = ["base", "reg", "reg1"]

# Written for the LANDED layout (bae5a98): one unpadded counter per collect
# program, so plan.cnt is num_rows * SSPLIT long.
CHILD = r"""
import torch, triton
from importlib import import_module

M = "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
m = import_module(M)
dev = "cuda"
num_rows, vocab, top_k, stride0 = 64, 129280, 1024, 129280


def inputs(kind):
    torch.manual_seed(42)
    buf = torch.randn((num_rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
    if kind == "band":
        buf = 10.0 + 0.2 * torch.rand_like(buf)
    x = torch.as_strided(buf, (num_rows, vocab), (stride0, 1))
    assert x.stride(0) == stride0 and x.stride(1) == 1
    if kind == "partial":
        g = torch.Generator(device="cpu").manual_seed(7)
        st = torch.randint(0, 2000, (num_rows,), generator=g).to(torch.int32)
        en = (vocab - torch.randint(0, 2000, (num_rows,), generator=g)).to(torch.int32)
    else:
        st = torch.zeros(num_rows, dtype=torch.int32)
        en = torch.full((num_rows,), vocab, dtype=torch.int32)
    return x, st.to(dev), en.to(dev)


def check(kind):
    x, st, en = inputs(kind)
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


checks = [check(k) for k in ("normal", "band", "partial")]

x, st, en = inputs("normal")
out = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
assert m._can_sample(x, st, en, num_rows, stride0, 1, top_k)
plan = m._SPlan(x.device, x.dtype, num_rows, vocab, top_k)


def prep():
    plan.prepare(x, st, en, plan.hist, plan.thr, plan.cnt, stride0)


def prep_coll():
    prep()
    plan.collect(x, st, en, plan.thr, plan.cnt, plan.cand_idx, plan.cand_val, stride0)


b = triton.testing.do_bench
t_p = b(prep, warmup=100, rep=300) * 1e3
t_pc = b(prep_coll, warmup=100, rep=300) * 1e3
t_a = b(lambda: plan.run(x, st, en, out, stride0), warmup=100, rep=300) * 1e3

plan.run(x, st, en, out, stride0)
torch.cuda.synchronize()
per_row = plan.cnt.view(num_rows, m.SSPLIT).to(torch.int64).sum(1)
w = torch.arange(1, num_rows + 1, device=dev)
thr = plan.thr.to(torch.int64)
print(
    f"CHILD {t_p:.1f} {t_pc - t_p:.1f} {t_a - t_pc:.1f} {t_a:.1f}"
    f" {int(per_row.min())} {int(per_row.max())} {'/'.join(checks)}"
    f" thr_ck={int((thr * w).sum())} cnt_ck={int((per_row * w).sum())}"
    f" thr={int(thr.min())}-{int(thr.max())}"
)
"""


def main():
    dirty = pc.sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
    if dirty.strip():
        raise SystemExit("the override is already modified:\n" + dirty)
    pristine = OVERRIDE.read_text()

    print("### preflight", flush=True)
    variants = {}
    for arm in ARMS:
        v = variant(pristine, arm)
        compile(v, f"<{arm}>", "exec")
        pc.preflight(v, arm)
        variants[arm] = v
        print(f"      {arm:>5}: ok", flush=True)
    pc.occupancy("before")

    child, tests, bench, broken = {}, {}, {a: [] for a in ARMS}, {}
    try:
        for arm in ARMS:
            OVERRIDE.write_text(variants[arm])
            print(f"### child (budget + checks), arm {arm}", flush=True)
            r = subprocess.run(
                [sys.executable, "-c", CHILD],
                capture_output=True,
                text=True,
                env=dict(os.environ),
            )
            line = [x for x in r.stdout.splitlines() if x.startswith("CHILD")]
            if not line:
                broken[arm], tail = pc.why(r)
                print(f"      ! {arm}: {broken[arm]}", flush=True)
                for ln in tail:
                    print(f"        | {ln[:200]}", flush=True)
                continue
            child[arm] = line[0].split()[1:]
            print(f"      {' '.join(child[arm])}", flush=True)

            print(f"### tests, arm {arm}", flush=True)
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-rf"] + TESTS,
                capture_output=True,
                text=True,
                env=dict(os.environ),
            )
            tests[arm] = pc.parse_tests(r.stdout)
            print(f"      {tests[arm]}", flush=True)
            for ln in [x for x in r.stdout.splitlines() if x.startswith("FAILED")][:3]:
                print(f"        {ln[:220]}", flush=True)

        for p in range(PASSES):
            order = ARMS[p % len(ARMS) :] + ARMS[: p % len(ARMS)]
            for arm in order:
                if arm in broken:
                    continue
                OVERRIDE.write_text(variants[arm])
                print(f"### benchmark pass {p + 1}, arm {arm}", flush=True)
                r = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "-s"] + BENCH,
                    capture_output=True,
                    text=True,
                    env=dict(os.environ),
                )
                rows = pc.parse_bench(r.stdout)
                if not rows:
                    broken[arm], tail = pc.why(r)
                    print(f"      ! {arm}: {broken[arm]}", flush=True)
                    continue
                bench[arm].append(rows)
    finally:
        OVERRIDE.write_text(pristine)
        left = pc.sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
        print(f"### restored; git status: {left.strip() or 'clean'}", flush=True)

    print(f"\n  budget on {FOCUS[0]}x{FOCUS[1]}, do_bench on our own launches, us\n")
    print(
        f"  {'arm':>5} {'prepare':>8} {'collect':>8} {'finish':>8} {'whole':>8}"
        "   cand/row    normal/band/partial"
    )
    for arm in ARMS:
        if arm not in child:
            print(f"  {arm:>5}   FAILED: {broken.get(arm, '?')}")
            continue
        v = child[arm]
        print(
            f"  {arm:>5} {v[0]:>8} {v[1]:>8} {v[2]:>8} {v[3]:>8}"
            f"   {v[4]}-{v[5]:<6}  {v[6]}"
        )
    print("\n  thresholds and candidate counts -- base and reg must agree EXACTLY:")
    for arm in ARMS:
        if arm in child:
            print(f"      {arm:>5}: " + " ".join(child[arm][7:]))

    good = [a for a in ARMS if len(bench[a]) == PASSES]
    print(f"\n  benchmark SpeedUp on {FOCUS[0]}x{FOCUS[1]}, two passes\n")
    print(f"  {'arm':>5} {'pass 1':>9} {'pass 2':>9} {'vs base':>9}   tests")
    b0 = None
    if "base" in good:
        b0 = sum(bench["base"][p][FOCUS][0] for p in range(PASSES)) / PASSES
    for arm in ARMS:
        if arm not in good:
            print(f"  {arm:>5}   FAILED: {broken.get(arm, 'incomplete')}")
            continue
        v = [bench[arm][p][FOCUS][0] for p in range(PASSES)]
        rel = f"{sum(v) / PASSES / b0:9.3f}" if b0 else "        -"
        print(f"  {arm:>5} {v[0]:9.3f} {v[1]:9.3f} {rel}   {tests.get(arm, '-')}")

    print("\n  the other six shapes do not take the sampled path -- a control:")
    for arm in good:
        gm = []
        for p in range(PASSES):
            vals = [v[0] for k, v in bench[arm][p].items() if k != FOCUS]
            g = 1.0
            for x in vals:
                g *= x
            gm.append(g ** (1.0 / len(vals)))
        print(
            f"      {arm:>5}: geomean of the six " + " / ".join(f"{x:.3f}" for x in gm)
        )
    if good:
        lat = [bench[a][p][FOCUS][1] for a in good for p in range(PASSES)]
        print(f"\n  vLLM latency on that shape, max/min {max(lat) / min(lat):.2f}")
    pc.occupancy("after")


if __name__ == "__main__":
    main()
