"""Where the one-kernel sampled MVP spends its time, and whether warps are why.

tools/hygon_prefill_dense_sampled_mvp.py read each row ONCE with no
per-element atomic and still took 860 us on 12961x4100 against 895 for the
shipped kernel, whose histogram atomics alone are ~474 us. One plain read of
the rows is ~186 us, so ~670 us went somewhere not modelled. And 4 warps were
25-30% SLOWER than 2 -- a hint that the cost is the per-program serial chain
of cross-warp collectives: 22 tl.sum for the two lifted thresholds, one
cumsum per tile, a 512-wide sort, each through LDS and a barrier once a
program spans more than one wave. That is a hypothesis; this measures it.

TRUNCATION, as in the dense budget: every cut is a PREFIX, so nothing
downstream of it can do different work.

    cut0   the per-program prologue
    cut1   + the sample and both thresholds (22 reductions)
    cut2   + the one pass (a load, a key, a packed cumsum, three stores per tile)
    cut3   + the exact select of the band (512-wide final_network): the full MVP

at 2 warps (as the MVP ran) and at 1 warp, where every reduction, cumsum and
sort stays inside one wave: no LDS round trip, no barrier. num_warps=1 once
returned wrong answers on the generic dense kernel, so every full arm is
checked against torch.topk on its unflagged rows.

    lift3  the full MVP with the thresholds found 3 bits at a time: each step
           compares the sample against 7 candidate prefixes for BOTH
           thresholds at once, one [512, 16] reduction per step, 4 steps for
           11 bits -- 4 reductions where the MVP made 22

The compiled kernel's register count is printed per arm.

    tools/vendor_probe.sh tools/hygon_prefill_dense_mvp_budget.py hygon_prefill_dense_mvp_budget
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._hygon.fused._top_k_per_row_prefill_final_network import (  # noqa: E501
    final_network,
)


@triton.jit
def _key11(x):
    h = x.to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    sign_set = (bits & 0x8000) != 0
    inv = (~bits) & 0x7FFF
    mapped = tl.where(sign_set, bits, inv)
    return (mapped >> 5).to(tl.int32)


@triton.jit
def _kth(keys, valid, r):
    res = tl.zeros((), tl.int32)
    for b in tl.static_range(10, -1, -1):
        probe = res | (1 << b)
        cnt = tl.sum((valid & (keys < probe)).to(tl.int32), axis=0)
        res = tl.where(cnt < r, probe, res)
    return res


@triton.jit
def _lift3_step(
    keys, valid, res_hi, res_lo, r_hi, r_lo, SH: tl.constexpr, W: tl.constexpr
):
    q = tl.arange(0, 16)
    is_hi = q < 8
    j = q % 8
    probes = tl.where(is_hi, res_hi, res_lo) | (j << SH)
    cmp = (valid[:, None] & (keys[:, None] < probes[None, :])).to(tl.int32)
    cnt = tl.sum(cmp, axis=0)
    under = tl.where(is_hi, cnt < r_hi, cnt < r_lo) & (j > 0) & (j < (1 << W))
    step_hi = tl.sum((under & is_hi).to(tl.int32), axis=0)
    step_lo = tl.sum((under & (q >= 8)).to(tl.int32), axis=0)
    return res_hi | (step_hi << SH), res_lo | (step_lo << SH)


@triton.jit
def _classify(
    x, i, m, t_hi, t_lo, S, B, obase, vb, ib, TOPK: tl.constexpr, BCAP: tl.constexpr
):
    k = _key11(x)
    sure = m & (k < t_hi)
    band = m & (k >= t_hi) & (k < t_lo)
    packed = sure.to(tl.int32) + (band.to(tl.int32) << 16)
    cs = tl.cumsum(packed, axis=0) - packed
    tot = tl.sum(packed, axis=0)
    ps = S + (cs & 0xFFFF)
    pb = B + (cs >> 16)
    tl.store(obase + ps, i, mask=sure & (ps < TOPK))
    keep = band & (pb < BCAP)
    tl.store(vb + pb, x, mask=keep)
    tl.store(ib + pb, i, mask=keep)
    return S + (tot & 0xFFFF), B + (tot >> 16)


@triton.jit
def _mvp(
    x_ptr,
    starts_ptr,
    ends_ptr,
    out_ptr,
    bval_ptr,
    bidx_ptr,
    stat_ptr,
    stride0,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    NS: tl.constexpr,
    BCAP: tl.constexpr,
    HI: tl.constexpr,
    LO: tl.constexpr,
    CUT: tl.constexpr,
    LIFT3: tl.constexpr,
):
    row = tl.program_id(0)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    span = e - s
    base = x_ptr + row * stride0 + s
    if CUT == 0:
        tl.store(stat_ptr + row * 3, span)
    else:
        CH: tl.constexpr = NS // 8
        sl = tl.arange(0, NS)
        si = (sl // CH) * (span // 8) + (sl % CH)
        sv = si < span
        sk = _key11(tl.load(base + si, mask=sv, other=float("-inf")))
        ns_eff = tl.sum(sv.to(tl.int32), axis=0)
        expect = TOPK * ns_eff.to(tl.float32) / span.to(tl.float32)
        r_hi = (expect * HI / 100).to(tl.int32)
        r_lo = (expect * LO / 100).to(tl.int32) + 1
        if LIFT3:
            a = tl.zeros((), tl.int32)
            c = tl.zeros((), tl.int32)
            a, c = _lift3_step(sk, sv, a, c, r_hi, r_lo, 8, 3)
            a, c = _lift3_step(sk, sv, a, c, r_hi, r_lo, 5, 3)
            a, c = _lift3_step(sk, sv, a, c, r_hi, r_lo, 2, 3)
            a, c = _lift3_step(sk, sv, a, c, r_hi, r_lo, 0, 2)
            t_hi = a
            t_lo = c + 1
        else:
            t_hi = _kth(sk, sv, r_hi)
            t_lo = _kth(sk, sv, r_lo) + 1
        if CUT == 1:
            tl.store(stat_ptr + row * 3, t_hi)
            tl.store(stat_ptr + row * 3 + 1, t_lo)
        else:
            lane = tl.arange(0, BLOCK)
            obase = out_ptr + row * TOPK
            vb = bval_ptr + row * BCAP
            ib = bidx_ptr + row * BCAP
            S = tl.zeros((), tl.int32)
            B = tl.zeros((), tl.int32)
            n_full = span // BLOCK
            for t in tl.range(0, n_full):
                i = t * BLOCK + lane
                x = tl.load(base + i)
                S, B = _classify(
                    x, i, i >= 0, t_hi, t_lo, S, B, obase, vb, ib, TOPK, BCAP
                )
            i = n_full * BLOCK + lane
            m = i < span
            x = tl.load(base + i, mask=m, other=float("-inf"))
            S, B = _classify(x, i, m, t_hi, t_lo, S, B, obase, vb, ib, TOPK, BCAP)
            need = TOPK - S
            good = (S <= TOPK) & (need <= B) & (B <= BCAP)
            tl.store(stat_ptr + row * 3, S)
            tl.store(stat_ptr + row * 3 + 1, B)
            tl.store(stat_ptr + row * 3 + 2, 1 - good.to(tl.int32))
            if CUT >= 3:
                tl.debug_barrier()
                if good:
                    final_network(vb, ib, obase, B, S, need, CAP=BCAP)


def bench(fn):
    return min(triton.testing.do_bench(fn, warmup=20, rep=300) for _ in range(3)) * 1e3


ARMS = [
    # tag, warps, CUT, LIFT3
    ("w2 cut0", 2, 0, False),
    ("w2 cut1", 2, 1, False),
    ("w2 cut2", 2, 2, False),
    ("w2 cut3", 2, 3, False),
    ("w1 cut0", 1, 0, False),
    ("w1 cut1", 1, 1, False),
    ("w1 cut2", 1, 2, False),
    ("w1 cut3", 1, 3, False),
    ("w2 lift3", 2, 3, True),
    ("w1 lift3", 1, 3, True),
]


def main():
    from importlib import import_module

    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    dev = "cuda"
    top_k, bcap = 512, 512
    shapes = ((16383, 4095, 4352), (12961, 4100, 4360), (16380, 5115, 5376))
    res = {}
    for num_rows, vocab, stride0 in shapes:
        label = f"{num_rows}x{vocab}"
        torch.manual_seed(42)
        buf = torch.randn(
            (num_rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32
        )
        x = torch.as_strided(buf, (num_rows, vocab), (stride0, 1))
        st = torch.zeros(num_rows, dtype=torch.int32, device=dev)
        en = torch.full((num_rows,), vocab, dtype=torch.int32, device=dev)
        ref = torch.topk(x, top_k, dim=1).values
        out0 = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
        res[(label, "shipped")] = (
            bench(
                lambda: ov.top_k_per_row_prefill(
                    x, st, en, out0, num_rows, stride0, 1, top_k
                )
            ),
            "-",
            "-",
            "-",
        )
        bval = torch.empty((num_rows, bcap), dtype=torch.float32, device=dev)
        bidx = torch.empty((num_rows, bcap), dtype=torch.int32, device=dev)
        stat = torch.zeros((num_rows, 3), dtype=torch.int32, device=dev)
        for tag, warps, cut, lift3 in ARMS:
            out = torch.full((num_rows, top_k), -1, dtype=torch.int32, device=dev)

            def run():
                return _mvp[(num_rows,)](
                    x,
                    st,
                    en,
                    out,
                    bval,
                    bidx,
                    stat,
                    stride0,
                    TOPK=top_k,
                    BLOCK=512,
                    NS=512,
                    BCAP=bcap,
                    HI=75,
                    LO=135,
                    CUT=cut,
                    LIFT3=lift3,
                    num_warps=warps,
                )

            try:
                k = run()
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001 - report and keep going
                res[(label, tag)] = (float("nan"), "?", "-", f"FAILED {exc!r}"[:120])
                print(f"  {label} {tag:>9}: FAILED {exc!r}"[:200], flush=True)
                continue
            ans, flagged = "-", "-"
            if cut == 3:
                bad = stat[:, 2] != 0
                good = ~bad
                flagged = str(int(bad.sum()))
                g = torch.gather(x[good], 1, out[good].long().clamp(min=0))
                g = g.sort(dim=1, descending=True)[0]
                err = float((g - ref[good]).abs().max())
                pads = int((out[good] < 0).sum())
                ans = "ok" if err == 0.0 and pads == 0 else f"WRONG({err:.1e},{pads})"
            t = bench(run)
            res[(label, tag)] = (t, getattr(k, "n_regs", "?"), flagged, ans)
            print(
                f"  {label} {tag:>9}: {t:8.1f} us  regs {res[(label, tag)][1]}"
                f"  flagged {flagged}  {ans}",
                flush=True,
            )
        buf = x = ref = out = out0 = None
        torch.cuda.empty_cache()

    labels = [f"{r}x{v}" for r, v, _ in shapes]
    print("\n  us per call (do_bench, min of 3)\n")
    print("  " + f"{'arm':>9} " + " ".join(f"{s:>12}" for s in labels))
    for tag in ["shipped"] + [a[0] for a in ARMS]:
        print(
            f"  {tag:>9} "
            + " ".join(f"{res[(s, tag)][0]:12.1f}" for s in labels if (s, tag) in res)
        )
    print("\n  phase costs, us (differences of successive cuts)\n")
    for w in ("w2", "w1"):
        for name, lo, hi in (
            ("prologue", None, "cut0"),
            ("sample + 2 thresholds", "cut0", "cut1"),
            ("one pass", "cut1", "cut2"),
            ("exact select", "cut2", "cut3"),
        ):
            cells = []
            for s in labels:
                h = res[(s, f"{w} {hi}")][0]
                v = h - (res[(s, f"{w} {lo}")][0] if lo else 0.0)
                cells.append(f"{v:12.1f}")
            print(f"  {w} {name:<24}" + " ".join(cells))
        print()


if __name__ == "__main__":
    main()
