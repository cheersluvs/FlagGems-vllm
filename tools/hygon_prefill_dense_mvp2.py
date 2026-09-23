"""MVP 2: the one-kernel sampled dense top-k at one warp, with a cheap exact select.

tools/hygon_prefill_dense_mvp_budget.py showed the cost was cross-warp
collectives: at ONE warp the MVP ran 642 us on 12961x4100 against 896 for
the shipped kernel (1.40x; 1.44x and 1.60x on the other two), correct on every
unflagged row. Its phases at one warp:

    prologue 49 | sample + 2 thresholds 42 | one pass 334 | exact select 217

The select is a full tl.sort of 512 slots, where only ~150 of a ~340-element
band are wanted. This replaces it with the same tool the thresholds use:
bitwise lifting for the need-th smallest 32-bit ordered key K over the band,
then one cumsum to emit keys < K and one to fill ties at K. Inside one wave,
no LDS, no barrier.

    sort       the MVP's select (final_network, CAP 512) -- the control;
               must reproduce ~642 us
    lift       32 lifting steps
    liftp      lifting over the bits where the band's keys differ only: the
               common prefix of min and max is taken as known (the highest
               differing bit from the float exponent of min ^ max, which can
               only round UP, i.e. lift one bit too many -- never too few)

Lifting costs per tile rather than per CAP^2, so the band can grow: BCAP 1024
should flag far fewer rows than 512 (the MVP's flags were mostly band
overflow, max 539-587). Also tried: a 1024-wide pass tile, and a wider band.

Every arm at num_warps=1. Correctness per arm on normal, narrow-band and
partial rows against torch.topk on the rows it did not flag, with the flag
counts reported per input (a narrow band collapses the 11-bit key, so
flagging most of those rows is expected -- they would take the retry).
Implied SpeedUp scales the shipped operator's accepted figure by the time
ratio on the same inputs.

    tools/vendor_probe.sh tools/hygon_prefill_dense_mvp2.py hygon_prefill_dense_mvp2
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.top_k_per_row_prefill import _convert_to_uint32
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
def _kth11(keys, valid, r):
    res = tl.zeros((), tl.int32)
    for b in tl.static_range(10, -1, -1):
        probe = res | (1 << b)
        cnt = tl.sum((valid & (keys < probe)).to(tl.int32), axis=0)
        res = tl.where(cnt < r, probe, res)
    return res


@triton.jit
def _classify(
    x,
    i,
    m,
    t_hi,
    t_lo,
    S,
    B,
    obase,
    vb,
    kb,
    ib,
    TOPK: tl.constexpr,
    BCAP: tl.constexpr,
    SELECT: tl.constexpr,
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
    if SELECT == 0:
        tl.store(vb + pb, x, mask=keep)
    else:
        tl.store(kb + pb, _convert_to_uint32(x).to(tl.int32, bitcast=True), mask=keep)
    tl.store(ib + pb, i, mask=keep)
    return S + (tot & 0xFFFF), B + (tot >> 16)


@triton.jit
def _mvp2(
    x_ptr,
    starts_ptr,
    ends_ptr,
    out_ptr,
    bval_ptr,
    bkey_ptr,
    bidx_ptr,
    stat_ptr,
    stride0,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    NS: tl.constexpr,
    BCAP: tl.constexpr,
    HI: tl.constexpr,
    LO: tl.constexpr,
    SELECT: tl.constexpr,
):
    row = tl.program_id(0)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    span = e - s
    base = x_ptr + row * stride0 + s

    CH: tl.constexpr = NS // 8
    sl = tl.arange(0, NS)
    si = (sl // CH) * (span // 8) + (sl % CH)
    sv = si < span
    sk = _key11(tl.load(base + si, mask=sv, other=float("-inf")))
    ns_eff = tl.sum(sv.to(tl.int32), axis=0)
    expect = TOPK * ns_eff.to(tl.float32) / span.to(tl.float32)
    t_hi = _kth11(sk, sv, (expect * HI / 100).to(tl.int32))
    t_lo = _kth11(sk, sv, (expect * LO / 100).to(tl.int32) + 1) + 1

    lane = tl.arange(0, BLOCK)
    obase = out_ptr + row * TOPK
    vb = bval_ptr + row * BCAP
    kb = bkey_ptr + row * BCAP
    ib = bidx_ptr + row * BCAP
    S = tl.zeros((), tl.int32)
    B = tl.zeros((), tl.int32)
    n_full = span // BLOCK
    for t in tl.range(0, n_full):
        i = t * BLOCK + lane
        x = tl.load(base + i)
        S, B = _classify(
            x, i, i >= 0, t_hi, t_lo, S, B, obase, vb, kb, ib, TOPK, BCAP, SELECT
        )
    i = n_full * BLOCK + lane
    m = i < span
    x = tl.load(base + i, mask=m, other=float("-inf"))
    S, B = _classify(x, i, m, t_hi, t_lo, S, B, obase, vb, kb, ib, TOPK, BCAP, SELECT)

    need = TOPK - S
    good = (S <= TOPK) & (need <= B) & (B <= BCAP)
    tl.store(stat_ptr + row * 3, S)
    tl.store(stat_ptr + row * 3 + 1, B)
    tl.store(stat_ptr + row * 3 + 2, 1 - good.to(tl.int32))
    tl.debug_barrier()
    if good:
        if SELECT == 0:
            final_network(vb, ib, obase, B, S, need, CAP=BCAP)
        else:
            q = tl.arange(0, BCAP)
            bv = q < B
            bk = tl.load(kb + q, mask=bv, other=0).to(tl.uint32, bitcast=True)
            one = tl.full((), 1, tl.uint32)
            if SELECT == 1:
                res = tl.zeros((), tl.uint32)
                for b in tl.static_range(31, -1, -1):
                    probe = res | (one << b)
                    cnt = tl.sum((bv & (bk < probe)).to(tl.int32), axis=0)
                    res = tl.where(cnt < need, probe, res)
            else:
                kmin = tl.min(
                    tl.where(bv, bk, tl.full([BCAP], 0xFFFFFFFF, tl.uint32)), axis=0
                )
                kmax = tl.max(tl.where(bv, bk, tl.zeros([BCAP], tl.uint32)), axis=0)
                d = kmin ^ kmax
                hb = (d.to(tl.float32).to(tl.int32, bitcast=True) >> 23) - 127
                hb = tl.minimum(tl.maximum(hb, 0), 31)
                nb = hb + 1
                low = tl.where(
                    nb >= 32,
                    tl.full((), 0xFFFFFFFF, tl.uint32),
                    (one << nb.to(tl.uint32)) - one,
                )
                res = kmin & ~low
                for ii in tl.range(0, nb):
                    probe = res | (one << (hb - ii).to(tl.uint32))
                    cnt = tl.sum((bv & (bk < probe)).to(tl.int32), axis=0)
                    res = tl.where(cnt < need, probe, res)
            idx = tl.load(ib + q, mask=bv, other=0)
            lt = bv & (bk < res)
            lti = lt.to(tl.int32)
            nlt = tl.sum(lti, axis=0)
            tl.store(obase + S + tl.cumsum(lti, axis=0) - lti, idx, mask=lt)
            eq = bv & (bk == res)
            eqi = eq.to(tl.int32)
            pe = S + nlt + tl.cumsum(eqi, axis=0) - eqi
            tl.store(obase + pe, idx, mask=eq & (pe < TOPK))


def bench(fn):
    return min(triton.testing.do_bench(fn, warmup=20, rep=300) for _ in range(3)) * 1e3


ARMS = [
    # tag, BLOCK, BCAP, HI, LO, SELECT
    ("sort", 512, 512, 75, 135, 0),
    ("lift", 512, 512, 75, 135, 1),
    ("liftp", 512, 512, 75, 135, 2),
    ("liftp cap1k", 512, 1024, 75, 135, 2),
    ("liftp blk1k", 1024, 1024, 75, 135, 2),
    ("liftp 70/150", 512, 1024, 70, 150, 2),
]


def inputs(num_rows, vocab, stride0, kind, dev="cuda"):
    torch.manual_seed(42)
    buf = torch.randn((num_rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
    if kind == "band":
        buf = 10.0 + 0.2 * torch.rand_like(buf)
    x = torch.as_strided(buf, (num_rows, vocab), (stride0, 1))
    if kind == "partial":
        g = torch.Generator(device="cpu").manual_seed(7)
        st = torch.randint(0, 200, (num_rows,), generator=g).to(torch.int32)
        en = (vocab - torch.randint(0, 200, (num_rows,), generator=g)).to(torch.int32)
    else:
        st = torch.zeros(num_rows, dtype=torch.int32)
        en = torch.full((num_rows,), vocab, dtype=torch.int32)
    return x, st.to(dev), en.to(dev)


def main():
    from importlib import import_module

    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    dev = "cuda"
    top_k = 512
    accepted = {(16383, 4095): 1.041, (12961, 4100): 0.761, (16380, 5115): 0.927}
    for num_rows, vocab, stride0 in (
        (16383, 4095, 4352),
        (12961, 4100, 4360),
        (16380, 5115, 5376),
    ):
        label = f"{num_rows}x{vocab}"
        x, st, en = inputs(num_rows, vocab, stride0, "normal")
        out0 = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
        t0 = bench(
            lambda: ov.top_k_per_row_prefill(
                x, st, en, out0, num_rows, stride0, 1, top_k
            )
        )
        print(
            f"\n  {label}: shipped {t0:.1f} us (accepted {accepted[(num_rows, vocab)]})"
        )
        print(
            f"  {'arm':>13} {'us':>8} {'x shipped':>9} {'implied':>8} {'regs':>5}"
            "   flagged normal/band/partial   answer normal/band/partial   band mean/max"
        )
        for tag, blk, bcap, hi, lo, select in ARMS:
            bval = torch.empty((num_rows, bcap), dtype=torch.float32, device=dev)
            bkey = torch.empty((num_rows, bcap), dtype=torch.int32, device=dev)
            bidx = torch.empty((num_rows, bcap), dtype=torch.int32, device=dev)
            stat = torch.zeros((num_rows, 3), dtype=torch.int32, device=dev)

            def launch(xx, ss, ee, oo):
                return _mvp2[(num_rows,)](
                    xx,
                    ss,
                    ee,
                    oo,
                    bval,
                    bkey,
                    bidx,
                    stat,
                    stride0,
                    TOPK=top_k,
                    BLOCK=blk,
                    NS=512,
                    BCAP=bcap,
                    HI=hi,
                    LO=lo,
                    SELECT=select,
                    num_warps=1,
                )

            flags, answers, regs, band_stat = [], [], "?", ""
            try:
                for kind in ("normal", "band", "partial"):
                    xx, ss, ee = inputs(num_rows, vocab, stride0, kind)
                    oo = torch.full(
                        (num_rows, top_k), -1, dtype=torch.int32, device=dev
                    )
                    k = launch(xx, ss, ee, oo)
                    torch.cuda.synchronize()
                    regs = getattr(k, "n_regs", "?")
                    bad = stat[:, 2] != 0
                    good = ~bad
                    flags.append(str(int(bad.sum())))
                    if kind == "normal":
                        band_stat = f"{float(stat[:, 1].float().mean()):.0f}/{int(stat[:, 1].max())}"
                    if int(good.sum()) == 0:
                        answers.append("none")
                        continue
                    col = torch.arange(vocab, device=dev)[None, :]
                    inside = (col >= ss[:, None].long()) & (col < ee[:, None].long())
                    ref = torch.topk(
                        xx.masked_fill(~inside, float("-inf")), top_k, dim=1
                    ).values[good]
                    got = torch.gather(
                        xx, 1, ss[:, None].long() + oo.long().clamp(min=0)
                    )
                    got = got[good].sort(dim=1, descending=True)[0]
                    err = float((got - ref).abs().max())
                    pads = int((oo[good] < 0).sum())
                    answers.append(
                        "ok" if err == 0.0 and pads == 0 else f"WRONG({err:.0e},{pads})"
                    )
                    xx = ss = ee = oo = ref = got = inside = col = None
                    torch.cuda.empty_cache()
                t = bench(lambda: launch(x, st, en, out0))
                print(
                    f"  {tag:>13} {t:8.1f} {t0 / t:9.3f}"
                    f" {accepted[(num_rows, vocab)] * t0 / t:8.3f} {regs!s:>5}"
                    f"   {'/'.join(flags):>26}   {'/'.join(answers):>26}   {band_stat:>13}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"  {tag:>13} FAILED {exc!r}"[:220], flush=True)
        x = st = en = out0 = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
