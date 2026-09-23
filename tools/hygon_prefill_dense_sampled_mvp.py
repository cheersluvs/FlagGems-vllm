"""MVP: a one-kernel sampled top-k for the big dense prefill shapes.

WHY. The dense budget and the bandwidth probe settled what the shipped dense
kernel pays on 12961x4100 (895 us, 0.761 of vLLM):

    histogram atomics      ~474 us   bound by distinct addresses touched;
                                     every narrower-bin variant lost
    first read + key       ~209 us   the row read runs at 88% of peak already
    second read            ~184 us   NOT served by L2 (v2twice: same as first)
    clear + prologue         89 us
    final select             11 us

A design that reads the row ONCE and does no per-element atomics removes the
two biggest items. Floyd-Rivest shape, one program per row, one launch:

  1. SAMPLE 512 elements from 8 chunks spread over the row. Two thresholds by
     bitwise lifting on the 11-bit key -- 11 steps, each one tl.sum over the
     sample, no histogram, no atomic:
         T_hi  ~0.75 * top_k of the row lies above it  -> surely in
         T_lo  ~1.35 * top_k lies above it             -> the band ends here
  2. ONE PASS over the row. Surely-in elements go straight to the output; the
     band [T_hi, T_lo) goes to a 512-slot buffer (value + index). Both slot
     positions come from ONE cumsum of a packed int (sure + band << 16), with
     a register running offset -- one program owns the row, so no atomic.
  3. EXACT SELECT of the missing top_k - S from the band with the shipped
     `final_network` (tl.sort of 512 ordered-key codes), CAP 512.

A row whose sure set exceeds top_k, whose band is too small to finish it, or
whose band overflows 512 is FLAGGED (it would retry in a real version). This
MVP only counts them; correctness is checked on the unflagged rows.

What it is for: whether this shape of algorithm lands near 400 us on
12961x4100. do_bench on the MVP kernel against the shipped operator on the same
inputs, both correct-checked; the implied SpeedUp scales the shipped one's
accepted 0.761 / 1.041 / 0.927 by the time ratio.

    tools/vendor_probe.sh tools/hygon_prefill_dense_sampled_mvp.py hygon_prefill_dense_sampled_mvp
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
    """The r-th smallest 11-bit key (1-based) among the valid lanes."""
    res = tl.zeros((), tl.int32)
    for b in tl.static_range(10, -1, -1):
        probe = res | (1 << b)
        cnt = tl.sum((valid & (keys < probe)).to(tl.int32), axis=0)
        res = tl.where(cnt < r, probe, res)
    return res


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
def _dense_sampled(
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
):
    row = tl.program_id(0)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    span = e - s
    base = x_ptr + row * stride0 + s

    # 1. sample: 8 chunks of NS // 8 contiguous elements spread over the row
    CH: tl.constexpr = NS // 8
    sl = tl.arange(0, NS)
    si = (sl // CH) * (span // 8) + (sl % CH)
    sv = si < span
    sk = _key11(tl.load(base + si, mask=sv, other=float("-inf")))
    ns_eff = tl.sum(sv.to(tl.int32), axis=0)
    expect = TOPK * ns_eff.to(tl.float32) / span.to(tl.float32)
    r_hi = (expect * HI / 100).to(tl.int32)
    r_lo = (expect * LO / 100).to(tl.int32) + 1
    t_hi = _kth(sk, sv, r_hi)
    t_lo = _kth(sk, sv, r_lo) + 1

    # 2. one pass: sure -> output, band -> buffer
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
        S, B = _classify(x, i, i >= 0, t_hi, t_lo, S, B, obase, vb, ib, TOPK, BCAP)
    i = n_full * BLOCK + lane
    m = i < span
    x = tl.load(base + i, mask=m, other=float("-inf"))
    S, B = _classify(x, i, m, t_hi, t_lo, S, B, obase, vb, ib, TOPK, BCAP)

    need = TOPK - S
    good = (S <= TOPK) & (need <= B) & (B <= BCAP)
    tl.store(stat_ptr + row * 3, S)
    tl.store(stat_ptr + row * 3 + 1, B)
    tl.store(stat_ptr + row * 3 + 2, 1 - good.to(tl.int32))
    tl.debug_barrier()

    # 3. exact select of the missing `need` from the band
    if good:
        final_network(vb, ib, obase, B, S, need, CAP=BCAP)


def bench(fn):
    return min(triton.testing.do_bench(fn, warmup=20, rep=300) for _ in range(3)) * 1e3


def main():
    from importlib import import_module

    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    dev = "cuda"
    top_k = 512
    accepted = {(16383, 4095): 1.041, (12961, 4100): 0.761, (16380, 5115): 0.927}
    configs = [(512, 2, 75, 135), (512, 4, 75, 135), (512, 2, 70, 150)]
    print(
        f"  {'shape':>12} {'impl':>22} {'us':>8} {'x shipped':>9} {'implied':>8}"
        "   flagged   S mean   band mean/max   answer"
    )
    for num_rows, vocab, stride0 in (
        (16383, 4095, 4352),
        (12961, 4100, 4360),
        (16380, 5115, 5376),
    ):
        torch.manual_seed(42)
        buf = torch.randn(
            (num_rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32
        )
        x = torch.as_strided(buf, (num_rows, vocab), (stride0, 1))
        st = torch.zeros(num_rows, dtype=torch.int32, device=dev)
        en = torch.full((num_rows,), vocab, dtype=torch.int32, device=dev)
        ref = torch.topk(x, top_k, dim=1).values

        out = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
        ov.top_k_per_row_prefill(x, st, en, out, num_rows, stride0, 1, top_k)
        torch.cuda.synchronize()
        got = torch.gather(x, 1, out.long().clamp(min=0)).sort(dim=1, descending=True)[
            0
        ]
        ok = "ok" if float((got - ref).abs().max()) == 0.0 else "WRONG"
        t0 = bench(
            lambda: ov.top_k_per_row_prefill(
                x, st, en, out, num_rows, stride0, 1, top_k
            )
        )
        label = f"{num_rows}x{vocab}"
        print(
            f"  {label:>12} {'shipped':>22} {t0:8.1f} {1.0:9.3f} {accepted[(num_rows, vocab)]:8.3f}"
            f"   {'-':>7}   {'-':>6}   {'-':>13}   {ok}"
        )

        for blk, warps, hi, lo in configs:
            bcap = 512
            bval = torch.empty((num_rows, bcap), dtype=torch.float32, device=dev)
            bidx = torch.empty((num_rows, bcap), dtype=torch.int32, device=dev)
            stat = torch.empty((num_rows, 3), dtype=torch.int32, device=dev)
            out2 = torch.full((num_rows, top_k), -1, dtype=torch.int32, device=dev)

            def run():
                _dense_sampled[(num_rows,)](
                    x,
                    st,
                    en,
                    out2,
                    bval,
                    bidx,
                    stat,
                    stride0,
                    TOPK=top_k,
                    BLOCK=blk,
                    NS=512,
                    BCAP=bcap,
                    HI=hi,
                    LO=lo,
                    num_warps=warps,
                )

            run()
            torch.cuda.synchronize()
            flagged = stat[:, 2] != 0
            good = ~flagged
            if int(good.sum()):
                g = torch.gather(x[good], 1, out2[good].long().clamp(min=0))
                g = g.sort(dim=1, descending=True)[0]
                err = float((g - ref[good]).abs().max())
                pads = int((out2[good] < 0).sum())
                ans = "ok" if err == 0.0 and pads == 0 else f"WRONG({err:.1e},{pads})"
            else:
                ans = "no rows"
            t = bench(run)
            tag = f"mvp B{blk} w{warps} {hi}/{lo}"
            print(
                f"  {label:>12} {tag:>22} {t:8.1f} {t0 / t:9.3f}"
                f" {accepted[(num_rows, vocab)] * t0 / t:8.3f}"
                f"   {int(flagged.sum()):>7}   {float(stat[:, 0].float().mean()):6.0f}"
                f"   {float(stat[:, 1].float().mean()):6.0f}/{int(stat[:, 1].max()):<6}   {ans}"
            )
        buf = x = ref = out = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
