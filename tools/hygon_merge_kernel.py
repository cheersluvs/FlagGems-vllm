"""A merge kernel sized for the candidates, against the generic operator.

Both sampled pipelines finish by handing a few thousand candidates per row to
the generic top-k kernel and then remapping its positions. That kernel is
built for a 262144-wide row, and it is now the biggest remaining term in both:

    prefill (64,129280)   merge 83.2 us of a 308.8 us pipeline
    prefill (4,8193)      merge 20.5 us of a  47.9 us pipeline, for 2127
                          candidates -- fixed cost, not work
    decode 8 and 16 rows  fixup + merge + remap ~38 us, of which ~26 is the
                          bare launch floor; both shapes sit below 0.9

The replacement does the whole tail in ONE program per row:

    the fallback check    a row with too few or too many candidates is the
                          caller's problem; here it only clamps
    exact top-k           four 8-bit radix rounds over the FULL 32-bit ordered
                          key (_convert_to_uint32: ascending uint32 is
                          descending float), narrowing desired/desired_mask
                          until the k-th key is known. This is what the
                          generic kernel's own final select does, and doing it
                          over the candidates directly means no 11-bit
                          pre-pass and no threshold-bin special case -- it is
                          exact by construction, with no fp16 granularity to
                          reason about.
    the remap             the output is cand_idx[pos], so no second launch

Measured against "generic merge + remap" at the candidate counts and row
counts the two operators actually produce, over a 2-D BLOCK x warps grid --
geometry was worth 2.9x on the select kernel, so it is not assumed here.

    tools/vendor_probe.sh tools/hygon_merge_kernel.py hygon_merge_kernel
"""

import sys
from importlib import import_module

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

_generic = import_module("flaggems_vllm.ops.top_k_per_row_decode")
_u32 = _generic._convert_to_uint32
_key11 = _generic._convert_to_trt_uint16_hi11

CAP = 8192
TOPK = 512
ROWS = (4, 16, 64, 512)
CANDS = (1024, 2048, 4096, 8192)
GEOMS = [(256, 4), (512, 4), (512, 8), (1024, 8), (1024, 16)]
RADIX = 256
NB = 2048  # 11-bit bins, as the operator's own STEP 0 uses


def device_us(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = 0.0
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0)
        total += t or 0.0
    return total / iters


@triton.jit
def _merge(
    cand_val_ptr,
    cand_idx_ptr,
    cnt_ptr,
    out_ptr,
    counts_ptr,
    slot_ptr,
    CAP: tl.constexpr,
    TOPK: tl.constexpr,
    RADIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Exact top-k of a row's candidates, written straight out as row indices.

    One program owns the row, so its radix counters are its own and a
    tl.debug_barrier() is all the ordering the phases need.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    bins = tl.arange(0, RADIX)
    ones = tl.full([BLOCK], 1, tl.int32)
    n = tl.minimum(tl.load(cnt_ptr + row), CAP)
    vbase = cand_val_ptr + row * CAP
    ibase = cand_idx_ptr + row * CAP
    obase = out_ptr + row * TOPK
    cbase = counts_ptr + row * RADIX
    tiles = tl.cdiv(n, BLOCK)

    if n <= TOPK:
        # fewer candidates than asked for: everything goes out, -1 pads
        for t in tl.static_range((TOPK + BLOCK - 1) // BLOCK):
            j = t * BLOCK + lane
            idx = tl.load(ibase + j, mask=j < n, other=-1)
            tl.store(obase + j, tl.where(j < n, idx, -1), mask=j < TOPK)
        return

    desired = tl.zeros((), tl.uint32)
    desired_mask = tl.zeros((), tl.uint32)
    k_to_find = TOPK + 1
    for digit_pos in tl.static_range(24, -1, -8):
        if k_to_find > 1:
            tl.store(cbase + bins, tl.zeros([RADIX], tl.int32))
            tl.debug_barrier()
            for t in tl.range(0, tiles):
                pos = t * BLOCK + lane
                valid = pos < n
                x = tl.load(vbase + pos, mask=valid, other=0.0)
                key = _u32(x)
                digit = ((key >> digit_pos) & (RADIX - 1)).to(tl.int32)
                tl.atomic_add(
                    cbase + digit,
                    ones,
                    mask=valid & ((key & desired_mask) == desired),
                    sem="relaxed",
                    scope="cta",
                )
            tl.debug_barrier()
            counts = tl.load(cbase + bins)
            prefix = tl.cumsum(counts, axis=0) - counts
            hit = (prefix < k_to_find) & (prefix + counts >= k_to_find)
            tb = tl.min(tl.where(hit, bins, RADIX), axis=0).to(tl.int32)
            tb = tl.where(tb == RADIX, RADIX - 1, tb)
            counts_lt = tl.max(tl.where(bins == tb, prefix, 0), axis=0).to(tl.int32)
            desired = desired | (tb.to(tl.uint32) << digit_pos)
            desired_mask = desired_mask | (
                tl.full((), RADIX - 1, tl.uint32) << digit_pos
            )
            k_to_find = k_to_find - counts_lt

    thr_key = desired
    tl.store(slot_ptr + row, 0)
    tl.debug_barrier()
    slots = slot_ptr + row + tl.zeros([BLOCK], tl.int32)
    # everything strictly better than the k-th, then fill from its equals;
    # after the rounds above the first group is smaller than TOPK and the two
    # together are at least TOPK, so this lands exactly on TOPK
    for equal in tl.static_range(2):
        for t in tl.range(0, tiles):
            pos = t * BLOCK + lane
            valid = pos < n
            x = tl.load(vbase + pos, mask=valid, other=0.0)
            key = _u32(x)
            if equal == 0:
                take = valid & (key < thr_key)
            else:
                take = valid & (key == thr_key)
            p = tl.atomic_add(slots, ones, mask=take, sem="relaxed", scope="cta")
            idx = tl.load(ibase + pos, mask=take, other=-1)
            tl.store(obase + p, idx, mask=take & (p < TOPK))
        tl.debug_barrier()


@triton.jit
def _scan_rank(base, target, NB: tl.constexpr, BLOCK: tl.constexpr):
    """Lowest bin whose inclusive prefix reaches `target`, and that bin's
    EXCLUSIVE prefix -- i.e. how many elements are strictly better."""
    lane = tl.arange(0, BLOCK)
    carry = tl.zeros([], tl.int32)
    tb = tl.full([], NB - 1, tl.int32)
    lt = tl.zeros([], tl.int32)
    found = tl.full([], False, tl.int1)
    for t in tl.static_range(NB // BLOCK):
        bins = t * BLOCK + lane
        c = tl.load(base + bins)
        pre = carry + tl.cumsum(c, axis=0) - c
        hit = (pre < target) & (pre + c >= target) & (not found)
        cand = tl.min(tl.where(hit, bins, NB - 1), axis=0)
        candlt = tl.max(tl.where(hit, pre, 0), axis=0)
        if (not found) & (tl.max(hit.to(tl.int32), axis=0) > 0):
            tb = cand
            lt = candlt
            found = tl.full([], True, tl.int1)
        carry += tl.sum(c, axis=0)
    return tb, lt


@triton.jit
def _merge2(
    cand_val_ptr,
    cand_idx_ptr,
    cnt_ptr,
    out_ptr,
    hist_ptr,
    counts_ptr,
    surv_ptr,
    slot_ptr,
    CAP: tl.constexpr,
    TOPK: tl.constexpr,
    NB: tl.constexpr,
    RADIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Exact top-k of a row's candidates in TWO passes over them.

    v1 ran the 32-bit radix over every candidate, four rounds plus two output
    passes -- six passes, and its per-candidate cost came out 2.7x the generic
    kernel's, which narrows first. So narrow first here too: one 11-bit
    histogram pass finds the bin holding the k-th, one partition pass sends
    the strictly-better bins straight to the output and the k-th bin's
    elements to a survivor list, and the exact 32-bit rounds then run over the
    survivors alone -- a few dozen, since 2048 bins cut finely even across the
    narrow range the candidates occupy.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    bins = tl.arange(0, RADIX)
    ones = tl.full([BLOCK], 1, tl.int32)
    n = tl.minimum(tl.load(cnt_ptr + row), CAP)
    vbase = cand_val_ptr + row * CAP
    ibase = cand_idx_ptr + row * CAP
    obase = out_ptr + row * TOPK
    hbase = hist_ptr + row * NB
    cbase = counts_ptr + row * RADIX
    sbase = surv_ptr + row * CAP
    tiles = tl.cdiv(n, BLOCK)

    if n <= TOPK:
        for t in tl.static_range((TOPK + BLOCK - 1) // BLOCK):
            j = t * BLOCK + lane
            idx = tl.load(ibase + j, mask=j < n, other=-1)
            tl.store(obase + j, tl.where(j < n, idx, -1), mask=j < TOPK)
        return

    for t in tl.static_range(NB // BLOCK):
        tl.store(hbase + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))
    tl.store(slot_ptr + 2 * row, 0)
    tl.store(slot_ptr + 2 * row + 1, 0)
    tl.debug_barrier()
    for t in tl.range(0, tiles):
        pos = t * BLOCK + lane
        valid = pos < n
        x = tl.load(vbase + pos, mask=valid, other=0.0)
        tl.atomic_add(hbase + _key11(x), ones, mask=valid, sem="relaxed", scope="cta")
    tl.debug_barrier()
    tb, nbetter = _scan_rank(hbase, TOPK, NB, BLOCK)

    slot_a = slot_ptr + 2 * row + tl.zeros([BLOCK], tl.int32)
    slot_b = slot_ptr + 2 * row + 1 + tl.zeros([BLOCK], tl.int32)
    for t in tl.range(0, tiles):
        pos = t * BLOCK + lane
        valid = pos < n
        x = tl.load(vbase + pos, mask=valid, other=0.0)
        k = _key11(x)
        idx = tl.load(ibase + pos, mask=valid, other=-1)
        better = valid & (k < tb)
        pa = tl.atomic_add(slot_a, ones, mask=better, sem="relaxed", scope="cta")
        tl.store(obase + pa, idx, mask=better & (pa < TOPK))
        eq = valid & (k == tb)
        pb = tl.atomic_add(slot_b, ones, mask=eq, sem="relaxed", scope="cta")
        tl.store(sbase + pb, pos.to(tl.int32), mask=eq & (pb < CAP))
    tl.debug_barrier()
    ns = tl.minimum(tl.load(slot_ptr + 2 * row + 1), CAP)
    stiles = tl.cdiv(ns, BLOCK)

    desired = tl.zeros((), tl.uint32)
    desired_mask = tl.zeros((), tl.uint32)
    k_to_find = TOPK - nbetter + 1
    for digit_pos in tl.static_range(24, -1, -8):
        if k_to_find > 1:
            tl.store(cbase + bins, tl.zeros([RADIX], tl.int32))
            tl.debug_barrier()
            for t in tl.range(0, stiles):
                j = t * BLOCK + lane
                valid = j < ns
                pos = tl.load(sbase + j, mask=valid, other=0)
                x = tl.load(vbase + pos, mask=valid, other=0.0)
                key = _u32(x)
                digit = ((key >> digit_pos) & (RADIX - 1)).to(tl.int32)
                tl.atomic_add(
                    cbase + digit,
                    ones,
                    mask=valid & ((key & desired_mask) == desired),
                    sem="relaxed",
                    scope="cta",
                )
            tl.debug_barrier()
            counts = tl.load(cbase + bins)
            prefix = tl.cumsum(counts, axis=0) - counts
            hit = (prefix < k_to_find) & (prefix + counts >= k_to_find)
            rb = tl.min(tl.where(hit, bins, RADIX), axis=0).to(tl.int32)
            rb = tl.where(rb == RADIX, RADIX - 1, rb)
            counts_lt = tl.max(tl.where(bins == rb, prefix, 0), axis=0).to(tl.int32)
            desired = desired | (rb.to(tl.uint32) << digit_pos)
            desired_mask = desired_mask | (
                tl.full((), RADIX - 1, tl.uint32) << digit_pos
            )
            k_to_find = k_to_find - counts_lt

    thr_key = desired
    tl.store(slot_ptr + 2 * row, nbetter)
    tl.debug_barrier()
    for equal in tl.static_range(2):
        for t in tl.range(0, stiles):
            j = t * BLOCK + lane
            valid = j < ns
            pos = tl.load(sbase + j, mask=valid, other=0)
            x = tl.load(vbase + pos, mask=valid, other=0.0)
            key = _u32(x)
            if equal == 0:
                take = valid & (key < thr_key)
            else:
                take = valid & (key == thr_key)
            idx = tl.load(ibase + pos, mask=take, other=-1)
            pa = tl.atomic_add(slot_a, ones, mask=take, sem="relaxed", scope="cta")
            tl.store(obase + pa, idx, mask=take & (pa < TOPK))
        tl.debug_barrier()


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    gen = _generic
    dev = "cuda"
    block = gen.NUM_THREADS_PER_BLOCK
    print(
        f"top_k {TOPK}, buffer {CAP}; device us for the whole tail "
        f"(generic merge + remap, against one kernel)\n"
    )
    print(
        f"  {'rows':>5} {'cands':>6} {'generic':>9}"
        + "".join(
            f" {v:>8} {'x':>6} {'geom':>9} {'ans':>6}" for v in ("v1 all", "v2 narrow")
        )
    )

    for rows in ROWS:
        for c in CANDS:
            torch.manual_seed(rows * 31 + c)
            cand_val = torch.randn((rows, CAP), dtype=torch.float32, device=dev)
            cand_idx = (
                torch.arange(CAP, dtype=torch.int32, device=dev)
                .repeat(rows, 1)
                .contiguous()
            )
            cnt = torch.full((rows,), c, dtype=torch.int32, device=dev)
            out = torch.empty((rows, TOPK), dtype=torch.int32, device=dev)
            merged = torch.empty((rows, TOPK), dtype=torch.int32, device=dev)
            counts = torch.empty((rows, RADIX), dtype=torch.int32, device=dev)
            slot = torch.empty((rows * 2,), dtype=torch.int32, device=dev)
            hist = torch.empty((rows, NB), dtype=torch.int32, device=dev)
            surv = torch.empty((rows, CAP), dtype=torch.int32, device=dev)
            scratch = (
                torch.empty((rows, gen.NUM_BINS), dtype=torch.int32, device=dev),
                torch.empty(
                    (rows, gen.NUM_FILNAL_ITEMS), dtype=torch.float32, device=dev
                ),
                torch.empty((rows,), dtype=torch.int32, device=dev),
                torch.empty((rows,), dtype=torch.int32, device=dev),
                torch.empty((rows,), dtype=torch.int32, device=dev),
                torch.empty((rows,), dtype=torch.int32, device=dev),
            )
            want = torch.topk(cand_val[:, :c], TOPK, dim=1).values.sort(dim=1).values

            lmerge = ov._Launch(
                gen.non_tle_top_k_per_row_decode,
                (rows,),
                {"TOPK": TOPK, "BLOCK_SIZE": block},
                gen._num_warps(block),
            )
            lremap = ov._Launch(
                ov._remap,
                (rows,),
                {"CAP": CAP, "TOPK": TOPK, "BLOCK": triton.next_power_of_2(TOPK)},
                4,
            )

            def old():
                lmerge(cand_val, merged, cnt, 1, CAP, 1, CAP, *scratch)
                lremap(cand_idx, merged, out)

            t_old = device_us(old)
            bests = []
            for version in (1, 2):
                best = (1e9, None, False)
                for blk, warps in GEOMS:
                    if version == 1:
                        lded = ov._Launch(
                            _merge,
                            (rows,),
                            {
                                "CAP": CAP,
                                "TOPK": TOPK,
                                "RADIX": RADIX,
                                "BLOCK": blk,
                            },
                            warps,
                        )

                        def new(lded=lded):
                            lded(cand_val, cand_idx, cnt, out, counts, slot)

                    else:
                        lded = ov._Launch(
                            _merge2,
                            (rows,),
                            {
                                "CAP": CAP,
                                "TOPK": TOPK,
                                "NB": NB,
                                "RADIX": RADIX,
                                "BLOCK": blk,
                            },
                            warps,
                        )

                        def new(lded=lded):
                            lded(
                                cand_val,
                                cand_idx,
                                cnt,
                                out,
                                hist,
                                counts,
                                surv,
                                slot,
                            )

                    out.fill_(-9)
                    new()
                    torch.cuda.synchronize()
                    got = (
                        cand_val.gather(1, out.long().clamp(0, CAP - 1))
                        .sort(dim=1)
                        .values
                    )
                    ok = torch.allclose(got, want) and bool((out >= 0).all())
                    t = device_us(new)
                    if t < best[0]:
                        best = (t, f"{blk}x{warps}", ok)
                bests.append(best)
            print(
                f"  {rows:>5} {c:>6} {t_old:>9.1f}"
                + "".join(
                    f" {b[0]:>8.1f} {t_old / b[0]:>6.2f} {b[1]:>9}"
                    f" {'OK' if b[2] else 'WRONG':>6}"
                    for b in bests
                )
            )


if __name__ == "__main__":
    sys.exit(main())
