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

CAP = 8192
TOPK = 512
ROWS = (4, 16, 64, 512)
CANDS = (1024, 2048, 4096, 8192)
GEOMS = [(256, 4), (512, 4), (512, 8), (1024, 8), (1024, 16)]
RADIX = 256


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
        f"  {'rows':>5} {'cands':>6} {'generic':>9} {'dedicated':>10} "
        f"{'speedup':>8} {'geom':>10} {'answer':>8}"
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
            slot = torch.empty((rows,), dtype=torch.int32, device=dev)
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
            best = (1e9, None, False)
            for blk, warps in GEOMS:
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

                out.fill_(-9)
                new()
                torch.cuda.synchronize()
                got = (
                    cand_val.gather(1, out.long().clamp(0, CAP - 1)).sort(dim=1).values
                )
                ok = torch.allclose(got, want) and bool((out >= 0).all())
                t = device_us(new)
                if t < best[0]:
                    best = (t, f"{blk}x{warps}", ok)
            print(
                f"  {rows:>5} {c:>6} {t_old:>9.1f} {best[0]:>10.1f} "
                f"{t_old / best[0]:>8.2f} {best[1]:>10} "
                f"{'OK' if best[2] else 'WRONG':>8}"
            )


if __name__ == "__main__":
    sys.exit(main())
