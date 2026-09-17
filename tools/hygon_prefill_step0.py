"""Does prefill's STEP 0 converge, and could it use fewer bins?

STEP 0 histograms the row into 2048 bins on an 11-bit fp16 key, finds the bin
holding rank top_k, and then bets that this bin holds at most
NUM_FINAL_ITEMS = 2048 elements -- in which case its contents go straight to
the exact final select and STEP 1-3 never run. If the bet usually fails, those
steps are real passes and STEP 0 is a pass spent to skip nothing.

Nobody has checked which happens on this card, and the answer settles two
questions at once, because both turn on the size of that one bin:

  does STEP 0 pay for itself   bin <= 2048 means steps 1-3 are dead code for
                               these shapes, so STEP 0 IS the algorithm and
                               removing it would cost, not save
  can STEP 0 use fewer bins    halving the bins roughly doubles the threshold
                               bin. 512 bins would make the clear and the scan
                               a quarter of what they are, but only if the bin
                               still fits under 2048

So this measures the threshold bin's size per row at 2048, 1024, 512 and 256
bins, over every benchmark shape, and reports the distribution -- the median
row, the worst row, and how many rows would overflow NUM_FINAL_ITEMS.

This is a property of the data and the key, not a timing, so it needs no
interleaving and no repetition.

    tools/vendor_probe.sh tools/hygon_prefill_step0.py hygon_prefill_step0
"""

import sys
from importlib import import_module

import torch
import triton
import triton.language as tl

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
_key11 = _generic._convert_to_trt_uint16_hi11
NUM_FINAL_ITEMS = 2048

# (num_rows, vocab, top_k, stride0) -- the benchmark's shapes
SHAPES = [
    (64, 129280, 1024, 129280),
    (4, 8193, 512, 8456),
    (4, 16385, 512, 16648),
    (16383, 4095, 512, 4352),
    (12961, 4100, 512, 4360),
    (16380, 5115, 512, 5376),
    (4100, 1025, 512, 1288),
]
BINS = (2048, 1024, 512, 256)


@triton.jit
def _bin_stats(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    binsize_ptr,
    nlt_ptr,
    stride0,
    TOPK: tl.constexpr,
    NB: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
):
    """One program per row: histogram on the operator's own key, shifted down
    to NB bins, then the bin holding rank TOPK, its size, and how many
    elements are strictly better."""
    row = tl.program_id(0)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    n = e - s
    base = hist_ptr + row * NB
    bins = tl.arange(0, NB)
    tl.store(base + bins, tl.zeros([NB], tl.int32))
    tl.debug_barrier()
    lane = tl.arange(0, BLOCK)[:, None] * VEC + tl.arange(0, VEC)[None, :]
    for t in tl.range(0, tl.cdiv(n, BLOCK * VEC)):
        i = t * BLOCK * VEC + lane
        m = i < n
        x = tl.load(logits_ptr + row * stride0 + s + i, mask=m, other=0.0)
        tl.atomic_add(
            base + (_key11(x) >> SHIFT),
            tl.full([BLOCK, VEC], 1, tl.int32),
            mask=m,
            sem="relaxed",
            scope="cta",
        )
    tl.debug_barrier()
    counts = tl.load(base + bins)
    pre = tl.cumsum(counts, axis=0) - counts
    hit = (pre < TOPK) & (pre + counts >= TOPK)
    thr = tl.min(tl.where(hit, bins, NB - 1), axis=0).to(tl.int32)
    tl.store(binsize_ptr + row, tl.max(tl.where(bins == thr, counts, 0), axis=0))
    tl.store(nlt_ptr + row, tl.max(tl.where(bins == thr, pre, 0), axis=0))


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    dev = "cuda"
    print(
        f"threshold-bin size per row; the operator keeps at most "
        f"{NUM_FINAL_ITEMS} of them\n"
    )
    for rows, vocab, top_k, stride0 in SHAPES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        binsize = torch.empty((rows,), dtype=torch.int32, device=dev)
        nlt = torch.empty((rows,), dtype=torch.int32, device=dev)
        print(f"  {rows} x {vocab}, top_k {top_k}")
        print(
            f"    {'bins':>5} {'median':>8} {'p99':>8} {'worst':>8} "
            f"{'rows over':>10} {'nlt median':>11}"
        )
        for nb in BINS:
            hist = torch.empty((rows, nb), dtype=torch.int32, device=dev)
            block = min(512, triton.next_power_of_2(max(vocab // 4, 64)))
            ov._Launch(
                _bin_stats,
                (rows,),
                {
                    "TOPK": top_k,
                    "NB": nb,
                    "SHIFT": 11 - int(nb).bit_length() + 1,
                    "BLOCK": block,
                    "VEC": 4,
                },
                8,
            )(logits, starts, ends, hist, binsize, nlt, stride0)
            torch.cuda.synchronize()
            b = binsize.float()
            over = int((binsize > NUM_FINAL_ITEMS).sum())
            print(
                f"    {nb:>5} {b.median().item():>8.0f} "
                f"{b.quantile(0.99).item():>8.0f} {b.max().item():>8.0f} "
                f"{over:>4} / {rows:<5} {nlt.float().median().item():>11.0f}",
                flush=True,
            )
        print(flush=True)
    print(
        "  'rows over' is how many rows would blow NUM_FINAL_ITEMS at that bin"
        "\n  count, i.e. how many would need STEP 1-3 to run at all. Zero at"
        "\n  2048 bins means those steps are dead code today and STEP 0 is the"
        "\n  whole algorithm; zero at 512 means the bins can be cut."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
