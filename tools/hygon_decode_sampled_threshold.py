"""Decode with a sampled threshold: can one pass replace stage one's two?

Stage one of the split pipeline runs the full radix algorithm per chunk --
a histogram pass over the chunk, a threshold scan, then a second pass that
writes the definite elements and collects the threshold bin. It is 91-94% of
decode's device time, and geometry, split factor and the helper kernels are
all exhausted (0.535-0.805 of vLLM at 16-56 rows).

But stage one does more than is needed: the merge only needs a SUPERSET of the
row's top-k, not each chunk's exact top-k. So:

    sample   histogram of every STRIDE-th TILE          (~1/STRIDE of a pass)
    thresh   scan it for the bin holding rank k/STRIDE, with a safety factor
    select   ONE pass over the row, appending every element at or above that
             bin to a per-row candidate buffer (index and value)
    merge    the existing kernel over the candidates, then remap

That is ~1.06 passes against 2. The estimate is not a bound, so a row whose
candidate count exceeds the buffer (ties: the 8-LSB test has 256 distinct
values over 262144 elements) or falls short of k must fall back to the exact
pipeline -- which needs the count on the host, i.e. a device sync this
operator does not do today. The sync is timed here as part of the cost.

Reports, per row count: wall us and ratio for the shipped pipeline and for
this one, how many candidates the threshold actually admits, how often it
overflows or undershoots, and correctness against torch.topk.

    tools/vendor_probe.sh tools/hygon_decode_sampled_threshold.py hygon_decode_sampled
"""

import sys
from importlib import import_module

import torch
import triton
import triton.language as tl

import flaggems_vllm

V, K = 262144, 512
ROWS = (1, 4, 8, 16, 24, 32, 56)
NB = 2048  # bins, as in the operator's STEP 0
STRIDE = 16  # sample one element in STRIDE
SAFETY = 2  # admit ~SAFETY * k elements
CAP = 8192  # candidate buffer per row
BLOCK = 512
WARPS = 8


@triton.jit
def _key(x):
    """The operator's STEP-0 key: fp16 bits mapped so that ascending uint16
    means descending float, then the top 11 bits."""
    h = x.to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    sign_set = (bits & tl.full(bits.shape, 0x8000, tl.uint16)) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
    mapped = tl.where(sign_set, bits, inv)
    return (mapped >> 5).to(tl.int32)


@triton.jit
def k_zero_hist(hist_ptr, NB: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    for t in tl.static_range(NB // BLOCK):
        tl.store(hist_ptr + row * NB + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))


@triton.jit
def k_sample_hist(
    logits_ptr,
    seq_ptr,
    hist_ptr,
    stride0,
    NB: tl.constexpr,
    STRIDE: tl.constexpr,
    CHUNK: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Every STRIDE-th TILE, one program per chunk.

    Round 1 sampled every STRIDE-th ELEMENT from one program per row and the
    whole pipeline came out flat at ~420 us, worse than the shipped path: a
    strided element sample touches one cache line per value, so it reads all
    of the row's lines for 1/STRIDE of the data, and one program per row does
    not shrink with row count. Contiguous tiles read 1/STRIDE of the bytes.
    """
    pid = tl.program_id(0)
    row = pid // SPLIT
    chunk = pid % SPLIT
    lane = tl.arange(0, BLOCK)
    base = hist_ptr + row * NB
    n = tl.load(seq_ptr + row)
    start = chunk * CHUNK
    end = tl.minimum(start + CHUNK, n)
    for t in tl.range(0, tl.cdiv(CHUNK, BLOCK * STRIDE)):
        i = start + t * BLOCK * STRIDE + lane
        m = i < end
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        tl.atomic_add(
            base + _key(x),
            tl.full([BLOCK], 1, tl.int32),
            mask=m,
            sem="relaxed",
            scope="cta",
        )


@triton.jit
def k_threshold(
    hist_ptr, thr_ptr, TARGET: tl.constexpr, NB: tl.constexpr, BLOCK: tl.constexpr
):
    """Lowest bin index whose prefix count reaches TARGET; bin 0 holds the
    largest values, so 'at or above' means bin_idx <= thr."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = hist_ptr + row * NB
    carry = tl.zeros([], tl.int32)
    thr = tl.full([], NB - 1, tl.int32)
    found = tl.full([], False, tl.int1)
    for t in tl.static_range(NB // BLOCK):
        bins = t * BLOCK + lane
        c = tl.load(base + bins)
        pre = carry + tl.cumsum(c, axis=0)
        hit = (pre >= TARGET) & (not found)
        cand = tl.min(tl.where(hit, bins, NB - 1), axis=0)
        if (not found) & (tl.max(hit.to(tl.int32), axis=0) > 0):
            thr = cand
            found = tl.full([], True, tl.int1)
        carry += tl.sum(c, axis=0)
    tl.store(thr_ptr + row, thr)


@triton.jit
def k_select(
    logits_ptr,
    seq_ptr,
    thr_ptr,
    cnt_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    stride0,
    CHUNK: tl.constexpr,
    SPLIT: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One pass: append every element at or above the row's threshold bin."""
    pid = tl.program_id(0)
    row = pid // SPLIT
    chunk = pid % SPLIT
    lane = tl.arange(0, BLOCK)
    thr = tl.load(thr_ptr + row)
    n = tl.load(seq_ptr + row)
    start = chunk * CHUNK
    end = tl.minimum(start + CHUNK, n)
    cnt_ptrs = cnt_ptr + row + tl.zeros([BLOCK], tl.int32)
    for t in tl.range(0, tl.cdiv(CHUNK, BLOCK)):
        i = start + t * BLOCK + lane
        m = i < end
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        take = m & (_key(x) <= thr)
        pos = tl.atomic_add(
            cnt_ptrs,
            tl.full([BLOCK], 1, tl.int32),
            mask=take,
            sem="relaxed",
            scope="cta",
        )
        keep = take & (pos < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos, i.to(tl.int32), mask=keep)
        tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)


@triton.jit
def k_remap(
    cand_idx_ptr,
    merged_ptr,
    out_ptr,
    CAP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    m = j < TOPK
    pos = tl.load(merged_ptr + row * TOPK + j, mask=m, other=-1)
    live = m & (pos >= 0)
    idx = tl.load(cand_idx_ptr + row * CAP + pos, mask=live, other=-1)
    tl.store(out_ptr + row * TOPK + j, tl.where(live, idx, -1), mask=m)


def wall_us(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000


def main():
    gen = import_module("flaggems_vllm.ops.top_k_per_row_decode")
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    import vllm._custom_ops  # noqa: F401

    dev = "cuda"
    target = max(1, (K // STRIDE) * SAFETY)
    print(
        f"vocab {V}, top_k {K}; sample 1/{STRIDE}, admit rank {target} of the "
        f"sample (~{target * STRIDE} elements), buffer {CAP}\n"
    )
    print(
        f"  {'rows':>4} {'split':>5} {'shipped us':>11} {'sampled us':>11} "
        f"{'+sync us':>9} {'ratio now':>10} {'ratio new':>10} {'cands':>7} {'answer':>8}"
    )
    for rows in ROWS:
        torch.manual_seed(rows)
        logits = torch.randn(rows, V, dtype=torch.float32, device=dev)
        want = torch.topk(logits, K, dim=1).values.sort(dim=1).values
        lens = torch.full((rows,), V, dtype=torch.int32, device=dev)
        idx = torch.empty(rows, K, dtype=torch.int32, device=dev)
        split = ov._split_factor(rows, V, K)
        chunk = V // split

        hist = torch.empty(rows * NB, dtype=torch.int32, device=dev)
        thr = torch.empty(rows, dtype=torch.int32, device=dev)
        cnt = torch.zeros(rows, dtype=torch.int32, device=dev)
        cand_idx = torch.empty(rows * CAP, dtype=torch.int32, device=dev)
        cand_val = torch.empty(rows * CAP, dtype=torch.float32, device=dev)
        merged = torch.empty(rows, K, dtype=torch.int32, device=dev)
        mlens = torch.empty(rows, dtype=torch.int32, device=dev)
        scratch = [
            torch.empty(s, dtype=d, device=dev)
            for s, d in (
                ((rows, gen.NUM_BINS), torch.int32),
                ((rows, gen.NUM_FILNAL_ITEMS), torch.float32),
                ((rows,), torch.int32),
                ((rows,), torch.int32),
                ((rows,), torch.int32),
                ((rows,), torch.int32),
            )
        ]

        def sampled(sync=False):
            cnt.zero_()
            k_zero_hist[(rows,)](hist, NB=NB, BLOCK=BLOCK, num_warps=WARPS)
            k_sample_hist[(rows * split,)](
                logits,
                lens,
                hist,
                V,
                NB=NB,
                STRIDE=STRIDE,
                CHUNK=chunk,
                SPLIT=split,
                BLOCK=BLOCK,
                num_warps=WARPS,
            )
            k_threshold[(rows,)](
                hist, thr, TARGET=target, NB=NB, BLOCK=BLOCK, num_warps=WARPS
            )
            k_select[(rows * split,)](
                logits,
                lens,
                thr,
                cnt,
                cand_idx,
                cand_val,
                V,
                CHUNK=chunk,
                SPLIT=split,
                CAP=CAP,
                BLOCK=BLOCK,
                num_warps=WARPS,
            )
            mlens.copy_(cnt.clamp(max=CAP))
            gen.non_tle_top_k_per_row_decode[(rows,)](
                cand_val,
                merged,
                mlens,
                1,
                CAP,
                1,
                CAP,
                *scratch,
                TOPK=K,
                BLOCK_SIZE=gen.NUM_THREADS_PER_BLOCK,
                num_warps=gen._num_warps(gen.NUM_THREADS_PER_BLOCK),
            )
            k_remap[(rows,)](
                cand_idx,
                merged,
                idx,
                CAP=CAP,
                TOPK=K,
                BLOCK=triton.next_power_of_2(K),
                num_warps=4,
            )
            if sync:
                return int(cnt.max()), int(cnt.min())
            return None

        t_vllm = wall_us(
            lambda: torch.ops._C.top_k_per_row_decode(
                logits, 1, lens, idx, rows, V, 1, K
            )
        )
        t_ship = wall_us(
            lambda: flaggems_vllm.top_k_per_row_decode(
                logits, 1, lens, idx, rows, V, 1, K
            )
        )
        hi, lo = sampled(sync=True)
        torch.cuda.synchronize()
        got = logits.gather(1, idx.long().clamp(0, V - 1)).sort(dim=1).values
        ok = torch.allclose(got, want) and bool((idx >= 0).all())
        t_new = wall_us(lambda: sampled(False))
        t_sync = wall_us(lambda: sampled(True))
        print(
            f"  {rows:>4} {split:>5} {t_ship:>11.1f} {t_new:>11.1f} {t_sync:>9.1f} "
            f"{t_vllm / t_ship:>10.3f} {t_vllm / t_sync:>10.3f} {hi:>7} "
            f"{'OK' if ok else 'WRONG':>8}"
        )
    print(
        "\n  'cands' is the largest candidate count admitted (buffer is "
        f"{CAP}); under {K} would mean the threshold was too strict."
    )
    print("  'ratio new' includes the sync that the fallback decision needs.")


if __name__ == "__main__":
    sys.exit(main())
