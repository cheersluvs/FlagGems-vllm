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
candidate count exceeds the buffer or falls short of k has to be redone
exactly -- and the operator does not sync, so that decision has to be made on
the device.

Rounds 1 and 2 both came out flat at ~420-460 us whatever the row count, and
both times the flatness was the probe, not the idea:

  round 1  sampled every STRIDE-th ELEMENT: one cache line per value, so it
           read the whole row for 1/STRIDE of the data
  round 2  launched all five kernels through the JIT: ~110 us of host dispatch
           each, 550 us of serial host work hiding every device difference,
           while the shipped pipeline it was compared against uses cached
           direct launches at ~12 us

Round 3: every launch direct (the recipe from hygon_decode_split_direct.py),
no torch ops in the pipeline (zeroing and clamping are kernels), and the
sample is one program per ROW over 1/64 of the tiles -- with one program per
chunk the ~19 us per-program floor dominated a pass that only reads 1/16.

Round 4, once round 3 showed the idea pays (0.85-1.54 of vLLM against the
shipped path's 0.55-0.92, and five of the seven stages sitting at the ~11 us
launch floor):

  * zero, sample and threshold fuse into ONE kernel. All three are one program
    per row and that program owns the row's histogram outright, so a
    tl.debug_barrier() between the phases is all the ordering they need.
  * the host sync is gone. A fixup kernel reads the count on the device: rows
    in range just get their merge length written and return; a row out of
    range is redone exactly inside the kernel -- histogram, rank-k threshold,
    then the definite bins appended before the threshold bin, so a buffer that
    still overflows can only drop elements that share an 11-bit key.
  * the merge-length clamp folds into the fixup, so the pipeline is five
    launches against round 3's seven, with no sync.

Correctness is checked on random normals (which never leave the range, so they
exercise the fast path) AND on a tie-heavy row of few distinct values, which
does.

Reports, per row count: wall us and ratio for the shipped pipeline and for
this one, how many candidates the threshold admits, and both correctness
checks.

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
STRIDE = 64  # sample one tile in STRIDE
SAFETY = 4  # admit ~SAFETY * k elements
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
def _scan_threshold(base, TARGET, NB: tl.constexpr, BLOCK: tl.constexpr):
    """Lowest bin index whose prefix count reaches TARGET; bin 0 holds the
    largest values, so 'at or above' means bin_idx <= thr."""
    lane = tl.arange(0, BLOCK)
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
    return thr


@triton.jit
def _hist_pass(
    logits_ptr, base, row, stride0, n, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    """Histogram every STRIDE-th tile of the row into base (STRIDE=1: all)."""
    lane = tl.arange(0, BLOCK)
    for t in tl.range(0, tl.cdiv(n, BLOCK * STRIDE)):
        i = t * BLOCK * STRIDE + lane
        m = i < n
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        tl.atomic_add(
            base + _key(x),
            tl.full([BLOCK], 1, tl.int32),
            mask=m,
            sem="relaxed",
            scope="cta",
        )


@triton.jit
def k_prepare(
    logits_ptr,
    seq_ptr,
    hist_ptr,
    thr_ptr,
    cnt_ptr,
    stride0,
    TARGET: tl.constexpr,
    NB: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Zero, sample and threshold in one program per row.

    Tiles, not elements: a strided element sample touches one cache line per
    value and costs a whole pass. One program per row, not per chunk: the
    kernel floor is ~11 us per launch and ~19 us per program, and the sample
    only reads 1/STRIDE of the bytes, so splitting it buys nothing.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = hist_ptr + row * NB
    for t in tl.static_range(NB // BLOCK):
        tl.store(base + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))
    tl.store(cnt_ptr + row, 0)
    tl.debug_barrier()
    _hist_pass(logits_ptr, base, row, stride0, tl.load(seq_ptr + row), STRIDE, BLOCK)
    tl.debug_barrier()
    tl.store(thr_ptr + row, _scan_threshold(base, TARGET, NB, BLOCK))


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
def _select_pass(
    logits_ptr,
    row,
    stride0,
    n,
    thr,
    exact,
    cnt_ptrs,
    cand_idx_ptr,
    cand_val_ptr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Append the row's elements whose key is <= thr (exact: == thr)."""
    lane = tl.arange(0, BLOCK)
    for t in tl.range(0, tl.cdiv(n, BLOCK)):
        i = t * BLOCK + lane
        m = i < n
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        k = _key(x)
        if exact:
            take = m & (k == thr)
        else:
            take = m & (k < thr)
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
def k_fixup(
    logits_ptr,
    seq_ptr,
    hist_ptr,
    cnt_ptr,
    mlen_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    stride0,
    TOPK: tl.constexpr,
    NB: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The device-side stand-in for a host sync.

    A row whose sampled threshold admitted between TOPK and CAP candidates is
    already a superset of its top-k: write the merge length and leave. A row
    outside that range is redone exactly here -- full histogram, threshold at
    rank TOPK, then the strictly-better bins appended BEFORE the threshold bin
    so that a buffer which still overflows can only ever drop elements sharing
    an 11-bit key with the k-th.
    """
    row = tl.program_id(0)
    c = tl.load(cnt_ptr + row)
    if (c >= TOPK) & (c <= CAP):
        tl.store(mlen_ptr + row, c)
        return
    lane = tl.arange(0, BLOCK)
    base = hist_ptr + row * NB
    n = tl.load(seq_ptr + row)
    for t in tl.static_range(NB // BLOCK):
        tl.store(base + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))
    tl.debug_barrier()
    _hist_pass(logits_ptr, base, row, stride0, n, 1, BLOCK)
    tl.debug_barrier()
    thr = _scan_threshold(base, TOPK, NB, BLOCK)
    tl.store(cnt_ptr + row, 0)
    tl.debug_barrier()
    cnt_ptrs = cnt_ptr + row + tl.zeros([BLOCK], tl.int32)
    _select_pass(
        logits_ptr,
        row,
        stride0,
        n,
        thr,
        False,
        cnt_ptrs,
        cand_idx_ptr,
        cand_val_ptr,
        CAP,
        BLOCK,
    )
    tl.debug_barrier()
    _select_pass(
        logits_ptr,
        row,
        stride0,
        n,
        thr,
        True,
        cnt_ptrs,
        cand_idx_ptr,
        cand_val_ptr,
        CAP,
        BLOCK,
    )
    tl.debug_barrier()
    tl.store(mlen_ptr + row, tl.minimum(tl.load(cnt_ptr + row), CAP))


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
    stages = []
    print(
        f"  {'rows':>4} {'split':>5} {'shipped us':>11} {'sampled us':>11} "
        f"{'fixup us':>9} {'ratio now':>10} {'ratio new':>10} {'cands':>7} "
        f"{'normal':>7} {'tied':>7}"
    )
    for rows in ROWS:
        torch.manual_seed(rows)
        logits = torch.randn(rows, V, dtype=torch.float32, device=dev)
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

        L = ov._Launch
        lprep = L(
            k_prepare,
            (rows,),
            {"TARGET": target, "NB": NB, "STRIDE": STRIDE, "BLOCK": BLOCK},
            WARPS,
        )
        lsel = L(
            k_select,
            (rows * split,),
            {"CHUNK": chunk, "SPLIT": split, "CAP": CAP, "BLOCK": BLOCK},
            WARPS,
        )
        lfix = L(
            k_fixup,
            (rows,),
            {"TOPK": K, "NB": NB, "CAP": CAP, "BLOCK": BLOCK},
            WARPS,
        )
        lmerge = L(
            gen.non_tle_top_k_per_row_decode,
            (rows,),
            {"TOPK": K, "BLOCK_SIZE": gen.NUM_THREADS_PER_BLOCK},
            gen._num_warps(gen.NUM_THREADS_PER_BLOCK),
        )
        lremap = L(
            k_remap,
            (rows,),
            {"CAP": CAP, "TOPK": K, "BLOCK": triton.next_power_of_2(K)},
            4,
        )

        def sampled(src):
            lprep(src, lens, hist, thr, cnt, V)
            lsel(src, lens, thr, cnt, cand_idx, cand_val, V)
            lfix(src, lens, hist, cnt, mlens, cand_idx, cand_val, V)
            lmerge(cand_val, merged, mlens, 1, CAP, 1, CAP, *scratch)
            lremap(cand_idx, merged, idx)

        def check(src):
            want = torch.topk(src, K, dim=1).values.sort(dim=1).values
            sampled(src)
            torch.cuda.synchronize()
            got = src.gather(1, idx.long().clamp(0, V - 1)).sort(dim=1).values
            return torch.allclose(got, want) and bool((idx >= 0).all())

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
        ok = check(logits)
        hi = int(cnt.max())
        # few distinct values: the sampled threshold overflows, so every row
        # takes the fixup's exact path
        ok_tied = check((logits * 4).round() / 4)
        t_new = wall_us(lambda: sampled(logits))
        sampled(logits)
        t_fix = wall_us(
            lambda: lfix(logits, lens, hist, cnt, mlens, cand_idx, cand_val, V)
        )
        stages.append(
            (
                rows,
                [
                    wall_us(f)
                    for f in (
                        lambda: lprep(logits, lens, hist, thr, cnt, V),
                        lambda: lsel(logits, lens, thr, cnt, cand_idx, cand_val, V),
                        lambda: lfix(
                            logits, lens, hist, cnt, mlens, cand_idx, cand_val, V
                        ),
                        lambda: lmerge(
                            cand_val, merged, mlens, 1, CAP, 1, CAP, *scratch
                        ),
                        lambda: lremap(cand_idx, merged, idx),
                    )
                ],
            )
        )
        print(
            f"  {rows:>4} {split:>5} {t_ship:>11.1f} {t_new:>11.1f} {t_fix:>9.1f} "
            f"{t_vllm / t_ship:>10.3f} {t_vllm / t_new:>10.3f} {hi:>7} "
            f"{'OK' if ok else 'WRONG':>7} {'OK' if ok_tied else 'WRONG':>7}"
        )
    print(
        "\n  'cands' is the largest candidate count admitted (buffer is "
        f"{CAP}); under {K} would mean the threshold was too strict."
    )
    print(
        "  'fixup us' is the fallback check launched alone on rows that are "
        "all in range: what the no-sync check costs when it fires on nothing."
    )
    names = ("prepare", "select", "fixup", "merge", "remap")
    print(
        "\n  per-stage wall us (each launch timed alone, so each carries one"
        " host submit)\n"
    )
    print("  " + f"{'rows':>4}" + "".join(f"{n:>9}" for n in names) + f"{'sum':>9}")
    for rows, ts in stages:
        print(
            "  " + f"{rows:>4}" + "".join(f"{t:>9.1f}" for t in ts) + f"{sum(ts):>9.1f}"
        )


if __name__ == "__main__":
    sys.exit(main())
