"""Stage-split against the shipped prefill, interleaved, to kill the drift.

tools/hygon_prefill_stagesplit.py confirmed what the counters predicted: on
(64,129280) the stage-split's device time falls monotonically as workgroups go
64 -> 512 (517 -> 275 us), and the same on both four-row shapes. The occupancy
diagnosis holds.

What it could NOT settle is whether that beats the shipped operator, because
the denominator moves: this shape's shipped device time has measured 232.3,
238.1, 310.7 and 362.0 us across runs -- it is the only shape with fewer
programs than SMs and the only one that drifts like that. 274.7 sits in the
middle of that spread, so "1.32x" may be mostly a high denominator.

So measure them ALTERNATELY in one process, per-call CUDA events with the
median taken the way do_bench does, several rounds. Drift then hits both
implementations equally and the per-round ratios show their own spread.

Two things the previous run established and this one keeps in view: at split 1
the pipeline is 1.4x SLOWER than the shipped kernel (517 against 362), so all
of its gain comes from parallelism and none from the pipeline itself; and its
2.5x on the four-row shapes' WALL time is only cached direct launch beating
Triton's dispatch, which kernel mode does not see -- there it is 1.03x.

    tools/vendor_probe.sh tools/hygon_prefill_ab.py hygon_prefill_ab
"""

import sys
from importlib import import_module

import torch
import triton
import triton.language as tl

import flaggems_vllm

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
_key11 = _generic._convert_to_trt_uint16_hi11

# (num_rows, vocab, top_k, stride0, split, BLOCK, warps) -- the split and
# geometry each shape measured best at in the sweep
CASES = [
    (64, 129280, 1024, 129280, 8, 256, 4),
    (4, 16385, 512, 16648, 8, 512, 8),
    (4, 8193, 512, 8456, 4, 512, 8),
]
NB = 2048
VEC = 4
RADIX = 256


@triton.jit
def _hist_split(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    stride0,
    CHUNK: tl.constexpr,
    SPLIT: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
):
    """SPLIT programs share one row's histogram, so the atomic is device
    scoped. Vectorised like the operator's own loop: each lane takes VEC
    consecutive elements."""
    pid = tl.program_id(0)
    row = pid // SPLIT
    chunk = pid % SPLIT
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    start = s + chunk * CHUNK
    end = tl.minimum(start + CHUNK, e)
    base = hist_ptr + row * NB
    lane = tl.arange(0, BLOCK)[:, None] * VEC + tl.arange(0, VEC)[None, :]
    for t in tl.range(0, tl.cdiv(CHUNK, BLOCK * VEC)):
        i = start + t * BLOCK * VEC + lane
        m = i < end
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        tl.atomic_add(
            base + _key11(x),
            tl.full([BLOCK, VEC], 1, tl.int32),
            mask=m,
            sem="relaxed",
            scope="gpu",
        )


@triton.jit
def _thresh(
    hist_ptr,
    thr_ptr,
    cnt_a_ptr,
    cnt_b_ptr,
    TOPK: tl.constexpr,
    NB: tl.constexpr,
):
    """One program per row: the bin holding rank TOPK and how many are
    strictly better, then zero the histogram for the next call."""
    row = tl.program_id(0)
    bins = tl.arange(0, NB)
    base = hist_ptr + row * NB
    counts = tl.load(base + bins)
    pre = tl.cumsum(counts, axis=0) - counts
    hit = (pre < TOPK) & (pre + counts >= TOPK)
    thr = tl.min(tl.where(hit, bins, NB - 1), axis=0).to(tl.int32)
    nlt = tl.max(tl.where(bins == thr, pre, 0), axis=0).to(tl.int32)
    tl.store(thr_ptr + row, thr)
    tl.store(cnt_a_ptr + row, 0)
    tl.store(cnt_b_ptr + row, nlt)
    tl.store(base + bins, tl.zeros([NB], tl.int32))


@triton.jit
def _select_split(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    thr_ptr,
    cnt_a_ptr,
    cnt_b_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    stride0,
    CHUNK: tl.constexpr,
    SPLIT: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
):
    """Strictly-better elements take slots [0, nlt) through counter A and the
    threshold bin's take [nlt, CAP) through counter B, which thresh seeded at
    nlt. The two regions cannot collide, so the programs need no ordering."""
    pid = tl.program_id(0)
    row = pid // SPLIT
    chunk = pid % SPLIT
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    thr = tl.load(thr_ptr + row)
    start = s + chunk * CHUNK
    end = tl.minimum(start + CHUNK, e)
    lane = tl.arange(0, BLOCK)[:, None] * VEC + tl.arange(0, VEC)[None, :]
    a_ptrs = cnt_a_ptr + row + tl.zeros([BLOCK, VEC], tl.int32)
    b_ptrs = cnt_b_ptr + row + tl.zeros([BLOCK, VEC], tl.int32)
    ones = tl.full([BLOCK, VEC], 1, tl.int32)
    for t in tl.range(0, tl.cdiv(CHUNK, BLOCK * VEC)):
        i = start + t * BLOCK * VEC + lane
        m = i < end
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        k = _key11(x)
        better = m & (k < thr)
        pos = tl.atomic_add(a_ptrs, ones, mask=better, sem="relaxed", scope="gpu")
        keep = better & (pos >= 0) & (pos < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos, (i - s).to(tl.int32), mask=keep)
        tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)
        eq = m & (k == thr)
        pos2 = tl.atomic_add(b_ptrs, ones, mask=eq, sem="relaxed", scope="gpu")
        keep2 = eq & (pos2 >= 0) & (pos2 < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos2, (i - s).to(tl.int32), mask=keep2)
        tl.store(cand_val_ptr + row * CAP + pos2, x, mask=keep2)


def median_us(fn, calls=30, warmup=10):
    """Per-call CUDA events, median -- what do_bench reports."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(calls)]
    for a, b in evs:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) * 1000 for a, b in evs)
    return ts[len(ts) // 2]


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    dev = "cuda"
    rounds = 5
    print(
        "per-call CUDA events, median, shipped and stage-split alternating "
        f"in one process, {rounds} rounds\n"
    )
    for rows, vocab, top_k, stride0, split, block, warps in CASES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)
        want = torch.topk(logits, top_k, dim=1).values.sort(dim=1).values
        cap = triton.next_power_of_2(top_k * 8)
        chunk = triton.cdiv(vocab, split)
        hist = torch.zeros((rows, NB), dtype=torch.int32, device=dev)
        thr = torch.empty((rows,), dtype=torch.int32, device=dev)
        cnt_a = torch.empty((rows,), dtype=torch.int32, device=dev)
        cnt_b = torch.empty((rows,), dtype=torch.int32, device=dev)
        cand_idx = torch.empty((rows, cap), dtype=torch.int32, device=dev)
        cand_val = torch.empty((rows, cap), dtype=torch.float32, device=dev)
        counts = torch.empty((rows, RADIX), dtype=torch.int32, device=dev)
        slot = torch.empty((rows,), dtype=torch.int32, device=dev)

        lh = ov._Launch(
            _hist_split,
            (rows * split,),
            {"CHUNK": chunk, "SPLIT": split, "NB": NB, "BLOCK": block, "VEC": VEC},
            warps,
        )
        lt = ov._Launch(_thresh, (rows,), {"TOPK": top_k, "NB": NB}, 8)
        ls = ov._Launch(
            _select_split,
            (rows * split,),
            {
                "CHUNK": chunk,
                "SPLIT": split,
                "CAP": cap,
                "BLOCK": block,
                "VEC": VEC,
            },
            warps,
        )
        lx = ov._Launch(
            ov._tail,
            (rows,),
            {"TOPK": top_k, "NB": NB, "CAP": cap, "RADIX": RADIX, "BLOCK": 512},
            8,
        )

        def shipped():
            flaggems_vllm.top_k_per_row_prefill(
                logits, starts, ends, idx, rows, stride0, 1, top_k
            )

        def staged():
            lh(logits, starts, ends, hist, stride0)
            lt(hist, thr, cnt_a, cnt_b)
            ls(
                logits,
                starts,
                ends,
                thr,
                cnt_a,
                cnt_b,
                cand_idx,
                cand_val,
                stride0,
            )
            lx(
                logits,
                ends,
                hist,
                cnt_b,
                cand_idx,
                cand_val,
                idx,
                counts,
                slot,
                stride0,
            )

        idx.fill_(-9)
        staged()
        torch.cuda.synchronize()
        got = logits.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
        ok = torch.allclose(got, want) and bool((idx >= 0).all())

        pairs = []
        for _ in range(rounds):
            a = median_us(shipped)
            b = median_us(staged)
            pairs.append((a, b))
        ratios = sorted(a / b for a, b in pairs)
        print(
            f"  {rows} x {vocab}, top_k {top_k}, split {split} "
            f"({rows * split} workgroups, {block}x{warps}): "
            f"{'OK' if ok else 'WRONG'}"
        )
        for a, b in pairs:
            print(f"      shipped {a:>8.1f}   staged {b:>8.1f}   {a / b:>6.3f}x")
        print(
            f"      median ratio {ratios[len(ratios) // 2]:.3f}, "
            f"spread {ratios[0]:.3f} to {ratios[-1]:.3f}\n"
        )


if __name__ == "__main__":
    sys.exit(main())
