"""Probe a vocab/8 sampled threshold for sparse, long Hygon prefill rows.

This is a probe only; it does not change the Hygon production override.

The old sampled-prefill experiment sampled a fixed number of tiles.  This
probe samples every eighth valid element instead, and only runs the sparse /
long-row shapes where that trade-off is plausible:

    (64, 129280, 1024), (4, 8193, 512), (4, 16385, 512)

The candidate pipeline is:

    sampled STEP-0 histogram -> candidate collect -> exact candidate merge

If the sampled candidate count is outside [top_k, CAP], the fixup uses four
8-bit rounds over the full 32-bit ordered float key.  Retrying the coarse
STEP-0 key is deliberately not used: a narrow value band can collapse into a
single STEP-0 bin and make a truncated candidate buffer incorrect.

Timing reports both the preallocated steady-state device time and a simple
allocation-inclusive synchronized wall time.  The latter is intentionally
conservative: the candidate scratch is allocated on every call, while the
production Hygon path may reuse scratch.

Run on the Hygon host with:

    tools/vendor_probe.sh tools/hygon_prefill_sparse_sample8.py \\
        hygon_prefill_sparse_sample8
"""

from importlib import import_module
import statistics
import sys
import time

import torch
import triton
import triton.language as tl

import flaggems_vllm  # noqa: F401

# Reuse the previously written probe's sampled collect and remap kernels.
# Importing it is side-effect free because its main is guarded.
from hygon_prefill_sampled import (  # noqa: E402
    BLOCK,
    CAP_FACTOR,
    NB,
    SAFETY,
    WARPS,
    _remap,
    _select,
    device_us,
    split_factor,
)

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
_full_key = _generic._convert_to_uint32
_sample_key = _generic._convert_to_trt_uint16_hi11

# Sparse / long rows only.  n/k is 126, 16 and 32 respectively.
SHAPES = (
    (64, 129280, 1024, 129280),
    (4, 8193, 512, 8456),
    (4, 16385, 512, 16648),
)

FULL_KEY_BINS = 256


@triton.jit
def _scan_threshold_v8(base, target, NBINS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    lane = tl.arange(0, BLOCK_SIZE)
    carry = tl.zeros([], tl.int32)
    threshold = tl.full([], NBINS - 1, tl.int32)
    found = tl.full([], False, tl.int1)
    for tile in tl.static_range(0, NBINS, BLOCK_SIZE):
        bins = tile + lane
        counts = tl.load(base + bins)
        prefix = carry + tl.cumsum(counts, axis=0)
        hit = (prefix >= target) & (not found)
        candidate = tl.min(tl.where(hit, bins, NBINS - 1), axis=0)
        if (not found) & (tl.max(hit.to(tl.int32), axis=0) > 0):
            threshold = candidate
            found = tl.full([], True, tl.int1)
        carry += tl.sum(counts, axis=0)
    return threshold


@triton.jit
def _prepare_v8(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    thr_ptr,
    cnt_ptr,
    stride0,
    TOPK: tl.constexpr,
    SAFETY: tl.constexpr,
    NBINS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sample valid positions s, s+8, s+16, ... and estimate the threshold."""

    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)
    base = hist_ptr + row * NBINS
    for tile in tl.static_range(0, NBINS, BLOCK_SIZE):
        tl.store(base + tile + lane, tl.zeros([BLOCK_SIZE], tl.int32))
    tl.store(cnt_ptr + row, 0)
    tl.debug_barrier()

    start = tl.load(starts_ptr + row)
    end = tl.load(ends_ptr + row)
    sample_count = tl.cdiv(end - start, 8)
    for tile in tl.range(0, tl.cdiv(sample_count, BLOCK_SIZE)):
        sample_pos = tile * BLOCK_SIZE + lane
        pos = start + sample_pos * 8
        mask = pos < end
        x = tl.load(logits_ptr + row * stride0 + pos, mask=mask, other=0.0)
        bin_idx = _sample_key(x)
        tl.atomic_add(
            base + bin_idx,
            tl.full([BLOCK_SIZE], 1, tl.int32),
            mask=mask,
            sem="relaxed",
            scope="cta",
        )
    tl.debug_barrier()

    total = tl.zeros([], tl.int32)
    for tile in tl.static_range(0, NBINS, BLOCK_SIZE):
        total += tl.sum(tl.load(base + tile + lane), axis=0)
    row_len = tl.maximum(end - start, 1)
    target = tl.maximum(tl.cdiv(TOPK * total, row_len), 1) * SAFETY
    tl.store(thr_ptr + row, _scan_threshold_v8(base, target, NBINS, BLOCK_SIZE))


@triton.jit
def _fixup_full32(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    cnt_ptr,
    merge_lens_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    stride0,
    TOPK: tl.constexpr,
    NB: tl.constexpr,
    CAP: tl.constexpr,
    RADIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Keep the cheap sampled result, or redo selection with the full key."""

    row = tl.program_id(0)
    c = tl.load(cnt_ptr + row)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    need = (c < tl.minimum(TOPK, e - s)) | (c > CAP)
    if not need:
        tl.store(merge_lens_ptr + row, c)
        return

    lane = tl.arange(0, BLOCK)
    bins = tl.arange(0, RADIX)
    base = hist_ptr + row * NB
    n_tiles = tl.cdiv(e - s, BLOCK)
    desired = tl.zeros((), dtype=tl.uint32)
    digit_mask = tl.zeros((), dtype=tl.uint32)
    rank_left = TOPK + 1

    # The ordered uint32 key sorts largest float values first.  Locate the
    # (top_k + 1)-th key from the high byte to the low byte.
    for digit_pos in tl.static_range(24, -1, -8):
        if rank_left > 1:
            tl.store(base + bins, tl.zeros([RADIX], tl.int32))
            tl.debug_barrier()
            for tile in tl.range(0, n_tiles):
                i = s + tile * BLOCK + lane
                mask = i < e
                x = tl.load(logits_ptr + row * stride0 + i, mask=mask, other=0.0)
                key = _full_key(x)
                digit = ((key >> digit_pos) & (RADIX - 1)).to(tl.int32)
                tl.atomic_add(
                    base + digit,
                    tl.full([BLOCK], 1, tl.int32),
                    mask=mask & ((key & digit_mask) == desired),
                    sem="relaxed",
                    scope="cta",
                )
            tl.debug_barrier()
            counts = tl.load(base + bins)
            prefix = tl.cumsum(counts, axis=0) - counts
            hit = (prefix < rank_left) & (prefix + counts >= rank_left)
            digit_value = tl.min(
                tl.where(hit, bins, RADIX), axis=0
            ).to(tl.int32)
            digit_value = tl.where(
                digit_value == RADIX, RADIX - 1, digit_value
            )
            below = tl.max(tl.where(bins == digit_value, prefix, 0), axis=0)
            desired = desired | (digit_value.to(tl.uint32) << digit_pos)
            digit_mask = digit_mask | (
                tl.full((), RADIX - 1, tl.uint32) << digit_pos
            )
            rank_left = rank_left - below

    # Rebuild the candidate buffer from the exact key.  Strictly-better values
    # are written first and exact ties second, matching the generic contract.
    tl.store(cnt_ptr + row, 0)
    tl.debug_barrier()
    cnt_ptrs = cnt_ptr + row + tl.zeros([BLOCK], tl.int32)
    for equal_pass in tl.static_range(2):
        for tile in tl.range(0, n_tiles):
            i = s + tile * BLOCK + lane
            mask = i < e
            x = tl.load(logits_ptr + row * stride0 + i, mask=mask, other=0.0)
            key = _full_key(x)
            if equal_pass == 0:
                take = mask & (key < desired)
            else:
                take = mask & (key == desired)
            pos = tl.atomic_add(
                cnt_ptrs,
                tl.full([BLOCK], 1, tl.int32),
                mask=take,
                sem="relaxed",
                scope="cta",
            )
            keep = take & (pos < CAP)
            tl.store(
                cand_idx_ptr + row * CAP + pos,
                (i - s).to(tl.int32),
                mask=keep,
            )
            tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)
        tl.debug_barrier()
    tl.store(merge_lens_ptr + row, tl.minimum(tl.load(cnt_ptr + row), CAP))


def _wall_us(fn, iters=8, warmup=3):
    """Synchronized host wall time, including Python-side allocations."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def _make_inputs(rows, vocab, top_k, stride0, case):
    torch.manual_seed(42 if case != "ties" else 43)
    storage = torch.randn(
        (rows - 1) * stride0 + vocab,
        dtype=torch.float32,
        device="cuda",
    )
    logits = torch.as_strided(storage, (rows, vocab), (stride0, 1))
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), vocab, dtype=torch.int32, device="cuda")
    if case == "band":
        logits.copy_(10.0 + 0.2 * torch.rand_like(logits))
    elif case == "ties":
        logits.copy_(torch.round(logits * 8.0) / 8.0)
    elif case == "partial":
        starts.fill_(17)
        ends.fill_(vocab - 23)
    return logits, starts, ends


def _oracle_values(logits, starts, ends, top_k):
    rows = logits.shape[0]
    live = torch.full_like(logits, float("-inf"))
    for row in range(rows):
        s = int(starts[row].item())
        e = int(ends[row].item())
        live[row, : e - s] = logits[row, s:e]
    return torch.topk(live, top_k, dim=1).values.sort(dim=1).values


def _check(logits, starts, ends, indices, want):
    rows, top_k = indices.shape
    if not bool((indices >= 0).all()):
        return False
    lengths = ends - starts
    if not bool((indices < lengths[:, None]).all()):
        return False
    absolute = indices.long() + starts.long()[:, None]
    got = logits.gather(1, absolute).sort(dim=1).values
    return bool(torch.allclose(got, want, rtol=1e-5, atol=1e-6))


def _build_pipeline(logits, starts, ends, top_k, stride0, stride1):
    rows, vocab = logits.shape
    decode = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    sms = decode._sm_count()
    launch = decode._Launch
    split = split_factor(rows, vocab, sms)
    chunk = triton.cdiv(vocab, split)
    cap = max(BLOCK, triton.next_power_of_2(top_k * CAP_FACTOR))

    hist = torch.empty((rows, NB), dtype=torch.int32, device=logits.device)
    thr = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    cnt = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    cand_idx = torch.empty((rows, cap), dtype=torch.int32, device=logits.device)
    cand_val = torch.empty((rows, cap), dtype=logits.dtype, device=logits.device)
    merged = torch.empty((rows, top_k), dtype=torch.int32, device=logits.device)
    merge_lens = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    zeros = torch.zeros((rows,), dtype=torch.int32, device=logits.device)
    scratch = (
        torch.empty((rows, _generic.NUM_BINS), dtype=torch.int32, device=logits.device),
        torch.empty(
            (rows, _generic.NUM_FILNAL_ITEMS),
            dtype=torch.float32,
            device=logits.device,
        ),
        torch.empty((rows,), dtype=torch.int32, device=logits.device),
        torch.empty((rows,), dtype=torch.int32, device=logits.device),
        torch.empty((rows,), dtype=torch.int32, device=logits.device),
        torch.empty((rows,), dtype=torch.int32, device=logits.device),
    )

    prepare = launch(
        _prepare_v8,
        (rows,),
        {
            "TOPK": top_k,
            "SAFETY": SAFETY,
            "NBINS": NB,
            "BLOCK_SIZE": BLOCK,
        },
        WARPS,
    )
    select = launch(
        _select,
        (rows * split,),
        {"CHUNK": chunk, "SPLIT": split, "CAP": cap, "BLOCK": BLOCK},
        WARPS,
    )
    fixup = launch(
        _fixup_full32,
        (rows,),
        {
            "TOPK": top_k,
            "NB": NB,
            "CAP": cap,
            "RADIX": FULL_KEY_BINS,
            "BLOCK": BLOCK,
        },
        WARPS,
    )
    merge = launch(
        _generic.non_tle_top_k_per_row_prefill,
        (rows,),
        {"TOPK": top_k, "BLOCK_SIZE": BLOCK, "ROW_OFFSET": 0},
        WARPS,
    )
    remap = launch(
        _remap,
        (rows,),
        {"CAP": cap, "TOPK": top_k, "BLOCK": triton.next_power_of_2(top_k)},
        4,
    )

    def run():
        prepare(logits, starts, ends, hist, thr, cnt, stride0)
        select(logits, starts, ends, thr, cnt, cand_idx, cand_val, stride0)
        fixup(
            logits,
            starts,
            ends,
            hist,
            cnt,
            merge_lens,
            cand_idx,
            cand_val,
            stride0,
        )
        merge(cand_val, merged, zeros, merge_lens, cap, 1, cap, *scratch)
        remap(cand_idx, merged, indices,)

    indices = torch.empty((rows, top_k), dtype=torch.int32, device=logits.device)

    def run_with_counts():
        prepare(logits, starts, ends, hist, thr, cnt, stride0)
        select(logits, starts, ends, thr, cnt, cand_idx, cand_val, stride0)
        sampled_counts = cnt.clone()
        fixup(
            logits,
            starts,
            ends,
            hist,
            cnt,
            merge_lens,
            cand_idx,
            cand_val,
            stride0,
        )
        merge(cand_val, merged, zeros, merge_lens, cap, 1, cap, *scratch)
        remap(cand_idx, merged, indices)
        return sampled_counts

    # The first closure is kept allocation-free for timing; `run_with_counts`
    # is used for validation and candidate-count reporting.
    return run, run_with_counts, indices, cap


def _allocation_inclusive_call(logits, starts, ends, top_k, stride0, stride1):
    """Construct and execute the candidate pipeline, including scratch allocs."""

    run, _, _, _ = _build_pipeline(logits, starts, ends, top_k, stride0, stride1)
    run()


def main():
    import vllm._custom_ops  # noqa: F401

    print(
        "sample policy: every 8th valid element (sample_count ~= row_len/8); "
        "full-32-bit fallback on candidate overflow\n"
    )
    print(
        "  rows vocab top_k n/k split sampled-cands fallback normal band partial "
        "vllm-dev ship-dev sample-dev vllm/ship vllm/sample sample-wall"
    )

    for rows, vocab, top_k, stride0 in SHAPES:
        logits, starts, ends = _make_inputs(
            rows, vocab, top_k, stride0, "normal"
        )
        stride1 = 1
        run, run_with_counts, indices, cap = _build_pipeline(
            logits, starts, ends, top_k, stride0, stride1
        )
        want = _oracle_values(logits, starts, ends, top_k)
        counts = run_with_counts()
        torch.cuda.synchronize()
        sampled_counts = counts.cpu()
        fallback_rows = int(
            ((sampled_counts < top_k) | (sampled_counts > cap)).sum().item()
        )
        normal_ok = _check(logits, starts, ends, indices, want)

        band_logits, band_starts, band_ends = _make_inputs(
            rows, vocab, top_k, stride0, "band"
        )
        _, band_with_counts, band_indices, _ = _build_pipeline(
            band_logits, band_starts, band_ends, top_k, stride0, stride1
        )
        band_want = _oracle_values(band_logits, band_starts, band_ends, top_k)
        band_with_counts()
        torch.cuda.synchronize()
        band_ok = _check(band_logits, band_starts, band_ends, band_indices, band_want)

        partial_logits, partial_starts, partial_ends = _make_inputs(
            rows, vocab, top_k, stride0, "partial"
        )
        _, partial_with_counts, partial_indices, _ = _build_pipeline(
            partial_logits, partial_starts, partial_ends, top_k, stride0, stride1
        )
        partial_want = _oracle_values(
            partial_logits, partial_starts, partial_ends, top_k
        )
        partial_with_counts()
        torch.cuda.synchronize()
        partial_ok = _check(
            partial_logits,
            partial_starts,
            partial_ends,
            partial_indices,
            partial_want,
        )

        def vllm():
            torch.ops._C.top_k_per_row_prefill(
                logits,
                starts,
                ends,
                indices,
                rows,
                stride0,
                stride1,
                top_k,
            )

        def shipped():
            flaggems_vllm.top_k_per_row_prefill(
                logits,
                starts,
                ends,
                indices,
                rows,
                stride0,
                stride1,
                top_k,
            )

        vllm_dev = device_us(vllm)
        ship_dev = device_us(shipped)
        sample_dev = device_us(run)
        sample_wall = _wall_us(run)
        alloc_wall = _wall_us(
            lambda: _allocation_inclusive_call(
                logits, starts, ends, top_k, stride0, stride1
            ),
            iters=3,
            warmup=1,
        )
        split = split_factor(rows, vocab, 80)
        print(
            f"  {rows:>4} {vocab:>6} {top_k:>5} {vocab // top_k:>3} {split:>5} "
            f"{int(sampled_counts.max()):>13} {fallback_rows:>8} "
            f"{str(normal_ok):>6} {str(band_ok):>4} {str(partial_ok):>7} "
            f"{vllm_dev:>8.1f} {ship_dev:>8.1f} {sample_dev:>10.1f} "
            f"{vllm_dev / ship_dev:>9.3f} {vllm_dev / sample_dev:>10.3f} "
            f"{sample_wall:>10.1f} ({alloc_wall:>7.1f})"
        )

    print(
        "\n`sample-dev` is steady-state device time with candidate scratch "
        "preallocated; `sample-wall` includes synchronized host dispatch; "
        "the parenthesized value allocates the candidate scratch per call."
    )


if __name__ == "__main__":
    sys.exit(main())
