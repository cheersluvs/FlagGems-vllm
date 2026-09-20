"""Joint Hygon probes for long-row splitting and cross-step candidate reuse.

This is an analysis probe only.  It does not change the shipped operator.

Part 1 re-benchmarks the exact chunk-split pipeline on the current checkout:
each chunk runs an exact local top-k, then a gather and an exact final merge.
The split is only interesting for the grid-starved long rows, so the sweep is
kept to those rows plus one many-row control.

Part 2 mirrors the four radix decisions with device tensors.  It reports:
  * the boundary-bin size and the number of rows reaching each STEP;
  * whether the STEP-1 boundary bucket is contained in STEP-0's fp16 bucket;
  * how much of that bucket is covered by a small adjacent-bin guard; and
  * the scan elements that could be removed if STEP-2/3 consumed a materialized
    candidate workset instead of rereading the whole row.

The containment check matters: STEP 0 uses rounded fp16 keys while STEP 1
uses fp32 high bits.  A naive STEP-0 -> STEP-1 workset can therefore be
incorrect at a rounding boundary.  The probe measures that rather than
assuming the proposed data flow is valid.

Run on BW1000, for example:

    tools/vendor_probe.sh tools/hygon_prefill_split_workset.py \
        hygon_prefill_split_workset_v1
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import torch
import triton


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import hygon_prefill_split as split_probe  # noqa: E402


GENERIC = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
SHAPES = (
    # Grid-starved targets.
    (64, 129280, 1024, 129280),
    (4, 8193, 512, 8456),
    (4, 16385, 512, 16648),
    # Control: enough rows that splitting should not help.
    (16383, 4095, 512, 4352),
)
SPLITS = (1, 2, 4, 8, 16)
MIN_CHUNK = 1024
NB = 2048


def event_us(fn, iters=20, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iters


def make_inputs(rows, vocab, stride0, seed):
    torch.manual_seed(seed)
    buf = torch.randn(
        (rows - 1) * stride0 + vocab,
        device="cuda",
        dtype=torch.float32,
    )
    logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), vocab, dtype=torch.int32, device="cuda")
    return buf, logits, starts, ends


def run_chunk_split(logits, starts, ends, rows, vocab, top_k, stride0, split):
    """Build one exact split pipeline, matching the old direct probe."""
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    chunk = triton.cdiv(vocab, split)
    nv = rows * split
    ncand = split * top_k
    cap = triton.next_power_of_2(ncand)
    block = GENERIC.NUM_THREADS_PER_BLOCK

    cstart = torch.empty((nv,), dtype=torch.int32, device="cuda")
    cend = torch.empty((nv,), dtype=torch.int32, device="cuda")
    cand = torch.empty((nv, top_k), dtype=torch.int32, device="cuda")
    cand_val = torch.empty((rows, cap), dtype=torch.float32, device="cuda")
    cand_idx = torch.empty((rows, cap), dtype=torch.int32, device="cuda")
    cnt = torch.full((rows,), ncand, dtype=torch.int32, device="cuda")
    hist = torch.empty((rows, NB), dtype=torch.int32, device="cuda")
    counts = torch.empty((rows, split_probe.RADIX), dtype=torch.int32, device="cuda")
    slot = torch.empty((rows,), dtype=torch.int32, device="cuda")
    scratch = (
        torch.empty((nv, GENERIC.NUM_BINS), dtype=torch.int32, device="cuda"),
        torch.empty(
            (nv, GENERIC.NUM_FILNAL_ITEMS), dtype=torch.float32, device="cuda"
        ),
        torch.empty((nv,), dtype=torch.int32, device="cuda"),
        torch.empty((nv,), dtype=torch.int32, device="cuda"),
        torch.empty((nv,), dtype=torch.int32, device="cuda"),
        torch.empty((nv,), dtype=torch.int32, device="cuda"),
    )
    out = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    floor = torch.finfo(torch.float32).min
    gblock = min(1024, triton.next_power_of_2(ncand))

    lb = ov._Launch(
        split_probe._bounds,
        (rows,),
        {"SPLIT": split, "CHUNK": chunk, "BLOCK": triton.next_power_of_2(split)},
        1,
    )
    l1 = ov._Launch(
        GENERIC.non_tle_top_k_per_row_prefill,
        (nv,),
        {"TOPK": top_k, "BLOCK_SIZE": block, "ROW_OFFSET": 0},
        GENERIC._num_warps(block),
    )
    lg = ov._Launch(
        split_probe._gather,
        (rows, triton.cdiv(ncand, gblock)),
        {
            "SPLIT": split,
            "TOPK": top_k,
            "CHUNK": chunk,
            "NCAND": ncand,
            "BLOCK": gblock,
        },
        4,
    )
    lt = ov._Launch(
        ov._tail,
        (rows,),
        {
            "TOPK": top_k,
            "NB": NB,
            "CAP": cap,
            "RADIX": split_probe.RADIX,
            "BLOCK": 512,
        },
        8,
    )

    def run():
        lb(starts, ends, cstart, cend, stride0)
        l1(logits, cand, cstart, cend, 0, 1, chunk, *scratch)
        lg(logits, starts, cand, cand_val, cand_idx, stride0, floor)
        lt(
            logits,
            ends,
            hist,
            cnt,
            cand_idx,
            cand_val,
            out,
            counts,
            slot,
            stride0,
        )

    return run, out


def check_split(out, logits, top_k, want):
    idx = out.long().clamp(0, logits.shape[1] - 1)
    got = logits.gather(1, idx).sort(dim=1).values
    return bool(torch.allclose(got, want) and bool((out >= 0).all()))


def benchmark_splits():
    print("\n## split probe: current checkout, exact chunk split + merge")
    for rows, vocab, top_k, stride0 in SHAPES:
        _, logits, starts, ends = make_inputs(rows, vocab, stride0, 42)
        want = torch.topk(logits, top_k, dim=1).values.sort(dim=1).values
        baseline_out = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")

        def baseline():
            import flaggems_vllm

            flaggems_vllm.top_k_per_row_prefill(
                logits, starts, ends, baseline_out, rows, stride0, 1, top_k
            )

        base_us = event_us(baseline)
        print(
            f"shape={rows}x{vocab} k={top_k} baseline_event_us={base_us:.3f}",
            flush=True,
        )
        for split in SPLITS:
            chunk = triton.cdiv(vocab, split)
            if split > 1 and chunk < MIN_CHUNK:
                continue
            run, out = run_chunk_split(
                logits, starts, ends, rows, vocab, top_k, stride0, split
            )
            run()
            torch.cuda.synchronize()
            ok = check_split(out, logits, top_k, want)
            if not ok:
                raise AssertionError(f"split={split} failed exact validation")
            us = event_us(run)
            print(
                f"  split={split:2d} chunk={chunk:6d} programs={rows * split:6d} "
                f"event_us={us:9.3f} ratio={base_us / us:6.3f} exact=OK",
                flush=True,
            )


def fp16_key(x):
    bits = x.to(torch.float16).contiguous().view(torch.int16).to(torch.int64)
    bits = bits & 0xFFFF
    sign = (bits & 0x8000) != 0
    inv = (~bits) & 0x7FFF
    mapped = torch.where(sign, bits, inv)
    return mapped >> 5


def fp32_key(x):
    bits = x.contiguous().view(torch.int32).to(torch.int64)
    bits = bits & 0xFFFFFFFF
    sign = (bits & 0x80000000) != 0
    inv = (~bits) & 0x7FFFFFFF
    return torch.where(sign, bits, inv)


def threshold(key, mask, target, bins):
    rows = key.shape[0]
    hist = torch.zeros((rows, bins), dtype=torch.int32, device=key.device)
    values = key.clamp(0, bins - 1).to(torch.int64)
    weights = mask.to(torch.int32)
    hist.scatter_add_(1, values, weights)
    prefix = hist.cumsum(dim=1)
    hit = prefix >= target[:, None]
    idx = hit.to(torch.int32).argmax(dim=1)
    at = hist.gather(1, idx[:, None]).squeeze(1)
    before = prefix.gather(1, idx[:, None]).squeeze(1) - at
    return idx, before, at


def qsummary(values):
    values = torch.cat(values).to(torch.float32).cpu()
    if values.numel() == 0:
        return "n=0"
    q = torch.quantile(values, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0]))
    return (
        f"n={values.numel()} min={q[0].item():.0f} med={q[1].item():.0f} "
        f"p90={q[2].item():.0f} p99={q[3].item():.0f} max={q[4].item():.0f}"
    )


def workset_case(rows, vocab, top_k, stride0, case, seed):
    _, logits, _, _ = make_inputs(rows, vocab, stride0, seed)
    if case == "tied":
        logits = (logits * 4).round() / 4
    elif case == "constant":
        logits = torch.zeros_like(logits)

    # Keep temporary key/hist tensors bounded on the dense 16K-row controls.
    row_chunk = min(rows, 256)
    stats = {name: [] for name in (
        "c0", "c1", "c2", "c3", "steps", "baseline", "workset", "saved"
    )}
    active1 = []
    active2 = []
    active3 = []
    leak01 = []
    guard_hits = {g: [] for g in (0, 1, 2, 4)}
    bytes_candidates = []

    for lo in range(0, rows, row_chunk):
        hi = min(rows, lo + row_chunk)
        x = logits[lo:hi, :vocab].contiguous()
        n = hi - lo
        key = fp32_key(x)
        key0 = fp16_key(x)
        key1 = key >> 21
        key2 = (key >> 10) & 0x7FF
        key3 = key & 0x3FF
        all_mask = torch.ones((n, vocab), dtype=torch.bool, device="cuda")

        thr0, before0, c0 = threshold(key0, all_mask, torch.full((n,), top_k, device="cuda", dtype=torch.int32), NB)
        mask0 = key0 == thr0[:, None]
        target1 = (top_k - before0).clamp_min(1)
        thr1, before1, c1 = threshold(key1, all_mask, target1, NB)
        mask1 = key1 == thr1[:, None]
        active_1 = c0 > NB

        # STEP 0 is fp16-rounded while STEP 1 is fp32-based.  Report the
        # exact leakage and the coverage of adjacent coarse-bin guards.
        leak = (mask1 & ~mask0).sum(dim=1)
        leak01.append(leak[active_1].cpu())
        for guard in guard_hits:
            lo_bin = (thr0 - guard).clamp_min(0)
            hi_bin = (thr0 + guard).clamp_max(NB - 1)
            guarded = (key0 >= lo_bin[:, None]) & (key0 <= hi_bin[:, None])
            covered = (mask1 & guarded).sum(dim=1)
            guard_hits[guard].append(
                (covered[active_1].to(torch.float32) / c1[active_1].clamp_min(1)).cpu()
            )

        target2 = (top_k - before1).clamp_min(1)
        thr2, before2, c2 = threshold(key2, mask1, target2, NB)
        mask2 = mask1 & (key2 == thr2[:, None])
        active_2 = active_1 & (c1 > NB)

        target3 = (top_k - before2).clamp_min(1)
        thr3, _, c3 = threshold(key3, mask2, target3, 1024)
        active_3 = active_2 & (c2 > NB)

        steps = torch.where(
            c0 <= NB,
            torch.ones_like(c0),
            torch.where(c1 <= NB, torch.full_like(c0, 2),
                        torch.where(c2 <= NB, torch.full_like(c0, 3), torch.full_like(c0, 4))),
        )
        # Existing implementation does two full-row passes per executed step.
        baseline = steps.to(torch.int64) * (2 * vocab)
        # Candidate reuse can only start after STEP 1 has established the
        # fp32 boundary bucket.  STEP 0 and STEP 1 remain full-row scans.
        workset = torch.where(
            steps <= 2,
            baseline,
            torch.where(steps == 3, 4 * vocab + 2 * c1,
                        4 * vocab + 2 * c1 + 2 * c2),
        )

        for name, value in (
            ("c0", c0), ("c1", c1), ("c2", c2), ("c3", c3),
            ("steps", steps), ("baseline", baseline),
            ("workset", workset), ("saved", baseline - workset),
        ):
            stats[name].append(value.cpu())
        active1.append(active_1.cpu())
        active2.append(active_2.cpu())
        active3.append(active_3.cpu())
        bytes_candidates.append((8 * (c1 + c2)).cpu())

    def cat(name):
        return torch.cat(stats[name])

    active1_t = torch.cat(active1)
    active2_t = torch.cat(active2)
    active3_t = torch.cat(active3)
    leak_t = torch.cat(leak01) if leak01 else torch.empty(0)
    print(
        f"  shape={rows}x{vocab} k={top_k} case={case} "
        f"step_counts="
        f"{[(s, int((cat('steps') == s).sum())) for s in (1, 2, 3, 4)]}",
        flush=True,
    )
    for name in ("c0", "c1", "c2", "c3", "saved"):
        print(f"    {name}: {qsummary(stats[name])}", flush=True)
    print(
        f"    reaches_step1={int(active1_t.sum())} "
        f"step2={int(active2_t.sum())} step3={int(active3_t.sum())} "
        f"/ {rows}",
        flush=True,
    )
    if leak_t.numel():
        print(
            f"    step0_to_step1 leakage among step1 rows: {qsummary([leak_t])} "
            f"nonzero={int((leak_t > 0).sum())}/{leak_t.numel()}",
            flush=True,
        )
        for guard, values in guard_hits.items():
            print(
                f"    adjacent_guard={guard} step1_bucket_coverage: "
                f"{qsummary(values)}",
                flush=True,
            )
    total_base = int(cat("baseline").sum())
    total_work = int(cat("workset").sum())
    total_bytes = int(torch.cat(bytes_candidates).sum())
    print(
        f"    scan_elements baseline={total_base} workset={total_work} "
        f"saved={total_base - total_work} "
        f"({(total_base - total_work) / max(total_base, 1):.3%}); "
        f"candidate_materialization_bytes={total_bytes}",
        flush=True,
    )


def candidate_workset_stats():
    print("\n## workset probe: exact key mirror and safe-carry analysis")
    for shape in SHAPES:
        for case in ("normal", "tied", "constant"):
            workset_case(*shape, case, 42)


def main():
    import vllm._custom_ops  # noqa: F401

    benchmark_splits()
    candidate_workset_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
