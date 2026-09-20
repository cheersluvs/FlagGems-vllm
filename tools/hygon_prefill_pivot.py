"""BW1000 exact-pivot/partition cost gate; production operator is untouched.

The pivot is a heuristic from 256 row samples, not a correctness assumption.
Only rows with >=k values strictly greater than the pivot and <=CAP such
values are eligible for a second exact selection; every other row must use a
fallback in a future implementation. This probe deliberately does not time or
claim a complete top-k implementation.
"""

import argparse
import math
import statistics
import subprocess
import sys
from importlib import import_module

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

from hygon_prefill_audit import Plan, device_time, emit, inputs, oracle, validate


SHAPES = (
    (64, 129280, 1024, 129280),
    (16383, 4095, 512, 4352),
    (4, 16385, 512, 16648),
    (4100, 1025, 512, 1288),
)
SEEDS = (42, 43)
CAP_FACTOR = 4


def z_for_shape(vocab, k):
    # Aim for ~1.25*k survivors, leaving some room for sampling error.
    tail = min(0.99, 1.25 * k / vocab)
    return statistics.NormalDist().inv_cdf(1.0 - tail)


@triton.jit
def estimate_pivot(logits, starts, ends, pivots, STRIDE0: tl.constexpr,
                   Z: tl.constexpr, SAMPLE: tl.constexpr):
    row = tl.program_id(0)
    lo = tl.load(starts + row)
    hi = tl.load(ends + row)
    n = hi - lo
    lane = tl.arange(0, SAMPLE)
    off = tl.minimum((lane * n) // SAMPLE, tl.maximum(n - 1, 0))
    x = tl.load(logits + row * STRIDE0 + lo + off, mask=n > 0, other=0.0)
    mean = tl.sum(x, 0) / SAMPLE
    d = x - mean
    sigma = tl.sqrt(tl.maximum(tl.sum(d * d, 0) / SAMPLE, 0.0))
    pivot = mean + Z * sigma
    # Non-finite samples make the heuristic unusable; the exact count/fallback
    # still decides safety. Zero is only a diagnostic placeholder here.
    pivot = tl.where(tl.abs(pivot) < float("inf"), pivot, 0.0)
    tl.store(pivots + row, pivot)


@triton.jit
def partition_gt(logits, starts, ends, pivots, counts_gt, counts_eq,
                 out_values, out_indices, STRIDE0: tl.constexpr,
                 CAP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lo = tl.load(starts + row)
    hi = tl.load(ends + row)
    n = hi - lo
    pivot = tl.load(pivots + row)
    lane = tl.arange(0, BLOCK)
    total_gt = 0
    total_eq = 0
    for tile in tl.range(0, tl.cdiv(n, BLOCK)):
        off = tile * BLOCK + lane
        valid = off < n
        x = tl.load(logits + row * STRIDE0 + lo + off,
                    mask=valid, other=float("-inf"))
        take = valid & (x > pivot)
        equal = valid & (x == pivot)
        flag = take.to(tl.int32)
        rank = tl.cumsum(flag, 0) - 1
        pos = total_gt + rank
        store = take & (pos < CAP)
        tl.store(out_values + row * CAP + pos, x, mask=store)
        tl.store(out_indices + row * CAP + pos, off, mask=store)
        total_gt += tl.sum(flag, 0)
        total_eq += tl.sum(equal.to(tl.int32), 0)
    tl.store(counts_gt + row, total_gt)
    tl.store(counts_eq + row, total_eq)


class PivotPlan:
    def __init__(self, tensors, k):
        x, starts, ends = tensors
        rows, vocab = x.shape
        self.rows = rows
        self.k = k
        self.cap = triton.next_power_of_2(CAP_FACTOR * k)
        self.pivots = torch.empty((rows,), dtype=torch.float32, device=x.device)
        self.counts_gt = torch.empty((rows,), dtype=torch.int32, device=x.device)
        self.counts_eq = torch.empty_like(self.counts_gt)
        self.values = torch.empty((rows, self.cap), dtype=torch.float32, device=x.device)
        self.indices = torch.empty((rows, self.cap), dtype=torch.int32, device=x.device)
        self.sample = lambda: estimate_pivot[(rows,)](
            x, starts, ends, self.pivots, x.stride(0), z_for_shape(vocab, k),
            256, num_warps=4,
        )
        self.partition = lambda: partition_gt[(rows,)](
            x, starts, ends, self.pivots, self.counts_gt, self.counts_eq,
            self.values, self.indices, x.stride(0), self.cap, 256,
            num_warps=4,
        )

    def __call__(self):
        self.sample()
        self.partition()


def measure_pair(plan, iters):
    for _ in range(3):
        plan()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            plan()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if getattr(e.device_type, "name", "") == "CUDA"]
    stages = {}
    for tag in ("estimate_pivot", "partition_gt"):
        times = [e.time_range.elapsed_us() for e in events if tag in e.name]
        if len(times) != iters or not all(math.isfinite(t) and t > 0 for t in times):
            raise RuntimeError(f"{tag}: expected {iters} kernel events, got {len(times)}")
        stages[tag] = statistics.median(times)
    return {"us": sum(stages.values()), "stages_us": stages,
            "kernel_events": len(events)}


def validate_partition(plan, tensors, want):
    x, starts, ends = tensors
    rows, vocab = x.shape
    plan()
    torch.cuda.synchronize()
    col = torch.arange(vocab, device=x.device)[None, :]
    valid = (col >= starts[:, None]) & (col < ends[:, None])
    gt = (valid & (x > plan.pivots[:, None])).sum(1)
    eq = (valid & (x == plan.pivots[:, None])).sum(1)
    if not torch.equal(gt.to(torch.int32), plan.counts_gt):
        raise AssertionError("Partition > counts differ from exact torch counts")
    if not torch.equal(eq.to(torch.int32), plan.counts_eq):
        raise AssertionError("Partition == counts differ from exact torch counts")
    eligible = (gt >= plan.k) & (gt <= plan.cap)
    safe = torch.minimum(gt, torch.tensor(plan.cap, device=x.device))
    pos = torch.arange(plan.cap, device=x.device)[None, :]
    used = pos < safe[:, None]
    # All stores are bounds-checked, including rows that overflow CAP. Only
    # eligible rows are required to hold the complete >pivot candidate set.
    cand = plan.values.masked_fill(~used, float("-inf"))
    selected = cand.topk(plan.k, dim=1).values.sort(dim=1).values
    if bool((selected[eligible] != want[eligible]).any()):
        raise AssertionError("Eligible compacted candidates lost exact top-k values")
    inds = plan.indices.masked_fill(~used, -1)
    bounds = (inds >= 0) & (inds < (ends - starts)[:, None])
    if bool((~bounds[used]).any()):
        raise AssertionError("Compacted candidate index outside row bounds")
    return {
        "eligible_rows": int(eligible.sum()), "fallback_rows": int((~eligible).sum()),
        "overflow_rows": int((gt > plan.cap).sum()),
        "below_k_rows": int((gt < plan.k).sum()),
        "median_gt": int(gt.median()), "max_gt": int(gt.max()),
        "median_eq": int(eq.median()), "max_eq": int(eq.max()),
        "median_shrink_fraction": round(float((gt.float() / (ends-starts).clamp_min(1)).median()), 5),
    }


def worker(shape_id, rounds, iters):
    shape = SHAPES[shape_id]
    rows, vocab, k, stride0 = shape
    ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
    dense = vocab <= ov.DENSE_VOCAB_PER_TOPK * k
    mod = ov._dense_carry if dense else ov._sparse
    if mod is None or mod.HAS_TLE:
        raise RuntimeError("Expected production non-TLE Hygon path")
    block, warps = ov._geometry(rows, vocab) or (512, 8)
    emit("pivot_config", shape=shape, geometry=[block, warps], cap=triton.next_power_of_2(CAP_FACTOR*k),
         z=z_for_shape(vocab, k), note="two-kernel lower bound; final exact select and fallback not timed")
    for case in ("tied", "constant", "partial", "short", "special"):
        n = min(rows, 8)
        probe_shape = inputs(n, vocab, max(stride0, vocab + 8), k, 123, case)
        probe_want = oracle(probe_shape, k)
        case_stats = validate_partition(PivotPlan(probe_shape, k), probe_shape, probe_want)
        emit("pivot_adversarial", shape_id=shape_id, case=case, **case_stats)
    for seed in SEEDS:
        tensors = inputs(rows, vocab, stride0, k, seed)
        want = oracle(tensors, k)
        baseline = Plan(mod, tensors, k, block, warps)
        validate(baseline, tensors, k, want)
        pivot = PivotPlan(tensors, k)
        stats = validate_partition(pivot, tensors, want)
        emit("pivot_validation", seed=seed, shape_id=shape_id, **stats)
        ratios = []
        for round_id in range(rounds):
            order = ("baseline", "pivot", "pivot", "baseline") if round_id % 2 == 0 else ("pivot", "baseline", "baseline", "pivot")
            records = []
            for arm in order:
                result = device_time(baseline, iters) if arm == "baseline" else measure_pair(pivot, iters)
                records.append((arm, result))
            a, b = records[0][1], records[1][1]
            c, d = records[2][1], records[3][1]
            pair1 = a["us"] / b["us"] if order[0] == "baseline" else b["us"] / a["us"]
            pair2 = d["us"] / c["us"] if order[3] == "baseline" else c["us"] / d["us"]
            ratios += [pair1, pair2]
            emit("pivot_round", seed=seed, round=round_id, order=order,
                 readings=records, baseline_over_two_stage=[pair1, pair2])
        emit("pivot_seed_summary", seed=seed, shape_id=shape_id,
             median_ratio=statistics.median(ratios), min_ratio=min(ratios), max_ratio=max(ratios))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args()
    if args.worker is not None:
        worker(args.worker, args.rounds, args.iters)
        return
    failures = []
    for shape_id in range(len(SHAPES)):
        emit("pivot_worker_start", shape_id=shape_id, shape=SHAPES[shape_id])
        try:
            result = subprocess.run([sys.executable, "-u", __file__, "--worker", str(shape_id),
                                     "--rounds", str(args.rounds), "--iters", str(args.iters)],
                                    timeout=1800, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
        emit("pivot_worker_exit", shape_id=shape_id, code=code)
        if code:
            failures.append(shape_id)
    emit("pivot_suite_summary", failures=failures)
    if failures:
        raise RuntimeError(f"Pivot workers failed: {failures}")


if __name__ == "__main__":
    main()
