"""Feasibility of a many-CTA, block-private histogram on 64x129280.

This times ONLY the first stage, not a complete top-k operator. It is a
necessary-cost gate for a future exact coarse-histogram + refine algorithm.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import traceback
from pathlib import Path

from hygon_prefill_audit import SHAPES, emit, inputs, occupancy
from hygon_prefill_audit_source import GENERIC, OVERRIDE

ROOT = Path(__file__).resolve().parents[1]
SHAPE = SHAPES[0]
CONFIGS = ((1024, 4), (2048, 4), (2048, 8), (4096, 8), (4096, 16))
BINS = 256


def device_time(fn, iters):
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    gpu = [e for e in prof.events() if getattr(e.device_type, "name", "") == "CUDA"]
    if len(gpu) != iters:
        raise RuntimeError(
            f"Expected {iters} single-kernel events, got {len(gpu)}: "
            f"{sorted(set(e.name for e in gpu))}"
        )
    samples = [e.time_range.elapsed_us() for e in gpu]
    if not all(x > 0 for x in samples):
        raise RuntimeError(f"Invalid kernel durations: {samples}")
    return dict(us=statistics.median(samples), samples_us=samples)


class HistPlan:
    def __init__(self, tensors, block, warps):
        from importlib import import_module

        import torch
        import triton
        from hygon_prefill_private_hist_kernel import private_hist256

        launcher = import_module(
            "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
        )._Launch
        x, starts, ends = tensors
        rows, vocab = x.shape
        self.chunks = triton.cdiv(vocab, block)
        self.scratch = torch.empty(
            (rows, self.chunks, BINS), device=x.device, dtype=torch.int32
        )
        self.launch = launcher(
            private_hist256,
            (rows * self.chunks,),
            dict(CHUNKS=self.chunks, BLOCK=block),
            warps,
        )
        self.args = (x, starts, ends, self.scratch, x.stride(0))

    def __call__(self):
        self.launch(*self.args)


def verify(plan, tensors, top_k, case):
    import torch

    x, starts, ends = tensors
    plan()
    torch.cuda.synchronize()
    counts = plan.scratch.sum(dim=(1, 2))
    lengths = ends - starts
    if not torch.equal(counts.to(torch.int32), lengths):
        raise AssertionError("Private histogram does not count each live element once")
    merged = plan.scratch.sum(dim=1)
    for row in range(min(x.shape[0], 4)):
        lo, hi = int(starts[row].item()), int(ends[row].item())
        values = x[row, lo:hi].to(torch.float16)
        bits = values.view(torch.int16).to(torch.int32) & 0xFFFF
        mapped = torch.where((bits & 0x8000) != 0, bits, (~bits) & 0x7FFF)
        key = (mapped >> 8).cpu().to(torch.int64)
        expected = torch.bincount(key, minlength=BINS)
        if not torch.equal(merged[row].cpu().to(torch.int64), expected):
            raise AssertionError(f"Private histogram bin mismatch on row {row}")
    if case == "full_normal":
        # A coarse threshold bin larger than 2048 requires refinement or
        # fallback in a complete operator; never silently truncate it.
        bin_counts = merged.cpu().tolist()
        threshold_sizes = []
        for row_counts in bin_counts:
            seen = 0
            for size in row_counts:
                seen += size
                if seen >= top_k:
                    threshold_sizes.append(size)
                    break
        if len(threshold_sizes) != x.shape[0]:
            raise AssertionError("Missing top-k threshold bin")
        emit(
            "threshold_bin",
            min=min(threshold_sizes),
            median=statistics.median(threshold_sizes),
            max=max(threshold_sizes),
            above_2048=sum(size > 2048 for size in threshold_sizes),
        )
    emit("validation", case=case, rows=x.shape[0], ok=True)


def paired(plan, baseline, rounds, iters, seed):
    ratios, stage_us, baseline_us = [], [], []
    for round_id in range(rounds):
        order = (
            ("vllm", "private_hist", "private_hist", "vllm")
            if round_id % 2 == 0
            else ("private_hist", "vllm", "vllm", "private_hist")
        )
        funcs = {"vllm": baseline, "private_hist": plan}
        readings = [(arm, device_time(funcs[arm], iters)) for arm in order]
        round_ratios = []
        for i, j in ((0, 1), (3, 2)):
            pair = dict(readings[p] for p in (i, j))
            round_ratios.append(pair["private_hist"]["us"] / pair["vllm"]["us"])
            stage_us.append(pair["private_hist"]["us"])
            baseline_us.append(pair["vllm"]["us"])
        ratios.extend(round_ratios)
        emit(
            "paired_round",
            seed=seed,
            round=round_id,
            order=order,
            readings=readings,
            stage_fraction=round_ratios,
        )
    return ratios, stage_us, baseline_us


def worker(args):
    import torch
    import triton
    import vllm._custom_ops  # noqa: F401 - registers torch.ops._C

    import flaggems_vllm  # noqa: F401 - activates Hygon backend

    if not hasattr(torch.ops._C, "top_k_per_row_prefill"):
        raise RuntimeError("vLLM prefill baseline is unavailable")
    rows, vocab, top_k, stride0 = SHAPE
    block, warps = CONFIGS[args.worker]
    emit(
        "device",
        shape=SHAPE,
        block=block,
        warps=warps,
        chunks=triton.cdiv(vocab, block),
        torch=torch.__version__,
        triton=triton.__version__,
        props=str(torch.cuda.get_device_properties(torch.cuda.current_device())),
    )
    for case in ("tied", "constant", "partial", "short", "special", "strided"):
        tensors = inputs(4, vocab, stride0, top_k, 123, case)
        plan = HistPlan(tensors, block, warps)
        verify(plan, tensors, top_k, case)
        del plan, tensors
    all_ratios, all_stage_us, all_baseline_us = [], [], []
    for seed in args.seeds:
        tensors = inputs(rows, vocab, stride0, top_k, seed)
        plan = HistPlan(tensors, block, warps)
        verify(plan, tensors, top_k, "full_normal")
        x, starts, ends = tensors
        out = torch.empty((rows, top_k), device=x.device, dtype=torch.int32)

        def baseline(x=x, starts=starts, ends=ends, out=out):
            torch.ops._C.top_k_per_row_prefill(
                x, starts, ends, out, rows, stride0, 1, top_k
            )

        ratios, stage_us, baseline_us = paired(
            plan, baseline, args.rounds, args.iters, seed
        )
        all_ratios.extend(ratios)
        all_stage_us.extend(stage_us)
        all_baseline_us.extend(baseline_us)
        del plan, tensors, out
    emit(
        "worker_summary",
        config=(block, warps),
        stage_us=statistics.median(all_stage_us),
        vllm_us=statistics.median(all_baseline_us),
        stage_fraction=statistics.median(all_ratios),
        seed_fractions=[
            statistics.median(
                all_ratios[i * args.rounds * 2 : (i + 1) * args.rounds * 2]
            )
            for i in range(len(args.seeds))
        ],
        fractions=all_ratios,
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, choices=range(len(CONFIGS)))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43))
    args = parser.parse_args()
    if args.rounds < 2 or args.iters < 2 or not args.seeds:
        parser.error("At least two rounds/iterations and one seed are required")
    if args.worker is not None:
        return worker(args)
    emit(
        "probe",
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        host=platform.node(),
        shape=SHAPE,
        configs=CONFIGS,
        env={
            key: os.environ.get(key)
            for key in (
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "FLAGGEMS_FORCE_TLE",
            )
        },
        source_sha256={
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (GENERIC, OVERRIDE)
        },
    )
    occupancy("before")
    failed, summaries = [], []
    for target_id, config in enumerate(CONFIGS):
        emit("worker_start", target_id=target_id, config=config)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(target_id),
            "--rounds",
            str(args.rounds),
            "--iters",
            str(args.iters),
            "--seeds",
            *map(str, args.seeds),
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=ROOT, capture_output=True, text=True, timeout=1200
            )
            print(proc.stdout, end="", flush=True)
            print(proc.stderr, end="", flush=True)
            if proc.returncode:
                failed.append(target_id)
            for line in proc.stdout.splitlines():
                if line.startswith('{"'):
                    record = json.loads(line)
                    if record.get("kind") == "worker_summary":
                        summaries.append(record)
            emit("worker_exit", target_id=target_id, code=proc.returncode)
        except subprocess.TimeoutExpired as exc:
            failed.append(target_id)
            print((exc.stdout or b"").decode(errors="replace"), flush=True)
            emit("worker_timeout", target_id=target_id, seconds=1200)
    occupancy("after")
    emit("probe_complete", ok=not failed, failed=failed, summaries=summaries)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
