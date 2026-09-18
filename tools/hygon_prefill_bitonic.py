"""Compare on-chip bitonic top-k against the shipped Hygon dense carry path.

The parent is pure stdlib, each (shape, warps) candidate runs in its own
process, and no result is timed before full and adversarial correctness tests.
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

from hygon_prefill_audit import (
    SHAPES,
    Plan,
    check_output,
    emit,
    inputs,
    occupancy,
    oracle,
    validate,
)
from hygon_prefill_audit_source import GENERIC, OVERRIDE, build

ROOT = Path(__file__).resolve().parents[1]
# Start with dense shapes. The 129280-column row is too large for this
# single-CTA method; it needs a hierarchical algorithm, not a wider BLOCK.
TARGETS = ((2, 8), (2, 16), (4, 16), (5, 16), (6, 4), (6, 8))
EXPECTED_CARRY_SHA256 = (
    "869cf66cd5408b353333684c63a35a29f1f0d68f082eca50635beb3a7ccde777"
)


def device_time(fn, kernel_name, iters):
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
    kernels = [e for e in gpu if kernel_name in e.name]
    if len(kernels) != iters or len(gpu) != iters:
        raise RuntimeError(
            f"Expected {iters} single-kernel events for {kernel_name}; "
            f"matched={len(kernels)}, total={len(gpu)}, "
            f"names={sorted(set(e.name for e in gpu))}"
        )
    samples = [e.time_range.elapsed_us() for e in kernels]
    if not all(x > 0 for x in samples):
        raise RuntimeError(f"Invalid device durations: {samples}")
    return dict(us=statistics.median(samples), samples_us=samples)


class CandidatePlan:
    def __init__(self, tensors, top_k, warps):
        from importlib import import_module

        import torch
        import triton
        from hygon_prefill_bitonic_kernel import bitonic_topk_indices

        launcher = import_module(
            "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
        )._Launch
        x, starts, ends = tensors
        rows, vocab = x.shape
        block = triton.next_power_of_2(vocab)
        self.guard = torch.full(
            (rows * top_k + 32,), -123456, device=x.device, dtype=torch.int32
        )
        self.out = self.guard[16:-16].view(rows, top_k)
        self.args = (x, starts, ends, self.out, x.stride(0))
        self.launch = launcher(
            bitonic_topk_indices,
            (rows,),
            dict(TOPK=top_k, BLOCK=block),
            warps,
        )

    def __call__(self):
        self.launch(*self.args)


def check_cases(make, shape):
    rows, vocab, top_k, stride0 = shape
    for case in ("tied", "constant", "partial", "short", "special", "strided"):
        n = min(rows, 32)
        tensors = inputs(n, vocab, max(stride0, vocab + 8), top_k, 123, case)
        want = oracle(tensors, top_k)
        for arm, factory in make.items():
            plan = factory(tensors)
            validate(plan, tensors, top_k, want)
            emit("validation", arm=arm, case=case, ok=True)
            del plan


def paired(plans, rounds, iters, seed):
    all_ratios = []
    names = {
        "default": "non_tle_top_k_per_row_prefill",
        "candidate": "bitonic_topk_indices",
    }
    for round_id in range(rounds):
        order = (
            ("default", "candidate", "candidate", "default")
            if round_id % 2 == 0
            else ("candidate", "default", "default", "candidate")
        )
        readings = [(arm, device_time(plans[arm], names[arm], iters)) for arm in order]
        ratios = []
        for i, j in ((0, 1), (3, 2)):
            pair = dict(readings[p] for p in (i, j))
            ratios.append(pair["default"]["us"] / pair["candidate"]["us"])
        all_ratios.extend(ratios)
        emit(
            "paired_round",
            seed=seed,
            round=round_id,
            order=order,
            readings=readings,
            ratios=ratios,
        )
    return all_ratios


def worker(args):
    from importlib import import_module

    import torch
    import triton

    import flaggems_vllm

    shape_id, warps = TARGETS[args.worker]
    rows, vocab, top_k, stride0 = SHAPES[shape_id]
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    if flaggems_vllm.top_k_per_row_prefill is not ov.top_k_per_row_prefill:
        raise RuntimeError("Public operator is not the Hygon override")
    if (
        ov._dense_carry is None
        or not ov._ONESCAN_PATH
        or not ov._ENABLED
        or not ov._GEOMETRY
    ):
        raise RuntimeError("Expected the shipped default carry route")
    carry_source = Path(ov._dense_carry.__file__).read_text()
    source_sha = hashlib.sha256(carry_source.encode()).hexdigest()
    if source_sha != EXPECTED_CARRY_SHA256 or carry_source != build(
        ROOT, True, "carry"
    ):
        raise RuntimeError(f"Default carry source drifted: {source_sha}")
    if vocab > ov.DENSE_VOCAB_PER_TOPK * top_k:
        raise RuntimeError("A sparse shape reached the dense feasibility probe")
    geo = ov._geometry(rows, vocab)
    block, base_warps = geo or (512, 8)
    emit(
        "device",
        shape=SHAPES[shape_id],
        candidate_block=triton.next_power_of_2(vocab),
        candidate_warps=warps,
        default_block=block,
        default_warps=base_warps,
        carry_sha256=source_sha,
        props=str(torch.cuda.get_device_properties(torch.cuda.current_device())),
        torch=torch.__version__,
        triton=triton.__version__,
    )
    factories = {
        "default": lambda tensors: Plan(
            ov._dense_carry, tensors, top_k, block, base_warps
        ),
        "candidate": lambda tensors: CandidatePlan(tensors, top_k, warps),
    }
    check_cases(factories, SHAPES[shape_id])
    ratios = []
    for seed in args.seeds:
        tensors = inputs(rows, vocab, stride0, top_k, seed)
        want = oracle(tensors, top_k)
        plans = {arm: make(tensors) for arm, make in factories.items()}
        for arm, plan in plans.items():
            validate(plan, tensors, top_k, want)
            emit("validation", arm=arm, case="full_normal", seed=seed, ok=True)
        x, starts, ends = tensors
        public_out = torch.empty_like(plans["default"].out)
        flaggems_vllm.top_k_per_row_prefill(
            x, starts, ends, public_out, rows, stride0, 1, top_k
        )
        check_output(public_out, tensors, top_k, want)
        ratios.extend(paired(plans, args.rounds, args.iters, seed))
        del tensors, want, plans, public_out
    emit(
        "worker_summary",
        shape_id=shape_id,
        shape=SHAPES[shape_id],
        warps=warps,
        median=statistics.median(ratios),
        seed_medians=[
            statistics.median(ratios[i * args.rounds * 2 : (i + 1) * args.rounds * 2])
            for i in range(len(args.seeds))
        ],
        min=min(ratios),
        max=max(ratios),
        ratios=ratios,
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, choices=range(len(TARGETS)))
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
        targets=[(SHAPES[shape_id], warps) for shape_id, warps in TARGETS],
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
    for target_id, (shape_id, warps) in enumerate(TARGETS):
        emit("worker_start", target_id=target_id, shape_id=shape_id, warps=warps)
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
