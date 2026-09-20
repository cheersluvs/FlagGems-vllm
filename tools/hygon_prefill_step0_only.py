"""Isolated normal-input STEP-0-only code-size/performance upper bound.

The candidate intentionally lacks STEP-1..3 and is INVALID when the first
boundary bin overflows. Never route production inputs to this candidate.
Only validated normal seeds are timed, with independent A/A controls.
"""

import argparse
import importlib.util
import statistics
import subprocess
import sys
import tempfile
import traceback
from importlib import import_module
from pathlib import Path

from hygon_prefill_audit import Plan, SHAPES, emit, inputs, oracle, validate
from hygon_prefill_next import readings
from hygon_prefill_step0_source import digest, variants


ROOT = Path(__file__).resolve().parents[1]


def load_module(folder, source, dense):
    name = f"flaggems_vllm.ops._hygon_step0_{int(dense)}"
    path = Path(folder) / f"_hygon_step0_{int(dense)}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if mod.HAS_TLE:
        raise RuntimeError("STEP-0 upper bound expects the non-TLE path")
    return mod


def code_size(mod, plan, k, block, warps, rows):
    import torch

    ck = mod.non_tle_top_k_per_row_prefill.run(
        *plan.args, TOPK=k, BLOCK_SIZE=block, ROW_OFFSET=0,
        num_warps=warps, grid=(rows,), warmup=False,
    )
    torch.cuda.synchronize()
    if ck is None:
        raise RuntimeError("JIT returned no compiled kernel")
    target = ck.asm.get("amdgcn")
    if not isinstance(target, str):
        raise RuntimeError(f"Expected AMDGCN text, got {list(ck.asm)}")
    return dict(
        target_bytes=len(target.encode()), registers=ck.n_regs,
        spills=ck.n_spills, shared_bytes=ck.metadata.shared,
        static_barriers=sum(line.strip().startswith("s_barrier") for line in target.splitlines()),
    )


def worker(args):
    from flaggems_vllm import vendor_name

    if vendor_name != "hygon":
        raise RuntimeError("Hygon-only probe")
    shape = SHAPES[args.shape_id]
    rows, vocab, k, stride0 = shape
    ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
    dense = vocab <= ov.DENSE_VOCAB_PER_TOPK * k
    production = ov._dense_carry if dense else ov._sparse
    if production is None or production.HAS_TLE:
        raise RuntimeError("Expected current non-TLE production module")
    control, fast = variants(ROOT, dense)
    if Path(production.__file__).read_text() != control:
        raise RuntimeError("Generated control differs from current production source")
    block, warps = ov._geometry(rows, vocab) or (512, 8)
    emit("step0_config", shape_id=args.shape_id, shape=shape,
         geometry=[block, warps], dense=dense,
         source_control=digest(control), source_fast=digest(fast),
         restriction="normal-only upper bound; no overflow fallback")
    with tempfile.TemporaryDirectory(prefix="hygon_step0_") as folder:
        candidate_mod = load_module(folder, fast, dense)
        sizes_recorded = False
        for seed in args.seeds:
            tensors = inputs(rows, vocab, stride0, k, seed)
            want = oracle(tensors, k)
            plans = {
                "control_a": Plan(production, tensors, k, block, warps),
                "control_b": Plan(production, tensors, k, block, warps),
                "fast": Plan(candidate_mod, tensors, k, block, warps),
            }
            # A silent overflow must fail validation before any timing.
            for arm, plan in plans.items():
                validate(plan, tensors, k, want)
                emit("step0_validation", shape_id=args.shape_id,
                     seed=seed, arm=arm, ok=True)
            if not sizes_recorded:
                emit("step0_codegen", shape_id=args.shape_id,
                     control=code_size(production, plans["control_a"], k, block, warps, rows),
                     fast=code_size(candidate_mod, plans["fast"], k, block, warps, rows))
                sizes_recorded = True
            aa = readings(plans, "control_a", "control_b", args.rounds, args.iters, seed)
            ab = readings(plans, "control_a", "fast", args.rounds, args.iters, seed)
            emit("step0_seed_summary", shape_id=args.shape_id, seed=seed,
                 aa_median=statistics.median(aa), aa_range=[min(aa), max(aa)],
                 ab_median=statistics.median(ab), ab_range=[min(ab), max(ab)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shape-id", type=int)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43))
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.rounds < 2 or args.iters < 2 or args.timeout < 1 or not args.seeds:
        parser.error("rounds/iters >= 2, timeout >= 1, nonempty seeds required")
    if args.worker:
        if args.shape_id is None or args.shape_id not in range(len(SHAPES)):
            parser.error("worker requires a valid --shape-id")
        worker(args)
        return
    failures = []
    for shape_id, shape in enumerate(SHAPES):
        emit("step0_worker_start", shape_id=shape_id, shape=shape)
        command = [sys.executable, "-u", __file__, "--worker",
                   "--shape-id", str(shape_id), "--rounds", str(args.rounds),
                   "--iters", str(args.iters), "--seeds", *map(str, args.seeds)]
        try:
            result = subprocess.run(command, timeout=args.timeout, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
        emit("step0_worker_exit", shape_id=shape_id, code=code)
        if code:
            failures.append(shape_id)
    emit("step0_suite_summary", failures=failures)
    if failures:
        raise RuntimeError(f"STEP-0 workers failed: {failures}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
