"""BW1000 follow-up: four-row launch sweep, then rank8 x carry factorial.

Run each stage through hygon_prefill_next_run.sh on the Hygon worktree.
The parent uses one subprocess per shape/config to preserve partial reports
if a Triton configuration faults. This is a device-kernel-only experiment;
production routing and kernels are never modified.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from hygon_prefill_audit import (
    SHAPES,
    Plan,
    check_output,
    device_time,
    emit,
    inputs,
    load_module,
    occupancy,
    oracle,
    validate,
)
from hygon_prefill_audit_source import GENERIC, OVERRIDE, build

ROOT = Path(__file__).resolve().parents[1]
FOUR_ROW_IDS = (1, 3)
DENSE_IDS = (2, 4, 5, 6)
BASELINE = (512, 8)
# Do not repeat the old num_warps=1 wrong answers or B1024/w2,w4 faults.
LAUNCH_CONFIGS = (
    (256, 2),
    (256, 4),
    (256, 8),
    (512, 4),
    (512, 16),
    (1024, 8),
    (1024, 16),
)
COMBO_PAIRS = (("control", "rank8"), ("control", "carry"), ("carry", "rank8_carry"))


def readings(plans, base, candidate, rounds, iters, seed):
    ratios = []
    for round_id in range(rounds):
        order = (
            (base, candidate, candidate, base)
            if round_id % 2 == 0
            else (candidate, base, base, candidate)
        )
        timed = [(arm, device_time(plans[arm], iters)) for arm in order]
        for left, right in ((0, 1), (3, 2)):
            pair = dict(timed[p] for p in (left, right))
            ratios.append(pair[base]["us"] / pair[candidate]["us"])
        emit(
            "paired_round",
            seed=seed,
            round=round_id,
            base=base,
            candidate=candidate,
            order=order,
            readings=timed,
            ratios=ratios[-2:],
        )
    return ratios


def check_cases(plans, shape, label):
    rows, vocab, top_k, stride0 = shape
    for case in ("tied", "constant", "partial", "short", "special", "strided"):
        # Five rows make the short case cover lengths 0,1,k-1,k,k+1.
        n = 5 if rows == 4 else min(rows, 32)
        tensors = inputs(n, vocab, max(stride0, vocab + 8), top_k, 123, case)
        want = oracle(tensors, top_k)
        for arm, make_plan in plans.items():
            plan = make_plan(tensors)
            validate(plan, tensors, top_k, want)
            emit("validation", stage=label, case=case, arm=arm, ok=True)
            del plan


def launch_worker(args, ov):
    import torch

    shape = SHAPES[args.shape_id]
    if args.shape_id not in FOUR_ROW_IDS:
        raise ValueError("Launch sweep must use a four-row shape")
    rows, vocab, top_k, stride0 = shape
    if ov._geometry(rows, vocab) is not None or ov._sparse.HAS_TLE:
        raise RuntimeError("Expected the shipped sparse, non-TLE four-row path")
    if (ov._sparse.NUM_THREADS_PER_BLOCK, ov._sparse._num_warps(512)) != BASELINE:
        raise RuntimeError("Production default launch geometry drifted")
    config = LAUNCH_CONFIGS[args.config_id]
    mod = ov._sparse
    emit(
        "config", shape=shape, baseline=BASELINE, candidate=config, module=mod.__file__
    )
    factories = {
        "default": lambda tensors: Plan(mod, tensors, top_k, *BASELINE),
        "candidate": lambda tensors: Plan(mod, tensors, top_k, *config),
    }
    check_cases(factories, shape, "launch")
    all_ratios = []
    for seed in args.seeds:
        tensors = inputs(rows, vocab, stride0, top_k, seed)
        want = oracle(tensors, top_k)
        plans = {arm: make(tensors) for arm, make in factories.items()}
        for arm, plan in plans.items():
            validate(plan, tensors, top_k, want)
            emit(
                "validation",
                stage="launch",
                case="full_normal",
                seed=seed,
                arm=arm,
                ok=True,
            )
        public_out = torch.empty_like(plans["default"].out)
        x, starts, ends = tensors
        import flaggems_vllm

        flaggems_vllm.top_k_per_row_prefill(
            x, starts, ends, public_out, rows, stride0, 1, top_k
        )
        check_output(public_out, tensors, top_k, want)
        all_ratios.extend(
            readings(plans, "default", "candidate", args.rounds, args.iters, seed)
        )
        del plans, tensors, want, public_out
    emit(
        "worker_summary",
        stage="launch",
        shape_id=args.shape_id,
        shape=shape,
        candidate=config,
        ratio_median=statistics.median(all_ratios),
        ratio_min=min(all_ratios),
        ratio_max=max(all_ratios),
        ratios=all_ratios,
    )


def combo_worker(args, ov):
    if args.shape_id not in DENSE_IDS:
        raise ValueError("Carry combination must use a dense shape")
    shape = SHAPES[args.shape_id]
    rows, vocab, top_k, stride0 = shape
    if vocab > ov.DENSE_VOCAB_PER_TOPK * top_k or ov._dense_carry is None:
        raise RuntimeError("Expected the shipped dense carry path")
    production = Path(ov._dense_carry.__file__).read_text()
    expected = build(ROOT, True, "carry")
    if production != expected:
        raise RuntimeError("Audited carry arm no longer equals production source")
    emit(
        "production_carry_sha256",
        sha256=hashlib.sha256(production.encode()).hexdigest(),
    )
    block, warps = ov._geometry(rows, vocab) or BASELINE
    emit(
        "config",
        shape=shape,
        block=block,
        warps=warps,
        arms=("control", "rank8", "carry", "rank8_carry"),
    )
    with tempfile.TemporaryDirectory(prefix="hygon_prefill_combo_") as folder:
        modules = {
            arm: load_module(folder, True, arm)
            for arm in ("control", "rank8", "carry", "rank8_carry")
        }
        factories = {
            arm: (lambda tensors, mod=mod: Plan(mod, tensors, top_k, block, warps))
            for arm, mod in modules.items()
        }
        check_cases(factories, shape, "combo")
        ratios = {f"{base}->{candidate}": [] for base, candidate in COMBO_PAIRS}
        for seed in args.seeds:
            tensors = inputs(rows, vocab, stride0, top_k, seed)
            want = oracle(tensors, top_k)
            plans = {arm: make(tensors) for arm, make in factories.items()}
            for arm, plan in plans.items():
                validate(plan, tensors, top_k, want)
                emit(
                    "validation",
                    stage="combo",
                    case="full_normal",
                    seed=seed,
                    arm=arm,
                    ok=True,
                )
            for base, candidate in COMBO_PAIRS:
                ratios[f"{base}->{candidate}"].extend(
                    readings(plans, base, candidate, args.rounds, args.iters, seed)
                )
            del plans, tensors, want
    emit(
        "worker_summary",
        stage="combo",
        shape_id=args.shape_id,
        shape=shape,
        medians={key: statistics.median(values) for key, values in ratios.items()},
        ranges={key: (min(values), max(values)) for key, values in ratios.items()},
        ratios=ratios,
    )


def worker(args):
    from importlib import import_module

    import torch
    import triton

    import flaggems_vllm

    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    if flaggems_vllm.top_k_per_row_prefill is not ov.top_k_per_row_prefill:
        raise RuntimeError("Public operator is not the Hygon override")
    if not ov._ONESCAN_PATH or not ov._ENABLED or not ov._GEOMETRY:
        raise RuntimeError("Expected default one-scan, slot-scan and geometry")
    emit(
        "device",
        props=str(torch.cuda.get_device_properties(torch.cuda.current_device())),
        torch=torch.__version__,
        triton=triton.__version__,
        shape_id=args.shape_id,
    )
    if args.stage == "launch":
        launch_worker(args, ov)
    else:
        combo_worker(args, ov)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("launch", "combo"))
    parser.add_argument("--shape-id", type=int)
    parser.add_argument("--config-id", type=int)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43))
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.rounds < 2 or args.iters < 2 or not args.seeds:
        parser.error("At least two rounds/iterations and one seed are required")
    if args.worker:
        return worker(args)
    targets = (
        [(sid, cid) for sid in FOUR_ROW_IDS for cid in range(len(LAUNCH_CONFIGS))]
        if args.stage == "launch"
        else [(sid, None) for sid in DENSE_IDS]
    )
    emit(
        "probe",
        stage=args.stage,
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        host=platform.node(),
        shapes=[SHAPES[sid] for sid, _ in targets],
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
    summaries, failed = [], []
    for shape_id, config_id in targets:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            args.stage,
            "--worker",
            "--shape-id",
            str(shape_id),
            "--rounds",
            str(args.rounds),
            "--iters",
            str(args.iters),
            "--seeds",
            *map(str, args.seeds),
        ]
        if config_id is not None:
            cmd.extend(("--config-id", str(config_id)))
        emit("worker_start", shape_id=shape_id, config_id=config_id)
        try:
            proc = subprocess.run(
                cmd, cwd=ROOT, capture_output=True, text=True, timeout=1200
            )
            print(proc.stdout, end="", flush=True)
            print(proc.stderr, end="", flush=True)
            if proc.returncode:
                failed.append((shape_id, config_id))
            for line in proc.stdout.splitlines():
                if line.startswith('{"'):
                    record = json.loads(line)
                    if record.get("kind") == "worker_summary":
                        summaries.append(record)
            emit(
                "worker_exit",
                shape_id=shape_id,
                config_id=config_id,
                code=proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            failed.append((shape_id, config_id))
            print((exc.stdout or b"").decode(errors="replace"), flush=True)
            emit("worker_timeout", shape_id=shape_id, config_id=config_id, seconds=1200)
    occupancy("after")
    emit(
        "probe_complete",
        stage=args.stage,
        ok=not failed,
        failed=failed,
        summaries=summaries,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
