"""Focused BW1000 follow-up for gaps-v1 correctness and A/A timing anomalies."""

import argparse
import statistics
import subprocess
import sys
import tempfile
import traceback
from importlib import import_module
from pathlib import Path

from hygon_prefill_audit import emit, inputs, occupancy, oracle, validate
from hygon_prefill_gaps import CROSSOVER, GEOMETRIES, GapPlan, baseline, module
from hygon_prefill_gaps_source import variant
from hygon_prefill_next import check_cases, readings

ROOT = Path(__file__).resolve().parents[1]
FAIL_SHAPES = (12, 13)
TIMING_SHAPE = 15
TIMING_CONFIGS = (3, 5)


def production_module():
    import flaggems_vllm

    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    if flaggems_vllm.vendor_name != "hygon":
        raise RuntimeError("Hygon-only focus probe")
    shape = CROSSOVER[FAIL_SHAPES[0]]
    dense = shape[1] <= ov.DENSE_VOCAB_PER_TOPK * shape[2]
    shipped = ov._dense_carry if dense else ov._sparse
    if shipped is None or shipped.HAS_TLE:
        raise RuntimeError("Expected production non-TLE module")
    if Path(shipped.__file__).read_text() != variant(ROOT, dense, "control"):
        raise RuntimeError("Generated control differs from production")
    return ov, dense


def mismatch_details(plan, tensors, k, want):
    import torch

    x, starts, ends = tensors
    lens = ends - starts
    valid = torch.arange(k, device=x.device)[None, :] < lens[:, None]
    bounds = torch.where(
        valid, (plan.out >= 0) & (plan.out < lens[:, None]), plan.out == -1
    )
    safe = torch.where(
        valid, (plan.out + starts[:, None]).clamp(0, x.shape[1] - 1), 0
    )
    got = torch.where(valid, x.gather(1, safe.long()), float("-inf"))
    got = got.sort(dim=1).values
    bad_values = got != want
    bad_rows = (~bounds.all(dim=1)) | bad_values.any(dim=1)
    row_ids = torch.nonzero(bad_rows).flatten()
    if row_ids.numel() == 0:
        return {"bad_rows": 0, "note": "Mismatch was transient on second inspection"}
    row = int(row_ids[0])
    positions = torch.nonzero(bad_values[row]).flatten()[:8].cpu().tolist()
    result = {
        "bad_rows": int(row_ids.numel()),
        "first_bad_row": row,
        "row_start": int(starts[row]),
        "row_end": int(ends[row]),
        "bad_value_positions": positions,
        "got_values": [float(got[row, pos]) for pos in positions],
        "want_values": [float(want[row, pos]) for pos in positions],
        "bounds_ok": bool(bounds[row].all()),
        "duplicate_indices": bool(
            (
                plan.out[row].sort().values[1:]
                == plan.out[row].sort().values[:-1]
            ).any()
        ),
        "count": int(plan.count[row]),
        "step": int(plan.step[row]),
        "bin_size": int(plan.bin_size[row]),
        "found": int(plan.found[row]),
    }
    return result


def correctness_worker(args, ov, dense):
    import torch

    shape = CROSSOVER[args.shape_id]
    rows, vocab, k, stride0 = shape
    block, warps = ov._geometry(rows, vocab) or (512, 8)
    cb, cw = GEOMETRIES[args.config_id]
    with tempfile.TemporaryDirectory(prefix="hygon_focus_") as folder:
        mod = module(folder, dense, "control")
        factories = {
            "control": lambda ts: GapPlan(mod, ts, k, block, warps),
            "candidate": lambda ts: GapPlan(mod, ts, k, cb, cw),
        }
        emit(
            "focus_config", mode="correctness", shape=shape,
            control=[block, warps], candidate=[cb, cw],
        )
        check_cases(factories, shape, "focus")
        failures = 0
        for repeat in range(args.repeats):
            for seed in args.seeds:
                tensors = inputs(rows, vocab, stride0, k, seed)
                want = oracle(tensors, k)
                plans = {name: make(tensors) for name, make in factories.items()}
                for name, plan in plans.items():
                    try:
                        validate(plan, tensors, k, want)
                    except AssertionError as exc:
                        failures += 1
                        emit(
                            "focus_mismatch",
                            shape_id=args.shape_id,
                            config_id=args.config_id,
                            arm=name,
                            seed=seed,
                            repeat=repeat,
                            error=str(exc),
                            details=mismatch_details(plan, tensors, k, want),
                        )
                    else:
                        emit(
                            "focus_validation", shape_id=args.shape_id,
                            config_id=args.config_id, arm=name, seed=seed,
                            repeat=repeat, ok=True,
                        )
                del plans, tensors, want
                torch.cuda.synchronize()
        emit(
            "focus_summary", mode="correctness", shape_id=args.shape_id,
            config_id=args.config_id, failures=failures,
            attempts=args.repeats * len(args.seeds) * 2,
        )
        if failures:
            raise RuntimeError(f"{failures} exact top-k validation failures")


def swap_scratch(a, b):
    aa, bb = a.args, b.args
    a.args = (*aa[:7], *bb[7:])
    b.args = (*bb[:7], *aa[7:])


def paired(plans, label, args, seed):
    ratios = readings(plans, "control", "candidate", args.rounds, args.iters, seed)
    emit(
        "focus_result", mode="timing", comparison=label, seed=seed,
        median=statistics.median(ratios), min=min(ratios), max=max(ratios),
    )


def timing_worker(args, ov, dense):
    shape = CROSSOVER[TIMING_SHAPE]
    rows, vocab, k, stride0 = shape
    cb, cw = GEOMETRIES[args.config_id]
    block, warps = ov._geometry(rows, vocab) or (512, 8)
    with tempfile.TemporaryDirectory(prefix="hygon_focus_") as folder:
        mod = module(folder, dense, "control")
        emit(
            "focus_config", mode="timing", shape=shape,
            control=[block, warps], candidate=[cb, cw],
        )
        for seed in args.seeds:
            tensors = inputs(rows, vocab, stride0, k, seed)
            want = oracle(tensors, k)
            a = GapPlan(mod, tensors, k, block, warps)
            b = GapPlan(mod, tensors, k, block, warps)
            for plan in (a, b):
                validate(plan, tensors, k, want)
            emit(
                "workspace", seed=seed, phase="initial",
                control=[int(t.data_ptr() % 4096) for t in a.args[7:]],
                candidate=[int(t.data_ptr() % 4096) for t in b.args[7:]],
            )
            paired({"control": a, "candidate": a}, "same_object", args, seed)
            paired({"control": a, "candidate": b}, "independent_aa", args, seed)
            swap_scratch(a, b)
            for plan in (a, b):
                validate(plan, tensors, k, want)
            paired({"control": a, "candidate": b}, "swapped_workspace_aa", args, seed)
            c = GapPlan(mod, tensors, k, cb, cw)
            validate(c, tensors, k, want)
            paired({"control": a, "candidate": c}, "geometry_vs_a", args, seed)
            paired({"control": b, "candidate": c}, "geometry_vs_b", args, seed)


def worker(args):
    import torch
    import triton

    ov, dense = production_module()
    emit(
        "device", mode=args.mode, shape_id=args.shape_id,
        torch=torch.__version__, triton=triton.__version__,
        warp_size=torch.cuda.get_device_properties(0).warp_size,
    )
    if args.mode == "correctness":
        correctness_worker(args, ov, dense)
    else:
        timing_worker(args, ov, dense)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("correctness", "timing"))
    parser.add_argument("--shape-id", type=int, default=TIMING_SHAPE)
    parser.add_argument("--config-id", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43))
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if min(args.repeats, args.rounds, args.iters, args.timeout) <= 0:
        parser.error("repeats, rounds, iters and timeout must be positive")
    if args.worker:
        try:
            worker(args)
            return 0
        except Exception:
            traceback.print_exc()
            return 1
    tasks = [("correctness", sid, cid) for sid in FAIL_SHAPES for cid in (0, 1)]
    tasks += [("timing", TIMING_SHAPE, cid) for cid in TIMING_CONFIGS]
    occupancy("before")
    baseline()
    failures = []
    for mode, sid, cid in tasks:
        emit("worker_start", mode=mode, shape_id=sid, config_id=cid)
        cmd = [
            sys.executable, "-u", __file__, "--worker", "--mode", mode,
            "--shape-id", str(sid), "--config-id", str(cid),
            "--repeats", str(args.repeats), "--rounds", str(args.rounds),
            "--iters", str(args.iters), "--seeds", *map(str, args.seeds),
        ]
        try:
            result = subprocess.run(cmd, timeout=args.timeout, capture_output=True, text=True)
            print(result.stdout, end="", flush=True)
            print(result.stderr, end="", file=sys.stderr, flush=True)
            code = result.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            print(stdout, end="", flush=True)
            print(stderr, end="", file=sys.stderr, flush=True)
            code = 124
        emit("worker_exit", mode=mode, shape_id=sid, config_id=cid, code=code)
        if code:
            failures.append((mode, sid, cid, code))
    occupancy("after")
    emit("suite_summary", requested_workers=len(tasks), failures=failures)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
