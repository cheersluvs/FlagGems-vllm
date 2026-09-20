"""BW1000 probe for the 16-elements-per-lane hypothesis.

The earlier launch sweep reached 16 elements/lane only for a subset of the
shapes, and the VEC sweep kept the production warp count fixed.  This probe
does the missing joint experiment: it builds the exact production source for
VEC=2/4/8 and measures several BLOCK x num_warps pairs whose tile maps to 16
elements per wave lane.  Each shape/config is isolated in a child process so
a bad HIP configuration cannot discard the rest of the report.

This is a measurement-only probe.  It never changes the Hygon override.
"""

import argparse
import importlib.util
import math
import os
import statistics
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from importlib import import_module

from hygon_prefill_audit import (
    SHAPES,
    Plan,
    device_time,
    emit,
    inputs,
    oracle,
    occupancy,
    validate,
)
from hygon_prefill_audit_source import digest
from hygon_prefill_vec_source import variants

ROOT = Path(__file__).resolve().parents[1]

# Every tuple satisfies BLOCK * VEC / (num_warps * 64) == 16.  The
# num_warps=1 alternatives are intentionally omitted: an earlier Hygon sweep
# showed wrong answers for those launches.
CONFIGS = (
    (4, 512, 2),
    (4, 1024, 4),
    (8, 256, 2),
    (8, 512, 4),
    (2, 1024, 2),
)


def load_variant(folder, source, dense, vec):
    name = f"flaggems_vllm.ops._hygon_lane16_{int(dense)}_{vec}"
    path = Path(folder) / f"_hygon_lane16_{int(dense)}_{vec}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if mod.HAS_TLE:
        raise RuntimeError("Expected non-TLE VEC variant")
    return mod


def paired(plans, rounds, iters, seed):
    ratios = []
    readings = []
    for round_id in range(rounds):
        order = (
            ("control", "candidate", "candidate", "control")
            if round_id % 2 == 0
            else ("candidate", "control", "control", "candidate")
        )
        timed = [(arm, device_time(plans[arm], iters)["us"]) for arm in order]
        for left, right in ((0, 1), (3, 2)):
            pair = dict(timed[p] for p in (left, right))
            ratios.append(pair["control"] / pair["candidate"])
        readings.append(timed)
    return ratios, readings


def worker(args):
    import triton

    rows, vocab, top_k, stride0 = SHAPES[args.shape_id]
    vec, block, warps = CONFIGS[args.config_id]
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.warp_size != 64:
        raise RuntimeError(f"Expected Hygon HIP wave64, got {target}")

    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    dense = vocab <= ov.DENSE_VOCAB_PER_TOPK * top_k
    production = ov._dense_carry if dense else ov._sparse
    if production is None or production.HAS_TLE:
        raise RuntimeError("Expected shipped non-TLE production path")
    base_block, base_warps = ov._geometry(rows, vocab) or (512, 8)
    emit(
        "lane16_config",
        shape_id=args.shape_id,
        shape=SHAPES[args.shape_id],
        dense=dense,
        baseline=[base_block, base_warps],
        candidate=[block, warps],
        vec=vec,
        elements_per_lane=block * vec / (warps * target.warp_size),
        source_sha256=digest(variants(ROOT, dense)[vec]),
    )

    with tempfile.TemporaryDirectory(prefix="hygon_lane16_") as folder:
        source = variants(ROOT, dense)
        candidate_mod = load_variant(folder, source[vec], dense, vec)
        errors = []
        for case in ("tied", "constant", "partial", "short", "special", "strided"):
            n = 5 if rows == 4 else min(rows, 32)
            tensors = inputs(n, vocab, max(stride0, vocab + 8), top_k, 123, case)
            want = oracle(tensors, top_k)
            for label, mod, b, w in (
                ("control", production, base_block, base_warps),
                ("candidate", candidate_mod, block, warps),
            ):
                try:
                    validate(Plan(mod, tensors, top_k, b, w), tensors, top_k, want)
                    emit("lane16_validation", shape_id=args.shape_id, config_id=args.config_id,
                         arm=label, case=case, ok=True)
                except Exception as exc:
                    errors.append((label, case, repr(exc)))
                    emit("lane16_validation", shape_id=args.shape_id, config_id=args.config_id,
                         arm=label, case=case, ok=False, error=repr(exc))
                    if label == "control":
                        raise
        if errors:
            raise RuntimeError(f"candidate validation failed: {errors}")

        ratios = []
        for seed in args.seeds:
            tensors = inputs(rows, vocab, stride0, top_k, seed)
            want = oracle(tensors, top_k)
            plans = {
                "control": Plan(production, tensors, top_k, base_block, base_warps),
                "candidate": Plan(candidate_mod, tensors, top_k, block, warps),
            }
            for label, plan in plans.items():
                validate(plan, tensors, top_k, want)
                emit("lane16_validation", shape_id=args.shape_id, config_id=args.config_id,
                     arm=label, case="normal", seed=seed, ok=True)
            values, readings = paired(plans, args.rounds, args.iters, seed)
            ratios.extend(values)
            emit(
                "lane16_timing",
                shape_id=args.shape_id,
                config_id=args.config_id,
                seed=seed,
                ratios=values,
                readings=readings,
            )
        emit(
            "lane16_worker_summary",
            shape_id=args.shape_id,
            config_id=args.config_id,
            shape=SHAPES[args.shape_id],
            config=[vec, block, warps],
            ratio_median=statistics.median(ratios),
            ratio_min=min(ratios),
            ratio_max=max(ratios),
            ratios=ratios,
        )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shape-id", type=int)
    ap.add_argument("--config-id", type=int)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    args = ap.parse_args()
    if args.worker:
        try:
            return worker(args)
        except Exception:
            traceback.print_exc()
            return 1

    occupancy("before")
    failures = []
    for config_id, config in enumerate(CONFIGS):
        for shape_id, shape in enumerate(SHAPES):
            emit("lane16_worker_start", config_id=config_id, config=config,
                 shape_id=shape_id, shape=shape)
            cmd = [
                sys.executable, "-u", __file__, "--worker",
                "--shape-id", str(shape_id), "--config-id", str(config_id),
                "--rounds", str(args.rounds), "--iters", str(args.iters),
                "--seeds", *map(str, args.seeds),
            ]
            try:
                code = subprocess.run(
                    cmd, timeout=args.timeout,
                    env=dict(os.environ, FLAGGEMS_FORCE_TLE="0"),
                ).returncode
            except subprocess.TimeoutExpired:
                code = 124
            emit("lane16_worker_exit", config_id=config_id, shape_id=shape_id, code=code)
            if code:
                failures.append([config_id, shape_id])
    occupancy("after")
    emit("lane16_suite_summary", configs=CONFIGS, failures=failures,
         all_are_16_elems_per_lane=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
