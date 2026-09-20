"""Full-operator BW1000 VEC sweep at the shipped routing and geometry."""

import argparse
import importlib.util
import os
import statistics
import subprocess
import sys
import tempfile
import traceback
from importlib import import_module
from pathlib import Path

from hygon_prefill_audit import Plan, SHAPES, emit, inputs, occupancy, oracle, validate
from hygon_prefill_audit_source import digest
from hygon_prefill_next import readings
from hygon_prefill_vec_source import variants

ROOT = Path(__file__).resolve().parents[1]


def load(folder, source, vec):
    name = f"flaggems_vllm.ops._hygon_vec_{vec}"
    path = Path(folder) / f"_hygon_vec_{vec}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if mod.HAS_TLE:
        raise RuntimeError("Expected non-TLE variant")
    return mod


def metadata(mod, plan, k, block, warps, rows):
    import torch

    ck = mod.non_tle_top_k_per_row_prefill.run(
        *plan.args, TOPK=k, BLOCK_SIZE=block, ROW_OFFSET=0,
        num_warps=warps, grid=(rows,), warmup=False,
    )
    torch.cuda.synchronize()
    if ck is None:
        raise RuntimeError("JIT did not return a compiled kernel")
    asm = ck.asm.get("amdgcn")
    return dict(
        registers=ck.n_regs, spills=ck.n_spills,
        shared_bytes=ck.metadata.shared,
        target_bytes=len(asm.encode()) if isinstance(asm, str) else None,
        static_barriers=(
            sum(line.strip().startswith("s_barrier") for line in asm.splitlines())
            if isinstance(asm, str) else None
        ),
    )


def worker(args):
    import torch
    import triton
    from flaggems_vllm import vendor_name

    if vendor_name != "hygon":
        raise RuntimeError("Hygon-only VEC sweep")
    rows, vocab, k, stride0 = SHAPES[args.shape_id]
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.warp_size != 64:
        raise RuntimeError(f"Expected HIP wave64, got {target}")
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    dense = vocab <= ov.DENSE_VOCAB_PER_TOPK * k
    production = ov._dense_carry if dense else ov._sparse
    if production is None or production.HAS_TLE:
        raise RuntimeError("Expected production non-TLE kernel")
    sources = variants(ROOT, dense)
    if Path(production.__file__).read_text() != sources[4]:
        raise RuntimeError("Generated VEC=4 control differs from production")
    block, warps = ov._geometry(rows, vocab) or (512, 8)
    emit(
        "vec_config", shape_id=args.shape_id, shape=SHAPES[args.shape_id],
        dense=dense, geometry=[block, warps],
        sources={str(vec): digest(src) for vec, src in sources.items()},
        warp_size=target.warp_size,
    )
    errors = []
    with tempfile.TemporaryDirectory(prefix="hygon_vec_") as folder:
        modules = {4: production}
        modules.update(
            {v: load(folder, s, v) for v, s in sources.items() if v != 4}
        )
        viable = set(modules)
        for case in ("tied", "constant", "partial", "short", "special", "strided"):
            n = 5 if rows == 4 else min(rows, 32)
            tensors = inputs(n, vocab, max(stride0, vocab + 8), k, 123, case)
            want = oracle(tensors, k)
            for vec in sorted(viable):
                try:
                    plan = Plan(modules[vec], tensors, k, block, warps)
                    validate(plan, tensors, k, want)
                    emit(
                        "vec_validation", shape_id=args.shape_id,
                        vec=vec, case=case, ok=True,
                    )
                except Exception as exc:
                    viable.remove(vec)
                    errors.append([vec, case, str(exc)])
                    emit("vec_validation", shape_id=args.shape_id, vec=vec, case=case,
                         ok=False, error=str(exc))
        if 4 not in viable:
            raise RuntimeError(f"Production control failed: {errors}")
        for seed in args.seeds:
            tensors = inputs(rows, vocab, stride0, k, seed)
            want = oracle(tensors, k)
            plans = {
                "control_a": Plan(production, tensors, k, block, warps),
                "control_b": Plan(production, tensors, k, block, warps),
            }
            for vec in sorted(viable - {4}):
                plans[f"vec{vec}"] = Plan(modules[vec], tensors, k, block, warps)
            for arm in list(plans):
                try:
                    validate(plans[arm], tensors, k, want)
                    emit("vec_validation", shape_id=args.shape_id, seed=seed,
                         arm=arm, case="normal", ok=True)
                except Exception as exc:
                    if arm.startswith("control"):
                        raise
                    vec = int(arm[3:])
                    viable.remove(vec)
                    errors.append([vec, f"normal-{seed}", str(exc)])
                    del plans[arm]
                    emit("vec_validation", shape_id=args.shape_id, seed=seed,
                         arm=arm, case="normal", ok=False, error=str(exc))
            if seed == args.seeds[0]:
                for arm, plan in plans.items():
                    mod = modules[int(arm[3:])] if arm.startswith("vec") else production
                    emit(
                        "vec_codegen", shape_id=args.shape_id, arm=arm,
                        **metadata(mod, plan, k, block, warps, rows),
                    )
            aa = readings(plans, "control_a", "control_b", args.rounds, args.iters, seed)
            emit("vec_summary", shape_id=args.shape_id, seed=seed,
                 arm="control_b", median=statistics.median(aa),
                 min=min(aa), max=max(aa))
            for vec in sorted(viable - {4}):
                arm = f"vec{vec}"
                ab = readings(plans, "control_a", arm, args.rounds, args.iters, seed)
                emit("vec_summary", shape_id=args.shape_id, seed=seed, arm=arm,
                     median=statistics.median(ab), min=min(ab), max=max(ab))
    emit(
        "vec_worker_summary", shape_id=args.shape_id,
        viable=sorted(viable), errors=errors,
    )
    return int(bool(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shape-id", type=int)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if min(args.rounds, args.iters, args.timeout) <= 0 or not args.seeds:
        parser.error("Positive rounds, iterations, timeout and seeds required")
    if args.worker:
        try:
            return worker(args)
        except Exception:
            traceback.print_exc()
            return 1
    occupancy("before")
    failures = []
    for shape_id in range(len(SHAPES)):
        emit("vec_worker_start", shape_id=shape_id, shape=SHAPES[shape_id])
        cmd = [
            sys.executable, "-u", __file__, "--worker", "--shape-id",
            str(shape_id), "--rounds", str(args.rounds), "--iters",
            str(args.iters), "--seeds", *map(str, args.seeds),
        ]
        try:
            code = subprocess.run(
                cmd, timeout=args.timeout,
                env=dict(os.environ, FLAGGEMS_FORCE_TLE="0"),
            ).returncode
        except subprocess.TimeoutExpired:
            code = 124
        emit("vec_worker_exit", shape_id=shape_id, code=code)
        if code:
            failures.append(shape_id)
    occupancy("after")
    emit("vec_suite_summary", failures=failures)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
