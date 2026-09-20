"""Compile-only register/LDS budget sweep for the 16-elements/lane candidates.

The lane probe measures end-to-end time.  This companion records the compiler
resource footprint for the same candidates: registers, spills, shared bytes,
barriers, and target-assembly size.  It is intentionally one child process per
shape/config, because a bad resource combination can fault the Hygon process.
"""

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import traceback
from importlib import import_module
from pathlib import Path

import torch

from hygon_prefill_audit import SHAPES, Plan, emit, inputs, occupancy
from hygon_prefill_audit_source import digest
from hygon_prefill_vec_source import variants

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    (4, 512, 2),
    (4, 1024, 4),
    (8, 256, 2),
    (8, 512, 4),
    (2, 1024, 2),
)
SHAPE_IDS = (0, 1, 2, 3, 6)


def load_variant(folder, source, dense, vec):
    name = f"flaggems_vllm.ops._hygon_budget_{int(dense)}_{vec}"
    path = Path(folder) / f"_hygon_budget_{int(dense)}_{vec}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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
    source = variants(ROOT, dense)[vec]
    base_block, base_warps = ov._geometry(rows, vocab) or (512, 8)
    emit("budget_config", shape_id=args.shape_id, shape=SHAPES[args.shape_id],
         config_id=args.config_id, config=[vec, block, warps],
         baseline=[base_block, base_warps],
         elements_per_lane=block * vec / (warps * target.warp_size),
         source_sha256=digest(source))

    with tempfile.TemporaryDirectory(prefix="hygon_budget_") as folder:
        mod = load_variant(folder, source, dense, vec)
        tensors = inputs(rows, vocab, stride0, top_k, 42)
        plan = Plan(mod, tensors, top_k, block, warps)
        compiled = mod.non_tle_top_k_per_row_prefill.run(
            *plan.args, TOPK=top_k, BLOCK_SIZE=block, ROW_OFFSET=0,
            num_warps=warps, grid=(rows,), warmup=False,
        )
        torch.cuda.synchronize()
        if compiled is None:
            raise RuntimeError("JIT returned no compiled kernel")
        metadata = getattr(compiled, "metadata", None)
        asm_map = getattr(compiled, "asm", {})
        asm = asm_map.get("amdgcn")
        barriers = None
        cvt = None
        if isinstance(asm, str):
            lines = [line.strip() for line in asm.splitlines()]
            barriers = sum(line.startswith("s_barrier") for line in lines)
            cvt = sum("v_cvt_f16_f32" in line for line in lines)
        regs = getattr(compiled, "n_regs", None)
        spills = getattr(compiled, "n_spills", None)
        shared = getattr(metadata, "shared", None)
        lanes = warps * target.warp_size
        emit(
            "budget_codegen",
            shape_id=args.shape_id,
            config_id=args.config_id,
            shape=SHAPES[args.shape_id],
            config=[vec, block, warps],
            registers=regs,
            spills=spills,
            shared_bytes=shared,
            wave_lanes=lanes,
            register_bytes_per_program=(regs * lanes * 4 if regs is not None else None),
            barriers=barriers,
            cvt_f16_f32=cvt,
            target_bytes=(len(asm.encode()) if isinstance(asm, str) else None),
            target_sha256=(hashlib.sha256(asm.encode()).hexdigest() if isinstance(asm, str) else None),
        )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shape-id", type=int)
    ap.add_argument("--config-id", type=int)
    ap.add_argument("--timeout", type=int, default=1200)
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
        for shape_id in SHAPE_IDS:
            emit("budget_worker_start", config_id=config_id, config=config,
                 shape_id=shape_id, shape=SHAPES[shape_id])
            cmd = [sys.executable, "-u", __file__, "--worker",
                   "--shape-id", str(shape_id), "--config-id", str(config_id)]
            try:
                code = subprocess.run(
                    cmd, timeout=args.timeout,
                    env=dict(os.environ, FLAGGEMS_FORCE_TLE="0"),
                ).returncode
            except subprocess.TimeoutExpired:
                code = 124
            emit("budget_worker_exit", config_id=config_id, shape_id=shape_id,
                 code=code)
            if code:
                failures.append([config_id, shape_id])
    occupancy("after")
    emit("budget_suite_summary", configs=CONFIGS, shape_ids=SHAPE_IDS,
         failures=failures)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
