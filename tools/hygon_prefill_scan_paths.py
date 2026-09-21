"""BW1000 full-operator probes: full-row tail, then wave64 slot scan.

No production routing is changed. Each candidate is a copy of the *current*
selected module, with only the indicated source transform. Every launch uses
the same preallocated scratch and geometry as its control.
"""

import argparse
import hashlib
import importlib.util
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import traceback
from importlib import import_module
from pathlib import Path

from hygon_prefill_audit import SHAPES, check_output, emit, inputs, occupancy, oracle
from hygon_prefill_scan_paths_source import fullrow_variant, wave64_variant

ROOT = Path(__file__).resolve().parents[1]
OVERRIDE = "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
STAGES = {"fullrow": tuple(range(1, 7)), "wave64": (2, 4, 5, 6)}


def load_copy(source, name, directory):
    path = Path(directory) / f"{name.rsplit('.', 1)[-1]}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if mod.HAS_TLE:
        raise RuntimeError("Expected non-TLE Hygon module")
    return mod


class Plan:
    def __init__(self, mod, tensors, top_k, block, warps):
        import torch

        launcher = import_module(
            "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
        )._Launch
        x, starts, ends = tensors
        rows, vocab = x.shape
        self.guard = torch.empty(
            (rows * top_k + 32,), device=x.device, dtype=torch.int32
        )
        self.out = self.guard[16:-16].view(rows, top_k)
        self.scratch = (
            torch.empty((rows, 2048), device=x.device, dtype=torch.int32),
            torch.empty((rows, 2048), device=x.device, dtype=torch.float32),
            *(
                torch.empty((rows,), device=x.device, dtype=torch.int32)
                for _ in range(4)
            ),
        )
        self.args = (
            x,
            self.out,
            starts,
            ends,
            x.stride(0),
            x.stride(1),
            vocab,
            *self.scratch,
        )
        self.launch = launcher(
            mod.non_tle_top_k_per_row_prefill,
            (rows,),
            dict(TOPK=top_k, BLOCK_SIZE=block, ROW_OFFSET=0),
            warps,
        )

    def __call__(self):
        self.launch(*self.args)


def verify(plan, tensors, top_k, want):
    import torch

    plan.guard.fill_(-123456)
    for repeat in range(2):
        plan.out.fill_(-9)
        plan()
        torch.cuda.synchronize()
        if not bool(
            (plan.guard[:16] == -123456).all() & (plan.guard[-16:] == -123456).all()
        ):
            raise AssertionError("Output guard overwritten")
        check_output(plan.out, tensors, top_k, want)
    return repeat + 1


def event_us(plan, iters=8):
    import torch

    for _ in range(3):
        plan()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        plan()
    end.record()
    torch.cuda.synchronize()
    return 1000 * start.elapsed_time(end) / iters


def paired(control, candidate, stage, shape_id, seed):
    ratios = []
    for round_id in range(3):
        order = (
            ("control", "candidate", "candidate", "control")
            if round_id % 2 == 0
            else ("candidate", "control", "control", "candidate")
        )
        readings = [
            (name, event_us({"control": control, "candidate": candidate}[name]))
            for name in order
        ]
        for a, b in ((0, 1), (3, 2)):
            pair = dict(readings[i] for i in (a, b))
            ratios.append(pair["control"] / pair["candidate"])
        emit(
            "scan_paths_round",
            stage=stage,
            shape_id=shape_id,
            seed=seed,
            round=round_id,
            readings=readings,
            ratios=ratios[-2:],
        )
    emit(
        "scan_paths_summary",
        stage=stage,
        shape_id=shape_id,
        seed=seed,
        ratio_median=statistics.median(ratios),
        ratio_min=min(ratios),
        ratio_max=max(ratios),
        metric="preallocated_full_kernel_event_us",
    )


def metadata(plan, stage, shape_id, arm):
    import torch

    ck = plan.launch.jit.run(
        *plan.args,
        **plan.launch.constexprs,
        num_warps=plan.launch.num_warps,
        grid=plan.launch.grid,
        warmup=False,
    )
    torch.cuda.synchronize()
    asm = ck.asm.get("amdgcn", "")
    emit(
        "scan_paths_codegen",
        stage=stage,
        shape_id=shape_id,
        arm=arm,
        registers=getattr(ck, "n_regs", None),
        spills=getattr(ck, "n_spills", None),
        asm_sha256=hashlib.sha256(asm.encode()).hexdigest(),
        ds_bpermute=asm.count("ds_bpermute"),
        barriers=asm.count("s_barrier"),
    )


def worker(stage, shape_id):
    import torch
    import triton
    import vllm._custom_ops  # noqa: F401 - register the C++ baseline

    import flaggems_vllm

    if flaggems_vllm.vendor_name != "hygon":
        raise RuntimeError(f"Expected Hygon, got {flaggems_vllm.vendor_name}")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.warp_size != 64:
        raise RuntimeError(f"Expected HIP wave64, got {target}")
    if not hasattr(torch.ops._C, "top_k_per_row_prefill"):
        raise RuntimeError("The vLLM C++ benchmark baseline is unavailable")
    ov = import_module(OVERRIDE)
    rows, vocab, top_k, stride0 = SHAPES[shape_id]
    meta = type("Meta", (), {"shape": (rows, vocab), "dtype": torch.float32})()
    control_mod = ov._select_module(meta, rows, top_k)
    if control_mod.HAS_TLE:
        raise RuntimeError("Expected non-TLE production control")
    geo = ov._geometry(rows, vocab) if ov._GEOMETRY else None
    if geo is None:
        block = control_mod.NUM_THREADS_PER_BLOCK
        warps = control_mod._num_warps(block)
    else:
        block, warps = geo
    control_source = Path(control_mod.__file__).read_text()
    build = fullrow_variant if stage == "fullrow" else wave64_variant
    candidate_source = build(control_source)
    emit(
        "scan_paths_config",
        stage=stage,
        shape_id=shape_id,
        shape=SHAPES[shape_id],
        geometry=[block, warps],
        control_path=control_mod.__file__,
        control_sha256=hashlib.sha256(control_source.encode()).hexdigest(),
        candidate_sha256=hashlib.sha256(candidate_source.encode()).hexdigest(),
    )
    with tempfile.TemporaryDirectory(prefix="hygon_scan_paths_") as directory:
        candidate_mod = load_copy(
            candidate_source,
            f"flaggems_vllm.ops._probe_{stage}_{shape_id}",
            directory,
        )
        # Full official shape, same normal input and seed as the benchmark.
        for seed in (42, 43):
            tensors = inputs(rows, vocab, stride0, top_k, seed)
            want = oracle(tensors, top_k)
            control = Plan(control_mod, tensors, top_k, block, warps)
            candidate = Plan(candidate_mod, tensors, top_k, block, warps)
            verify(control, tensors, top_k, want)
            verify(candidate, tensors, top_k, want)
            emit(
                "scan_paths_validation",
                stage=stage,
                shape_id=shape_id,
                case="normal_full",
                seed=seed,
                ok=True,
            )
            if seed == 42:
                metadata(control, stage, shape_id, "control")
                metadata(candidate, stage, shape_id, "candidate")
            paired(control, candidate, stage, shape_id, seed)
            del tensors, want, control, candidate
        # A reduced batch still exercises full-row tails and the unchanged
        # partial/short/special paths without allocating a second huge oracle.
        for case in ("tied", "constant", "partial", "short", "special"):
            small_rows = 5 if case == "short" else 8
            tensors = inputs(small_rows, vocab, stride0, top_k, 123, case)
            want = oracle(tensors, top_k)
            plan = Plan(candidate_mod, tensors, top_k, block, warps)
            verify(plan, tensors, top_k, want)
            emit(
                "scan_paths_validation",
                stage=stage,
                shape_id=shape_id,
                case=case,
                seed=123,
                ok=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--stage", choices=("all", "fullrow", "wave64"), default="all")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]))
        return 0
    emit(
        "scan_paths_probe",
        host=platform.node(),
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    occupancy("before")
    failures = []
    stages = STAGES if args.stage == "all" else {args.stage: STAGES[args.stage]}
    for stage, shape_ids in stages.items():
        emit("scan_paths_stage_start", stage=stage, shape_ids=shape_ids)
        for shape_id in shape_ids:
            env = dict(
                os.environ,
                FLAGGEMS_FORCE_TLE="0",
                FLAGGEMS_HYGON_TOPK_FINAL_NETWORK="1",
            )
            command = [sys.executable, "-u", __file__, "--worker", stage, str(shape_id)]
            try:
                code = subprocess.run(
                    command, cwd=ROOT, env=env, timeout=1800
                ).returncode
            except subprocess.TimeoutExpired:
                code = 124
            emit("scan_paths_worker_exit", stage=stage, shape_id=shape_id, code=code)
            if code:
                failures.append([stage, shape_id, code])
        emit(
            "scan_paths_stage_complete",
            stage=stage,
            failures=[f for f in failures if f[0] == stage],
        )
    occupancy("after")
    emit("scan_paths_complete", failures=failures)
    return int(bool(failures))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
