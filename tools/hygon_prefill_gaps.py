"""Remaining BW1000 probes. See hygon_prefill_gaps_trial.md; no production edits."""

import argparse
import importlib.util
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from importlib import import_module
from pathlib import Path

from hygon_prefill_audit import (
    SHAPES,
    Plan,
    check_output,
    emit,
    inputs,
    occupancy,
    oracle,
    stats,
    validate,
)
from hygon_prefill_audit_source import diagnostic_source, digest
from hygon_prefill_gaps_source import variant
from hygon_prefill_next import check_cases, readings

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("preflight", "radix", "counters", "scratch", "compression", "geometry", "tle")
GEOMETRIES = ((256, 2), (256, 4), (512, 4), (512, 8), (1024, 8), (1024, 16))
CROSSOVER = tuple(
    (r, v, 1024 if v == 129280 else 512, v + 8)
    for v in (4096, 16385, 129280)
    for r in (4, 64, 160, 320, 640, 1280, 2560)
)


def module(folder, dense, arm, diagnostic=False):
    source = variant(ROOT, dense, arm)
    if diagnostic:
        source = diagnostic_source(source)
    name = f"flaggems_vllm.ops._gaps_{arm}_{int(dense)}_{int(diagnostic)}"
    path = Path(folder) / (name.rsplit(".", 1)[-1] + ".py")
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if mod.HAS_TLE:
        raise RuntimeError("Non-TLE worker unexpectedly enabled TLE")
    emit("source", arm=arm, diagnostic=diagnostic, sha256=digest(source))
    return mod


class GapPlan(Plan):
    def __init__(self, mod, tensors, top_k, block, warps, radix=False):
        super().__init__(mod, tensors, top_k, block, warps)
        if radix:
            import torch

            # The first 2048 entries hold candidate indices during final select.
            # Keep the 256 radix counters separate, with explicit per-row stride.
            self.hist = torch.empty(
                (tensors[0].shape[0], 2304), device=tensors[0].device, dtype=torch.int32
            )
            self.args = (*self.args[:7], self.hist, *self.args[8:])

    def fresh(self):
        import torch

        # Same launcher and output; only workspace lifetime differs.
        scratch = tuple(torch.empty_like(t) for t in self.args[7:])
        self.launch(*self.args[:7], *scratch)


class FreshValidationPlan(GapPlan):
    def __call__(self):
        self.fresh()


def baseline():
    import torch
    import vllm
    import vllm._custom_ops  # noqa: F401

    name = "_C::top_k_per_row_prefill"
    if not hasattr(torch.ops._C, "top_k_per_row_prefill"):
        raise RuntimeError("Compiled vLLM baseline absent after custom-op import")
    table = torch._C._dispatch_dump_table(name)
    if not torch._C._dispatch_has_kernel_for_dispatch_key(name, "CUDA"):
        raise RuntimeError("No CUDA registration for vLLM baseline")
    emit(
        "baseline",
        version=vllm.__version__,
        schema=str(torch.ops._C.top_k_per_row_prefill.default._schema),
        dispatch=table,
    )
    tensors = inputs(5, 4095, 4104, 512, 42, "partial")
    x, starts, ends = tensors
    out = torch.empty((5, 512), device=x.device, dtype=torch.int32)
    torch.ops._C.top_k_per_row_prefill(x, starts, ends, out, 5, x.stride(0), 1, 512)
    torch.cuda.synchronize()
    check_output(out, tensors, 512, oracle(tensors, 512))
    emit("baseline_validation", ok=True)


def wall_time(fn, iters):
    import torch

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - start) / (iters * 1000)


def radix_cases(factories, shape):
    import torch

    _, vocab, k, stride0 = shape
    # Deliberately fill the final buffer; normal logits often leave <64 items.
    # Also straddle both dispatch gates with exactly one remaining output slot.
    for count in (1, 63, 64, 65, 255, 256, 257, min(2048, vocab)):
        tensors = inputs(5, vocab, max(stride0, vocab + 8), k, 123)
        x, starts, ends = tensors
        prefix = k - 1 if count + k - 1 <= vocab else 0
        # Include a losing value where possible so count=1 does not early-return.
        ends.fill_(min(vocab, prefix + count + 1))
        x.fill_(-100.0)
        x[:, :prefix] = 100.0
        lane = torch.arange(count, device=x.device, dtype=torch.float32)
        x[:, prefix : prefix + count] = 1.0 + lane / (max(count, 1) * 65536.0)
        want = oracle(tensors, k)
        for arm, make in factories.items():
            plan = make(tensors)
            validate(plan, tensors, k, want)
            if not bool((plan.count == count).all()):
                raise AssertionError(
                    "Radix stress did not reach the intended final count"
                )
            emit(
                "validation",
                stage="radix",
                case="final_count_boundary",
                arm=arm,
                count=count,
                remaining=k - prefix,
                ok=True,
            )


def compression(plan, seed):
    import torch

    plan()
    torch.cuda.synchronize()
    stats(plan, "full_normal", seed)
    counts = plan.count.clamp(0, 2048)
    valid = torch.arange(2048, device=counts.device)[None, :] < counts[:, None]
    # Preserve all fp32 bits, including signed zero, for lossless diagnostics.
    bits = plan.values.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    keys = torch.where((bits & 0x80000000) != 0, bits, ~bits & 0x7FFFFFFF)
    high = keys >> 16
    lo = torch.where(valid, high, 65536).min(dim=1).values
    hi = torch.where(valid, high, -1).max(dim=1).values
    eligible = (counts > 0) & (lo == hi)
    rebuilt = (keys & 65535) | (lo[:, None] << 16)
    if not bool(
        torch.all(torch.where(valid & eligible[:, None], rebuilt == keys, True))
    ):
        raise AssertionError("Lossless key reconstruction failed")
    emit(
        "compression_eligibility",
        seed=seed,
        rows=int(counts.numel()),
        nonempty_rows=int((counts > 0).sum()),
        common_high16_rows=int(eligible.sum()),
        final_items=int(counts.sum()),
        allocated_value_bytes=plan.values.numel() * 4,
        live_value_bytes=int(counts.sum()) * 4,
        note="Untimed diagnostic; capacity is not measured DRAM traffic; no compressed kernel",
    )


def tle_worker(args, shape):
    import torch
    from torch.profiler import ProfilerActivity, profile

    generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
    if not generic.HAS_TLE:
        raise RuntimeError("Forced TLE unavailable")
    baseline()
    rows, vocab, k, stride0 = shape
    for case in (
        "tied",
        "constant",
        "partial",
        "short",
        "special",
        "strided",
    ):
        # Include the first radix-tail row when the host splits at 12288.
        n = (
            generic.SORTING_ALGORITHM_THRESHOLD + 1
            if rows > generic.SORTING_ALGORITHM_THRESHOLD
            else max(5, min(rows, 32))
        )
        tensors = inputs(n, vocab, max(stride0, vocab + 8), k, 42, case)
        x, starts, ends = tensors
        guard = torch.full((n * k + 32,), -123456, device=x.device, dtype=torch.int32)
        out = guard[16:-16].view(n, k)
        generic.top_k_per_row_prefill(x, starts, ends, out, n, x.stride(0), 1, k)
        torch.cuda.synchronize()
        check_output(out, tensors, k, oracle(tensors, k))
        assert bool((guard[:16] == -123456).all() & (guard[-16:] == -123456).all())
        emit("validation", stage="tle", case=case, ok=True)

    def run():
        generic.top_k_per_row_prefill(x, starts, ends, out, rows, x.stride(0), 1, k)

    def reference():
        torch.ops._C.top_k_per_row_prefill(
            x, starts, ends, out, rows, x.stride(0), 1, k
        )

    for seed in args.seeds:
        tensors = inputs(rows, vocab, stride0, k, seed)
        x, starts, ends = tensors
        guard = torch.full(
            (rows * k + 32,), -123456, device=x.device, dtype=torch.int32
        )
        out = guard[16:-16].view(rows, k)
        want = oracle(tensors, k)
        for label, fn in (("tle", run), ("vllm", reference)):
            out.fill_(-9)
            fn()
            torch.cuda.synchronize()
            check_output(out, tensors, k, want)
            assert bool((guard[:16] == -123456).all() & (guard[-16:] == -123456).all())
            emit(
                "validation",
                stage="tle",
                arm=label,
                case="full_normal",
                seed=seed,
                ok=True,
            )
        for round_id in range(args.rounds):
            order = (
                ("tle", run),
                ("vllm", reference),
                ("vllm", reference),
                ("tle", run),
            )
            if round_id % 2:
                order = (
                    ("vllm", reference),
                    ("tle", run),
                    ("tle", run),
                    ("vllm", reference),
                )
            samples = []
            for label, fn in order:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                ) as prof:
                    for _ in range(args.iters):
                        fn()
                    torch.cuda.synchronize()
                events = [
                    e
                    for e in prof.events()
                    if getattr(e.device_type, "name", "") == "CUDA"
                ]
                if not events or len(events) % args.iters:
                    raise RuntimeError("Invalid GPU event count for TLE comparison")
                us = sum(e.time_range.elapsed_us() for e in events) / args.iters
                samples.append((label, us))
                emit(
                    "tle_timing",
                    arm=label,
                    seed=seed,
                    round=round_id,
                    device_us_per_call=us,
                    events=len(events),
                    names=sorted({e.name for e in events}),
                )
            pairs = [dict(samples[p] for p in pair) for pair in ((0, 1), (3, 2))]
            emit(
                "tle_paired",
                seed=seed,
                round=round_id,
                ratios=[pair["vllm"] / pair["tle"] for pair in pairs],
            )


def worker(args):
    import torch
    import triton

    import flaggems_vllm

    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    target = triton.runtime.driver.active.get_current_target()
    emit(
        "device",
        props=str(props),
        torch=torch.__version__,
        triton=triton.__version__,
        warp_size=target.warp_size,
        backend=target.backend,
        stage=args.stage,
        arm=args.arm,
    )
    if flaggems_vllm.vendor_name != "hygon":
        raise RuntimeError("Hygon-only probe")
    if target.warp_size != 64:
        raise RuntimeError("Expected the measured warp64 BW1000 target")
    if args.stage == "preflight":
        baseline()
        return
    shape = (
        CROSSOVER[args.shape_id] if args.stage == "geometry" else SHAPES[args.shape_id]
    )
    emit("shape", shape=shape, shape_id=args.shape_id)
    if args.stage == "tle":
        tle_worker(args, shape)
        return
    rows, vocab, k, stride0 = shape
    dense = vocab <= ov.DENSE_VOCAB_PER_TOPK * k
    block, warps = ov._geometry(rows, vocab) or (512, 8)
    shipped = ov._dense_carry if dense else ov._sparse
    if shipped is None or shipped.HAS_TLE:
        raise RuntimeError("Expected production carry/sparse non-TLE route")
    if Path(shipped.__file__).read_text() != variant(ROOT, dense, "control"):
        raise RuntimeError("Generated control differs from production")
    with tempfile.TemporaryDirectory(prefix="hygon_gaps_") as folder:
        control = module(folder, dense, "control", args.stage == "compression")
        arm = args.arm if args.stage in ("radix", "counters") else "control"
        candidate = module(folder, dense, arm) if arm != "control" else control
        cb, cw = (
            GEOMETRIES[args.config_id] if args.stage == "geometry" else (block, warps)
        )
        factories = {
            "control": lambda ts: GapPlan(control, ts, k, block, warps),
            "candidate": lambda ts: GapPlan(
                candidate, ts, k, cb, cw, arm.startswith("radix")
            ),
        }
        emit("config", dense=dense, control=[block, warps], candidate=[cb, cw], arm=arm)
        check_cases(factories, shape, args.stage)
        if args.stage == "radix":
            radix_cases(factories, shape)
        if args.stage == "scratch":
            for case in ("tied", "constant", "partial", "short", "special", "strided"):
                tensors = inputs(
                    max(5, min(rows, 32)), vocab, max(stride0, vocab + 8), k, 123, case
                )
                plan = FreshValidationPlan(control, tensors, k, block, warps)
                validate(plan, tensors, k, oracle(tensors, k))
                emit("validation", stage="scratch", case=case, arm="fresh", ok=True)
        ratios = []
        for seed in args.seeds:
            tensors = inputs(rows, vocab, stride0, k, seed)
            want = oracle(tensors, k)
            plans = {key: make(tensors) for key, make in factories.items()}
            for key, plan in plans.items():
                validate(plan, tensors, k, want)
                emit("validation", case="full_normal", seed=seed, arm=key, ok=True)
            if args.stage == "compression":
                compression(plans["control"], seed)
            elif args.stage == "scratch":
                plan = plans["control"]
                plan.fresh()
                torch.cuda.synchronize()
                check_output(plan.out, tensors, k, want)
                variants = {"cached": plan, "fresh": plan.fresh}
                ratios.extend(
                    readings(variants, "fresh", "cached", args.rounds, args.iters, seed)
                )
                for round_id in range(args.rounds):
                    order = ("fresh", "cached", "cached", "fresh")
                    if round_id % 2:
                        order = order[::-1]
                    emit(
                        "scratch_wall",
                        seed=seed,
                        round=round_id,
                        samples=[
                            (key, wall_time(variants[key], args.iters)) for key in order
                        ],
                        note="synchronized batch host end-to-end us/call, separate from device time",
                    )
            else:
                ratios.extend(
                    readings(
                        plans, "control", "candidate", args.rounds, args.iters, seed
                    )
                )
            del plans, tensors, want
        emit(
            "worker_summary",
            stage=args.stage,
            arm=args.arm,
            shape=shape,
            ratios=ratios,
            median=statistics.median(ratios) if ratios else None,
        )


def jobs(args):
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "preflight":
            yield stage, 0, "control", 0
            continue
        ids = args.shape_ids
        if ids is None:
            ids = range(len(CROSSOVER) if stage == "geometry" else len(SHAPES))
        for sid in ids:
            if stage == "geometry":
                if not 0 <= sid < len(CROSSOVER):
                    raise ValueError("Invalid geometry shape id")
                for config_id in range(len(GEOMETRIES)):
                    yield stage, sid, "control", config_id
            else:
                if not 0 <= sid < len(SHAPES):
                    raise ValueError("Invalid benchmark shape id")
                dense = SHAPES[sid][1] <= 10 * SHAPES[sid][2]
                arms = ("control",)
                if stage == "radix":
                    arms = ("radix0", "radix64", "radix256")
                elif stage == "counters":
                    arms = (
                        ("final_scan",)
                        if dense
                        else ("found_scan", "final_scan", "both_scan")
                    )
                for arm in arms:
                    yield stage, sid, arm, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", *STAGES))
    parser.add_argument("--shape-ids", nargs="+", type=int)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shape-id", type=int, default=0)
    parser.add_argument("--config-id", type=int, default=0)
    parser.add_argument("--arm", default="control")
    args = parser.parse_args()
    if min(args.rounds, args.iters, args.timeout) <= 0:
        parser.error("rounds, iters and timeout must be positive")
    if args.worker:
        try:
            worker(args)
            return 0
        except Exception:
            traceback.print_exc()
            return 1
    targets = list(jobs(args))
    if args.stage not in ("all", "preflight"):
        targets.insert(0, ("preflight", 0, "control", 0))
    occupancy("before")
    failures = []
    for stage, sid, arm, config_id in targets:
        cmd = [
            sys.executable,
            "-u",
            __file__,
            stage,
            "--worker",
            "--shape-id",
            str(sid),
            "--arm",
            arm,
            "--config-id",
            str(config_id),
            "--rounds",
            str(args.rounds),
            "--iters",
            str(args.iters),
            "--seeds",
            *map(str, args.seeds),
        ]
        env = dict(os.environ, FLAGGEMS_FORCE_TLE="1" if stage == "tle" else "0")
        emit("worker_start", stage=stage, shape_id=sid, arm=arm, config_id=config_id)
        try:
            code = subprocess.run(cmd, env=env, timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            code = 124
        emit(
            "worker_exit",
            stage=stage,
            shape_id=sid,
            arm=arm,
            config_id=config_id,
            code=code,
        )
        if code:
            failures.append([stage, sid, arm, config_id, code])
            if stage == "preflight":
                break
    occupancy("after")
    emit("suite_summary", failures=failures, requested_workers=len(targets))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
