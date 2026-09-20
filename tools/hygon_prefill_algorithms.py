"""Four exact algorithm experiments; see hygon_prefill_algorithms.md.

Parent and --check are stdlib-only. GPU imports happen only inside isolated
workers. Baseline is the public prefill API, including its current routing.
"""

import argparse
import gc
import hashlib
import importlib.util
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from importlib import import_module
from pathlib import Path

from hygon_prefill_algorithms_source import final_variant
from hygon_prefill_audit import SHAPES, check_output, emit, inputs, occupancy, oracle

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("threshold", "streaming", "final", "delegate")
CASES = (
    "tied",
    "constant",
    "partial",
    "short",
    "special",
    "strided",
    "ascending",
    "descending",
    "heavy_tail",
    "clustered",
)


def tasks(family):
    if family == "threshold":
        return [
            (s, (kind, warps))
            for s in (6, 2, 4, 5)
            for kind in ("binary", "quaternary")
            for warps in (4, 8)
        ]
    if family == "streaming":
        return [(s, (kind, 8)) for s in (0, 1, 3) for kind in ("filter", "always")]
    if family == "final":
        return [
            (s, (kind, 0)) for s in range(len(SHAPES)) for kind in ("network", "prefix")
        ]
    if family == "delegate":
        return [(0, (block, 8)) for block in (32, 64)]
    raise ValueError(family)


def actual_module(ov, rows, vocab, k):
    # Same gates as the current public dispatcher, including short bins/VEC2.
    if ov._ENABLED and vocab <= ov.DENSE_VOCAB_PER_TOPK * k:
        if (
            ov._dense_short_bins is not None
            and k == ov.SHORT_BINS_TOPK
            and vocab <= ov.SHORT_BINS_MAX_VOCAB
        ):
            mod = ov._dense_short_bins
        elif ov._dense_vec2 is not None:
            mod = ov._dense_vec2
        else:
            mod = ov._dense_carry if ov._dense_carry is not None else ov._dense
    else:
        mod = ov._sparse
    geo = ov._geometry(rows, vocab) if ov._GEOMETRY else None
    default_block, default_warps = ov._GENERIC_DEFAULTS[id(mod)]
    return mod, geo or (default_block, default_warps(default_block))


class PublicPlan:
    kernel_count = 1

    def __init__(self, tensors, k):
        import torch

        import flaggems_vllm

        self.data = tensors
        self.k = k
        self.fn = flaggems_vllm.top_k_per_row_prefill
        x = tensors[0]
        self.guard = torch.empty(
            (x.shape[0] * k + 32,), device=x.device, dtype=torch.int32
        )
        self.out = self.guard[16:-16].view(x.shape[0], k)

    def __call__(self):
        x, starts, ends = self.data
        self.fn(x, starts, ends, self.out, x.shape[0], x.stride(0), 1, self.k)


class CandidatePlan(PublicPlan):
    def __init__(self, tensors, k, family, config, mod=None, geo=None, diag=False):
        import hygon_prefill_algorithms_kernel as kernels
        import torch
        import triton

        super().__init__(tensors, k)
        launcher = import_module(
            "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
        )._Launch
        self.calls = []
        self.family = family
        self.config = config
        self.diag = diag
        x, starts, ends = tensors
        rows, vocab = x.shape
        self.stats = torch.empty((rows,), device=x.device, dtype=torch.int32)

        def add(kernel, grid, args, meta, warps):
            self.calls.append((launcher(kernel, grid, meta, warps), args))

        if family == "threshold":
            kind, warps = config
            add(
                kernels.threshold_select,
                (rows,),
                (x, starts, ends, self.out, self.stats, x.stride(0)),
                dict(
                    K=k,
                    B=triton.next_power_of_2(vocab),
                    QUATERNARY=kind == "quaternary",
                    DIAG=diag,
                ),
                warps,
            )
        elif family in ("streaming", "delegate"):
            kind, warps = config
            block = kind if family == "delegate" else 64
            groups = triton.cdiv(vocab, block)
            self.maxima = torch.empty(
                (rows, groups) if family == "delegate" else (1,),
                device=x.device,
                dtype=torch.uint32,
            )
            self.bounds = torch.empty((rows,), device=x.device, dtype=torch.uint32)
            self.delegate_stats = torch.empty(
                (rows, 3), device=x.device, dtype=torch.int32
            )
            if family == "delegate":
                add(
                    kernels.delegate_max,
                    (rows, groups),
                    (x, starts, ends, self.maxima, x.stride(0)),
                    dict(GROUPS=groups, BLOCK=block),
                    4,
                )
                add(
                    kernels.delegate_bound,
                    (rows,),
                    (self.maxima, starts, ends, self.bounds, self.delegate_stats),
                    dict(
                        K=k,
                        GROUPS=groups,
                        BLOCK=block,
                        B=triton.next_power_of_2(groups),
                        DIAG=diag,
                    ),
                    4,
                )
            add(
                kernels.streaming_select,
                (rows,),
                (
                    x,
                    starts,
                    ends,
                    self.out,
                    self.maxima,
                    self.bounds,
                    self.stats,
                    x.stride(0),
                ),
                dict(
                    K=k,
                    FILTER=kind != "always",
                    DELEGATE=family == "delegate",
                    GROUPS=groups,
                    BLOCK=block,
                    DIAG=diag,
                ),
                warps,
            )
        else:
            block, warps = geo
            self.hist = torch.empty((rows, 2048), device=x.device, dtype=torch.int32)
            self.values = torch.empty(
                (rows, 2048), device=x.device, dtype=torch.float32
            )
            self.counters = [
                torch.empty((rows,), device=x.device, dtype=torch.int32)
                for _ in range(4)
            ]
            add(
                mod.non_tle_top_k_per_row_prefill,
                (rows,),
                (
                    x,
                    self.out,
                    starts,
                    ends,
                    x.stride(0),
                    1,
                    vocab,
                    self.hist,
                    self.values,
                    *self.counters,
                ),
                dict(TOPK=k, BLOCK_SIZE=block, ROW_OFFSET=0),
                warps,
            )
        self.kernel_count = len(self.calls)

    def __call__(self):
        for launch, args in self.calls:
            launch(*args)

    def metadata(self):
        import torch

        for launch, args in self.calls:
            ck = launch.jit.run(
                *args,
                **launch.constexprs,
                num_warps=launch.num_warps,
                grid=launch.grid,
                warmup=False,
            )
            torch.cuda.synchronize()
            asm = ck.asm.get("amdgcn", "")
            emit(
                "algorithm_codegen",
                kernel=ck.name,
                warps=launch.num_warps,
                registers=getattr(ck, "n_regs", None),
                spills=getattr(ck, "n_spills", None),
                shared_bytes=getattr(ck.metadata, "shared", None),
                asm_sha256=hashlib.sha256(asm.encode()).hexdigest(),
            )


def make_inputs(rows, vocab, stride, k, seed, case):
    import torch

    data = inputs(rows, vocab, stride, k, seed, case)
    x = data[0]
    if case in ("ascending", "descending"):
        values = torch.arange(vocab, device=x.device, dtype=x.dtype)
        x.copy_(values if case == "ascending" else -values)
    elif case == "heavy_tail":
        # Finite Cauchy-like values; deliberately unlike the normal benchmark.
        x.copy_(torch.tan(torch.empty_like(x).uniform_(-1.56, 1.56)))
    elif case == "clustered":
        x.fill_(1.0)
        x[:, ::31] = 1.000001
        x[:, 1::37] = -1.0
    return data


def verify(plan, tensors, k, want):
    import torch

    plan.guard.fill_(-123456)
    for _ in range(2):
        plan.out.fill_(-9)
        plan()
        torch.cuda.synchronize()
        if not bool(
            (plan.guard[:16] == -123456).all() & (plan.guard[-16:] == -123456).all()
        ):
            raise AssertionError("output guard overwritten")
        check_output(plan.out, tensors, k, want)


def kernel_time(plan, iters):
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        plan()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            plan()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if getattr(e.device_type, "name", "") == "CUDA"]
    if len(events) != iters * plan.kernel_count:
        raise RuntimeError(
            f"Unexpected kernel events: {len(events)}; expected {iters * plan.kernel_count}"
        )
    events.sort(key=lambda e: e.time_range.start)
    times = [e.time_range.elapsed_us() for e in events]
    if not all(math.isfinite(t) and t > 0 for t in times):
        raise RuntimeError(f"Invalid device times: {times}")
    n = plan.kernel_count
    totals = [sum(times[i : i + n]) for i in range(0, len(times), n)]
    return dict(
        us=statistics.median(totals),
        samples_us=totals,
        stage_us=[statistics.median(times[j::n]) for j in range(n)],
    )


def wall_time(make, iters):
    import torch

    # Caller output allocation is charged to both arms. Candidate workspace and
    # launcher construction are included; the public API keeps its own cache.
    for _ in range(2):
        p = make()
        p()
        torch.cuda.synchronize()
        del p
    start = time.perf_counter_ns()
    for _ in range(iters):
        p = make()
        p()
        torch.cuda.synchronize()
        del p
    return (time.perf_counter_ns() - start) / (iters * 1000)


def compare(control, candidate, factories, args, label, seed, allocation=True):
    plans = dict(control=control, candidate=candidate)
    ratios = []
    for round_id in range(args.rounds):
        order = (
            ("control", "candidate", "candidate", "control")
            if round_id % 2 == 0
            else ("candidate", "control", "control", "candidate")
        )
        readings = [(name, kernel_time(plans[name], args.iters)) for name in order]
        for a, b in ((0, 1), (3, 2)):
            pair = dict((readings[i][0], readings[i][1]["us"]) for i in (a, b))
            ratios.append(pair["control"] / pair["candidate"])
        emit(
            "algorithm_timing",
            case=label,
            seed=seed,
            round=round_id,
            readings=readings,
            ratios=ratios[-2:],
            metric="sum_device_kernel_us",
        )
    emit(
        "algorithm_summary",
        case=label,
        seed=seed,
        ratio_median=statistics.median(ratios),
        ratio_min=min(ratios),
        ratio_max=max(ratios),
        metric="sum_device_kernel_us",
    )
    if allocation:
        readings = [
            (name, wall_time(factories[name], args.iters))
            for name in ("control", "candidate", "candidate", "control")
        ]
        c = statistics.mean(value for name, value in readings if name == "control")
        t = statistics.mean(value for name, value in readings if name == "candidate")
        emit(
            "algorithm_wall",
            case=label,
            seed=seed,
            control_us=c,
            candidate_us=t,
            ratio=c / t,
            metric="allocation_inclusive_synchronized_wall_us",
            readings=readings,
        )


def tail_checks(mode, warps):
    """Exercise count/remain edges even if the full benchmark rarely reaches them."""
    import hygon_prefill_algorithms_kernel as kernels
    import torch

    fn = kernels.final_network if mode == "network" else kernels.final_prefix
    counts = (0, 1, 63, 64, 65, 127, 128, 129, 255, 256)
    if mode == "prefix":
        counts += (257, 1024, 2048)
    for count in counts:
        cap = max(64, 1 << (max(1, count) - 1).bit_length())
        vals = torch.randn(cap, device="cuda")
        vals = (vals * 3).round() / 3
        ids = torch.arange(cap, device="cuda", dtype=torch.int32)
        if count >= 4:
            vals[:4] = torch.tensor(
                [float("inf"), -float("inf"), -0.0, 0.0], device="cuda"
            )
        base = 3
        guard = torch.empty(cap + base + 32, device="cuda", dtype=torch.int32)
        for remain in sorted({0, min(1, count), count // 2, max(0, count - 1), count}):
            guard.fill_(-123456)
            fn[(1,)](
                vals, ids, guard[16:], count, base, remain, CAP=cap, num_warps=warps
            )
            torch.cuda.synchronize()
            out = guard[16 + base : 16 + base + remain]
            if not bool(
                (guard[: 16 + base] == -123456).all()
                & (guard[16 + base + remain :] == -123456).all()
            ):
                raise AssertionError("tail guard overwrite")
            if remain:
                if (
                    not bool(((out >= 0) & (out < count)).all())
                    or out.unique().numel() != remain
                ):
                    raise AssertionError("tail indices invalid")
                want = vals[:count].topk(remain).values.sort().values
                if not torch.equal(vals[out.long()].sort().values, want):
                    raise AssertionError("tail value mismatch")
                host_values = vals[:count].cpu().tolist()
                expected = sorted(
                    range(count), key=lambda i: (host_values[i], i), reverse=True
                )[:remain]
                if sorted(out.cpu().tolist()) != sorted(expected):
                    raise AssertionError("tail later-position tie rule mismatch")
        emit("tail_validation", mode=mode, count=count, ok=True)


def worker(args):
    import torch
    import triton

    from flaggems_vllm import vendor_name

    if vendor_name != "hygon":
        raise RuntimeError(f"Expected Hygon, got {vendor_name}")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.warp_size != 64:
        raise RuntimeError(f"Expected HIP wave64, got {target}")
    shape_id, config = tasks(args.family)[args.worker]
    rows, vocab, k, stride = SHAPES[shape_id]
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    production, geo = actual_module(ov, rows, vocab, k)
    if production.HAS_TLE:
        raise RuntimeError("Expected non-TLE production control")
    emit(
        "algorithm_config",
        family=args.family,
        shape=SHAPES[shape_id],
        shape_id=shape_id,
        config=config,
        public_module=production.__file__,
        geometry=geo,
        baseline_sha256=hashlib.sha256(
            Path(production.__file__).read_bytes()
        ).hexdigest(),
        device=str(torch.cuda.get_device_properties(0)),
        torch=torch.__version__,
        triton=triton.__version__,
        scratch_reuse=ov._scratch_reuse_enabled(),
    )

    with tempfile.TemporaryDirectory(prefix="hygon_algorithm_") as folder:
        mod = None
        if args.family == "final":
            source = final_variant(Path(production.__file__).read_text(), config[0])
            name = f"flaggems_vllm.ops._probe_final_{config[0]}"
            path = Path(folder) / "final_variant.py"
            path.write_text(source)
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            emit("algorithm_source", sha256=hashlib.sha256(source.encode()).hexdigest())
            tail_checks(config[0], geo[1])

        def make(ts, diag=False):
            return CandidatePlan(ts, k, args.family, config, mod, geo, diag)

        # All correctness checks must pass before normal-shape performance.
        n = 5 if rows == 4 else min(rows, 8)
        for case in CASES:
            ts = make_inputs(n, vocab, max(stride, vocab + 8), k, 123, case)
            want = oracle(ts, k)
            control, candidate = PublicPlan(ts, k), make(ts)
            verify(control, ts, k, want)
            verify(candidate, ts, k, want)
            emit("algorithm_validation", case=case, rows=n, ok=True)
            if case in ("constant", "ascending", "heavy_tail"):
                compare(
                    control,
                    candidate,
                    None,
                    args,
                    "validation_size_" + case,
                    123,
                    allocation=False,
                )
            del control, candidate, want, ts
        for seed in args.seeds:
            ts = make_inputs(rows, vocab, stride, k, seed, "normal")
            want = oracle(ts, k)
            control, candidate = PublicPlan(ts, k), make(ts)
            verify(control, ts, k, want)
            verify(candidate, ts, k, want)
            emit(
                "algorithm_validation",
                case="normal_full",
                seed=seed,
                rows=rows,
                ok=True,
            )
            if seed == args.seeds[0]:
                candidate.metadata()
            factories = dict(
                control=lambda data=ts: PublicPlan(data, k),
                candidate=lambda data=ts: make(data),
            )
            compare(control, candidate, factories, args, "normal_full", seed)
            if args.family == "final":
                counts = candidate.counters[0].cpu().tolist()
                bases = candidate.counters[3].cpu().tolist()
                emit(
                    "final_diagnostic",
                    seed=seed,
                    count_min=min(counts),
                    count_median=statistics.median(counts),
                    count_max=max(counts),
                    rows_within_capacity={
                        str(cap): sum(c <= cap for c in counts)
                        for cap in (64, 128, 256)
                    },
                    rows_remaining_one=sum(
                        min(k - b, c) == 1 for c, b in zip(counts, bases)
                    ),
                    rows_remaining_all=sum(
                        c > 0 and k - b >= c for c, b in zip(counts, bases)
                    ),
                )
            if args.family != "final":
                diagnostic = make(ts, diag=True)
                verify(diagnostic, ts, k, want)
                values = diagnostic.stats.cpu().tolist()
                emit(
                    "algorithm_diagnostic",
                    seed=seed,
                    min=min(values),
                    max=max(values),
                    median=statistics.median(values),
                    meaning=(
                        "search_rounds"
                        if args.family == "threshold"
                        else "queue_merges"
                    ),
                )
                if args.family == "delegate":
                    counts = diagnostic.delegate_stats.cpu().tolist()
                    emit(
                        "delegate_diagnostic",
                        seed=seed,
                        retained_block_fraction=sum(v[1] for v in counts)
                        / sum(v[0] for v in counts),
                        threshold_rounds_max=max(v[2] for v in counts),
                    )
                del diagnostic
            del control, candidate, want, ts
            gc.collect()
    emit("algorithm_worker_complete", family=args.family, task_id=args.worker, ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("family", choices=("all",) + FAMILIES, nargs="?", default="all")
    ap.add_argument("--worker", type=int)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--suite-timeout", type=int, default=13000)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        from hygon_prefill_algorithms_check import check

        check()
        return 0
    if min(args.rounds, args.iters, args.timeout, args.suite_timeout) <= 0:
        ap.error("rounds/iters/timeouts must be positive")
    if args.worker is not None:
        if args.family == "all" or not 0 <= args.worker < len(tasks(args.family)):
            ap.error("invalid worker")
        try:
            worker(args)
            return 0
        except Exception:
            traceback.print_exc()
            return 1
    families = FAMILIES if args.family == "all" else (args.family,)
    occupancy("before")
    deadline = time.monotonic() + args.suite_timeout
    failures, skipped = [], []
    for family in families:
        for task_id, (shape_id, config) in enumerate(tasks(family)):
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                skipped.append([family, task_id])
                continue
            emit(
                "algorithm_worker_start",
                family=family,
                task_id=task_id,
                shape=SHAPES[shape_id],
                config=config,
            )
            cmd = [
                sys.executable,
                "-u",
                __file__,
                family,
                "--worker",
                str(task_id),
                "--rounds",
                str(args.rounds),
                "--iters",
                str(args.iters),
                "--seeds",
                *map(str, args.seeds),
            ]
            try:
                code = subprocess.run(
                    cmd,
                    timeout=min(args.timeout, remaining),
                    env=dict(os.environ, FLAGGEMS_FORCE_TLE="0"),
                ).returncode
            except subprocess.TimeoutExpired:
                code = 124
            emit("algorithm_worker_exit", family=family, task_id=task_id, code=code)
            if code:
                failures.append([family, task_id, code])
    occupancy("after")
    emit(
        "algorithm_suite_complete",
        ok=not (failures or skipped),
        failures=failures,
        skipped=skipped,
    )
    return int(bool(failures or skipped))


if __name__ == "__main__":
    raise SystemExit(main())
