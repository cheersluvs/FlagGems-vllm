"""Paired full-operator audit; see hygon_prefill_audit.md for the contract.

Run on an idle HCU via vendor_probe.sh. Each shape is a subprocess so a GPU
fault cannot erase the other shapes' reports. Production code is untouched.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import traceback
from importlib import import_module
from pathlib import Path

from hygon_prefill_audit_source import GENERIC, OVERRIDE, build, digest

ROOT = Path(__file__).resolve().parents[1]
SHAPES = (
    (64, 129280, 1024, 129280),
    (4, 8193, 512, 8456),
    (16383, 4095, 512, 4352),
    (4, 16385, 512, 16648),
    (12961, 4100, 512, 4360),
    (16380, 5115, 512, 5376),
    (4100, 1025, 512, 1288),
)
ARMS = ("control", "rank8", "rank16", "carry")


def emit(kind, **fields):
    print(json.dumps(dict(kind=kind, **fields), sort_keys=True), flush=True)


def occupancy(label):
    try:
        proc = subprocess.run(["hy-smi"], capture_output=True, text=True, timeout=15)
        emit(
            "occupancy",
            label=label,
            exit=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        emit("occupancy_unavailable", label=label, error=str(exc))


def load_module(folder, dense, arm, diagnostic=False):
    source = build(ROOT, dense, arm, diagnostic)
    name = f"flaggems_vllm.ops._hygon_audit_{arm}_{int(dense)}_{int(diagnostic)}"
    path = Path(folder) / (name.rsplit(".", 1)[-1] + ".py")
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if mod.HAS_TLE:
        raise RuntimeError("Audit requires the shipped non-TLE path")
    emit("source", arm=arm, dense=dense, diagnostic=diagnostic, sha256=digest(source))
    return mod


class Plan:
    def __init__(self, mod, tensors, top_k, block, warps):
        import torch

        launcher = import_module(
            "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
        )._Launch
        x, starts, ends = tensors
        rows, vocab = x.shape
        self.guard = torch.full(
            (rows * top_k + 32,), -123456, device=x.device, dtype=torch.int32
        )
        self.out = self.guard[16:-16].view(rows, top_k)
        self.hist = torch.empty((rows, 2048), device=x.device, dtype=torch.int32)
        self.values = torch.empty((rows, 2048), device=x.device, dtype=torch.float32)
        self.count = torch.full((rows,), -1, device=x.device, dtype=torch.int32)
        self.step = torch.full_like(self.count, -1)
        self.bin_size = torch.full_like(self.count, -1)
        self.found = torch.full_like(self.count, -1)
        self.args = (
            x,
            self.out,
            starts,
            ends,
            x.stride(0),
            x.stride(1),
            vocab,
            self.hist,
            self.values,
            self.count,
            self.step,
            self.bin_size,
            self.found,
        )
        self.launch = launcher(
            mod.non_tle_top_k_per_row_prefill,
            (rows,),
            dict(TOPK=top_k, BLOCK_SIZE=block, ROW_OFFSET=0),
            warps,
        )

    def __call__(self):
        self.launch(*self.args)


def inputs(rows, vocab, stride0, top_k, seed, case="normal", stride1=1):
    import torch

    torch.manual_seed(seed)
    buf = torch.randn(
        (rows - 1) * stride0 + (vocab - 1) * stride1 + 1,
        device="cuda",
        dtype=torch.float32,
    )
    if case == "tied":
        buf = (buf * 4).round() / 4
    elif case == "constant":
        buf.fill_(0.0)
    elif case == "special":
        buf[::11] = float("inf")
        buf[1::11] = float("-inf")
        buf[2::11] = -0.0
    x = torch.as_strided(buf, (rows, vocab), (stride0, stride1))
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.full_like(starts, vocab)
    if case in ("partial", "short"):
        r = torch.arange(rows, device="cuda", dtype=torch.int32)
        starts = r % 7  # includes all float4 alignment offsets
        if case == "short":
            lengths = torch.tensor([0, 1, top_k - 1, top_k, top_k + 1], device="cuda")
            ends = starts + lengths[r.long() % len(lengths)].to(torch.int32)
        else:
            ends = starts + (vocab // 2 + r % 31)
        ends = ends.clamp_max(vocab)
    return x, starts, ends


def oracle(tensors, top_k):
    import torch

    x, starts, ends = tensors
    cols = torch.arange(x.shape[1], device=x.device)[None, :]
    live = (cols >= starts[:, None]) & (cols < ends[:, None])
    want = torch.topk(torch.where(live, x, float("-inf")), top_k, dim=1).values
    return want.sort(dim=1).values


def check_output(out, tensors, top_k, want):
    import torch

    x, starts, ends = tensors
    lens = ends - starts
    valid = torch.arange(top_k, device=x.device)[None, :] < lens[:, None]
    if not bool(
        torch.all(torch.where(valid, (out >= 0) & (out < lens[:, None]), out == -1))
    ):
        raise AssertionError(
            "Index bounds or -1 padding violated (checked BEFORE gather)"
        )
    sorted_idx = out.sort(dim=1).values
    duplicate = (sorted_idx[:, 1:] == sorted_idx[:, :-1]) & (sorted_idx[:, 1:] >= 0)
    if bool(duplicate.any()):
        raise AssertionError("Duplicate valid output index")
    absolute = torch.where(valid, out + starts[:, None], 0).long()
    got = torch.where(valid, x.gather(1, absolute), float("-inf")).sort(dim=1).values
    if not torch.equal(got, want):
        raise AssertionError("Selected value multiset differs from exact torch.topk")


def validate(plan, tensors, top_k, want):
    import torch

    plan.out.fill_(-9)
    plan()
    torch.cuda.synchronize()
    if not bool(
        (plan.guard[:16] == -123456).all() & (plan.guard[-16:] == -123456).all()
    ):
        raise AssertionError("Output guard overwritten")
    check_output(plan.out, tensors, top_k, want)


def device_time(fn, iters):
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
    kernels = [e for e in gpu if "non_tle_top_k_per_row_prefill" in e.name]
    if len(kernels) != iters or len(gpu) != iters:
        raise RuntimeError(
            f"Expected {iters} single-kernel events; matched={len(kernels)}, "
            f"total={len(gpu)}, names={sorted(set(e.name for e in gpu))}"
        )
    durations = [e.time_range.elapsed_us() for e in kernels]
    if not all(math.isfinite(x) and x > 0 for x in durations):
        raise RuntimeError(f"Invalid device durations: {durations}")
    return dict(
        us=statistics.median(durations),
        mean_us=statistics.mean(durations),
        kernel_events=len(kernels),
        samples_us=durations,
    )


def stats(plan, label, seed):
    import torch

    plan()
    torch.cuda.synchronize()
    fields = {}
    for name in ("count", "bin_size", "found"):
        v = getattr(plan, name).float()
        fields[name] = dict(
            min=v.min().item(),
            median=v.median().item(),
            p99=v.quantile(0.99).item(),
            max=v.max().item(),
        )
    steps, counts = plan.step.unique(return_counts=True)
    emit(
        "candidate_stats",
        case=label,
        seed=seed,
        **fields,
        last_step=dict(zip(map(str, steps.tolist()), counts.tolist())),
    )


def adversarial_checks(modules, diag, shape, block, warps):
    rows, vocab, top_k, stride0 = shape
    failures = []
    # Keep the exact timed geometry/routing while bounding expensive tie cases.
    for case in ("tied", "constant", "partial", "short", "special", "strided"):
        n = min(rows, 32)
        # The production contract is column-contiguous.  Exercise the
        # supported non-contiguous case with padded rows; a column stride of
        # two makes the control implementation fail before candidate timing.
        step = 1
        stride = max(stride0, vocab + 8)
        tensors = inputs(n, vocab, stride, top_k, 123, case, step)
        want = oracle(tensors, top_k)
        for arm, module in modules.items():
            plan = Plan(module, tensors, top_k, block, warps)
            try:
                validate(plan, tensors, top_k, want)
                emit("validation", arm=arm, case=case, ok=True)
            except Exception as exc:
                failures.append(f"{arm}/{case}: {exc}")
                emit("validation", arm=arm, case=case, ok=False, error=str(exc))
            del plan
        diagnostic = Plan(diag, tensors, top_k, block, warps)
        validate(diagnostic, tensors, top_k, want)
        stats(diagnostic, case, 123)
        del diagnostic, tensors, want
    return failures


def worker(args):
    import torch
    import triton

    import flaggems_vllm

    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    if flaggems_vllm.top_k_per_row_prefill is not ov.top_k_per_row_prefill:
        raise RuntimeError("Public operator is not the Hygon override")
    if not ov._ONESCAN_PATH or not ov._ENABLED or not ov._GEOMETRY:
        raise RuntimeError("Expected shipped one-scan, slot-scan and geometry settings")
    rows, vocab, top_k, stride0 = SHAPES[args.worker]
    dense = vocab <= ov.DENSE_VOCAB_PER_TOPK * top_k
    geo = ov._geometry(rows, vocab)
    mod = ov._dense if dense else ov._sparse
    block, warps = geo or (
        mod.NUM_THREADS_PER_BLOCK,
        mod._num_warps(mod.NUM_THREADS_PER_BLOCK),
    )
    emit(
        "device",
        shape=SHAPES[args.worker],
        dense=dense,
        block=block,
        warps=warps,
        props=str(torch.cuda.get_device_properties(torch.cuda.current_device())),
        torch=torch.__version__,
        triton=triton.__version__,
    )
    failures = []
    ratios = {arm: [] for arm in ARMS[1:] if arm != "carry" or dense}
    with tempfile.TemporaryDirectory(prefix="hygon_prefill_audit_") as folder:
        modules = {a: load_module(folder, dense, a) for a in ("control", *ratios)}
        diag = load_module(folder, dense, "control", diagnostic=True)
        failures = adversarial_checks(modules, diag, SHAPES[args.worker], block, warps)
        if failures:
            emit("validation_failed", shape_id=args.worker, failures=failures)
            return 1
        for seed in args.seeds:
            tensors = inputs(rows, vocab, stride0, top_k, seed)
            want = oracle(tensors, top_k)
            plans = {
                a: Plan(m, tensors, top_k, block, warps) for a, m in modules.items()
            }
            for arm, plan in plans.items():
                validate(plan, tensors, top_k, want)
                emit("validation", arm=arm, case="full_normal", seed=seed, ok=True)
            public_out = torch.empty_like(plans["control"].out)

            def public(tensors=tensors, public_out=public_out):
                x, starts, ends = tensors
                flaggems_vllm.top_k_per_row_prefill(
                    x, starts, ends, public_out, rows, stride0, 1, top_k
                )

            public()
            check_output(public_out, tensors, top_k, want)
            # Public/copy kernel agreement is measured with the same device timer.
            emit(
                "public_control",
                seed=seed,
                public=device_time(public, args.iters),
                control=device_time(plans["control"], args.iters),
            )
            diagnostic = Plan(diag, tensors, top_k, block, warps)
            validate(diagnostic, tensors, top_k, want)
            stats(diagnostic, "full_normal", seed)
            del diagnostic
            seed_ratios = {arm: [] for arm in ratios}
            for round_id in range(args.rounds):
                candidates = list(ratios)
                shift = round_id % len(candidates)
                candidates = candidates[shift:] + candidates[:shift]
                for arm in candidates:
                    order = (
                        ("control", arm, arm, "control")
                        if round_id % 2 == 0
                        else (arm, "control", "control", arm)
                    )
                    readings = [(a, device_time(plans[a], args.iters)) for a in order]
                    paired = []
                    for i, j in ((0, 1), (3, 2)):
                        pair = dict(readings[p] for p in (i, j))
                        paired.append(pair["control"]["us"] / pair[arm]["us"])
                    seed_ratios[arm].extend(paired)
                    emit(
                        "paired_round",
                        seed=seed,
                        round=round_id,
                        arm=arm,
                        order=order,
                        readings=readings,
                        ratios=paired,
                    )
            for arm, values in seed_ratios.items():
                ratios[arm].extend(values)
                emit(
                    "seed_summary",
                    seed=seed,
                    arm=arm,
                    median_ratio=statistics.median(values),
                    min_ratio=min(values),
                    max_ratio=max(values),
                    ratios=values,
                )
            del plans, want, public_out, tensors, public

    emit(
        "shape_summary",
        shape_id=args.worker,
        shape=SHAPES[args.worker],
        medians={a: statistics.median(v) for a, v in ratios.items()},
        failures=failures,
    )
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, choices=range(len(SHAPES)))
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument(
        "--shapes",
        type=int,
        nargs="+",
        choices=range(len(SHAPES)),
        default=list(range(7)),
    )
    args = parser.parse_args()
    if args.rounds < 2 or args.iters < 2:
        parser.error("At least two rounds and iterations are required")
    if args.worker is not None:
        return worker(args)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    emit(
        "audit",
        commit=commit,
        host=platform.node(),
        shapes=args.shapes,
        env={
            k: os.environ.get(k)
            for k in (
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "FLAGGEMS_FORCE_TLE",
            )
        },
        source_sha256={
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (GENERIC, OVERRIDE)
        },
    )
    occupancy("before")
    summaries, failed = [], []
    for shape_id in args.shapes:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(shape_id),
            "--rounds",
            str(args.rounds),
            "--iters",
            str(args.iters),
            "--seeds",
            *map(str, args.seeds),
        ]
        emit("worker_start", shape_id=shape_id, shape=SHAPES[shape_id])
        try:
            proc = subprocess.run(
                cmd, cwd=ROOT, capture_output=True, text=True, timeout=900
            )
            print(proc.stdout, end="", flush=True)
            print(proc.stderr, end="", flush=True)
            if proc.returncode:
                failed.append(shape_id)
            for line in proc.stdout.splitlines():
                if line.startswith('{"'):
                    record = json.loads(line)
                    if record.get("kind") == "shape_summary":
                        summaries.append(record)
            emit("worker_exit", shape_id=shape_id, code=proc.returncode)
        except subprocess.TimeoutExpired as exc:
            failed.append(shape_id)
            print((exc.stdout or b"").decode(errors="replace"), flush=True)
            emit("worker_timeout", shape_id=shape_id, seconds=900)
    occupancy("after")
    # Sparse carry is exactly control; count it as identity, not a measured win.
    if not failed and len(summaries) == 7 and set(args.shapes) == set(range(7)):
        emit(
            "aggregate",
            geomean={
                a: math.exp(
                    sum(math.log(s["medians"].get(a, 1.0)) for s in summaries) / 7
                )
                for a in ARMS[1:]
            },
        )
    emit(
        "audit_complete", ok=not failed, failed_shapes=failed, summaries=len(summaries)
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
