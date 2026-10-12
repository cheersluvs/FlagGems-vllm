# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Standalone, correctness-gated MUSA A/B/C experiment. See README.md."""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from plan import cache_stride, geometry, positive_list

HERE = Path(__file__).resolve().parent
ARMS = ("A", "A2", "B", "B1", "C")
BASE_SHA256 = "04a4eeddc6dda1f66cfed8cef51bcfe8b004750eca9d753db247d66655e77a8d"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="512,1024,2048,8192,32768")
    parser.add_argument("--heads", default="64")
    parser.add_argument("--head-tiles", default="4")
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/mthreads-abc.json"))
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".csv").exists():
        parser.error(
            "report already exists; choose a new --output to preserve earlier trials"
        )
    if args.quick:
        args.tokens, args.rounds, args.repeat = "512,1024", 3, 10
    try:
        args.tokens = positive_list(args.tokens)
        args.heads = positive_list(args.heads)
        args.head_tiles = positive_list(args.head_tiles, powers_of_two=True)
        if min(args.rounds, args.repeat, args.block_size) <= 0 or args.warmup < 0:
            raise ValueError("rounds/repeat/block-size must be positive; warmup >= 0")
        if args.warps not in (1, 2, 4, 8) or max(args.head_tiles) > 32:
            raise ValueError("warps must be 1/2/4/8; head tiles <= 32")
        if args.device < 0:
            raise ValueError("device index must be >= 0")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def command_output(command):
    try:
        return subprocess.check_output(
            command, cwd=HERE, text=True, stderr=subprocess.STDOUT, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(name, filename):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def preflight(args, report):
    report["source"] = {
        p.name: {"path": str(p), "sha256": sha(p)} for p in sorted(HERE.glob("*.py"))
    }
    report["git"] = {
        "head": command_output(["git", "rev-parse", "HEAD"]),
        "branch": command_output(["git", "branch", "--show-current"]),
        "remote": command_output(["git", "remote", "-v"]),
        "status": command_output(["git", "status", "--short", "--", str(HERE)]),
    }
    if sha(HERE / "frozen_a.py") != BASE_SHA256:
        raise RuntimeError("frozen A differs from the baseline: stop, do not compare")
    # No flag_gems/flaggems_vllm/vllm import: avoid stale installs/platform plugins.
    import torch
    import torch_musa
    import triton

    env = {
        "python": sys.executable,
        "torch": {"version": torch.__version__, "path": torch.__file__},
        "torch_musa": {
            "version": getattr(torch_musa, "__version__", None),
            "path": torch_musa.__file__,
        },
        "triton": {"version": triton.__version__, "path": triton.__file__},
        "selected_env": {
            k: os.getenv(k)
            for k in (
                "MUSA_VISIBLE_DEVICES",
                "MUSA_HOME",
                "TRITON_CACHE_DIR",
                "TRITON_LLC_PATH",
            )
        },
    }
    try:
        env["flagtree_distribution"] = importlib.metadata.version("flagtree")
    except importlib.metadata.PackageNotFoundError:
        env["flagtree_distribution"] = None
    llc = Path(triton.__file__).resolve().parent / "backends/mthreads/bin/llc"
    env["bundled_llc"] = {"path": str(llc), "exists": llc.is_file()}
    if llc.is_file():
        env["bundled_llc"].update(
            sha256=sha(llc), md5=hashlib.md5(llc.read_bytes()).hexdigest()
        )
    else:
        env["compiler_warning"] = (
            "no bundled llc; older toolkit llc may fail instruction selection"
        )
    report["environment"] = env
    print(json.dumps(env, ensure_ascii=False, indent=2), flush=True)
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError(
            "torch.musa is unavailable; this probe requires a Moore Threads card"
        )
    env["device_count"] = torch.musa.device_count()
    if args.device >= env["device_count"]:
        raise ValueError("selected device index is out of range")
    torch.musa.set_device(args.device)
    env["device_name"] = torch.musa.get_device_name(args.device)
    device = torch.device(f"musa:{args.device}")
    frozen = load_source("abc_frozen_a", "frozen_a.py")
    candidates = load_source("abc_candidates", "candidates.py")
    return torch, device, frozen, candidates


def make_case(torch, device, n, h, bs, seed, mode="normal"):
    torch.manual_seed(seed)
    torch.musa.manual_seed_all(seed)
    q0 = torch.randn((n, h, 512), dtype=torch.bfloat16, device=device)
    kv = torch.randn((n, 512), dtype=torch.bfloat16, device=device)
    # Unique reversed slot order exercises nontrivial paged-cache addressing.
    slots_cpu = torch.arange(n - 1, -1, -1, dtype=torch.int64)
    if mode == "mixed":
        slots_cpu = slots_cpu[: max(0, n - 3)].clone()
        slots_cpu[::3] = -1
    elif mode == "empty":
        slots_cpu = slots_cpu[:0]
    elif mode == "zeros":
        q0.zero_()
        kv.zero_()
    elif mode == "quant_edges":
        values = torch.tensor(
            [0.0, -0.0, 0.0009765625, -0.0029296875, 1.0625, -1.1875, 224, -448],
            dtype=torch.bfloat16,
        )
        kv[:, :448] = values.repeat(56).to(device)
    positions = (torch.arange(n, dtype=torch.int64) * 7 + 3) % 257
    angle = torch.arange(257, dtype=torch.float32)[:, None] * (
        10000 ** (-torch.arange(32, dtype=torch.float32)[None, :] / 32)
    )
    cs = torch.cat((angle.cos(), angle.sin()), 1).to(device)
    blocks = max(1, (n + bs - 1) // bs)
    cache = torch.full(
        (blocks, cache_stride(bs)), 165, dtype=torch.uint8, device=device
    )
    return {
        "q0": q0,
        "q": torch.empty_like(q0),
        "kv": kv,
        "kv_cpu": kv.cpu(),
        "cache": cache,
        "slots": slots_cpu.to(device),
        "slots_cpu": slots_cpu,
        "positions": positions.to(device),
        "positions_cpu": positions,
        "cs": cs,
        "cs_cpu": cs.cpu(),
        "n": n,
        "h": h,
        "bs": bs,
        "eps": 1e-6,
    }


def launch(case, arm, hpp, warps, frozen, candidates):
    n, h, bs = case["n"], case["h"], case["bs"]
    grid, kv_tpp, merge = geometry(arm, n, h, hpp)
    if arm in ("A", "A2"):
        return frozen._tiled_kernel[(grid,)](
            case["q"],
            case["kv"],
            case["cache"],
            case["slots"],
            case["positions"],
            case["cs"],
            case["eps"],
            bs,
            n,
            h,
            case["cache"].stride(0),
            case["slots"].numel(),
            TPP=4,
            num_warps=4,
            num_stages=2,
        )
    return candidates.head_kernel[(grid,)](
        case["q"],
        case["kv"],
        case["cache"],
        case["slots"],
        case["positions"],
        case["cs"],
        case["eps"],
        n,
        case["cache"].stride(0),
        case["slots"].numel(),
        H=h,
        BS=bs,
        HPP=hpp,
        KV_TPP=kv_tpp,
        MERGE=merge,
        num_warps=warps,
        num_stages=2,
    )


def reset(case):
    case["q"].copy_(case["q0"])
    case["cache"].fill_(165)


def rotate_cpu(torch, value, positions, cs):
    rope = value[..., 448:].reshape(*value.shape[:-1], 32, 2)
    cos, sin = cs[positions, :32], cs[positions, 32:]
    if value.ndim == 3:
        cos, sin = cos[:, None, :], sin[:, None, :]
    even, odd = rope[..., 0], rope[..., 1]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


def close_cpu(torch, actual, expected, label):
    # Avoid torch_musa's historical assert_close diff-reporting failures.
    a, b = actual.float(), expected.float()
    error = (a - b).abs()
    bad = (~torch.isfinite(a)) | (error > 1e-2 + 1e-2 * b.abs())
    if bool(bad.any()):
        raise AssertionError(
            f"{label}: {int(bad.sum())} violations, max abs={float(error.max())}"
        )
    return float(error.max()) if error.numel() else 0.0


def q_oracle_check(torch, case):
    max_error = 0.0
    # Bounded CPU temporaries, and no >2^31-element device reference operation.
    step = max(1, 32768 // case["h"])
    for start in range(0, case["n"], step):
        end = min(case["n"], start + step)
        value = case["q0"][start:end].cpu().float()
        normalized = value * torch.rsqrt(
            (value * value).mean(-1, keepdim=True) + case["eps"]
        )
        expected = normalized.to(torch.bfloat16)
        expected[..., 448:] = rotate_cpu(
            torch, normalized, case["positions_cpu"][start:end], case["cs_cpu"]
        ).to(torch.bfloat16)
        max_error = max(
            max_error,
            close_cpu(torch, case["q"][start:end].cpu(), expected, "Q oracle"),
        )
    return max_error


def cache_oracle(torch, case):
    bs, n = case["bs"], case["n"]
    expected = torch.full(
        ((n + bs - 1) // bs, cache_stride(bs)), 165, dtype=torch.uint8
    )
    rope_bytes = torch.zeros_like(expected, dtype=torch.bool)
    slots = case["slots_cpu"]
    token = torch.arange(slots.numel())[slots >= 0]
    slot = slots[slots >= 0]
    if not token.numel():
        return expected, rope_bytes
    kv = case["kv_cpu"][token].float()
    rope = rotate_cpu(torch, kv, case["positions_cpu"][token], case["cs_cpu"])
    groups = kv[:, :448].reshape(-1, 7, 64)
    exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.0))
    scale = ((exponent.to(torch.int32) + 127) << 23).contiguous().view(torch.float32)
    encoded = (exponent + 127).clamp(0, 255).to(torch.uint8)
    quant = (
        (groups / scale[..., None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .reshape(-1, 448)
    )
    block, pos = slot // bs, slot % bs
    data = pos * 576
    expected[block[:, None], data[:, None] + torch.arange(448)] = quant
    rope_indices = data[:, None] + 448 + torch.arange(128)
    expected[block[:, None], rope_indices] = (
        rope.to(torch.bfloat16).contiguous().view(torch.uint8)
    )
    rope_bytes[block[:, None], rope_indices] = True
    scales = bs * 576 + pos * 8
    expected[block[:, None], scales[:, None] + torch.arange(7)] = encoded
    expected[block, scales + 7] = 0
    return expected, rope_bytes


def resources(compiled):
    result = {}
    for key in ("n_regs", "n_spills"):
        value = getattr(compiled, key, None)
        if value is not None:
            result[key] = value
    metadata = getattr(compiled, "metadata", None)
    if metadata is not None:
        result["metadata"] = (
            metadata._asdict() if hasattr(metadata, "_asdict") else str(metadata)
        )
    return result


def validate(torch, case, hpp, warps, frozen, candidates):
    expected, rope_mask = cache_oracle(torch, case)
    result, baseline = {}, None
    for arm in ("A", "B", "B1", "C"):
        reset(case)
        compiled = launch(case, arm, hpp, warps, frozen, candidates)
        torch.musa.synchronize()
        error = q_oracle_check(torch, case)
        actual = case["cache"].cpu()
        bad = int(((actual != expected) & ~rope_mask).sum())
        if bad:
            raise AssertionError(
                f"{arm}: {bad} quant/scale/padding bytes differ from CPU oracle"
            )
        rope_error = close_cpu(
            torch,
            actual[rope_mask].contiguous().view(torch.bfloat16),
            expected[rope_mask].contiguous().view(torch.bfloat16),
            f"{arm} KV RoPE",
        )
        if baseline is None:
            baseline = actual
        elif not torch.equal(actual, baseline):
            raise AssertionError(
                f"{arm}: {int((actual != baseline).sum())} full-cache bytes differ from A"
            )
        if not torch.equal(case["kv"].cpu(), case["kv_cpu"]):
            raise AssertionError(f"{arm}: read-only KV was modified")
        result[arm] = {
            "q_max_abs": error,
            "rope_max_abs": rope_error,
            "cache_bytes_differing_vs_A": 0,
            "compiled": resources(compiled),
        }
    return result


def benchmark(torch, case, args, hpp, frozen, candidates):
    rng = random.Random(args.seed)
    for arm in ARMS:
        for _ in range(args.warmup):
            reset(case)
            launch(case, arm, hpp, args.warps, frozen, candidates)
        torch.musa.synchronize()
    samples = {arm: [] for arm in ARMS}
    start, end = torch.musa.Event(enable_timing=True), torch.musa.Event(
        enable_timing=True
    )
    # Synchronize BEFORE start so CPU dispatch delay is part of eager event latency;
    # do not describe these timings as graph replay or pure hardware cycle counts.
    for round_id in range(args.rounds):
        current = {arm: [] for arm in ARMS}
        for rep in range(args.repeat):
            order = list(ARMS)
            rng.shuffle(order)
            for arm in order:
                reset(case)
                torch.musa.synchronize()
                start.record()
                launch(case, arm, hpp, args.warps, frozen, candidates)
                end.record()
                end.synchronize()
                ms = float(start.elapsed_time(end))
                if not math.isfinite(ms) or ms <= 0:
                    raise RuntimeError(f"invalid event latency {ms} for {arm}")
                current[arm].append(ms)
        for arm in ARMS:
            samples[arm].append(current[arm])
        print(
            f"  round {round_id + 1}: "
            + " ".join(f"{a}={statistics.median(current[a]):.5f}" for a in ARMS),
            flush=True,
        )
    rounds = {a: [statistics.median(r) for r in samples[a]] for a in ARMS}
    median = {a: statistics.median(rounds[a]) for a in ARMS}
    ratios = {
        "A/B": median["A"] / median["B"],
        "A/B1": median["A"] / median["B1"],
        "A/C": median["A"] / median["C"],
        "B/B1": median["B"] / median["B1"],
        "B1/C": median["B1"] / median["C"],
        "A/A2": median["A"] / median["A2"],
    }
    return {
        "median_ms": median,
        "round_medians_ms": rounds,
        "samples_ms": samples,
        "ratios": ratios,
    }


def save(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    rows = []
    for item in report["shapes"]:
        if "timing" in item:
            rows.append(
                {
                    "tokens": item["tokens"],
                    "heads": item["heads"],
                    "hpp": item["hpp"],
                    "warps": item["warps"],
                    **{f"{k}_ms": v for k, v in item["timing"]["median_ms"].items()},
                    **item["timing"]["ratios"],
                }
            )
    if rows:
        with path.with_suffix(".csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    args = arguments()
    report = {
        "status": "running",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": vars(args),
        "boundary_tests": [],
        "shapes": [],
        "method": "eager MUSA events; reset outside timed interval; CPU oracle; forced A tile, not public dispatch",
    }
    phase = "preflight"
    try:
        torch, device, frozen, candidates = preflight(args, report)
        phase = "boundary correctness"
        cases = [
            (1, 1, 16, "normal"),
            (17, 7, 16, "mixed"),
            (9, 65, 64, "empty"),
            (17, 64, 64, "zeros"),
            (17, 64, 64, "quant_edges"),
        ]
        for n, h, bs, mode in cases:
            for hpp in args.head_tiles:
                case = make_case(torch, device, n, h, bs, args.seed, mode)
                report["current"] = {
                    "phase": phase,
                    "tokens": n,
                    "heads": h,
                    "hpp": hpp,
                    "mode": mode,
                }
                checks = validate(torch, case, hpp, args.warps, frozen, candidates)
                report["boundary_tests"].append({**report["current"], "checks": checks})
                print(f"PASS boundary {n}x{h}, bs={bs}, HPP={hpp}, {mode}", flush=True)
                del case
                torch.musa.empty_cache()
                save(report, args.output)
        for n in args.tokens:
            for h in args.heads:
                for hpp in args.head_tiles:
                    item = {
                        "tokens": n,
                        "heads": h,
                        "hpp": hpp,
                        "warps": args.warps,
                        "block_size": args.block_size,
                    }
                    report["shapes"].append(item)
                    report["current"] = item
                    phase = "shape allocation"
                    # q0 + q + device KV/cache and modest overhead; CPU oracle is chunked.
                    required = 2 * n * h * 512 * 2 + n * (512 * 2 + 584) + (64 << 20)
                    if hasattr(torch.musa, "mem_get_info"):
                        free, _ = torch.musa.mem_get_info()
                        if required > free * 0.8:
                            item.update(
                                status="skipped_memory",
                                required_bytes=required,
                                free_bytes=free,
                            )
                            print(f"SKIP memory {n}x{h}", flush=True)
                            continue
                    case = make_case(torch, device, n, h, args.block_size, args.seed)
                    phase = "shape correctness"
                    item["checks"] = validate(
                        torch, case, hpp, args.warps, frozen, candidates
                    )
                    print(f"PASS shape {n}x{h}, HPP={hpp}", flush=True)
                    if not args.validate_only:
                        phase = "timing"
                        item["timing"] = benchmark(
                            torch, case, args, hpp, frozen, candidates
                        )
                        print(
                            "  ratios " + json.dumps(item["timing"]["ratios"]),
                            flush=True,
                        )
                    item["status"] = "passed"
                    del case
                    torch.musa.empty_cache()
                    save(report, args.output)
        report["aggregate"] = {}
        for hpp in args.head_tiles:
            values = [x for x in report["shapes"] if x["hpp"] == hpp and "timing" in x]
            if values:
                report["aggregate"][str(hpp)] = {
                    metric: {
                        "min": min(v),
                        "max": max(v),
                        "arithmetic_mean": statistics.mean(v),
                        "count": len(v),
                    }
                    for metric in ("A/B", "A/C", "B1/C", "A/A2")
                    for v in [[x["timing"]["ratios"][metric] for x in values]]
                }
        skipped = any(x.get("status") == "skipped_memory" for x in report["shapes"])
        report["status"] = "incomplete_memory" if skipped else "passed"
        save(report, args.output)
        print(f"RESULT {report['status']}: {args.output}", flush=True)
        print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2), flush=True)
        return 2 if skipped else 0
    except Exception as exc:
        report.update(
            status="failed",
            failure_phase=phase,
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        save(report, args.output)
        traceback.print_exc()
        print(f"FAILED during {phase}; partial report: {args.output}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
