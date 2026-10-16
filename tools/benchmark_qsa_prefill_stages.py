# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Measure production QSA prefill stages and pinned host A/B in one runtime.

No native vLLM import, installation, source patch or alternative Top-K. Each
case runs in a fresh process. Use --list-cases without GPU dependencies.
"""

import argparse
import cProfile
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import platform
import pstats
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 2
SOURCES = [
    "tools/benchmark_qsa_prefill_stages.py",
    "tools/qsa_prefill_validation_baseline.py",
    "tools/qsa_prefill_stride_baseline.py",
    "tools/qsa_prefill_chunk_baseline.py",
    "tools/qsa_prefill_single_row_trial.py",
    "tools/qsa_prefill_gpu_guard.py",
    "tools/qsa_prefill_score_benchmark.py",
    "tools/qsa_prefill_score_resume.py",
    "tools/qsa_prefill_upstream_workloads.py",
    "benchmark/base.py",
    "benchmark/conftest.py",
    "benchmark/consts.py",
    "tools/run_qsa_prefill_validation_ab_h800.sh",
    "tests/qsa_select_prefill_utils.py",
    "tests/test_qsa_select_paged_prefill.py",
    "tests/test_qsa_prefill_single_row_trial.py",
    "tests/conftest.py",
    "src/flaggems_vllm/ops/qsa_select_paged_prefill.py",
    "src/flaggems_vllm/ops/persistent_topk.py",
    "src/flaggems_vllm/utils/libentry.py",
    "src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml",
]
METHODS = {
    "wall_us": "perf_counter, synchronize before/after a batch; host inclusive",
    "event_us": "device events around an eager batch; includes host supply gaps",
    "graph_us": "device events around graph replays / captured calls per replay",
}


def ab_configuration(options):
    if options.single_row_ab:
        return (
            "single_row_ab",
            "full_matmul_baseline",
            "tools.qsa_prefill_single_row_trial",
            "single-row-vector",
        )
    if options.host_ab:
        return (
            "host_ab",
            "full_host_baseline",
            "tools.qsa_prefill_chunk_baseline",
            "chunk-setup",
        )
    if options.validation_ab:
        module = (
            "tools.qsa_prefill_stride_baseline"
            if options.validation_baseline == "overlap-retained"
            else "tools.qsa_prefill_validation_baseline"
        )
        return (
            "validation_ab",
            "full_validation_baseline",
            module,
            options.validation_baseline,
        )
    return None


def cases(single_row=False):
    # Historical diagnostic fixtures, available only via explicit legacy suites.
    core = [
        (1, 1, 2048, 16, 512),
        (1, 64, 4096, 16, 512),
        (4, 128, 16384, 64, 512),
        (1, 2048, 65536, 64, 512),
        (1, 64, 4096, 16, 1024),
        (1, 64, 8192, 16, 2048),
    ]
    result = [
        dict(name=f"core_{i}", kind="core", shape=list(shape), dtype="bfloat16")
        for i, shape in enumerate(core)
    ]
    for dtype in ("bfloat16", "float8_e4m3fn"):
        for slack in (0, 1792):
            result.append(
                dict(
                    name=f"upstream_{dtype}_s{slack}",
                    kind="upstream",
                    slack=slack,
                    dtype=dtype,
                )
            )
    result.append(dict(name="zero_visible", kind="zero", dtype="bfloat16"))
    for dtype in ("bfloat16", "float8_e4m3fn"):
        for heads, dim in ((1, 32), (3, 64), (5, 96), (8, 256)):
            result.append(
                dict(
                    name=f"hd_{heads}_{dim}_{dtype}",
                    kind="hd",
                    heads=heads,
                    dim=dim,
                    dtype=dtype,
                    extended=True,
                )
            )
        result.append(
            dict(
                name=f"one_row_{dtype}",
                kind="upstream",
                slack=0,
                budget_mb=0,
                dtype=dtype,
                extended=True,
            )
        )
    result.append(
        dict(name="mixed_padded", kind="mixed", dtype="bfloat16", extended=True)
    )
    for dtype in ("bfloat16", "float8_e4m3fn"):
        result.append(
            dict(
                name=f"tail_chunk_{dtype}",
                kind="tail",
                dtype=dtype,
                budget_mb=1,
                extended=True,
            )
        )
    if single_row:
        for dtype in ("bfloat16", "float8_e4m3fn"):
            for heads, dim in ((1, 32), (3, 64), (4, 128), (5, 96), (8, 256), (17, 33)):
                result.append(
                    dict(
                        name=f"single_{heads}_{dim}_{dtype}",
                        kind="single",
                        heads=heads,
                        dim=dim,
                        dtype=dtype,
                        extended=True,
                    )
                )
            result.append(
                dict(
                    name=f"tail_one_{dtype}",
                    kind="tail_one",
                    dtype=dtype,
                    budget_mb=1,
                    extended=True,
                )
            )
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def upstream_workloads_module():
    spec = importlib.util.spec_from_file_location(
        "qsa_prefill_upstream_workloads",
        ROOT / "tools/qsa_prefill_upstream_workloads.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def suite_cases(options):
    if options.suite == "upstream-prefill":
        return upstream_workloads_module().cases()
    return cases(options.single_row_ab)


def command(args):
    try:
        proc = subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, timeout=30
        )
        return dict(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(error=str(exc))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def gpu_guard_module():
    spec = importlib.util.spec_from_file_location(
        "qsa_prefill_gpu_guard", ROOT / "tools/qsa_prefill_gpu_guard.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_resume_module():
    spec = importlib.util.spec_from_file_location(
        "qsa_score_resume", ROOT / "tools/qsa_prefill_score_resume.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def idle_gpu_gate(summary, directory, env, label, guard):
    for attempt in range(1, 4):
        check = guard.idle(env.get("CUDA_VISIBLE_DEVICES", ""), command)
        check.update(label=label, attempt=attempt)
        summary.setdefault("idle_gpu_checks", []).append(check)
        write_json(directory / "stage-summary.json", summary)
        if check["status"] == "passed":
            return True
        if not check.get("retryable_utilization") or attempt == 3:
            break
        print(
            "No GPU processes; rechecking trailing utilization in one second",
            flush=True,
        )
        time.sleep(1)
    if check["status"] != "passed":
        if check.get("processes"):
            check["process_details"] = guard.process_details(
                [p["pid"] for p in check["processes"]], command
            )
        summary["environment_rejected"] = True
        summary.setdefault("accuracy", dict(status="not_run_environment_rejected"))
        print(f"GPU environment rejected: {check['reasons']}", flush=True)
    write_json(directory / "stage-summary.json", summary)
    return check["status"] == "passed"


def guarded_telemetry(uuid, label, result, report, guard):
    point = telemetry(uuid, label)
    device_id = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
    check = guard.timing(
        point, guard.process_query(device_id, command), device_id, guard.self_pids()
    )
    point["gpu_guard"] = check
    if check["status"] != "passed" and check.get("processes"):
        check["process_details"] = guard.process_details(
            [p["pid"] for p in check["processes"]], command
        )
        check["nvml_process_names"] = guard.process_names(device_id, command)
    result.setdefault("telemetry", []).append(point)
    result["performance_guard"] = dict(
        status="observed_clean" if check["status"] == "passed" else "rejected",
        last_label=label,
        reasons=check["reasons"],
    )
    # Persist rejection evidence before the worker exception handler runs.
    write_json(report, result)
    if check["status"] != "passed":
        raise RuntimeError(f"GPU environment rejected: {check['reasons']}")
    return point


def stats(samples):
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError(f"Invalid timing samples: {samples}")
    return dict(
        samples=samples,
        median=statistics.median(samples),
        mean=statistics.mean(samples),
        min=min(samples),
        max=max(samples),
        stdev=statistics.stdev(samples) if len(samples) > 1 else 0.0,
    )


def accuracy_results(path):
    if not path.is_file():
        return {}, "Correctness runner did not write its result file"
    records = json.loads(path.read_text())
    counts = {}
    unexpected = []
    for name, record in records.items():
        status = record.get("result")
        counts[status] = counts.get(status, 0) + 1
        optional_second_gpu = (
            name.endswith("::test_noncurrent_cuda_device")
            and status == "skipped"
            and "a second SM90 device" in record.get("reason", "")
        )
        if status != "passed" and not optional_second_gpu:
            unexpected.append(name)
    if not counts.get("passed") or unexpected:
        return (
            counts,
            f"Correctness missing passes or unexpected non-passes: {unexpected}",
        )
    return counts, None


def timed_batch(fn, device, iterations, method, divisor=1):
    device.synchronize()
    if method == "wall_us":
        begin = time.perf_counter()
        for _ in range(iterations):
            fn()
        device.synchronize()
        elapsed_us = (time.perf_counter() - begin) * 1e6
    else:
        begin = device.Event(enable_timing=True)
        end = device.Event(enable_timing=True)
        # Initialize events outside the measured interval.
        begin.record()
        end.record()
        end.synchronize()
        begin.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        elapsed_us = begin.elapsed_time(end) * 1000
    return elapsed_us / (iterations * divisor)


def timing_plan(options):
    if options.mode == "kernel":
        return dict(
            mode="kernel",
            scope="score_only_no_topk",
            rounds=options.rounds,
            timer="benchmark.base.Benchmark.get_latency",
            warmup_ms=options.kernel_warmup_ms,
            rep_ms=options.kernel_rep_ms,
            order="alternate_baseline_candidate_each_round",
        )
    batch = options.graph_batch
    calls = options.iterations
    if not options.no_graph:
        calls = (calls + batch - 1) // batch * batch
    return dict(
        calls_per_round=calls,
        eager_iterations=calls,
        graph_replays=0 if options.no_graph else calls // batch,
        graph_batch=batch,
        rounds=options.rounds,
        order="rotate_stages_and_modes_each_round",
    )


def telemetry(uuid, label):
    if uuid == "unavailable":
        return dict(label=label, status="unavailable", reason="GPU UUID missing")
    device_id = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
    fields = "timestamp,uuid,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw"
    return dict(
        label=label,
        monotonic_seconds=time.monotonic(),
        clocks=command(
            [
                "nvidia-smi",
                "-i",
                device_id,
                f"--query-gpu={fields}",
                "--format=csv",
            ]
        ),
        performance=command(["nvidia-smi", "-i", device_id, "-q", "-d", "PERFORMANCE"]),
    )


def measure_rounds(functions, graphs, device, options, monitor):
    """Equal work per mode, with no fixed wall -> event -> graph phase."""
    plan = timing_plan(options)
    names = list(functions)
    modes = ["wall_us", "event_us", "graph_us"]
    samples = {name: {mode: [] for mode in modes} for name in names}
    observations = [monitor("before_timing")]
    order = []
    for repeat in range(options.rounds):
        stage_order = names[repeat % len(names) :] + names[: repeat % len(names)]
        mode_order = modes[repeat % 3 :] + modes[: repeat % 3]
        measured = []
        for name in stage_order:
            for method in mode_order:
                if method == "graph_us":
                    if name not in graphs:
                        continue
                    fn = graphs[name].replay
                    iterations = plan["graph_replays"]
                    divisor = plan["graph_batch"]
                else:
                    fn = functions[name]
                    iterations = plan["eager_iterations"]
                    divisor = 1
                samples[name][method].append(
                    timed_batch(fn, device, iterations, method, divisor)
                )
                measured.append([name, method])
        order.append(measured)
        observations.append(monitor(f"after_round_{repeat + 1}"))
    return samples, observations, order


def host_microbench(fn, device, iterations, rounds):
    # The CPU clock excludes sleeping; wall can include allocator/driver waits.
    wall, cpu = [], []
    for _ in range(rounds):
        device.synchronize()
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        for _ in range(iterations):
            fn()
        cpu.append((time.process_time() - start_cpu) * 1e6 / iterations)
        wall.append((time.perf_counter() - start_wall) * 1e6 / iterations)
        device.synchronize()
    return dict(wall_us=stats(wall), process_cpu_us=stats(cpu))


def python_profile(fn, device, iterations):
    device.synchronize()
    profile = cProfile.Profile()
    profile.enable()
    try:
        for _ in range(iterations):
            fn()
    finally:
        profile.disable()
    device.synchronize()
    stream = io.StringIO()
    summary = pstats.Stats(profile, stream=stream).sort_stats("cumulative")
    summary.print_stats(80)
    rows = []
    for (filename, line, name), (
        primitive,
        calls,
        self_time,
        cumulative,
        _,
    ) in summary.stats.items():
        rows.append(
            dict(
                filename=filename,
                line=line,
                function=name,
                primitive_calls=primitive,
                total_calls=calls,
                self_seconds=self_time,
                cumulative_seconds=cumulative,
            )
        )
    rows.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
    return dict(
        invocations=iterations,
        total_instrumented_seconds=summary.total_tt,
        top_functions=rows[:100],
        text=stream.getvalue(),
    )


def host_diagnostics(op, args, chunks, torch, device, options, baseline=None):
    if args[0].shape[0] > 512:
        return dict(
            status="not_collected_large_case",
            reason="Host investigation limited to <=512 rows",
        )
    functions = dict(
        noop=lambda: None,
        validation=lambda: op._validate_inputs(*args),
    )

    def guard():
        with device.device(args[0].device):
            pass

    functions["device_guard"] = guard
    if baseline is not None:
        functions["validation_baseline"] = lambda: baseline.__globals__[
            "_validate_inputs"
        ](*args)
    if chunks:
        logits, scratch = chunks[0]["score_args"][5], chunks[0]["topk_args"][3]

        def allocate():
            return (
                torch.empty(logits.shape, dtype=logits.dtype, device=logits.device),
                torch.empty(scratch.shape, dtype=scratch.dtype, device=scratch.device),
            )

        functions["outer_workspace_allocation"] = allocate
    for fn in functions.values():
        for _ in range(options.warmup):
            fn()
    result = dict(
        status="passed",
        iterations=options.profile_iterations,
        microbench={
            name: host_microbench(
                fn, device, options.profile_iterations, options.rounds
            )
            for name, fn in functions.items()
        },
        profiles={
            "validation": python_profile(
                functions["validation"], device, options.profile_iterations
            ),
            "public_entry": python_profile(
                lambda: op.qsa_select_paged_prefill(*args),
                device,
                options.profile_iterations,
            ),
        },
    )
    if baseline is not None:
        result["profiles"]["validation_baseline"] = python_profile(
            functions["validation_baseline"], device, options.profile_iterations
        )
        result["profiles"]["baseline_entry"] = python_profile(
            lambda: baseline(*args), device, options.profile_iterations
        )
    return result


def make_case(case, torch, helpers):
    dtype = getattr(torch, case["dtype"])
    if case["kind"] == "upstream_perf":
        return upstream_workloads_module().make_inputs(case, helpers, torch)
    if case["kind"] == "single":
        return helpers.make_inputs(
            (1,), (8192,), dtype=dtype, num_heads=case["heads"], head_dim=case["dim"]
        )
    if case["kind"] == "tail_one":
        return helpers.make_inputs((33, 32), (4096, 8192), query_base=113, dtype=dtype)
    if case["kind"] == "tail":
        return helpers.make_inputs((39, 34), (4096, 8192), query_base=113, dtype=dtype)
    if case["kind"] == "core":
        requests, length, context, page, k = case["shape"]
        return helpers.make_inputs(
            (length,) * requests,
            (context,) * requests,
            page_size=page,
            k=k,
            query_base=31,
        )
    if case["kind"] == "upstream":
        return helpers.make_upstream_prefill_inputs(
            seq_len_slack=case["slack"], dtype=dtype
        )
    if case["kind"] == "zero":
        return helpers.make_upstream_no_complete_blocks_inputs()
    if case["kind"] == "hd":
        return helpers.make_inputs(
            (3, 33),
            (2048, 4096),
            query_base=31,
            num_heads=case["heads"],
            head_dim=case["dim"],
            dtype=dtype,
        )
    return helpers.make_inputs(
        (4, 0, 5),
        (1025, 16, 1537),
        query_base=37,
        padded=True,
        visible=[0, 1, 511, 1025, 512, 513, 800, 1024, 1537],
    )


def fingerprint(args, torch):
    result = []
    for i, value in enumerate(args):
        if not isinstance(value, torch.Tensor):
            result.append(dict(argument=i, value=value))
            continue
        entry = dict(
            argument=i,
            shape=list(value.shape),
            stride=list(value.stride()),
            dtype=str(value.dtype),
        )
        # Output storage is caller supplied and may be uninitialized.
        if i != 8:
            raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            entry["sha256"] = sha256(raw)
        result.append(entry)
    return result


def observe_pipeline(op, args):
    """Record actual production calls, including chunk widths and tail sizes.

    Observation/cloning is untimed. Restore globals before any warmup/timing.
    Frozen logits are separate per chunk for isolated Top-K measurements.
    """
    chunks = []
    score, topk = op._prefill_logits, op.persistent_topk

    def observe_score(*a, **kw):
        logits = score(*a, **kw)
        chunks.append(dict(score_args=a, score_kwargs=kw, frozen=logits.clone()))
        return logits

    def observe_topk(*a, **kw):
        if not chunks or "topk_args" in chunks[-1]:
            raise RuntimeError("Production pipeline no longer alternates score/Top-K")
        chunks[-1].update(topk_args=a, topk_kwargs=kw)
        return topk(*a, **kw)

    with patch.object(op, "_prefill_logits", observe_score), patch.object(
        op, "persistent_topk", observe_topk
    ):
        op.qsa_select_paged_prefill(*args)
    if any("topk_args" not in chunk for chunk in chunks):
        raise RuntimeError("Missing Top-K call for observed score chunk")
    return chunks


def stage_functions(op, args, chunks):
    def scores():
        for chunk in chunks:
            op._prefill_logits(*chunk["score_args"], **chunk.get("score_kwargs", {}))

    def topk():
        for chunk in chunks:
            a = chunk["topk_args"]
            op.persistent_topk(chunk["frozen"], *a[1:], **chunk["topk_kwargs"])

    def pipeline():
        for chunk in chunks:
            logits = op._prefill_logits(
                *chunk["score_args"], **chunk.get("score_kwargs", {})
            )
            op.persistent_topk(logits, *chunk["topk_args"][1:], **chunk["topk_kwargs"])

    functions = dict(full=lambda: op.qsa_select_paged_prefill(*args))
    if chunks:
        functions.update(score=scores, topk=topk, pipeline_preallocated=pipeline)
    return functions


def sample_rows(count):
    return (
        sorted({i * (count - 1) // min(count - 1, 7) for i in range(min(count, 8))})
        if count > 1
        else [0]
    )


def validate_outputs(args, torch, helpers):
    """Independent FP32 reference on <=8 entire rows, outside all timings.

    This is a benchmark guard, not a substitute for the full correctness suite.
    Use the existing selection tolerances, including the FP8 sqrt(D) scale.
    """
    q, cache, table, starts, visible = args[:5]
    offsets = (starts - starts[0]).cpu().tolist()
    lengths = visible.cpu().tolist()
    rows = sample_rows(q.shape[0])
    fp8 = q.dtype == torch.float8_e4m3fn
    for row in rows:
        request = next(
            i for i in range(len(offsets) - 1) if offsets[i] <= row < offsets[i + 1]
        )
        length = lengths[row]
        columns = torch.arange(length, device=q.device)
        pages = table[request, columns // cache.shape[1]].long()
        keys = cache[pages, columns % cache.shape[1], 0].float()
        logits = (q[row].float() @ keys.T).clamp_min(0).sum(0)[None, :]
        helpers.assert_selection(
            logits,
            visible[row : row + 1],
            args[8][row : row + 1],
            rtol=1e-3 if fp8 else 2e-4,
            atol=1e-3 * q.shape[2] ** 0.5 if fp8 else 2e-3,
        )
    return rows


def validate_stage(name, fn, args, chunks, torch, helpers, device):
    if name == "score":
        # Earlier chunks intentionally share/overwrite the same production buffer.
        last = chunks[-1]
        last["score_args"][5].fill_(float("nan"))
        fn()
        device.synchronize()
        expected = last["frozen"]
        actual = last["topk_args"][0]
        visible = last["topk_args"][1]
        for row in sample_rows(actual.shape[0]):
            length = int(visible[row].item())
            torch.testing.assert_close(
                actual[row, :length], expected[row, :length], rtol=0, atol=0
            )
    else:
        args[8].fill_(-17)
        fn()
        device.synchronize()
        validate_outputs(args, torch, helpers)


def score_configs(op, kernel_name="_qsa_prefill_score_kernel"):
    # Capture the actual LibEntry cache, not Autotuner.best_config, which may
    # describe only the last cache miss rather than every observed chunk size.
    records = []
    for cache in getattr(op, kernel_name).kernel_cache:
        for entry in cache.values():
            kernel, constexprs = entry[:2]
            records.append(
                dict(
                    constexprs=constexprs,
                    metadata=str(kernel.metadata),
                    ptx_sha256=sha256(kernel.asm["ptx"].encode()),
                    cubin_sha256=sha256(kernel.asm["cubin"]),
                )
            )
    return records


def run_score_kernel(
    options, op, baseline, args, chunks, result, report, torch, device, guard
):
    from tools import qsa_prefill_score_benchmark as scoring

    result["stages"] = {}
    result["host_diagnostics"] = dict(status="not_collected_score_kernel_mode")
    result["timing_plan"] = timing_plan(options)
    if options.resume_from:
        from tools import qsa_prefill_score_resume as resume

        metadata, _ = resume.load(
            options.resume_from,
            ROOT,
            suite_cases(options),
            options,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        if resume.fingerprint(result["runtime"]) != metadata["runtime"]:
            result["performance_guard"] = dict(
                status="rejected", reasons=["Resume GPU/runtime identity changed"]
            )
            write_json(report, result)
            raise RuntimeError(
                "Resume GPU/runtime identity changed; retained measurements cannot be combined"
            )
    if not chunks:
        result.update(
            status="passed",
            score_kernel=dict(status="not_applicable_no_score", includes_topk=False),
        )
        write_json(report, result)
        return
    launchers = dict(
        baseline=baseline.__globals__["_prefill_logits"], candidate=op._prefill_logits
    )
    functions = {
        name: scoring.score_function(fn, chunks) for name, fn in launchers.items()
    }
    result["score_validation_rows"] = {
        name: scoring.validate_scores(fn, chunks, args, torch, sample_rows, device)
        for name, fn in launchers.items()
    }
    timer, timer_info = scoring.make_timer(
        options.kernel_warmup_ms, options.kernel_rep_ms
    )
    if options.resume_from:
        if resume.timer_fingerprint(timer_info) != metadata["timer"]:
            result["performance_guard"] = dict(
                status="rejected", reasons=["Resume GPU/runtime/timer identity changed"]
            )
            write_json(report, result)
            raise RuntimeError(
                "Resume GPU/runtime/timer identity changed; retained measurements cannot be combined"
            )
    result["score_timer"] = timer_info
    result["score_configs"] = score_configs(op)
    result["vector_score_configs"] = score_configs(
        op, "_qsa_prefill_single_row_score_kernel"
    )
    write_json(report, result)
    result["score_kernel"], observations, order = scoring.measure_scores(
        timer,
        functions["baseline"],
        functions["candidate"],
        options.rounds,
        lambda label: (
            guarded_telemetry(result["runtime"]["uuid"], label, result, report, guard)
            if guard
            else telemetry(result["runtime"]["uuid"], label)
        ),
        stats,
    )
    result["telemetry"], result["measurement_order"] = observations, order
    for fn in launchers.values():
        scoring.validate_scores(fn, chunks, args, torch, sample_rows, device)
    result.update(status="passed", peak_allocated_bytes=device.max_memory_allocated())
    result["runtime"]["nvidia_smi_after"] = command(["nvidia-smi"])
    write_json(report, result)


def runtime_info(torch, triton, runtime):
    device = runtime.torch_device_fn
    props = device.get_device_properties(device.current_device())
    compiler = importlib.import_module("triton.backends.nvidia.compiler")
    ptxas = compiler.get_ptxas(90)
    ptxas = Path(getattr(ptxas, "path", ptxas)).resolve()
    compiler_files = {}
    for name in (
        "triton.backends.nvidia.compiler",
        "triton.backends.nvidia.driver",
        "triton.knobs",
        "triton._C.libtriton",
    ):
        path = Path(importlib.import_module(name).__file__).resolve()
        compiler_files[name] = dict(path=str(path), sha256=sha256(path.read_bytes()))
    versions = {}
    for name in ("torch", "triton", "flagtree", "vllm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(
        python=sys.executable,
        platform=platform.platform(),
        torch_version=torch.__version__,
        torch_source=torch.__file__,
        triton_version=triton.__version__,
        triton_source=triton.__file__,
        triton_init_sha256=sha256(Path(triton.__file__).read_bytes()),
        compiler_files=compiler_files,
        ptxas=dict(
            path=str(ptxas),
            sha256=sha256(ptxas.read_bytes()),
            version=command([str(ptxas), "--version"]),
        ),
        cuda=torch.version.cuda,
        vendor=runtime.device.vendor_name,
        device=str(props),
        uuid=str(getattr(props, "uuid", "unavailable")),
        versions=versions,
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        environment={
            key: os.environ.get(key)
            for key in (
                "PYTHONPATH",
                "QSA_TRITON_RUNTIME_PATH",
                "CUDA_VISIBLE_DEVICES",
                "CUDA_LAUNCH_BLOCKING",
                "LD_LIBRARY_PATH",
                "TRITON_CACHE_DIR",
                "FLAGGEMS_CACHE_DIR",
                "TRITON_PTXAS_PATH",
                "TRITON_DISABLE_LINE_INFO",
                "TRITON_OVERRIDE_ARCH",
                "DISABLE_PTXAS_OPT",
                "PTXAS_OPTIONS",
                "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB",
            )
        },
        nvidia_smi=command(["nvidia-smi"]),
    )


def worker(options, case, report):
    # Freeze routing before any Torch/Triton/FlagGems import.
    runtime_path = os.environ.get("QSA_TRITON_RUNTIME_PATH")
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)] + (
        [runtime_path] if runtime_path else []
    )
    import torch
    import triton

    import flaggems_vllm
    from tests import qsa_select_prefill_utils as helpers

    op = importlib.import_module("flaggems_vllm.ops.qsa_select_paged_prefill")
    assert (
        Path(op.__file__).resolve()
        == ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
    )
    assert Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert torch.__version__ == importlib.metadata.version("torch")
    if runtime_path:
        assert (
            Path(triton.__file__).resolve().is_relative_to(Path(runtime_path).resolve())
        )
    else:
        assert (
            Path(triton.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        )
    topk = importlib.import_module("flaggems_vllm.ops.persistent_topk")
    assert topk.HAS_TLE and op.persistent_topk is topk.persistent_topk
    device = flaggems_vllm.runtime.torch_device_fn
    assert flaggems_vllm.vendor_name == "nvidia" and device.get_device_capability() == (
        9,
        0,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = str(case.get("budget_mb", 512))
    result = dict(
        schema=SCHEMA,
        case=case,
        status="running",
        runtime=runtime_info(torch, triton, flaggems_vllm.runtime),
    )
    if case["kind"] == "upstream_perf":
        result["workload"] = upstream_workloads_module().geometry(case)
    write_json(report, result)
    guard = gpu_guard_module() if options.require_idle_gpu else None
    if guard:
        actual_uuid = result["runtime"]["uuid"]
        if not actual_uuid.startswith("GPU-"):
            actual_uuid = f"GPU-{actual_uuid}"
        if actual_uuid != os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError(
                "Actual CUDA device differs from selected guard GPU UUID"
            )
    baseline = None
    ab = ab_configuration(options)
    if options.single_row_ab:
        baseline_module = importlib.import_module(ab[2])
        baseline = op.qsa_select_paged_prefill
        op = baseline_module.candidate_module(op)
        result["candidate_source"] = str(Path(op.__file__).resolve())
        result["score_kernel_policy"] = (
            "candidate vector for one query H<=8 D<=256; shared production matmul otherwise"
        )
    args = make_case(case, torch, helpers)
    result["inputs"] = fingerprint(args, torch)
    chunks = observe_pipeline(op, args)
    device.synchronize()
    result["validation_rows"] = validate_outputs(args, torch, helpers)
    result["chunks"] = [
        dict(
            offset=c["score_args"][-2],
            rows=c["score_args"][-1],
            width=c["score_args"][-3],
            workspace_shape=list(c["score_args"][5].shape),
            prepared_metadata="launch_metadata" in c.get("score_kwargs", {}),
            logits_is_workspace=c["topk_args"][0] is c["score_args"][5],
            score_path=(
                "vector"
                if options.single_row_ab
                and c["score_args"][-1] == 1
                and c["score_args"][0].shape[1] <= 8
                and c["score_args"][0].shape[2] <= 256
                else "matmul"
            ),
        )
        for c in chunks
    ]
    if case["kind"] == "upstream_perf":
        write_json(report, result)
        expected = result["workload"]
        if (
            [c["rows"] for c in result["chunks"]] != expected["expected_chunk_rows"]
            or any(c["width"] != expected["logits_width"] for c in result["chunks"])
            or list(args[0].shape) != expected["q_shape"]
            or list(args[1].shape) != expected["cache_shape"]
            or list(args[2].shape) != expected["page_table_shape"]
        ):
            raise RuntimeError(
                "Observed long-prefill inputs/chunks differ from workload geometry"
            )
    functions = stage_functions(op, args, chunks)
    if ab:
        ab_field, baseline_stage, module_name, kind = ab
        baseline_module = importlib.import_module(module_name)
        if baseline is None:
            baseline = baseline_module.baseline_entry(op)
        baseline(*args)
        device.synchronize()
        result["baseline_validation_rows"] = validate_outputs(args, torch, helpers)
        functions = dict(
            full=functions.pop("full"),
            **{baseline_stage: lambda: baseline(*args), **functions},
        )
        result[ab_field] = dict(
            kind=kind,
            baseline_commit=baseline_module.BASELINE_COMMIT,
            baseline_op_sha256=baseline_module.BASELINE_OP_SHA256,
            same_public_bytecode=baseline.__code__
            is op.qsa_select_paged_prefill.__code__,
            shared_score=baseline.__globals__["_prefill_logits"] is op._prefill_logits,
            shared_score_kernel=baseline.__globals__["_qsa_prefill_score_kernel"]
            is op._qsa_prefill_score_kernel,
            shared_topk=baseline.__globals__["persistent_topk"] is op.persistent_topk,
            shared_fill_kernel=baseline.__globals__["_select_all_kernel"]
            is op._select_all_kernel,
            shared_validator_bytecode=baseline.__globals__["_validate_inputs"].__code__
            is op._validate_inputs.__code__,
            shared_overlap_helper=baseline.__globals__[
                "_validate_inputs"
            ].__globals__.get("_output_overlaps_inputs")
            is op._output_overlaps_inputs,
        )
    if options.mode == "kernel":
        run_score_kernel(
            options, op, baseline, args, chunks, result, report, torch, device, guard
        )
        return
    result["stages"] = {name: {} for name in functions}
    if not chunks:
        result["stages"].update(
            {
                name: dict(status="not_executed_select_all")
                for name in ("score", "topk", "pipeline_preallocated")
            }
        )
    for fn in functions.values():
        for _ in range(options.warmup):
            fn()
    device.synchronize()
    result["score_configs"] = score_configs(op)
    if options.single_row_ab:
        result["vector_score_configs"] = score_configs(
            op, "_qsa_prefill_single_row_score_kernel"
        )
    write_json(report, result)
    # Prepare every optional graph before measurement so mode/stage order can
    # rotate within each round. Retain graph pools until all rounds complete.
    graphs = {}
    graph = None
    for name, fn in functions.items():
        stage = result["stages"][name]
        if options.no_graph:
            stage["graph_us"] = dict(status="disabled")
            continue
        # Graph support is optional. Capture failures are explicit, and a poisoned
        # context or failed replay/validation still fails the entire case.
        graph = None
        try:
            stream = device.Stream()
            stream.wait_stream(device.current_stream())
            with device.stream(stream):
                for _ in range(options.warmup):
                    fn()
            device.synchronize()
            graph = device.CUDAGraph()
            with device.graph(graph, stream=stream):
                for _ in range(options.graph_batch):
                    fn()
        except RuntimeError as exc:
            stage["graph_us"] = dict(status="capture_unavailable", reason=str(exc))
            device.synchronize()
            validate_stage(name, fn, args, chunks, torch, helpers, device)
        else:
            validate_stage(name, graph.replay, args, chunks, torch, helpers, device)
            graphs[name] = graph
            stage["graph_us"] = dict(status="captured_and_validated")
        write_json(report, result)
    graph = None
    result["timing_plan"] = timing_plan(options)
    try:
        samples, observations, order = measure_rounds(
            functions,
            graphs,
            device,
            options,
            lambda label: (
                guarded_telemetry(
                    result["runtime"]["uuid"], label, result, report, guard
                )
                if guard
                else telemetry(result["runtime"]["uuid"], label)
            ),
        )
        result["telemetry"] = observations
        result["measurement_order"] = order
        for name, metrics in samples.items():
            for method, values in metrics.items():
                if values:
                    result["stages"][name][method] = dict(
                        status="passed", **stats(values)
                    )
        write_json(report, result)
        for name, fn in functions.items():
            validate_stage(name, fn, args, chunks, torch, helpers, device)
            if name in graphs:
                validate_stage(
                    name, graphs[name].replay, args, chunks, torch, helpers, device
                )
    finally:
        graphs.clear()
    result["host_diagnostics"] = (
        host_diagnostics(op, args, chunks, torch, device, options, baseline)
        if options.host_profile
        else dict(status="not_requested")
    )
    if options.host_profile:
        # Profiles execute the real entry after all formal timings and must
        # retain the same output contract.
        validate_outputs(args, torch, helpers)
        profiles = result["host_diagnostics"].get("profiles", {})
        if profiles:
            (report.parent / f"{case['name']}.host-profile.txt").write_text(
                "Instrumentation changes latency; these are CPU diagnostics.\n\n"
                + "\n\n".join(
                    f"{name}\n{entry['text']}" for name, entry in profiles.items()
                )
            )
    result.update(status="passed", peak_allocated_bytes=device.max_memory_allocated())
    if baseline is not None:
        result[ab_field]["speedup"] = {}
        for method in METHODS:
            old = result["stages"][baseline_stage].get(method, {})
            new = result["stages"]["full"].get(method, {})
            if "samples" in old and "samples" in new:
                result[ab_field]["speedup"][method] = dict(
                    ratio_of_medians=old["median"] / new["median"],
                    paired_round_ratios=stats(
                        [a / b for a, b in zip(old["samples"], new["samples"])]
                    ),
                )
    result["runtime"]["nvidia_smi_after"] = command(["nvidia-smi"])
    write_json(report, result)


def parent(options, selected, directory):
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Reports: {directory}", flush=True)
    env = os.environ.copy()
    runtime = env.get("QSA_TRITON_RUNTIME_PATH")
    if runtime and (
        not Path(runtime).is_absolute() or not (Path(runtime) / "triton").is_dir()
    ):
        raise ValueError("QSA_TRITON_RUNTIME_PATH must be absolute and contain triton/")
    if env.get("CUDA_LAUNCH_BLOCKING", "0") != "0":
        raise ValueError("Unset CUDA_LAUNCH_BLOCKING before performance measurement")
    env.update(
        PYTHONPATH=os.pathsep.join(
            [str(ROOT / "src"), str(ROOT)] + ([runtime] if runtime else [])
        ),
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        DNN_VENDOR="nvidia",
        VLLM_PLUGINS="",
    )
    env.pop("PYTHONHOME", None)
    env.pop("QSA_NATIVE_REFERENCE_PYTHON", None)
    # Force the report-local SQLite caches rather than a shared tuning DB.
    env.pop("FLAGGEMS_DB_URL", None)
    sources = {
        path: dict(
            sha256=sha256((ROOT / path).read_bytes()), text=(ROOT / path).read_text()
        )
        for path in SOURCES
    }
    write_json(directory / "sources.json", sources)
    manifest = dict(
        schema=SCHEMA,
        options=vars(options),
        cases=selected,
        methods=(
            dict(
                score_kernel_us="benchmark.base.Benchmark.get_latency, BenchMode.KERNEL; score only"
            )
            if options.mode == "kernel"
            else METHODS
        ),
        profiler_tools=dict(ncu=shutil.which("ncu"), nsys=shutil.which("nsys")),
        timing_plan=timing_plan(options),
        git_head=command(["git", "rev-parse", "HEAD"]),
        git_status=command(["git", "status", "--porcelain"]),
        source_hashes={path: value["sha256"] for path, value in sources.items()},
        policy=(
            "Production score self baseline only; no native-vLLM speedup. "
            "Top-K excluded from both timed callables. Shared kernel-mode do_bench defaults, including cache policy."
            if options.mode == "kernel"
            else "Production self baseline only; no native-vLLM speedup. "
            "Stage times are not additive. Warm repeated inputs, no cache flush."
        ),
    )
    if options.suite == "upstream-prefill":
        module = upstream_workloads_module()
        manifest["workloads"] = {
            case["name"]: module.geometry(case) for case in selected
        }
        rows = [
            "case\trequests\tqueries_per_request\tprefix_tokens\tsequence_tokens\t"
            "compressed_context\tpage_size\tlogits_width\tchunk_rows"
        ]
        for case in selected:
            info = manifest["workloads"][case["name"]]
            rows.append(
                "\t".join(
                    map(
                        str,
                        [
                            case["name"],
                            case["requests"],
                            case["query_length"],
                            case["prefix_tokens"],
                            info["sequence_lengths"][0],
                            info["compressed_lengths"][0],
                            case["page_size"],
                            info["logits_width"],
                            ",".join(map(str, info["expected_chunk_rows"])),
                        ],
                    )
                )
            )
        (directory / "upstream-prefill-shapes.tsv").write_text("\n".join(rows) + "\n")
    write_json(directory / "manifest.json", manifest)
    summary = dict(schema=SCHEMA, manifest=manifest, cases=[])
    if options.mode == "kernel":
        manifest["active_set"] = selected
        summary["score_case_progress"] = dict(
            completed=0, active=len(selected), pending=len(selected)
        )
        write_json(directory / "manifest.json", manifest)
    if options.resume_from:
        try:
            module = score_resume_module()
            metadata, retained = module.load(
                options.resume_from,
                ROOT,
                suite_cases(options),
                options,
                env.get("CUDA_VISIBLE_DEVICES"),
            )
        except (ValueError, KeyError, TypeError, OSError) as exc:
            summary["resume_error"] = str(exc)
            write_json(directory / "stage-summary.json", summary)
            print(f"Resume rejected: {exc}", flush=True)
            return 2
        summary["retained_cases"] = retained
        summary["retained_accuracy"] = metadata["prior_accuracy"]
        summary["score_case_progress"] = dict(
            completed=len(retained),
            active=len(suite_cases(options)),
            pending=len(suite_cases(options)) - len(retained),
        )
        manifest["resume"] = metadata
        selected = [
            case for case in selected if case["name"] not in metadata["completed_names"]
        ]
        manifest["cases"] = selected
        manifest["active_set"] = suite_cases(options)
        write_json(directory / "manifest.json", manifest)
        print(
            f"Retained {len(retained)} completed cases; measuring {len(selected)} pending cases",
            flush=True,
        )
    guard = gpu_guard_module() if options.require_idle_gpu else None
    if guard and not idle_gpu_gate(summary, directory, env, "before_accuracy", guard):
        return 2
    if options.skip_accuracy:
        summary["accuracy"] = dict(status="not_run_explicit_request")
    else:
        print(
            "Running full independent correctness suite (native excluded)", flush=True
        )
        accuracy_env = env.copy()
        accuracy_env.update(
            TRITON_CACHE_DIR=str(directory / "cache" / "accuracy" / "triton"),
            FLAGGEMS_CACHE_DIR=str(directory / "cache" / "accuracy" / "gems"),
            QSA_REQUIRE_VLLM_REFERENCE="0",
        )
        with (directory / "accuracy-full.txt").open("w") as log:
            try:
                accuracy_tests = ["tests/test_qsa_select_paged_prefill.py"]
                if options.single_row_ab:
                    accuracy_tests.append("tests/test_qsa_prefill_single_row_trial.py")
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "-ra",
                        "-x",
                        *accuracy_tests,
                        "-k",
                        "not vllm",
                        "--record",
                        "json",
                        "--output",
                        str(directory / "accuracy-full.json"),
                    ],
                    cwd=ROOT,
                    env=accuracy_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=options.timeout,
                )
                error = (
                    None if proc.returncode == 0 else f"pytest exit {proc.returncode}"
                )
            except subprocess.TimeoutExpired:
                error = f"accuracy timeout after {options.timeout}s"
        counts = {}
        if not error:
            counts, error = accuracy_results(directory / "accuracy-full.json")
        summary["accuracy"] = dict(
            status="failed" if error else "passed", counts=counts, error=error
        )
        write_json(directory / "stage-summary.json", summary)
        if error:
            print(f"Accuracy gate failed: {error}; no timings collected", flush=True)
            return 1
    for case in selected:
        name = case["name"]
        if guard and not idle_gpu_gate(
            summary, directory, env, f"before_{name}", guard
        ):
            summary["stopped_before_case"] = name
            write_json(directory / "stage-summary.json", summary)
            break
        output = directory / f"{name}.json"
        case_env = env.copy()
        case_env.update(
            TRITON_CACHE_DIR=str(directory / "cache" / name / "triton"),
            FLAGGEMS_CACHE_DIR=str(directory / "cache" / name / "gems"),
        )
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            name,
            "--report-dir",
            str(directory),
            "--warmup",
            str(options.warmup),
            "--iterations",
            str(options.iterations),
            "--rounds",
            str(options.rounds),
            "--graph-batch",
            str(options.graph_batch),
            "--mode",
            options.mode,
            "--suite",
            options.suite,
            "--kernel-warmup-ms",
            str(options.kernel_warmup_ms),
            "--kernel-rep-ms",
            str(options.kernel_rep_ms),
        ]
        if options.no_graph:
            cmd.append("--no-graph")
        if options.host_profile:
            cmd += [
                "--host-profile",
                "--profile-iterations",
                str(options.profile_iterations),
            ]
        if options.validation_ab:
            cmd += [
                "--validation-ab",
                "--validation-baseline",
                options.validation_baseline,
            ]
        if options.host_ab:
            cmd.append("--host-ab")
        if options.single_row_ab:
            cmd.append("--single-row-ab")
        if options.require_idle_gpu:
            cmd.append("--require-idle-gpu")
        if options.resume_from:
            cmd.extend(["--resume-from", str(Path(options.resume_from).resolve())])
        print(f"Running {name}", flush=True)
        with (directory / f"{name}.txt").open("w") as log:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    env=case_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=options.timeout,
                )
                error = (
                    None if proc.returncode == 0 else f"worker exit {proc.returncode}"
                )
            except subprocess.TimeoutExpired:
                error = f"worker timeout after {options.timeout}s"
        result = json.loads(output.read_text()) if output.exists() else dict(case=case)
        if error or result.get("status") != "passed":
            result.update(status="failed", error=error or "Worker did not complete")
        summary["cases"].append(result)
        write_json(directory / "stage-summary.json", summary)
        print(f"{name}: {result['status']}", flush=True)
        if result.get("performance_guard", {}).get("status") == "rejected":
            summary["environment_rejected"] = True
            write_json(directory / "stage-summary.json", summary)
            print(
                "Stopping remaining cases after GPU environment rejection", flush=True
            )
            break
    if options.mode == "kernel":
        complete = len(summary.get("retained_cases", [])) + sum(
            c["status"] == "passed" for c in summary["cases"]
        )
        active = len(manifest["active_set"])
        summary["score_case_progress"] = dict(
            completed=complete,
            active=active,
            pending=active - complete,
        )
        write_json(directory / "stage-summary.json", summary)
    lines = [
        "case\tcase_status\tstage\tstage_status\twall_median_us\t"
        "event_median_us\tgraph_median_us\tgraph_status"
    ]
    for result in summary["cases"]:
        for name, stage in result.get("stages", {}).items():
            values = [
                str(stage.get(method, {}).get("median", "NA")) for method in METHODS
            ]
            lines.append(
                "\t".join(
                    [
                        result["case"]["name"],
                        result["status"],
                        name,
                        stage.get(
                            "status",
                            (
                                "passed"
                                if result["status"] == "passed"
                                else "partial_or_failed"
                            ),
                        ),
                        *values,
                        stage.get("graph_us", {}).get("status", "NA"),
                    ]
                )
            )
    (directory / "stage-summary.tsv").write_text("\n".join(lines) + "\n")
    if options.host_profile:
        host_lines = ["case\tstatus\tprobe\twall_median_us\tprocess_cpu_median_us"]
        for case in summary["cases"]:
            info = case.get("host_diagnostics", {})
            for name, values in info.get("microbench", {}).items():
                host_lines.append(
                    "\t".join(
                        [
                            case["case"]["name"],
                            info["status"],
                            name,
                            str(values["wall_us"]["median"]),
                            str(values["process_cpu_us"]["median"]),
                        ]
                    )
                )
        (directory / "host-summary.tsv").write_text("\n".join(host_lines) + "\n")
    ab = ab_configuration(options)
    if options.mode == "kernel":
        rows = [
            "case\tstatus\tbaseline_us\tcandidate_us\tspeedup\tpaired_ratio_median\tmeasurement_origin"
        ]
        for case in summary.get("retained_cases", []) + summary["cases"]:
            info = case.get("score_kernel", {})
            if case["status"] != "passed":
                info = dict(status=case["status"])
            rows.append(
                "\t".join(
                    map(
                        str,
                        [
                            case["case"]["name"],
                            info.get("status", case["status"]),
                            info.get("baseline_us", {}).get("median", "NA"),
                            info.get("candidate_us", {}).get("median", "NA"),
                            info.get("speedup", "NA"),
                            info.get("paired_ratios", {}).get("median", "NA"),
                            "retained" if "measurement_origin" in case else "current",
                        ],
                    )
                )
            )
        (directory / "score-kernel-ab.tsv").write_text("\n".join(rows) + "\n")
    elif ab:
        ab_field, baseline_stage, _, _ = ab
        rows = [
            "case\tstatus\tmetric\tbaseline_median_us\tcandidate_median_us\tratio_of_medians\tpaired_ratio_median"
        ]
        for case in summary["cases"]:
            for method, comparison in case.get(ab_field, {}).get("speedup", {}).items():
                old = case["stages"][baseline_stage][method]["median"]
                new = case["stages"]["full"][method]["median"]
                rows.append(
                    "\t".join(
                        map(
                            str,
                            [
                                case["case"]["name"],
                                case["status"],
                                method,
                                old,
                                new,
                                comparison["ratio_of_medians"],
                                comparison["paired_round_ratios"]["median"],
                            ],
                        )
                    )
                )
        filename = (
            "single-row-ab.tsv"
            if options.single_row_ab
            else "host-ab.tsv" if options.host_ab else "validation-ab.tsv"
        )
        (directory / filename).write_text("\n".join(rows) + "\n")
    print(f"Return top-level JSON/TXT/TSV files from: {directory}", flush=True)
    return (
        2
        if summary.get("environment_rejected")
        else int(any(case["status"] != "passed" for case in summary["cases"]))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("core", "comprehensive", "upstream-prefill"),
        default="upstream-prefill",
        help="Default: the published twelve layouts; core/comprehensive are legacy diagnostics",
    )
    parser.add_argument(
        "--cases", nargs="+", help="Explicit case names; overrides suite"
    )
    parser.add_argument(
        "--mode",
        choices=("stages", "kernel"),
        default="stages",
        help="kernel: score-only shared benchmark timer, Top-K excluded",
    )
    parser.add_argument("--kernel-warmup-ms", type=float, default=1000)
    parser.add_argument("--kernel-rep-ms", type=float, default=100)
    parser.add_argument(
        "--resume-from",
        help="Retain completed guarded kernel cases from this stage-summary.json on the same GPU/runtime",
    )
    parser.add_argument(
        "--require-idle-gpu",
        action="store_true",
        help="Require one full GPU UUID; reject contention/thermal slowdown and save diagnostics",
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--report-dir")
    parser.add_argument(
        "--warmup", type=int, default=25, help="Untimed calls per stage after JIT"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Minimum stage calls per round; rounded up to a graph-batch multiple for every mode",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=7,
        help="Independent batches, not per-call percentiles",
    )
    parser.add_argument(
        "--graph-batch", type=int, default=8, help="Calls captured in each graph"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Seconds per isolated case, including JIT",
    )
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument(
        "--host-profile",
        action="store_true",
        help="Untimed Python CPU diagnostics after performance measurements",
    )
    parser.add_argument("--profile-iterations", type=int, default=200)
    ab_group = parser.add_mutually_exclusive_group()
    ab_group.add_argument(
        "--validation-ab",
        action="store_true",
        help="Compare frozen old validation against production with shared GPU launchers",
    )
    ab_group.add_argument(
        "--host-ab",
        action="store_true",
        help="Compare prepared chunk setup against pinned old host entry/launcher with shared GPU kernels",
    )
    ab_group.add_argument(
        "--single-row-ab",
        action="store_true",
        help="Experimental one-query vector scorer vs unchanged production in the same compiler",
    )
    parser.add_argument(
        "--validation-baseline",
        choices=("original", "overlap-retained"),
        default="original",
        help="Frozen validation baseline; overlap-retained isolates the stride trial",
    )
    parser.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="Explicit smoke/diagnosis only; recorded in manifest",
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.resume_from and (
        options.mode != "kernel"
        or not options.single_row_ab
        or not options.require_idle_gpu
        or options.cases
        or options.suite not in ("comprehensive", "upstream-prefill")
    ):
        parser.error(
            "--resume-from requires comprehensive or upstream-prefill "
            "--single-row-ab --mode kernel --require-idle-gpu, without --cases"
        )
    if options.mode == "kernel" and not options.single_row_ab:
        parser.error("Score-only --mode kernel currently requires --single-row-ab")
    if (
        options.host_ab or options.single_row_ab
    ) and options.validation_baseline != "original":
        parser.error("--validation-baseline requires --validation-ab")
    all_cases = suite_cases(options)
    if options.list_cases:
        print(json.dumps(all_cases, indent=2))
        return 0
    if any(
        getattr(options, name) <= 0
        for name in (
            "warmup",
            "iterations",
            "rounds",
            "graph_batch",
            "timeout",
            "profile_iterations",
            "kernel_warmup_ms",
            "kernel_rep_ms",
        )
    ):
        parser.error(
            "Warmup, iteration counts, rounds, graph batch and timeout must be positive"
        )
    names = {case["name"] for case in all_cases}
    if options.cases and (
        set(options.cases) - names or len(options.cases) != len(set(options.cases))
    ):
        parser.error("Unknown or duplicate case names; use --list-cases")
    if options.worker:
        if options.worker not in names or not options.report_dir:
            parser.error("Worker requires a known case and report directory")
        case = next(case for case in all_cases if case["name"] == options.worker)
        report = Path(options.report_dir) / f"{options.worker}.json"
        try:
            worker(options, case, report)
        except Exception:
            failure = traceback.format_exc()
            print(failure, flush=True)
            result = (
                json.loads(report.read_text()) if report.exists() else dict(case=case)
            )
            result.update(status="failed", traceback=failure)
            write_json(report, result)
            return 1
        return 0
    selected = [
        case
        for case in all_cases
        if (
            case["name"] in options.cases
            if options.cases
            else options.suite == "comprehensive" or not case.get("extended")
        )
    ]
    directory = (
        Path(options.report_dir).resolve()
        if options.report_dir
        else Path(tempfile.mkdtemp(prefix="qsa-stage-h800."))
    )
    if directory.exists() and any(directory.iterdir()):
        parser.error(
            "Report directory must be empty; existing reports are never overwritten"
        )
    return parent(options, selected, directory)


if __name__ == "__main__":
    sys.exit(main())
