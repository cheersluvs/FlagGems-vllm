# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Focused QSA timing and separate, bounded Nsight Compute score capture."""

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
CASES = (
    "perf_n1_q16384_p102400_bfloat16",
    "fp8_long100k_float8_e4m3fn",
    "fp8_tail112_float8_e4m3fn",
    "observed_short96_bfloat16",
    "observed_short_batch3_bfloat16",
)
SECTIONS = (
    "LaunchStats",
    "Occupancy",
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "ComputeWorkloadAnalysis",
    "SchedulerStats",
)


def score_pairs(functions):
    return {
        "native": functions["score"]["baseline"],
        "production": functions["score_vs_production"]["baseline"],
        "candidate": functions["score"]["candidate"],
    }


def self_controls(options, result, path, functions, timer, monitor):
    from tools import benchmark_qsa_prefill_stages as support
    from tools import qsa_prefill_score_benchmark as scoring

    records = result.setdefault("diagnostics", {}).setdefault("self_controls", {})
    for stage in ("score", "pipeline"):
        functions_for_stage = {
            "native": functions[stage]["baseline"],
            "production": functions[stage + "_vs_production"]["baseline"],
            "candidate": functions[stage]["candidate"],
        }
        for side, fn in functions_for_stage.items():
            name = side + ":" + stage
            measured, telemetry, order = scoring.measure_scores(
                timer,
                fn,
                fn,
                options.rounds,
                lambda label: monitor("AA:" + name + ":" + label),
                support.stats,
            )
            records[name] = dict(
                measured,
                telemetry=telemetry,
                measurement_order=order,
                policy="identical callable and buffers; descriptive noise control only",
            )
            support.write_json(path, result)


def configuration_signature(record):
    """Actual per-chunk dispatch/binary identity, excluding addresses/cache sizes."""
    traces = record["cache_granularity"]
    result = {
        side: [
            {
                **{
                    key: t[key]
                    for key in ("offset", "rows", "width", "route", "constants")
                },
                "cubin_sha256": t["compiled"]["cubin_sha256"],
            }
            for t in traces[side]
        ]
        for side in ("production", "production_baseline")
    }
    native = record.get("kernel_resources", {}).get("native", {})
    if native.get("status") != "collected":
        raise RuntimeError(
            "Native binary identity unavailable for profile/timing match"
        )
    result["native_cubins"] = sorted(k["cubin_sha256"] for k in native["kernels"])
    return result


def capture(options, result, path, functions, validate, torch, device, monitor):
    """Run unmodified scorers with real stores, after all JIT/autotune/validation."""
    from tools import benchmark_qsa_prefill_stages as support
    from tools import benchmark_qsa_prefill_triton_stages as bench

    measured = json.loads(
        (Path(options.report_dir) / (options.worker + ".json")).read_text()
    )
    if measured.get("status") != "passed":
        raise RuntimeError("Profiling requires a valid independent timing record")
    if (
        configuration_signature(measured) != configuration_signature(result)
        or measured["inputs"] != result["inputs"]
        or bench.runtime_identity(measured["runtime"])
        != bench.runtime_identity(result["runtime"])
    ):
        raise RuntimeError(
            "Profile/timing inputs, actual configs, binary or runtime differ"
        )
    names = (
        "score",
        "score_vs_production",
        "topk",
        "pipeline",
        "pipeline_vs_production",
    )
    record = result["profiling"] = dict(
        status="validating",
        acceptance_eligible=False,
        kind="hardware_counters_only_not_benchmark",
        config_and_input_match=True,
        validation_before={name: validate(name) for name in names},
        captures=[],
        chunks=result["chunks"],
        expected_profiled_launches=3 * len(result["chunks"]),
        policy="all actual chunks once per side; no TopK in capture; real stores retained",
    )
    support.write_json(path, result)
    # Prime all paths before collection. These launches do not supply latency data.
    for fn in score_pairs(functions).values():
        fn()
    device.synchronize()
    monitor("before_score_capture")
    profiler = torch.cuda.cudart()
    started = False
    try:
        rc = profiler.cudaProfilerStart()
        if int(rc) != 0:
            raise RuntimeError("cudaProfilerStart failed: " + str(rc))
        started = True
        for side, fn in score_pairs(functions).items():
            label = "qsa_score/" + side
            torch.cuda.nvtx.range_push(label)
            try:
                fn()
                device.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
            record["captures"].append(dict(side=side, nvtx_range=label))
    finally:
        if started:
            rc = profiler.cudaProfilerStop()
            if int(rc) != 0:
                raise RuntimeError("cudaProfilerStop failed: " + str(rc))
    monitor("after_score_capture")
    record["validation_after"] = {name: validate(name) for name in names}
    record["status"] = "captured_pending_counter_export"
    support.write_json(path, result)


def run_process(command, *, timeout, env=None, log=None):
    """Bound a private process group, including profiler-launched descendants."""
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT if log else None,
        start_new_session=True,
    )
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return 124


def ncu_command(ncu, csv_path, worker):
    return [
        ncu,
        "--profile-from-start",
        "off",
        "--replay-mode",
        "kernel",
        "--cache-control",
        "none",
        "--clock-control",
        "none",
        "--nvtx",
        "--csv",
        "--page",
        "raw",
        "--log-file",
        str(csv_path),
        *[arg for section in SECTIONS for arg in ("--section", section)],
        *worker,
    ]


def exported_metrics(path):
    """Require actual metric rows; an error-only CSV is not a counter result."""
    lines = Path(path).read_text(errors="replace").splitlines()
    for index, line in enumerate(lines):
        header = next(csv.reader([line]), [])
        if "Metric Name" not in header or "Metric Value" not in header:
            continue
        rows = list(csv.DictReader(lines[index:]))
        valid = [
            row
            for row in rows
            if row.get("Metric Name")
            and row.get("Metric Value") not in (None, "", "n/a")
        ]
        return dict(
            metric_rows=len(valid),
            kernel_ids=sorted({row.get("ID", "") for row in valid}),
        )
    return dict(metric_rows=0, kernel_ids=[])


def engine_command(options):
    return [
        sys.executable,
        "-I",
        str(ROOT / "tools/benchmark_qsa_prefill_triton_stages.py"),
        "--compiler",
        "flagtree",
        "--flagtree",
        options.flagtree,
        "--candidate",
        "query-stage-joint",
        "--compare-previous-tuning",
        "--previous-tuning",
        "combined",
        "--suite",
        "scored-prefill",
        "--accuracy-profile",
        "iteration",
        "--rounds",
        "2",
        "--kernel-warmup-ms",
        "200",
        "--kernel-rep-ms",
        "100",
        "--timeout",
        str(options.timeout),
        "--max-initial-temp-c",
        "80",
        "--report-dir",
        options.report_dir,
    ]


def find_ncu(explicit=None):
    """Bounded discovery for CUDA/Nsight container layouts; never install tools."""
    if explicit:
        return explicit
    found = shutil.which("ncu")
    if found:
        return found
    for parent, patterns in (
        (
            Path("/usr/local"),
            (
                "cuda*/bin/ncu",
                "cuda*/nsight-compute*/ncu",
                "NVIDIA-Nsight-Compute*/ncu",
            ),
        ),
        (
            Path("/opt"),
            (
                "nvidia/nsight-compute/*/ncu",
                "nvidia/nsight-compute/ncu",
                "nsight-compute*/ncu",
            ),
        ),
    ):
        for pattern in patterns:
            candidates = sorted(parent.glob(pattern), reverse=True)
            for path in candidates:
                if path.is_file() and os.access(path, os.X_OK):
                    return str(path.resolve())
    return None


def collect_counters(options, facts, command, *, cache_report=None):
    from tools import benchmark_qsa_prefill_stages as support
    from tools import benchmark_qsa_prefill_triton_stages as bench

    report = Path(options.report_dir)
    output = report / "profiling-summary.json"
    facts["ncu"] = find_ncu(options.ncu)
    facts["profiles"] = []
    if not facts["ncu"]:
        facts.update(status="profiler_unavailable", hardware_counters_complete=False)
        print("未找到 ncu；独立计时与 A/A 已保存，硬件计数器尚未采集。", flush=True)
        return 2
    facts["ncu_version"] = support.command([facts["ncu"], "--version"])
    facts["ncu_sections"] = support.command([facts["ncu"], "--list-sections"])
    if any(
        facts[key].get("returncode") != 0 for key in ("ncu_version", "ncu_sections")
    ):
        raise RuntimeError("Nsight Compute executable/section preflight failed")
    if any(section not in facts["ncu_sections"]["stdout"] for section in SECTIONS):
        raise RuntimeError("Requested Nsight Compute sections unavailable")
    facts["capture_policy"] = dict(
        sections=SECTIONS,
        replay_mode="kernel",
        cache_control="none",
        clock_control="none",
        profile_from_start=False,
    )
    reference = str((cache_report or report) / "qsa_indexer_vllm.py")
    for name in options.cases:
        check = support.gpu_guard_module().idle(
            os.environ["CUDA_VISIBLE_DEVICES"],
            support.command,
            max_initial_temp_c=80,
        )
        facts.setdefault("profile_admission", []).append(check)
        if check["status"] != "passed":
            raise RuntimeError("GPU guard rejected profiling: " + repr(check))
        csv_path = report / ("profile-" + name + ".csv")
        worker = [
            *command,
            "--reference-file",
            reference,
            "--profile-capture",
            "--worker",
            name,
        ]
        cmd = ncu_command(facts["ncu"], csv_path, worker)
        record = dict(case=name, command=cmd, status="running")
        facts["profiles"].append(record)
        support.write_json(output, facts)
        with (report / ("profile-" + name + ".txt")).open("w") as log:
            rc = run_process(
                cmd,
                timeout=options.timeout,
                env=bench.child_environment(cache_report or report, name),
                log=log,
            )
        record["exit_code"] = rc
        capture_path = report / ("profile-" + name + ".json")
        captured = (
            json.loads(capture_path.read_text()) if capture_path.is_file() else {}
        )
        record["worker_status"] = captured.get("status")
        record["counters"] = exported_metrics(csv_path) if csv_path.is_file() else {}
        expected = captured.get("profiling", {}).get("expected_profiled_launches")
        record["expected_profiled_launches"] = expected
        record["status"] = (
            "passed"
            if rc == 0
            and captured.get("status") == "passed"
            and record["counters"].get("metric_rows", 0) > 0
            and len(record["counters"].get("kernel_ids", [])) == expected
            else "failed"
        )
        print("Profile", name, record["status"], flush=True)
        if record["status"] != "passed":
            # Permission/toolchain failures normally affect every case.
            # Preserve the first full error, do not repeat it five times.
            raise RuntimeError("Counter capture failed; see " + str(csv_path))
    facts.update(status="passed", hardware_counters_complete=True)
    return 0


def coordinate(options):
    from tools import benchmark_qsa_prefill_stages as support
    from tools import qsa_prefill_flagtree_ab as profile
    from tools import validate_qsa_ptxas_h800 as validation

    report = Path(options.report_dir)
    facts = dict(
        status="preparing",
        acceptance_eligible=False,
        hardware_counters_complete=False,
        cases=options.cases,
    )
    output = report / "profiling-summary.json"
    try:
        if not profile.select_gpu(facts):
            raise RuntimeError("No exclusive idle H800 available")
        import tempfile

        tools = validation.toolchain(
            Path(tempfile.mkdtemp(prefix="qsa-profile-tools."))
        )
        if tools["ptxas"]["sha256"] != profile.PTXAS_SHA256:
            raise RuntimeError("Unexpected ptxas binary")
        os.environ["TRITON_PTXAS_PATH"] = tools["ptxas"]["path"]
        support.write_json(report / "flagtree134-tools.json", dict(toolchain=tools))
        command = engine_command(options)
        # The benchmark process has no profiler attached. It retains all stage
        # numeric guards and adds A/A with unchanged original timers.
        rc = run_process(
            [
                *command,
                "--skip-accuracy",
                "--diagnostic-self-control",
                "--cases",
                *options.cases,
            ],
            timeout=options.timeout * (len(options.cases) + 1),
        )
        facts.update(benchmark_exit_code=rc, status="benchmark_completed")
        if rc:
            raise RuntimeError("Independent focused benchmark failed: " + str(rc))
        summary = json.loads((report / "stage-summary.json").read_text())
        if summary.get("status") != "passed":
            raise RuntimeError("Independent benchmark incomplete")
        with (report / "diagnostic-timings.tsv").open("w") as handle:
            handle.write("case\tstage\tbaseline_us\tcandidate_us\tspeedup\n")
            for record in summary["cases"]:
                for stage in (
                    "score",
                    "topk",
                    "pipeline",
                    "score_vs_production",
                    "pipeline_vs_production",
                ):
                    row = record["stages"][stage]
                    handle.write(
                        f"{record['case']['name']}\t{stage}\t{row['baseline_us']['median']}\t"
                        f"{row['candidate_us']['median']}\t{row['speedup']}\n"
                    )
        return collect_counters(options, facts, command)
    except BaseException:
        facts.update(status="failed", error=traceback.format_exc())
        print(facts["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        support.write_json(output, facts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--flagtree", default="/opt/flagtree")
    parser.add_argument("--ncu", help="Explicit Nsight Compute CLI executable")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    options = parser.parse_args()
    if options.timeout <= 0 or len(set(options.cases)) != len(options.cases):
        parser.error("Positive timeout and unique cases required")
    # Called through with_cuda_compat134 in a fresh process, before any CUDA import.
    return coordinate(options)


if __name__ == "__main__":
    sys.exit(main())
