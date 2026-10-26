# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Replay captured TTIR for pass localization; no GPU kernel execution."""

import argparse
import hashlib
import importlib
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = "63ae460e1abcd1b923ef9a939bea60b01125ac95"
PRODUCTION = "0d7e80e240966565a7c78151c8344f17e0627e73920f5229aadcc0f0fb42f10e"
CASES = ("perf_n1_q8192_p102400_bfloat16", "perf_n4_q4096_p102400_bfloat16")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def cache_input(cache, expected_ptx):
    matches = sorted(p for p in Path(cache).rglob("*.ptx") if sha(p) == expected_ptx)
    for ptx in matches:
        ir, metadata = ptx.with_suffix(".ttir"), ptx.with_suffix(".json")
        if ir.is_file() and metadata.is_file():
            return ir, metadata
    raise RuntimeError(
        "Original PTX-matched TTIR/metadata cache missing: " + str(cache)
    )


def sections(text):
    headers = list(
        re.finditer(
            r"^//.*(?:IR Dump (?:Before|After)|internal IR Dump After):?.*$", text, re.M
        )
    )
    result = []
    for i, m in enumerate(headers):
        label = m[0].strip()
        body = text[
            m.end() : headers[i + 1].start() if i + 1 < len(headers) else len(text)
        ]
        if not re.search(r"latenc|schedul|pipelin|LowerLoops|ExpandLoops", label, re.I):
            continue
        events = []
        for number, line in enumerate(body.splitlines(), 1):
            if re.search(
                r"tt\.load|async_copy_global_to_local|warp_group_dot|tt\.dot\b|"
                r"tt\.store|scf\.for|loop.stage|loop.cluster|tt.latency",
                line,
            ):
                events.append(dict(line=number, text=line.strip()))
        result.append(dict(pass_header=label, events=events))
    return result


def coverage(events):
    labels = "\n".join(v["pass_header"] for v in events)
    return {
        name: bool(re.search(pattern, labels, re.I))
        for name, pattern in (
            ("latencies", "AssignLatencies|assign-latencies"),
            ("schedule", "ScheduleLoops|schedule-loops"),
            ("lower_loads", "LowerLoops"),
            ("expand", "ExpandLoops"),
        )
    }


def prepare(source, report):
    summary = json.loads((source / "async-pipeline-summary.json").read_text())
    if (
        summary["git_head"]["stdout"].strip() != CODE
        or summary["status"] != "not_promising"
    ):
        raise RuntimeError("Requires the captured stages3 negative probe at " + CODE)
    if (
        summary["source"]["production_sha256"] != PRODUCTION
        or sha(ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py") != PRODUCTION
    ):
        raise RuntimeError("Production source changed")
    jobs = []
    for case in CASES:
        result = json.loads((source / (case + ".json")).read_text())
        if result["source"] != summary["source"] or result["status"] != "not_promising":
            raise RuntimeError("Captured case identity changed")
        snapshots = [
            json.loads(p.read_text())
            for p in source.glob("compile-" + case + "-*.json")
        ]
        snapshots = [v for v in snapshots if not v["boundary"]]
        if len(snapshots) != 1:
            raise RuntimeError("Requires one main compile snapshot per case")
        for side in ("baseline", "candidate"):
            record = snapshots[0]["records"][side][0]
            ir, metadata = cache_input(
                result["runtime"]["environment"]["TRITON_CACHE_DIR"],
                record["ptx_sha256"],
            )
            job_dir = report / (case + "-" + side)
            job_dir.mkdir()
            for p in (ir, metadata, ir.with_suffix(".ptx")):
                (job_dir / p.name).write_bytes(p.read_bytes())
            job = dict(
                name=job_dir.name,
                directory=str(job_dir),
                ir=str(job_dir / ir.name),
                metadata=str(job_dir / metadata.name),
                ir_sha256=sha(ir),
                metadata_sha256=sha(metadata),
                expected_ptx_sha256=record["ptx_sha256"],
                runtime=result["runtime"],
                original_resources=record,
            )
            save(job_dir / "spec.json", job)
            jobs.append(job)
    return jobs


def restore_options(metadata):
    # These CUDAOptions fields are tuples in memory but arrays in cached JSON.
    # Restore only their container types; preserve every element and all other
    # fields. extern_libs is a tuple of (name, path) pairs after __post_init__.
    names = {
        "cluster_dims",
        "supported_fp8_dtypes",
        "deprecated_fp8_dot_operand_dtypes",
        "allowed_dot_input_precisions",
        "extern_libs",
    }

    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, list) else value

    return {
        name: tuples(value) if name in names else value
        for name, value in metadata.items()
    }


def child(spec, flagtree):
    job = json.loads(Path(spec).read_text())
    folder = Path(job["directory"])
    os.environ.update(
        MLIR_ENABLE_DUMP="1",
        MLIR_DUMP_PATH=str(folder / "passes.txt"),
        TRITON_ALWAYS_COMPILE="1",
        TRITON_CACHE_DIR=str(folder / "replay-cache"),
        CUDA_VISIBLE_DEVICES="",
    )
    for name, value in job["runtime"]["environment"].items():
        if name in (
            "TRITON_PTXAS_PATH",
            "TRITON_DISABLE_LINE_INFO",
            "TRITON_OVERRIDE_ARCH",
            "DISABLE_PTXAS_OPT",
            "PTXAS_OPTIONS",
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    for name in (
        "TRITON_KERNEL_OVERRIDE",
        "TRITON_KERNEL_DUMP",
        "TRITON_OVERRIDE_DIR",
        "LLVM_IR_ENABLE_DUMP",
    ):
        os.environ.pop(name, None)
    sys.path.insert(0, str(Path(flagtree).resolve()))
    import triton
    from triton.backends.compiler import GPUTarget

    for name, expected in job["runtime"]["compiler_files"].items():
        actual = Path(importlib.import_module(name).__file__).resolve()
        if (
            actual != Path(expected["path"]).resolve()
            or sha(actual) != expected["sha256"]
        ):
            raise RuntimeError("Compiler identity mismatch: " + name)
    if sha(Path(triton.__file__)) != job["runtime"]["triton_init_sha256"]:
        raise RuntimeError("Triton module identity mismatch")
    if sha(os.environ["TRITON_PTXAS_PATH"]) != job["runtime"]["ptxas"]["sha256"]:
        raise RuntimeError("Assembler identity mismatch")
    if (
        sha(job["ir"]) != job["ir_sha256"]
        or sha(job["metadata"]) != job["metadata_sha256"]
    ):
        raise RuntimeError("Replay inputs changed")
    metadata = json.loads(Path(job["metadata"]).read_text())
    target = metadata["target"]
    if target != dict(backend="cuda", arch=90, warp_size=32):
        raise RuntimeError("Unexpected cached target: " + str(target))
    # Explicit target and asm access only: do not use .run, .warmup, .n_regs,
    # .n_spills or a launcher, which would initialize GPU binary handles.
    kernel = triton.compile(
        job["ir"], target=GPUTarget(**target), options=restore_options(metadata)
    )
    ptx = kernel.asm["ptx"]
    (folder / "replayed-ptx.txt").write_text(ptx)
    (folder / "replayed-ttgir.txt").write_text(kernel.asm["ttgir"])
    actual = hashlib.sha256(ptx.encode()).hexdigest()
    output = dict(
        status="passed" if actual == job["expected_ptx_sha256"] else "replay_mismatch",
        actual_ptx_sha256=actual,
        expected_ptx_sha256=job["expected_ptx_sha256"],
        gpu_kernel_executed=False,
        target=target,
    )
    save(folder / "result.json", output)
    return 0 if output["status"] == "passed" else 2


def run_job(command, env, log, timeout):
    with Path(log).open("w") as stream:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise RuntimeError("Compile-only child timed out") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--flagtree", default="/opt/flagtree")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--worker-spec")
    args = parser.parse_args()
    if args.worker_spec:
        return child(args.worker_spec, args.flagtree)
    report = Path(args.report_dir).resolve()
    report.mkdir(exist_ok=True, parents=True)
    summary = dict(status="preparing", gpu_kernel_executed=False, timings={}, jobs=[])
    save(
        report / "trace-sources.json",
        {
            name: dict(sha256=sha(ROOT / name), text=(ROOT / name).read_text())
            for name in (
                "tools/qsa_prefill_pipeline_trace.py",
                "tools/test_qsa_prefill_pipeline_trace.py",
                "tools/run_qsa_prefill_pipeline_trace.sh",
                "docs/qsa_prefill_pipeline_dependency_trace.md",
            )
        },
    )
    try:
        jobs = prepare(Path(args.source_report).resolve(), report)
        for job in jobs:
            folder = Path(job["directory"])
            command = [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--worker-spec",
                str(folder / "spec.json"),
            ]
            env = dict(
                os.environ,
                PYTHONNOUSERSITE="1",
                PYTHONUNBUFFERED="1",
                CUDA_VISIBLE_DEVICES="",
            )
            code = run_job(command, env, folder / "compiler.txt", args.timeout)
            events = [
                dict(origin=path.name, **event)
                for path in (folder / "passes.txt", folder / "compiler.txt")
                if path.is_file()
                for event in sections(path.read_text())
            ]
            save(folder / "pass-events.json", events)
            result = (
                json.loads((folder / "result.json").read_text())
                if (folder / "result.json").is_file()
                else {}
            )
            summary["jobs"].append(
                dict(
                    name=job["name"],
                    exit_code=code,
                    result=result,
                    pass_sections=len(events),
                    coverage=coverage(events),
                )
            )
            print(
                job["name"],
                "exit",
                code,
                "PTX identity",
                result.get("status"),
                "pass sections",
                len(events),
                flush=True,
            )
            for event in events:
                print(
                    " ",
                    event["pass_header"],
                    "events",
                    len(event["events"]),
                    flush=True,
                )
                if re.search(
                    "LowerLoops|ExpandLoops|TritonGPUPipeline", event["pass_header"]
                ):
                    for operation in event["events"][:12]:
                        attrs = re.findall(
                            r"(?:loop[._](?:stage|cluster)|tt.latency)\s*=\s*[-0-9]+",
                            operation["text"],
                        )
                        print(
                            "   ", operation["text"][:100], " ".join(attrs), flush=True
                        )
            if (
                code
                or result.get("status") != "passed"
                or not all(coverage(events).values())
            ):
                print("Compiler log:", folder / "compiler.txt", flush=True)
                print(
                    "\n".join((folder / "compiler.txt").read_text().splitlines()[-30:]),
                    flush=True,
                )
                raise RuntimeError("Replay/trace did not validate: " + job["name"])
        summary["status"] = "passed"
        return 0
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        print(summary["error"], flush=True)
        return 1
    finally:
        save(report / "pipeline-trace-summary.json", summary)
        with tarfile.open(report / "pass-traces.tar.gz", "w:gz") as archive:
            for directory in sorted(report.iterdir()):
                if directory.is_dir() and directory.name.startswith("perf_"):
                    for path in sorted(directory.iterdir()):
                        if path.is_file():
                            archive.add(
                                path,
                                arcname=str(path.relative_to(report)),
                                recursive=False,
                            )
        print("Compile-only trace archive:", report / "pass-traces.tar.gz", flush=True)


if __name__ == "__main__":
    sys.exit(main())
