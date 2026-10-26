# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Compile-gated two-buffer producer-first accuracy and two-case stage A/B."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import qsa_prefill_double_buffer as double  # noqa: E402
from tools import qsa_prefill_producer_runtime as runtime  # noqa: E402

ORIGINAL_PROFILES = runtime.profiles
ORIGINAL_WORKER = runtime.worker


def profiles(*, allow_create=False):
    result = ORIGINAL_PROFILES(allow_create=allow_create)
    source = Path(os.environ["QSA_DOUBLE_BUFFER_REPORT"]).resolve()
    summary = json.loads((source / "double-buffer-summary.json").read_text())
    if (
        summary["status"] != "compile_feasible_pending_accuracy"
        or len(summary["jobs"]) != 4
        or len(summary["decisions"]) != 2
        or not all(v["reason"] is None for v in summary["decisions"])
    ):
        raise RuntimeError("Both two-buffer compile gates must pass")
    sources = json.loads((source / "sources.json").read_text())
    if set(sources) != set(double.SOURCES) or any(
        double.trace.sha(ROOT / n) != v["sha256"] for n, v in sources.items()
    ):
        raise RuntimeError("Two-buffer compiler proof source changed")
    proof = json.loads((source / "protocol-model.json").read_text())
    if (
        proof["status"] != "passed"
        or len(proof["checks"]) != 34
        or not all(v["safe"] for v in proof["checks"])
        or proof["missing_reuse_barrier_counterexample"]["safe"]
    ):
        raise RuntimeError("Incomplete lifetime model proof")
    for capture in result:
        case = capture["case"]
        control, candidate = (
            source / (case + suffix) for suffix in ("-control", "-two")
        )
        a, b = (
            json.loads((p / "result.json").read_text()) for p in (control, candidate)
        )
        for folder, record in ((control, a), (candidate, b)):
            saved = next(j for j in summary["jobs"] if j["name"] == folder.name)
            if (
                saved["exit_code"]
                or saved["result"] != record
                or record["status"] != "compiled"
            ):
                raise RuntimeError("Invalid two-buffer compile record")
            if (
                double.trace.sha(folder / "ptx.txt") != record["ptx_sha256"]
                or double.trace.sha(folder / "kernel.cubin") != record["cubin_sha256"]
            ):
                raise RuntimeError("Two-buffer binary changed")
            if (
                double.ptx_protocol((folder / "ptx.txt").read_text())
                != record["PTX_protocol"]
            ):
                raise RuntimeError("PTX reuse protocol changed")
        if a["ptx_sha256"] != capture["candidate_ptx_sha256"] or double.gate(a, b):
            raise RuntimeError("Three-buffer control/two-buffer gate mismatch")
        original = (control / "input.ttgir").read_text()
        if double.trace.sha(control / "input.ttgir") != capture["override_sha256"]:
            raise RuntimeError("Captured three-buffer IR differs")
        changed, lifetime = double.two_slots(original)
        if (candidate / "input.ttgir").read_text() != changed:
            raise RuntimeError("Two-buffer IR changed non-ring instructions")
        spec = json.loads((candidate / "spec.json").read_text())
        if (
            spec["lifetime_proof"] != lifetime
            or double.trace.sha(candidate / "input.ttgir") != spec["ir_sha256"]
        ):
            raise RuntimeError("Ring rewrite proof changed")
        path = Path(os.environ["QSA_PRODUCER_RUNTIME_REPORT"]) / (
            "double-buffer-" + case + ".ttgir"
        )
        if path.exists() and path.read_text() != changed:
            raise RuntimeError("Prepared two-buffer IR changed")
        if not path.exists():
            if not allow_create:
                raise RuntimeError("Prepared two-buffer IR disappeared")
            path.write_text(changed)
        capture.update(
            candidate=b,
            candidate_ptx_sha256=b["ptx_sha256"],
            override_file=str(path),
            override_sha256=double.trace.sha(path),
            variant="two_buffer_producer_first",
            lifetime_proof=lifetime,
            lifetime_model_sha256=double.trace.sha(source / "protocol-model.json"),
            compile_summary_sha256=double.trace.sha(
                source / "double-buffer-summary.json"
            ),
            three_buffer_control_ptx_sha256=a["ptx_sha256"],
        )
    return result


def worker(options):
    code = ORIGINAL_WORKER(options)
    path = Path(options.report_dir) / (options.worker_case + ".json")
    if path.is_file():
        result = json.loads(path.read_text())
        result["algorithm"] = "producer_first_two_buffer_verified_IR"
        path.write_text(json.dumps(result, indent=2) + "\n")
    return code


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--double-buffer-report", default=os.environ.get("QSA_DOUBLE_BUFFER_REPORT")
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--compile-timeout", type=int, default=300)
    own, remaining = parser.parse_known_args()
    if any(v in remaining for v in ("-h", "--help")):
        print(
            __doc__
            + "\nExtra flags: --compile-only, --compile-timeout, --double-buffer-report"
        )
        sys.argv = [sys.argv[0], *remaining]
        return runtime.main()
    read = argparse.ArgumentParser(add_help=False)
    read.add_argument("--report-dir", required=True)
    read.add_argument(
        "--producer-report", default=os.environ.get("QSA_PRODUCER_REPORT")
    )
    read.add_argument("--flagtree", default="/opt/flagtree")
    values, _ = read.parse_known_args(remaining)
    if not values.producer_report:
        read.error("--producer-report is required")
    report = Path(values.report_dir).resolve()
    report.mkdir(parents=True, exist_ok=True)
    compiled = (
        Path(own.double_buffer_report).resolve()
        if own.double_buffer_report
        else report / "compile"
    )
    if not own.double_buffer_report:
        command = [
            sys.executable,
            "-I",
            str(ROOT / "tools/qsa_prefill_double_buffer.py"),
            "--report-dir",
            str(compiled),
            "--source-report",
            str(Path(values.producer_report).resolve()),
            "--flagtree",
            values.flagtree,
            "--timeout",
            str(own.compile_timeout),
        ]
        # Inherit stdout for visible compile/proof results. The compiler's own
        # child processes use CUDA_VISIBLE_DEVICES="" and explicit SM90 target.
        import subprocess

        code = subprocess.call(command)
    else:
        code = 0
    for name, new in (
        ("double-buffer-summary.json", "double-buffer-compile-summary.json"),
        ("compile-artifacts.tar.gz", "double-buffer-compile-artifacts.tar.gz"),
    ):
        if (compiled / name).is_file():
            shutil.copyfile(compiled / name, report / new)
    path = compiled / "double-buffer-summary.json"
    summary = json.loads(path.read_text()) if path.is_file() else {}
    if (
        code
        or summary.get("status") != "compile_feasible_pending_accuracy"
        or own.compile_only
    ):
        double.trace.save(
            report / "double-buffer-runtime-summary.json",
            dict(
                status=(
                    "failed"
                    if code
                    else "compile_only" if own.compile_only else "compile_not_promising"
                ),
                compile_status=summary.get("status"),
                gpu_kernel_executed=False,
                timings={},
            ),
        )
        print(
            "Stopped before GPU execution;compile status:",
            summary.get("status"),
            flush=True,
        )
        return code
    os.environ["QSA_DOUBLE_BUFFER_REPORT"] = str(compiled)
    sys.argv = [sys.argv[0], *remaining]
    is_worker = "--worker-case" in remaining
    with patch.object(runtime, "profiles", profiles), patch.object(
        runtime, "worker", worker
    ), patch.object(runtime, "__file__", str(Path(__file__).resolve())), patch.object(
        runtime, "__doc__", __doc__
    ), patch.object(
        runtime, "EXTRA_SOURCES", runtime.EXTRA_SOURCES | set(double.SOURCES)
    ):
        code = runtime.main()
    if not is_worker:
        path = report / "producer-runtime-summary.json"
        if path.is_file():
            path.replace(report / "double-buffer-runtime-summary.json")
    return code


if __name__ == "__main__":
    sys.exit(main())
