# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Resume only QSA hardware capture from an existing valid profiling report."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def checked_file(path, expected):
    path = Path(path).resolve()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Preserved source/tool mismatch: " + str(path))
    return path


def gpu_uuid(value):
    from tools.qsa_prefill_gpu_guard import UUID_PATTERN

    value = str(value)
    value = value if value.startswith("GPU-") else "GPU-" + value
    if not UUID_PATTERN.fullmatch(value):
        raise RuntimeError("Invalid recorded GPU UUID: " + value)
    return value


def prepare(options):
    """Preserve original runtime/source paths so the old worker can compare exactly."""
    original = Path(options.resume_report).resolve()
    output = Path(options.report_dir).resolve()
    if original == output:
        raise ValueError("Resume output must differ from the original report")
    summary = json.loads((original / "stage-summary.json").read_text())
    if summary.get("status") != "passed" or not summary.get("cases"):
        raise RuntimeError("Resume requires completed independent measurements")
    measured_options = summary["manifest"]["options"]
    if measured_options["candidate"] != "query-stage-joint":
        raise RuntimeError("Resume supports the retained stage-joint diagnostic only")
    records = summary["cases"]
    if any(c.get("status") != "passed" for c in records):
        raise RuntimeError(
            "Incomplete benchmark cases cannot supply profiling baselines"
        )
    binding = records[0]["runtime"]["operator_binding"]
    code = Path(binding["canonical_source"]).resolve().parents[3]
    sources = json.loads((original / "sources.json").read_text())
    for name, source in sources.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError("Invalid archived source path")
        if hashlib.sha256(source["text"].encode()).hexdigest() != source["sha256"]:
            raise RuntimeError("Archived source hash mismatch: " + name)
        path = (
            original / "qsa_indexer_vllm.py"
            if name.startswith("pinned_vllm/")
            else code / name
        )
        checked_file(path, source["sha256"])
    compat = json.loads((original / "compat134.json").read_text())
    checked_file(compat["libcuda"], compat["libcuda_sha256"])
    toolchain = json.loads((original / "flagtree134-tools.json").read_text())[
        "toolchain"
    ]
    checked_file(toolchain["ptxas"]["path"], toolchain["ptxas"]["sha256"])
    if options.flagtree != measured_options["flagtree"]:
        raise RuntimeError("Use the original FlagTree directory for a matching profile")
    options.cases = [c["case"]["name"] for c in records]
    uuid = gpu_uuid(records[0]["runtime"]["uuid"])
    if any(gpu_uuid(c["runtime"]["uuid"]) != uuid for c in records):
        raise RuntimeError("Mixed baseline GPU identities")
    for source in original.iterdir():
        if source.is_file() and source.suffix in (".json", ".txt", ".tsv"):
            # Original measurements/provenance are copied, never edited.
            name = source.name
            if name in ("runner.txt", "run-status.txt", "profiling-summary.json"):
                name = "baseline-" + name
            shutil.copy2(source, output / name)
    shutil.copy2(original / "qsa_indexer_vllm.py", output / "qsa_indexer_vllm.py")
    os.environ.update(
        LD_LIBRARY_PATH=compat["child_ld_library_path"],
        TRITON_PTXAS_PATH=toolchain["ptxas"]["path"],
        CUDA_VISIBLE_DEVICES=uuid,
    )
    from tools import qsa_prefill_profiling as profiling

    command = profiling.engine_command(options)
    command[2] = str(code / "tools/benchmark_qsa_prefill_triton_stages.py")
    return (
        original,
        command,
        dict(
            status="resuming_counters",
            acceptance_eligible=False,
            hardware_counters_complete=False,
            independent_benchmark_rerun=False,
            original_report=str(original),
            original_git_head=summary["manifest"]["git_head"],
            original_code_root=str(code),
            verified_original_sources=len(sources),
            gpu_uuid=uuid,
            cases=options.cases,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-report", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--flagtree", default="/opt/flagtree")
    parser.add_argument("--ncu")
    parser.add_argument("--prepare-ncu", action="store_true")
    parser.add_argument(
        "--reuse-profile-report", help="Preserve verified prior hardware captures"
    )
    parser.add_argument(
        "--ncu-archive", help="Already downloaded pinned Nsight Compute archive"
    )
    parser.add_argument("--timeout", type=int, default=1800)
    options = parser.parse_args()
    if options.timeout <= 0:
        parser.error("Timeout must be positive")
    output = Path(options.report_dir)
    facts = dict(status="preparing_resume", hardware_counters_complete=False)
    try:
        original, command, facts = prepare(options)
        from tools import qsa_prefill_profiling as profiling

        sources = {
            name: dict(
                sha256=hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                text=(ROOT / name).read_text(),
            )
            for name in (
                "tools/qsa_prefill_profile_resume.py",
                "tools/qsa_ncu_portable.py",
                "tools/qsa_prefill_profiling.py",
                "tools/qsa_prefill_report.py",
                "tools/run_qsa_prefill_profiling_h800.sh",
            )
        }
        (output / "resume-sources.json").write_text(
            json.dumps(sources, indent=2) + "\n"
        )
        if options.reuse_profile_report:
            facts["reused_profiles"] = profiling.reuse_profiles(
                original, options.reuse_profile_report, output, options.cases
            )
        if options.prepare_ncu and not profiling.find_ncu(options.ncu):
            from tools.qsa_ncu_portable import prepare as prepare_ncu

            options.ncu = prepare_ncu(output, options.ncu_archive)
        from tools.qsa_prefill_report import show_report

        print("复用已保存的独立计时；本次不重跑 benchmark/A/A。", flush=True)
        show_report(output, timing_only=True)
        print("复用已有五组计时与 A/A，仅采集硬件计数器：", original, flush=True)
        return profiling.collect_counters(
            options, facts, command, cache_report=original
        )
    except BaseException:
        facts.update(status="failed", error=traceback.format_exc())
        print(facts["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        (output / "profiling-summary.json").write_text(
            json.dumps(facts, indent=2) + "\n"
        )


if __name__ == "__main__":
    sys.exit(main())
