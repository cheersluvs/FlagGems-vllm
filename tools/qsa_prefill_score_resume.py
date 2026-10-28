# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Validate score-only checkpoints without accepting interrupted measurements."""

import hashlib
import json
import math
import statistics
from pathlib import Path

# These files only add scheduling/diagnostics; math, timer and configs must match.
SCHEDULING_SOURCES = {
    "tools/benchmark_qsa_prefill_stages.py",
    "tools/qsa_prefill_gpu_guard.py",
}
RUNTIME_KEYS = (
    "python",
    "torch_version",
    "torch_source",
    "triton_version",
    "triton_source",
    "triton_init_sha256",
    "compiler_files",
    "ptxas",
    "uuid",
    "allow_tf32",
)


def fingerprint(runtime):
    return {key: runtime[key] for key in RUNTIME_KEYS}


def timer_fingerprint(timer):
    return {
        **{
            key: timer[key]
            for key in (
                "mode",
                "unit",
                "warmup_ms",
                "rep_ms",
                "return_mode",
                "cache_policy",
                "includes_topk",
            )
        },
        "source_hashes": {
            key: value["sha256"] for key, value in timer["sources"].items()
        },
    }


def load(path, root, all_cases, options, gpu_uuid):
    raw = Path(path).read_bytes()
    saved = json.loads(raw)
    manifest = saved["manifest"]
    previous = manifest["options"]
    if previous.get("suite", "comprehensive") != getattr(
        options, "suite", "comprehensive"
    ):
        raise ValueError("Checkpoint benchmark suite changed")
    if (
        previous["mode"] != "kernel"
        or not previous["single_row_ab"]
        or not previous["require_idle_gpu"]
    ):
        raise ValueError(
            "Checkpoint must be a guarded single-row score --mode kernel run"
        )
    for key in ("rounds", "kernel_warmup_ms", "kernel_rep_ms"):
        if previous[key] != getattr(options, key):
            raise ValueError(f"Checkpoint measurement parameter changed: {key}")
    hashes = manifest["source_hashes"]
    for name, expected in hashes.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Checkpoint source path is outside repository")
        if name in SCHEDULING_SOURCES:
            continue
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint frozen source changed: {name}")
    if not {
        "tools/qsa_prefill_score_benchmark.py",
        "benchmark/base.py",
        "src/flaggems_vllm/ops/qsa_select_paged_prefill.py",
    }.issubset(hashes):
        raise ValueError("Checkpoint lacks required scoring/timer source identities")
    definitions = {case["name"]: case for case in all_cases}
    retained = []
    seen = set()
    runtime = None
    timer = None
    # Retained entries preserve original measurements and their provenance.
    for case in saved.get("retained_cases", []) + saved["cases"]:
        if case.get("status") != "passed":
            continue
        name = case["case"]["name"]
        if name in seen or definitions.get(name) != case["case"]:
            raise ValueError(f"Duplicate or changed checkpoint case: {name}")
        info = case.get("score_kernel", {})
        if info.get("status") == "not_applicable_no_score":
            if name != "zero_visible" or "speedup" in info:
                raise ValueError("Invalid no-score checkpoint result")
        elif info.get("status") == "passed":
            if case.get("performance_guard", {}).get("status") != "observed_clean":
                raise ValueError(f"Checkpoint timing guard did not pass: {name}")
            if (
                case["score_timer"]["mode"] != "kernel"
                or case["score_timer"]["includes_topk"]
            ):
                raise ValueError("Checkpoint score timer scope differs")
            timer_value = timer_fingerprint(case["score_timer"])
            if timer is not None and timer != timer_value:
                raise ValueError("Checkpoint mixes score timer identities")
            timer = timer_value
            for field in ("baseline_us", "candidate_us"):
                samples = info[field]["samples"]
                if len(samples) != options.rounds or not all(
                    math.isfinite(v) and v > 0 for v in samples
                ):
                    raise ValueError(f"Incomplete checkpoint samples: {name}")
                if not math.isclose(
                    info[field]["median"], statistics.median(samples), rel_tol=1e-12
                ):
                    raise ValueError(f"Incorrect checkpoint median: {name}")
            expected_ratio = (
                info["baseline_us"]["median"] / info["candidate_us"]["median"]
            )
            if not math.isclose(info["speedup"], expected_ratio, rel_tol=1e-12):
                raise ValueError(f"Incorrect checkpoint speedup: {name}")
            paired = [
                a / b
                for a, b in zip(
                    info["baseline_us"]["samples"], info["candidate_us"]["samples"]
                )
            ]
            if info.get("paired_ratios", {}).get(
                "samples"
            ) != paired or not math.isclose(
                info["paired_ratios"]["median"],
                statistics.median(paired),
                rel_tol=1e-12,
            ):
                raise ValueError(f"Incorrect checkpoint paired ratios: {name}")
            if len(case.get("telemetry", [])) != options.rounds + 1 or any(
                p["gpu_guard"]["status"] != "passed" for p in case["telemetry"]
            ):
                raise ValueError(
                    f"Checkpoint telemetry is incomplete or rejected: {name}"
                )
        else:
            raise ValueError(f"Completed case lacks a valid score result: {name}")
        current_runtime = fingerprint(case["runtime"])
        if runtime is not None and runtime != current_runtime:
            raise ValueError("Checkpoint mixes GPU/runtime identities")
        runtime = current_runtime
        item = dict(case)
        item.setdefault(
            "measurement_origin",
            dict(
                checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                code_commit=manifest["git_head"]["stdout"].strip(),
            ),
        )
        retained.append(item)
        seen.add(name)
    if runtime is None or timer is None:
        raise ValueError("Checkpoint has no completed measurements to retain")
    accuracy = saved.get("retained_accuracy", saved.get("accuracy"))
    if (
        not accuracy
        or accuracy.get("status") != "passed"
        or not accuracy.get("counts", {}).get("passed")
    ):
        raise ValueError("Checkpoint lacks passing full correctness evidence")
    uuid = runtime["uuid"]
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    if gpu_uuid != uuid:
        raise ValueError(
            "Resume requires the same GPU UUID; use a fresh full run on a different GPU"
        )
    return (
        dict(
            checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
            checkpoint_path=str(Path(path).resolve()),
            completed_names=sorted(seen),
            runtime=runtime,
            timer=timer,
            prior_accuracy=accuracy,
        ),
        retained,
    )
