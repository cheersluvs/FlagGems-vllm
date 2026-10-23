# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Replay measured upstream12 P1 choices; paired query subdivision is diagnostic."""

import csv
import hashlib
import json
import math
import statistics
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/qsa_prefill_bf16_query_configs.json"
MANIFEST_SHA256 = "3edbb5d7df489c908e99bec52fbeca4ae4a1053443003f39108b63c036545c22"
SOURCE_COMMIT = "f6a41ae371cbc0907bb2b2aa4e2eca2c49966a18"
STAGES = ("score", "topk", "pipeline")


def load_manifest():
    from tools import qsa_prefill_bf16_query_probe as probe
    from tools.qsa_prefill_upstream_workloads import cases, geometry

    data = MANIFEST.read_bytes()
    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
        raise RuntimeError("Frozen upstream config manifest SHA256 mismatch")
    manifest = json.loads(data)
    if (
        manifest["schema"] != 1
        or manifest["source_commit"] != SOURCE_COMMIT
        or manifest["candidate_source_sha256"] != probe.OLD_KERNEL_SHA
        or [c["case"] for c in manifest["cases"]] != cases()
    ):
        raise RuntimeError("Frozen upstream suite/source identity changed")
    for item in manifest["cases"]:
        shape = geometry(item["case"])
        expected, offset = [], 0
        for rows in shape["expected_chunk_rows"]:
            expected.append((offset, rows, shape["logits_width"]))
            offset += rows
        traces = item["traces"]
        if [(t["offset"], t["rows"], t["width"]) for t in traces] != expected:
            raise RuntimeError("Frozen public chunk geometry changed")
        if any(
            t["route"] != "rectangular"
            or t["constants"]["QUERY_PARTS"] != 1
            or t["constants"]["TILE_R"] not in (32, 64)
            for t in traces
        ):
            raise RuntimeError(
                "Upstream12 probe requires the captured large-query tiles"
            )
    return manifest


def saved_record(name):
    manifest = load_manifest()
    item = next((c for c in manifest["cases"] if c["case"]["name"] == name), None)
    if item is None:
        raise ValueError("Unknown upstream12 case: " + name)
    return dict(item, compiler_identity=manifest["compiler_identity"]), item["traces"]


def validate_record(record, first, rounds=None):
    from tools import qsa_prefill_bf16_query_policy as policy
    from tools import qsa_prefill_bf16_query_probe as probe

    query_policy = record.get("query_policy", "all")
    name = record["case"]["name"]
    saved, traces = saved_record(name)
    if (
        record.get("status") != "passed"
        or record.get("case") != saved["case"]
        or record.get("inputs") != saved["inputs"]
        or record.get("fixed_traces") != traces
        or record.get("source") != probe.verify_source()
        or not record.get("compiler_matches_capture")
        or not record.get("same_buffers")
        or not record.get("same_topk")
        or record.get("query_parts_pair") != policy.pair_label(query_policy)
        or record.get("mode") != "kernel"
        or record.get("production_modified") is not False
        or record.get("kernels_before") != record.get("kernels_after")
        or record.get("common_before") != record.get("common_after")
        or not record.get("common_before")
        or record.get("acceptance_eligible") is not False
    ):
        raise RuntimeError("Invalid fixed upstream probe identity: " + name)
    if probe.compiler_identity(record["runtime"]) != saved["compiler_identity"]:
        raise RuntimeError("Compiler changed in upstream probe")
    for side in ("baseline", "candidate"):
        expected = {
            (t["rows"], t["offset"], t["width"]): dict(
                t["constants"],
                QUERY_PARTS=policy.parts(t["constants"], query_policy, side),
            )
            for t in traces
        }
        kernels = record["kernels_before"][side]
        actual = {tuple(k["chunk"]): k["constants"] for k in kernels}
        if actual != expected or len(kernels) != len(expected):
            raise RuntimeError(
                "Actual upstream fixed configs differ from frozen manifest"
            )
    if query_policy == "large-tile":
        policy.verify_fallback(
            record["kernels_before"]["baseline"],
            record["kernels_before"]["candidate"],
            query_policy,
        )
    checks = record.get("boundary_checks", [])
    if name == first:
        if len(checks) != 2 or any(
            c.get("status") != "passed" or c.get("all_rows_checked") != 289
            for c in checks
        ):
            raise RuntimeError(
                "First upstream case must pass all small boundary checks"
            )
    elif checks != [dict(status="covered_by_first_case", case=first)]:
        raise RuntimeError("Missing same-suite boundary provenance")
    if not record.get("telemetry") or any(
        t.get("gpu_guard", {}).get("status") != "passed" for t in record["telemetry"]
    ):
        raise RuntimeError("Missing or rejected upstream GPU guards")
    if set(record.get("self_controls", {})) != {
        f"{side}:{stage}" for side in ("P1", "P2") for stage in ("score", "pipeline")
    }:
        raise RuntimeError("Missing upstream A/A controls")
    for label, stage in {**record["stages"], **record["self_controls"]}.items():
        if stage.get("status") != "passed" or not stage.get("measurement_validated"):
            raise RuntimeError("Unvalidated upstream comparison: " + label)
        count = rounds or len(stage["baseline_us"]["samples"])
        expected_order = [
            ["baseline", "candidate"] if i % 2 == 0 else ["candidate", "baseline"]
            for i in range(count)
        ]
        if (
            count not in (1, 2)
            or any(
                len(stage[side]["samples"]) != count
                for side in ("baseline_us", "candidate_us")
            )
            or stage.get("measurement_order") != expected_order
        ):
            raise RuntimeError("Upstream probe round count/order changed")
        if (
            len(stage.get("validation_before", [])) != 2
            or len(stage.get("validation_after", [])) != 2
        ):
            raise RuntimeError("Missing pre/post numerical validation")
        samples = [
            v
            for side in ("baseline_us", "candidate_us")
            for v in stage[side]["samples"]
        ]
        if not samples or any(not math.isfinite(v) or v <= 0 for v in samples):
            raise RuntimeError("Invalid upstream timing samples")
        if not math.isclose(
            stage["speedup"],
            stage["baseline_us"]["median"] / stage["candidate_us"]["median"],
            rel_tol=1e-12,
        ):
            raise RuntimeError("Invalid recorded upstream speedup")
    if set(record["stages"]) != set(STAGES):
        raise RuntimeError("Missing score/TopK/pipeline stage")


def aggregate(records):
    values = {}
    for stage in STAGES:
        rows = [(r["case"]["name"], r["stages"][stage]["speedup"]) for r in records]
        values[stage] = dict(
            count=len(rows),
            arithmetic_mean=statistics.mean(v for _, v in rows) if rows else None,
            minimum=min((v for _, v in rows), default=None),
            below_one=[name for name, value in rows if value < 1],
        )
    return values


def show(summary):
    from tools.qsa_prefill_report import fmt, table

    print(
        f"\nUpstream BF16 P1/candidate probe: {len(summary['cases'])}/{len(summary['selected_cases'])} requested cases",
        flush=True,
    )
    table(
        ("stage", "count", "arithmetic_mean", "minimum", "below_1_count"),
        [
            (
                stage,
                value["count"],
                fmt(value["arithmetic_mean"], 4),
                fmt(value["minimum"], 4),
                len(value["below_one"]),
            )
            for stage, value in aggregate(summary["cases"]).items()
        ],
    )
    print("query policy:", summary.get("query_policy", "all"), flush=True)
    print("增量对照 P1/candidate；没有新测原生 vLLM，不能作为全30组验收。", flush=True)


def write_results(report, summary):
    from tools.benchmark_qsa_prefill_stages import write_json

    summary["aggregate"] = aggregate(summary["cases"])
    all_names = [c["case"]["name"] for c in load_manifest()["cases"]]
    completed = [c["case"]["name"] for c in summary["cases"]]
    summary["full_upstream12_complete"] = len(completed) == len(all_names) and set(
        completed
    ) == set(all_names)
    write_json(report / "probe-summary.json", summary)
    with (report / "upstream-probe-timings.tsv").open("w") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            (
                "case",
                "stage",
                "P1_us",
                "candidate_us",
                "speedup",
                "paired_min",
                "paired_max",
            )
        )
        for record in summary["cases"]:
            for name in STAGES:
                value = record["stages"][name]
                writer.writerow(
                    (
                        record["case"]["name"],
                        name,
                        value["baseline_us"]["median"],
                        value["candidate_us"]["median"],
                        value["speedup"],
                        value["paired_ratios"]["min"],
                        value["paired_ratios"]["max"],
                    )
                )


def coordinate(options):
    from tools import benchmark_qsa_prefill_triton_stages as bench
    from tools import qsa_prefill_bf16_query_probe as probe
    from tools import qsa_prefill_profiling as profiling

    manifest = load_manifest()
    selected = options.cases or [c["case"]["name"] for c in manifest["cases"]]
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate upstream probe cases")
    report = Path(options.report_dir)
    summary = dict(
        status="preparing",
        cases=[],
        selected_cases=selected,
        acceptance_eligible=False,
        production_modified=False,
        source=probe.verify_source(),
        query_policy=getattr(options, "query_policy", "all"),
        manifest_sha256=MANIFEST_SHA256,
        config_source_commit=SOURCE_COMMIT,
        options=vars(options),
        git_head=bench.support.command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        git_status=bench.support.command(
            ["git", "-C", str(ROOT), "status", "--porcelain"]
        ),
    )
    try:
        probe.prepare_environment(options, summary)
        for index, name in enumerate(selected):
            if index and not bench.idle_gate(
                options,
                summary,
                report,
                "before_" + name,
                bench.support.gpu_guard_module(),
            ):
                raise RuntimeError("GPU rejected before " + name)
            command = [
                sys.executable,
                "-I",
                str(Path(probe.__file__).resolve()),
                *sys.argv[1:],
                "--worker",
                "--worker-case",
                name,
            ]
            summary["active_case"] = name
            write_results(report, summary)
            code = profiling.run_process(command, timeout=options.timeout)
            summary.setdefault("worker_exit_codes", {})[name] = code
            if code:
                raise RuntimeError(f"Upstream probe worker failed: {name}: exit={code}")
            record = json.loads((report / (name + ".json")).read_text())
            if record.get("query_policy", "all") != summary["query_policy"]:
                raise RuntimeError("Worker query policy differs from requested policy")
            validate_record(record, selected[0], options.rounds)
            summary["cases"].append(record)
            write_results(report, summary)
            show(summary)
        summary.update(status="passed", active_case=None)
        return 0
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        print(summary["error"], flush=True)
        return 1
    finally:
        write_results(report, summary)
        show(summary)
