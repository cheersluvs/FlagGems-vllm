# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Summarize Ascend QSA INT8 kernel benchmarks and candidate stage timings."""

import argparse
import json
import math
import statistics
from pathlib import Path

from tools.qsa_int8_ten_workloads_910b import cases

OP_NAME = "qsa_select_paged_prefill_int8_910b"
TOPK_SEGMENT = 4096
TOPK_POLICIES = ("auto", "radix_3seg", "prefill_3seg")
TOPK_ROUTES = ("auto", "radix", "prefill")
STAGE_CASES = ("int8_first_chunk", "int8_compact", "int8_budget_exact", "int8_long100k")
SHORT_SCORE_CASES = ("int8_first_chunk", "int8_compact")
TILE_HANDOFF_CASE = "int8_budget_exact"
GROUP_VISIBLE_CASES = ("int8_budget_exact", "int8_long100k")
GROUP_VISIBLE_WALL_CASES = ("int8_two_skewed", "int8_two_balanced")


def topk_route(case, policy):
    if policy not in TOPK_POLICIES:
        raise ValueError(f"Unknown QSA harness TopK policy: {policy}")
    width = case["logits_width"]
    segments = (width + TOPK_SEGMENT - 1) // TOPK_SEGMENT
    if policy != "auto" and case["k"] == 512 and segments == 3:
        return policy.removesuffix("_3seg")
    return "auto"


def merge_case_reports(
    directory,
    *,
    stages=False,
    tile_handoff=False,
    group_visible=False,
    group_visible_wall=False,
):
    """Preserve completed harness records when a later shape fails."""
    details, outcomes = [], {}
    stage_details = {f"{OP_NAME}_{stage}": [] for stage in ("score", "topk")}
    for case in cases():
        name = case["name"]
        if (
            (stages and name not in STAGE_CASES)
            or (tile_handoff and name != TILE_HANDOFF_CASE)
            or (group_visible and name not in GROUP_VISIBLE_CASES)
            or (group_visible_wall and name not in GROUP_VISIBLE_WALL_CASES)
        ):
            continue
        path = directory / f"harness-{name}.json"
        if not path.is_file():
            outcomes[name] = "missing"
            continue
        case_report = json.loads(path.read_text())
        entry = case_report.get(OP_NAME, {})
        outcome = entry.get("result", "missing")
        if outcome == "passed":
            case_details = entry.get("details", [])
            shapes = [
                row.get("shape_detail")
                for detail in case_details
                for row in detail["result"]
            ]
            if shapes != [[name]] and not any(
                shapes == [[name, route]] for route in TOPK_ROUTES
            ):
                outcome = "invalid_shape_record"
            else:
                details.extend(case_details)
                if stages:
                    for op, collected in stage_details.items():
                        collected.extend(case_report.get(op, {}).get("details", []))
        outcomes[name] = outcome
    report = {
        OP_NAME: dict(
            result=(
                "passed"
                if all(v == "passed" for v in outcomes.values())
                else "incomplete"
            ),
            details=details,
            case_outcomes=outcomes,
        )
    }
    if stages:
        report.update(
            {op: dict(details=details) for op, details in stage_details.items()}
        )
    return report


def summarize(
    report,
    pytest_exit_code,
    *,
    stages=False,
    short_score=False,
    tile_handoff=False,
    group_visible=False,
    group_visible_wall=False,
):
    if pytest_exit_code != 0:
        raise ValueError(f"pytest returned {pytest_exit_code}")
    entry = report[OP_NAME]
    if entry.get("result") != "passed":
        raise ValueError(f"Benchmark outcome: {entry.get('result')}")
    expected = {
        case["name"]: case
        for case in cases()
        if (not stages or case["name"] in STAGE_CASES)
        and (not tile_handoff or case["name"] == TILE_HANDOFF_CASE)
        and (not group_visible or case["name"] in GROUP_VISIBLE_CASES)
        and (not group_visible_wall or case["name"] in GROUP_VISIBLE_WALL_CASES)
    }
    rows = {}
    for detail in entry["details"]:
        if (
            detail["op_name"] != OP_NAME
            or detail["mode"] != "kernel"
            or detail["dtype"] != "torch.int8"
        ):
            raise ValueError("Expected the INT8 kernel benchmark")
        for result in detail["result"]:
            name, *route_values = result["shape_detail"]
            route = route_values[0] if route_values else "auto"
            if name not in expected or name in rows or result.get("error_msg"):
                raise ValueError(f"Invalid or duplicate case: {name}")
            if len(route_values) > 1 or route not in TOPK_ROUTES:
                raise ValueError(f"Invalid TopK route record for {name}")
            baseline, candidate, ratio = (
                result["latency_base"],
                result["latency"],
                result["speedup"],
            )
            if not all(
                isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                for value in (baseline, candidate, ratio)
            ):
                raise ValueError(f"Invalid kernel timing for {name}")
            if not math.isclose(ratio, baseline / candidate, rel_tol=1e-9):
                raise ValueError(f"Kernel speedup does not match latencies for {name}")
            rows[name] = dict(
                case=name,
                baseline_ms=baseline,
                candidate_ms=candidate,
                speedup=ratio,
                topk_route=route,
            )
    if set(rows) != set(expected):
        raise ValueError(f"Missing kernel cases: {sorted(set(expected) - set(rows))}")
    for policy in TOPK_POLICIES:
        if all(
            rows[name]["topk_route"] == topk_route(case, policy)
            for name, case in expected.items()
        ):
            break
    else:
        raise ValueError("TopK routes do not match a single harness policy")
    summary = dict(
        exit_code=0,
        mode="kernel",
        benchmark=OP_NAME,
        baseline="torch_npu.npu_quant_lightning_indexer (CANN V1)",
        candidate="Prepared INT8 selection with shared page proof",
        topk_policy=policy,
        cases=[rows[name] for name in expected],
        arithmetic_mean_speedup=(
            None
            if stages or tile_handoff or group_visible or group_visible_wall
            else statistics.mean(row["speedup"] for row in rows.values())
        ),
    )
    if stages:
        for stage in ("score", "topk"):
            op = f"{OP_NAME}_{stage}"
            stage_rows = {}
            for detail in report[op]["details"]:
                if (
                    detail["op_name"] != op
                    or detail["mode"] != "kernel"
                    or detail["dtype"] != "torch.int8"
                ):
                    raise ValueError(f"Invalid stage benchmark: {op}")
                for result in detail["result"]:
                    name, route = result["shape_detail"]
                    if (
                        name not in rows
                        or name in stage_rows
                        or route != rows[name]["topk_route"]
                        or result.get("error_msg")
                    ):
                        raise ValueError(f"Invalid stage case: {op} {name}")
                    latency = result["latency"]
                    if (
                        not isinstance(latency, (int, float))
                        or not math.isfinite(latency)
                        or latency <= 0
                    ):
                        raise ValueError(f"Invalid stage latency: {op} {name}")
                    control, ratio = result.get("latency_base"), result.get("speedup")
                    if short_score and stage == "score" and name in SHORT_SCORE_CASES:
                        if not all(
                            isinstance(value, (int, float))
                            and math.isfinite(value)
                            and value > 0
                            for value in (control, ratio)
                        ) or not math.isclose(ratio, control / latency, rel_tol=1e-9):
                            raise ValueError(f"Invalid paired scorer timing: {name}")
                        rows[name].update(score_control_ms=control, score_speedup=ratio)
                    elif control is not None or ratio is not None:
                        raise ValueError(
                            f"Stage must report candidate latency only: {op}"
                        )
                    stage_rows[name] = latency
            if set(stage_rows) != set(rows):
                raise ValueError(f"Missing {stage} cases")
            for name, latency in stage_rows.items():
                rows[name][f"{stage}_ms"] = latency
        for row in rows.values():
            total = row["score_ms"] + row["topk_ms"]
            row["score_fraction_of_isolated_sum"] = row["score_ms"] / total
            row["isolated_sum_over_combined"] = total / row["candidate_ms"]
        summary["stage_profile"] = True
        summary["stage_scope"] = (
            "Candidate-only score and TopK measured separately with the stock harness; "
            "TopK reads materialized scores. CANN V1 is measured as a fused operator."
        )
        if short_score:
            summary["stage_scope"] = (
                "Score and TopK measured separately with the stock harness. "
                "TopK reads materialized scores. CANN V1 is measured as a fused operator."
            )
    if short_score:
        summary["short_score_trial"] = (
            "N64 query16 versus the current N128 score on the same tensors; "
            "both arms use the current TopK route."
        )
    return summary


def summarize_online_chunk_stages(directory, pytest_exit_code):
    if pytest_exit_code != 0:
        raise ValueError(f"pytest returned {pytest_exit_code}")
    rows = []
    for name in ("int8_budget_exact", "int8_compact"):
        report = json.loads((directory / f"harness-{name}.json").read_text())
        if report[OP_NAME].get("result") != "passed":
            raise ValueError(f"Chunk preflight failed: {name}")
        row = {"case": name, "rows": 40}
        for stage in ("score", "topk", "serial", "online"):
            op = f"{OP_NAME}_chunk_{stage}"
            entry = report[op]
            if entry.get("result", "passed") != "passed":
                raise ValueError(f"Stage failed: {name} {stage}")
            details = entry["details"]
            if len(details) != 1 or details[0]["mode"] != "kernel":
                raise ValueError(f"Unexpected stage records: {name} {stage}")
            records = details[0]["result"]
            if len(records) != 1 or records[0]["shape_detail"] != [name, "auto"]:
                raise ValueError(f"Unexpected stage shape: {name} {stage}")
            value = records[0]["latency"]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid stage time: {name} {stage}")
            row[f"{stage}_ms"] = value
        profile = json.loads((directory / f"{name}-chunk-profile.json").read_text())
        row.update(
            width=profile["width"], profile_status=profile["profile_status"],
            current_short_query16=profile.get("current_short_query16"),
            current_query16_chunks=profile.get("current_query16_chunks"),
            serial_over_online=row["serial_ms"] / row["online_ms"],
        )
        if profile["profile_status"] == "completed":
            row["device_stages"] = profile["timeline"]["stages"]
        else:
            row["profile_error"] = profile.get("profile_error")
        rows.append(row)
    return dict(
        exit_code=0, mode="kernel", scope="Two matched 40-row chunks",
        arithmetic_mean_speedup=None, cases=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("pytest_exit_code", type=int)
    parser.add_argument("--merge-case-reports", action="store_true")
    parser.add_argument("--stages", action="store_true")
    parser.add_argument("--short-score", action="store_true")
    parser.add_argument("--fused-binary", action="store_true")
    parser.add_argument("--online-topk", action="store_true")
    parser.add_argument("--online-stages", action="store_true")
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument("--tile-handoff", action="store_true")
    diagnostic.add_argument("--group-visible", action="store_true")
    diagnostic.add_argument("--group-visible-full", action="store_true")
    diagnostic.add_argument("--group-visible-wall", action="store_true")
    args = parser.parse_args()
    if args.online_stages:
        try:
            summary = summarize_online_chunk_stages(args.report.parent, args.pytest_exit_code)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            summary = dict(exit_code=1, error=str(exc), arithmetic_mean_speedup=None)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        if summary["exit_code"] == 0:
            print("case | score ms | TopK ms | serial ms | online ms | serial/online")
            for row in summary["cases"]:
                print(
                    f"{row['case']} | {row['score_ms']:.6f} | {row['topk_ms']:.6f} | "
                    f"{row['serial_ms']:.6f} | {row['online_ms']:.6f} | "
                    f"{row['serial_over_online']:.6f}x [profile {row['profile_status']}]"
                )
        else:
            print(summary["error"])
        return summary["exit_code"]
    try:
        if args.merge_case_reports:
            report = merge_case_reports(
                args.report.parent,
                stages=args.stages,
                tile_handoff=args.tile_handoff,
                group_visible=args.group_visible,
                group_visible_wall=args.group_visible_wall,
            )
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        else:
            report = json.loads(args.report.read_text())
        summary = summarize(
            report,
            args.pytest_exit_code,
            stages=args.stages,
            short_score=args.short_score,
            tile_handoff=args.tile_handoff,
            group_visible=args.group_visible,
            group_visible_wall=args.group_visible_wall,
        )
        if args.fused_binary or args.online_topk:
            prefix = "QSA_INT8_HARNESS_FUSED_BINARY "
            chunk_key = "fused_chunks"
            summary["candidate"] = "Double-buffered INT8 scoring, fused binary prefix and Torch NPU TopK"
            if args.online_topk:
                prefix = "QSA_INT8_HARNESS_ONLINE_TOPK "
                chunk_key = "online_chunks"
                summary["candidate"] = "Scoped double-buffered INT8 scoring with online Top512 and explicit UB state"
            summary["production_control"] = "Prepared INT8 selection with shared page proof"
            for row in summary["cases"]:
                name = row["case"]
                log = args.report.with_name(f"harness-{name}-console.txt")
                records = [
                    json.loads(line[len(prefix):])
                    for line in log.read_text().splitlines()
                    if line.startswith(prefix)
                ]
                if len(records) != 1 or records[0]["case"] != name:
                    raise ValueError(f"Missing mixed candidate control for {name}")
                record = records[0]
                production_ms = record["production_ms"]
                if (
                    not math.isfinite(production_ms)
                    or production_ms <= 0
                    or not 0 < record[chunk_key] <= record["total_chunks"]
                ):
                    raise ValueError(f"Invalid mixed candidate control for {name}")
                row.update(
                    production_ms=production_ms,
                    production_vs_cann_speedup=row["baseline_ms"] / production_ms,
                    speedup_vs_production=production_ms / row["candidate_ms"],
                    total_chunks=record["total_chunks"],
                    sampled_rows=record["sampled_rows"],
                )
                row[chunk_key] = record[chunk_key]
                if "control_short_query16_enabled" in record:
                    row.update(
                        control_short_query16_enabled=record["control_short_query16_enabled"],
                        control_short_query16_active=record["control_short_query16_active"],
                        control_query16_chunks=record["control_query16_chunks"],
                    )
                if args.online_topk:
                    row["chunk_rows"] = record["chunk_rows"]
            summary["arithmetic_mean_speedup_vs_production"] = statistics.mean(
                row["speedup_vs_production"] for row in summary["cases"]
            )
            summary["production_arithmetic_mean_speedup_vs_cann"] = statistics.mean(
                row["production_vs_cann_speedup"] for row in summary["cases"]
            )
        if args.tile_handoff:
            log = args.report.with_name(f"harness-{TILE_HANDOFF_CASE}-console.txt")
            prefix = "QSA_INT8_HARNESS_TILE_HANDOFF "
            records = [
                json.loads(line[len(prefix) :])
                for line in log.read_text().splitlines()
                if line.startswith(prefix)
            ]
            if len(records) != 1 or records[0]["hits"] <= 0:
                raise ValueError("Missing N128 handoff measurements")
            summary["tile_handoff"] = records[0]
        if args.group_visible or args.group_visible_full or args.group_visible_wall:
            summary["group_visible"] = {}
            prefix = "QSA_INT8_HARNESS_GROUP_VISIBLE "
            for row in summary["cases"]:
                name = row["case"]
                log = args.report.with_name(f"harness-{name}-console.txt")
                records = [
                    json.loads(line[len(prefix) :])
                    for line in log.read_text().splitlines()
                    if line.startswith(prefix)
                ]
                if len(records) != 1 or (
                    name in GROUP_VISIBLE_CASES and records[0]["hits"] <= 0
                ):
                    raise ValueError(f"Missing GroupVisible measurements for {name}")
                if not 0 <= records[0]["hits"] <= records[0]["total_chunks"]:
                    raise ValueError(f"Invalid GroupVisible route counts for {name}")
                if args.group_visible_wall and (
                    not records[0]["hits"]
                    or len(records[0]["fresh_wall"]["samples"]) != 12
                    or len(records[0]["host_phases"]["samples"]) != 6
                ):
                    raise ValueError(
                        f"Incomplete GroupVisible wall diagnosis for {name}"
                    )
                summary["group_visible"][name] = records[0]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        summary = dict(
            exit_code=1,
            pytest_exit_code=args.pytest_exit_code,
            arithmetic_mean_speedup=None,
            error=str(exc),
        )
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if summary["exit_code"] == 0 and args.stages:
        print(
            "case | score ms | TopK ms | combined ms | CANN V1 ms | full speedup | "
            "score fraction | stage sum/combined"
        )
        for row in summary["cases"]:
            print(
                f"{row['case']} | {row['score_ms']:.3f} | {row['topk_ms']:.3f} | "
                f"{row['candidate_ms']:.3f} | {row['baseline_ms']:.3f} | {row['speedup']:.6f} | "
                f"{row['score_fraction_of_isolated_sum']:.3%} | "
                f"{row['isolated_sum_over_combined']:.3f}"
            )
        if args.short_score:
            print("case | current score ms | trial score ms | score speedup")
            for row in summary["cases"]:
                if row["case"] in SHORT_SCORE_CASES:
                    print(
                        f"{row['case']} | {row['score_control_ms']:.3f} | "
                        f"{row['score_ms']:.3f} | {row['score_speedup']:.6f}x"
                    )
        print(
            "QSA INT8 kernel stage profile complete; ten-case acceptance mean pending"
        )
    elif summary["exit_code"] == 0 and (args.fused_binary or args.online_topk):
        print("case | CANN V1 ms | current INT8 ms | candidate ms | vs CANN V1 | vs current")
        for row in summary["cases"]:
            print(
                f"{row['case']} | {row['baseline_ms']:.3f} | {row['production_ms']:.3f} | "
                f"{row['candidate_ms']:.3f} | {row['speedup']:.6f}x | "
                f"{row['speedup_vs_production']:.6f}x"
            )
        print(
            f"Arithmetic mean: candidate vs CANN V1 {summary['arithmetic_mean_speedup']:.6f}x, "
            f"candidate vs current {summary['arithmetic_mean_speedup_vs_production']:.6f}x, "
            f"current vs CANN V1 {summary['production_arithmetic_mean_speedup_vs_cann']:.6f}x"
        )
    elif summary["exit_code"] == 0 and args.tile_handoff:
        row = summary["cases"][0]
        print(f"{TILE_HANDOFF_CASE}: N128 vs CANN V1 {row['speedup']:.6f}x")
    elif summary["exit_code"] == 0 and (args.group_visible or args.group_visible_wall):
        for row in summary["cases"]:
            print(f"{row['case']}: GroupVisible vs CANN V1 {row['speedup']:.6f}x")
    elif summary["exit_code"] == 0:
        print(
            f"QSA INT8 harness kernel arithmetic mean ({len(summary['cases'])} cases): "
            f"{summary['arithmetic_mean_speedup']:.6f}x [TopK {summary['topk_policy']}]"
        )
    else:
        print(f"QSA INT8 harness summary unavailable: {summary['error']}")
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
