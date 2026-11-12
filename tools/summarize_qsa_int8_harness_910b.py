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
import os
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
    acceptance_stages=False,
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
            (stages and not acceptance_stages and name not in STAGE_CASES)
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
    acceptance_stages=False,
    short_score=False,
    tile_handoff=False,
    group_visible=False,
    group_visible_wall=False,
):
    if acceptance_stages and (
        not stages or short_score or tile_handoff or group_visible or group_visible_wall
    ):
        raise ValueError(
            "Full acceptance requires all ten ordinary cases with candidate-only stage timing"
        )
    if pytest_exit_code != 0:
        raise ValueError(f"pytest returned {pytest_exit_code}")
    entry = report[OP_NAME]
    if entry.get("result") != "passed":
        raise ValueError(f"Benchmark outcome: {entry.get('result')}")
    expected = {
        case["name"]: case
        for case in cases()
        if (not stages or acceptance_stages or case["name"] in STAGE_CASES)
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
            if (stages and not acceptance_stages)
            or tile_handoff
            or group_visible
            or group_visible_wall
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
    if acceptance_stages:
        summary.update(
            acceptance_stages=True,
            candidate="Unregistered PreparedSharedMapInt8Selection INT8 integrated entry",
            acceptance_scope="All ten full inputs; arithmetic mean of stock harness kernel fused-call speedups",
            correctness_scope=(
                "All-row output structure; CPU selected-score reference on recorded sample rows; "
                "all-chunk scorer control comparison where query16 is active"
            ),
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
        row = {"case": name}
        profile = json.loads((directory / f"{name}-chunk-profile.json").read_text())
        stages = ("score", "topk", "serial", "online")
        if "update_grains" in profile:
            stages += ("online256", "online512")
        if "handoff_grains" in profile:
            stages += ("handoff256", "handoff512")
        if "query_packs" in profile:
            stages += ("query4",)
        if "mixed_prefix" in profile:
            stages += ("prefix_score", "prefix_sort", "prefix_merge")
        if "score_handoff_prefix" in profile:
            stages += ("qk_only", "qk_handoff", "qk_score")
        if "cube_tiles" in profile:
            stages += ("n128",)
        for stage in stages:
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
        row.update(
            rows=profile["rows"], width=profile["width"], profile_status=profile["profile_status"],
            current_short_query16=profile.get("current_short_query16"),
            current_query16_chunks=profile.get("current_query16_chunks"),
            serial_over_online=row["serial_ms"] / row["online_ms"],
        )
        if any(key in profile for key in ("update_grains", "handoff_grains", "query_packs", "mixed_prefix", "score_handoff_prefix", "cube_tiles")):
            samples = profile["paired_kernel_samples_ms"]
            for stage in stages[3:]:
                if len(samples[stage]) != 2 or not all(
                    isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                    for value in samples[stage]
                ):
                    raise ValueError(f"Invalid paired grain times: {name} {stage}")
            row["paired_kernel_samples_ms"] = samples
            if "update_grains" in profile:
                row["grain_speedups"] = {
                    stage: statistics.median(samples["online"]) / statistics.median(samples[stage])
                    for stage in ("online256", "online512")
                }
            if "handoff_grains" in profile:
                row["handoff_speedups"] = {
                    stage: statistics.median(samples["online"]) / statistics.median(samples[stage])
                    for stage in ("handoff256", "handoff512")
                }
            if "query_packs" in profile:
                row["query_pack_speedup"] = (
                    statistics.median(samples["online"]) / statistics.median(samples["query4"])
                )
            if "mixed_prefix" in profile:
                medians = {stage: statistics.median(samples[stage]) for stage in profile["mixed_prefix"]}
                row["prefix_medians_ms"] = medians
                row["prefix_deltas_ms"] = {
                    "local_sort": medians["prefix_sort"] - medians["prefix_score"],
                    "merge_and_final_output": medians["prefix_merge"] - medians["prefix_sort"],
                }
                row["prefix_caveat"] = (
                    "All prefixes store identical-size per-tile diagnostic buffers and use MTE3 release; "
                    "deltas include compiler scheduling changes and are not additive stage costs. "
                    "The original online control uses MTE2 release without per-tile stores."
                )
            if "score_handoff_prefix" in profile:
                medians = {stage: statistics.median(samples[stage]) for stage in profile["score_handoff_prefix"]}
                row["handoff_prefix_medians_ms"] = medians
                row["handoff_prefix_deltas_ms"] = {
                    "ring_handoff": medians["qk_handoff"] - medians["qk_only"],
                    "dequant_reduce": medians["qk_score"] - medians["qk_handoff"],
                }
                row["handoff_prefix_spread_pct"] = {
                    stage: (max(samples[stage]) - min(samples[stage])) / statistics.median(samples[stage]) * 100
                    for stage in ("online", *profile["score_handoff_prefix"])
                }
                row["handoff_prefix_unstable"] = any(value > 15 for value in row["handoff_prefix_spread_pct"].values())
                row["handoff_prefix_caveat"] = (
                    "QK-only uses the existing pure Cube control and stores all M16 rows, including eight padded rows; "
                    "its diagnostic write bytes are twice those of mixed arms. Mixed arms store four heads per query "
                    "and add GM ring traffic and MTE3 release. Scoring replicates its output across four lanes. "
                    "Deltas include different Cube launch configuration, write volume, scheduling and output paths; "
                    "they are diagnostic comparisons, not isolated synchronization costs. "
                    "A paired range/median above 15% is flagged as unstable, not an acceptance failure."
                )
            if "cube_tiles" in profile:
                row["n128_speedup"] = statistics.median(samples["online"]) / statistics.median(samples["n128"])
                row["n128_spread_pct"] = {
                    stage: (max(samples[stage]) - min(samples[stage])) / statistics.median(samples[stage]) * 100
                    for stage in ("online", "n128")
                }
                row["n128_unstable"] = any(value > 15 for value in row["n128_spread_pct"].values())
                row["cube_tile_calls"] = profile["cube_tile_calls"]
                row["topk_update_keys"] = profile["topk_update_keys"]
                row["topk_update_counts"] = profile["topk_update_counts"]
        if profile["profile_status"] == "completed":
            row["device_stages"] = profile["timeline"]["stages"]
        else:
            row["profile_error"] = profile.get("profile_error")
        rows.append(row)
    return dict(
        exit_code=0, mode="kernel", scope="Two matched request-local chunks",
        arithmetic_mean_speedup=None, cases=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("pytest_exit_code", type=int)
    parser.add_argument("--merge-case-reports", action="store_true")
    parser.add_argument("--stages", action="store_true")
    parser.add_argument("--acceptance-stages", action="store_true")
    parser.add_argument("--short-score", action="store_true")
    parser.add_argument("--fused-binary", action="store_true")
    parser.add_argument("--online-topk", action="store_true")
    parser.add_argument("--online-stages", action="store_true")
    parser.add_argument("--cann-serial", action="store_true")
    parser.add_argument("--cann-mixed-recheck", action="store_true")
    parser.add_argument("--cann-fused-replay", action="store_true")
    parser.add_argument("--fused-ten-chunks", action="store_true")
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument("--tile-handoff", action="store_true")
    diagnostic.add_argument("--group-visible", action="store_true")
    diagnostic.add_argument("--group-visible-full", action="store_true")
    diagnostic.add_argument("--group-visible-wall", action="store_true")
    args = parser.parse_args()
    if os.environ.get("QSA_EXPLICIT_CUBE_HANDOFF") == "1":
        from tools.probe_qsa_explicit_cube_handoff_910b import summarize_probe

        return summarize_probe(args.report.parent, args.output, args.pytest_exit_code)
    if args.acceptance_stages:
        args.stages = True
        if any(
            (
                args.short_score,
                args.tile_handoff,
                args.group_visible,
                args.group_visible_full,
                args.group_visible_wall,
                args.fused_binary,
                args.online_topk,
                args.online_stages,
                args.cann_serial,
                args.fused_ten_chunks,
            )
        ):
            parser.error(
                "Acceptance cannot be combined with experimental or subset modes"
            )
    if args.fused_ten_chunks and os.environ.get("QSA_GENERIC_LOCAL_HANDOFF") == "1":
        from tools.probe_qsa_generic_local_buffer_910b import summarize_probe
        return summarize_probe(args.report.parent, args.output, args.pytest_exit_code)
    if args.fused_ten_chunks:
        records, error = [], None
        diagnostic = os.environ.get("QSA_QUERY16_REDUCTION_DIAG") == "1"
        plans = [p for p in cases() if not diagnostic or p["name"] in ("int8_budget_exact", "int8_long100k")]
        try:
            for plan in plans:
                name = plan["name"]
                record = json.loads((args.report.parent / f"{name}-fused-second-cube.json").read_text())
                records.append(record)
                if diagnostic and record.get("reduction_diagnostics", {}).get("status") != "completed":
                    raise ValueError(f"Incomplete reduction diagnostics: {name}")
                if record["status"] not in ("completed", "completed_with_candidate_rejection"):
                    raise ValueError(f"Incomplete probe: {name}")
                for arm, entry in record["arms"].items():
                    if entry["status"] != "passed":
                        continue
                    if not entry["selection"]["checks_passed"]:
                        raise ValueError(f"Selection failed for accepted arm: {name}/{arm}")
                    for times in (entry["samples_ms"], entry["component_samples_ms"]["topk"], entry["component_samples_ms"]["score_plus_topk"]):
                        if len(times) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in times):
                            raise ValueError(f"Invalid chunk timing: {name}/{arm}")
                if record.get("production_comparison_requested"):
                    if record["arms"]["current_int8_integrated"]["status"] != "passed":
                        raise ValueError(f"Current integrated INT8 control failed: {name}")
                baseline = "current_int8_integrated" if record.get("incremental_query16_cube") else "qk_vector_fp32"
                if record["arms"][baseline]["status"] != "passed":
                    raise ValueError(f"FP32 control failed: {name}")
            if args.pytest_exit_code:
                raise ValueError(f"pytest returned {args.pytest_exit_code}")
        except (OSError, KeyError, ValueError, TypeError) as exc:
            error = str(exc)
        rejected = [r for r in records if r["status"] != "completed"]
        result = dict(exit_code=int(error is not None), error=error, cases=records,
            all_candidates_passed=error is None and not rejected and len(records)==len(plans),
            arithmetic_mean_speedup=None, scope=f"{len(plans)} representative 32-row tail chunks; not full-input acceptance")
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
        for plan, record in zip(plans, records):
            print(plan["name"], record["status"])
            for arm, entry in record["arms"].items():
                print(arm, entry["status"], "score us", entry.get("median_us"),
                      "components us", entry.get("component_median_us"), "selection", entry.get("selection"), "vs current score/total", entry.get("speedup_over_current_score"), entry.get("speedup_over_current_total"))
        print("ALL_CANDIDATES_PASSED", result["all_candidates_passed"])
        return result["exit_code"]
    if args.cann_serial:
        rows = []
        error = None
        try:
            names = ("int8_budget_exact",) if args.cann_mixed_recheck else ("int8_budget_exact", "int8_compact")
            for name in names:
                record = json.loads((args.report.parent / f"{name}-cann-serial.json").read_text())
                rows.append(record)
                if record["status"] != "passed":
                    raise ValueError(f"Serial control failed: {name}")
                if args.cann_fused_replay:
                    if not record.get("replay_only") or record["raw_qk_isolation"]["status"] != "passed":
                        raise ValueError("Missing or failed raw-QK replay")
                    fused = record["fused_second_cube_probe"]
                    if fused["status"] != "completed":
                        raise ValueError("Both fused scorers must pass replay")
                    for arm in ("qk_vector_fp32", "qk_second_cube"):
                        entry = fused["arms"][arm]
                        if entry["status"] != "passed" or len(entry["samples_ms"]) != 2 or not all(
                            isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in entry["samples_ms"]):
                            raise ValueError(f"Invalid fused replay timings: {arm}")
                    continue
                required_stages = ("score", "topk", "serial", "production_control")
                if record.get("compared_query_groups") == [4, 16]:
                    required_stages += ("peer_score", "peer_serial")
                elif record.get("score_variant") == "shared_n64":
                    required_stages += ("reference_serial",)
                if record.get("mixed_checks"):
                    required_stages += ("mixed_serial", "mixed_double")
                for stage in required_stages:
                    samples = record["samples_ms"][stage]
                    if len(samples) != 2 or not all(
                        isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                        for value in samples
                    ):
                        record["timing_status"] = "invalid"
                        raise ValueError(f"Invalid serial kernel timing: {name} {stage}")
            if args.pytest_exit_code:
                raise ValueError(f"pytest returned {args.pytest_exit_code}")
        except (OSError, ValueError, KeyError) as exc:
            error = str(exc)
        result = dict(exit_code=int(error is not None), mode="kernel", cases=rows,
                      error=error, arithmetic_mean_speedup=None, scope="Serial structural control, not acceptance")
        def clean(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, list):
                return [clean(item) for item in value]
            return value
        args.output.write_text(json.dumps(clean(result), indent=2, allow_nan=False))
        for row in rows:
            print(row["case"], row["status"], row.get("medians_ms"), row.get("spread_pct"))
        if error:
            print(error)
        return result["exit_code"]
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
                if "grain_speedups" in row:
                    samples = row["paired_kernel_samples_ms"]
                    print(
                        f"{row['case']} paired update 64/256/512 us: "
                        f"{statistics.median(samples['online']) * 1000:.3f} / "
                        f"{statistics.median(samples['online256']) * 1000:.3f} / "
                        f"{statistics.median(samples['online512']) * 1000:.3f}; "
                        f"speedup {row['grain_speedups']['online256']:.3f}x / "
                        f"{row['grain_speedups']['online512']:.3f}x"
                    )
                if "handoff_speedups" in row:
                    samples = row["paired_kernel_samples_ms"]
                    print(
                        f"{row['case']} paired handoff 64/256/512 us: "
                        f"{statistics.median(samples['online']) * 1000:.3f} / "
                        f"{statistics.median(samples['handoff256']) * 1000:.3f} / "
                        f"{statistics.median(samples['handoff512']) * 1000:.3f}, "
                        f"speedup {row['handoff_speedups']['handoff256']:.3f}x / "
                        f"{row['handoff_speedups']['handoff512']:.3f}x"
                    )
                if "query_pack_speedup" in row:
                    samples = row["paired_kernel_samples_ms"]
                    print(
                        f"{row['case']} paired query pack 2/4 us: "
                        f"{statistics.median(samples['online']) * 1000:.3f} / "
                        f"{statistics.median(samples['query4']) * 1000:.3f}; "
                        f"speedup {row['query_pack_speedup']:.3f}x"
                    )
                if "prefix_medians_ms" in row:
                    medians = row["prefix_medians_ms"]
                    print(
                        f"{row['case']} paired prefix score/sort/merge us: "
                        + " / ".join(f"{medians[stage] * 1000:.3f}" for stage in
                                     ("prefix_score", "prefix_sort", "prefix_merge"))
                    )
                    print(f"{row['case']} prefix deltas us: " + ", ".join(
                        f"{stage}={value * 1000:.3f}" for stage, value in row["prefix_deltas_ms"].items()
                    ))
                    print(row["prefix_caveat"])
                if "handoff_prefix_medians_ms" in row:
                    medians = row["handoff_prefix_medians_ms"]
                    print(f"{row['case']} paired QK / handoff / score us: " + " / ".join(
                        f"{medians[stage] * 1000:.3f}" for stage in ("qk_only", "qk_handoff", "qk_score")
                    ))
                    print(f"{row['case']} diagnostic deltas us: " + ", ".join(
                        f"{stage}={value * 1000:.3f}" for stage, value in row["handoff_prefix_deltas_ms"].items()
                    ))
                    print(f"{row['case']} paired range/median %: {row['handoff_prefix_spread_pct']}; "
                          f"unstable={row['handoff_prefix_unstable']}")
                    print(row["handoff_prefix_caveat"])
                if "n128_speedup" in row:
                    samples = row["paired_kernel_samples_ms"]
                    print(f"{row['case']} paired N64/N128 us: "
                          f"{statistics.median(samples['online']) * 1000:.3f} / "
                          f"{statistics.median(samples['n128']) * 1000:.3f}; speedup {row['n128_speedup']:.3f}x")
                    print(f"{row['case']} paired range/median %: {row['n128_spread_pct']}; "
                          f"unstable={row['n128_unstable']}; Cube calls={row['cube_tile_calls']}; "
                          f"TopK update=64, counts={row['topk_update_counts']}")
        else:
            print(summary["error"])
        return summary["exit_code"]
    try:
        if args.merge_case_reports:
            report = merge_case_reports(
                args.report.parent,
                stages=args.stages,
                acceptance_stages=args.acceptance_stages,
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
            acceptance_stages=args.acceptance_stages,
            short_score=args.short_score,
            tile_handoff=args.tile_handoff,
            group_visible=args.group_visible,
            group_visible_wall=args.group_visible_wall,
        )
        if args.acceptance_stages:
            summary["correctness_records"] = {}
            for row in summary["cases"]:
                name = row["case"]
                check = json.loads(
                    (args.report.parent / f"{name}-acceptance-checks.json").read_text()
                )
                plan = next(p for p in cases() if p["name"] == name)
                if (
                    check["case"] != name
                    or check["rows"] != plan["q_shape"][0]
                    or check["width"] != plan["logits_width"]
                    or not check["sampled_rows"]
                    or check["topk_policy"] != "auto"
                    or check["topk_route"] != row["topk_route"]
                    or set(check["accuracy"]) != {"CANN V1", "candidate"}
                    or not all(e["checks_passed"] for e in check["accuracy"].values())
                ):
                    raise ValueError(
                        f"Incomplete full-input correctness record: {name}"
                    )
                summary["correctness_records"][name] = check
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
        if args.acceptance_stages:
            print(
                f"QSA INT8 harness kernel arithmetic mean ({len(summary['cases'])} full cases): "
                f"{summary['arithmetic_mean_speedup']:.6f}x [TopK {summary['topk_policy']}]"
            )
        else:
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
