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

"""Summarize the full Ascend QSA INT8 kernel matrix recorded by pytest."""

import argparse
import json
import math
import statistics
from pathlib import Path

from tools.qsa_int8_ten_workloads_910b import cases

OP_NAME = "qsa_select_paged_prefill_int8_910b"
TOPK_SEGMENT = 4096


def topk_route(case, policy):
    if policy not in ("auto", "radix_3seg"):
        raise ValueError(f"Unknown QSA harness TopK policy: {policy}")
    width = case["logits_width"]
    segments = (width + TOPK_SEGMENT - 1) // TOPK_SEGMENT
    if policy == "radix_3seg" and case["k"] == 512 and segments == 3:
        return "radix"
    return "auto"


def merge_case_reports(directory):
    """Preserve completed harness records when a later shape fails."""
    details, outcomes = [], {}
    for case in cases():
        name = case["name"]
        path = directory / f"harness-{name}.json"
        if not path.is_file():
            outcomes[name] = "missing"
            continue
        entry = json.loads(path.read_text()).get(OP_NAME, {})
        outcome = entry.get("result", "missing")
        if outcome == "passed":
            case_details = entry.get("details", [])
            shapes = [
                row.get("shape_detail")
                for detail in case_details
                for row in detail["result"]
            ]
            if shapes not in ([[name]], [[name, "auto"]], [[name, "radix"]]):
                outcome = "invalid_shape_record"
            else:
                details.extend(case_details)
        outcomes[name] = outcome
    return {
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


def summarize(report, pytest_exit_code):
    if pytest_exit_code != 0:
        raise ValueError(f"pytest returned {pytest_exit_code}")
    entry = report[OP_NAME]
    if entry.get("result") != "passed":
        raise ValueError(f"Benchmark outcome: {entry.get('result')}")
    expected = {case["name"]: case for case in cases()}
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
            if len(route_values) > 1 or route not in ("auto", "radix"):
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
    for policy in ("auto", "radix_3seg"):
        if all(
            rows[name]["topk_route"] == topk_route(case, policy)
            for name, case in expected.items()
        ):
            break
    else:
        raise ValueError("TopK routes do not match a single harness policy")
    return dict(
        exit_code=0,
        mode="kernel",
        benchmark=OP_NAME,
        baseline="torch_npu.npu_quant_lightning_indexer (CANN V1)",
        candidate="Prepared INT8 selection with shared page proof",
        topk_policy=policy,
        cases=[rows[name] for name in expected],
        arithmetic_mean_speedup=statistics.mean(
            row["speedup"] for row in rows.values()
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("pytest_exit_code", type=int)
    parser.add_argument("--merge-case-reports", action="store_true")
    args = parser.parse_args()
    try:
        if args.merge_case_reports:
            report = merge_case_reports(args.report.parent)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        else:
            report = json.loads(args.report.read_text())
        summary = summarize(report, args.pytest_exit_code)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        summary = dict(
            exit_code=1,
            pytest_exit_code=args.pytest_exit_code,
            arithmetic_mean_speedup=None,
            error=str(exc),
        )
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if summary["exit_code"] == 0:
        print(
            f"QSA INT8 harness kernel arithmetic mean ({len(summary['cases'])} cases): "
            f"{summary['arithmetic_mean_speedup']:.6f}x [TopK {summary['topk_policy']}]"
        )
    else:
        print(f"QSA INT8 harness summary unavailable: {summary['error']}")
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
