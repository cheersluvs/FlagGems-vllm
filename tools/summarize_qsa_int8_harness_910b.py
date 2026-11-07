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


def summarize(report, pytest_exit_code):
    if pytest_exit_code != 0:
        raise ValueError(f"pytest returned {pytest_exit_code}")
    entry = report[OP_NAME]
    if entry.get("result") != "passed":
        raise ValueError(f"Benchmark outcome: {entry.get('result')}")
    expected = [case["name"] for case in cases()]
    rows = {}
    for detail in entry["details"]:
        if (
            detail["op_name"] != OP_NAME
            or detail["mode"] != "kernel"
            or detail["dtype"] != "torch.int8"
        ):
            raise ValueError("Expected the INT8 kernel benchmark")
        for result in detail["result"]:
            (name,) = result["shape_detail"]
            if name not in expected or name in rows or result.get("error_msg"):
                raise ValueError(f"Invalid or duplicate case: {name}")
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
                case=name, baseline_ms=baseline, candidate_ms=candidate, speedup=ratio
            )
    if set(rows) != set(expected):
        raise ValueError(f"Missing kernel cases: {sorted(set(expected) - set(rows))}")
    return dict(
        exit_code=0,
        mode="kernel",
        benchmark=OP_NAME,
        baseline="torch_npu.npu_quant_lightning_indexer (CANN V1)",
        candidate="Prepared INT8 selection with shared page proof",
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
    args = parser.parse_args()
    try:
        summary = summarize(json.loads(args.report.read_text()), args.pytest_exit_code)
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
            f"{summary['arithmetic_mean_speedup']:.6f}x"
        )
    else:
        print(f"QSA INT8 harness summary unavailable: {summary['error']}")
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
