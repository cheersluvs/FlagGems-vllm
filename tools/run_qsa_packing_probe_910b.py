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

"""Run packing diagnostics or a history update comparison."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from statistics import mean

from qsa_history_policy_910b import load_policy
from run_qsa_raw_serial_harness_910b import cases, run_worker


def main():
    suite = [p["name"] for p in cases()]
    parser = argparse.ArgumentParser()
    probe = parser.add_mutually_exclusive_group(required=True)
    probe.add_argument("--packing-probe", action="store_true")
    probe.add_argument("--paired-history", action="store_true")
    probe.add_argument("--paired-handoff", action="store_true")
    probe.add_argument("--combined-history", action="store_true")
    probe.add_argument("--early-release", action="store_true")
    probe.add_argument("--sort-barrier-ten", action="store_true")
    probe.add_argument("--sort32-barrier-ten", action="store_true")
    probe.add_argument("--final-mrgsort-reuse-long", action="store_true")
    probe.add_argument("--history-compact32", action="store_true")
    probe.add_argument("--history-sort-census", action="store_true")
    probe.add_argument(
        "--short-history", "--short-history-compact", action="store_true"
    )
    for flag in (
        "serial-harness",
        "mixed-harness",
        "tle-sort-harness",
        "history-prefix-harness",
    ):
        parser.add_argument("--" + flag, action="store_true", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume-report", type=Path, required=True)
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--case-timeout", type=int, default=1800)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cases", nargs="+", choices=suite)
    selection.add_argument("--all-cases", action="store_true")
    args = parser.parse_args()
    sort_barrier_suite = args.sort_barrier_ten or args.sort32_barrier_ten
    if args.early_release:
        from qsa_paired_history_910b import CASES

        args.cases = args.cases or list(CASES)
        if args.all_cases or any(name not in CASES for name in args.cases):
            parser.error("Early release supports int8_budget_exact and int8_long100k")
        mode, result_key = "early-release", "early_release"
        scope = "Two full inputs with paired TopK and 512-key double-buffered handoff"
        log_key = "EARLY_RELEASE"
    elif args.history_compact32:
        if args.all_cases or (args.cases and args.cases != ["int8_long100k"]):
            parser.error("Compact32 benchmark selects int8_long100k")
        args.cases = ["int8_long100k"]
        mode, result_key = "history-compact32", "history_compact32"
        scope = "Paired local32 versus cutoff compaction and narrow sort on the complete input"
        log_key = "HISTORY_COMPACT32"
    elif args.history_sort_census:
        if args.all_cases or (args.cases and args.cases != ["int8_long100k"]):
            parser.error("History sort census selects int8_long100k")
        args.cases = ["int8_long100k"]
        mode, result_key = "history-sort-census", "history_sort_census"
        scope = "Conservative survivors on full int8_long100k with paired out=local32"
        log_key = "HISTORY_SORT_CENSUS"
    elif args.final_mrgsort_reuse_long:
        if args.all_cases or (args.cases and args.cases != ["int8_long100k"]):
            parser.error("Final mrgsort reuse benchmark selects int8_long100k")
        args.cases = ["int8_long100k"]
        mode, result_key = "final-mrgsort-reuse-long", "final_mrgsort_reuse_long"
        scope = "Full int8_long100k input with paired TopK and final mrgsort out=local32"
        log_key = "FINAL_MRGSORT_REUSE_LONG"
    elif args.sort32_barrier_ten:
        args.cases = args.cases or suite
        mode, result_key = "sort32-barrier-ten", "sort32_barrier_ten"
        scope = "Complete inputs with paired TopK and the explicit barrier after sort32 removed"
        log_key = "SORT32_BARRIER_TEN"
    elif args.sort_barrier_ten:
        args.cases = args.cases or suite
        mode, result_key = "sort-barrier-ten", "sort_barrier_ten"
        scope = "Complete inputs with paired TopK and one explicit local-sort barrier removed"
        log_key = "SORT_BARRIER_TEN"
    elif args.combined_history:
        args.cases = args.cases or suite
        mode, result_key = "combined-history", "combined_history"
        scope = "Ten-input 512-key handoff with short routes and paired prefix updates"
        log_key = "COMBINED_HISTORY"
    elif args.short_history:
        args.cases = suite if args.all_cases else args.cases or ["int8_compact"]
        mode, result_key = "short-history", "short_history"
        scope = "Complete fixtures with two rounds of stock kernel timing"
        log_key = "SHORT_HISTORY"
    elif args.paired_history or args.paired_handoff:
        from qsa_paired_history_910b import CASES

        args.cases = args.cases or list(CASES)
        if args.all_cases or any(name not in CASES for name in args.cases):
            parser.error("Paired probes support int8_budget_exact and int8_long100k")
        if args.paired_handoff:
            mode, result_key = "paired-handoff", "paired_handoff"
            scope = (
                "512 versus 1024-key handoff with paired TopK and stock kernel timing"
            )
            log_key = "PAIRED_HANDOFF"
        else:
            mode, result_key = "paired-history", "paired_history"
            scope = "Two long inputs with stock kernel timing and fixed 512-key handoff"
            log_key = "PAIRED_HISTORY"
    else:
        if args.all_cases:
            parser.error("--all-cases requires --short-history or --combined-history")
        args.cases = args.cases or [
            "int8_first_chunk",
            "int8_budget_exact",
            "int8_compact",
            "int8_long100k",
        ]
        mode, result_key = "packing-cost-probe", "packing_probe"
        scope = "Four full fixtures, diagnostic stages only; no acceptance mean"
        log_key = "PACKING"
    root = Path(__file__).resolve().parents[1]
    args.report.mkdir(exist_ok=True, parents=True)
    policy = load_policy(args.resume_report)
    for path in args.resume_report.iterdir():
        if path.is_file() and (
            path.name == "history-policy.json" or path.name.startswith("boundary-")
        ):
            shutil.copy2(path, args.report / path.name)
    result = dict(
        status="running",
        mode=mode,
        production_changed=False,
        scope=scope,
        history_policy=policy,
        cases=[],
    )

    def save():
        (args.report / "raw-cube-chain.json").write_text(json.dumps(result, indent=2))

    save()
    env = dict(
        os.environ,
        QSA_RAW_SERIAL_REPORT=str(args.report.resolve()),
        QSA_DEVICE=str(args.device),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        QSA_RAW_MIXED_HARNESS="1",
        QSA_RAW_TLE_SORT_HARNESS="1",
        QSA_RAW_HISTORY_HARNESS="1",
        QSA_PACKING_PROBE="1" if args.packing_probe else "0",
        QSA_SHORT_HISTORY="1" if args.short_history else "0",
        QSA_PAIRED_HISTORY="1" if args.paired_history or args.paired_handoff else "0",
        QSA_PAIRED_HANDOFF="1" if args.paired_handoff else "0",
        QSA_FINAL_MRGSORT_REUSE_LONG="1" if args.final_mrgsort_reuse_long else "0",
        QSA_HISTORY_COMPACT32="1" if args.history_compact32 else "0",
        QSA_HISTORY_SORT_CENSUS="1" if args.history_sort_census else "0",
        QSA_SORT32_BARRIER_TEN="1" if args.sort32_barrier_ten else "0",
        QSA_SORT_BARRIER_TEN="1" if args.sort_barrier_ten else "0",
        QSA_EARLY_RELEASE="1" if args.early_release else "0",
        QSA_COMBINED_HISTORY="1" if args.combined_history else "0",
        QSA_HISTORY_POLICY=str((args.report / "history-policy.json").resolve()),
    )
    for name in dict.fromkeys(args.cases):
        env["QSA_TIMER_CASE"] = name
        cmd = [
            sys.executable,
            "-u",
            str(root / "tools/qsa_timer_diagnostic_worker_910b.py"),
            "-q",
            "-s",
            f"benchmark/test_qsa_raw_serial_int8_perf.py::test_qsa_raw_serial_kernel[{name}]",
            "--mode",
            "kernel",
        ]
        rc = run_worker(
            cmd,
            root,
            env,
            args.report / (name + "-" + mode + "-console.txt"),
            args.case_timeout,
        )
        row = dict(case=name, exit_code=rc, status="failed")
        try:
            d = json.loads((args.report / (name + "-all-score.json")).read_text())
            if d.get("status") != "passed":
                raise ValueError(d.get("traceback", d.get("error", "Worker failed")))
            detail = d["serial_harness"]
            checks = [d["precheck"], d["postcheck"]]
            if not args.history_sort_census:
                checks.extend([detail["precheck"], detail["postcheck"]])
            if rc or not all(c.get("passed") is True for c in checks):
                raise ValueError("Full pre/post correctness or worker failed")
            measurement = detail[result_key]
            if measurement.get("status") != "passed":
                raise ValueError("Measurement did not finish")
            if (
                args.paired_history
                or args.paired_handoff
                or args.combined_history
                or args.early_release
                or sort_barrier_suite
                or args.history_compact32
                or args.final_mrgsort_reuse_long
            ):
                check = json.loads(
                    (args.report / (name + "-three-way-merge.json")).read_text()
                )
                if check.get("status") != "passed":
                    raise ValueError("Three-way merge check did not pass")
                row["three_way_checked_rows"] = check["checked_rows"]
            row.update(status="passed")
            row["packing" if args.packing_probe else result_key] = measurement
        except (KeyError, ValueError, OSError) as exc:
            row["error"] = repr(exc)
        result["cases"].append(row)
        print(log_key + "_CASE", json.dumps(row), flush=True)
        save()
        if rc in (124, 134, -6, 139, -11):
            break
    result["status"] = (
        "passed"
        if len(result["cases"]) == len(set(args.cases))
        and all(r["status"] == "passed" for r in result["cases"])
        else "incomplete"
    )
    if (
        (args.short_history or args.combined_history or sort_barrier_suite)
        and result["status"] == "passed"
        and set(args.cases) == set(suite)
    ):
        measurements = [row[result_key] for row in result["cases"]]
        if sort_barrier_suite:
            result["paired_pieces"] = sum(
                sum(m["paired_pieces_by_blocks"].values()) for m in measurements
            )
            result["paired_odd_tail_pieces"] = sum(
                m["paired_odd_tail_pieces"] for m in measurements
            )
        for ratio in (
            "cann_over_candidate",
            "cann_over_control",
            "control_over_candidate",
        ):
            result["arithmetic_mean_" + ratio] = mean(m[ratio] for m in measurements)
    save()
    print(log_key + "_SUMMARY", json.dumps(result), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
