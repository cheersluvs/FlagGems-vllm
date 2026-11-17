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

"""Run four complete fixtures with existing kernels and diagnostic-only stages."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from qsa_history_policy_910b import load_policy
from run_qsa_raw_serial_harness_910b import cases, run_worker


def main():
    parser = argparse.ArgumentParser()
    for flag in (
        "packing-probe",
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
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[p["name"] for p in cases()],
        default=[
            "int8_first_chunk",
            "int8_budget_exact",
            "int8_compact",
            "int8_long100k",
        ],
    )
    args = parser.parse_args()
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
        mode="packing-cost-probe",
        production_changed=False,
        scope="Four full fixtures, diagnostic stages only; no acceptance mean",
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
        QSA_PACKING_PROBE="1",
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
            args.report / (name + "-packing-console.txt"),
            args.case_timeout,
        )
        row = dict(case=name, exit_code=rc, status="failed")
        try:
            d = json.loads((args.report / (name + "-all-score.json")).read_text())
            if d.get("status") != "passed":
                raise ValueError(d.get("traceback", d.get("error", "Worker failed")))
            detail = d["serial_harness"]
            if (
                rc
                or d["status"] != "passed"
                or not all(
                    c.get("passed") is True
                    for c in (
                        d["precheck"],
                        d["postcheck"],
                        detail["precheck"],
                        detail["postcheck"],
                    )
                )
            ):
                raise ValueError("Full pre/post correctness or worker failed")
            row.update(status="passed", packing=detail["packing_probe"])
        except (KeyError, ValueError, OSError) as exc:
            row["error"] = repr(exc)
        result["cases"].append(row)
        print("PACKING_CASE", json.dumps(row), flush=True)
        save()
        if rc in (124, 134, -6, 139, -11):
            break
    result["status"] = (
        "passed"
        if len(result["cases"]) == len(set(args.cases))
        and all(r["status"] == "passed" for r in result["cases"])
        else "incomplete"
    )
    save()
    print("PACKING_SUMMARY", json.dumps(result), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
