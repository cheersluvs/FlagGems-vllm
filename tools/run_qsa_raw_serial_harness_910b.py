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

"""Run isolated stock harness workers and retain only complete ten-input means."""

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.qsa_int8_ten_workloads_910b import cases  # noqa: E402

OP_NAME = "qsa_raw_serial_int8_910b"
MIXED_OP_NAME = "qsa_raw_mixed_int8_910b"


def summarize(report, outcomes, mixed=False):
    rows = []
    for plan in cases():
        name = plan["name"]
        path = report / (name + "-all-score.json")
        entry = dict(case=name, exit_code=outcomes.get(name), status="failed")
        try:
            detail = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, ValueError) as exc:
            entry["error"] = repr(exc)
            rows.append(entry)
            continue
        if entry["exit_code"] == 0 and detail.get("status") == "passed":
            try:
                payload = json.loads(
                    (report / ("harness-" + name + ".json")).read_text()
                )
                outcome = payload["qsa_select_paged_prefill_int8_910b"]
                expected_test = f"benchmark/test_qsa_raw_serial_int8_perf.py::test_qsa_raw_serial_kernel[{name}]"
                if (
                    outcome.get("result") != "passed"
                    or outcome.get("test_case") != expected_test
                ):
                    raise ValueError("Harness outcome or test identity mismatch")
                bench = payload[MIXED_OP_NAME if mixed else OP_NAME]
                if not bench["details"] or any(
                    d.get("mode") != "kernel" or d.get("dtype") != "torch.int8"
                    for d in bench["details"]
                ):
                    raise ValueError("Expected INT8 kernel-mode details")
                values = [r for d in bench["details"] for r in d["result"]]
                if len(values) != 1 or values[0]["shape_detail"] != [
                    name,
                    "online512_slots2" if mixed else "auto",
                ]:
                    raise ValueError("Unexpected harness case records")
                metric = values[0]
                if metric.get("error_msg"):
                    raise ValueError("Harness timing reported an error")
                base, cand, ratio = (
                    metric["latency_base"],
                    metric["latency"],
                    metric["speedup"],
                )
                if not all(
                    isinstance(v, (float, int)) and math.isfinite(v) and v > 0
                    for v in (base, cand, ratio)
                ) or not math.isclose(ratio, base / cand, rel_tol=1e-9):
                    raise ValueError("Invalid harness timing")
                checks = detail["serial_harness"]
                if not all(
                    (
                        detail["precheck"]["passed"],
                        detail["postcheck"]["passed"],
                        checks["precheck"]["passed"],
                        checks["postcheck"]["passed"],
                    )
                ):
                    raise ValueError("Missing pre/post correctness")
                if mixed:
                    if checks["mixed_identity"]["slots"] != 2 or any(
                        "mixed" not in checks[k].get("extra_candidates", [])
                        or checks[k].get("extra_score_trace_rows") != plan["q_shape"][0]
                        for k in ("precheck", "postcheck")
                    ):
                        raise ValueError("Missing mixed pre/post trace coverage")
                stages = checks["stage_ms"]
                if not all(
                    isinstance(stages.get(key), (int, float))
                    and math.isfinite(stages[key])
                    and stages[key] > 0
                    for key in (
                        "current_combined",
                        "current_score",
                        "raw_score",
                        "raw_topk",
                    )
                ):
                    raise ValueError("Invalid auxiliary stage timing")
                if mixed and not all(
                    isinstance(stages.get(k), (int, float))
                    and math.isfinite(stages[k])
                    and stages[k] > 0
                    for k in ("serial_combined", "online_topk")
                ):
                    raise ValueError("Missing mixed comparison stages")
                entry.update(
                    status="passed",
                    cann_ms=base,
                    **(
                        {
                            "mixed_combined_ms": cand,
                            "serial_over_mixed": stages["serial_combined"] / cand,
                        }
                        if mixed
                        else {"raw_combined_ms": cand}
                    ),
                    speedup=ratio,
                    **{
                        "current_over_mixed" if mixed else "current_over_raw": stages[
                            "current_combined"
                        ]
                        / cand
                    },
                    stage_ms=stages,
                    changed_index_sets=(
                        checks["postcheck"]["extra_current_changed_index_sets"]["mixed"]
                        if mixed
                        else checks["postcheck"]["raw_current_changed_index_sets"]
                    ),
                )
            except (KeyError, ValueError, TypeError, OSError) as exc:
                entry["error"] = repr(exc)
        else:
            entry["error"] = detail.get("error", "Worker or postcheck failed")
        rows.append(entry)
    valid = len(rows) == 10 and all(r["status"] == "passed" for r in rows)
    summary = dict(
        status="passed" if valid else "incomplete",
        cases=rows,
        mode="kernel",
        baseline="CANN V1",
        production_changed=False,
        scope=(
            "Stock harness whole-call speedup of prepared two-slot mixed candidate"
            if mixed
            else "Stock harness whole-call speedup of unregistered serial raw candidate"
        ),
    )
    if valid:
        summary["arithmetic_mean_speedup"] = sum(r["speedup"] for r in rows) / 10
        if mixed:
            summary["arithmetic_mean_serial_over_mixed"] = (
                sum(r["serial_over_mixed"] for r in rows) / 10
            )
    return summary


def run_worker(command, root, env, log, timeout):
    with log.open("w") as output:
        proc = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def stream():
            for line in proc.stdout:
                output.write(line)
                output.flush()
                print(line, end="", flush=True)

        thread = threading.Thread(target=stream, daemon=True)
        thread.start()
        previous = signal.getsignal(signal.SIGTERM)

        def terminate(signum, frame):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, terminate)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            rc = 124
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            thread.join()
            proc.stdout.close()
            signal.signal(signal.SIGTERM, previous)
    return rc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-harness", action="store_true", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case-timeout", type=int, default=1800)
    parser.add_argument("--mixed-harness", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--source-report", type=Path)
    args = parser.parse_args()
    if args.summarize_only != (args.source_report is not None):
        parser.error("--summarize-only requires --source-report and vice versa")
    args.report.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        source = args.source_report.resolve()
        if source == args.report.resolve():
            raise ValueError("Use a fresh destination to preserve original reports")
        old = json.loads((source / "serial-harness-summary.json").read_text())
        outcomes = {row["case"]: row.get("exit_code") for row in old["cases"]}
        for path in source.iterdir():
            if path.is_file() and (
                path.suffix in (".json", ".txt", ".csv")
                or path.name.endswith((".json.gz", ".bc.gz", ".bin.gz"))
            ):
                name = path.name
                if name in (
                    "serial-harness-summary.json",
                    "raw-cube-chain.json",
                    "console.txt",
                    "code-commit.txt",
                    "exit-code.txt",
                ):
                    name = "original-" + name
                shutil.copy2(path, args.report / name)
        summary = summarize(args.report, outcomes, mixed=args.mixed_harness)
        summary["recovered_without_npu_execution"] = True
        summary["source_report"] = str(source)
        (args.report / "serial-harness-summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        for row in summary["cases"]:
            print("RAW_SERIAL_HARNESS_RESULT", json.dumps(row), flush=True)
        print("RAW_SERIAL_HARNESS_SUMMARY", json.dumps(summary), flush=True)
        return 0 if summary["status"] == "passed" else 1
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        QSA_RAW_SERIAL_REPORT=str(args.report.resolve()),
        QSA_DEVICE=str(args.device),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        QSA_RAW_MIXED_HARNESS="1" if args.mixed_harness else "0",
    )
    outcomes = {}
    for plan in cases():
        name = plan["name"]
        command = [
            sys.executable,
            "-u",
            "-m",
            "pytest",
            "-q",
            "-s",
            f"benchmark/test_qsa_raw_serial_int8_perf.py::test_qsa_raw_serial_kernel[{name}]",
            "--mode",
            "kernel",
            "--record",
            "json",
            "--output",
            str(args.report / ("harness-" + name + ".json")),
        ]
        print("RAW_SERIAL_HARNESS_WORKER", name, flush=True)
        try:
            # Stream stdout while retaining the exact case log; bound worker lifetime.
            log = args.report / ("harness-" + name + "-console.txt")
            rc = run_worker(command, root, env, log, args.case_timeout)
        except OSError as exc:
            print(repr(exc), flush=True)
            rc = 1
        outcomes[name] = rc
        summary = summarize(args.report, outcomes, mixed=args.mixed_harness)
        (args.report / "serial-harness-summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        # A fatal NPU error needs explicit diagnosis before more device work.
        if rc in (134, -6, 139, -11, 124):
            break
    summary = summarize(args.report, outcomes, mixed=args.mixed_harness)
    (args.report / "serial-harness-summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    for row in summary["cases"]:
        print("RAW_SERIAL_HARNESS_RESULT", json.dumps(row), flush=True)
    print("RAW_SERIAL_HARNESS_SUMMARY", json.dumps(summary), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
