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
FIXED_OP_NAME = "qsa_raw_tle_sort_int8_910b"
HISTORY_OP_NAME = "qsa_raw_history_prefix_int8_910b"


def summarize(report, outcomes, mixed=False, fixed=False, history=False):
    rows = []
    policy = None
    policy_error = None
    if history:
        try:
            from qsa_history_policy_910b import load_policy

            policy = load_policy(report)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            policy_error = repr(exc)
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
                bench = payload[
                    (
                        HISTORY_OP_NAME
                        if history
                        else (
                            FIXED_OP_NAME
                            if fixed
                            else MIXED_OP_NAME if mixed else OP_NAME
                        )
                    )
                ]
                if not bench["details"] or any(
                    d.get("mode") != "kernel" or d.get("dtype") != "torch.int8"
                    for d in bench["details"]
                ):
                    raise ValueError("Expected INT8 kernel-mode details")
                values = [r for d in bench["details"] for r in d["result"]]
                if len(values) != 1 or values[0]["shape_detail"] != [
                    name,
                    (
                        "history_prefix_slots2"
                        if history
                        else (
                            "tle_sort512_slots2"
                            if fixed
                            else "online512_slots2" if mixed else "auto"
                        )
                    ),
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
                if fixed:
                    identity = checks["mixed_identity"]
                    control = checks["generic_identity"]
                    if (
                        not mixed
                        or identity.get("local_sort") != "sort32_mrgsort_fixed512"
                        or identity.get("native") != "block512_query16_cube"
                        or control.get("local_sort")
                        != ("sort32_mrgsort_fixed512" if history else "sort_1d_pack")
                        or control.get("native") != identity.get("native")
                        or control.get("same_packing_and_schedule") is not True
                        or checks["generic_routes"] != checks["mixed_routes"]
                        or any(
                            "generic" not in checks[k].get("extra_candidates", [])
                            for k in ("precheck", "postcheck")
                        )
                    ):
                        raise ValueError(
                            "Fixed candidate/control identity or coverage mismatch"
                        )
                if fixed:
                    for artifact_key, expected_calls in (
                        (
                            "mixed_compiled_artifacts",
                            (
                                dict(
                                    sort32=1,
                                    mrgsort=2,
                                    merge_exhaust_sort4=1,
                                    unpack_sort=1,
                                )
                                if history
                                else dict(sort_1d_pack=0, sort32=1, mrgsort=2)
                            ),
                        ),
                        (
                            "generic_compiled_artifacts",
                            (
                                dict(
                                    sort32=1,
                                    mrgsort=2,
                                    merge_exhaust_sort4=1,
                                    unpack_sort=1,
                                )
                                if history
                                else dict(sort_1d_pack=1, sort32=0, mrgsort=0)
                            ),
                        ),
                    ):
                        artifacts = checks[artifact_key]
                        if not artifacts or any(
                            any(
                                (a.get("primitive_ir") or {}).get("calls", {}).get(k)
                                != v
                                for k, v in expected_calls.items()
                            )
                            for a in artifacts
                        ):
                            raise ValueError(
                                "Missing or mismatched compiled primitive evidence"
                            )
                if history:
                    from qsa_history_policy_910b import use_prefix

                    if policy_error:
                        raise ValueError(policy_error)
                    if (
                        not fixed
                        or checks.get("history_policy") != policy
                        or checks["mixed_identity"].get("history")
                        != "prefix_by_threshold"
                        or checks["generic_identity"].get("history") != "snapshot"
                    ):
                        raise ValueError("History policy or control identity mismatch")
                    routes = checks["mixed_routes"]
                    dispatch = checks["history_dispatch"]
                    if len(routes) != len(dispatch) or not routes:
                        raise ValueError("History dispatch missing")
                    for route, piece in zip(routes, dispatch):
                        if any(
                            route[k] != piece[k] for k in ("first", "count", "blocks")
                        ):
                            raise ValueError("History dispatch geometry mismatch")
                        expected = (
                            "prefix"
                            if use_prefix(piece["blocks"], policy)
                            else "snapshot"
                        )
                        if piece["mode"] != expected:
                            raise ValueError("History fallback policy violated")
                    artifacts = checks["mixed_compiled_artifacts"]
                    if {a.get("history_mode") for a in artifacts} != {
                        p["mode"] for p in dispatch
                    } or any(
                        a["history_mode"] == "prefix"
                        and a["primitive_ir"].get("gather_references") != 0
                        for a in artifacts
                    ):
                        raise ValueError("History dispatch compiled evidence mismatch")
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
                if fixed and not all(
                    isinstance(stages.get(k), (int, float))
                    and math.isfinite(stages[k])
                    and stages[k] > 0
                    for k in ("generic_combined", "generic_online_topk")
                ):
                    raise ValueError("Missing generic control or TopK timing")
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
                if fixed:
                    entry["generic_over_fixed"] = stages["generic_combined"] / cand
                if history:
                    entry["snapshot_over_prefix"] = stages["generic_combined"] / cand
                    entry["prefix_pieces"] = sum(
                        r["mode"] == "prefix" for r in checks["history_dispatch"]
                    )
                    entry["snapshot_pieces"] = sum(
                        r["mode"] == "snapshot" for r in checks["history_dispatch"]
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
            "Stock harness CANN/dispatched history prefix candidate;fixed-sort snapshot control"
            if history
            else (
                "Stock harness whole-call CANN/fixed TLE sort candidate;block512 generic control"
                if fixed
                else (
                    "Stock harness whole-call speedup of prepared two-slot mixed candidate"
                    if mixed
                    else "Stock harness whole-call speedup of unregistered serial raw candidate"
                )
            )
        ),
    )
    if history:
        summary["history_policy"] = policy
    provenance = report / "resume-provenance.json"
    if provenance.is_file():
        summary["resume"] = json.loads(provenance.read_text())
        reused = summary["resume"]["reused_cases"]
        for row in rows:
            row["reused_from_previous_run"] = row["case"] in reused
    if valid:
        summary["arithmetic_mean_speedup"] = sum(r["speedup"] for r in rows) / 10
        if history:
            summary["arithmetic_mean_snapshot_over_prefix"] = (
                sum(r["snapshot_over_prefix"] for r in rows) / 10
            )
        if fixed:
            summary["arithmetic_mean_generic_over_fixed"] = (
                sum(r["generic_over_fixed"] for r in rows) / 10
            )
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


def resume_cases(root, source, report, mixed, fixed=False, history=False, rerun=()):
    source = source.resolve()
    if source == report.resolve():
        raise ValueError("Resume requires a fresh destination")
    rerun = set(rerun)
    if not rerun.issubset({plan["name"] for plan in cases()}):
        raise ValueError("Unknown case requested for remeasurement")
    commit = (source / "code-commit.txt").read_text().strip()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Invalid source code commit")
    changed = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--name-only", commit, "HEAD"], text=True
    ).splitlines()
    allowed = {
        "tools/run_qsa_raw_serial_harness_910b.py",
        "tools/test_qsa_raw_serial_selection.py",
        "tools/qsa_timer_diagnostic_worker_910b.py",
        "tools/test_qsa_timer_diagnostic.py",
    }
    if any(p not in allowed and not p.startswith("docs/") for p in changed):
        raise ValueError(
            "Candidate, baseline or timing code changed; cannot reuse timings"
        )
    old = json.loads((source / "serial-harness-summary.json").read_text())
    outcomes = {row["case"]: row.get("exit_code") for row in old["cases"]}
    checked = summarize(source, outcomes, mixed=mixed, fixed=fixed, history=history)
    reusable = [
        row["case"]
        for row in checked["cases"]
        if row["status"] == "passed" and row["case"] not in rerun
    ]
    for path in source.iterdir():
        if path.is_file() and any(
            path.name.startswith(name + "-")
            or path.name.startswith("harness-" + name + ".")
            or path.name.startswith("harness-" + name + "-")
            for name in reusable
        ):
            shutil.copy2(path, report / path.name)
    (report / "resume-provenance.json").write_text(
        json.dumps(
            dict(
                source_report=str(source),
                source_code_commit=commit,
                reused_cases=reusable,
                requested_rerun_cases=sorted(rerun),
                note="Verified completed cases retained; remaining cases measured separately",
            ),
            indent=2,
        )
    )
    return {name: 0 for name in reusable}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-harness", action="store_true", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case-timeout", type=int, default=1800)
    parser.add_argument("--mixed-harness", action="store_true")
    parser.add_argument("--tle-sort-harness", action="store_true")
    parser.add_argument("--history-prefix-harness", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--resume-report", type=Path)
    parser.add_argument(
        "--rerun-cases",
        nargs="+",
        choices=[plan["name"] for plan in cases()],
        default=[],
    )
    args = parser.parse_args()
    if args.history_prefix_harness and not args.tle_sort_harness:
        parser.error("--history-prefix-harness requires --tle-sort-harness")
    if args.tle_sort_harness and not args.mixed_harness:
        parser.error("--tle-sort-harness requires --mixed-harness")
    if args.summarize_only != (args.source_report is not None):
        parser.error("--summarize-only requires --source-report and vice versa")
    if args.resume_report and args.summarize_only:
        parser.error("Resume and summarize-only are mutually exclusive")
    if args.rerun_cases and not args.resume_report:
        parser.error("--rerun-cases requires --resume-report")
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
        summary = summarize(
            args.report,
            outcomes,
            mixed=args.mixed_harness,
            fixed=args.tle_sort_harness,
            history=args.history_prefix_harness,
        )
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
        QSA_RAW_TLE_SORT_HARNESS="1" if args.tle_sort_harness else "0",
        QSA_RAW_HISTORY_HARNESS="1" if args.history_prefix_harness else "0",
    )
    outcomes = (
        resume_cases(
            root,
            args.resume_report,
            args.report,
            args.mixed_harness,
            fixed=args.tle_sort_harness,
            history=args.history_prefix_harness,
            rerun=args.rerun_cases,
        )
        if args.resume_report
        else {}
    )
    if args.history_prefix_harness:
        from qsa_history_policy_910b import load_policy, preflight

        try:
            if args.resume_report:
                load_policy(args.resume_report)
                for path in args.resume_report.iterdir():
                    if path.is_file() and (
                        path.name == "history-policy.json"
                        or path.name.startswith("boundary-")
                    ):
                        shutil.copy2(path, args.report / path.name)
            else:
                preflight(root, args.report, args.device)
        except Exception as exc:
            (args.report / "history-preflight-failure.json").write_text(
                json.dumps(dict(status="failed", error=repr(exc)), indent=2)
            )
            print("HISTORY_PREFLIGHT_FAILED", repr(exc), flush=True)
            return 2
        env["QSA_HISTORY_POLICY"] = str((args.report / "history-policy.json").resolve())
    for plan in cases():
        name = plan["name"]
        if name in outcomes:
            print("RAW_SERIAL_HARNESS_REUSED", name, flush=True)
            continue
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
        if os.environ.get("QSA_TIMER_DIAGNOSTICS") == "1":
            command[2:4] = [str(root / "tools/qsa_timer_diagnostic_worker_910b.py")]
            env["QSA_TIMER_CASE"] = name
        print("RAW_SERIAL_HARNESS_WORKER", name, flush=True)
        try:
            # Stream stdout while retaining the exact case log; bound worker lifetime.
            log = args.report / ("harness-" + name + "-console.txt")
            rc = run_worker(command, root, env, log, args.case_timeout)
        except OSError as exc:
            print(repr(exc), flush=True)
            rc = 1
        outcomes[name] = rc
        summary = summarize(
            args.report,
            outcomes,
            mixed=args.mixed_harness,
            fixed=args.tle_sort_harness,
            history=args.history_prefix_harness,
        )
        (args.report / "serial-harness-summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        # A fatal NPU error needs explicit diagnosis before more device work.
        if rc in (134, -6, 139, -11, 124):
            break
    summary = summarize(
        args.report,
        outcomes,
        mixed=args.mixed_harness,
        fixed=args.tle_sort_harness,
        history=args.history_prefix_harness,
    )
    (args.report / "serial-harness-summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    for row in summary["cases"]:
        print("RAW_SERIAL_HARNESS_RESULT", json.dumps(row), flush=True)
    print("RAW_SERIAL_HARNESS_SUMMARY", json.dumps(summary), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
