# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture CANN TimelineDetail for one current score launch per fixture."""

import argparse
import collections
import csv
import gzip
import json
import os
import re
import shutil
import sys
from pathlib import Path

from probe_qsa_int8_score_sol_910b import CHUNKS
from probe_qsa_int8_ten_shapes_910b import run_child

EXPECTED = {
    "int8_budget_exact": (
        "_qsa_page64_page_reuse_score",
        "fb0e8f02f645f6b591111e2793469e3fad37db09ec34a103dbdcc278ed2b3015",
    ),
    "int8_long100k": (
        "_qsa_page64_page_reuse_full_score",
        "39243371a6a2fe5af99548ba8d80a332389c79356f58f02a156608dd8c10a0f3",
    ),
}
MAX_RAW_BYTES = 64 * 1024**2
MAX_ARCHIVE_BYTES = 24 * 1024**2
ENTRY = Path(__file__).with_name("probe_qsa_int8_score_sol_910b.py")


def find_profiler():
    root = os.getenv("ASCEND_HOME_PATH")
    for name in ("msopprof", "msprof"):
        candidates = []
        if root:
            candidates.append(Path(root) / "bin" / name)
        if tool := shutil.which(name):
            candidates.append(Path(tool))
        for path in candidates:
            if path.is_file() and os.access(path, os.X_OK):
                return [str(path.resolve())] + (["op"] if name == "msprof" else [])
    return None


def category(label):
    label = label.lower()
    if re.search(r"wait[_ ]?(flag|event)|waitflag", label):
        return "wait"
    if re.search(r"set[_ ]?(flag|event)|setflag|barrier|sync", label):
        return "signal_or_barrier"
    if re.search(r"fix[_ ]?(pipe|p)|l0c", label):
        return "fixpipe_or_l0c"
    if re.search(r"mte[123]|copy|dma", label):
        return "transfer"
    if re.search(r"mmad|matmul|cube", label):
        return "cube"
    if "vector" in label:
        return "vector"
    return "other"


def inspect_trace(path):
    payload = json.loads(path.read_bytes())
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    counts = collections.Counter()
    examples = {}
    for event in events:
        if event.get("ph") != "X" or not isinstance(event.get("dur"), (int, float)):
            continue
        args = event.get("args") or {}
        tag = " ".join(
            str(value)
            for value in (
                event.get("name", ""),
                event.get("cat", ""),
                args.get("instr", ""),
                args.get("pipe", ""),
            )
        )
        label = category(tag)
        counts[label] += 1
        if label != "other":
            sample = dict(
                name=event.get("name"),
                pid=event.get("pid"),
                tid=event.get("tid"),
                ts=event.get("ts"),
                dur=event["dur"],
                args={k: str(v)[:240] for k, v in list(args.items())[:20]},
            )
            if label not in examples or event["dur"] > examples[label]["dur"]:
                examples[label] = sample
    return dict(timed_events=dict(counts), largest_event_in_trace_units=examples)


def inspect_instructions(path):
    counts = collections.Counter()
    examples = {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as source:
        for row in csv.DictReader(source):
            cols = {str(k).strip().lower(): v for k, v in row.items()}
            instr = cols.get("instr") or cols.get("instruction")
            if not instr:
                continue
            pipe = cols.get("pipe") or ""
            label = category(f"{instr} {pipe}")
            counts[label] += 1
            if label != "other":
                examples.setdefault(
                    label,
                    dict(
                        instr=instr,
                        pipe=pipe,
                        addr=cols.get("addr"),
                        cycles=cols.get("cycles"),
                        call_count=cols.get("call_count"),
                        detail=(cols.get("detail") or "")[:512],
                    ),
                )
    return dict(instruction_rows=dict(counts), example=examples)


def collect(case, directory, report):
    paths = sorted(
        p for p in directory.rglob("*") if p.is_file() and not p.is_symlink()
    )
    inventory = [
        dict(path=str(p.relative_to(directory)), bytes=p.stat().st_size) for p in paths
    ]
    selected = [
        p
        for p in paths
        if (
            p.name in ("trace.json", "visualize_data.bin")
            or "_instr_exe" in p.name
            and p.suffix == ".csv"
        )
    ]
    # Preserve compact instruction tables first, then per-core traces before aggregate traces.
    selected.sort(
        key=lambda p: (
            p.suffix != ".csv",
            "core" not in p.parent.name,
            p.stat().st_size,
        )
    )
    remaining = MAX_ARCHIVE_BYTES
    evidence = []
    for index, p in enumerate(selected):
        entry = dict(path=str(p.relative_to(directory)), bytes=p.stat().st_size)
        evidence.append(entry)
        if entry["bytes"] > MAX_RAW_BYTES:
            entry["status"] = "raw_size_limit"
            continue
        if p.suffix in (".json", ".csv"):
            try:
                parsed = (
                    inspect_trace(p) if p.suffix == ".json" else inspect_instructions(p)
                )
                entry.update(parsed)
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                entry.update(status="unparsed", error=str(exc))
        if remaining:
            archive = gzip.compress(p.read_bytes(), compresslevel=3)
            if len(archive) <= remaining:
                dest = report / f"{case}-{index:02d}-{p.name}.gz"
                dest.write_bytes(archive)
                entry["archive"] = dest.name
                remaining -= len(archive)
    usable = any(
        e.get("instruction_rows")
        or any(count for k, count in e.get("timed_events", {}).items() if k != "other")
        for e in evidence
    )
    return dict(
        file_count=len(inventory),
        files=inventory[:1000],
        evidence=evidence,
        instruction_evidence=usable,
    )


def read_entry(directory, case):
    path = directory / f"{case}.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def run_case(args, case, profiler, report):
    kernel, expected_hash = EXPECTED[case]
    root = report / case
    for part in ("reference", "profiled", "capture"):
        (root / part).mkdir(parents=True, exist_ok=True)

    def app(part):
        return [
            sys.executable,
            "-u",
            str(ENTRY),
            "--instruction-entry",
            "--report",
            str(root / part),
            "--case",
            case,
            "--device",
            str(args.device),
        ]

    ref_status, ref_timeout = run_child(
        app("reference") + ["--verify-output"],
        report / f"{case}-reference.txt",
        min(args.timeout, 120),
    )
    reference = read_entry(root / "reference", case)
    ref_entry = reference.get("instruction_entry", {})
    result = dict(
        case=case,
        kernel=kernel,
        reference=reference,
        reference_exit=ref_status,
        reference_timeout=ref_timeout,
        historical_binary_sha256=expected_hash,
    )
    if (
        ref_status != 0
        or ref_timeout
        or ref_entry.get("kernel") != kernel
        or not reference.get("checks")
    ):
        return dict(
            result,
            status="reference_failed",
            exit_code=2 if ref_status == 2 and not ref_timeout else 1,
        )
    result["reference_matches_historical_binary"] = (
        ref_entry.get("binary_sha256") == expected_hash
    )

    command = [
        *profiler,
        "--aic-metrics=TimelineDetail",
        f"--core-id={args.core_id}",
        f"--kernel-name={kernel}",
        "--launch-count=1",
        f"--output={root / 'capture'}",
        *app("profiled"),
    ]
    status, timed_out = run_child(
        command, report / f"{case}-timeline-detail.txt", args.timeout
    )
    profiled = read_entry(root / "profiled", case)
    captured = collect(case, root / "capture", report)
    op_tables = sorted((root / "capture").rglob("OpBasicInfo*.csv"))
    operator_match = (
        any(kernel in p.read_text(errors="replace") for p in op_tables)
        if op_tables
        else None
    )
    matched = profiled.get("instruction_entry") == ref_entry
    ok = (
        status == 0
        and not timed_out
        and matched
        and operator_match is not False
        and captured["instruction_evidence"]
    )
    return dict(
        result,
        status="captured" if ok else "capture_incomplete",
        exit_code=0 if ok else 1,
        command=command,
        profiler_exit=status,
        timed_out=timed_out,
        profiled=profiled,
        profiled_binary_and_launch_match=matched,
        operator_table_matches=operator_match,
        **captured,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--core-id", default="0", help="Logical cores, separated by |")
    parser.add_argument(
        "--timeout", type=int, default=360, help="Seconds per profiling case"
    )
    parser.add_argument("--cases", choices=CHUNKS, nargs="+", default=list(CHUNKS))
    args = parser.parse_args()
    cores = args.core_id.split("|")
    if args.timeout < 1 or not all(
        c.isascii() and c.isdecimal() and 0 <= int(c) < 50 for c in cores
    ):
        parser.error("Choose logical core IDs from 0 to 49 and a positive timeout")
    report = Path(args.report).resolve()
    report.mkdir(parents=True, exist_ok=True)
    profiler = find_profiler()
    result = dict(
        profiler=profiler,
        scope="CANN TimelineDetail simulator-derived instruction trace from a device launch. "
        "Raw event durations are tool units and do not establish device latency.",
        cases=[],
    )
    if not profiler:
        result.update(status="profiler_unavailable", exit_code=2)
        print("SCORE_INSTRUCTION_TOOL_UNAVAILABLE", flush=True)
    else:
        help_status, help_timeout = run_child(
            profiler + ["--help"], report / "timeline-detail-help.txt", 30
        )
        result.update(help_exit=help_status, help_timeout=help_timeout)
        for case in dict.fromkeys(args.cases):
            capture = run_case(args, case, profiler, report)
            result["cases"].append(capture)
            print(
                "SCORE_INSTRUCTIONS",
                json.dumps(
                    {
                        k: v
                        for k, v in capture.items()
                        if k
                        in (
                            "case",
                            "status",
                            "exit_code",
                            "profiled_binary_and_launch_match",
                            "reference_matches_historical_binary",
                            "operator_table_matches",
                            "file_count",
                            "evidence",
                        )
                    }
                ),
                flush=True,
            )
        codes = {c["exit_code"] for c in result["cases"]}
        result["exit_code"] = 1 if 1 in codes else 2 if 2 in codes else 0
    (report / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
