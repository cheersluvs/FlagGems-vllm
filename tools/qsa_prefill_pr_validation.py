# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""PR-byte identity and retained correctness coverage; no GPU imports."""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PINS = ROOT / "docs/qsa_select_paged_prefill/pr-validation-pins.json"


def verify_checkout():
    pins = json.loads(PINS.read_text())
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", pins["pr_commit"], "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    for name, expected in pins["files"].items():
        path = ROOT / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise RuntimeError("PR file changed: " + name)
    return pins


def validate_records(records):
    required = json.loads(PINS.read_text())["required_test_functions"]
    for name in required:
        matches = [
            r
            for node, r in records.items()
            if node.split("::")[-1].split("[")[0] == name
        ]
        if not matches:
            return "Missing PR correctness function: " + name
        for record in matches:
            if record.get("result") == "passed":
                continue
            if (
                name == "test_noncurrent_cuda_device"
                and record.get("result") == "skipped"
                and "a second SM90 device" in record.get("reason", "")
            ):
                continue
            return "Nonpassing PR correctness function: " + name
    return None


def fresh_report_error(directory, *, flagtree):
    """Permit bootstrap inputs, never existing benchmark/accuracy outputs."""
    allowed = {"runner.txt"}
    if flagtree:
        from tools.qsa_prefill_pr_dependencies import WHEEL

        allowed |= {
            "compat134.json",
            "compat134-package.json",
            "flagtree134-tools.json",
            "toolchain-downloads.json",
            "flag-gems-dependency.json",
            "python-deps",
            WHEEL,
        }
    unexpected = sorted(
        p.name for p in Path(directory).iterdir() if p.name not in allowed
    )
    if unexpected:
        return (
            "Report directory must be fresh; unexpected existing entries: "
            + ", ".join(unexpected)
        )
    return None


def summarize(report):
    path = Path(report)
    file = path / "stage-summary.json"
    record = json.loads(file.read_text()) if file.is_file() else {}
    expected_path = ROOT / "tools/qsa_prefill_fp8_workloads.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("_pr_fp8_workloads", expected_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {case["name"] for case in module.cases()}
    cases = record.get("cases", [])
    names = [case.get("case", {}).get("name") for case in cases]
    pins = verify_checkout()
    source_hashes = record.get("manifest", {}).get("source_hashes", {})
    bytes_match = all(
        source_hashes.get(name) == digest for name, digest in pins["files"].items()
    )
    passed = (
        bytes_match
        and record.get("status") == "passed"
        and record.get("accuracy", {}).get("status") == "passed"
        and set(names) == expected
        and len(names) == len(expected)
    )
    rows, ratios = [], {stage: [] for stage in ("score", "topk", "pipeline")}
    print("\nPR H800 revalidation: baseline_us / production_us / speedup", flush=True)
    for case in cases:
        row = {"case": case.get("case", {}).get("name"), "stages": {}}
        values = []
        for name in ratios:
            data = case.get("stages", {}).get(name, {})
            valid = (
                data.get("status") == "passed"
                and data.get("measurement_validated") is True
            )
            passed = passed and valid
            if valid:
                value = float(data["speedup"])
                ratios[name].append(value)
                row["stages"][name] = {
                    "baseline_us": data["baseline_us"]["median"],
                    "production_us": data["candidate_us"]["median"],
                    "speedup": value,
                }
                values.append(f"{name}={value:.4f}x")
            else:
                values.append(name + "=incomplete")
        rows.append(row)
        print(row["case"], " | ".join(values), flush=True)
    means = {
        name: statistics.mean(values) if values else None
        for name, values in ratios.items()
    }
    result = {
        "status": "passed" if passed else "incomplete_or_failed",
        "pr_commit": pins["pr_commit"],
        "reported_pr_bytes_match": bytes_match,
        "correctness": record.get("accuracy", {}),
        "cases": rows,
        "arithmetic_means": means,
        "acceptance_metric": "score-only kernel speedup; TopK excluded",
        "scope": "complete retained PR test file and ten FP8 stage benchmarks; not entire repository suite",
        "historical_30_case_acceptance_not_applicable": True,
    }
    (path / "pr-validation-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("Arithmetic means:", means, flush=True)
    print("PR revalidation status:", result["status"], flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.report:
        raise SystemExit(summarize(args.report))
    pins = verify_checkout()
    print("Verified PR commit:", pins["pr_commit"])
    print("Verified pinned files:", len(pins["files"]))
