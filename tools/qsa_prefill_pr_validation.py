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
        if not any(record.get("result") == "passed" for record in matches):
            return "No executed passing cases for PR correctness function: " + name
        expected_count = json.loads(PINS.read_text())["expected_test_counts"][name]
        if len(matches) != expected_count:
            return f"Incomplete correctness collection for {name}: {len(matches)}/{expected_count}"
        for record in matches:
            if record.get("result") == "passed":
                continue
            allowed_skips = (
                "The upstream fixture uses BF16 inputs",
                "Native cross-check is limited to validated head/dimension pairs",
                "Native BF16 H8/D256 exceeds H800 shared memory",
            )
            if record.get("result") == "skipped" and any(
                reason in record.get("reason", "") for reason in allowed_skips
            ):
                continue
            return "Nonpassing PR correctness function: " + name
    return None


def comparable_accuracy_runtime(value):
    """Ignore relocated temporary paths, retaining content hashes and GPU ID."""
    if isinstance(value, dict):
        return {
            key: comparable_accuracy_runtime(item)
            for key, item in value.items()
            if key
            not in {"path", "source", "canonical_source", "candidate_source", "libcuda"}
        }
    if isinstance(value, list):
        return [comparable_accuracy_runtime(item) for item in value]
    return value


def reuse_accuracy(previous, destination, runtime, identity_fn):
    previous, destination = Path(previous).resolve(), Path(destination).resolve()
    pins = verify_checkout()
    manifest = json.loads((previous / "manifest.json").read_text())
    for name, digest in pins["files"].items():
        if manifest["source_hashes"].get(name) != digest:
            raise RuntimeError("Previous correctness source differs: " + name)
    records = json.loads((previous / "accuracy-full.json").read_text())
    error = validate_records(records)
    if error:
        raise RuntimeError(error)
    saved = json.loads((previous / "accuracy-runtime.json").read_text())
    if saved.get("status") != "passed" or saved.get("accuracy_profile") != "full":
        raise RuntimeError("Previous correctness process did not finish successfully")
    if comparable_accuracy_runtime(
        identity_fn(saved["runtime"])
    ) != comparable_accuracy_runtime(identity_fn(runtime)):
        raise RuntimeError(
            "Previous correctness compiler/GPU/native-library identity differs"
        )
    evidence = {
        "previous_report": str(previous),
        "pr_commit": pins["pr_commit"],
        "files": {},
    }
    for name in ("accuracy-full.json", "accuracy-full.txt", "accuracy-runtime.json"):
        content = (previous / name).read_bytes()
        (destination / name).write_bytes(content)
        evidence["files"][name] = hashlib.sha256(content).hexdigest()
    (destination / "accuracy-reuse.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    print("Reusing verified full correctness:", previous, flush=True)
    return evidence


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
