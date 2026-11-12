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

"""Reuse only complete unchanged-code acceptance cases in a fresh report directory."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

from tools.qsa_int8_ten_workloads_910b import cases
from tools.summarize_qsa_int8_harness_910b import OP_NAME

SOURCE_COMMIT = "c49ca4ef7e5d3b1657144638115fc3efa326eebc"
ALLOWED_CHANGES = {
    "tools/run_qsa_int8_integrated_910b.sh",
    "tools/qsa_int8_acceptance_resume.py",
    "docs/qsa_int8_acceptance_resume_20261112.md",
}


def completed(directory, repo):
    names = []
    for plan in cases():
        name = plan["name"]
        try:
            report = json.loads((directory / f"harness-{name}.json").read_text())
            checks = json.loads(
                (directory / f"{name}-acceptance-checks.json").read_text()
            )
        except (OSError, ValueError):
            continue
        if report.get(OP_NAME, {}).get("result") != "passed":
            continue
        if (
            checks["rows"] != plan["q_shape"][0]
            or checks["width"] != plan["logits_width"]
            or checks["topk_policy"] != "auto"
            or not checks["sampled_rows"]
            or not all(c["checks_passed"] for c in checks["accuracy"].values())
        ):
            raise ValueError(f"Invalid acceptance correctness record: {name}")
        for path, digest in checks["source_sha256"].items():
            if hashlib.sha256((repo / path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Changed measured source: {path}")
        topk = repo / "src/flaggems_vllm/runtime/backend/_ascend/ops/persistent_topk.py"
        if (
            hashlib.sha256(topk.read_bytes()).hexdigest()
            != checks["topk_identity"]["sha256"]
        ):
            raise ValueError("Changed measured TopK")
        good = True
        for suffix in ("", "_score", "_topk"):
            op = OP_NAME + suffix
            entry = report.get(op, {})
            rows = [r for d in entry.get("details", []) for r in d["result"]]
            details = entry.get("details", [])
            if (
                (not suffix and entry.get("result") != "passed")
                or len(rows) != 1
                or any(
                    d["mode"] != "kernel" or d["dtype"] != "torch.int8" for d in details
                )
            ):
                good = False
                break
            r = rows[0]
            if (
                r["shape_detail"] != [name, "auto"]
                or r.get("error_msg")
                or not math.isfinite(r["latency"])
                or r["latency"] <= 0
            ):
                good = False
                break
            if not suffix and (
                not math.isfinite(r["latency_base"])
                or r["latency_base"] <= 0
                or not math.isclose(
                    r["speedup"], r["latency_base"] / r["latency"], rel_tol=1e-9
                )
            ):
                good = False
                break
        if good:
            names.append(name)
    return names


def prepare(repo, source, destination):
    changed = set(
        subprocess.check_output(
            ["git", "-C", str(repo), "diff", "--name-only", SOURCE_COMMIT, "HEAD"]
        )
        .decode()
        .splitlines()
    )
    if changed - ALLOWED_CHANGES:
        raise ValueError(
            f"Measured code differs from frozen source: {sorted(changed - ALLOWED_CHANGES)}"
        )
    options = (
        sorted(
            Path("/tmp").glob("qsa-ascend910b-int8-harness.*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if source == "auto"
        else [Path(source)]
    )
    for candidate in options:
        marker = candidate / "code-commit.txt"
        if marker.is_file() and marker.read_text().strip() == SOURCE_COMMIT:
            source = candidate
            break
    else:
        raise ValueError(
            "Retained c49ca4ef acceptance report not found; specify QSA_RESUME_REPORT"
        )
    source = Path(source).resolve()
    if source == destination.resolve():
        raise ValueError("Resume destination must be fresh")
    reused = completed(source, repo)
    if not reused:
        raise ValueError("No complete acceptance cases to reuse")
    artifacts = {}
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)
            artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("summary.json", "exit-code.txt"):
        path = destination / name
        if path.is_file():
            shutil.copy2(path, destination / ("before-resume-" + name))
    pending = [p["name"] for p in cases() if p["name"] not in reused]
    for name in pending:
        for suffix in (".json", "-console.txt"):
            path = destination / ("harness-" + name + suffix)
            if path.is_file():
                shutil.copy2(path, destination / ("before-resume-" + path.name))
    provenance = dict(
        source_report=str(source),
        original_code_commit=SOURCE_COMMIT,
        reuse_cases=reused,
        pending_cases=pending,
        source_artifact_sha256=artifacts,
        scope="Merge unchanged-code full-input cases across sessions; no stability/repeat claim",
    )
    (destination / "resume-provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )
    print("REUSE CASES:", ", ".join(reused), flush=True)
    print("PENDING CASES:", ", ".join(pending), flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", default="auto")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo, args.source, args.destination)


if __name__ == "__main__":
    main()
