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

"""Torch-free boundary policy and matched prepared-kernel dispatch."""

import json
import math
import shutil
import sys
from functools import partial


def boundary_cases():
    return tuple(
        (rows, 512 * steps) for steps in (1, 2, 4, 8) for rows in (16, 128)
    ) + ((17, 8256), (112, 29696))


def choose_policy(data):
    if data.get("status") != "passed" or data.get("mode") != "history_boundary_tuning":
        raise ValueError("Boundary run did not complete")
    cases = data.get("cases", [])
    if len(cases) != len(boundary_cases()) or {
        (c["rows"], c["width"]) for c in cases
    } != set(boundary_cases()):
        raise ValueError("Boundary set incomplete")
    evidence = []
    for case in cases:
        if case.get("status") != "passed":
            raise ValueError("Boundary case failed")
        for phase in ("precheck", "postcheck"):
            for arm in ("snapshot", "prefix"):
                check = case[phase][arm]
                if not all(
                    check.get(k) is True
                    for k in (
                        "passed",
                        "trace_scores_exact",
                        "selected_scores_exact",
                        "output_guards_unchanged",
                    )
                ):
                    raise ValueError("Boundary correctness missing")
        counts = case["single_capture_counts"]
        if (
            not counts.get("expected_host_counts")
            or counts["expected_host_counts"] != counts.get("observed_host_counts")
            or not counts.get("tasks_per_window")
            or counts["tasks_per_window"] != counts.get("expected_tasks_per_window")
        ):
            raise ValueError("Boundary timing counts invalid")
        artifacts = case["history_ir"]["prefix"]
        if set(artifacts) != {"False", "True"}:
            raise ValueError("Boundary primitive evidence missing")
        for item in artifacts.values():
            if item.get("gather_references") != 0:
                raise ValueError("Prefix gather retained")
            if item.get("calls") != dict(
                sort32=1, mrgsort=2, merge_exhaust_sort4=1, unpack_sort=1
            ):
                raise ValueError("Boundary primitive structure invalid")
        samples = case["samples"]
        if len(samples["snapshot"]) != 2 or len(samples["prefix"]) != 2:
            raise ValueError("Two boundary rounds required")
        ratios = []
        for a, b in zip(samples["snapshot"], samples["prefix"]):
            av, bv = a["kernel_us"], b["kernel_us"]
            if not all(
                isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                for v in (av, bv)
            ):
                raise ValueError("Invalid boundary latency")
            ratios.append(av / bv)
        evidence.append(
            dict(
                rows=case["rows"],
                blocks=(case["width"] + 511) // 512,
                round_ratios=ratios,
            )
        )
    thresholds = sorted({r["blocks"] for r in evidence})
    threshold = next(
        (
            k
            for k in thresholds
            if all(min(r["round_ratios"]) >= 1.02 for r in evidence if r["blocks"] >= k)
        ),
        None,
    )
    if threshold is None:
        raise ValueError("No qualified prefix threshold;full benchmark not started")
    return dict(
        status="passed",
        threshold=threshold,
        max_blocks=max(thresholds),
        margin=1.02,
        rule="all tested larger cases and both rounds pass margin",
        boundary_evidence=evidence,
    )


def use_prefix(blocks, policy):
    if (
        policy.get("status") != "passed"
        or not 1 <= policy["threshold"] <= policy["max_blocks"]
    ):
        raise ValueError("Invalid history policy")
    return policy["threshold"] <= blocks <= policy["max_blocks"]


def load_policy(report):
    policy = json.loads((report / "history-policy.json").read_text())
    validated = choose_policy(
        json.loads((report / "boundary-raw-cube-chain.json").read_text())
    )
    if any(policy.get(k) != v for k, v in validated.items()):
        raise ValueError("Policy does not match boundary evidence")
    return policy


def dispatcher(module, policy):
    from qsa_raw_history_merge_910b import ir_evidence

    def choose(blocks):
        prefix = use_prefix(blocks, policy)
        mode = "prefix" if prefix else "snapshot"
        return (
            getattr(module, "raw_history_prepared_" + mode),
            getattr(module, "raw_history_topk_" + mode),
            partial(ir_evidence, complete=False, expect_no_gather=prefix),
            mode,
        )

    return choose


def preflight(root, report, device):
    import os
    import subprocess

    from run_qsa_raw_serial_harness_910b import run_worker

    dest = report / "history-boundary-run"
    dest.mkdir()
    command = [
        sys.executable,
        "-u",
        str(root / "tools/probe_qsa_raw_mixed_online_910b.py"),
        "--raw-mixed-online",
        "--boundaries",
        "--cube-block512",
        "--history-full-merge",
        "--history-boundary",
        "--rounds",
        "2",
        "--device",
        str(device),
        "--report",
        str(dest),
    ]
    rc = run_worker(
        command, root, dict(os.environ), report / "history-boundary-console.txt", 1800
    )
    for path in dest.iterdir():
        if path.is_file() and (
            path.suffix in (".json", ".txt", ".csv")
            or path.name.endswith((".json.gz", ".bc.gz", ".bin.gz"))
        ):
            shutil.copy2(path, report / ("boundary-" + path.name))
    if rc:
        raise RuntimeError("History boundary preflight failed: " + str(rc))
    data = json.loads((dest / "raw-cube-chain.json").read_text())
    policy = choose_policy(data)
    policy["source_code_commit"] = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    (report / "history-policy.json").write_text(json.dumps(policy, indent=2))
    print("HISTORY_POLICY", json.dumps(policy), flush=True)
    return policy
