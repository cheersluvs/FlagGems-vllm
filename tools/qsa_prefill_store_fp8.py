# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Frozen FP8 choices, boundary reuse and transparent same-kernel controls."""

import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/qsa_prefill_store_fp8_configs.json"
MANIFEST_SHA = "636c1fd9931b65487789c54345422f6bbea7df45ddc421e5b909b4733789681e"
SOURCE_COMMIT = "f6a41ae371cbc0907bb2b2aa4e2eca2c49966a18"


def load_manifest():
    from tools import qsa_prefill_observed_workloads as workloads
    from tools import qsa_prefill_query_joint_policy as policy
    from tools import qsa_prefill_store_probe as probe

    raw = MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise RuntimeError("Frozen FP8 manifest hash changed")
    manifest = json.loads(raw)
    if (
        manifest["schema"] != 1
        or manifest["source_commit"] != SOURCE_COMMIT
        or manifest["candidate_source_sha256"] != probe.shared.OLD_KERNEL_SHA
        or [item["case"] for item in manifest["cases"]]
        != workloads.cases("fp8-prefill")
    ):
        raise RuntimeError("FP8 fixture/source identity changed")
    for item in manifest["cases"]:
        shape = workloads.geometry(item["case"])
        offset, expected = 0, []
        for rows in shape["expected_chunk_rows"]:
            expected.append((offset, rows, shape["logits_width"]))
            offset += rows
        if [(t["offset"], t["rows"], t["width"]) for t in item["traces"]] != expected:
            raise RuntimeError("Frozen FP8 chunk geometry changed")
        policy.validate_choices(item["traces"], True)
    return manifest


def names():
    return [item["case"]["name"] for item in load_manifest()["cases"]]


def saved_record(name):
    data = load_manifest()
    item = next((item for item in data["cases"] if item["case"]["name"] == name), None)
    if item is None:
        raise ValueError("Unknown frozen FP8 case: " + name)
    return dict(item, compiler_identity=data["compiler_identity"]), item["traces"]


def boundary_evidence(options, info, run):
    from tools import qsa_prefill_store_probe as probe

    path = Path(options.report_dir) / "store-boundary.json"
    identity = dict(
        first_case=options.cases[0],
        source=probe.verify_source(),
        policy=options.store_policy,
        compiler_identity=probe.compiler_identity(info),
        driver_sha256=probe.sha(Path(probe.__file__)),
        suite_sha256=probe.sha(Path(__file__)),
    )
    if options.worker_case == options.cases[0]:
        evidence = dict(identity, checks=run())
        validate_boundary(evidence, identity)
        path.write_text(json.dumps(evidence, indent=2) + "\n")
        return evidence["checks"]
    evidence = json.loads(path.read_text())
    validate_boundary(evidence, identity)
    return [
        dict(
            status="covered_by_first_case",
            case=options.cases[0],
            evidence_sha256=probe.sha(path),
        )
    ]


def validate_boundary(evidence, identity):
    if any(evidence.get(k) != v for k, v in identity.items()):
        raise RuntimeError("Boundary source/compiler/policy identity changed")
    checks = evidence.get("checks", [])
    expected = {
        (dtype, width, shift)
        for dtype in ("torch.bfloat16", "torch.float8_e4m3fn")
        for width, shift in ((80, 64), (81, 64), (80, 65))
    }
    if (
        len(checks) != 6
        or {(c["dtype"], c["width"], c["shift"]) for c in checks} != expected
    ):
        raise RuntimeError("Incomplete BF16/FP8 gate boundary evidence")
    for c in checks:
        active = (
            c["dtype"] == "torch.float8_e4m3fn"
            and c["width"] == 80
            and c["shift"] == 64
        )
        if (
            c.get("status") != "passed"
            or c.get("all_rows_checked") != 15
            or c.get("bulk_active") != active
            or (not active and c.get("fallback_binary_verified") is not True)
        ):
            raise RuntimeError("FP8 gate/fallback boundary proof missing")


def gate_records(case, traces, chunks, resources, policy):
    from tools import qsa_prefill_store_probe as probe

    maps = {
        side: {tuple(r["chunk"]): r for r in resources[side]}
        for side in ("baseline", "candidate")
    }
    result = []
    for trace, chunk in zip(traces, chunks):
        key = (trace["rows"], trace["offset"], trace["width"])
        active = (
            trace["route"] == "rectangular"
            and (policy == "aligned" or case["dtype"] == "float8_e4m3fn")
            and probe.eligible(chunk["score_args"][5], trace["width"])
        )
        if not active:
            for field in ("ptx_sha256", "cubin_sha256"):
                if not maps["baseline"][key].get(field) or maps["baseline"][key][
                    field
                ] != maps["candidate"][key].get(field):
                    raise RuntimeError("Fallback differs from original scorer binary")
        result.append(
            dict(
                chunk=list(key),
                route=trace["route"],
                bulk_active=active,
                fallback_binary_verified=not active,
            )
        )
    if len(result) != len(traces) or len(result) != len(chunks):
        raise RuntimeError("Incomplete store-gate chunk evidence")
    return result


def show_summary(cases):
    from tools.qsa_prefill_report import fmt, table

    print(
        "\nFP8 suite: combined/fp8-only；仅本次已完成子集，不是30组原生vLLM验收。",
        flush=True,
    )
    rows = []
    for item in cases:
        for stage, v in item.get("stages", {}).items():
            if v.get("measurement_validated"):
                rows.append(
                    (
                        item["case"],
                        stage,
                        fmt(v["baseline_us"]["median"]),
                        fmt(v["candidate_us"]["median"]),
                        fmt(v["speedup"], 4),
                    )
                )
    table(("case", "stage", "original_us", "candidate_us", "speedup"), rows)
    means = {}
    for stage in ("score", "topk", "pipeline"):
        values = [
            item["stages"][stage]["speedup"]
            for item in cases
            if item.get("status") == "passed" and stage in item.get("stages", {})
        ]
        means[stage] = dict(
            count=len(values),
            arithmetic_mean=statistics.mean(values) if values else None,
            minimum=min(values) if values else None,
        )
    table(
        ("stage", "count", "arithmetic_mean", "min"),
        [
            (stage, v["count"], fmt(v["arithmetic_mean"], 4), fmt(v["minimum"], 4))
            for stage, v in means.items()
        ],
    )
    return means
