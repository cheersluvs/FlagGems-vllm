# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Full10 FP8 regression with source-only R32 paired issue on validated routes."""

import contextlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_fp8_dot_runtime as runtime  # noqa: E402
from tools import qsa_prefill_observed_workloads as workloads  # noqa: E402

fp8 = runtime.fp8
MANIFEST = ROOT / "docs/qsa_prefill_fp8_dot_full_configs.json"
MANIFEST_SHA = "2b875a887ad327873a3d5753ccb02fd0432e70167f086d40bcffac8477c0a7da"
ORIGINAL_VARIANTS = runtime.VARIANTS
PROOF, VERIFY, WORKER = runtime.proof, runtime.verify_source, runtime.worker
ELIGIBLE = runtime.eligible
TWO_MANIFEST, TWO_CASES = fp8.manifest, fp8.CASES
SOURCES = {
    "tools/qsa_prefill_fp8_dot_full.py",
    "tools/test_qsa_prefill_fp8_dot_full.py",
    "tools/run_qsa_prefill_fp8_dot_full_h800.sh",
    "docs/qsa_prefill_fp8_dot_full.md",
    "docs/qsa_prefill_fp8_dot_full_configs.json",
}


def manifest():
    if runtime.base.sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("Full10 FP8 manifest changed")
    data = json.loads(MANIFEST.read_text())
    if data["production_sha256"] != runtime.probe.compiler.PRODUCTION_SHA:
        raise RuntimeError("Full10 production changed")
    cases = workloads.cases("fp8-prefill")
    if [c["case"] for c in data["cases"]] != cases or len(cases) != 10:
        raise RuntimeError("Full10 fixture set changed")
    for item in data["cases"]:
        shape = workloads.geometry(item["case"])
        expected, offset = [], 0
        for count in shape["expected_chunk_rows"]:
            expected.append((offset, count, shape["logits_width"]))
            offset += count
        if [(t["offset"], t["rows"], t["width"]) for t in item["traces"]] != expected:
            raise RuntimeError("Full10 chunk geometry changed")
    return data


def proof():
    # The immutable compile report contains three variants. Runtime selects two.
    with patch.object(runtime, "VARIANTS", ORIGINAL_VARIANTS):
        return PROOF()


def eligible(named):
    return (
        os.environ.get("QSA_DOT_REQUESTS", "1") == "1"
        and named.get("num_requests", 1) == 1
        and ELIGIBLE(named)
    )


def active(item):
    return len(item["case"]["query_lengths"]) == 1 and any(
        t["route"] == "fp8"
        and ELIGIBLE(dict(IS_FP8=True, NUM_HEADS=4, HEAD_DIM=128, **t["constants"]))
        and t["constants"]["TILE_R"] == 64
        for t in item["traces"]
    )


def verify_source():
    return dict(
        VERIFY(),
        full10_manifest_sha256=MANIFEST_SHA,
        candidate_scope="single-request original R64/P2 main only;other routes unchanged",
    )


def boundary_checks(production, helpers, torch, device):
    # Retain the two validated main layouts, rather than assuming every full10
    # fixture is single-request or forcing its different config into the probe.
    with patch.dict(os.environ, {"QSA_DOT_REQUESTS": "1"}), patch.object(
        fp8, "CASES", TWO_CASES
    ), patch.object(fp8, "manifest", TWO_MANIFEST), patch.object(
        fp8, "make_pair", lambda prod, traces: tuple(runtime.make_modules(prod, traces))
    ), patch.object(
        fp8, "freeze_records", runtime.freeze_records
    ):
        return fp8.boundary_checks(production, helpers, torch, device)


def worker(options):
    item = next(
        c for c in manifest()["cases"] if c["case"]["name"] == options.worker_case
    )
    with patch.dict(
        os.environ, {"QSA_DOT_REQUESTS": str(len(item["case"]["query_lengths"]))}
    ):
        return WORKER(options)


def coordinate(options):
    code = COORDINATE(options)
    path = Path(options.report_dir) / "fp8-dot-runtime-summary.json"
    if path.is_file():
        import statistics

        value = json.loads(path.read_text())
        changed = {c["case"]["name"] for c in manifest()["cases"] if active(c)}
        value["groups"] = {}
        for label, changed_group in (("changed", True), ("original_controls", False)):
            rows = [
                r
                for r in value.get("cases", [])
                if (r["case"] in changed) == changed_group and r["status"] == "passed"
            ]
            if rows:
                value["groups"][label] = dict(
                    count=len(rows),
                    cases=[r["case"] for r in rows],
                    **{
                        stage: statistics.mean(
                            r["stages"][stage]["speedup"] for r in rows
                        )
                        for stage in ("score", "topk", "pipeline")
                    },
                )
        value["probe"] = "full10_fp8_restricted_source_r32_paired"
        (path.parent / "fp8-dot-full-summary.json").write_text(
            json.dumps(value, indent=2) + "\n"
        )
        print("Full10 arithmetic means:", value.get("aggregate"), flush=True)
        print(
            "Changed/control groups:", json.dumps(value["groups"], indent=2), flush=True
        )
        path.unlink()
    return code


def main():
    names = tuple(c["case"]["name"] for c in manifest()["cases"])
    with contextlib.ExitStack() as stack:
        for module, name, value in (
            (fp8, "manifest", manifest),
            (fp8, "CASES", names),
            (runtime, "VARIANTS", ("production", "paired32")),
            (runtime, "PAIRS", ((0, 1),)),
            (runtime, "proof", proof),
            (runtime, "verify_source", verify_source),
            (runtime, "eligible", eligible),
            (runtime, "coordinate", coordinate),
            (runtime, "worker", worker),
            (runtime, "boundary_checks", boundary_checks),
            (runtime, "EXTRA", runtime.EXTRA | SOURCES),
            (runtime, "__file__", str(Path(__file__).resolve())),
            (runtime, "__doc__", __doc__),
        ):
            stack.enter_context(patch.object(module, name, value))
        return runtime.main()


COORDINATE = runtime.coordinate


if __name__ == "__main__":
    sys.exit(main())
