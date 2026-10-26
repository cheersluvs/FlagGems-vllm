# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Seven-launch FP8 production hardware capture; historical stage times only."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_observed_workloads as workloads  # noqa: E402
from tools import qsa_prefill_production_profile as collector  # noqa: E402

CASES = (
    "fp8_first_chunk_float8_e4m3fn",
    "fp8_tail113_float8_e4m3fn",
    "fp8_long100k_float8_e4m3fn",
)
MANIFEST = ROOT / "docs/qsa_prefill_fp8_profile_configs.json"
MANIFEST_SHA = "4f29130c68f17c528ce49c6d4e4ab1351ec36359bc45269cfa170b2948737caa"
EXTRA_SOURCES = collector.EXTRA_SOURCES | {
    "tools/qsa_prefill_fp8_profile.py",
    "tools/test_qsa_prefill_fp8_profile.py",
    "tools/run_qsa_prefill_fp8_profile_h800.sh",
    "docs/qsa_prefill_fp8_profile.md",
    "docs/qsa_prefill_fp8_profile_configs.json",
}


def manifest():
    if collector.base.sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("FP8 profile manifest changed")
    data = json.loads(MANIFEST.read_text())
    if data["production_sha256"] != collector.stage.PRODUCTION_SHA:
        raise RuntimeError("FP8 production snapshot changed")
    if tuple(v["case"]["name"] for v in data["cases"]) != CASES:
        raise RuntimeError("FP8 profile case set changed")
    expected = {c["name"]: c for c in workloads.cases("fp8-prefill")}
    for item in data["cases"]:
        case, traces = item["case"], item["traces"]
        if expected.get(case["name"]) != case or case["dtype"] != "float8_e4m3fn":
            raise RuntimeError("FP8 profile workload changed")
        geometry = workloads.geometry(case)
        chunks = geometry["expected_chunk_rows"]
        wanted = [
            (sum(chunks[:i]), n, geometry["logits_width"]) for i, n in enumerate(chunks)
        ]
        if [(v["offset"], v["rows"], v["width"]) for v in traces] != wanted:
            raise RuntimeError("FP8 chunk geometry changed")
        if any(
            t["route"] != "fp8" or t["constants"]["QUERY_PARTS"] not in (1, 2)
            for t in traces
        ):
            raise RuntimeError("Expected existing FP8 store scorer")
    return data


def verify_source():
    manifest()
    return dict(
        collector.stage.verify_source(),
        fp8_profile_manifest_sha256=MANIFEST_SHA,
        profile_driver_sha256=collector.base.sha(Path(__file__)),
        profile_sources={
            n: collector.base.sha(ROOT / n) for n in sorted(EXTRA_SOURCES)
        },
    )


def main():
    # Keep the original collection, numerical validation, binary matching,
    # resume parsing and GPU guards byte-for-byte. Child workers enter this
    # adapter too, so they cannot silently revert to the old BF16 case list.
    with patch.object(collector, "CASES", CASES), patch.object(
        collector, "manifest", manifest
    ), patch.object(collector, "verify_source", verify_source), patch.object(
        collector, "EXTRA_SOURCES", EXTRA_SOURCES
    ), patch.object(
        collector, "__file__", str(Path(__file__).resolve())
    ), patch.object(
        collector, "__doc__", __doc__
    ):
        return collector.main()


if __name__ == "__main__":
    sys.exit(main())
