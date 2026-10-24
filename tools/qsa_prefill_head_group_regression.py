# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Thirty-case regression of the unchanged restricted BF16 head-pair candidate."""

import contextlib
import json
import statistics
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_head_group_gated as gated  # noqa: E402

GATED_PATH = Path(gated.__file__).resolve()
BOUNDARIES = gated.base.boundary_checks
COORDINATE = gated.base.coordinate
MANIFEST = ROOT / "docs/qsa_prefill_head_group_regression_configs.json"
MANIFEST_SHA = "9289031364bcf701f47ff4c2d0176232349b230c9a75cb6b346aca1f469d371b"
EXTRA_SOURCES = {
    "tools/qsa_prefill_head_group_regression.py",
    "tools/test_qsa_prefill_head_group_regression.py",
    "tools/run_qsa_prefill_head_group_regression_h800.sh",
    "docs/qsa_prefill_head_group_regression_configs.json",
    "docs/qsa_prefill_head_group_regression.md",
}


def manifest():
    from tools import qsa_prefill_observed_workloads as workloads
    from tools import qsa_prefill_triton_ab as ab

    if gated.base.sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("Regression manifest changed")
    data = json.loads(MANIFEST.read_text())
    if data["production_sha256"] != ab.INTEGRATED_PRODUCTION_SHA256:
        raise RuntimeError("Regression production changed")
    if [v["case"] for v in data["cases"]] != workloads.cases("scored-prefill"):
        raise RuntimeError("Expected unchanged scored-prefill30")
    for item in data["cases"]:
        shape = workloads.geometry(item["case"])
        expected = [
            (sum(shape["expected_chunk_rows"][:i]), n, shape["logits_width"])
            for i, n in enumerate(shape["expected_chunk_rows"])
        ]
        if [(t["offset"], t["rows"], t["width"]) for t in item["traces"]] != expected:
            raise RuntimeError("Regression chunk geometry changed")
    return data


def verify_source():
    data = manifest()
    sha = gated.base.sha
    production = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
    if (
        sha(production) != data["production_sha256"]
        or sha(ROOT / "tools/qsa_prefill_head_group_kernel.py") != gated.KERNEL_SHA
    ):
        raise RuntimeError("Frozen regression production/kernel changed")
    return dict(
        production_sha256=sha(production),
        candidate_sha256=gated.KERNEL_SHA,
        manifest_sha256=MANIFEST_SHA,
        driver_sha256=sha(Path(__file__)),
        gated_driver_sha256=sha(GATED_PATH),
        base_driver_sha256=sha(gated.BASE_PATH),
    )


def fixtures():
    # Group numbers correspond to the original three evidence records.
    return (
        [
            dict(group=1, rows=n, heads=4, dim=128, page=64 if n == 64 else 196)
            for n in (1, 63, 64, 65, 129)
        ]
        + [
            dict(group=0, rows=65, heads=h, dim=d, page=196)
            for h, d in ((2, 64), (8, 128), (4, 256))
        ]
        + [dict(group=2, rows=65, heads=4, dim=128, page=64, parts=p) for p in (1, 2)]
    )


def check_fixture(production, helpers, torch, device, spec):
    from tools import qsa_prefill_score_benchmark as scoring

    group, rows = spec["group"], spec["rows"]
    tile, warps = (32, 8) if group == 0 else (64, 4)
    dtype = torch.float8_e4m3fn if group == 2 else torch.bfloat16
    lengths = [rows] if rows == 1 else [1, rows - 1]
    levels = (0, 1, 63, 64, 65, 195, 196, 197, 511, 512, 513, 1024, 1025)
    visible = [levels[i % len(levels)] for i in range(rows)]
    visible[-1] = 1025
    args = helpers.make_inputs(
        lengths,
        [1025] * len(lengths),
        dtype=dtype,
        padded=True,
        query_base=97,
        page_size=spec["page"],
        visible=visible,
        num_heads=spec["heads"],
        head_dim=spec["dim"],
    )
    width, offset = 1088, 0
    # The 129-row fixture has a chunk crossing requests and a 65-row tile tail.
    counts = [1] if rows == 1 else ([64, 65] if rows == 129 else [1, rows - 1])
    traces, chunks, buffers = [], [], []
    for count in counts:
        storage = torch.full(
            (count * width + 128,), -12345.0, dtype=torch.float32, device=args[0].device
        )
        config = dict(
            TILE_R=tile,
            BLOCK_N=64,
            K_TILES=16,
            num_warps=warps,
            num_stages=2,
            num_ctas=1,
        )
        if group == 2:
            config["QUERY_PARTS"] = spec["parts"]
        traces.append(
            dict(
                offset=offset,
                rows=count,
                width=width,
                route="fp8" if group == 2 else "rectangular",
                constants=config,
            )
        )
        chunks.append(
            dict(
                score_args=(
                    *args[:5],
                    storage[64:-64].view(count, width),
                    args[7],
                    width,
                    offset,
                    count,
                ),
                score_kwargs={},
            )
        )
        buffers.append(storage)
        offset += count
    pair = gated.make_pair(production, traces)
    for module in pair:
        for storage in buffers:
            storage.fill_(-12345.0)
        scoring.validate_scores(
            module._prefill_logits,
            chunks,
            args,
            torch,
            lambda n: list(range(n)),
            device,
        )
        for chunk, storage in zip(chunks, buffers):
            off, count = chunk["score_args"][-2:]
            logits = chunk["score_args"][5]
            mask = (
                torch.arange(width, device=args[0].device)[None, :]
                < args[4][off : off + count, None]
            )
            if (
                not bool((storage[:64] == -12345.0).all())
                or not bool((storage[-64:] == -12345.0).all())
                or not bool((logits[~mask] == -12345.0).all())
            ):
                raise RuntimeError(
                    "Expanded head-group fixture changed masked/canary output"
                )
    records = gated.freeze_records(pair)
    active = sum(v["head_group_gate"]["active"] for v in records["candidate"])
    if active != (len(counts) if group == 1 else 0):
        raise RuntimeError("Expanded fixture entered incorrect gate")
    return dict(
        **spec,
        status="passed",
        tile=tile,
        warps=warps,
        active_chunks=active,
        original_binary_fallback_chunks=len(counts) - active,
        all_rows_checked=True,
    )


def boundary_checks(production, helpers, torch, device):
    checks = BOUNDARIES(production, helpers, torch, device)
    for group in checks:
        group["expanded"] = []
    for spec in fixtures():
        result = check_fixture(production, helpers, torch, device, spec)
        checks[spec["group"]]["expanded"].append(result)
        print("Expanded correctness fixture:", result, flush=True)
    return checks


def coordinate(options):
    code = COORDINATE(options)
    path = Path(options.report_dir) / "head-group-summary.json"
    if code or not path.is_file():
        return code
    result = json.loads(path.read_text())
    groups = {"upstream_bf16": [], "observed_bf16": [], "fp8": []}
    cases = {v["case"]["name"]: v["case"] for v in manifest()["cases"]}
    for item in result["cases"]:
        case = cases[item["case"]]
        group = (
            "fp8"
            if case["dtype"] == "float8_e4m3fn"
            else (
                "upstream_bf16" if case["kind"] == "upstream_perf" else "observed_bf16"
            )
        )
        groups[group].append(item)
    result["group_aggregates"] = {
        name: dict(
            count=len(items),
            **{
                stage: statistics.mean(v["stages"][stage]["speedup"] for v in items)
                for stage in ("score", "topk", "pipeline")
            },
        )
        for name, items in groups.items()
        if items
    }
    path.write_text(json.dumps(result, indent=2) + "\n")
    from tools.qsa_prefill_report import fmt, table

    print("\nIncremental production / gated candidate (same compiler):", flush=True)
    table(
        ("group", "count", "score", "topk", "pipeline"),
        [
            (
                name,
                values["count"],
                *(fmt(values[k], 4) for k in ("score", "topk", "pipeline")),
            )
            for name, values in result["group_aggregates"].items()
        ],
    )
    return code


def main():
    # Reuse the proven CLI, fixed-config worker and subprocess launch protocol.
    # __file__ is scoped so child processes invoke this regression script.
    with contextlib.ExitStack() as stack:
        for name, value in (
            ("__doc__", __doc__),
            ("manifest", manifest),
            ("verify_source", verify_source),
            ("__file__", str(Path(__file__).resolve())),
            ("EXTRA_SOURCES", gated.EXTRA_SOURCES | EXTRA_SOURCES),
        ):
            stack.enter_context(patch.object(gated, name, value))
        stack.enter_context(
            patch.object(gated.base, "boundary_checks", boundary_checks)
        )
        stack.enter_context(patch.object(gated.base, "coordinate", coordinate))
        return gated.main()


if __name__ == "__main__":
    sys.exit(main())
