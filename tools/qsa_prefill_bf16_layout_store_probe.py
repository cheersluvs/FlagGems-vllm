# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""BF16 query-major layout with aligned bulk/tail stores, fixed production configs."""

import contextlib
import json
import re
import statistics
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_bf16_stage_depth_probe as stage  # noqa: E402

base = stage.base
FIXED_TRACES = stage.fixed_traces
FREEZE = stage.FREEZE
STAGE_PATH = Path(stage.__file__).resolve()
KERNEL = ROOT / "tools/qsa_prefill_bf16_layout_store_kernel.py"
KERNEL_SHA = "3f9a895e0e5807ab4502e42bdfbfe88de5ce56b24944a7b8b1d3c5bcee21b19f"
EXTRA_SOURCES = stage.EXTRA_SOURCES | {
    "tools/qsa_prefill_bf16_layout_store_probe.py",
    "tools/qsa_prefill_bf16_layout_store_kernel.py",
    "tools/test_qsa_prefill_bf16_layout_store_probe.py",
    "tools/run_qsa_prefill_bf16_layout_store_h800.sh",
    "docs/qsa_prefill_bf16_layout_store_probe.md",
}
manifest = stage.manifest
RESOURCE_DIR = None
RESOURCE_INDEX = 0


def verify_source():
    source = stage.verify_source()
    if base.sha(KERNEL) != KERNEL_SHA:
        raise RuntimeError("Private layout kernel changed")
    source.update(
        stage_driver_sha256=base.sha(STAGE_PATH),
        driver_sha256=base.sha(Path(__file__)),
        candidate_sha256=KERNEL_SHA,
    )
    return source


def eligible(named):
    return (
        named.get("IS_FP8") is False
        and named.get("NUM_HEADS") == 4
        and named.get("HEAD_DIM") == 128
        and named.get("TILE_R") in (32, 64)
        and named.get("K_TILES") == 16
        and named.get("store_aligned") is True
    )


def aligned_logits(named):
    logits = named.get("logits_ptr")
    width = named.get("logits_width", 0)
    return (
        logits is not None
        and str(logits.dtype) == "torch.float32"
        and logits.ndim == 2
        and logits.shape[1] == width
        and logits.is_contiguous()
        and width > 0
        and width % 4 == 0
        and logits.data_ptr() % 16 == 0
    )


def store_sites(pair):
    result = {}
    for name, _ in base.ENTRIES:
        for key, (kernel, _) in getattr(pair[1], name).compiled.items():
            ptx = kernel.asm.get("ptx", "")
            # Ignore comments; these are static sites, not dynamic traffic.
            ptx = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)
            result[key] = {
                kind: len(re.findall(pattern, ptx))
                for kind, pattern in (
                    ("v4", r"\bst\.global(?:\.[\w:]+)*\.v4\.b32\b"),
                    ("v2", r"\bst\.global(?:\.[\w:]+)*\.v2\.b32\b"),
                    ("scalar", r"\bst\.global\.b32\b"),
                )
            }
    return result


class RoutedJIT:
    def __init__(self, original, candidate, arg_names, route):
        self.original, self.candidate, self.arg_names, self.route = (
            original,
            candidate,
            arg_names,
            route,
        )
        self.decisions = {}

    def __getitem__(self, grid):
        def launch(*values, **kwargs):
            named = dict(zip(self.arg_names, values))
            named.update(kwargs)
            named["store_aligned"] = aligned_logits(named)
            active = self.route == "rectangular" and eligible(named)
            self.decisions[base.fixed.signature(named)] = dict(
                active=active,
                IS_FP8=named["IS_FP8"],
                NUM_HEADS=named["NUM_HEADS"],
                HEAD_DIM=named["HEAD_DIM"],
                TILE_R=named["TILE_R"],
                K_TILES=named["K_TILES"],
                store_aligned=named["store_aligned"],
                logits_width=named["logits_width"],
            )
            return (self.candidate if active else self.original)[grid](
                *values, **kwargs
            )

        return launch


def split_traces(traces, candidate=False):
    return FIXED_TRACES(traces, False)


def make_pair(production, traces, kernel=None):
    if kernel is None:
        from tools.qsa_prefill_bf16_layout_store_kernel import (
            _layout_store_score_kernel,
        )

        kernel = _layout_store_score_kernel
    with patch.object(stage, "fixed_traces", split_traces):
        pair = stage.make_pair(production, traces)
    for name, route in base.ENTRIES:
        entry = getattr(pair[1], name)
        entry.jit = RoutedJIT(entry.jit, kernel, entry.arg_names, route)
    return pair


def freeze_records(pair):
    records = FREEZE(pair)
    original = {tuple(r["chunk"]): r for r in records["baseline"]}
    decisions = {}
    for name, _ in base.ENTRIES:
        decisions.update(getattr(pair[1], name).jit.decisions)
    if set(decisions) != set(original):
        raise RuntimeError("Missing/extra actual layout decisions")
    sites = store_sites(pair)
    for record in records["candidate"]:
        key = tuple(record["chunk"])
        record["store_sites"] = sites[key]
        old = original[key]
        decision = decisions[key]
        expected = dict(old["constants"])
        if record["constants"] != expected or record["route"] != old["route"]:
            raise RuntimeError("Layout changed production configurations")
        record["layout_gate"] = decision
        record["head_group_preserved"] = record["constants"].get("HEAD_GROUP") == old[
            "constants"
        ].get("HEAD_GROUP")
        if not decision["active"]:
            if any(record[k] != old[k] for k in ("ptx_sha256", "cubin_sha256")):
                raise RuntimeError("Layout fallback changed original binary")
            record["original_binary_verified"] = True
    resource_gate(records, pair)
    return records


def resource_gate(records, pair):
    global RESOURCE_INDEX
    if RESOURCE_DIR is not None:
        RESOURCE_INDEX += 1
        prefix = f"resource-{RESOURCE_INDEX:03d}"
        artifacts = base.artifact_records(pair, RESOURCE_DIR, prefix)
        (RESOURCE_DIR / (prefix + ".json")).write_text(
            json.dumps(dict(records=records, artifacts=artifacts), indent=2) + "\n"
        )
    for record in records["candidate"]:
        if not record["layout_gate"]["active"]:
            continue
        print(
            "Layout compile resources:",
            record["chunk"],
            "registers",
            record["registers_per_thread"],
            "shared",
            record["shared_bytes"],
            "spills",
            record["spill_count"],
            "stores",
            record["store_sites"],
            flush=True,
        )
        if (
            record["spill_count"] is None
            or record["spill_count"] != 0
            or record["registers_per_thread"] is None
            or record["shared_bytes"] is None
        ):
            raise RuntimeError(
                "Layout resource gate rejected; artifacts retained; no timing"
            )

        if record["store_sites"]["v4"] <= 0:
            raise RuntimeError(
                "Layout v4 store gate rejected; artifacts retained; no timing"
            )


def boundary_fixture(
    production,
    helpers,
    torch,
    device,
    dtype,
    tile,
    warps,
    counts,
    *,
    width=1088,
    shift=64,
):
    from tools import qsa_prefill_score_benchmark as scoring

    rows = 259
    levels = (0, 1, 31, 32, 33, 63, 64, 65, 195, 196, 197, 511, 512, 513, 1025)
    visible = [levels[i % len(levels)] for i in range(rows)]
    visible[-1] = 1025
    args = helpers.make_inputs(
        [1, 129, 0, 129],
        [1025] * 4,
        dtype=dtype,
        padded=True,
        query_base=97,
        page_size=196,
        visible=visible,
    )
    traces, chunks, buffers = [], [], []
    offset = 0
    fp8 = dtype == torch.float8_e4m3fn
    for count in counts:
        storage = torch.full(
            (count * width + shift + 64,),
            -12345.0,
            dtype=torch.float32,
            device=args[0].device,
        )
        config = dict(
            TILE_R=tile,
            BLOCK_N=64,
            K_TILES=16,
            num_warps=warps,
            num_stages=2,
            num_ctas=1,
        )
        if fp8:
            config["QUERY_PARTS"] = 2
        else:
            config["HEAD_GROUP"] = tile == 64
        traces.append(
            dict(
                offset=offset,
                rows=count,
                width=width,
                route="fp8" if fp8 else "rectangular",
                constants=config,
            )
        )
        chunks.append(
            dict(
                score_args=(
                    *args[:5],
                    storage[shift:-64].view(count, width),
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
    if offset != rows:
        raise RuntimeError("Layout boundary row mismatch")
    pair = make_pair(production, traces)
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
                not bool((storage[:shift] == -12345.0).all())
                or not bool((storage[-64:] == -12345.0).all())
                or not bool((logits[~mask] == -12345.0).all())
            ):
                raise RuntimeError("Layout boundary changed canary/masked output")
    records = freeze_records(pair)
    active = sum(v["layout_gate"]["active"] for v in records["candidate"])
    expected_active = not fp8 and tile in (32, 64) and width % 4 == 0 and shift % 4 == 0
    if active != (len(counts) if expected_active else 0):
        raise RuntimeError("Layout boundary dispatch mismatch")
    return dict(
        rows=rows,
        width=width,
        shift=shift,
        counts=list(counts),
        all_rows_checked=True,
        active_chunks=active,
        status="passed",
    )


def boundary_checks(production, helpers, torch, device):
    checks = []
    for dtype, tile, warps in (
        (torch.bfloat16, 32, 8),
        (torch.bfloat16, 64, 4),
        (torch.float8_e4m3fn, 64, 4),
    ):
        schedules = [
            boundary_fixture(
                production, helpers, torch, device, dtype, tile, warps, counts
            )
            for counts in ((259,), (127, 128, 4))
        ]
        fallback_checks = []
        if dtype == torch.bfloat16:
            for width, shift in ((1088, 65), (1087, 64)):
                fallback_checks.append(
                    boundary_fixture(
                        production,
                        helpers,
                        torch,
                        device,
                        dtype,
                        tile,
                        warps,
                        (259,),
                        width=width,
                        shift=shift,
                    )
                )
        item = dict(
            dtype=str(dtype),
            tile=tile,
            warps=warps,
            rows=259,
            schedules=schedules,
            fallback_checks=fallback_checks,
            status="passed",
        )
        print("Layout all-row boundary evidence:", item, flush=True)
        checks.append(item)
    return checks


def show(result):
    from tools.qsa_prefill_report import fmt, table

    print(
        "\nBF16 layout probe:",
        result.get("case", {}).get("name"),
        result.get("status"),
        flush=True,
    )
    for title, rows in (
        (
            "Current production / query-major layout plus bulk/tail stores",
            result.get("stages", {}),
        ),
        ("Identical-callable A/A", result.get("self_controls", {})),
    ):
        print(title, flush=True)
        table(
            (
                "stage",
                "production_us",
                "candidate_us",
                "speedup",
                "paired_min",
                "paired_max",
            ),
            [
                (
                    k,
                    *(fmt(v[f]["median"], 3) for f in ("baseline_us", "candidate_us")),
                    fmt(v["speedup"], 5),
                    fmt(v["paired_ratios"]["min"], 5),
                    fmt(v["paired_ratios"]["max"], 5),
                )
                for k, v in rows.items()
                if v.get("measurement_validated")
            ],
        )
    if result.get("error"):
        print(result["error"], flush=True)


def worker(options):
    with patch.object(sys.modules[__name__], "RESOURCE_DIR", Path(options.report_dir)):
        code = base.worker(options)
    path = Path(options.report_dir) / (options.worker_case + ".json")
    if path.is_file():
        result = json.loads(path.read_text())
        records = result.get("kernels_before", {}).get("candidate", [])
        result.update(
            algorithm="bf16_query_major_layout_bulk_tail",
            config_scope=(
                "rectangular BF16 query-major dot/reduction with aligned bulk/tail stores; same tiles, head grouping, "
                "key tiling, warp and stage"
            ),
            rounds=options.rounds,
            gate_summary=dict(
                active_chunks=(
                    sum(bool(r.get("layout_gate", {}).get("active")) for r in records)
                    if records
                    else None
                ),
                original_binary_fallback_chunks=(
                    sum(bool(r.get("original_binary_verified")) for r in records)
                    if records
                    else None
                ),
            ),
        )
        result["self_controls"] = {
            k.replace("head_group:", "layout:"): v
            for k, v in result.get("self_controls", {}).items()
        }
        path.write_text(json.dumps(result, indent=2) + "\n")
        tsv = path.with_suffix(".tsv")
        if tsv.is_file():
            tsv.write_text(tsv.read_text().replace("head_group_us", "layout_us", 1))
        print("Actual layout dispatch:", result["gate_summary"], flush=True)
        show(result)
    return code


def coordinate(options):
    code = base.coordinate(options)
    old = Path(options.report_dir) / "head-group-summary.json"
    if old.is_file():
        result = json.loads(old.read_text())
        result["probe"] = "bf16_query_major_layout_bulk_tail"
        result["group_aggregates"] = {}
        for label, active in (
            ("layout_active", True),
            ("unchanged_controls", False),
        ):
            values = []
            for c in result["cases"]:
                if c["status"] != "passed":
                    continue
                detail = json.loads((old.parent / (c["case"] + ".json")).read_text())
                if bool(detail["gate_summary"]["active_chunks"]) == active:
                    values.append(c)
            if values:
                result["group_aggregates"][label] = dict(
                    count=len(values),
                    **{
                        k: statistics.mean(c["stages"][k]["speedup"] for c in values)
                        for k in ("score", "topk", "pipeline")
                    },
                )
        print(
            "\nLayout group arithmetic means:",
            json.dumps(result["group_aggregates"], indent=2),
            flush=True,
        )
        (old.parent / "layout-store-summary.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        old.unlink()
    return code


@contextlib.contextmanager
def scoped():
    from tools import benchmark_qsa_prefill_triton_stages as bench

    with contextlib.ExitStack() as stack:
        for name, value in (
            ("manifest", manifest),
            ("verify_source", verify_source),
            ("make_pair", make_pair),
            ("freeze_records", freeze_records),
            ("boundary_checks", boundary_checks),
            ("show", lambda result: None),
            ("__file__", str(Path(__file__).resolve())),
        ):
            stack.enter_context(patch.object(base, name, value))
        stack.enter_context(
            patch.object(bench, "SOURCES", sorted(set(bench.SOURCES) | EXTRA_SOURCES))
        )
        yield


DEFAULT_CASES = (
    "perf_n1_q8192_p102400_bfloat16",
    "perf_n4_q4096_p102400_bfloat16",
    "fp8_long100k_float8_e4m3fn",
)


def main():
    if "--cases" not in sys.argv:
        sys.argv.extend(["--cases", *DEFAULT_CASES])
    with patch.object(stage, "__doc__", __doc__), patch.object(
        stage, "scoped", scoped
    ), patch.object(stage, "worker", worker), patch.object(
        stage, "coordinate", coordinate
    ):
        return stage.main()


if __name__ == "__main__":
    sys.exit(main())
