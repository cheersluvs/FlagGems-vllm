# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Full-head query-half decomposition versus current head-group production."""

import contextlib
import json
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
KERNEL = ROOT / "tools/qsa_prefill_bf16_query_retest_kernel.py"
KERNEL_SHA = "9af2cbdba90d7decc2123d788718491fdb8fc83c27550c46714a354e023b4585"
EXTRA_SOURCES = stage.EXTRA_SOURCES | {
    "tools/qsa_prefill_bf16_query_retest.py",
    "tools/qsa_prefill_bf16_query_retest_kernel.py",
    "tools/test_qsa_prefill_bf16_query_retest.py",
    "tools/run_qsa_prefill_bf16_query_retest_h800.sh",
    "docs/qsa_prefill_bf16_query_retest.md",
}
manifest = stage.manifest


def verify_source():
    source = stage.verify_source()
    if base.sha(KERNEL) != KERNEL_SHA:
        raise RuntimeError("Private query-split kernel changed")
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
        and named.get("TILE_R") == 64
        and named.get("K_TILES") == 16
    )


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
            active = self.route == "rectangular" and eligible(named)
            self.decisions[base.fixed.signature(named)] = dict(
                active=active,
                IS_FP8=named["IS_FP8"],
                NUM_HEADS=named["NUM_HEADS"],
                HEAD_DIM=named["HEAD_DIM"],
                TILE_R=named["TILE_R"],
                K_TILES=named["K_TILES"],
            )
            return (self.candidate if active else self.original)[grid](
                *values, **kwargs
            )

        return launch


def split_traces(traces, candidate=False):
    result = FIXED_TRACES(traces, False)
    if candidate:
        for t in result:
            config = t["constants"]
            if (
                t["route"] == "rectangular"
                and config["TILE_R"] == 64
                and config["K_TILES"] == 16
            ):
                config["HEAD_GROUP"] = False
    return result


def make_pair(production, traces, kernel=None):
    if kernel is None:
        from tools.qsa_prefill_bf16_query_retest_kernel import _query_split_score_kernel

        kernel = _query_split_score_kernel
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
        raise RuntimeError("Missing/extra actual query-split decisions")
    for record in records["candidate"]:
        key = tuple(record["chunk"])
        old = original[key]
        decision = decisions[key]
        expected = dict(old["constants"])
        if decision["active"]:
            expected["HEAD_GROUP"] = False
        if record["constants"] != expected or record["route"] != old["route"]:
            raise RuntimeError("Query split changed non-algorithm configurations")
        record["query_split_gate"] = decision
        record["head_group_replaced"] = bool(
            decision["active"] and old["constants"].get("HEAD_GROUP")
        )
        if not decision["active"]:
            if any(record[k] != old[k] for k in ("ptx_sha256", "cubin_sha256")):
                raise RuntimeError("Query-split fallback changed original binary")
            record["original_binary_verified"] = True
    return records


def boundary_fixture(production, helpers, torch, device, dtype, tile, warps, counts):
    from tools import qsa_prefill_score_benchmark as scoring

    rows, width = 129, 1088
    levels = (0, 1, 31, 32, 33, 63, 64, 65, 195, 196, 197, 511, 512, 513, 1025)
    visible = [levels[i % len(levels)] for i in range(rows)]
    visible[-1] = 1025
    args = helpers.make_inputs(
        [1, 65, 0, 63],
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
    if offset != rows:
        raise RuntimeError("Query split boundary row mismatch")
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
                not bool((storage[:64] == -12345.0).all())
                or not bool((storage[-64:] == -12345.0).all())
                or not bool((logits[~mask] == -12345.0).all())
            ):
                raise RuntimeError("Query split boundary changed canary/masked output")
    records = freeze_records(pair)
    active = sum(v["query_split_gate"]["active"] for v in records["candidate"])
    if active != (len(counts) if not fp8 and tile == 64 else 0):
        raise RuntimeError("Query split boundary dispatch mismatch")
    return dict(
        rows=rows,
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
            for counts in ((129,), (33, 64, 32))
        ]
        item = dict(
            dtype=str(dtype),
            tile=tile,
            warps=warps,
            rows=129,
            schedules=schedules,
            status="passed",
        )
        print("Query split all-row boundary evidence:", item, flush=True)
        checks.append(item)
    return checks


def show(result):
    from tools.qsa_prefill_report import fmt, table

    print(
        "\nBF16 query-split probe:",
        result.get("case", {}).get("name"),
        result.get("status"),
        flush=True,
    )
    for title, rows in (
        ("Production head groups / full-head query halves", result.get("stages", {})),
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
    code = base.worker(options)
    path = Path(options.report_dir) / (options.worker_case + ".json")
    if path.is_file():
        result = json.loads(path.read_text())
        records = result.get("kernels_before", {}).get("candidate", [])
        result.update(
            algorithm="bf16_large_tile_query_split_retest",
            config_scope=(
                "current head-group baseline vs full-head query halves; "
                "same tile/warp/stage; active HEAD_GROUP=false"
            ),
            rounds=options.rounds,
            gate_summary=dict(
                active_chunks=sum(
                    bool(r.get("query_split_gate", {}).get("active")) for r in records
                ),
                original_binary_fallback_chunks=sum(
                    bool(r.get("original_binary_verified")) for r in records
                ),
            ),
        )
        result["self_controls"] = {
            k.replace("head_group:", "query_split:"): v
            for k, v in result.get("self_controls", {}).items()
        }
        path.write_text(json.dumps(result, indent=2) + "\n")
        tsv = path.with_suffix(".tsv")
        if tsv.is_file():
            tsv.write_text(
                tsv.read_text().replace("head_group_us", "query_split_us", 1)
            )
        print("Actual query-split dispatch:", result["gate_summary"], flush=True)
        show(result)
    return code


def coordinate(options):
    code = base.coordinate(options)
    old = Path(options.report_dir) / "head-group-summary.json"
    if old.is_file():
        result = json.loads(old.read_text())
        result["probe"] = "bf16_large_tile_query_split_retest"
        result["group_aggregates"] = {}
        for label, active in (
            ("query_split_active", True),
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
            "\nQuery-split group arithmetic means:",
            json.dumps(result["group_aggregates"], indent=2),
            flush=True,
        )
        (old.parent / "query-split-summary.json").write_text(
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


def main():
    with patch.object(stage, "__doc__", __doc__), patch.object(
        stage, "scoped", scoped
    ), patch.object(stage, "worker", worker), patch.object(
        stage, "coordinate", coordinate
    ):
        return stage.main()


if __name__ == "__main__":
    sys.exit(main())
