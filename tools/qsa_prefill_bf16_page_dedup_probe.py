# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Fixed-config BF16 page-ID deduplication against current integrated production."""

import contextlib
import json
import statistics
import sys
import types
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_bf16_stage_depth_probe as stage  # noqa: E402

base = stage.base
FIXED_TRACES = stage.fixed_traces
FREEZE = stage.FREEZE
BOUNDARIES = stage.BOUNDARIES
STAGE_PATH = Path(stage.__file__).resolve()
KERNEL = ROOT / "tools/qsa_prefill_bf16_page_dedup_kernel.py"
KERNEL_SHA = "6563310a78e6789a88cd66612995964a6022aefcc63270633ccf2c076febcbd5"
EXTRA_SOURCES = stage.EXTRA_SOURCES | {
    "tools/qsa_prefill_bf16_page_dedup_probe.py",
    "tools/qsa_prefill_bf16_page_dedup_kernel.py",
    "tools/test_qsa_prefill_bf16_page_dedup_probe.py",
    "tools/run_qsa_prefill_bf16_page_dedup_h800.sh",
    "docs/qsa_prefill_bf16_page_dedup_probe.md",
}
manifest = stage.manifest


def verify_source():
    source = stage.verify_source()
    if base.sha(KERNEL) != KERNEL_SHA:
        raise RuntimeError("Private page-dedup kernel changed")
    source.update(
        stage_driver_sha256=base.sha(STAGE_PATH),
        driver_sha256=base.sha(Path(__file__)),
        candidate_sha256=KERNEL_SHA,
    )
    return source


def eligible(named):
    block, page = named.get("BLOCK_N"), named.get("PAGE_SIZE")
    return (
        named.get("IS_FP8") is False
        and block is not None
        and page is not None
        and 0 < block <= page
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
                PAGE_SIZE=named["PAGE_SIZE"],
                BLOCK_N=named["BLOCK_N"],
            )
            return (self.candidate if active else self.original)[grid](
                *values, **kwargs
            )

        return launch


def unchanged_traces(traces, candidate=False):
    del candidate
    return FIXED_TRACES(traces, False)


def make_pair(production, traces, kernel=None):
    if kernel is None:
        from tools.qsa_prefill_bf16_page_dedup_kernel import _page_dedup_score_kernel

        kernel = _page_dedup_score_kernel
    with patch.object(stage, "fixed_traces", unchanged_traces):
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
        raise RuntimeError("Missing/extra actual page-dedup decisions")
    for record in records["candidate"]:
        key = tuple(record["chunk"])
        old = original[key]
        if record["constants"] != old["constants"] or record["route"] != old["route"]:
            raise RuntimeError("Page dedup changed production configurations")
        decision = decisions[key]
        record["page_dedup_gate"] = decision
        if not decision["active"]:
            if any(record[k] != old[k] for k in ("ptx_sha256", "cubin_sha256")):
                raise RuntimeError("Page-dedup fallback changed original binary")
            record["original_binary_verified"] = True
    return records


def boundary_checks(production, helpers, torch, device):
    checks = None
    for page in (196, 64, 16):

        def inputs(*args, **kwargs):
            return helpers.make_inputs(*args, **dict(kwargs, page_size=page))

        current = BOUNDARIES(
            production, types.SimpleNamespace(make_inputs=inputs), torch, device
        )
        for item in current:
            item.update(
                page_size=page,
                page_dedup_active=item["dtype"] == "torch.bfloat16" and page >= 64,
                num_stages=2,
                head_group=item["dtype"] == "torch.bfloat16" and item["tile"] == 64,
            )
        if checks is None:
            checks = current
            for item in checks:
                item["extra_page_checks"] = []
        else:
            for item, extra in zip(checks, current):
                item["extra_page_checks"].append(extra)
        print("Page-dedup boundary evidence:", current, flush=True)
    return checks


def show(result):
    from tools.qsa_prefill_report import fmt, table

    print(
        "\nBF16 page-dedup probe:",
        result.get("case", {}).get("name"),
        result.get("status"),
        flush=True,
    )
    for title, rows in (
        ("Production / dedup candidate", result.get("stages", {})),
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
            algorithm="bf16_scalar_page_id_dedup",
            config_scope="same captured production configs/head groups/stages; page-ID fetch only",
            rounds=options.rounds,
            gate_summary=dict(
                active_chunks=sum(
                    bool(r.get("page_dedup_gate", {}).get("active")) for r in records
                ),
                original_binary_fallback_chunks=sum(
                    bool(r.get("original_binary_verified")) for r in records
                ),
            ),
        )
        result["self_controls"] = {
            k.replace("head_group:", "page_dedup:"): v
            for k, v in result.get("self_controls", {}).items()
        }
        path.write_text(json.dumps(result, indent=2) + "\n")
        tsv = path.with_suffix(".tsv")
        if tsv.is_file():
            tsv.write_text(tsv.read_text().replace("head_group_us", "page_dedup_us", 1))
        print("Actual page-dedup dispatch:", result["gate_summary"], flush=True)
        show(result)
    return code


def coordinate(options):
    code = base.coordinate(options)
    old = Path(options.report_dir) / "head-group-summary.json"
    if old.is_file():
        result = json.loads(old.read_text())
        result["probe"] = "bf16_scalar_page_id_dedup"
        result["group_aggregates"] = {}
        dtype = {v["case"]["name"]: v["case"]["dtype"] for v in manifest()["cases"]}
        for label, fp8 in (("changed_bf16", False), ("unchanged_fp8_control", True)):
            values = [
                c
                for c in result["cases"]
                if c["status"] == "passed"
                and (dtype[c["case"]] == "float8_e4m3fn") == fp8
            ]
            if values:
                result["group_aggregates"][label] = dict(
                    count=len(values),
                    **{
                        k: statistics.mean(c["stages"][k]["speedup"] for c in values)
                        for k in ("score", "topk", "pipeline")
                    },
                )
        print(
            "\nPage-dedup group arithmetic means:",
            json.dumps(result["group_aggregates"], indent=2),
            flush=True,
        )
        (old.parent / "page-dedup-summary.json").write_text(
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
