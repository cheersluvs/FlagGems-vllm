# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Fixed-tile BF16 stage2 versus stage1 probe on the integrated production JIT."""

import argparse
import contextlib
import copy
import json
import statistics
import sys
import types
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_head_group_probe as base  # noqa: E402

BASE_PATH = Path(base.__file__).resolve()
FREEZE = base.freeze_records
BOUNDARIES = base.boundary_checks
MANIFEST = ROOT / "docs/qsa_prefill_bf16_stage_depth_configs.json"
MANIFEST_SHA = "19efb9803313e303ad503b0864c48c5253f3fc3674daef1fcdf8a652b702f583"
PRODUCTION_SHA = "0d7e80e240966565a7c78151c8344f17e0627e73920f5229aadcc0f0fb42f10e"
EXTRA_SOURCES = {
    "tools/qsa_prefill_bf16_stage_depth_probe.py",
    "tools/test_qsa_prefill_bf16_stage_depth_probe.py",
    "tools/run_qsa_prefill_bf16_stage_depth_h800.sh",
    "docs/qsa_prefill_bf16_stage_depth_probe.md",
    "docs/qsa_prefill_bf16_stage_depth_configs.json",
}


def manifest():
    from tools import qsa_prefill_observed_workloads as workloads

    if base.sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("Stage-depth manifest changed")
    data = json.loads(MANIFEST.read_text())
    if data["production_sha256"] != PRODUCTION_SHA or len(data["cases"]) != 6:
        raise RuntimeError("Stage-depth production/fixture set changed")
    expected = {c["name"]: c for c in workloads.cases("scored-prefill")}
    for item in data["cases"]:
        case = item["case"]
        if expected.get(case["name"]) != case:
            raise RuntimeError("Stage-depth fixture changed")
        shape = workloads.geometry(case)
        geometry = [
            (sum(shape["expected_chunk_rows"][:i]), n, shape["logits_width"])
            for i, n in enumerate(shape["expected_chunk_rows"])
        ]
        if [(t["offset"], t["rows"], t["width"]) for t in item["traces"]] != geometry:
            raise RuntimeError("Stage-depth chunk geometry changed")
        for t in item["traces"]:
            if t["route"] == "rectangular" and (
                case["dtype"] != "bfloat16" or t["constants"]["num_stages"] != 2
            ):
                raise RuntimeError("Stage-depth edits require BF16 rectangular stage2")
    return data


def verify_source():
    manifest()
    from tools import qsa_prefill_triton_ab as ab

    production = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
    if (
        base.sha(production) != PRODUCTION_SHA
        or ab.INTEGRATED_PRODUCTION_SHA256 != PRODUCTION_SHA
    ):
        raise RuntimeError("Integrated head-group production changed")
    return dict(
        production_sha256=PRODUCTION_SHA,
        manifest_sha256=MANIFEST_SHA,
        driver_sha256=base.sha(Path(__file__)),
        base_driver_sha256=base.sha(BASE_PATH),
    )


def fixed_traces(traces, candidate=False):
    result = copy.deepcopy(traces)
    for t in result:
        if t["route"] == "rectangular":
            config = t["constants"]
            # The three inherited small fixtures predate HEAD_GROUP metadata.
            # Their BF16/H4/D128 R64/W4 fixture now also checks grouped mode.
            config.setdefault(
                "HEAD_GROUP", config["TILE_R"] == 64 and config["num_warps"] == 4
            )
            if candidate:
                if config["num_stages"] != 2:
                    raise RuntimeError("Expected stage2 baseline")
                config["num_stages"] = 1
    return result


def make_pair(production, traces):
    result = []
    for index in (0, 1):
        module = types.ModuleType(production.__name__ + f".stage_depth{index}")
        module.__dict__.update(vars(production))
        for name, route in base.ENTRIES:
            setattr(
                module,
                name,
                base.fixed.FixedEntry(
                    getattr(production, name), fixed_traces(traces, bool(index)), route
                ),
            )
        for name in ("_prefill_logits", "qsa_select_paged_prefill"):
            original = getattr(production, name)
            fn = types.FunctionType(
                original.__code__,
                vars(module),
                name,
                original.__defaults__,
                original.__closure__,
            )
            fn.__kwdefaults__ = original.__kwdefaults__
            setattr(module, name, fn)
        result.append(module)
    if result[0].persistent_topk is not result[1].persistent_topk:
        raise RuntimeError("Stage probe changed TopK")
    return tuple(result)


def freeze_records(pair):
    records = FREEZE(pair)
    left = {tuple(r["chunk"]): r for r in records["baseline"]}
    if {tuple(r["chunk"]) for r in records["candidate"]} != set(left):
        raise RuntimeError("Stage probe chunk mismatch")
    for r in records["candidate"]:
        original = left[tuple(r["chunk"])]
        changed = r["route"] == "rectangular"
        expected = dict(original["constants"])
        if changed:
            if expected["num_stages"] != 2:
                raise RuntimeError("Stage probe baseline is not stage2")
            expected["num_stages"] = 1
        if r["constants"] != expected or r["route"] != original["route"]:
            raise RuntimeError("Stage probe changed more than pipeline depth")
        r["stage_depth_changed"] = changed
        if not changed:
            if any(r[k] != original[k] for k in ("ptx_sha256", "cubin_sha256")):
                raise RuntimeError("Stage probe fallback changed original binary")
            r["original_binary_verified"] = True
    return records


def boundary_checks(production, helpers, torch, device):
    checks = BOUNDARIES(production, helpers, torch, device)
    for item in checks:
        bf16 = item["dtype"] == "torch.bfloat16"
        item.update(
            baseline_num_stages=2,
            candidate_num_stages=1 if bf16 else 2,
            head_group=bf16 and item["tile"] == 64 and item["warps"] == 4,
            original_binary_fallback_verified=not bf16,
        )
    return checks


def show(result):
    from tools.qsa_prefill_report import fmt, table

    print(
        "\nBF16 stage-depth probe:",
        result.get("case", {}).get("name"),
        result.get("status"),
        flush=True,
    )
    for label, values in (
        ("Current production / candidate (FP8 unchanged)", result.get("stages", {})),
        ("Identical-callable A/A", result.get("self_controls", {})),
    ):
        print(label, flush=True)
        table(
            (
                "stage",
                "baseline_us",
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
                for k, v in values.items()
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
        result.update(
            algorithm="bf16_fixed_stage2_vs_stage1",
            config_scope="same production JIT/configs; only rectangular BF16 num_stages2->1",
            rounds=options.rounds,
        )
        result["self_controls"] = {
            k.replace("head_group:", "stage1:"): v
            for k, v in result.get("self_controls", {}).items()
        }
        path.write_text(json.dumps(result, indent=2) + "\n")
        tsv = path.with_suffix(".tsv")
        if tsv.is_file():
            tsv.write_text(
                tsv.read_text().replace("head_group_us", "candidate_stage1_us", 1)
            )
        show(result)
    return code


def coordinate(options):
    code = base.coordinate(options)
    old = Path(options.report_dir) / "head-group-summary.json"
    if old.is_file():
        result = json.loads(old.read_text())
        result["probe"] = "bf16_fixed_stage2_vs_stage1"
        result["baseline_code_commit"] = manifest()["baseline_code_commit"]
        result["group_aggregates"] = {}
        dtype = {v["case"]["name"]: v["case"]["dtype"] for v in manifest()["cases"]}
        for label, fp8 in (("changed_bf16", False), ("unchanged_fp8_control", True)):
            items = [
                c
                for c in result["cases"]
                if (dtype[c["case"]] == "float8_e4m3fn") == fp8
                and c["status"] == "passed"
            ]
            if items:
                result["group_aggregates"][label] = dict(
                    count=len(items),
                    **{
                        k: statistics.mean(v["stages"][k]["speedup"] for v in items)
                        for k in ("score", "topk", "pipeline")
                    },
                )
        print(
            "\nStage-depth group arithmetic means:",
            json.dumps(result["group_aggregates"], indent=2),
            flush=True,
        )
        (old.parent / "stage-depth-summary.json").write_text(
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--flagtree", default="/opt/flagtree")
    parser.add_argument("--rounds", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--kernel-warmup-ms", type=int, default=200)
    parser.add_argument("--kernel-rep-ms", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-initial-temp-c", type=int, default=80)
    parser.add_argument("--cooldown-timeout", type=int, default=120)
    parser.add_argument("--cooldown-interval", type=int, default=10)
    names = [v["case"]["name"] for v in manifest()["cases"]]
    parser.add_argument("--cases", choices=names, nargs="+", default=names)
    parser.add_argument("--worker-case", choices=names, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if (
        min(
            options.kernel_warmup_ms,
            options.kernel_rep_ms,
            options.timeout,
            options.cooldown_interval,
        )
        <= 0
        or options.cooldown_timeout < 0
        or not 1 <= options.max_initial_temp_c <= 150
    ):
        parser.error("Invalid timing/thermal limits")
    if len(set(options.cases)) != len(options.cases) or (
        options.worker_case and options.worker_case not in options.cases
    ):
        parser.error("Distinct cases and selected worker case required")
    options.report_dir = str(Path(options.report_dir).resolve())
    options.baseline_report = str(Path(options.baseline_report).resolve())
    options.compiler, options.candidate = "flagtree", "production"
    from tools.qsa_prefill_flagtree_ab import configure

    configure(options.flagtree)
    with scoped():
        return worker(options) if options.worker_case else coordinate(options)


if __name__ == "__main__":
    sys.exit(main())
