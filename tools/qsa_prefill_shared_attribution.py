# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Map production shared-memory dataflow and collect source-correlated PC samples."""

import argparse
import ast
import collections
import importlib
import json
import re
import shutil
import sys
import traceback
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_production_profile as current  # noqa: E402

RAW_SOURCE_METRICS = (
    "inst_executed",
    "memory_l1_wavefronts_shared",
    "memory_l1_wavefronts_shared_ideal",
)
FILES = {
    "tools/qsa_prefill_shared_attribution.py",
    "tools/test_qsa_prefill_shared_attribution.py",
    "tools/run_qsa_prefill_shared_attribution_h800.sh",
    "docs/qsa_prefill_shared_attribution.md",
}
SOURCE = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
# Exact pre-fix collector revision; production and all capture helpers are unchanged.
LEGACY_EXPORT_SOURCES = {
    "docs/qsa_prefill_shared_attribution.md": "9c23b0cada6aa42fe6dcb13c5d72f2ac26d13d5b24a4f5a94edc7077febd9e58",
    "tools/qsa_prefill_shared_attribution.py": "e9b94145361bbeab24d3e98ec23670deb982b8f044abdd1825003dbcc033c41c",
    "tools/test_qsa_prefill_shared_attribution.py": "bdda7e463fe19efdb6926b0ed79f024c8b6adce6bdbdc6d7d41822f30c03f63c",
}


@lru_cache(maxsize=1)
def source_regions():
    regions = []
    for fn in ast.parse(SOURCE.read_text()).body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        if fn.name == "_fp8_bulk_tail_store":
            regions.append((fn.lineno, fn.end_lineno, "output_layout_or_store"))
        if fn.name not in (
            "_qsa_prefill_score_kernel",
            "_qsa_prefill_fp8_store_score_kernel",
        ):
            continue
        for n in ast.walk(fn):
            region = None
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                name = n.targets[0].id if isinstance(n.targets[0], ast.Name) else ""
                f = n.value.func
                method = f.attr if isinstance(f, ast.Attribute) else ""
                if method == "load" and name.startswith("query"):
                    region = "Q_operand_staging"
                elif method == "load" and name == "keys":
                    region = "K_operand_feed"
                elif name == "max_visible":
                    region = "visible_max_reduction"
                elif method == "dot":
                    region = "dot_" + (name if name in ("low", "high") else "full")
                elif method == "sum":
                    region = "head_reduction"
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
                f = n.value.func
                if (isinstance(f, ast.Attribute) and f.attr == "store") or (
                    isinstance(f, ast.Name) and f.id == "_fp8_bulk_tail_store"
                ):
                    region = "output_layout_or_store"
            if region:
                regions.append((n.lineno, n.end_lineno, region))
            if isinstance(n, ast.For):
                regions.append((n.lineno, n.lineno, "key_loop_pipeline_or_control"))
    return regions


def region(file, line):
    if Path(file or "").name != SOURCE.name or not line:
        return "unmapped_or_inlined"
    matches = [
        (end - start, label)
        for start, end, label in source_regions()
        if start <= line <= end
    ]
    return min(matches)[1] if matches else "other_source_region"


def family(instruction):
    text = instruction.lower()
    for label, patterns in (
        ("shared_store", ("st.shared", "sts.")),
        ("shared_load", ("ld.shared", "ldmatrix", "lds.")),
        ("cta_barrier", ("bar.sync", "barrier.sync", "bar.sync")),
        ("mma_wait", ("wgmma.wait", "warpgroup.depbar", "warpgroup.wait")),
        ("async_copy_wait", ("cp.async.wait",)),
        ("dependency_wait", ("depbar",)),
        ("mma_fence", ("wgmma.fence", "warpgroup.arrive", "fence.proxy")),
        ("async_K_copy", ("cp.async", "ldgsts")),
        ("matrix_dot", ("wgmma.mma", "mma.sync", "hgmma", "hmma")),
        ("shuffle", ("shfl.",)),
        ("global_store", ("st.global", "stg.")),
    ):
        if any(v in text for v in patterns):
            return label
    return "other_instruction"


def static_map(ptx, ir):
    files = {
        int(i): path
        for i, path in re.findall(r'(?m)^\s*\.file\s+(\d+)\s+"([^"]+)"', ptx)
    }
    mapped, own_file, own_line = [], "", 0
    for index, line in enumerate(ptx.splitlines(), 1):
        loc = re.search(r"\.loc\s+(\d+)\s+(\d+)", line)
        if loc:
            own_file, own_line = files.get(int(loc[1]), ""), int(loc[2])
            outer = re.findall(r"qsa_select_paged_prefill\.py:(\d+):\d+", line)
            if outer:
                own_file, own_line = SOURCE.name, int(outer[-1])
            continue
        kind = family(line)
        if kind != "other_instruction" and not line.lstrip().startswith(("//", ".")):
            mapped.append(
                dict(
                    ptx_line=index,
                    source_file=own_file,
                    source_line=own_line,
                    region=region(own_file, own_line),
                    family=kind,
                    instruction=line.strip(),
                )
            )
    definitions = dict(re.findall(r"(?m)^(#loc\d*) = loc\((.*)\)$", ir))

    def resolve(key, seen=()):
        if key in seen:
            return []
        value = definitions.get(key, "")
        found = [
            (file, int(line))
            for file, line in re.findall(r'"([^"]+\.py)":(\d+):\d+', value)
        ]
        for ref in re.findall(r"#loc\d*", value):
            found.extend(resolve(ref, (*seen, key)))
        return found

    operations = []
    loop_indent = None
    for i, line in enumerate(ir.splitlines(), 1):
        indent = len(line) - len(line.lstrip())
        if "scf.for " in line:
            loop_indent = indent
        elif (
            loop_indent is not None
            and indent <= loop_indent
            and line.lstrip().startswith("}")
        ):
            loop_indent = None
        match = re.search(
            r"(ttg\.(?:convert_layout|local_alloc|local_load|async_wait)|"
            r"ttng\.(?:warp_group_dot_wait|warp_group_dot)|"
            r"tle\.wgmma_shared_operand_fence|tt\.reduce)\b(?!\.)",
            line,
        )
        loc = re.search(r"loc\((#loc\d*)\)", line)
        if match:
            locations = resolve(loc[1]) if loc else []
            own = next(
                (
                    (file, n)
                    for file, n in reversed(locations)
                    if Path(file).name == SOURCE.name
                ),
                ("", 0),
            )
            operations.append(
                dict(
                    ir_line=i,
                    scope=(
                        "per_key_iteration"
                        if loop_indent is not None
                        else "CTA_setup_or_drain"
                    ),
                    op=match[1],
                    source_file=own[0],
                    source_line=own[1],
                    region=region(*own),
                    text=line.strip(),
                )
            )
    counts = collections.Counter((v["region"], v["family"]) for v in mapped)
    return dict(
        kind="static_sites_not_cycles",
        ptx=mapped,
        ir=operations,
        counts=[
            dict(region=a, family=b, sites=n) for (a, b), n in sorted(counts.items())
        ],
    )


def map_record(report, name):
    record = json.loads((report / ("profile-" + name + ".json")).read_text())
    if (
        current.base.sha(SOURCE) != current.stage.PRODUCTION_SHA
        or record["source"]["production_sha256"] != current.stage.PRODUCTION_SHA
    ):
        raise RuntimeError("Static source identity changed")
    current.checked_artifacts(record, report)
    artifact = next(v for v in record["artifacts"] if v["chunk"][1] == 0)
    return static_map(
        (report / artifact["files"]["ptx"]["path"]).read_text(),
        (report / artifact["files"]["ttgir"]["path"]).read_text(),
    )


def select_source_metrics(ncu, command, path):
    query = command(
        [ncu, "--query-metrics-collection", "source", "--query-metrics-mode", "base"]
    )
    path.write_text(json.dumps(query, indent=2) + "\n")
    if query["returncode"]:
        raise RuntimeError("NCU source-metric query failed before capture")
    names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", query["stdout"]))
    return dict(
        selected=[m for m in RAW_SOURCE_METRICS if m in names],
        missing=[m for m in RAW_SOURCE_METRICS if m not in names],
    )


def report_module(ncu):
    roots = list(Path(ncu).resolve().parents)[:5]
    for root in roots:
        directory = root / "extras/python"
        if (directory / "ncu_report.py").is_file() or (
            directory / "ncu_report"
        ).is_dir():
            sys.path.insert(0, str(directory))
            return importlib.import_module("ncu_report")
    raise RuntimeError(
        "NCU extras/python/ncu_report unavailable; .ncu-rep and source exports remain reusable"
    )


def pc_rows(action):
    rows = {}
    metrics = []
    for name in action.metric_names():
        if not (
            name.startswith("smsp__pcsamp_")
            or name
            in (
                "inst_executed",
                "memory_l1_wavefronts_shared",
                "memory_l1_wavefronts_shared_ideal",
                "derived__memory_l1_conflicts_shared_nway",
            )
        ):
            continue
        metric = action.metric_by_name(name)
        if not metric.has_correlation_ids():
            continue
        ids = metric.correlation_ids()
        if ids.num_instances() != metric.num_instances():
            raise RuntimeError("PC correlation length mismatch: " + name)
        metrics.append(name)
        for i in range(metric.num_instances()):
            pc = int(ids.value(i))
            value = current.number(metric.value(i))
            if value is None:
                continue
            row = rows.setdefault(pc, dict(pc=pc, pc_hex=hex(pc), metrics={}))
            if name in row["metrics"]:
                raise RuntimeError("Duplicate PC value for metric: " + name)
            row["metrics"][name] = value
    for pc, row in rows.items():
        source = action.source_info(pc)
        file, line = (source.file_name(), source.line()) if source else ("", 0)
        instruction = action.sass_by_pc(pc)
        row.update(
            file=file,
            line=line,
            instruction=instruction,
            region=region(file, line),
            family=family(instruction),
        )
        values = row["metrics"]
        count = values.get("inst_executed")
        actual = values.get("memory_l1_wavefronts_shared")
        ideal = values.get("memory_l1_wavefronts_shared_ideal")
        row["shared_wavefronts_per_instruction"] = (
            actual / count if count and actual is not None else None
        )
        row["ideal_shared_wavefronts_per_instruction"] = (
            ideal / count if count and ideal is not None else None
        )
        row["excess_shared_wavefronts"] = (
            actual - ideal if actual is not None and ideal is not None else None
        )
    # Each reason is summarized separately. These fractions are not runtime %.
    reasons = []
    for name in metrics:
        if not re.fullmatch(r"smsp__pcsamp_warps_issue_stalled_[a-z_]+", name):
            continue
        groups = collections.defaultdict(float)
        for row in rows.values():
            groups[(row["region"], row["family"])] += row["metrics"].get(name, 0)
        total = sum(groups.values())
        reasons.append(
            dict(
                metric=name,
                total_samples=total,
                groups=[
                    dict(
                        region=a,
                        family=b,
                        samples=v,
                        fraction_of_this_reason=(v / total if total else None),
                    )
                    for (a, b), v in sorted(groups.items(), key=lambda v: -v[1])
                    if v > 0
                ],
            )
        )
    return dict(
        metrics=metrics,
        rows=sorted(rows.values(), key=lambda v: v["pc"]),
        reasons=reasons,
        missing_focus_reasons=[
            "smsp__pcsamp_warps_issue_stalled_" + v
            for v in ("barrier", "short_scoreboard", "long_scoreboard", "mio_throttle")
            if "smsp__pcsamp_warps_issue_stalled_" + v not in metrics
        ],
        positive_stall_samples=any(
            v["total_samples"] > 0
            and not v["metric"].endswith(("_selected", "_not_selected"))
            for v in reasons
        ),
    )


def export_source(ncu, report, name, command):
    rep = report / ("source-" + name + ".ncu-rep")
    for view, suffix in (("sass", ".sass.csv"), ("cuda,sass", ".source.txt")):
        cmd = [
            ncu,
            "--import",
            str(rep),
            "--page",
            "source",
            "--print-source",
            view,
            "--section",
            "SourceCounters",
        ]
        if suffix.endswith(".csv"):
            cmd.append("--csv")
        result = command(cmd)
        (report / ("source-" + name + suffix)).write_text(
            result.get("stdout", "") + result.get("stderr", "")
        )
        if result["returncode"]:
            raise RuntimeError("Source export failed: " + suffix)
    mod = report_module(ncu)
    context = mod.load_report(str(rep))
    actions = [
        context.range_by_idx(i).action_by_idx(j)
        for i in range(context.num_ranges())
        for j in range(context.range_by_idx(i).num_actions())
    ]
    if len(actions) != 1:
        raise RuntimeError("Expected exactly first score chunk in binary report")
    result = pc_rows(actions[0])
    result.update(
        case=name,
        kind="PC_samples_not_runtime_percent",
        report_path=str(rep),
        report_sha256=current.base.sha(rep),
        static=map_record(report, name),
    )
    (report / ("attribution-" + name + ".json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    show(result)
    return result


def show(result):
    print("\n指令级 shared/sync 归因：", result["case"], flush=True)
    current.table(
        ("source region", "instruction family", "static PTX sites"),
        [
            (v["region"], v["family"], v["sites"])
            for v in result["static"]["counts"]
            if v["family"]
            in (
                "shared_load",
                "shared_store",
                "cta_barrier",
                "async_copy_wait",
                "mma_wait",
                "mma_fence",
            )
        ],
    )
    if result.get("missing_focus_reasons"):
        print("缺失的主要 PC 原因：", result["missing_focus_reasons"], flush=True)
    shared = [
        v
        for v in result["rows"]
        if v["family"] in ("shared_load", "shared_store")
        and v["excess_shared_wavefronts"] is not None
    ]
    if shared:
        print("逐指令 shared wavefront（实际与理想同时保留；不是耗时）：", flush=True)
        current.table(
            (
                "PC",
                "source line",
                "region",
                "instruction",
                "actual/inst",
                "ideal/inst",
                "excess wavefronts",
            ),
            [
                (
                    v["pc_hex"],
                    v["line"],
                    v["region"],
                    v["instruction"],
                    current.fmt(v["shared_wavefronts_per_instruction"]),
                    current.fmt(v["ideal_shared_wavefronts_per_instruction"]),
                    current.fmt(v["excess_shared_wavefronts"]),
                )
                for v in sorted(shared, key=lambda v: -v["excess_shared_wavefronts"])[
                    :12
                ]
            ],
        )
    for reason in result["reasons"]:
        if not any(
            v in reason["metric"] for v in ("barrier", "scoreboard", "mio", "warpgroup")
        ):
            continue
        print(
            reason["metric"],
            "samples=",
            reason["total_samples"],
            "（样本占比不是耗时占比）",
            flush=True,
        )
        current.table(
            ("region", "instruction family", "samples", "fraction of this reason"),
            [
                (
                    v["region"],
                    v["family"],
                    current.fmt(v["samples"]),
                    current.fmt(v["fraction_of_this_reason"]),
                )
                for v in reason["groups"][:8]
            ],
        )
        current.table(
            ("PC", "file:line", "region", "instruction", "samples"),
            [
                (
                    v["pc_hex"],
                    f"{Path(v['file']).name}:{v['line']}",
                    v["region"],
                    v["instruction"],
                    current.fmt(v["metrics"].get(reason["metric"])),
                )
                for v in sorted(
                    result["rows"], key=lambda v: -v["metrics"].get(reason["metric"], 0)
                )[:8]
                if v["metrics"].get(reason["metric"], 0) > 0
            ],
        )
    if not result["positive_stall_samples"]:
        print("没有正的 PC 停顿样本，不能完成动态归因。", flush=True)


def capture_command(ncu, csv_path, rep, metrics, worker):
    return current.profiling.ncu_command(ncu, csv_path, [], metrics=metrics) + [
        "--section",
        "SourceCounters",
        "--launch-count",
        "1",
        "--import-source",
        "yes",
        "--source-folders",
        str(ROOT / "src"),
        "--export",
        str(rep),
        *worker,
    ]


def capture_source(record, prepared):
    """Permit only the exact 81206ae export-only revision when resuming."""
    with patch.object(current, "EXTRA_SOURCES", current.EXTRA_SOURCES | FILES):
        expected = current.verify_source()
    if record.get("source") != prepared.get("source"):
        raise RuntimeError("Preparation/capture source mismatch")
    if record.get("source") == expected:
        return expected
    legacy = dict(expected, profile_sources=dict(expected["profile_sources"]))
    legacy["profile_sources"].update(LEGACY_EXPORT_SOURCES)
    if record.get("source") != legacy:
        raise RuntimeError("Unknown capture source revision; refusing resume")
    return legacy


def validate_capture(report, name, selection):
    item = next(v for v in current.manifest()["cases"] if v["case"]["name"] == name)
    record = json.loads((report / ("profile-" + name + ".json")).read_text())
    prepared = json.loads((report / ("prepared-" + name + ".json")).read_text())
    source = capture_source(record, prepared)
    with patch.object(current, "verify_source", return_value=source):
        current.checked_capture(record, item, prepared)
    current.checked_artifacts(record, report)
    current.checked_artifacts(prepared, report)
    raw = current.profiling.read_metrics(report / ("profile-" + name + ".csv"))
    if len(raw["kernels"]) != 1:
        raise RuntimeError("Expected one profiled chunk, no unlabelled extra launches")
    data = next(iter(raw["kernels"].values()))
    labels = " ".join(
        str(v) for k, v in data["metadata"].items() if "Range" in k or "NVTX" in k
    )
    if not re.search(r"qsa_score/production/chunk_0(?!\d)", labels):
        raise RuntimeError("First production chunk identity missing")
    if any(current.number(data["metrics"].get(k)) is None for k in current.REQUIRED):
        raise RuntimeError("Raw traffic metrics incomplete")
    resources = next(v for v in record["compiled_before"] if v["chunk"][1] == 0)
    for key, wanted in (
        ("launch__registers_per_thread", resources["registers_per_thread"]),
        ("launch__shared_mem_per_block_dynamic", resources["shared_bytes"]),
    ):
        value = current.number(data["metrics"].get(key))
        if value is not None and value != wanted:
            raise RuntimeError("NCU first-chunk resource mismatch: " + key)
    return record


def coordinate(o):
    from tools import benchmark_qsa_prefill_triton_stages as bench

    report = Path(o.report_dir)
    facts = dict(
        status="preparing",
        source=current.verify_source(),
        production_modified=False,
        benchmark_rerun=False,
        options=vars(o),
        cases=[],
        git_head=bench.support.command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        git_status=bench.support.command(
            ["git", "-C", str(ROOT), "status", "--porcelain"]
        ),
    )
    try:
        current.prepare(o, facts)
        ncu = facts["ncu"]
        sections = bench.support.command([ncu, "--list-sections"])
        if sections["returncode"] or "SourceCounters" not in sections["stdout"]:
            raise RuntimeError("NCU SourceCounters unavailable")
        report_module(ncu)  # fail before GPU if the report bridge is missing
        facts["source_metric_selection"] = select_source_metrics(
            ncu, bench.support.command, report / "source-metric-query.json"
        )
        facts["selection"]["metrics"] += facts["source_metric_selection"]["selected"]
        print("逐指令 shared 计数器：", facts["source_metric_selection"], flush=True)
        for name in current.CASES:
            item = next(
                v for v in current.manifest()["cases"] if v["case"]["name"] == name
            )
            entry = dict(case=name, status="preparing")
            facts["cases"].append(entry)
            prepared_path = report / ("prepared-" + name + ".json")
            if (
                o.resume_report
                and (Path(o.resume_report) / ("source-" + name + ".ncu-rep")).is_file()
            ):
                previous = Path(o.resume_report)
                for prefix in ("prepared-", "profile-"):
                    record = json.loads(
                        (previous / (prefix + name + ".json")).read_text()
                    )
                    for a in record["artifacts"]:
                        for f in a["files"].values():
                            if Path(f["path"]).name != f["path"]:
                                raise RuntimeError("Unsafe resume artifact path")
                            shutil.copy2(previous / f["path"], report / f["path"])
                    shutil.copy2(
                        previous / (prefix + name + ".json"),
                        report / (prefix + name + ".json"),
                    )
                for f in ("source-" + name + ".ncu-rep", "profile-" + name + ".csv"):
                    shutil.copy2(previous / f, report / f)
                entry["reused_from"] = str(previous)
            else:
                if not bench.idle_gate(
                    o, facts, report, "before_" + name, bench.support.gpu_guard_module()
                ):
                    raise RuntimeError("GPU admission rejected")
                cmd = [
                    sys.executable,
                    "-I",
                    str(Path(__file__).resolve()),
                    "--report-dir",
                    o.report_dir,
                    "--baseline-report",
                    o.baseline_report,
                    "--flagtree",
                    o.flagtree,
                    "--worker-case",
                    name,
                ]
                for prefix, command in (
                    ("prepared-", cmd + ["--prepare-only"]),
                    (
                        "profile-",
                        capture_command(
                            ncu,
                            report / ("profile-" + name + ".csv"),
                            report / ("source-" + name + ".ncu-rep"),
                            facts["selection"]["metrics"],
                            cmd,
                        ),
                    ),
                ):
                    if prefix == "profile-":
                        current.checked_preparation(
                            json.loads(prepared_path.read_text()), item
                        )
                        current.checked_artifacts(
                            json.loads(prepared_path.read_text()), report
                        )
                    with (report / (prefix + name + ".txt")).open("w") as log:
                        rc = current.profiling.run_process(
                            command,
                            timeout=o.timeout,
                            env=bench.child_environment(report, name),
                            log=log,
                        )
                    entry[prefix + "exit_code"] = rc
                    if rc:
                        print(
                            (report / (prefix + name + ".txt")).read_text(
                                errors="replace"
                            )[-12000:],
                            flush=True,
                        )
                        raise RuntimeError(prefix + "worker failed: " + name)
            validate_capture(report, name, facts["selection"])
            result = export_source(ncu, report, name, bench.support.command)
            entry.update(
                status="passed" if result["positive_stall_samples"] else "partial",
                report_sha256=result["report_sha256"],
            )
            bench.support.write_json(report / "shared-attribution-summary.json", facts)
        complete = all(v["status"] == "passed" for v in facts["cases"])
        facts["status"] = "passed" if complete else "partial"
        return 0 if complete else 2
    except BaseException:
        facts.update(status="failed", error=traceback.format_exc())
        print(facts["error"], flush=True)
        return 1
    finally:
        bench.support.write_json(report / "shared-attribution-summary.json", facts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir")
    p.add_argument("--baseline-report")
    p.add_argument("--flagtree", default="/opt/flagtree")
    p.add_argument("--ncu")
    p.add_argument("--prepare-ncu", action="store_true")
    p.add_argument("--resume-report")
    p.add_argument("--analyze-report")
    p.add_argument("--static-report")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--max-initial-temp-c", type=int, default=80)
    p.add_argument("--cooldown-timeout", type=int, default=120)
    p.add_argument("--cooldown-interval", type=int, default=10)
    p.add_argument("--worker-case", choices=current.CASES, help=argparse.SUPPRESS)
    p.add_argument("--prepare-only", action="store_true", help=argparse.SUPPRESS)
    o = p.parse_args()
    if o.static_report:
        report = Path(o.static_report)
        for name in current.CASES:
            result = map_record(report, name)
            print(name, json.dumps(result["counts"], ensure_ascii=False), flush=True)
        return 0
    if o.analyze_report:
        from tools import benchmark_qsa_prefill_stages as support

        report = Path(o.analyze_report)
        ncu = current.find_ncu(o.ncu)
        if not ncu:
            p.error("NCU needed for offline report export")
        results = []
        for name in current.CASES:
            if (report / ("source-" + name + ".ncu-rep")).is_file():
                validate_capture(report, name, None)
                results.append(export_source(ncu, report, name, support.command))
        if not results:
            p.error("No local .ncu-rep captures found")
        return 0 if all(v["positive_stall_samples"] for v in results) else 2
    if not o.report_dir or not o.baseline_report:
        p.error("--report-dir and --baseline-report required")
    if (
        o.timeout <= 0
        or o.cooldown_timeout < 0
        or o.cooldown_interval <= 0
        or not 1 <= o.max_initial_temp_c <= 150
    ):
        p.error("Invalid timeout/thermal limits")
    o.report_dir = str(Path(o.report_dir).resolve())
    Path(o.report_dir).mkdir(parents=True, exist_ok=True)
    o.compiler, o.candidate = "flagtree", "production"
    from tools.qsa_prefill_flagtree_ab import configure

    configure(o.flagtree)
    with patch.object(current, "EXTRA_SOURCES", current.EXTRA_SOURCES | FILES):
        return current.worker(o) if o.worker_case else coordinate(o)


if __name__ == "__main__":
    sys.exit(main())
