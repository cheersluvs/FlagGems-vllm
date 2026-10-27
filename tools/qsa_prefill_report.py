# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Readable QSA benchmark and hardware diagnostics, including saved reports."""

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

METRICS = (
    ("regs", ("launch__registers_per_thread",), False),
    ("smemKiB", ("launch__shared_mem_per_block_dynamic",), False),
    ("warp%", ("sm__warps_active.avg.pct_of_peak_sustained_active",), True),
    (
        "tensor%",
        ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",),
        True,
    ),
    (
        "DRAM%",
        (
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ),
        True,
    ),
    ("L2hit%", ("lts__t_sector_hit_rate.pct",), True),
    ("issue%", ("smsp__issue_active.avg.pct_of_peak_sustained_active",), True),
)


def number(value):
    try:
        result = float(str(value).replace(",", ""))
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def fmt(value, digits=3):
    value = number(value)
    return "N/A" if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    values = [list(map(str, headers)), *[list(map(str, row)) for row in rows]]
    widths = [max(len(row[i]) for row in values) for i in range(len(headers))]
    for index, row in enumerate(values):
        print(
            "  ".join(value.ljust(width) for value, width in zip(row, widths)),
            flush=True,
        )
        if index == 0:
            print("  ".join("-" * width for width in widths), flush=True)


def show_timing(record, *, source=None):
    case = record.get("case", {}).get("name", "unknown")
    print(f"\n=== {case} | benchmark {record.get('status', 'unknown')} ===", flush=True)
    if source:
        print("计时来源:", source, flush=True)
    print("candidate:", record.get("candidate_kind", "N/A"), flush=True)
    stages = record.get("stages", {})
    for title, names, label in (
        (
            f"原生 vLLM 对候选 [独立 {record.get('timer', {}).get('mode', 'kernel')} benchmark, us]",
            ("score", "topk", "pipeline"),
            "native_us",
        ),
        (
            "当前生产版对候选 [另一个独立计时段, us]",
            ("score_vs_production", "pipeline_vs_production"),
            "production_us",
        ),
    ):
        rows = []
        for name in names:
            row = stages.get(name)
            if not row:
                continue
            if row.get("status") != "passed":
                rows.append((name, "N/A", "N/A", "N/A", row.get("status", "unknown")))
                continue
            rows.append(
                (
                    name,
                    fmt(row.get("baseline_us", {}).get("median")),
                    fmt(row.get("candidate_us", {}).get("median")),
                    fmt(row.get("speedup"), 4),
                    row.get("measurement_validated", "N/A"),
                )
            )
        if rows:
            print(title, flush=True)
            table(("stage", label, "candidate_us", "speedup", "validated"), rows)
    controls = record.get("diagnostics", {}).get("self_controls", {})
    if controls:
        print("A/A [相同函数和 buffer；两轮描述性数据]", flush=True)
        table(
            ("side:stage", "ratio", "paired_min", "paired_max"),
            [
                (
                    name,
                    fmt(value.get("speedup"), 4),
                    fmt(value.get("paired_ratios", {}).get("min"), 4),
                    fmt(value.get("paired_ratios", {}).get("max"), 4),
                )
                for name, value in sorted(controls.items())
            ],
        )
    if record.get("error") or record.get("worker_error"):
        print("错误:", record.get("error") or record.get("worker_error"), flush=True)


def metric_values(row, units):
    values, anomalies = [], []
    for label, keys, percentage in METRICS:
        key = next((key for key in keys if key in row), keys[0])
        value = number(row.get(key))
        if value is not None and label == "smemKiB":
            if units.get(key) in ("byte/block", "byte", "bytes"):
                value /= 1024
            else:
                value = None  # Unknown unit is not assumed to be bytes.
        text = fmt(value, 0 if label == "regs" else 2)
        if value is not None and percentage and not 0 <= value <= 100:
            text += "*"
            anomalies.append(f"{key}={value}")
        values.append(text)
    return values, anomalies


def show_profile(directory, name, *, reused=False):
    from tools.qsa_prefill_profiling import read_metrics

    directory = Path(directory)
    print(
        f"\n--- {name} | 硬件计数器 [{'复用采集' if reused else '采集记录'}] ---",
        flush=True,
    )
    path = directory / ("profile-" + name + ".json")
    record = json.loads(path.read_text()) if path.is_file() else {}
    print("worker:", record.get("status", "not_captured"), flush=True)
    csv_path = directory / ("profile-" + name + ".csv")
    print("CSV:", csv_path, flush=True)
    if not csv_path.is_file():
        print("计数器 CSV: N/A", flush=True)
        return
    normalized = read_metrics(csv_path)
    chunks = record.get("chunks", [])
    traces = record.get("cache_granularity", {})
    rows, configs, flagged = [], [], []
    indexes = {}
    for identifier, kernel in sorted(
        normalized["kernels"].items(), key=lambda item: int(item[0])
    ):
        labels = " ".join(
            str(value)
            for key, value in kernel["metadata"].items()
            if "Range" in key or "NVTX" in key
        )
        match = re.search(r"qsa_score/(native|production|candidate)", labels)
        side = match[1] if match else "unlabeled"
        index = indexes.get(side, 0)
        indexes[side] = index + 1
        chunk = chunks[index] if index < len(chunks) else {}
        values, anomalies = metric_values(kernel["metrics"], normalized["units"])
        rows.append(
            (
                identifier,
                side,
                index,
                chunk.get("offset", "N/A"),
                chunk.get("rows", "N/A"),
                *values,
            )
        )
        flagged.extend(f"ID{identifier}: {message}" for message in anomalies)
        key = {"candidate": "production", "production": "production_baseline"}.get(side)
        trace = traces.get(key, [])
        if key and index < len(trace):
            constants = trace[index].get("constants", {})
            configs.append(
                (
                    side,
                    index,
                    "/".join(
                        f"{label}={constants[key]}"
                        for label, key in (
                            ("R", "TILE_R"),
                            ("N", "BLOCK_N"),
                            ("KT", "K_TILES"),
                            ("W", "num_warps"),
                            ("S", "num_stages"),
                            ("P", "QUERY_PARTS"),
                        )
                        if key in constants
                    ),
                )
            )
    table(
        ("ID", "side", "chunk", "offset", "rows", *[item[0] for item in METRICS]), rows
    )
    if configs:
        print("实际配置:", flush=True)
        table(("side", "chunk", "R/N/KT/warps/stages/query_parts"), configs)
    if flagged:
        print(
            "* 派生比例超出 [0,100]，保留原值，需复核:", "; ".join(flagged), flush=True
        )
    print(
        "warp/tensor/issue 为 active 周期口径，DRAM 为 elapsed；smem 为动态共享内存。",
        flush=True,
    )
    print("计数器来自 kernel replay；加速比取上面的独立 benchmark。", flush=True)


def show_report(directory, *, timing_only=False):
    directory = Path(directory).resolve()
    summary = json.loads((directory / "stage-summary.json").read_text())
    print("报告目录:", directory, flush=True)
    print(
        "计时状态:",
        summary.get("status"),
        "| standalone accuracy:",
        summary.get("accuracy", {}).get("status", "unknown"),
        flush=True,
    )
    print(
        "原始计时目录:",
        summary.get("manifest", {}).get("options", {}).get("report_dir", "N/A"),
        flush=True,
    )
    print(
        "各比较表是独立计时段；speedup=baseline/candidate，>1 表示候选更快。",
        flush=True,
    )
    for record in summary.get("cases", []):
        show_timing(record, source=directory / (record["case"]["name"] + ".json"))
    if not timing_only:
        path = directory / "profiling-summary.json"
        profile = json.loads(path.read_text()) if path.is_file() else {}
        print(
            "\n硬件采集状态:",
            profile.get("status", "not_run"),
            "| complete:",
            profile.get("hardware_counters_complete", False),
            flush=True,
        )
        for record in profile.get("profiles", []):
            show_profile(
                directory, record["case"], reused=bool(record.get("reused_from"))
            )
        if profile.get("error"):
            print("采集错误:", profile["error"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--timing-only", action="store_true")
    options = parser.parse_args()
    show_report(options.report_dir, timing_only=options.timing_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
