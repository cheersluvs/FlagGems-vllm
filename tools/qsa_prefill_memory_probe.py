# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Focused traffic/stall counters and exact cached scorer artifacts; no retiming."""

import argparse
import hashlib
import json
import re
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
CASES = ("perf_n1_q16384_p102400_bfloat16", "fp8_long100k_float8_e4m3fn")
REQUIRED = (
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
)
OPTIONAL = (
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_dynamic",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_selection(ncu, command, query_path=None):
    wanted = REQUIRED + OPTIONAL
    bases = sorted({m.split(".")[0] for m in wanted})
    query = command(
        [ncu, "--query-metrics-mode", "suffix", "--metrics", ",".join(bases)]
    )
    if query_path is not None:
        Path(query_path).write_text(json.dumps(query, indent=2) + "\n")
    if query.get("returncode") != 0:
        raise RuntimeError("NCU metric query failed: " + str(query))
    found = set(re.findall(r"\b\w+__\w+(?:\.\w+)*\b", query["stdout"]))
    missing = [m for m in REQUIRED if m not in found]
    if missing:
        raise RuntimeError(
            "Required traffic counters unavailable: " + ", ".join(missing)
        )
    return dict(
        metrics=[m for m in wanted if m in found],
        missing_optional=[m for m in OPTIONAL if m not in found],
        query=query,
    )


def numeric(row, key):
    from tools.qsa_prefill_report import number

    return number(row.get(key))


def ratio(a, b):
    return a / b if a is not None and b is not None and b > 0 else None


def payload(record, chunk):
    # Only the two pinned equal-geometry long fixtures. No guessed arbitrary visibility.
    case = record["case"]
    if case["name"] not in CASES or chunk["width"] != 29696:
        raise RuntimeError("Unknown memory-probe geometry")
    if (
        case.get("heads"),
        case.get("dim"),
        case.get("ratio"),
        case.get("page_size"),
    ) != (4, 128, 4, 196):
        raise RuntimeError("Memory-probe input metadata changed")
    if case["name"] == CASES[0]:
        lengths = (
            case.get("requests"),
            case.get("query_length"),
            case.get("prefix_tokens"),
        )
    else:
        query, prefix = case.get("query_lengths"), case.get("prefix_tokens")
        lengths = (
            (1, query[0], prefix[0])
            if (
                isinstance(query, list)
                and isinstance(prefix, list)
                and len(query) == len(prefix) == 1
                and case.get("query_base") == 0
                and case.get("batch_max_seq_len") == 118784
            )
            else None
        )
    first, count = chunk["offset"], chunk["rows"]
    if (
        lengths != (1, 16384, 102400)
        or first < 0
        or count <= 0
        or first + count > 16384
    ):
        raise RuntimeError("Memory-probe lengths changed")
    return (
        sum(min(29696, (102400 + row + 1) // 4) for row in range(first, first + count))
        * 4
    )


def traffic_report(report, name):
    from tools.qsa_prefill_profiling import read_metrics
    from tools.qsa_prefill_report import fmt, table

    report = Path(report)
    record = json.loads((report / ("profile-" + name + ".json")).read_text())
    data = read_metrics(report / ("profile-" + name + ".csv"))
    indexes, rows = {}, []
    for identifier, kernel in sorted(
        data["kernels"].items(), key=lambda item: int(item[0])
    ):
        labels = " ".join(
            str(v) for k, v in kernel["metadata"].items() if "Range" in k or "NVTX" in k
        )
        match = re.search(r"qsa_score/(native|production|candidate)", labels)
        if not match:
            raise RuntimeError("Missing per-side NVTX identity in memory CSV")
        side = match[1]
        index = indexes.get(side, 0)
        indexes[side] = index + 1
        chunk = record["chunks"][index]
        values = kernel["metrics"]
        missing = [key for key in REQUIRED if numeric(values, key) is None]
        if missing:
            raise RuntimeError(
                "Captured traffic counters missing/N/A: " + ", ".join(missing)
            )
        read_req, read_sectors, store_req, store_sectors, dram_read, dram_write = [
            numeric(values, k) for k in REQUIRED
        ]
        useful = payload(record, chunk)
        rows.append(
            dict(
                id=identifier,
                side=side,
                chunk=index,
                offset=chunk["offset"],
                rows=chunk["rows"],
                load_requests=read_req,
                load_sectors=read_sectors,
                store_requests=store_req,
                store_sectors=store_sectors,
                load_sectors_per_request=ratio(read_sectors, read_req),
                store_sectors_per_request=ratio(store_sectors, store_req),
                useful_logits_bytes=useful,
                store_sector_bytes_per_payload=ratio(store_sectors * 32, useful),
                dram_read_mib=dram_read / 1024**2,
                dram_write_mib=dram_write / 1024**2,
                optional={k: numeric(values, k) for k in OPTIONAL},
            )
        )
    if indexes != {
        s: len(record["chunks"]) for s in ("native", "production", "candidate")
    }:
        raise RuntimeError("Incomplete traffic capture by side/chunk")
    print("\n访存诊断:", name, flush=True)
    table(
        (
            "ID",
            "side",
            "chunk",
            "LDsec/req",
            "STsec/req",
            "STsector/payload",
            "DRAMreadMiB",
            "DRAMwriteMiB",
        ),
        [
            (
                r["id"],
                r["side"],
                r["chunk"],
                fmt(r["load_sectors_per_request"]),
                fmt(r["store_sectors_per_request"]),
                fmt(r["store_sector_bytes_per_payload"]),
                fmt(r["dram_read_mib"]),
                fmt(r["dram_write_mib"]),
            )
            for r in rows
        ],
    )
    keys = OPTIONAL[-6:]
    table(
        (
            "ID",
            "side",
            "sharedLDconf",
            "sharedSTconf",
            "longSB%",
            "shortSB%",
            "MIO%",
            "LG%",
        ),
        [(r["id"], r["side"], *[fmt(r["optional"][k]) for k in keys]) for r in rows],
    )
    print(
        "请求比值需结合指令宽度、活动lane和掩码；不是DRAM写放大或加速比。", flush=True
    )
    return rows


def export_artifacts(report, name, nvdisasm, command):
    report = Path(report)
    record = json.loads((report / ("profile-" + name + ".json")).read_text())
    cache = Path(record["runtime"]["environment"]["TRITON_CACHE_DIR"]).resolve()
    roles = {}
    for side, key in (
        ("production", "production_baseline"),
        ("candidate", "production"),
    ):
        for trace in record["cache_granularity"][key]:
            k = trace["compiled"]
            roles.setdefault((k["cubin_sha256"], k["ptx_sha256"]), set()).add(side)
    for k in record["kernel_resources"]["native"]["kernels"]:
        roles.setdefault((k["cubin_sha256"], k["ptx_sha256"]), set()).add("native")
    candidates = {}
    if cache.is_dir():
        for path in cache.rglob("*.cubin"):
            if path.resolve().is_relative_to(cache):
                candidates.setdefault(sha(path), []).append(path)
    if not roles:
        raise RuntimeError("No recorded scorer binaries")
    output = []
    for (cubin_hash, ptx_hash), sides in roles.items():
        entry = dict(
            sides=sorted(sides),
            cubin_sha256=cubin_hash,
            ptx_sha256=ptx_hash,
            status="missing",
        )
        output.append(entry)
        for path in sorted(candidates.get(cubin_hash, [])):
            ptx = path.with_suffix(".ptx")
            if (
                not ptx.is_file()
                or not ptx.resolve().is_relative_to(cache)
                or sha(ptx) != ptx_hash
            ):
                continue
            entry.update(status="exported", cache_cubin=str(path), files={})
            prefix = f"code-{name}-{cubin_hash[:12]}-{ptx_hash[:12]}"
            for suffix in ("ptx", "ttgir", "ttir", "llir"):
                src = path.with_suffix("." + suffix)
                if src.is_file() and src.resolve().is_relative_to(cache):
                    dst = report / (prefix + "." + suffix + ".txt")
                    shutil.copyfile(src, dst)
                    entry["files"][suffix] = dict(
                        path=dst.name, sha256=sha(dst), origin=str(src)
                    )
            if nvdisasm:
                response = command([nvdisasm, str(path)])
                dst = report / (prefix + ".sass.txt")
                dst.write_text(response.get("stdout", "") + response.get("stderr", ""))
                entry["disassembly_exit_code"] = response.get("returncode")
                entry["files"]["sass"] = dict(path=dst.name, sha256=sha(dst))
            break
    return dict(
        cache=str(cache),
        entries=output,
        all_ptx_exported=all(e["status"] == "exported" for e in output),
        all_ttgir_exported=all("ttgir" in e.get("files", {}) for e in output),
    )


def prepare_analysis(options, output):
    source = Path(options.analyze_report).resolve()
    if source == output:
        raise ValueError("Analysis output must differ from captured report")
    summaries = [
        json.loads(path.read_text())
        for path in (source / "memory-summary.json", source / "profiling-summary.json")
        if path.is_file()
    ]
    # memory-summary is saved after postprocessing failure; older capture drivers
    # wrote profiling-summary before the final worker finished. Require completed
    # per-worker evidence, never the postprocessor's overall status alone.
    summary = next(
        (
            s
            for s in summaries
            if s.get("hardware_counters_complete")
            and s.get("profiles")
            and all(p.get("status") == "passed" for p in s["profiles"])
        ),
        None,
    )
    if summary is None:
        raise RuntimeError("Analysis requires successful completed hardware captures")
    profiles = {p["case"]: p for p in summary["profiles"]}
    for name in CASES:
        profile = profiles.get(name, {})
        record = json.loads((source / ("profile-" + name + ".json")).read_text())
        if (
            profile.get("status") != "passed"
            or record.get("status") != "passed"
            or profile.get("expected_profiled_launches") != 3 * len(record["chunks"])
            or record["case"]["name"] != name
        ):
            raise RuntimeError("Incomplete or mismatched capture: " + name)
        for chunk in record["chunks"]:
            payload(record, chunk)
        if not (source / ("profile-" + name + ".csv")).is_file():
            raise RuntimeError("Missing saved counter CSV: " + name)
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".json", ".txt", ".tsv", ".csv"):
            prior = path.name in (
                "runner.txt",
                "terminal-summary.txt",
                "run-status.txt",
                "memory-summary.json",
                "memory-sources.json",
            )
            shutil.copy2(path, output / (("prior-" if prior else "") + path.name))
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--resume-report")
    mode.add_argument(
        "--analyze-report", help="Postprocess completed capture without GPU launches"
    )
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--flagtree", default="/opt/flagtree")
    parser.add_argument("--ncu")
    parser.add_argument("--timeout", type=int, default=1800)
    options = parser.parse_args()
    if options.timeout <= 0:
        parser.error("Positive timeout required")
    if options.resume_report and not options.ncu:
        parser.error("--ncu is required for counter capture")
    report = Path(options.report_dir).resolve()
    report.mkdir(parents=True, exist_ok=True)
    from tools import benchmark_qsa_prefill_stages as support
    from tools import qsa_prefill_profile_resume as resume
    from tools import qsa_prefill_profiling as profiling

    facts = dict(
        status="preparing", acceptance_eligible=False, independent_benchmark_rerun=False
    )
    try:
        if options.analyze_report:
            original = prepare_analysis(options, report)
            worker = None
            facts.update(
                analysis_only=True, original_report=str(original), cases=list(CASES)
            )
        else:
            original, worker, facts = resume.prepare(options)
            if not set(CASES) <= set(options.cases):
                raise RuntimeError("Missing original long-case benchmarks")
            options.cases = list(CASES)
            facts["cases"] = options.cases
        facts["question"] = (
            "global loads vs logits stores vs shared movement/stalls; no optimized kernel yet"
        )
        source_names = (
            "tools/qsa_prefill_memory_probe.py",
            "tools/test_qsa_prefill_memory_probe.py",
            "tools/qsa_prefill_profiling.py",
            "tools/qsa_prefill_profile_resume.py",
            "tools/run_qsa_prefill_memory_h800.sh",
            "docs/qsa_prefill_memory_probe.md",
        )
        support.write_json(
            report / "memory-sources.json",
            {
                n: dict(text=(ROOT / n).read_text(), sha256=sha(ROOT / n))
                for n in source_names
            },
        )
        if not options.analyze_report:
            selection = metric_selection(
                options.ncu, support.command, report / "memory-metric-query.json"
            )
            support.write_json(report / "memory-metric-selection.json", selection)
            print("只补采两个长序列场景；不重跑 benchmark/A/A/pytest。", flush=True)
            print(
                "Unavailable optional metrics:",
                selection["missing_optional"],
                flush=True,
            )
            from tools.qsa_prefill_report import show_timing

            print("以下为原始已保存计时，不是本次新 benchmark：", flush=True)
            for name in CASES:
                path = original / (name + ".json")
                show_timing(json.loads(path.read_text()), source=path)
            rc = profiling.collect_counters(
                options,
                facts,
                worker,
                cache_report=original,
                metric_names=selection["metrics"],
            )
            if rc:
                raise RuntimeError("Memory capture incomplete: " + str(rc))
        else:
            print(
                "只处理已完成的两组采集；不启动 CUDA/NCU，不运行 benchmark。",
                flush=True,
            )
        tools = json.loads((original / "flagtree134-tools.json").read_text())[
            "toolchain"
        ]
        dis = tools.get("nvdisasm", {})
        nvdisasm = None
        if dis and Path(dis["path"]).is_file() and sha(dis["path"]) == dis["sha256"]:
            nvdisasm = dis["path"]
        facts["disassembler"] = dict(path=nvdisasm, verified=bool(nvdisasm))
        facts["memory_cases"] = []
        for name in CASES:
            traffic = traffic_report(report, name)
            artifacts = export_artifacts(report, name, nvdisasm, support.command)
            facts["memory_cases"].append(
                dict(case=name, traffic=traffic, artifacts=artifacts)
            )
            print(
                "缓存编译产物:",
                name,
                "PTX complete:",
                artifacts["all_ptx_exported"],
                "TTGIR complete:",
                artifacts["all_ttgir_exported"],
                flush=True,
            )
        complete = all(
            c["artifacts"]["all_ptx_exported"] and c["artifacts"]["all_ttgir_exported"]
            for c in facts["memory_cases"]
        )
        facts.update(
            status="passed" if complete else "partial_artifacts",
            memory_counters_complete=True,
        )
        return 0 if complete else 2
    except BaseException:
        facts.update(status="failed", error=traceback.format_exc())
        print(facts["error"], flush=True)
        return 1
    finally:
        support.write_json(report / "memory-summary.json", facts)


if __name__ == "__main__":
    sys.exit(main())
