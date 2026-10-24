# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Current-production traffic, shared-memory and stall capture, without retiming."""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_bf16_stage_depth_probe as stage  # noqa: E402
from tools import qsa_prefill_memory_probe as memory  # noqa: E402
from tools import qsa_prefill_profiling as profiling  # noqa: E402
from tools.qsa_prefill_report import fmt, number, table  # noqa: E402

base = stage.base
CASES = (
    "perf_n1_q8192_p102400_bfloat16",
    "perf_n4_q4096_p102400_bfloat16",
    "fp8_long100k_float8_e4m3fn",
)
REQUIRED = memory.REQUIRED
OPTIONAL = memory.OPTIONAL + (
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
)
EXTRA_SOURCES = stage.EXTRA_SOURCES | {
    "tools/qsa_prefill_production_profile.py",
    "tools/run_qsa_prefill_production_profile_h800.sh",
    "tools/test_qsa_prefill_production_profile.py",
    "tools/qsa_prefill_memory_probe.py",
    "tools/qsa_prefill_profiling.py",
    "tools/qsa_prefill_report.py",
    "tools/qsa_ncu_portable.py",
    "docs/qsa_prefill_production_profile.md",
}


def manifest():
    data = stage.manifest()
    data["cases"] = [v for v in data["cases"] if v["case"]["name"] in CASES]
    if tuple(v["case"]["name"] for v in data["cases"]) != CASES:
        raise RuntimeError("Production profile fixture set changed")
    return data


def verify_source():
    return dict(
        stage.verify_source(),
        profile_driver_sha256=base.sha(Path(__file__)),
        profile_sources={n: base.sha(ROOT / n) for n in sorted(EXTRA_SOURCES)},
    )


def selected_metrics(ncu, command, path):
    wanted = REQUIRED + OPTIONAL
    query = command(
        [
            ncu,
            "--query-metrics-mode",
            "suffix",
            "--metrics",
            ",".join(sorted({v.split(".")[0] for v in wanted})),
        ]
    )
    path.write_text(json.dumps(query, indent=2) + "\n")
    if query["returncode"]:
        raise RuntimeError("NCU metric query failed; see " + str(path))
    found = set(re.findall(r"\b\w+__\w+(?:\.\w+)*\b", query["stdout"]))
    missing = [v for v in REQUIRED if v not in found]
    if missing:
        raise RuntimeError("Required traffic metrics unavailable: " + repr(missing))
    return dict(
        metrics=[v for v in wanted if v in found],
        missing_optional=[v for v in OPTIONAL if v not in found],
    )


def find_ncu(explicit=None):
    found = profiling.find_ncu(explicit)
    if found:
        return found
    # Bounded depth and prefix: retain existing isolated archive extractions.
    root = Path(tempfile.gettempdir())
    for pattern in (
        "qsa-ncu-portable.*/ncu",
        "qsa-ncu-portable.*/*/ncu",
        "qsa-ncu-portable.*/*/*/ncu",
    ):
        for p in sorted(root.glob(pattern), reverse=True):
            if p.is_file() and os.access(p, os.X_OK):
                return str(p.resolve())
    return None


def freeze(module, traces, *, historical=True):
    records = base.freeze_records((module, module))["baseline"]
    expected = {(t["rows"], t["offset"], t["width"]): t for t in traces}
    if {tuple(v["chunk"]) for v in records} != set(expected):
        raise RuntimeError("Missing production chunk")
    for record in records:
        t = expected[tuple(record["chunk"])]
        if record["constants"] != t["constants"] or record["route"] != t["route"]:
            raise RuntimeError("Production configuration changed")
        if historical and any(
            record[k] != t[k] for k in ("ptx_sha256", "cubin_sha256")
        ):
            raise RuntimeError("Captured production binary changed")
    return records


def artifact_records(module, report, name):
    records = []
    for entry, _ in base.ENTRIES:
        for key, (kernel, _) in getattr(module, entry).compiled.items():
            files = {}
            for kind in ("ptx", "ttgir"):
                path = report / f"{name}-{key[1]}-{kind}.txt"
                path.write_text(kernel.asm[kind])
                files[kind] = dict(path=path.name, sha256=base.sha(path))
            records.append(dict(chunk=list(key), files=files))
    return records


def worker(options):
    from tools import benchmark_qsa_prefill_triton_stages as bench

    support, scoring = bench.support, bench.scoring
    report = Path(options.report_dir)
    preparation = getattr(options, "prepare_only", False)
    prefix = "prepared-" if preparation else "profile-"
    path = report / (prefix + options.worker_case + ".json")
    result = dict(
        status="preparing",
        source=verify_source(),
        kind=(
            "untimed_production_preparation"
            if preparation
            else "hardware_only_not_benchmark"
        ),
        production_modified=False,
        measurement_performed=False,
    )
    try:
        item = next(
            v for v in manifest()["cases"] if v["case"]["name"] == options.worker_case
        )
        case, traces = item["case"], item["traces"]
        result.update(case=case, fixed_traces=traces)
        torch, _, production, _, helpers, device, info = bench.runtime(
            report / "qsa_indexer_vllm.py", path, options
        )
        result["runtime"] = info
        if base.compiler_identity(info) != manifest()["compiler_identity"]:
            raise RuntimeError("Production compiler/runtime changed")
        os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = str(case["budget_mb"])
        args = bench.workloads.make_inputs(case, helpers, torch)
        result["inputs"] = support.fingerprint(args, torch)
        if result["inputs"] != item["inputs"]:
            raise RuntimeError("Captured production input changed")
        # Baseline member uses the original production JIT and original configs.
        module = stage.make_pair(production, traces)[0]
        chunks = bench.ab.observe(module, "persistent_topk", args)
        bench.verify_geometry(case, args, chunks)
        if bench.ab.signatures(chunks) != [
            (v["offset"], v["rows"], v["width"]) for v in traces
        ]:
            raise RuntimeError("Public production dispatch changed")
        visible = args[4].tolist()
        result["chunks"] = [
            dict(
                offset=c["offset"],
                rows=c["rows"],
                width=c["width"],
                useful_logits_bytes=4
                * sum(
                    min(max(v, 0), c["width"])
                    for v in visible[c["offset"] : c["offset"] + c["rows"]]
                ),
            )
            for c in chunks
        ]

        def validate():
            return scoring.validate_scores(
                module._prefill_logits, chunks, args, torch, support.sample_rows, device
            )

        def scores():
            for c in chunks:
                module._prefill_logits(*c["score_args"], **c.get("score_kwargs", {}))

        result["validation_before"] = validate()
        scores()
        device.synchronize()
        # Persist actual artifacts before enforcing binary identity. A failed
        # match must still leave enough evidence to diagnose the difference.
        result["compiled_before"] = freeze(module, traces, historical=False)
        artifact_name = "prepared-" + case["name"] if preparation else case["name"]
        result["artifacts"] = artifact_records(module, report, artifact_name)
        result["historical_binary_match"] = historical_match(
            result["compiled_before"], traces
        )
        support.write_json(path, result)
        if not preparation:
            prepared = json.loads(
                (report / ("prepared-" + case["name"] + ".json")).read_text()
            )
            checked_preparation(prepared, item)
            if result["compiled_before"] != prepared["compiled_before"]:
                raise RuntimeError(
                    "NCU/current preparation binary mismatch; actual artifacts saved"
                )
        logits = tuple(c["score_args"][5] for c in chunks)
        result["logits_before"] = support.fingerprint(logits, torch)
        support.guarded_telemetry(
            info["uuid"],
            "before_current_score_capture",
            result,
            path,
            support.gpu_guard_module(),
        )
        cudart = torch.cuda.cudart()

        def launch(c):
            module._prefill_logits(*c["score_args"], **c.get("score_kwargs", {}))

        result["captures"] = (
            []
            if preparation
            else capture_sequence(cudart, torch.cuda.nvtx, device, chunks, launch)
        )
        support.guarded_telemetry(
            info["uuid"],
            "after_current_score_capture",
            result,
            path,
            support.gpu_guard_module(),
        )
        result["validation_after"] = validate()
        result["compiled_after"] = freeze(module, traces, historical=False)
        result["logits_after"] = support.fingerprint(logits, torch)
        if (
            result["compiled_before"] != result["compiled_after"]
            or result["logits_before"] != result["logits_after"]
            or support.fingerprint(args, torch) != item["inputs"]
            or result["source"] != verify_source()
        ):
            raise RuntimeError(
                "Input/logits/source/compiled kernel changed during capture"
            )
        result["status"] = "passed"
        return 0
    except BaseException:
        result.update(status="failed", error=traceback.format_exc())
        print(result["error"], flush=True)
        return 1
    finally:
        support.write_json(path, result)


def capture_sequence(cudart, nvtx, device, chunks, launch):
    """Only warmed, stored score invocations enter profiler collection."""
    rc = cudart.cudaProfilerStart()
    if int(rc):
        raise RuntimeError("cudaProfilerStart failed: " + str(rc))
    captures = []
    try:
        for index, c in enumerate(chunks):
            label = f"qsa_score/production/chunk_{index}"
            nvtx.range_push(label)
            try:
                launch(c)
                device.synchronize()
            finally:
                nvtx.range_pop()
            captures.append(
                dict(index=index, label=label, offset=c["offset"], rows=c["rows"])
            )
    finally:
        rc = cudart.cudaProfilerStop()
        if int(rc):
            raise RuntimeError("cudaProfilerStop failed: " + str(rc))
    return captures


def checked_common(record, item):
    if (
        record.get("status") != "passed"
        or record.get("source") != verify_source()
        or record.get("case") != item["case"]
        or record.get("inputs") != item["inputs"]
    ):
        raise RuntimeError("Invalid production capture provenance")
    if base.compiler_identity(record["runtime"]) != manifest()["compiler_identity"]:
        raise RuntimeError("Capture compiler changed")
    if (
        record.get("validation_before", 0) <= 0
        or record.get("validation_after", 0) <= 0
    ):
        raise RuntimeError("Missing before/after numeric checks")
    if (
        record["compiled_before"] != record["compiled_after"]
        or record["logits_before"] != record["logits_after"]
    ):
        raise RuntimeError("Capture binary/logits changed")
    binding = record["runtime"]["operator_binding"]
    if (
        binding["canonical_sha256"] != stage.PRODUCTION_SHA
        or not binding["candidate_is_canonical_module"]
        or not binding["exported_entry_is_candidate"]
    ):
        raise RuntimeError("Capture canonical production binding changed")
    traces = item["traces"]
    expected = {(t["rows"], t["offset"], t["width"]): t for t in traces}
    if {tuple(v["chunk"]) for v in record["compiled_before"]} != set(expected):
        raise RuntimeError("Capture chunk set changed")
    for v in record["compiled_before"]:
        t = expected[tuple(v["chunk"])]
        if v["route"] != t["route"] or any(v[k] != t[k] for k in ("constants",)):
            raise RuntimeError("Capture production configuration/binary changed")
    if [(v["offset"], v["rows"], v["width"]) for v in record["chunks"]] != [
        (t["offset"], t["rows"], t["width"]) for t in traces
    ]:
        raise RuntimeError("Capture geometry changed")
    if len(record.get("telemetry", [])) != 2 or any(
        v["gpu_guard"]["status"] != "passed" or v["gpu_guard"]["reasons"]
        for v in record["telemetry"]
    ):
        raise RuntimeError("Capture GPU guard failed")


def historical_match(records, traces):
    expected = {(t["rows"], t["offset"], t["width"]): t for t in traces}
    return all(
        all(
            r[k] == expected[tuple(r["chunk"])][k]
            for k in ("ptx_sha256", "cubin_sha256")
        )
        for r in records
    )


def checked_preparation(record, item):
    checked_common(record, item)
    if (
        record.get("kind") != "untimed_production_preparation"
        or record.get("captures") != []
    ):
        raise RuntimeError("Invalid unprofiled preparation")
    if record.get("historical_binary_match") != historical_match(
        record["compiled_before"], item["traces"]
    ):
        raise RuntimeError("Historical binary comparison changed")


def checked_capture(record, item, prepared):
    checked_common(record, item)
    checked_preparation(prepared, item)
    if record.get("kind") != "hardware_only_not_benchmark":
        raise RuntimeError("Invalid hardware capture")
    if (
        record["compiled_before"] != prepared["compiled_before"]
        or record["chunks"] != prepared["chunks"]
        or record["inputs"] != prepared["inputs"]
    ):
        raise RuntimeError("Capture/current preparation configuration/binary mismatch")
    traces = item["traces"]
    wanted = [
        dict(
            index=i,
            label=f"qsa_score/production/chunk_{i}",
            offset=t["offset"],
            rows=t["rows"],
        )
        for i, t in enumerate(traces)
    ]
    if record["captures"] != wanted:
        raise RuntimeError("Incomplete capture labels")


def checked_artifacts(record, report):
    expected_artifacts = {tuple(v["chunk"]): v for v in record["compiled_before"]}
    if {tuple(v["chunk"]) for v in record["artifacts"]} != set(expected_artifacts):
        raise RuntimeError("Incomplete compiled artifacts")
    for a in record["artifacts"]:
        if (
            a["files"]["ptx"]["sha256"]
            != expected_artifacts[tuple(a["chunk"])]["ptx_sha256"]
        ):
            raise RuntimeError("PTX artifact differs from compiled scorer")
        for f in a["files"].values():
            if base.sha(report / f["path"]) != f["sha256"]:
                raise RuntimeError("Captured artifact hash mismatch")


def counter_rows(report, name, selection):
    record = json.loads((report / ("profile-" + name + ".json")).read_text())
    item = next(v for v in manifest()["cases"] if v["case"]["name"] == name)
    prepared = json.loads((report / ("prepared-" + name + ".json")).read_text())
    checked_capture(record, item, prepared)
    checked_artifacts(record, report)
    checked_artifacts(prepared, report)
    raw = profiling.read_metrics(report / ("profile-" + name + ".csv"))
    rows, seen = [], set()
    for identifier, data in sorted(raw["kernels"].items(), key=lambda v: int(v[0])):
        labels = " ".join(
            str(v) for k, v in data["metadata"].items() if "Range" in k or "NVTX" in k
        )
        match = re.search(r"qsa_score/production/chunk_(\d+)(?!\d)", labels)
        if not match:
            raise RuntimeError("Missing production chunk NVTX label")
        index = int(match[1])
        if index in seen or index >= len(record["chunks"]):
            raise RuntimeError("Duplicate/extra captured score chunk")
        seen.add(index)
        metrics = {k: number(v) for k, v in data["metrics"].items()}
        if any(metrics.get(k) is None or metrics[k] < 0 for k in REQUIRED):
            raise RuntimeError("Missing/invalid required traffic values")
        chunk = record["chunks"][index]
        ldreq, ldsec, streq, stsec, dramrd, dramwr = [metrics[k] for k in REQUIRED]
        resources = next(
            v
            for v in record["compiled_before"]
            if v["chunk"] == [chunk["rows"], chunk["offset"], chunk["width"]]
        )
        for key, value in (
            ("launch__registers_per_thread", resources["registers_per_thread"]),
            ("launch__shared_mem_per_block_dynamic", resources["shared_bytes"]),
        ):
            if metrics.get(key) is not None and metrics[key] != value:
                raise RuntimeError("NCU/compiled resource mismatch: " + key)
        missing = [k for k in OPTIONAL if metrics.get(k) is None]
        anomalies = [
            k
            for k, v in metrics.items()
            if v is not None
            and (k.endswith(".pct") or ".pct_" in k)
            and not 0 <= v <= 100
        ]
        row = dict(
            id=identifier,
            index=index,
            **chunk,
            resources=resources,
            metrics=metrics,
            missing_optional=missing,
            anomalous_percentages=anomalies,
            load_sectors_per_request=memory.ratio(ldsec, ldreq),
            store_sectors_per_request=memory.ratio(stsec, streq),
            store_sector_bytes_per_payload=memory.ratio(
                stsec * 32, chunk["useful_logits_bytes"]
            ),
            dram_read_mib=dramrd / 1024**2,
            dram_write_mib=dramwr / 1024**2,
        )
        rows.append(row)
    if seen != set(range(len(record["chunks"]))):
        raise RuntimeError("Incomplete production score capture")
    return dict(
        case=name,
        status="passed",
        measurement_performed=False,
        historical_binary_match=prepared["historical_binary_match"],
        missing_queried_optional=selection["missing_optional"],
        diagnostics_complete=not any(v["missing_optional"] for v in rows),
        rows=rows,
    )


def display_counters(result):
    print("\n当前生产版硬件计数器（不是 benchmark）：", result["case"], flush=True)
    print(
        "与历史二进制一致：", result.get("historical_binary_match", "N/A"), flush=True
    )
    rows = result["rows"]
    table(
        (
            "chunk",
            "offset",
            "rows",
            "regs",
            "sharedKiB",
            "spill",
            "LDsec/req",
            "STsec/req",
            "STsector/payload",
            "DRAMreadMiB",
            "DRAMwriteMiB",
        ),
        [
            (
                v["index"],
                v["offset"],
                v["rows"],
                v["resources"]["registers_per_thread"],
                fmt(v["resources"]["shared_bytes"] / 1024),
                v["resources"]["spill_count"],
                *(
                    fmt(v[k])
                    for k in (
                        "load_sectors_per_request",
                        "store_sectors_per_request",
                        "store_sector_bytes_per_payload",
                        "dram_read_mib",
                        "dram_write_mib",
                    )
                ),
            )
            for v in rows
        ],
    )
    fields = (
        ("warp%", OPTIONAL[2]),
        ("tensor%", OPTIONAL[3]),
        ("issue%", "smsp__issue_active.avg.pct_of_peak_sustained_active"),
        ("DRAM%", OPTIONAL[4]),
        ("L2hit%", OPTIONAL[5]),
        ("eligible", "smsp__warps_eligible.avg.per_cycle_active"),
        ("longSB%", OPTIONAL[11]),
        ("shortSB%", OPTIONAL[12]),
        ("MIO%", OPTIONAL[13]),
        ("LG%", OPTIONAL[14]),
        ("barrier%", OPTIONAL[-2]),
        ("sharedLDconflicts", OPTIONAL[9]),
        ("sharedSTconflicts", OPTIONAL[10]),
    )
    # Explicit metric names avoid any interpretation of missing values as zero.
    table(
        ("chunk", *(label for label, _ in fields)),
        [(v["index"], *(fmt(v["metrics"].get(k)) for _, k in fields)) for v in rows],
    )
    for v in rows:
        if v["missing_optional"]:
            print("缺失可选计数器:", v["index"], v["missing_optional"], flush=True)
        if v["anomalous_percentages"]:
            print(
                "异常百分比原样保留:",
                v["index"],
                v["anomalous_percentages"],
                flush=True,
            )


def historical_timing(captured, name, report=None):
    if captured.get("status") != "passed" or captured["case"]["name"] != name:
        raise RuntimeError("Missing historical timing")
    rows = []
    for key in ("score", "topk", "pipeline"):
        v = captured["stages"][key]
        if not v.get("measurement_validated") or v["status"] != "passed":
            raise RuntimeError("Unvalidated historical stage")
        rows.append(
            (
                key,
                fmt(v["baseline_us"]["median"]),
                fmt(v["candidate_us"]["median"]),
                fmt(v["speedup"]),
            )
        )
    print("\n历史独立 kernel 计时，来源78c3ca4f（本次不重测）：", name, flush=True)
    table(("stage", "native_us", "production_us", "speedup"), rows)
    if report is not None:
        (report / ("historical-" + name + ".json")).write_text(
            json.dumps(captured, indent=2) + "\n"
        )


def prepare(options, facts):
    from tools import benchmark_qsa_prefill_triton_stages as bench
    from tools import qsa_prefill_flagtree_ab as profile

    report, previous = Path(options.report_dir), Path(options.baseline_report)
    old = json.loads((previous / "stage-summary.json").read_text())
    if (
        old["status"] != "passed"
        or old["manifest"]["git_head"]["stdout"].strip()
        != manifest()["baseline_code_commit"]
    ):
        raise RuntimeError("Requires successful production report78c3ca4f")
    tools = json.loads((previous / "flagtree134-tools.json").read_text())
    assembler = Path(tools["toolchain"]["ptxas"]["path"])
    if not assembler.is_file() or base.sha(assembler) != profile.PTXAS_SHA256:
        raise RuntimeError("Preserved verified ptxas unavailable")
    os.environ["TRITON_PTXAS_PATH"] = str(assembler)
    sources = json.loads((previous / "sources.json").read_text())
    reference = sources["pinned_vllm/qsa_indexer.py"]["text"]
    if hashlib.sha256(reference.encode()).hexdigest() != bench.ab.REFERENCE_SHA256:
        raise RuntimeError("Pinned native reference changed")
    (report / "qsa_indexer_vllm.py").write_text(reference)
    bench.support.write_json(report / "flagtree134-tools.json", tools)
    names = set(bench.SOURCES) | EXTRA_SOURCES
    bench.support.write_json(
        report / "sources.json",
        {
            n: dict(text=(ROOT / n).read_text(), sha256=base.sha(ROOT / n))
            for n in sorted(names)
        },
    )
    for item in manifest()["cases"]:
        name = item["case"]["name"]
        captured = json.loads((previous / (name + ".json")).read_text())
        if (
            captured.get("candidate_kind") != "production"
            or captured["case"] != item["case"]
            or captured["inputs"] != item["inputs"]
            or base.compiler_identity(captured["runtime"])
            != manifest()["compiler_identity"]
            or captured["runtime"]["operator_binding"]["canonical_sha256"]
            != stage.PRODUCTION_SHA
        ):
            raise RuntimeError("Captured production baseline changed: " + name)
        historical_timing(captured, name, report)
    facts["ncu"] = find_ncu(options.ncu)
    if not facts["ncu"] and options.prepare_ncu:
        from tools.qsa_ncu_portable import prepare as prepare_ncu

        facts["ncu"] = prepare_ncu(report)
    if not facts["ncu"]:
        raise RuntimeError("NCU unavailable; pass --ncu or --prepare-ncu")
    facts["ncu_version"] = bench.support.command([facts["ncu"], "--version"])
    if facts["ncu_version"]["returncode"]:
        raise RuntimeError("NCU --version failed")
    facts["selection"] = selected_metrics(
        facts["ncu"], bench.support.command, report / "metric-query.json"
    )
    if not profile.select_gpu(facts, options):
        raise RuntimeError("No exclusive idle H800")


def reuse_case(previous, report, name, selection):
    record = json.loads((previous / ("profile-" + name + ".json")).read_text())
    item = next(v for v in manifest()["cases"] if v["case"]["name"] == name)
    prepared = json.loads((previous / ("prepared-" + name + ".json")).read_text())
    checked_capture(record, item, prepared)
    candidates = ["profile-" + name + suffix for suffix in (".json", ".csv", ".txt")]
    candidates += ["prepared-" + name + ".json", "prepared-" + name + ".txt"]
    candidates += [
        f["path"]
        for r in (record, prepared)
        for a in r["artifacts"]
        for f in a["files"].values()
    ]
    for filename in candidates:
        if Path(filename).name != filename:
            raise RuntimeError("Unsafe resume path")
        shutil.copy2(previous / filename, report / filename)
    return counter_rows(report, name, selection)


def coordinate(options):
    from tools import benchmark_qsa_prefill_triton_stages as bench

    report = Path(options.report_dir)
    facts = dict(
        status="preparing",
        hardware_counters_complete=False,
        benchmark_rerun=False,
        production_modified=False,
        acceptance_eligible=False,
        source=verify_source(),
        options=vars(options),
        cases=[],
        git_head=bench.support.command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        git_status=bench.support.command(
            ["git", "-C", str(ROOT), "status", "--porcelain"]
        ),
    )
    try:
        prepare(options, facts)
        for name in CASES:
            if (
                options.resume_report
                and (
                    Path(options.resume_report) / ("counters-" + name + ".json")
                ).is_file()
            ):
                result = reuse_case(
                    Path(options.resume_report), report, name, facts["selection"]
                )
                entry = dict(
                    case=name,
                    status="passed",
                    reused_from=options.resume_report,
                    csv_sha256=base.sha(report / ("profile-" + name + ".csv")),
                )
            else:
                if not bench.idle_gate(
                    options,
                    facts,
                    report,
                    "before_" + name,
                    bench.support.gpu_guard_module(),
                ):
                    raise RuntimeError("GPU admission rejected")
                csv_path = report / ("profile-" + name + ".csv")
                prepare_cmd = [
                    sys.executable,
                    "-I",
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                    "--worker-case",
                    name,
                    "--prepare-only",
                ]
                with (report / ("prepared-" + name + ".txt")).open("w") as log:
                    prepare_rc = profiling.run_process(
                        prepare_cmd,
                        timeout=options.timeout,
                        env=bench.child_environment(report, name),
                        log=log,
                    )
                prepared_path = report / ("prepared-" + name + ".json")
                if prepare_rc:
                    print(
                        (report / ("prepared-" + name + ".txt")).read_text(
                            errors="replace"
                        ),
                        flush=True,
                    )
                    raise RuntimeError("Untimed preparation failed: " + name)
                prepared = json.loads(prepared_path.read_text())
                item = next(v for v in manifest()["cases"] if v["case"]["name"] == name)
                checked_preparation(prepared, item)
                checked_artifacts(prepared, report)
                print(
                    "当前独立准备完成；与历史二进制一致：",
                    name,
                    prepared["historical_binary_match"],
                    flush=True,
                )
                if not prepared["historical_binary_match"]:
                    print(
                        "历史时延仅供背景，不作为本次二进制的性能数据；本次采集必须与当前准备产物逐字节一致。",
                        flush=True,
                    )
                cmd = profiling.ncu_command(
                    facts["ncu"],
                    csv_path,
                    [
                        sys.executable,
                        "-I",
                        str(Path(__file__).resolve()),
                        *sys.argv[1:],
                        "--worker-case",
                        name,
                    ],
                    metrics=facts["selection"]["metrics"],
                )
                entry = dict(case=name, status="running", command=cmd)
                facts["cases"].append(entry)
                bench.support.write_json(
                    report / "production-profile-summary.json", facts
                )
                with (report / ("profile-" + name + ".txt")).open("w") as log:
                    rc = profiling.run_process(
                        cmd,
                        timeout=options.timeout,
                        env=bench.child_environment(report, name),
                        log=log,
                    )
                entry["exit_code"] = rc
                if rc:
                    print(
                        "\n".join(
                            (report / ("profile-" + name + ".txt"))
                            .read_text(errors="replace")
                            .splitlines()[-60:]
                        ),
                        flush=True,
                    )
                    raise RuntimeError(
                        "NCU worker failed: " + name + "; see profile-" + name + ".txt"
                    )
                result = counter_rows(report, name, facts["selection"])
                entry.update(status="passed", csv_sha256=base.sha(csv_path))
            if entry not in facts["cases"]:
                facts["cases"].append(entry)
            bench.support.write_json(report / ("counters-" + name + ".json"), result)
            display_counters(result)
        facts.update(
            status="passed",
            hardware_counters_complete=True,
            diagnostics_complete=all(
                json.loads((report / ("counters-" + n + ".json")).read_text())[
                    "diagnostics_complete"
                ]
                for n in CASES
            ),
        )
        return 0
    except BaseException:
        facts.update(status="failed", error=traceback.format_exc())
        print(facts["error"], flush=True)
        return 1
    finally:
        bench.support.write_json(report / "production-profile-summary.json", facts)


def show_report(report):
    report = Path(report)
    for name in CASES:
        p = report / ("historical-" + name + ".json")
        if p.is_file():
            historical_timing(json.loads(p.read_text()), name)
        p = report / ("counters-" + name + ".json")
        if p.is_file():
            display_counters(json.loads(p.read_text()))
    p = report / "production-profile-summary.json"
    if p.is_file():
        s = json.loads(p.read_text())
        print(
            "采集状态:",
            s["status"],
            "完整诊断计数器:",
            s.get("diagnostics_complete", False),
            flush=True,
        )
        if s.get("error"):
            print(s["error"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir")
    parser.add_argument("--baseline-report")
    parser.add_argument("--flagtree", default="/opt/flagtree")
    parser.add_argument("--ncu")
    parser.add_argument("--prepare-ncu", action="store_true")
    parser.add_argument("--resume-report")
    parser.add_argument("--show-report")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-initial-temp-c", type=int, default=80)
    parser.add_argument("--cooldown-timeout", type=int, default=120)
    parser.add_argument("--cooldown-interval", type=int, default=10)
    parser.add_argument("--worker-case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--prepare-only", action="store_true", help=argparse.SUPPRESS)
    o = parser.parse_args()
    if o.show_report:
        show_report(o.show_report)
        return 0
    if not o.report_dir or not o.baseline_report:
        parser.error("--report-dir and --baseline-report are required for capture")
    if (
        min(o.timeout, o.cooldown_interval) <= 0
        or o.cooldown_timeout < 0
        or not 1 <= o.max_initial_temp_c <= 150
    ):
        parser.error("Invalid timeout/thermal limits")
    o.report_dir = str(Path(o.report_dir).resolve())
    Path(o.report_dir).mkdir(parents=True, exist_ok=True)
    o.compiler, o.candidate = "flagtree", "production"
    from tools.qsa_prefill_flagtree_ab import configure

    configure(o.flagtree)
    return worker(o) if o.worker_case else coordinate(o)


if __name__ == "__main__":
    sys.exit(main())
