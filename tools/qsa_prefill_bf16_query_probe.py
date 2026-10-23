# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""One BF16 fixed-tile probe, with numerical guards and the original kernel timer."""

import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
CASE = "perf_n1_q16384_p102400_bfloat16"
CONFIG = dict(
    num_warps=4,
    num_stages=1,
    num_ctas=1,
    TILE_R=64,
    BLOCK_N=64,
    K_TILES=16,
    QUERY_PARTS=1,
)
OLD_KERNEL_SHA = "1ef12d431b65971c5617f0a34f614360008c652a36125d5679da00206b212d7c"
OLD_GATE = "SPLIT_QUERY: tl.constexpr = IS_FP8 and TILE_R >= 32 and QUERY_PARTS == 2"
NEW_GATE = "SPLIT_QUERY: tl.constexpr = TILE_R >= 32 and QUERY_PARTS == 2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source():
    old_path = ROOT / "tools/qsa_prefill_query_page_kernel.py"
    new_path = ROOT / "tools/qsa_prefill_bf16_query_probe_kernel.py"
    old, new = old_path.read_text(), new_path.read_text()
    if sha(old_path) != OLD_KERNEL_SHA or old.count(OLD_GATE) != 1:
        raise RuntimeError("Measured combined scorer source changed")
    # Ignore only the module description; all imports, decorators and math remain exact.
    old = old[old.index("import triton") :]
    new = new[new.index("import triton") :]
    if new != old.replace(OLD_GATE, NEW_GATE, 1):
        raise RuntimeError("Probe changed code beyond the single BF16 split gate")
    return dict(baseline_sha256=sha(old_path), candidate_sha256=sha(new_path))


def baseline_record(directory):
    path = Path(directory) / (CASE + ".json")
    record = json.loads(path.read_text())
    traces = record["cache_granularity"]["production"]
    expected = [
        (0, 4519, 29696),
        (4519, 4519, 29696),
        (9038, 4519, 29696),
        (13557, 2827, 29696),
    ]
    if (
        record.get("status") != "passed"
        or record.get("candidate_kind") != "query-stage-joint"
        or [(t["offset"], t["rows"], t["width"]) for t in traces] != expected
        or any(t["route"] != "rectangular" or t["constants"] != CONFIG for t in traces)
        or record["runtime"]["operator_binding"]["candidate_sha256"] != OLD_KERNEL_SHA
    ):
        raise RuntimeError(
            "Probe requires the original successful fixed-tile BF16 capture"
        )
    return record, traces


def compiler_identity(info):
    profile = info["flagtree_runtime"]
    return dict(
        torch=info["torch_version"],
        triton=info["triton_version"],
        vllm=info["versions"]["vllm"],
        compiler_files={k: v["sha256"] for k, v in info["compiler_files"].items()},
        flagtree_files={k: v["sha256"] for k, v in profile["files"].items()},
        ptxas=profile["ptxas"]["sha256"],
        libcuda=profile["cuda"]["libcuda_sha256"],
    )


def make_pair(production, traces, policy="all"):
    from tools import qsa_prefill_bf16_query_probe_kernel as kernel
    from tools import qsa_prefill_query_split_matched as fixed
    from tools import qsa_prefill_stage_joint_trial as joint

    source = joint.candidate_module(production)
    pair = fixed.modules(source, source, traces, query_parts=(1, 2))
    # Both entries bypass autotune. Only the candidate JIT admits BF16 splitting.
    pair[1]._qsa_prefill_score_kernel.jit = kernel._qsa_prefill_score_kernel.fn.fn
    if pair[0].persistent_topk is not pair[1].persistent_topk:
        raise RuntimeError("Probe changed TopK")
    from tools.qsa_prefill_bf16_query_policy import apply

    return apply(pair, policy)


def print_results(result):
    from tools.qsa_prefill_report import fmt, table

    print(
        "\nBF16 query probe:",
        result.get("case", {}).get("name", "preparing"),
        flush=True,
    )
    print(
        "original fixed P1 vs candidate; query policy:",
        result.get("query_policy", "all"),
        flush=True,
    )
    print("仅 probe，不作为生产/全30组验收；speedup=P1/candidate。", flush=True)
    table(
        ("stage", "P1_us", "candidate_us", "speedup", "paired_min", "paired_max"),
        [
            (
                name,
                fmt(v["baseline_us"]["median"]),
                fmt(v["candidate_us"]["median"]),
                fmt(v["speedup"], 4),
                fmt(v["paired_ratios"]["min"], 4),
                fmt(v["paired_ratios"]["max"], 4),
            )
            for name, v in result.get("stages", {}).items()
            if v.get("measurement_validated")
        ],
    )
    controls = result.get("self_controls", {})
    if controls:
        print("A/A: same callable and buffers", flush=True)
        table(
            ("side:stage", "ratio", "paired_min", "paired_max"),
            [
                (
                    name,
                    fmt(v["speedup"], 4),
                    fmt(v["paired_ratios"]["min"], 4),
                    fmt(v["paired_ratios"]["max"], 4),
                )
                for name, v in controls.items()
                if v.get("measurement_validated")
            ],
        )
    resources = result.get("kernels_before", {})
    rows = []
    for side in ("baseline", "candidate"):
        for kernel in resources.get(side, []):
            rows.append(
                (
                    side,
                    "/".join(map(str, kernel["chunk"])),
                    fmt(kernel.get("registers_per_thread"), 0),
                    fmt(kernel.get("shared_bytes"), 0),
                    fmt(kernel.get("spill_count"), 0),
                    "/".join(
                        f"{label}{kernel['constants'][key]}"
                        for label, key in (
                            ("R", "TILE_R"),
                            ("N", "BLOCK_N"),
                            ("KT", "K_TILES"),
                            ("W", "num_warps"),
                            ("S", "num_stages"),
                            ("CTA", "num_ctas"),
                            ("P", "QUERY_PARTS"),
                        )
                    ),
                )
            )
    if rows:
        print("编译资源（不是动态硬件计数器）", flush=True)
        table(
            ("side", "rows/offset/width", "regs", "shared_bytes", "spills", "config"),
            rows,
        )
    if result.get("error"):
        print(result["error"], flush=True)


def boundary_checks(production, helpers, torch, device):
    from tools import qsa_prefill_query_split_matched as fixed
    from tools import qsa_prefill_score_benchmark as scoring

    checks = []
    # Request tails around half/full tile, shuffled page196 and non-unit outer strides.
    lengths = [1, 31, 32, 33, 63, 64, 65]
    contexts = [791] * len(lengths)
    chunk_rows = [31, 1, 32, 1, 63, 64, 65, 32]
    for padded, base in ((False, 0), (True, 17)):
        args = helpers.make_inputs(
            lengths,
            contexts,
            dtype=torch.bfloat16,
            padded=padded,
            query_base=base,
            page_size=196,
        )
        args[4][0] = 0
        width = 832
        offset = 0
        traces, chunks, backing = [], [], []
        for count in chunk_rows:
            storage = torch.full(
                (count * width + 128,),
                -12345.0,
                dtype=torch.float32,
                device=args[0].device,
            )
            logits = storage[64:-64].view(count, width)
            traces.append(
                dict(
                    offset=offset,
                    rows=count,
                    width=width,
                    route="rectangular",
                    constants=dict(CONFIG),
                )
            )
            chunks.append(
                dict(
                    score_args=(*args[:5], logits, args[7], width, offset, count),
                    score_kwargs={},
                )
            )
            backing.append(storage)
            offset += count
        if offset != sum(lengths):
            raise RuntimeError("Boundary fixture chunks do not cover every row")
        pair = make_pair(production, traces)
        scores = []
        for module in pair:
            for storage in backing:
                storage.fill_(-12345.0)
            checked = scoring.validate_scores(
                module._prefill_logits,
                chunks,
                args,
                torch,
                lambda n: list(range(n)),
                device,
            )
            outputs = []
            for chunk, storage in zip(chunks, backing):
                call = chunk["score_args"]
                offset, count = call[-2:]
                logits = call[5]
                valid = (
                    torch.arange(width, device=logits.device)[None, :]
                    < args[4][offset : offset + count, None]
                )
                if not bool((storage[:64] == -12345.0).all()) or not bool(
                    (storage[-64:] == -12345.0).all()
                ):
                    raise RuntimeError("Query split overwrote workspace canary")
                if not bool((logits[~valid] == -12345.0).all()):
                    raise RuntimeError("Query split stored outside visible columns")
                outputs.append(logits[valid].clone())
            scores.append(outputs)
        for left, right in zip(*scores):
            torch.testing.assert_close(left, right, rtol=2e-4, atol=2e-3)
        fixed.freeze_and_records(pair, query_parts=(1, 2))
        checks.append(
            dict(
                status="passed",
                padded=padded,
                query_base=base,
                all_rows_checked=checked,
                chunk_rows=chunk_rows,
            )
        )
        print("边界校验 passed:", checks[-1], flush=True)
    return checks


def bind(module, chunks, common):
    from tools.qsa_prefill_score_benchmark import score_function

    def topk():
        for scores, c in zip(common, chunks):
            module.persistent_topk(scores, *c["topk_args"][1:], **c["topk_kwargs"])

    def pipeline():
        for c in chunks:
            logits = module._prefill_logits(*c["score_args"], **c["score_kwargs"])
            module.persistent_topk(logits, *c["topk_args"][1:], **c["topk_kwargs"])

    return dict(
        score=score_function(module._prefill_logits, chunks),
        topk=topk,
        pipeline=pipeline,
    )


def worker(options):
    from tools import benchmark_qsa_prefill_triton_stages as bench
    from tools import qsa_prefill_bf16_query_policy as policy

    support, scoring, ab = bench.support, bench.scoring, bench.ab
    report = Path(options.report_dir)
    upstream = getattr(options, "suite", "single") == "upstream-prefill"
    name = getattr(options, "worker_case", None) or CASE
    query_policy = getattr(options, "query_policy", "all")
    path = report / ((name + ".json") if upstream else "probe.json")
    result = dict(
        status="running",
        acceptance_eligible=False,
        mode="kernel",
        stages={},
        fixed_config=None if upstream else CONFIG,
        query_parts_pair=policy.pair_label(query_policy),
        query_policy=query_policy,
        source=verify_source(),
    )
    try:
        if upstream:
            from tools.qsa_prefill_bf16_query_upstream import saved_record

            saved, traces = saved_record(name)
        else:
            saved, traces = baseline_record(options.baseline_report)
        result["fixed_traces"] = traces
        torch, native, production, _, helpers, device, info = bench.runtime(
            report / "qsa_indexer_vllm.py", path, options
        )
        result["runtime"] = info
        expected_compiler = (
            saved["compiler_identity"]
            if upstream
            else compiler_identity(saved["runtime"])
        )
        if compiler_identity(info) != expected_compiler:
            raise RuntimeError(
                "Compiler/torch/vLLM identity differs from the captured baseline"
            )
        result["compiler_matches_capture"] = True
        first = (
            getattr(options, "cases", None) or [bench.workloads.cases()[0]["name"]]
        )[0]
        result["boundary_checks"] = (
            [dict(status="covered_by_first_case", case=first)]
            if upstream and name != first
            else boundary_checks(production, helpers, torch, device)
        )
        case = next(c for c in bench.workloads.cases() if c["name"] == name)
        result["case"] = case
        os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = str(case["budget_mb"])
        args = bench.workloads.make_inputs(case, helpers, torch)
        fingerprints = support.fingerprint(args, torch)
        if fingerprints != saved["inputs"]:
            raise RuntimeError("Probe inputs differ from the measured BF16 fixture")
        result["inputs"] = fingerprints
        pair = (
            make_pair(production, traces)
            if query_policy == "all"
            else make_pair(production, traces, policy=query_policy)
        )
        chunks = ab.observe(pair[0], "persistent_topk", args)
        bench.verify_geometry(case, args, chunks)
        common = [c.pop("common_logits") for c in chunks]
        bundles = [bind(module, chunks, common) for module in pair]
        bundles[1]["topk"] = bundles[0]["topk"]  # Identical callable and arguments.
        result["common_before"] = [support.fingerprint((v,), torch) for v in common]
        result["same_buffers"] = True
        result["same_topk"] = pair[0].persistent_topk is pair[1].persistent_topk

        def validate_side(index, stage):
            module = pair[index]
            if stage == "score":
                return scoring.validate_scores(
                    module._prefill_logits,
                    chunks,
                    args,
                    torch,
                    support.sample_rows,
                    device,
                )
            bundles[index][stage]()
            if stage == "topk":
                q = args[0]
                count = 0
                for scores, chunk in zip(common, chunks):
                    lengths = chunk["topk_args"][1]
                    selected = chunk["topk_args"][2]
                    for row in support.sample_rows(chunk["rows"]):
                        bench.validate_reference_selection(
                            scores[row : row + 1],
                            lengths[row : row + 1],
                            selected[row : row + 1],
                            q,
                            torch,
                            helpers,
                        )
                        count += 1
                return count
            # Validate immediately before another side overwrites shared buffers.
            fn = bundles[index]["pipeline"]
            return bench.validate_pipeline(
                dict(baseline=fn, candidate=fn), args, args, chunks, torch, helpers
            )

        for bundle in bundles:
            for fn in bundle.values():
                fn()
        device.synchronize()
        resources = policy.freeze(pair, query_policy)
        result["kernels_before"] = resources
        timer, metadata = scoring.make_timer(
            options.kernel_warmup_ms, options.kernel_rep_ms
        )
        result["timer"] = {k: v for k, v in metadata.items() if k != "includes_topk"}
        guard = support.gpu_guard_module()
        monitor = lambda label: support.guarded_telemetry(
            info["uuid"], label, result, path, guard
        )
        jobs = [(name, 0, 1) for name in ("score", "topk", "pipeline")]
        jobs += [
            (f"{side}:{name}", i, i)
            for i, side in enumerate(("P1", "P2"))
            for name in ("score", "pipeline")
        ]
        result["self_controls"] = {}
        for label, left, right in jobs:
            stage = label.split(":")[-1]
            bucket = result["self_controls"] if ":" in label else result["stages"]
            row = bucket[label] = dict(
                status="validating", includes_topk=stage != "score"
            )
            row["validation_before"] = [validate_side(i, stage) for i in (left, right)]
            support.write_json(path, result)
            measured, _, order = scoring.measure_scores(
                timer,
                bundles[left][stage],
                bundles[right][stage],
                options.rounds,
                lambda phase: monitor(label + ":" + phase),
                support.stats,
            )
            row.update(measured, status="pending_validation", measurement_order=order)
            row["validation_after"] = [validate_side(i, stage) for i in (left, right)]
            row.update(status="passed", measurement_validated=True)
            support.write_json(path, result)
            print_results(result)
        result["kernels_after"] = policy.freeze(pair, query_policy)
        result["common_after"] = [support.fingerprint((v,), torch) for v in common]
        if (
            result["common_after"] != result["common_before"]
            or verify_source() != result["source"]
            or result["kernels_after"] != resources
            or support.fingerprint(args, torch) != fingerprints
        ):
            raise RuntimeError(
                "Probe source/kernel/config/input/common-logits changed during timing"
            )
        monitor("after_all_validation")
        result.update(status="passed", production_modified=False)
        with (report / ((name + ".tsv") if upstream else "probe-timings.tsv")).open(
            "w"
        ) as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(
                ("stage", "P1_us", "candidate_us", "speedup", "paired_ratios")
            )
            for name, value in result["stages"].items():
                writer.writerow(
                    (
                        name,
                        value["baseline_us"]["median"],
                        value["candidate_us"]["median"],
                        value["speedup"],
                        value["paired_ratios"]["samples"],
                    )
                )
        return 0
    except BaseException:
        result.update(status="failed", error=traceback.format_exc())
        return 1
    finally:
        support.write_json(path, result)
        print_results(result)


def prepare_environment(options, summary):
    from tools import benchmark_qsa_prefill_triton_stages as bench
    from tools import qsa_prefill_flagtree_ab as profile

    report = Path(options.report_dir)
    baseline_record(options.baseline_report)
    tools = json.loads(
        (Path(options.baseline_report) / "flagtree134-tools.json").read_text()
    )
    assembler = Path(tools["toolchain"]["ptxas"]["path"])
    if sha(assembler) != profile.PTXAS_SHA256:
        raise RuntimeError("Previous ptxas13.4 binary missing or changed")
    os.environ["TRITON_PTXAS_PATH"] = str(assembler)
    bench.support.write_json(report / "flagtree134-tools.json", tools)
    if not profile.select_gpu(summary, options):
        raise RuntimeError("No exclusive idle H800 available")
    if not bench.idle_gate(
        options, summary, report, "before_probe", bench.support.gpu_guard_module()
    ):
        raise RuntimeError("Initial GPU guard rejected")
    reference = (Path(options.baseline_report) / "qsa_indexer_vllm.py").read_bytes()
    if hashlib.sha256(reference).hexdigest() != bench.ab.REFERENCE_SHA256:
        raise RuntimeError("Previous native reference source changed")
    (report / "qsa_indexer_vllm.py").write_bytes(reference)
    paths = set(bench.SOURCES) | {
        "tools/qsa_prefill_bf16_query_probe.py",
        "tools/qsa_prefill_bf16_query_probe_kernel.py",
        "tools/test_qsa_prefill_bf16_query_probe.py",
        "tools/qsa_prefill_bf16_query_upstream.py",
        "tools/qsa_prefill_bf16_query_policy.py",
        "tools/test_qsa_prefill_bf16_query_policy.py",
        "tools/test_qsa_prefill_bf16_query_upstream.py",
        "docs/qsa_prefill_bf16_query_configs.json",
        "tools/run_qsa_prefill_bf16_query_probe_h800.sh",
        "docs/qsa_prefill_bf16_query_probe.md",
    }
    bench.support.write_json(
        report / "sources.json",
        {
            name: dict(sha256=sha(ROOT / name), text=(ROOT / name).read_text())
            for name in sorted(paths)
        },
    )
    bench.support.write_json(
        report / "baseline-case.json", baseline_record(options.baseline_report)[0]
    )


def coordinate(options):
    from tools import benchmark_qsa_prefill_triton_stages as bench
    from tools import qsa_prefill_profiling as profiling

    report = Path(options.report_dir)
    summary = dict(
        status="preparing",
        acceptance_eligible=False,
        formal_acceptance_eligible=False,
        source=verify_source(),
        options=vars(options),
        git_head=bench.support.command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        git_status=bench.support.command(
            ["git", "-C", str(ROOT), "status", "--porcelain"]
        ),
    )
    output = report / "probe-summary.json"
    try:
        prepare_environment(options, summary)
        command = [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "--worker",
            *sys.argv[1:],
        ]
        code = profiling.run_process(command, timeout=options.timeout)
        summary["worker_exit_code"] = code
        if code:
            raise RuntimeError("Probe worker failed: " + str(code))
        result = json.loads((report / "probe.json").read_text())
        if result["status"] != "passed":
            raise RuntimeError("Probe did not pass validation")
        summary.update(status="passed", production_modified=False)
        return 0
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        print(summary["error"], flush=True)
        return 1
    finally:
        bench.support.write_json(output, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--flagtree", default="/opt/flagtree")
    parser.add_argument("--rounds", type=int, choices=(1, 2), default=2)
    parser.add_argument("--kernel-warmup-ms", type=int, default=200)
    parser.add_argument("--kernel-rep-ms", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-initial-temp-c", type=int, default=80)
    parser.add_argument("--cooldown-timeout", type=int, default=120)
    parser.add_argument("--cooldown-interval", type=int, default=10)
    from tools.qsa_prefill_upstream_workloads import cases as upstream_cases

    names = [c["name"] for c in upstream_cases()]
    parser.add_argument(
        "--suite", choices=("single", "upstream-prefill"), default="single"
    )
    parser.add_argument("--query-policy", choices=("all", "large-tile"), default="all")
    parser.add_argument("--cases", choices=names, nargs="+")
    parser.add_argument("--worker-case", choices=names, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    options.report_dir = str(Path(options.report_dir).resolve())
    options.baseline_report = str(Path(options.baseline_report).resolve())
    if (
        min(
            options.kernel_warmup_ms,
            options.kernel_rep_ms,
            options.timeout,
            options.cooldown_interval,
        )
        <= 0
        or options.cooldown_timeout < 0
    ):
        parser.error(
            "Positive timing/timeout/interval and nonnegative cooldown required"
        )
    if options.suite == "single" and (options.cases or options.worker_case):
        parser.error("--cases/--worker-case require --suite upstream-prefill")
    if (
        options.suite == "upstream-prefill"
        and options.worker
        and not options.worker_case
    ):
        parser.error("Suite workers require --worker-case")
    options.compiler, options.candidate = "flagtree", "production"
    if options.suite == "upstream-prefill" and not options.worker:
        from tools.qsa_prefill_bf16_query_upstream import (
            coordinate as upstream_coordinate,
        )

        return upstream_coordinate(options)
    return worker(options) if options.worker else coordinate(options)


if __name__ == "__main__":
    sys.exit(main())
