# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Fixed-choice bulk/tail store probe; two cases, no production edits or tuning."""

import argparse
import csv
import json
import os
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_bf16_query_probe as shared  # noqa: E402
from tools import qsa_prefill_query_split_matched as fixed  # noqa: E402

CASES = ("perf_n1_q16384_p102400_bfloat16", "fp8_long100k_float8_e4m3fn")
CANDIDATE_SHA = "715611ece37104b578bf600369956979fb710f9da94e679a23139befb637e0f9"
sha = shared.sha
compiler_identity = shared.compiler_identity
bind = shared.bind


def verify_source():
    old = ROOT / "tools/qsa_prefill_query_page_kernel.py"
    new = ROOT / "tools/qsa_prefill_store_probe_kernel.py"
    if sha(old) != shared.OLD_KERNEL_SHA or sha(new) != CANDIDATE_SHA:
        raise RuntimeError("Frozen baseline or private store-probe source changed")
    return dict(baseline_sha256=sha(old), candidate_sha256=sha(new))


def load_record(directory, name):
    from tools.qsa_prefill_memory_probe import payload

    record = json.loads((Path(directory) / (name + ".json")).read_text())
    traces = record["cache_granularity"]["production"]
    if (
        record.get("status") != "passed"
        or name not in CASES
        or record["case"]["name"] != name
        or record.get("candidate_kind") != "query-stage-joint"
        or record["runtime"]["operator_binding"]["candidate_sha256"]
        != shared.OLD_KERNEL_SHA
        or any(t["route"] != "rectangular" for t in traces)
    ):
        raise RuntimeError("Requires original measured combined long-case fixture")
    expected = [
        (0, 4519, 29696),
        (4519, 4519, 29696),
        (9038, 4519, 29696),
        (13557, 2827, 29696),
    ]
    if [(t["offset"], t["rows"], t["width"]) for t in traces] != expected:
        raise RuntimeError("Original public chunk geometry changed")
    for t in traces:
        payload(record, t)
        if t["constants"] != dict(
            shared.CONFIG, QUERY_PARTS=2 if name == CASES[1] else 1
        ):
            raise RuntimeError("Original long-case fixed configuration changed")
    return record, traces


def eligible(logits, width):
    return (
        width > 0
        and width % 4 == 0
        and logits.element_size() == 4
        and str(logits.dtype) == "torch.float32"
        and logits.is_contiguous()
        and logits.ndim == 2
        and logits.shape[1] == width
        and logits.data_ptr() % 16 == 0
    )


class StoreJIT:
    def __init__(self, original, bulk, names):
        self.original, self.bulk, self.names = original, bulk, names

    def __getitem__(self, grid):
        def launch(*values, **kwargs):
            named = dict(zip(self.names, values))
            named.update(kwargs)
            jit = (
                self.bulk
                if eligible(named["logits_ptr"], named["logits_width"])
                else self.original
            )
            return jit[grid](*values, **kwargs)

        return launch


def make_pair(production, traces):
    from tools import qsa_prefill_stage_joint_trial as joint
    from tools import qsa_prefill_store_probe_kernel as kernel

    source = joint.candidate_module(production)
    pair = fixed.modules(source, source, traces)
    entry = pair[1]._qsa_prefill_score_kernel
    entry.jit = StoreJIT(
        entry.jit, kernel._qsa_prefill_score_kernel.fn.fn, entry.arg_names
    )
    if pair[0].persistent_topk is not pair[1].persistent_topk:
        raise RuntimeError("Store probe changed TopK")
    return pair


def boundary_checks(production, helpers, torch, device):
    from tools import qsa_prefill_score_benchmark as scoring

    checks = []
    visible = [0, 1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65, 79, 80, 81]
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        for width, shift in ((80, 64), (81, 64), (80, 65)):
            args = helpers.make_inputs(
                [5, 10],
                [96, 96],
                dtype=dtype,
                padded=True,
                query_base=17,
                page_size=196,
                visible=visible,
            )
            traces, chunks, backing = [], [], []
            offset = 0
            for count in (4, 1, 10):
                storage = torch.full(
                    (count * width + shift + 64,),
                    -12345.0,
                    dtype=torch.float32,
                    device=args[0].device,
                )
                logits = storage[shift:-64].view(count, width)
                config = dict(
                    shared.CONFIG, QUERY_PARTS=2 if dtype == torch.float8_e4m3fn else 1
                )
                traces.append(
                    dict(
                        offset=offset,
                        rows=count,
                        width=width,
                        route="rectangular",
                        constants=config,
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
            pair = make_pair(production, traces)
            outputs = []
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
                values = []
                for c, storage in zip(chunks, backing):
                    call = c["score_args"]
                    offset, count = call[-2:]
                    logits = call[5]
                    mask = (
                        torch.arange(width, device=logits.device)[None, :]
                        < args[4][offset : offset + count, None]
                    )
                    if not bool((storage[:shift] == -12345.0).all()) or not bool(
                        (storage[-64:] == -12345.0).all()
                    ):
                        raise RuntimeError("Store probe overwrote a canary")
                    if not bool((logits[~mask] == -12345.0).all()):
                        raise RuntimeError("Store probe overwrote masked output")
                    values.append(logits[mask].clone())
                outputs.append(values)
            for a, b in zip(*outputs):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            resources = fixed.freeze_and_records(pair)
            active = width % 4 == 0 and shift % 4 == 0
            if not active:
                for a, b in zip(resources["baseline"], resources["candidate"]):
                    if (
                        a["cubin_sha256"] != b["cubin_sha256"]
                        or a["ptx_sha256"] != b["ptx_sha256"]
                    ):
                        raise RuntimeError(
                            "Unaligned fallback changed the original binary"
                        )
            checks.append(
                dict(
                    status="passed",
                    dtype=str(dtype),
                    width=width,
                    shift=shift,
                    all_rows_checked=checked,
                    bulk_active=active,
                )
            )
            print("完整边界校验:", checks[-1], flush=True)
    return checks


def artifact_records(pair, report, name):
    rows = []
    for side, module in zip(("baseline", "candidate"), pair):
        for chunk, (
            kernel,
            config,
        ) in module._qsa_prefill_score_kernel.compiled.items():
            ptx = kernel.asm["ptx"]
            # Static sites only, not dynamic transaction counts.
            ops = re.findall(r"\bst\.global[\w.]*", re.sub(r"//[^\n]*", "", ptx))
            prefix = f"{name}-{side}-{chunk[1]}"
            files = {}
            for suffix in ("ptx", "ttgir"):
                text = kernel.asm.get(suffix)
                if text is not None:
                    path = report / (prefix + "." + suffix + ".txt")
                    path.write_text(text)
                    files[suffix] = dict(path=path.name, sha256=sha(path))
            rows.append(
                dict(
                    side=side,
                    chunk=list(chunk),
                    constants=config,
                    files=files,
                    store_ops={op: ops.count(op) for op in sorted(set(ops))},
                )
            )
    return rows


def print_results(result):
    from tools.qsa_prefill_report import fmt, table

    print(
        "\nBulk/tail store probe:",
        result.get("case", {}).get("name", "preparing"),
        flush=True,
    )
    print(
        (
            "仅编译/正确性检查；无性能计时。"
            if result.get("compile_only")
            else "kernel模式；原combined/候选；不是全30组生产验收。"
        ),
        flush=True,
    )
    table(
        ("stage", "original_us", "bulk_tail_us", "speedup", "paired_min", "paired_max"),
        [
            (
                k,
                fmt(v["baseline_us"]["median"]),
                fmt(v["candidate_us"]["median"]),
                fmt(v["speedup"], 4),
                fmt(v["paired_ratios"]["min"], 4),
                fmt(v["paired_ratios"]["max"], 4),
            )
            for k, v in result.get("stages", {}).items()
            if v.get("measurement_validated")
        ],
    )
    table(
        ("A/A", "ratio", "min", "max"),
        [
            (
                k,
                fmt(v["speedup"], 4),
                fmt(v["paired_ratios"]["min"], 4),
                fmt(v["paired_ratios"]["max"], 4),
            )
            for k, v in result.get("self_controls", {}).items()
            if v.get("measurement_validated")
        ],
    )
    if result.get("error"):
        print(result["error"], flush=True)


def worker(options):
    from tools import benchmark_qsa_prefill_triton_stages as bench

    support, scoring, ab = bench.support, bench.scoring, bench.ab
    report = Path(options.report_dir)
    name = options.worker_case
    path = report / (name + ".json")
    result = dict(
        status="running",
        acceptance_eligible=False,
        mode="kernel",
        stages={},
        source=verify_source(),
        production_modified=False,
        compile_only=getattr(options, "compile_only", False),
    )
    try:
        saved, traces = load_record(options.baseline_report, name)
        result["fixed_traces"] = traces
        torch, native, production, _, helpers, device, info = bench.runtime(
            report / "qsa_indexer_vllm.py", path, options
        )
        result["runtime"] = info
        expected_compiler = compiler_identity(saved["runtime"])
        if compiler_identity(info) != expected_compiler:
            raise RuntimeError(
                "Compiler/torch/vLLM identity differs from the captured baseline"
            )
        result["compiler_matches_capture"] = True
        result["boundary_checks"] = boundary_checks(production, helpers, torch, device)
        case = next(
            c for c in bench.workloads.cases("scored-prefill") if c["name"] == name
        )
        result["case"] = case
        os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = str(case["budget_mb"])
        args = bench.workloads.make_inputs(case, helpers, torch)
        fingerprints = support.fingerprint(args, torch)
        if fingerprints != saved["inputs"]:
            raise RuntimeError(
                "Probe inputs differ from the measured long-case fixture"
            )
        result["inputs"] = fingerprints
        pair = make_pair(production, traces)
        if result["compile_only"]:
            from unittest.mock import patch

            # Capture original score geometry/common logits, while the observer
            # records TopK arguments without executing the selection kernel.
            with patch.object(pair[0], "persistent_topk", lambda *a, **kw: None):
                chunks = ab.observe(pair[0], "persistent_topk", args)
        else:
            chunks = ab.observe(pair[0], "persistent_topk", args)
        result["topk_skipped_for_compile_only"] = result["compile_only"]
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
            for fn in [bundle["score"]] if result["compile_only"] else bundle.values():
                fn()
        device.synchronize()
        resources = fixed.freeze_and_records(pair)
        result["kernels_before"] = resources
        result["artifacts"] = artifact_records(pair, report, name)
        for item in result["artifacts"]:
            print(
                "Actual compiled stores:",
                item["side"],
                item["chunk"],
                item["store_ops"],
                flush=True,
            )
        for side, entries in (
            ("baseline", resources["baseline"]),
            ("candidate", resources["candidate"]),
        ):
            for entry in entries:
                print(
                    "Resources:",
                    side,
                    entry["chunk"],
                    "regs",
                    entry.get("registers_per_thread"),
                    "smem",
                    entry.get("shared_bytes"),
                    "spills",
                    entry.get("spill_count"),
                    flush=True,
                )
        if result["compile_only"]:
            result["score_validation"] = [validate_side(i, "score") for i in (0, 1)]
            result["kernels_after"] = fixed.freeze_and_records(pair)
            result["common_after"] = [support.fingerprint((v,), torch) for v in common]
            if (
                result["kernels_after"] != resources
                or result["common_after"] != result["common_before"]
                or support.fingerprint(args, torch) != fingerprints
                or verify_source() != result["source"]
            ):
                raise RuntimeError("Compile-only inputs/source/binaries changed")
            support.guarded_telemetry(
                info["uuid"],
                "after_compile_validation",
                result,
                path,
                support.gpu_guard_module(),
            )
            result.update(status="passed", measurement_performed=False)
            return 0
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
            for i, side in enumerate(("original", "bulk_tail"))
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
        result["kernels_after"] = fixed.freeze_and_records(pair)
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
        with (report / (name + ".tsv")).open("w") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(
                ("stage", "original_us", "bulk_tail_us", "speedup", "paired_ratios")
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


def coordinate(options):
    from tools import benchmark_qsa_prefill_triton_stages as bench
    from tools import qsa_prefill_profiling as profiling

    report = Path(options.report_dir)
    summary = dict(
        status="preparing",
        acceptance_eligible=False,
        production_modified=False,
        source=verify_source(),
        options=vars(options),
        git_head=bench.support.command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
    )
    try:
        for name in options.cases:
            load_record(options.baseline_report, name)
        shared.prepare_environment(options, summary)
        names = (
            "tools/qsa_prefill_store_probe.py",
            "tools/qsa_prefill_store_probe_kernel.py",
            "tools/test_qsa_prefill_store_probe.py",
            "tools/run_qsa_prefill_store_probe_h800.sh",
            "docs/qsa_prefill_store_probe.md",
        )
        bench.support.write_json(
            report / "store-sources.json",
            {n: dict(text=(ROOT / n).read_text(), sha256=sha(ROOT / n)) for n in names},
        )
        summary["cases"] = []
        for name in options.cases:
            if not bench.idle_gate(
                options,
                summary,
                report,
                "before_" + name,
                bench.support.gpu_guard_module(),
            ):
                raise RuntimeError("GPU guard rejected before case")
            command = [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--worker-case",
                name,
            ]
            code = profiling.run_process(
                command,
                timeout=options.timeout,
                env=bench.child_environment(report, name),
            )
            result = (
                json.loads((report / (name + ".json")).read_text())
                if (report / (name + ".json")).is_file()
                else {}
            )
            summary["cases"].append(
                dict(
                    case=name,
                    exit_code=code,
                    status=result.get("status"),
                    stages=result.get("stages", {}),
                )
            )
            if code or result.get("status") != "passed":
                raise RuntimeError("Store-probe worker failed: " + name)
            if getattr(options, "compile_only", False):
                if (
                    not result.get("compile_only")
                    or result.get("measurement_performed") is not False
                    or result.get("stages")
                    or len(result.get("score_validation", [])) != 2
                ):
                    raise RuntimeError("Invalid compile-only result")
            elif set(result.get("stages", {})) != {"score", "topk", "pipeline"}:
                raise RuntimeError("Missing score/TopK/pipeline timings")
        summary.update(status="passed")
        return 0
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        print(summary["error"], flush=True)
        return 1
    finally:
        bench.support.write_json(report / "store-summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--flagtree", default="/opt/flagtree")
    parser.add_argument("--rounds", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile/validate/export actual artifacts; no performance timing",
    )
    parser.add_argument("--kernel-warmup-ms", type=int, default=200)
    parser.add_argument("--kernel-rep-ms", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-initial-temp-c", type=int, default=80)
    parser.add_argument("--cooldown-timeout", type=int, default=120)
    parser.add_argument("--cooldown-interval", type=int, default=10)
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--worker-case", choices=CASES, help=argparse.SUPPRESS)
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
    ):
        parser.error(
            "Positive timing/timeout/interval and nonnegative cooldown required"
        )
    if len(set(options.cases)) != len(options.cases):
        parser.error("Duplicate cases")
    options.report_dir = str(Path(options.report_dir).resolve())
    options.baseline_report = str(Path(options.baseline_report).resolve())
    Path(options.report_dir).mkdir(parents=True, exist_ok=True)
    options.compiler, options.candidate = "flagtree", "production"
    return worker(options) if options.worker_case else coordinate(options)


if __name__ == "__main__":
    sys.exit(main())
