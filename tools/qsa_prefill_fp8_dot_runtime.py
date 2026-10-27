# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Three-way FP8 source-JIT runtime probe; no IR override or compiler edits."""

import argparse
import contextlib
import copy
import importlib.util
import json
import os
import re
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_fp8_dot_issue as probe  # noqa: E402
from tools import qsa_prefill_fp8_producer_runtime as fp8  # noqa: E402

base, stage = fp8.base, fp8.stage
VARIANTS = ("production", "sequential32", "paired32")
PAIRS = ((0, 1), (0, 2), (1, 2))
SUMMARY_SHA = "4421fc08e618735df720de1fa56feab599aa3cc73cdf6b90f8d803b40f9b6dd2"
IR_SHA = dict(
    sequential32="d9ec6aa88520e1fbf7dcf143fd0b5b10078b203ecf46f87f477cc99085767f17",
    paired32="ac22a8c4abce758120f233abd2191fddfbfd229ab57aadc6f2e8f67b8882143e",
)
EXTRA = (
    set(probe.SOURCES)
    | fp8.EXTRA_SOURCES
    | {
        "tools/qsa_prefill_fp8_dot_runtime.py",
        "tools/test_qsa_prefill_fp8_dot_runtime.py",
        "tools/run_qsa_prefill_fp8_dot_runtime_h800.sh",
        "docs/qsa_prefill_fp8_dot_runtime.md",
    }
)


def lifetime(ir, ptx, kind):
    """Read-only checks; exact IR hash binds the detailed review in the board."""
    import hashlib

    if kind not in IR_SHA or hashlib.sha256(ir.encode()).hexdigest() != IR_SHA[kind]:
        raise RuntimeError("Unreviewed R32 IR")
    body = fp8.probe.body_lines(ir)
    dots = [
        i for i, line in enumerate(body) if re.search(r"\bttng.warp_group_dot\b", line)
    ]
    waits = [i for i, line in enumerate(body) if "ttng.warp_group_dot_wait" in line]
    copies = [
        i for i, line in enumerate(body) if "ttg.async_copy_global_to_local" in line
    ]
    if (len(dots), len(waits), len(copies)) != (2, 2, 1) or max(waits) >= copies[0]:
        raise RuntimeError("K overwrite precedes consumer completion")
    if "memdesc<2x64x128xf8E4M3FN" not in ir:
        raise RuntimeError("Changed K ring allocation")
    if not any(
        "ttg.async_wait" in line and "num = 0 : i32" in line
        for line in body[: min(dots)]
    ):
        raise RuntimeError("Missing current K-copy wait")
    clean = fp8.runtime.program(ptx)
    mma = [m.start() for m in re.finditer(r"\bwgmma.mma_async\.", clean)]
    done = [m for m in re.finditer(r"\bwgmma.wait_group.sync.aligned (\d+);", clean)]
    copies = [
        m.start() for m in re.finditer(r"\bcp.async.(?:cg|ca).shared.global\b", clean)
    ]
    expected = ["1", "0"] if kind == "paired32" else ["0", "0"]
    if len(mma) != 8 or [m[1] for m in done] != expected:
        raise RuntimeError("Changed final matrix/wait protocol")
    if not any(
        re.search(r"\bbar.sync\b", clean[done[-1].end() : c])
        for c in copies
        if c > max(mma)
    ):
        raise RuntimeError("Missing completion-to-next-copy CTA barrier")
    if not re.search(r"cp.async.wait_group 0;\s*bar.sync 0;", clean[max(copies) :]):
        raise RuntimeError("Missing final K-copy drain")
    # One-warp-group abstract ring model, including zero/single and wrap cases.
    for n in range(17):
        slots, read, write = [None, None], -1, 0
        if n:
            slots[0] = 0
        for tile in range(n):
            read, write = (read + 1) % 2, (write + 1) % 2
            if slots[read] != tile or read == write:
                raise RuntimeError("K ring phase model failed")
            # Both WGMMA consumers complete before issuing the next tile copy.
            if tile + 1 < n:
                slots[write] = tile + 1
    return dict(
        status="reviewed_under_explicit_sync_assumptions",
        slots=2,
        warp_groups=1,
        K_tile_bytes=8192,
        iterations_checked=list(range(17)),
        wait_protocol=expected,
        buffer_or_sync_rewrite=False,
    )


def proof():
    directory = Path(os.environ["QSA_DOT_COMPILE_REPORT"])
    summary = directory / "fp8-dot-issue-summary.json"
    if base.sha(summary) != SUMMARY_SHA:
        raise RuntimeError("Requires the exact successful R32 compile report")
    saved = json.loads(summary.read_text())
    if (
        saved["status"] != "pending_lifetime_and_accuracy"
        or saved["gpu_kernel_executed"]
    ):
        raise RuntimeError("R32 compile proof is not eligible")
    sources = json.loads((directory / "sources.json").read_text())
    if set(sources) != set(probe.SOURCES) or any(
        base.sha(ROOT / n) != r["sha256"] for n, r in sources.items()
    ):
        raise RuntimeError("Offline probe source identity changed")
    records = {}
    for item in saved["jobs"]:
        kind, result = item["kind"], item["result"]
        folder = directory / kind
        job = json.loads((folder / "spec.json").read_text())
        if result != json.loads((folder / "result.json").read_text()):
            raise RuntimeError("Compile result differs from summary")
        for filename, key in (
            ("ptx.txt", "ptx_sha256"),
            ("kernel.cubin", "cubin_sha256"),
        ):
            if base.sha(folder / filename) != result[key]:
                raise RuntimeError("Offline binary changed")
        expected_source = probe.source("paired" if kind == "paired32" else "production")
        if (folder / "kernel.py").read_text() != expected_source or base.sha(
            folder / "kernel.py"
        ) != job["module_sha256"]:
            raise RuntimeError("Offline Python source changed")
        if base.sha(folder / "options.json") != probe.OPTIONS_SHA:
            raise RuntimeError("Offline compiler options changed")
        records[kind] = dict(result=result, source=str(folder / "kernel.py"))
        if kind in IR_SHA:
            records[kind]["lifetime"] = lifetime(
                (folder / "ttgir.txt").read_text(),
                (folder / "ptx.txt").read_text(),
                kind,
            )
    if tuple(records) != VARIANTS or probe.tile32_gate(
        *(records[k]["result"] for k in VARIANTS)
    ):
        raise RuntimeError("Offline candidate gate changed")
    return records


def verify_source():
    return dict(
        fp8.baseline_source(),
        compile_summary_sha256=SUMMARY_SHA,
        profiles=proof(),
        sources={n: base.sha(ROOT / n) for n in sorted(EXTRA)},
    )


def eligible(config):
    return (
        config.get("IS_FP8") is True
        and config.get("NUM_HEADS") == 4
        and config.get("HEAD_DIM") == 128
        and all(
            config.get(k) == v for k, v in fp8.probe.CONFIG.items() if k != "TILE_R"
        )
        and config.get("TILE_R") in (32, 64)
    )


def traces_for(traces, kind):
    result = copy.deepcopy(traces)
    for t in result:
        if t["route"] == "fp8" and eligible(
            dict(IS_FP8=True, NUM_HEADS=4, HEAD_DIM=128, **t["constants"])
        ):
            if t["constants"]["TILE_R"] != 64:
                raise RuntimeError("Expected original R64 main trace")
            if kind != "production":
                t["constants"]["TILE_R"] = 32
    return result


class SourceJIT:
    def __init__(self, original, source, names, expected):
        self.original, self.source, self.names, self.expected = (
            original,
            source,
            names,
            expected,
        )
        self.prepared, self.frozen = {}, False

    def __getitem__(self, grid):
        def launch(*values, **kwargs):
            named = dict(zip(self.names, values), **kwargs)
            if not eligible(named):
                return self.original[grid](*values, **kwargs)
            key = base.fixed.signature(named)
            if key not in self.prepared:
                if self.frozen:
                    raise RuntimeError("New source specialization during timing")
                if kwargs.get("ir_override") is not None:
                    raise RuntimeError("IR overrides are forbidden")
                kernel = self.source.warmup(*values, grid=grid, **kwargs)
                check_kernel(kernel, self.expected)
                self.prepared[key] = kernel
            return self.source[grid](*values, **kwargs)

        return launch


def check_kernel(kernel, expected):
    if (
        fp8.runtime.code_sha(kernel.asm["ptx"]) != expected["code_sha256"]
        or kernel.metadata.shared != expected["shared_bytes"]
        or kernel.n_regs != expected["registers_per_thread"]
        or kernel.n_spills
    ):
        raise RuntimeError(
            "Source JIT executable/resources differ from offline proof;not launched"
        )


def make_modules(production, traces):
    captures = proof()
    modules = []
    for kind in VARIANTS:
        # Existing module cloning and FixedEntry grid construction, without stage-depth edits.
        with patch.object(
            stage, "fixed_traces", lambda ts, candidate=False: traces_for(ts, kind)
        ):
            module = stage.make_pair(production, traces)[0]
        module._dot_kind = kind
        entry = module._qsa_prefill_fp8_store_score_kernel
        if kind == "production":
            jit = entry.jit
        else:
            spec = importlib.util.spec_from_file_location(
                "qsa_dot_" + kind, captures[kind]["source"]
            )
            source_module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = source_module
            spec.loader.exec_module(source_module)
            jit = source_module._qsa_prefill_score_kernel
        entry.jit = SourceJIT(entry.jit, jit, entry.arg_names, captures[kind]["result"])
        modules.append(module)
    if any(m.persistent_topk is not modules[0].persistent_topk for m in modules):
        raise RuntimeError("TopK callable changed")
    return modules


def freeze_records(modules):
    result = {}
    fallback = {}
    for module in modules:
        kind = module._dot_kind
        rows = stage.FREEZE((module, module))["baseline"]
        for record in rows:
            named = dict(IS_FP8=True, NUM_HEADS=4, HEAD_DIM=128, **record["constants"])
            if record["route"] == "fp8" and eligible(named):
                entry = module._qsa_prefill_fp8_store_score_kernel
                key = tuple(record["chunk"])
                if key not in entry.jit.prepared:
                    raise RuntimeError("Source candidate did not pass prelaunch gate")
                check_kernel(entry.compiled[key][0], entry.jit.expected)
            else:
                key = tuple(record["chunk"])
                value = (
                    record["ptx_sha256"],
                    record["cubin_sha256"],
                    record["constants"],
                )
                if key in fallback and fallback[key] != value:
                    raise RuntimeError("Original small-tail fallback binary changed")
                fallback[key] = value
                record["original_binary_verified"] = True
        module._qsa_prefill_fp8_store_score_kernel.jit.frozen = True
        result[kind] = rows
    return result


def boundary_checks(production, helpers, torch, device):
    results = []
    for index in (1, 2):

        def pair(prod, traces):
            modules = make_modules(prod, traces)
            return (modules[0], modules[index])

        with patch.object(fp8, "make_pair", pair), patch.object(
            fp8, "freeze_records", freeze_records
        ):
            checks = fp8.boundary_checks(production, helpers, torch, device)
        results.append(dict(variant=VARIANTS[index], checks=checks))
    return [
        dict(
            scenario=scenario,
            status="passed",
            variants=[r["checks"][i] for r in results],
        )
        for i, scenario in enumerate(("mixed_edges", "single_tile", "zero_visible"))
    ]


def worker(options):
    from tools import benchmark_qsa_prefill_triton_stages as bench

    support, scoring, ab = bench.support, bench.scoring, bench.ab
    report = Path(options.report_dir)
    path = report / (options.worker_case + ".json")
    os.environ["QSA_PRODUCER_RUNTIME_REPORT"] = str(report)
    result = dict(
        status="running",
        acceptance_eligible=False,
        production_modified=False,
        algorithm="fp8_source_jit_r32_paired_issue",
        stages={},
        comparisons={},
        self_controls={},
        source=verify_source(),
        rounds=options.rounds,
    )
    try:
        item = next(
            v
            for v in fp8.manifest()["cases"]
            if v["case"]["name"] == options.worker_case
        )
        case, traces = item["case"], item["traces"]
        result.update(case=case, fixed_traces=traces)
        torch, _, production, _, helpers, device, info = bench.runtime(
            report / "qsa_indexer_vllm.py", path, options
        )
        result["runtime"] = info
        if base.compiler_identity(info) != fp8.manifest()["compiler_identity"]:
            raise RuntimeError("Compiler/torch/vLLM identity changed")
        evidence = report / "dot-boundaries.json"
        identity = dict(
            source=result["source"], compiler_identity=base.compiler_identity(info)
        )
        if options.worker_case == options.cases[0]:
            result["boundaries"] = boundary_checks(production, helpers, torch, device)
            support.write_json(
                evidence, dict(identity=identity, checks=result["boundaries"])
            )
        else:
            proof_data = json.loads(evidence.read_text())
            if (
                proof_data["identity"] != identity
                or len(proof_data["checks"]) != 3
                or any(v["status"] != "passed" for v in proof_data["checks"])
            ):
                raise RuntimeError("Boundary evidence mismatch")
            result["boundaries"] = dict(
                covered_by=options.cases[0], evidence_sha256=base.sha(evidence)
            )
        os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = str(case["budget_mb"])
        args = bench.workloads.make_inputs(case, helpers, torch)
        if support.fingerprint(args, torch) != item["inputs"]:
            raise RuntimeError("Captured input identity changed")
        result["inputs"] = item["inputs"]
        modules = make_modules(production, traces)
        chunks = ab.observe(modules[0], "persistent_topk", args)
        bench.verify_geometry(case, args, chunks)
        if ab.signatures(chunks) != [
            (t["offset"], t["rows"], t["width"]) for t in traces
        ]:
            raise RuntimeError("Captured chunk dispatch changed")
        common = [c.pop("common_logits") for c in chunks]
        bundles = [base.bind(module, chunks, common) for module in modules]
        for bundle in bundles:
            bundle["topk"] = bundles[0]["topk"]
        result.update(
            same_topk=True,
            same_buffers=True,
            common_before=[support.fingerprint((v,), torch) for v in common],
        )

        def validate_all():
            checks = {}
            for kind, module, bundle in zip(VARIANTS, modules, bundles):
                score_rows = scoring.validate_scores(
                    module._prefill_logits,
                    chunks,
                    args,
                    torch,
                    support.sample_rows,
                    device,
                )
                pipeline_rows = bench.validate_pipeline(
                    dict(baseline=bundle["pipeline"], candidate=bundle["pipeline"]),
                    args,
                    args,
                    chunks,
                    torch,
                    helpers,
                )
                checks[kind] = dict(score=score_rows, pipeline=pipeline_rows)
            bundles[0]["topk"]()
            checked = 0
            for scores, chunk in zip(common, chunks):
                lengths, selected = chunk["topk_args"][1:3]
                for row in support.sample_rows(chunk["rows"]):
                    bench.validate_reference_selection(
                        scores[row : row + 1],
                        lengths[row : row + 1],
                        selected[row : row + 1],
                        args[0],
                        torch,
                        helpers,
                    )
                    checked += 1
            checks["shared_topk"] = checked
            return checks

        result["validation_before"] = validate_all()
        device.synchronize()
        result["kernels_before"] = freeze_records(modules)
        result["artifacts"] = base.artifact_records(modules, report, case["name"])
        for kind, records in result["kernels_before"].items():
            for r in records:
                print(
                    "Resources:",
                    kind,
                    r["chunk"],
                    r["constants"],
                    "regs",
                    r["registers_per_thread"],
                    "shared",
                    r["shared_bytes"],
                    "spills",
                    r["spill_count"],
                    flush=True,
                )
        timer, metadata = scoring.make_timer(
            options.kernel_warmup_ms, options.kernel_rep_ms
        )
        result["timer"] = {k: v for k, v in metadata.items() if k != "includes_topk"}
        monitor = lambda label: support.guarded_telemetry(
            info["uuid"], label, result, path, support.gpu_guard_module()
        )

        def measure(label, left, right, stage_name):
            values, _, order = scoring.measure_scores(
                timer,
                bundles[left][stage_name],
                bundles[right][stage_name],
                options.rounds,
                lambda phase: monitor(label + ":" + phase),
                support.stats,
            )
            row = dict(
                values,
                measurement_order=order,
                includes_topk=stage_name != "score",
                status="passed",
                measurement_validated=False,
            )
            print(
                "Measured (pending final validation):",
                label,
                "left_us",
                row["baseline_us"]["median"],
                "right_us",
                row["candidate_us"]["median"],
                "ratio",
                row["speedup"],
                flush=True,
            )
            support.write_json(path, result)
            return row

        common_topk = measure("shared_topk", 0, 0, "topk")
        for left, right in PAIRS:
            label = VARIANTS[left] + "/" + VARIANTS[right]
            rows = result["comparisons"][label] = {}
            for stage_name in ("score", "pipeline"):
                rows[stage_name] = measure(
                    label + ":" + stage_name, left, right, stage_name
                )
            rows["topk"] = dict(common_topk, shared_callable_measurement=True)
        # Four A/A controls preserve the existing coordinator contract; all
        # three variants also participate in direct measured cross-comparisons.
        for index in (0, 2):
            for stage_name in ("score", "pipeline"):
                label = VARIANTS[index] + ":" + stage_name
                result["self_controls"][label] = measure(
                    label, index, index, stage_name
                )
        result["validation_after"] = validate_all()
        for rows in [*result["comparisons"].values(), result["self_controls"]]:
            for row in rows.values():
                row["measurement_validated"] = True
                row["validation_scope"] = (
                    "all three variants validated before/after entire timing block"
                )
        result["stages"] = result["comparisons"]["production/paired32"]
        result["kernels_after"] = freeze_records(modules)
        result["common_after"] = [support.fingerprint((v,), torch) for v in common]
        if (
            result["kernels_after"] != result["kernels_before"]
            or result["common_before"] != result["common_after"]
            or support.fingerprint(args, torch) != item["inputs"]
            or verify_source() != result["source"]
        ):
            raise RuntimeError(
                "Source/config/kernel/input/logits changed during timing"
            )
        monitor("after_all_validation")
        result["status"] = "passed"
        return 0
    except BaseException:
        result.update(status="failed", error=traceback.format_exc())
        return 1
    finally:
        support.write_json(path, result)
        for label, rows in result.get("comparisons", {}).items():
            print("\nDirect comparison:", label, flush=True)
            fp8.runtime.show(
                dict(case=result.get("case", {}), status=result["status"], stages=rows)
            )
        if result.get("error"):
            print(result["error"], flush=True)


@contextlib.contextmanager
def scoped():
    from tools import benchmark_qsa_prefill_triton_stages as bench

    with contextlib.ExitStack() as stack:
        for module, name, value in (
            (base, "manifest", fp8.manifest),
            (base, "verify_source", verify_source),
            (base, "__file__", str(Path(__file__).resolve())),
            (bench, "SOURCES", sorted(set(bench.SOURCES) | EXTRA)),
        ):
            stack.enter_context(patch.object(module, name, value))
        yield


def coordinate(options):
    code = base.coordinate(options)
    old = Path(options.report_dir) / "head-group-summary.json"
    if old.is_file():
        value = json.loads(old.read_text())
        value["probe"] = "fp8_r32_source_paired_issue"
        for row in value.get("cases", []):
            path = old.parent / (row["case"] + ".json")
            if path.is_file():
                row["comparisons"] = json.loads(path.read_text()).get("comparisons", {})
        (old.parent / "fp8-dot-runtime-summary.json").write_text(
            json.dumps(value, indent=2) + "\n"
        )
        old.unlink()
    return code


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--compile-report", default=os.environ.get("QSA_DOT_COMPILE_REPORT")
    )
    own, remaining = parser.parse_known_args()
    if not own.compile_report and not any(v in remaining for v in ("--help", "-h")):
        parser.error("--compile-report is required")
    if own.compile_report:
        os.environ["QSA_DOT_COMPILE_REPORT"] = str(Path(own.compile_report).resolve())
    sys.argv = [sys.argv[0], *remaining]
    for flag, values in (
        ("--cases", fp8.CASES),
        ("--kernel-warmup-ms", ("20",)),
        ("--kernel-rep-ms", ("20",)),
    ):
        if flag not in sys.argv:
            sys.argv.extend([flag, *values])
    with patch.object(stage, "manifest", fp8.manifest), patch.object(
        stage, "scoped", scoped
    ), patch.object(stage, "worker", worker), patch.object(
        stage, "coordinate", coordinate
    ), patch.object(
        stage, "__doc__", __doc__
    ):
        return stage.main()


if __name__ == "__main__":
    sys.exit(main())
