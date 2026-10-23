# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Six-case fixed-production head-group probe, never production dispatch."""

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_bf16_query_probe as shared  # noqa: E402
from tools import qsa_prefill_query_split_matched as fixed  # noqa: E402

MANIFEST = ROOT / "docs/qsa_prefill_head_group_configs.json"
MANIFEST_SHA = "439dfbac8043350ccecc2fae027db90bec58cf917f210606f11328dbc32c321d"
ENTRIES = (
    ("_qsa_prefill_score_kernel", "rectangular"),
    ("_qsa_prefill_compact_score_kernel", "compact"),
    ("_qsa_prefill_fp8_store_score_kernel", "fp8"),
)
sha, compiler_identity, bind = shared.sha, shared.compiler_identity, shared.bind


def manifest():
    from tools import qsa_prefill_observed_workloads as workloads
    from tools import qsa_prefill_triton_ab as ab

    if sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("Head-group manifest changed")
    data = json.loads(MANIFEST.read_text())
    if data["production_sha256"] != ab.INTEGRATED_PRODUCTION_SHA256:
        raise RuntimeError("Production baseline changed")
    expected = {c["name"]: c for c in workloads.cases("scored-prefill")}
    if len(data["cases"]) != 6 or len({v["case"]["name"] for v in data["cases"]}) != 6:
        raise RuntimeError("Expected six distinct captured cases")
    for item in data["cases"]:
        case = item["case"]
        if expected.get(case["name"]) != case:
            raise RuntimeError("Probe workload changed")
        shape = workloads.geometry(case)
        expected_chunks = [
            (sum(shape["expected_chunk_rows"][:i]), n, shape["logits_width"])
            for i, n in enumerate(shape["expected_chunk_rows"])
        ]
        if [
            (t["offset"], t["rows"], t["width"]) for t in item["traces"]
        ] != expected_chunks:
            raise RuntimeError("Probe chunk geometry changed")
        if case["heads"] != 4 or case["dim"] != 128:
            raise RuntimeError("Private head-pair probe requires H4/D128")
    return data


def verify_source():
    data = manifest()
    production = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
    if sha(production) != data["production_sha256"]:
        raise RuntimeError("Production source changed")
    return dict(
        production_sha256=sha(production),
        candidate_sha256=sha(ROOT / "tools/qsa_prefill_head_group_kernel.py"),
        manifest_sha256=sha(MANIFEST),
        driver_sha256=sha(Path(__file__)),
    )


def saved_record(name):
    return next(item for item in manifest()["cases"] if item["case"]["name"] == name)


def make_pair(production, traces):
    from tools.qsa_prefill_head_group_kernel import _head_group_score_kernel

    result = []
    for index in (0, 1):
        module = types.ModuleType(production.__name__ + f".head_group_fixed{index}")
        module.__dict__.update(vars(production))
        for name, route in ENTRIES:
            entry = fixed.FixedEntry(getattr(production, name), traces, route)
            if index == 1 and route != "compact":
                entry.jit = _head_group_score_kernel
            setattr(module, name, entry)
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
        raise RuntimeError("Head grouping changed TopK")
    return tuple(result)


def freeze_records(pair):
    from tools.qsa_prefill_kernel_resources import compiled_record

    records = []
    for module in pair:
        values = []
        for name, route in ENTRIES:
            entry = getattr(module, name)
            if set(entry.compiled) != set(entry.configs):
                raise RuntimeError("Fixed score preparation omitted a chunk")
            entry.frozen = True
            values.extend(
                dict(
                    chunk=list(key),
                    route=route,
                    constants=config,
                    **compiled_record(kernel),
                )
                for key, (kernel, config) in entry.compiled.items()
            )
        records.append(values)
    return dict(baseline=records[0], candidate=records[1])


def artifact_records(pair, report, case):
    artifacts = []
    for index, module in enumerate(pair):
        for name, route in ENTRIES:
            for key, (kernel, config) in getattr(module, name).compiled.items():
                files = {}
                for kind in ("ptx", "ttgir"):
                    content = kernel.asm[kind].encode()
                    path = report / f"{case}-side{index}-{key[1]}-{kind}.txt"
                    path.write_bytes(content)
                    files[kind] = dict(
                        path=path.name, sha256=hashlib.sha256(content).hexdigest()
                    )
                artifacts.append(
                    dict(
                        side=index,
                        chunk=list(key),
                        route=route,
                        constants=config,
                        files=files,
                    )
                )
    return artifacts


def boundary_checks(production, helpers, torch, device):
    from tools import qsa_prefill_score_benchmark as scoring

    checks = []
    visible = [
        0,
        1,
        2,
        3,
        4,
        5,
        511,
        512,
        513,
        514,
        515,
        1021,
        1022,
        1023,
        1024,
        1025,
        1025,
    ]
    choices = (
        (torch.bfloat16, 32, 8),
        (torch.bfloat16, 64, 4),
        (torch.float8_e4m3fn, 64, 4),
    )
    for dtype, tile, warps in choices:
        args = helpers.make_inputs(
            [5, 12],
            [1025, 1025],
            dtype=dtype,
            padded=True,
            query_base=97,
            page_size=196,
            visible=visible,
        )
        traces, chunks, buffers = [], [], []
        offset = 0
        for count in (4, 3, 10):
            width = 1088
            backing = torch.full(
                (count * width + 128,),
                -12345.0,
                dtype=torch.float32,
                device=args[0].device,
            )
            logits = backing[64:-64].view(count, width)
            config = dict(
                TILE_R=tile,
                BLOCK_N=64,
                K_TILES=16,
                num_warps=warps,
                num_stages=2,
                num_ctas=1,
            )
            if dtype == torch.float8_e4m3fn:
                config["QUERY_PARTS"] = 2
            traces.append(
                dict(
                    offset=offset,
                    rows=count,
                    width=width,
                    route="fp8" if dtype == torch.float8_e4m3fn else "rectangular",
                    constants=config,
                )
            )
            chunks.append(
                dict(
                    score_args=(*args[:5], logits, args[7], width, offset, count),
                    score_kwargs={},
                )
            )
            buffers.append(backing)
            offset += count
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
            for c, storage in zip(chunks, buffers):
                offset, count = c["score_args"][-2:]
                logits = c["score_args"][5]
                mask = (
                    torch.arange(width, device=args[0].device)[None, :]
                    < args[4][offset : offset + count, None]
                )
                if (
                    not bool((storage[:64] == -12345.0).all())
                    or not bool((storage[-64:] == -12345.0).all())
                    or not bool((logits[~mask] == -12345.0).all())
                ):
                    raise RuntimeError(
                        "Head-group boundary wrote masked output or canaries"
                    )
        freeze_records(pair)
        checks.append(
            dict(dtype=str(dtype), tile=tile, warps=warps, rows=17, status="passed")
        )
    return checks


def show(result):
    from tools.qsa_prefill_report import fmt, table

    print(
        "\nHead-group probe:",
        result.get("case", {}).get("name"),
        result.get("status"),
        flush=True,
    )
    for title, key in (
        ("Production / candidate [us]", "stages"),
        ("Identical-callable A/A controls", "self_controls"),
    ):
        values = result.get(key, {})
        if values:
            print(title, flush=True)
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
                        fmt(v["baseline_us"]["median"]),
                        fmt(v["candidate_us"]["median"]),
                        fmt(v["speedup"], 4),
                        fmt(v["paired_ratios"]["min"], 4),
                        fmt(v["paired_ratios"]["max"], 4),
                    )
                    for k, v in values.items()
                    if v.get("measurement_validated")
                ],
            )
    if result.get("error"):
        print(result["error"], flush=True)


def worker(options):
    from tools import benchmark_qsa_prefill_triton_stages as bench

    support, scoring, ab = bench.support, bench.scoring, bench.ab
    report = Path(options.report_dir)
    path = report / (options.worker_case + ".json")
    result = dict(
        status="running",
        acceptance_eligible=False,
        production_modified=False,
        stages={},
        algorithm="two_head_pairs_full_query",
        config_scope="same tiles/warps/stages; QUERY_PARTS is a baseline tag ignored by the candidate",
        source=verify_source(),
    )
    try:
        item = saved_record(options.worker_case)
        case, traces = item["case"], item["traces"]
        result.update(case=case, fixed_traces=traces)
        torch, _, production, _, helpers, device, info = bench.runtime(
            report / "qsa_indexer_vllm.py", path, options
        )
        result["runtime"] = info
        if compiler_identity(info) != manifest()["compiler_identity"]:
            raise RuntimeError("Captured compiler/torch/vLLM identity changed")
        evidence = report / "head-boundaries.json"
        identity = dict(
            source=result["source"], compiler_identity=compiler_identity(info)
        )
        if options.worker_case == options.cases[0]:
            result["boundaries"] = boundary_checks(production, helpers, torch, device)
            support.write_json(
                evidence, dict(identity=identity, checks=result["boundaries"])
            )
        else:
            proof = json.loads(evidence.read_text())
            if (
                proof["identity"] != identity
                or len(proof["checks"]) != 3
                or any(v["status"] != "passed" for v in proof["checks"])
            ):
                raise RuntimeError("Boundary evidence mismatch")
            result["boundaries"] = dict(
                covered_by=options.cases[0], evidence_sha256=sha(evidence)
            )
        os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = str(case["budget_mb"])
        args = bench.workloads.make_inputs(case, helpers, torch)
        if support.fingerprint(args, torch) != item["inputs"]:
            raise RuntimeError("Captured input identity changed")
        result["inputs"] = item["inputs"]
        pair = make_pair(production, traces)
        chunks = ab.observe(pair[0], "persistent_topk", args)
        bench.verify_geometry(case, args, chunks)
        if ab.signatures(chunks) != [
            (t["offset"], t["rows"], t["width"]) for t in traces
        ]:
            raise RuntimeError("Captured chunk dispatch changed")
        common = [c.pop("common_logits") for c in chunks]
        bundles = [bind(module, chunks, common) for module in pair]
        bundles[1]["topk"] = bundles[0]["topk"]
        result.update(
            same_topk=pair[0].persistent_topk is pair[1].persistent_topk,
            same_buffers=True,
            common_before=[support.fingerprint((v,), torch) for v in common],
        )

        def validate(index, stage):
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
                return checked
            fn = bundles[index]["pipeline"]
            return bench.validate_pipeline(
                dict(baseline=fn, candidate=fn), args, args, chunks, torch, helpers
            )

        for bundle in bundles:
            for fn in bundle.values():
                fn()
        device.synchronize()
        result["kernels_before"] = freeze_records(pair)
        result["artifacts"] = artifact_records(pair, report, case["name"])
        for side, records in result["kernels_before"].items():
            for v in records:
                print(
                    "Resources:",
                    side,
                    v["chunk"],
                    v["constants"],
                    "registers",
                    v["registers_per_thread"],
                    "shared",
                    v["shared_bytes"],
                    "spills",
                    v["spill_count"],
                    flush=True,
                )
        timer, metadata = scoring.make_timer(
            options.kernel_warmup_ms, options.kernel_rep_ms
        )
        result["timer"] = {k: v for k, v in metadata.items() if k != "includes_topk"}
        monitor = lambda label: support.guarded_telemetry(
            info["uuid"], label, result, path, support.gpu_guard_module()
        )
        result["self_controls"] = {}
        jobs = [(stage, 0, 1) for stage in ("score", "topk", "pipeline")] + [
            (f"{side}:{stage}", i, i)
            for i, side in enumerate(("production", "head_group"))
            for stage in ("score", "pipeline")
        ]
        for label, left, right in jobs:
            stage = label.split(":")[-1]
            bucket = result["self_controls"] if ":" in label else result["stages"]
            row = bucket[label] = dict(
                validation_before=[validate(i, stage) for i in (left, right)],
                includes_topk=stage != "score",
            )
            values, _, order = scoring.measure_scores(
                timer,
                bundles[left][stage],
                bundles[right][stage],
                options.rounds,
                lambda phase: monitor(label + ":" + phase),
                support.stats,
            )
            row.update(
                values,
                measurement_order=order,
                validation_after=[validate(i, stage) for i in (left, right)],
                status="passed",
                measurement_validated=True,
            )
            support.write_json(path, result)
        result["kernels_after"] = freeze_records(pair)
        result["common_after"] = [support.fingerprint((v,), torch) for v in common]
        if (
            result["kernels_after"] != result["kernels_before"]
            or result["common_before"] != result["common_after"]
            or support.fingerprint(args, torch) != item["inputs"]
            or verify_source() != result["source"]
        ):
            raise RuntimeError(
                "Source/config/kernel/input/common-logits changed during probe"
            )
        monitor("after_all_validation")
        with (report / (case["name"] + ".tsv")).open("w") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(("stage", "production_us", "head_group_us", "speedup"))
            for label, row in result["stages"].items():
                writer.writerow(
                    (
                        label,
                        row["baseline_us"]["median"],
                        row["candidate_us"]["median"],
                        row["speedup"],
                    )
                )
        result.update(status="passed")
        return 0
    except BaseException:
        result.update(status="failed", error=traceback.format_exc())
        return 1
    finally:
        support.write_json(path, result)
        show(result)


def coordinate(options):
    from tools import benchmark_qsa_prefill_triton_stages as bench
    from tools import qsa_prefill_flagtree_ab as profile
    from tools import qsa_prefill_profiling as profiling

    report, previous = Path(options.report_dir), Path(options.baseline_report)
    summary = dict(
        status="preparing",
        acceptance_eligible=False,
        production_modified=False,
        source=verify_source(),
        options=vars(options),
        git_head=bench.support.command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        git_status=bench.support.command(
            ["git", "-C", str(ROOT), "status", "--porcelain"]
        ),
        cases=[],
    )
    try:
        old = json.loads((previous / "stage-summary.json").read_text())
        if (
            old["status"] != "passed"
            or old["manifest"]["git_head"]["stdout"].strip()
            != "55a13f673a5b54a0d430a124f0e04ec5e2c155cf"
        ):
            raise RuntimeError("Requires successful e1c188 production report")
        tools = json.loads((previous / "flagtree134-tools.json").read_text())
        assembler = Path(tools["toolchain"]["ptxas"]["path"])
        if not assembler.is_file() or sha(assembler) != profile.PTXAS_SHA256:
            raise RuntimeError("Previous verified ptxas unavailable")
        os.environ["TRITON_PTXAS_PATH"] = str(assembler)
        sources = json.loads((previous / "sources.json").read_text())
        reference = sources["pinned_vllm/qsa_indexer.py"]["text"]
        if hashlib.sha256(reference.encode()).hexdigest() != bench.ab.REFERENCE_SHA256:
            raise RuntimeError("Native reference identity changed")
        (report / "qsa_indexer_vllm.py").write_text(reference)
        bench.support.write_json(report / "flagtree134-tools.json", tools)
        for name in options.cases:
            saved = saved_record(name)
            captured = json.loads((previous / (name + ".json")).read_text())
            if (
                captured["status"] != "passed"
                or captured["case"] != saved["case"]
                or captured["inputs"] != saved["inputs"]
                or compiler_identity(captured["runtime"])
                != manifest()["compiler_identity"]
            ):
                raise RuntimeError("Captured fixture changed: " + name)
        if not profile.select_gpu(summary, options):
            raise RuntimeError("No exclusive idle H800 available")
        names = set(bench.SOURCES) | {
            "tools/qsa_prefill_head_group_probe.py",
            "tools/qsa_prefill_head_group_kernel.py",
            "tools/test_qsa_prefill_head_group_probe.py",
            "tools/run_qsa_prefill_head_group_h800.sh",
            "docs/qsa_prefill_head_group_configs.json",
            "docs/qsa_prefill_head_group_probe.md",
        }
        bench.support.write_json(
            report / "sources.json",
            {
                n: dict(text=(ROOT / n).read_text(), sha256=sha(ROOT / n))
                for n in sorted(names)
            },
        )
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
            path = report / (name + ".json")
            result = json.loads(path.read_text()) if path.is_file() else {}
            summary["cases"].append(
                dict(
                    case=name,
                    exit_code=code,
                    status=result.get("status"),
                    stages=result.get("stages", {}),
                )
            )
            if code or result.get("status") != "passed":
                raise RuntimeError("Head-group worker failed: " + name)
            if (
                set(result.get("stages", {})) != {"score", "topk", "pipeline"}
                or not all(
                    v.get("measurement_validated") for v in result["stages"].values()
                )
                or len(result.get("self_controls", {})) != 4
            ):
                raise RuntimeError("Incomplete head-group measurements: " + name)
        summary["aggregate"] = {
            stage: statistics.mean(
                c["stages"][stage]["speedup"] for c in summary["cases"]
            )
            for stage in ("score", "topk", "pipeline")
        }
        print(
            "Incremental production / head-group arithmetic means:",
            summary["aggregate"],
            flush=True,
        )
        summary["status"] = "passed"
        return 0
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        print(summary["error"], flush=True)
        return 1
    finally:
        bench.support.write_json(report / "head-group-summary.json", summary)


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
    names = [item["case"]["name"] for item in manifest()["cases"]]
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
    ):
        parser.error(
            "Positive timing/timeout/interval and nonnegative cooldown required"
        )
    if len(set(options.cases)) != len(options.cases) or (
        options.worker_case and options.worker_case not in options.cases
    ):
        parser.error("Distinct cases and selected worker case required")
    options.report_dir = str(Path(options.report_dir).resolve())
    options.baseline_report = str(Path(options.baseline_report).resolve())
    options.compiler, options.candidate = "flagtree", "production"
    from tools.qsa_prefill_flagtree_ab import configure

    configure(options.flagtree)
    return worker(options) if options.worker_case else coordinate(options)


if __name__ == "__main__":
    sys.exit(main())
