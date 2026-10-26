# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Compile-gated three-stage indirect loads while retaining async K copies."""

import contextlib
import hashlib
import json
import os
import re
import statistics
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_bf16_dot_schedule_probe as harness  # noqa: E402

stage, base = harness.stage, harness.base
manifest = stage.manifest
MAKE_PAIR, CORE_SCOPED = harness.make_pair, harness.scoped
KERNEL = ROOT / "tools/qsa_prefill_async_pipeline_kernel.py"
KERNEL_SHA = "db9896ef6b23d99f718885693687a274d95d5808088fb90b3c59081178251272"
EXTRA_SOURCES = harness.EXTRA_SOURCES | {
    "tools/qsa_prefill_async_pipeline_probe.py",
    "tools/qsa_prefill_async_pipeline_kernel.py",
    "tools/test_qsa_prefill_async_pipeline_probe.py",
    "tools/run_qsa_prefill_async_pipeline_h800.sh",
    "docs/qsa_prefill_async_pipeline_probe.md",
}
DEFAULT_CASES = ("perf_n1_q8192_p102400_bfloat16", "perf_n4_q4096_p102400_bfloat16")


def verify_source():
    source = stage.verify_source()
    if base.sha(KERNEL) != KERNEL_SHA:
        raise RuntimeError("Async pipeline private source changed")
    return dict(
        source,
        candidate_sha256=KERNEL_SHA,
        pipeline_sources={n: base.sha(ROOT / n) for n in sorted(EXTRA_SOURCES)},
    )


def eligible(named):
    return (
        named.get("IS_FP8") is False
        and named.get("NUM_HEADS") == 4
        and named.get("HEAD_DIM") == 128
        and named.get("TILE_R") in (32, 64)
        and named.get("BLOCK_N") == 64
    )


def make_pair(production, traces, kernel=None):
    if kernel is None:
        from tools.qsa_prefill_async_pipeline_kernel import _async_pipeline_score_kernel

        kernel = _async_pipeline_score_kernel
    return MAKE_PAIR(production, traces, kernel)


def prefetch(ir, ptx):
    lines = ir.splitlines()
    loops = [
        (i, len(line) - len(line.lstrip()))
        for i, line in enumerate(lines)
        if re.search(r"\bscf\.for\b", line)
    ]
    result = dict(
        main_loops=len(loops),
        copies_before_first_dot=None,
        loop_copy_sites=None,
        first_dot_line=None,
        mma_sites=harness.schedule(ptx)["mma_sites"],
        ptx_async_copy_sites=len(
            re.findall(r"\bcp\.async\.(?:ca|cg)\.shared\.global\b", ptx)
        ),
    )
    if len(loops) != 1:
        return result
    start, indent = loops[0]
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if lines[i].startswith(" " * indent + "}")
        ),
        None,
    )
    if end is None:
        return result
    copies = [
        i
        for i in range(start + 1, end)
        if (
            "ttg.async_copy_global_to_local" in lines[i]
            or re.search(r"\btt.load\b", lines[i])
        )
        and re.search(r"tensor<64x128x!tt.ptr<bf16>", lines[i])
    ]
    async_keys = [i for i in copies if "ttg.async_copy_global_to_local" in lines[i]]
    metadata_copies = [
        i
        for i in range(start + 1, end)
        if "ttg.async_copy_global_to_local" in lines[i]
        and re.search(r"!tt.ptr<i32>", lines[i])
    ]
    slots = 0
    for line in lines:
        if "ttg.local_alloc" not in line:
            continue
        m = re.search(r"memdesc<((?:\d+x)+)bf16", line)
        if m:
            dims = [int(v) for v in m[1].rstrip("x").split("x")]
            if dims[-2:] == [64, 128]:
                count = 1
                for v in dims[:-2]:
                    count *= v
                slots += count
    dots = [
        i
        for i in range(start + 1, end)
        if re.search(r"\bttng\.warp_group_dot\b", lines[i])
    ]
    result.update(
        loop_start_line=start + 1,
        loop_end_line=end + 1,
        loop_copy_sites=len(copies),
        key_shared_slots=slots,
        async_key_copy_sites=len(async_keys),
        async_key_copy_lines=[i + 1 for i in async_keys],
        register_key_operand=any(
            re.search(r"\bttng\.warp_group_dot\b.*:\s*tensor<64x128xbf16", lines[i])
            for i in dots
        ),
        metadata_copy_sites=len(metadata_copies),
        key_read_instructions=[lines[i].strip() for i in copies],
        copy_lines=[i + 1 for i in copies],
    )
    if dots:
        result.update(
            first_dot_line=dots[0] + 1,
            copies_before_first_dot=sum(i < dots[0] for i in copies),
            async_keys_before_first_dot=sum(i < dots[0] for i in async_keys),
        )
    return result


def gate_reason(before, after):
    if before["spill_count"] != 0 or after["spill_count"] != 0:
        return "nonzero or unknown spills"
    if any(
        v[k] is None
        for v in (before, after)
        for k in ("shared_bytes", "registers_per_thread")
    ):
        return "unknown resources"
    if after["shared_bytes"] > before["shared_bytes"] + 1024:
        return "shared footprint exceeds fixed 1KiB metadata allowance"
    a, b = before["prefetch"], after["prefetch"]
    if any(
        v["main_loops"] != 1 or v["copies_before_first_dot"] is None for v in (a, b)
    ):
        return "unidentified loop/copy/dot schedule"
    if not a["mma_sites"] or b["mma_sites"] != a["mma_sites"]:
        return "matrix work changed or missing"
    if b["key_shared_slots"] != 2:
        return "K double buffering changed or unidentified"
    if not b["ptx_async_copy_sites"]:
        return "hardware async K copy path absent"
    if b["async_keys_before_first_dot"] <= a["async_keys_before_first_dot"]:
        return "no additional async K copy before first matrix dot"
    return None


def freeze_records(pair):
    records = stage.FREEZE(pair)
    original = {tuple(v["chunk"]): v for v in records["baseline"]}
    decisions = pair[1]._qsa_prefill_score_kernel.jit.decisions
    for side, module in zip(("baseline", "candidate"), pair):
        entries = {}
        for name, _ in base.ENTRIES:
            entries.update(getattr(module, name).compiled)
        for r in records[side]:
            asm = entries[tuple(r["chunk"])][0].asm
            r["prefetch"] = prefetch(asm["ttgir"], asm["ptx"])
    rejections = []
    for r in records["candidate"]:
        key = tuple(r["chunk"])
        old = original[key]
        if r["constants"] != old["constants"] or r["route"] != old["route"]:
            raise RuntimeError("Async pipeline changed launch configs/route")
        active = decisions.get(key, False) if r["route"] == "rectangular" else False
        r["pipeline_active"] = active
        if not active:
            if any(r[k] != old[k] for k in ("ptx_sha256", "cubin_sha256")):
                raise RuntimeError("Async pipeline fallback changed original binary")
            r["original_binary_verified"] = True
            continue
        reason = r["pipeline_gate_reason"] = gate_reason(old, r)
        print(
            "Async pipeline compile:",
            key,
            "early async K copies",
            old["prefetch"].get("async_keys_before_first_dot"),
            "->",
            r["prefetch"].get("async_keys_before_first_dot"),
            "PTX cp.async sites",
            r["prefetch"]["ptx_async_copy_sites"],
            "K shared slots",
            old["prefetch"].get("key_shared_slots"),
            "->",
            r["prefetch"].get("key_shared_slots"),
            "registers",
            old["registers_per_thread"],
            "->",
            r["registers_per_thread"],
            "shared",
            old["shared_bytes"],
            "->",
            r["shared_bytes"],
            "spills",
            r["spill_count"],
            "gate",
            reason or "passed",
            flush=True,
        )
        if reason and not harness.IN_BOUNDARY:
            rejections.append(dict(chunk=r["chunk"], reason=reason))
    if harness.RESOURCE_DIR is not None:
        harness.RESOURCE_INDEX += 1
        prefix = f"compile-{harness.RESOURCE_PREFIX}-{os.getpid()}-{harness.RESOURCE_INDEX:03d}"
        artifacts = base.artifact_records(pair, harness.RESOURCE_DIR, prefix)
        (harness.RESOURCE_DIR / (prefix + ".json")).write_text(
            json.dumps(
                dict(
                    records=records, artifacts=artifacts, boundary=harness.IN_BOUNDARY
                ),
                indent=2,
            )
            + "\n"
        )
    if not harness.IN_BOUNDARY and not any(
        r["pipeline_active"] for r in records["candidate"]
    ):
        raise RuntimeError("Expected rectangular BF16 route was not exercised")
    if rejections:
        harness.LAST_REJECTION = rejections
        raise RuntimeError(
            "ASYNC_PIPELINE_NOT_PROMISING: no timing; artifacts retained"
        )
    return records


def show(result):
    from tools.qsa_prefill_report import fmt, table

    print(
        "\nAsync K pipeline:",
        result.get("case", {}).get("name"),
        result.get("status"),
        flush=True,
    )
    if result.get("rejections"):
        print("编译探测未通过，跳过计时：", result["rejections"], flush=True)
    for label, rows in (
        ("kernel-mode production / candidate", result.get("stages", {})),
        ("identical-callable A/A", result.get("self_controls", {})),
    ):
        if rows:
            print(label, flush=True)
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
                        *(
                            fmt(v[f]["median"], 3)
                            for f in ("baseline_us", "candidate_us")
                        ),
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
    harness.LAST_REJECTION = None
    with patch.object(harness, "RESOURCE_DIR", Path(options.report_dir)), patch.object(
        harness, "RESOURCE_PREFIX", options.worker_case
    ):
        code = base.worker(options)
    path = Path(options.report_dir) / (options.worker_case + ".json")
    if path.is_file():
        result = json.loads(path.read_text())
        result.update(
            algorithm="async_k_pipeline_stage3",
            rounds=options.rounds,
            config_scope="original production body; only loop-scoped stages3",
        )
        if code and harness.LAST_REJECTION and not result.get("stages"):
            result.update(
                status="not_promising",
                rejections=harness.LAST_REJECTION,
                compile_rejection_trace=result.pop("error", None),
            )
            code = 0
        result["self_controls"] = {
            k.replace("head_group:", "async_pipeline:"): v
            for k, v in result.get("self_controls", {}).items()
        }
        path.write_text(json.dumps(result, indent=2) + "\n")
        show(result)
    return code


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
        baseline_commit = manifest().get(
            "baseline_code_commit", "55a13f673a5b54a0d430a124f0e04ec5e2c155cf"
        )
        if (
            old["status"] != "passed"
            or old["manifest"]["git_head"]["stdout"].strip() != baseline_commit
        ):
            raise RuntimeError(
                "Requires successful captured production report at the pinned code commit"
            )
        tools = json.loads((previous / "flagtree134-tools.json").read_text())
        assembler = Path(tools["toolchain"]["ptxas"]["path"])
        if not assembler.is_file() or base.sha(assembler) != profile.PTXAS_SHA256:
            raise RuntimeError("Previous verified ptxas unavailable")
        os.environ["TRITON_PTXAS_PATH"] = str(assembler)
        sources = json.loads((previous / "sources.json").read_text())
        reference = sources["pinned_vllm/qsa_indexer.py"]["text"]
        if hashlib.sha256(reference.encode()).hexdigest() != bench.ab.REFERENCE_SHA256:
            raise RuntimeError("Native reference identity changed")
        (report / "qsa_indexer_vllm.py").write_text(reference)
        bench.support.write_json(report / "flagtree134-tools.json", tools)
        for name in options.cases:
            saved = base.saved_record(name)
            captured = json.loads((previous / (name + ".json")).read_text())
            if (
                captured["status"] != "passed"
                or captured["case"] != saved["case"]
                or captured["inputs"] != saved["inputs"]
                or base.compiler_identity(captured["runtime"])
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
                n: dict(text=(ROOT / n).read_text(), sha256=base.sha(ROOT / n))
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
            if (
                code == 0
                and result.get("status") == "not_promising"
                and not result.get("stages")
                and result.get("rejections")
            ):
                summary["cases"][-1]["rejections"] = result["rejections"]
                print(
                    "Negative compile probe complete; continue with other configuration",
                    flush=True,
                )
                continue
            if code or result.get("status") != "passed":
                raise RuntimeError("Async-pipeline worker failed: " + name)
            if (
                set(result.get("stages", {})) != {"score", "topk", "pipeline"}
                or not all(
                    v.get("measurement_validated") for v in result["stages"].values()
                )
                or len(result.get("self_controls", {})) != 4
            ):
                raise RuntimeError("Incomplete async-pipeline measurements: " + name)
        measured = [c for c in summary["cases"] if c["status"] == "passed"]
        summary["aggregate"] = (
            {
                kind: statistics.mean(c["stages"][kind]["speedup"] for c in measured)
                for kind in ("score", "topk", "pipeline")
            }
            if measured
            else {}
        )
        summary["measured_cases"] = len(measured)
        print(
            "Incremental production / candidate arithmetic means:",
            summary["aggregate"],
            flush=True,
        )
        summary["status"] = (
            "passed"
            if len(measured) == len(summary["cases"])
            else "completed_with_rejections" if measured else "not_promising"
        )
        return 0
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        print(summary["error"], flush=True)
        return 1
    finally:
        bench.support.write_json(report / "async-pipeline-summary.json", summary)


@contextlib.contextmanager
def scoped():
    with contextlib.ExitStack() as stack:
        for name, value in (
            ("eligible", eligible),
            ("make_pair", make_pair),
            ("freeze_records", freeze_records),
            ("verify_source", verify_source),
            ("EXTRA_SOURCES", EXTRA_SOURCES),
            ("__file__", str(Path(__file__).resolve())),
        ):
            stack.enter_context(patch.object(harness, name, value))
        stack.enter_context(CORE_SCOPED())
        yield


def main():
    for flag, values in (
        ("--cases", DEFAULT_CASES),
        ("--kernel-warmup-ms", ("20",)),
        ("--kernel-rep-ms", ("20",)),
    ):
        if flag not in sys.argv:
            sys.argv.extend([flag, *values])
    with patch.object(stage, "__doc__", __doc__), patch.object(
        stage, "scoped", scoped
    ), patch.object(stage, "worker", worker), patch.object(
        stage, "coordinate", coordinate
    ):
        return stage.main()


if __name__ == "__main__":
    sys.exit(main())
