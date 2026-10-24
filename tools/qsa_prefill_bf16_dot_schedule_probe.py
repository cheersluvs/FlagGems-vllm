# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Compile-gated grouped BF16 dot scheduling; exact production configs and timers."""

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
from tools import qsa_prefill_bf16_stage_depth_probe as stage  # noqa: E402

base = stage.base
FIXED_TRACES = stage.fixed_traces
KERNEL = ROOT / "tools/qsa_prefill_bf16_dot_schedule_kernel.py"
KERNEL_SHA = "e140f8dbb79a996427fdd893ef1e1c52a0cf48add6f83c72bee099b4182d51a3"
EXTRA_SOURCES = stage.EXTRA_SOURCES | {
    "tools/qsa_prefill_bf16_dot_schedule_probe.py",
    "tools/qsa_prefill_bf16_dot_schedule_kernel.py",
    "tools/test_qsa_prefill_bf16_dot_schedule_probe.py",
    "tools/run_qsa_prefill_bf16_dot_schedule_h800.sh",
    "docs/qsa_prefill_bf16_dot_schedule_probe.md",
}
DEFAULT_CASES = (
    "perf_n4_q4096_p102400_bfloat16",
    "observed_chunk_boundary_bfloat16",
)
manifest = stage.manifest
RESOURCE_DIR = None
RESOURCE_PREFIX = "unbound"
RESOURCE_INDEX = 0
IN_BOUNDARY = False
LAST_REJECTION = None


def verify_source():
    result = stage.verify_source()
    if base.sha(KERNEL) != KERNEL_SHA:
        raise RuntimeError("Private dot-schedule source changed")
    return dict(
        result,
        candidate_sha256=KERNEL_SHA,
        schedule_sources={n: base.sha(ROOT / n) for n in sorted(EXTRA_SOURCES)},
    )


def eligible(named):
    return (
        named.get("IS_FP8") is False
        and named.get("NUM_HEADS") == 4
        and named.get("HEAD_DIM") == 128
        and named.get("TILE_R") == 64
        and named.get("HEAD_GROUP") is True
    )


class RoutedJIT:
    def __init__(self, original, candidate, arg_names):
        self.original, self.candidate, self.arg_names = original, candidate, arg_names
        self.decisions = {}

    def __getitem__(self, grid):
        def launch(*values, **kwargs):
            named = dict(zip(self.arg_names, values))
            named.update(kwargs)
            active = eligible(named)
            self.decisions[base.fixed.signature(named)] = active
            return (self.candidate if active else self.original)[grid](
                *values, **kwargs
            )

        return launch


def make_pair(production, traces, kernel=None):
    if kernel is None:
        from tools.qsa_prefill_bf16_dot_schedule_kernel import (
            _dot_schedule_score_kernel,
        )

        kernel = _dot_schedule_score_kernel
    with patch.object(
        stage,
        "fixed_traces",
        lambda traces, candidate=False: FIXED_TRACES(traces, False),
    ):
        pair = stage.make_pair(production, traces)
    entry = pair[1]._qsa_prefill_score_kernel
    entry.jit = RoutedJIT(entry.jit, kernel, entry.arg_names)
    return pair


def schedule(ptx):
    ptx = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)
    sequence = []
    mma_count, first_wait = 0, None
    for match in re.finditer(
        r"\bwgmma\.(mma_async|commit_group|wait_group)\b[^;]*;", ptx
    ):
        kind = match[1]
        if kind == "mma_async":
            mma_count += 1
        if kind == "wait_group" and first_wait is None:
            first_wait = mma_count
        sequence.append(
            dict(
                kind=kind,
                mma_issued_so_far=mma_count,
                instruction=match[0].split("{")[0].strip(),
            )
        )
    return dict(
        mma_sites=mma_count,
        wait_sites=sum(v["kind"] == "wait_group" for v in sequence),
        mma_before_first_wait=first_wait,
        sequence=sequence,
    )


def gate_reason(before, after):
    if after["spill_count"] != 0 or before["spill_count"] != 0:
        return "nonzero or unknown spill count"
    if any(
        v[k] is None
        for v in (before, after)
        for k in ("registers_per_thread", "shared_bytes")
    ):
        return "unknown register/shared resource footprint"
    a, b = before["schedule"], after["schedule"]
    if (
        a["mma_before_first_wait"] is None
        or b["mma_before_first_wait"] is None
        or not a["wait_sites"]
    ):
        return "missing WGMMA result-wait sequence"
    if b["mma_sites"] != a["mma_sites"] or b["wait_sites"] > a["wait_sites"]:
        return "WGMMA work changed or more wait sites introduced"
    if b["mma_before_first_wait"] <= a["mma_before_first_wait"]:
        return "no additional independent MMA issued before first result wait"
    return None


def freeze_records(pair):
    global RESOURCE_INDEX, LAST_REJECTION
    records = stage.FREEZE(pair)
    originals = {tuple(v["chunk"]): v for v in records["baseline"]}
    jit = pair[1]._qsa_prefill_score_kernel.jit
    decisions = jit.decisions
    for side, module in zip(("baseline", "candidate"), pair):
        entries = {}
        for name, _ in base.ENTRIES:
            entries.update(getattr(module, name).compiled)
        for record in records[side]:
            record["schedule"] = schedule(entries[tuple(record["chunk"])][0].asm["ptx"])
    rejections = []
    for r in records["candidate"]:
        key = tuple(r["chunk"])
        old = originals[key]
        active = decisions.get(key, False) if r["route"] == "rectangular" else False
        if r["constants"] != old["constants"] or r["route"] != old["route"]:
            raise RuntimeError("Dot scheduling changed production config/route")
        r["schedule_active"] = active
        if not active:
            if any(r[k] != old[k] for k in ("ptx_sha256", "cubin_sha256")):
                raise RuntimeError("Dot-schedule fallback changed original binary")
            r["original_binary_verified"] = True
            continue
        r["schedule_gate_reason"] = gate_reason(old, r)
        print(
            "Dot schedule compile:",
            key,
            "MMA-before-first-wait",
            old["schedule"]["mma_before_first_wait"],
            "->",
            r["schedule"]["mma_before_first_wait"],
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
            r["schedule_gate_reason"] or "passed",
            flush=True,
        )
        if r["schedule_gate_reason"] and not IN_BOUNDARY:
            rejections.append(dict(chunk=r["chunk"], reason=r["schedule_gate_reason"]))
    if RESOURCE_DIR is not None:
        RESOURCE_INDEX += 1
        prefix = f"compile-{RESOURCE_PREFIX}-{os.getpid()}-{RESOURCE_INDEX:03d}"
        artifacts = base.artifact_records(pair, RESOURCE_DIR, prefix)
        (RESOURCE_DIR / (prefix + ".json")).write_text(
            json.dumps(
                dict(records=records, artifacts=artifacts, boundary=IN_BOUNDARY),
                indent=2,
            )
            + "\n"
        )
    if not IN_BOUNDARY and not any(v["schedule_active"] for v in records["candidate"]):
        raise RuntimeError("Expected grouped R64 route was not exercised")
    if rejections:
        LAST_REJECTION = rejections
        raise RuntimeError("DOT_SCHEDULE_NOT_PROMISING: no timing; artifacts retained")
    return records


def boundary_checks(production, helpers, torch, device):
    with patch.object(sys.modules[__name__], "IN_BOUNDARY", True):
        return stage.BOUNDARIES(production, helpers, torch, device)


def show(result):
    from tools.qsa_prefill_report import fmt, table

    print(
        "\nBF16 dot scheduling:",
        result.get("case", {}).get("name"),
        result.get("status"),
        flush=True,
    )
    if result.get("rejections"):
        print(
            "编译探测未发现可接受的重叠空间，跳过计时：",
            result["rejections"],
            flush=True,
        )
    for title, rows in (
        ("kernel-mode production / candidate", result.get("stages", {})),
        ("identical-callable A/A", result.get("self_controls", {})),
    ):
        if rows:
            print(title, flush=True)
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
    global LAST_REJECTION
    LAST_REJECTION = None
    with patch.object(
        sys.modules[__name__], "RESOURCE_DIR", Path(options.report_dir)
    ), patch.object(sys.modules[__name__], "RESOURCE_PREFIX", options.worker_case):
        code = base.worker(options)
    path = Path(options.report_dir) / (options.worker_case + ".json")
    if path.is_file():
        result = json.loads(path.read_text())
        result.update(
            algorithm="bf16_grouped_dot_issue_before_reduce",
            rounds=options.rounds,
            config_scope="same production configs; issue high dot before low reduction",
        )
        if code and LAST_REJECTION and not result.get("stages"):
            result.update(
                status="not_promising",
                rejections=LAST_REJECTION,
                compile_rejection_trace=result.pop("error", None),
            )
            code = 0
        result["self_controls"] = {
            k.replace("head_group:", "schedule:"): v
            for k, v in result.get("self_controls", {}).items()
        }
        path.write_text(json.dumps(result, indent=2) + "\n")
        show(result)
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


# Keep the original coordinator's input/compiler/PTXAS/GPU/source checks and
# timing completeness gate. Only a fully captured negative compile probe can
# finish early without timings; implementation below is frozen at source level.


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
                summary.update(
                    status="not_promising",
                    rejections=result["rejections"],
                    timing_skipped=True,
                )
                print(
                    "Negative compile probe complete; no further cases/timing needed",
                    flush=True,
                )
                return 0
            if code or result.get("status") != "passed":
                raise RuntimeError("Dot-schedule worker failed: " + name)
            if (
                set(result.get("stages", {})) != {"score", "topk", "pipeline"}
                or not all(
                    v.get("measurement_validated") for v in result["stages"].values()
                )
                or len(result.get("self_controls", {})) != 4
            ):
                raise RuntimeError("Incomplete dot-schedule measurements: " + name)
        summary["aggregate"] = {
            stage: statistics.mean(
                c["stages"][stage]["speedup"] for c in summary["cases"]
            )
            for stage in ("score", "topk", "pipeline")
        }
        print(
            "Incremental production / candidate arithmetic means:",
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
        bench.support.write_json(report / "dot-schedule-summary.json", summary)


def main():
    if "--cases" not in sys.argv:
        sys.argv.extend(["--cases", *DEFAULT_CASES])
    with patch.object(stage, "__doc__", __doc__), patch.object(
        stage, "scoped", scoped
    ), patch.object(stage, "worker", worker), patch.object(
        stage, "coordinate", coordinate
    ):
        return stage.main()


if __name__ == "__main__":
    sys.exit(main())
