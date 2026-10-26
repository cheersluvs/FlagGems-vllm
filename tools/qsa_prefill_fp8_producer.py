# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Compile-only FP8 K producer-first feasibility; no GPU kernel execution."""

import argparse
import json
import os
import re
import sys
import tarfile
import traceback
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import qsa_prefill_double_buffer as protocol  # noqa: E402
from tools import qsa_prefill_fp8_profile as captured  # noqa: E402

producer, trace = protocol.producer, protocol.trace
CASES = ("fp8_tail113_float8_e4m3fn", "fp8_long100k_float8_e4m3fn")
SOURCES = (
    "tools/qsa_prefill_fp8_producer.py",
    "tools/test_qsa_prefill_fp8_producer.py",
    "tools/run_qsa_prefill_fp8_producer_h800.sh",
    "docs/qsa_prefill_fp8_producer.md",
    *producer.SOURCES,
    "tools/qsa_prefill_double_buffer.py",
    "tools/qsa_prefill_producer_runtime.py",
    "tools/qsa_prefill_fp8_profile.py",
    "docs/qsa_prefill_fp8_profile_configs.json",
)
CONFIG = dict(
    num_warps=4,
    num_stages=2,
    num_ctas=1,
    TILE_R=64,
    BLOCK_N=64,
    K_TILES=16,
    QUERY_PARTS=2,
)


def body_lines(ir):
    lines = ir.splitlines()
    loops = [i for i, line in enumerate(lines) if re.search(r"\bscf.for\b", line)]
    if len(loops) != 1:
        raise RuntimeError("Requires one main FP8 loop")
    start = loops[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = next(
        i
        for i in range(start + 1, len(lines))
        if lines[i].startswith(" " * indent + "}")
    )
    return lines[start + 1 : end]


def rewrite(text):
    body = body_lines(text)
    keys = [
        line
        for line in body
        if "tt.load" in line and "tensor<64x128x!tt.ptr<f8E4M3FN>" in line
    ]
    dots = [line for line in body if re.search(r"\bttng.warp_group_dot\b", line)]
    if len(keys) != 1 or len(dots) != 2:
        raise RuntimeError("Unsupported FP8 P2 load/dot schedule")
    if "loop.cluster = 1 : i32" not in keys[0] or "loop.stage = 0 : i32" not in keys[0]:
        raise RuntimeError("K load is not original stage0/cluster1")
    if any(
        "loop.cluster = 0 : i32" not in line or "loop.stage = 1 : i32" not in line
        for line in dots
    ):
        raise RuntimeError("P2 dots are not original stage1/cluster0")
    changed, count = producer.producer_first(text)
    if producer.producer_first(changed)[0] != text:
        raise RuntimeError("Cluster rewrite is not reversible")
    return changed, dict(
        cluster_changes=count,
        stage_numbers_unchanged=True,
        operations_operands_masks_and_math_unchanged=True,
        K_tile_bytes=8192,
        buffer_rewrite_performed=False,
    )


def motion(ptx):
    lines = protocol.runtime.program(ptx).splitlines()
    labels = {
        line[:-1]: i for i, line in enumerate(lines) if re.fullmatch(r"[\w.$]+:", line)
    }
    edges = []
    for i, line in enumerate(lines):
        m = re.fullmatch(r"(@!?%\w+\s+)?bra(?:\.uni)?\s+([\w.$]+);", line)
        if m and m[2] in labels and labels[m[2]] < i:
            edges.append((labels[m[2]], i))
    if len(edges) != 1:
        raise RuntimeError("Unknown final PTX main loop")
    a, b = edges[0]
    body = lines[a + 1 : b]
    if any(re.search(r"\bbra(?:\.uni)?\b", line) for line in body):
        raise RuntimeError("Internal PTX branch in copy-motion check")
    mma = [i for i, line in enumerate(body) if re.search(r"\bwgmma.mma_async\.", line)]
    copies = [
        i
        for i, line in enumerate(body)
        if re.search(
            r"\bcp.async.(?:ca|cg).shared.global \[[^]]+\], \[[^]]+\], (?:0x10|16)(?:,|;)",
            line,
        )
    ]
    if not mma or not copies:
        raise RuntimeError("No identifiable FP8 K16-byte copy/MMA")
    return dict(
        K16byte_copy_sites=len(copies),
        mma_sites=len(mma),
        before_first_MMA=sum(i < min(mma) for i in copies),
        after_last_MMA=sum(i > max(mma) for i in copies),
        all_K_fragments_before_first_MMA=max(copies) < min(mma),
    )


def inspect(ir, ptx):
    body = body_lines(ir)
    copies = [
        i
        for i, line in enumerate(body)
        if "ttg.async_copy_global_to_local" in line
        and "tensor<64x128x!tt.ptr<f8E4M3FN>" in line
    ]
    dots = [
        i for i, line in enumerate(body) if re.search(r"\bttng.warp_group_dot\b", line)
    ]
    waits = [line.strip() for line in body if "ttng.warp_group_dot_wait" in line]
    if len(copies) != 1 or len(dots) != 2 or len(waits) != 2:
        raise RuntimeError("Unidentified FP8 K/P2 dot/wait structure")
    allocations = [
        int(v)
        for line in ir.splitlines()
        if "ttg.local_alloc" in line
        for v in re.findall(r"memdesc<(\d+)x64x128xf8E4M3FN", line)
    ]
    if len(allocations) != 1:
        raise RuntimeError("Ambiguous FP8 K ring allocation")
    result = dict(
        K_slots=allocations[0],
        early_K_copies=int(copies[0] < dots[0]),
        dot_sites=len(dots),
        waits=waits,
        **motion(ptx),
    )
    try:
        result["PTX_protocol"] = protocol.ptx_protocol(ptx)
    except RuntimeError as exc:
        result["protocol_error"] = str(exc)
    return result


def gate(control, candidate):
    a, b = control["schedule"], candidate["schedule"]
    if candidate["spill_store_bytes"] or candidate["spill_load_bytes"]:
        return "candidate spills"
    if any(
        a[k] != b[k] for k in ("dot_sites", "mma_sites", "K16byte_copy_sites")
    ) or len(a["waits"]) != len(b["waits"]):
        return "matrix/copy/materialization work changed"
    if not b["early_K_copies"] or not b["all_K_fragments_before_first_MMA"]:
        return "actual K async copies did not all move before first MMA"
    if b.get("PTX_protocol", {}).get("status") != "passed":
        return b.get("protocol_error", "Missing final PTX synchronization protocol")
    if a["K_slots"] != 2 or b["K_slots"] not in (2, 3):
        return "unexpected compiler K buffering"
    if (
        candidate["shared_bytes"] - control["shared_bytes"]
        != (b["K_slots"] - a["K_slots"]) * 8192
    ):
        return "shared-memory change is not explained by FP8 K ring"
    return None


def prepare(source, report):
    collector = captured.collector
    summary = json.loads((source / "production-profile-summary.json").read_text())
    if (
        summary["status"] != "passed"
        or not summary["hardware_counters_complete"]
        or not summary["diagnostics_complete"]
        or summary["source"] != captured.verify_source()
    ):
        raise RuntimeError("Requires complete unchanged FP8 production profile")
    jobs = []
    with patch.object(collector, "manifest", captured.manifest), patch.object(
        collector, "verify_source", captured.verify_source
    ):
        for name in CASES:
            collector.counter_rows(source, name, summary["selection"])
            record = json.loads((source / ("profile-" + name + ".json")).read_text())
            item = next(
                v for v in captured.manifest()["cases"] if v["case"]["name"] == name
            )
            selected = next(v for v in record["compiled_before"] if v["chunk"][1] == 0)
            if (
                selected["route"] != "fp8"
                or selected["constants"] != CONFIG
                or selected["spill_count"]
            ):
                raise RuntimeError("Wrong FP8 main chunk/config")
            ir, meta = trace.cache_input(
                record["runtime"]["environment"]["TRITON_CACHE_DIR"],
                selected["ptx_sha256"],
            )
            cubin = ir.with_suffix(".cubin")
            if not cubin.is_file() or trace.sha(cubin) != selected["cubin_sha256"]:
                raise RuntimeError("PTX-matched cached cubin changed/missing")
            metadata = json.loads(meta.read_text())
            if metadata.get("ir_override") is not None or metadata["target"] != dict(
                backend="cuda", arch=90, warp_size=32
            ):
                raise RuntimeError("Unsupported replay options/target")
            folder = report / (name + "-replay")
            folder.mkdir()
            for p in (ir, meta, cubin, ir.with_suffix(".ptx")):
                (folder / p.name).write_bytes(p.read_bytes())
            job = dict(
                case=name,
                name=folder.name,
                kind="replay",
                directory=str(folder),
                ir=str(folder / ir.name),
                ir_sha256=trace.sha(ir),
                metadata=str(folder / meta.name),
                metadata_sha256=trace.sha(meta),
                runtime=record["runtime"],
                original_resources=selected,
                expected_ptx_sha256=selected["ptx_sha256"],
                expected_cubin_sha256=selected["cubin_sha256"],
                profile_chunk=selected["chunk"],
                case_definition=item["case"],
            )
            trace.save(folder / "spec.json", job)
            jobs.append(job)
    return jobs


def scheduled_jobs(replay, report):
    folder = Path(replay["directory"])
    text = producer.snapshot((folder / "passes.txt").read_text())
    changed, proof = rewrite(text)
    jobs = []
    for kind, ir_text in (("control", text), ("producer", changed)):
        path = report / (replay["case"] + "-" + kind)
        path.mkdir()
        ir = path / "scheduled.ttir"
        ir.write_text(ir_text)
        meta = path / "options.json"
        meta.write_bytes(Path(replay["metadata"]).read_bytes())
        job = dict(
            replay,
            name=path.name,
            kind=kind,
            directory=str(path),
            ir=str(ir),
            ir_sha256=trace.sha(ir),
            metadata=str(meta),
            metadata_sha256=trace.sha(meta),
            rewrite_proof=proof,
        )
        trace.save(path / "spec.json", job)
        jobs.append(job)
    return jobs


def child(args):
    job = json.loads(Path(args.worker_spec).read_text())
    os.environ["TRITON_DUMP_PTXAS_LOG"] = "1"
    if job["kind"] == "replay":
        return trace.child(args.worker_spec, args.flagtree)
    # Change the diagnostic parser only. Compilation uses the already-tested
    # in-memory resume adapter and original installed pipeline/tail passes.
    with patch.object(producer, "inspect", inspect):
        return producer.child(args)


def run(job, args, summary):
    folder = Path(job["directory"])
    cmd = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--worker-spec",
        str(folder / "spec.json"),
    ]
    rc = trace.run_job(
        cmd,
        dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            PYTHONNOUSERSITE="1",
            PYTHONUNBUFFERED="1",
        ),
        folder / "compiler.txt",
        args.timeout,
    )
    result = (
        json.loads((folder / "result.json").read_text())
        if (folder / "result.json").is_file()
        else {}
    )
    summary["jobs"].append(
        dict(name=job["name"], kind=job["kind"], exit_code=rc, result=result)
    )
    if rc or result.get("status") != (
        "passed" if job["kind"] == "replay" else "compiled"
    ):
        print(
            "\n".join((folder / "compiler.txt").read_text().splitlines()[-40:]),
            flush=True,
        )
        raise RuntimeError("Compiler/control failed: " + job["name"])
    if job["kind"] == "replay":
        if trace.sha(folder / "replayed-ptx.txt") != job["expected_ptx_sha256"]:
            raise RuntimeError("Full replay PTX changed")
        print(job["name"], "exact PTX replay passed", flush=True)
    else:
        result.update(producer.resources((folder / "compiler.txt").read_text()))
        trace.save(folder / "result.json", result)
        print(
            job["name"],
            "K-before-MMA",
            result["schedule"]["before_first_MMA"],
            "/",
            result["schedule"]["K16byte_copy_sites"],
            "slots",
            result["schedule"]["K_slots"],
            "regs",
            result["registers_per_thread"],
            "shared",
            result["shared_bytes"],
            "spill bytes",
            result["spill_store_bytes"],
            result["spill_load_bytes"],
            flush=True,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--flagtree", default="/opt/flagtree")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--worker-spec")
    args = parser.parse_args()
    if args.worker_spec:
        return child(args)
    report = Path(args.report_dir).resolve()
    report.mkdir(parents=True, exist_ok=True)
    summary = dict(
        status="preparing",
        gpu_kernel_executed=False,
        production_modified=False,
        timings={},
        jobs=[],
        decisions=[],
    )
    try:
        trace.save(
            report / "sources.json",
            {
                n: dict(text=(ROOT / n).read_text(), sha256=trace.sha(ROOT / n))
                for n in SOURCES
            },
        )
        jobs = prepare(Path(args.source_report).resolve(), report)
        for replay in jobs:
            run(replay, args, summary)
            control, candidate = scheduled_jobs(replay, report)
            a = run(control, args, summary)
            b = run(candidate, args, summary)
            reason = gate(a, b)
            decision = dict(
                case=replay["case"],
                status=(
                    "not_promising"
                    if reason
                    else "schedule_moved_pending_numeric_and_lifetime_review"
                ),
                reason=reason,
                extra_K_slots=b["schedule"]["K_slots"] - a["schedule"]["K_slots"],
                extra_shared_bytes=b["shared_bytes"] - a["shared_bytes"],
                extra_registers=b["registers_per_thread"] - a["registers_per_thread"],
            )
            summary["decisions"].append(decision)
            print("Decision:", json.dumps(decision), flush=True)
        summary["status"] = (
            "compile_feasible_pending_review"
            if all(v["reason"] is None for v in summary["decisions"])
            else "not_promising"
        )
        return 0
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        print(summary["error"], flush=True)
        return 1
    finally:
        trace.save(report / "fp8-producer-summary.json", summary)
        with tarfile.open(report / "fp8-producer-artifacts.tar.gz", "w:gz") as archive:
            for folder in sorted(report.iterdir()):
                if folder.is_dir() and folder.name.startswith("fp8_"):
                    for p in sorted(folder.iterdir()):
                        if p.is_file():
                            archive.add(
                                p, arcname=str(p.relative_to(report)), recursive=False
                            )
        print("Compile-only report:", report, flush=True)


if __name__ == "__main__":
    sys.exit(main())
