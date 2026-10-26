# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Compile-only producer-first cluster probe; no GPU kernel execution."""

import argparse
import ast
import copy
import importlib
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
from tools import qsa_prefill_pipeline_trace as trace  # noqa: E402

SOURCES = (
    "tools/qsa_prefill_producer_first.py",
    "tools/test_qsa_prefill_producer_first.py",
    "tools/run_qsa_prefill_producer_first.sh",
    "docs/qsa_prefill_producer_first_probe.md",
    "tools/qsa_prefill_pipeline_trace.py",
)


def snapshot(text):
    headers = list(re.finditer(r"^//.*IR Dump (?:Before|After).*$", text, re.M))
    matches = [i for i, h in enumerate(headers) if "Before TritonGPUPipeline " in h[0]]
    if len(matches) != 1:
        raise RuntimeError("Requires exactly one pre-pipeline snapshot")
    i = matches[0]
    return (
        text[
            headers[i].end() : (
                headers[i + 1].start() if i + 1 < len(headers) else len(text)
            )
        ].strip()
        + "\n"
    )


def producer_first(text):
    cluster = r"loop\.cluster = (-?\d+) : i32"
    if set(re.findall(cluster, text)) != {"0", "1"} or set(
        re.findall(r"loop\.stage = (-?\d+) : i32", text)
    ) != {"0", "1"}:
        raise RuntimeError("Requires the original two-stage/two-cluster schedule")
    if len(re.findall(r"\bscf.for\b", text)) != 1:
        raise RuntimeError("Requires one original score loop")
    result, count = re.subn(
        cluster, lambda m: "loop.cluster = " + str(1 - int(m[1])) + " : i32", text
    )
    if count < 2:
        raise RuntimeError("Incomplete cluster rewrite")
    return result, count


def resume_function(source, namespace):
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CUDABackend"
    )
    fn = copy.deepcopy(
        next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "make_ttgir"
        )
    )
    if [n.arg for n in fn.args.args] != ["mod", "metadata", "opt", "capability"]:
        raise RuntimeError("Unsupported backend method signature")
    branches = [
        (i, n)
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "capability // 10 in [8, 9]"
    ]
    if len(branches) != 1:
        raise RuntimeError("Unsupported SM90 pipeline structure")
    index, branch = branches[0]
    calls = [
        n
        for n in branch.body
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "passes.ttgpuir.add_pipeline"
    ]
    if (
        len(calls) != 1
        or ast.unparse(calls[0].value)
        != "passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)"
    ):
        raise RuntimeError("Unsupported pipeline entry")
    fn.name = "_qsa_resume_ttgir"
    fn.decorator_list = []
    fn.body = (
        ast.parse(
            "pm = ir.pass_manager(mod.context)\ndump_enabled = pm.enable_debug()\n"
        ).body
        + copy.deepcopy(calls)
        + fn.body[index + 1 :]
    )
    adapted = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    scope = dict(namespace)
    exec(compile(adapted, "<private-qsa-compiler-tail>", "exec"), scope)
    return scope[fn.name], ast.unparse(adapted) + "\n"


def inspect(ir, ptx):
    lines = ir.splitlines()
    loops = [i for i, line in enumerate(lines) if re.search(r"\bscf.for\b", line)]
    if len(loops) != 1:
        raise RuntimeError("Unidentified main loop")
    start = loops[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = next(
        i
        for i in range(start + 1, len(lines))
        if lines[i].startswith(" " * indent + "}")
    )
    body = lines[start + 1 : end]
    dots = [
        i for i, line in enumerate(body) if re.search(r"\bttng.warp_group_dot\b", line)
    ]
    copies = [
        i
        for i, line in enumerate(body)
        if "ttg.async_copy_global_to_local" in line
        and "tensor<64x128x!tt.ptr<bf16>" in line
    ]
    waits = [line.strip() for line in body if "ttng.warp_group_dot_wait" in line]
    slots = 0
    for line in lines:
        m = (
            re.search(r"memdesc<((?:\d+x)+)bf16", line)
            if "ttg.local_alloc" in line
            else None
        )
        if m:
            dims = [int(v) for v in m[1].rstrip("x").split("x")]
            if dims[-2:] == [64, 128]:
                n = 1
                for d in dims[:-2]:
                    n *= d
                slots += n
    if not dots or len(copies) != 1 or not waits:
        raise RuntimeError("Unidentified K-copy/dot/wait schedule")
    clean = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)
    return dict(
        K_slots=slots,
        early_K_copies=sum(i < dots[0] for i in copies),
        K_copy_line=start + 2 + copies[0],
        first_dot_line=start + 2 + dots[0],
        dot_sites=len(dots),
        waits=waits,
        mma_sites=len(re.findall(r"\bwgmma.mma_async\.", clean)),
        async_copy_sites=len(
            re.findall(r"\bcp.async.(?:ca|cg).shared.global\b", clean)
        ),
    )


def resources(text):
    regs = set(int(v) for v in re.findall(r"Used (\d+) registers", text))
    spill = set(
        (int(a), int(b))
        for a, b in re.findall(
            r"(\d+) bytes spill stores, (\d+) bytes spill loads", text
        )
    )
    if len(regs) != 1 or len(spill) != 1:
        raise RuntimeError("Unknown/ambiguous PTXAS register or spill resources")
    stores, loads = next(iter(spill))
    return dict(
        registers_per_thread=next(iter(regs)),
        spill_store_bytes=stores,
        spill_load_bytes=loads,
    )


def gate(control, candidate):
    a, b = control["schedule"], candidate["schedule"]
    if candidate["spill_store_bytes"] or candidate["spill_load_bytes"]:
        return "candidate spills"
    if a["mma_sites"] != b["mma_sites"] or a["dot_sites"] != b["dot_sites"]:
        return "matrix work changed"
    if len(a["waits"]) != len(b["waits"]):
        return "WGMMA materialization wait count changed"
    if not b["async_copy_sites"] or not b["early_K_copies"]:
        return "no early hardware-async K copy"
    if b["K_slots"] not in (2, 3):
        return "unexpected K buffer count"
    return None


def prepare(source, report):
    summary = json.loads((source / "pipeline-trace-summary.json").read_text())
    if (
        summary["status"] != "passed"
        or len(summary["jobs"]) != 4
        or summary["gpu_kernel_executed"]
    ):
        raise RuntimeError("Requires successful four-way compile-only trace")
    expected_names = {
        case + "-" + side for case in trace.CASES for side in ("baseline", "candidate")
    }
    if {j["name"] for j in summary["jobs"]} != expected_names or any(
        j["exit_code"]
        or not all(j["coverage"].values())
        or j["result"]["status"] != "passed"
        or j["result"]["actual_ptx_sha256"] != j["result"]["expected_ptx_sha256"]
        for j in summary["jobs"]
    ):
        raise RuntimeError("Incomplete four-way trace identity")
    jobs = []
    for case in trace.CASES:
        old = source / (case + "-baseline")
        spec = json.loads((old / "spec.json").read_text())
        prior = next(j for j in summary["jobs"] if j["name"] == old.name)
        if (
            prior["exit_code"]
            or not all(prior["coverage"].values())
            or prior["result"]["status"] != "passed"
        ):
            raise RuntimeError("Unvalidated captured production trace")
        if (
            trace.sha(old / "replayed-ptx.txt") != spec["expected_ptx_sha256"]
            or prior["result"]["actual_ptx_sha256"] != spec["expected_ptx_sha256"]
        ):
            raise RuntimeError("Captured PTX identity changed")
        if spec["original_resources"]["constants"]["num_stages"] != 2:
            raise RuntimeError("Requires original production stage2")
        original = snapshot((old / "passes.txt").read_text())
        changed, count = producer_first(original)
        metadata = old / Path(spec["metadata"]).name
        if trace.sha(metadata) != spec["metadata_sha256"]:
            raise RuntimeError("Captured compiler options changed")
        for kind, text in (("control", original), ("producer", changed)):
            folder = report / (case + "-" + kind)
            folder.mkdir()
            # .ttir selects the patched TTGPU resume entry; its content is the
            # already-scheduled GPU IR. No TTIR/early GPU passes are repeated.
            ir = folder / "scheduled.ttir"
            ir.write_text(text)
            meta = folder / "options.json"
            meta.write_bytes(metadata.read_bytes())
            job = dict(
                spec,
                name=folder.name,
                directory=str(folder),
                kind=kind,
                ir=str(ir),
                ir_sha256=trace.sha(ir),
                metadata=str(meta),
                metadata_sha256=trace.sha(meta),
                cluster_changes=count if kind == "producer" else 0,
            )
            trace.save(folder / "spec.json", job)
            jobs.append(job)
    return jobs


def child(options):
    job = json.loads(Path(options.worker_spec).read_text())
    folder = Path(job["directory"])
    os.environ.update(
        MLIR_ENABLE_DUMP="1",
        MLIR_DUMP_PATH=str(folder / "passes.txt"),
        TRITON_ALWAYS_COMPILE="1",
        TRITON_DUMP_PTXAS_LOG="1",
        TRITON_CACHE_DIR=str(folder / "cache"),
        CUDA_VISIBLE_DEVICES="",
    )
    for name in (
        "TRITON_PTXAS_PATH",
        "TRITON_DISABLE_LINE_INFO",
        "TRITON_OVERRIDE_ARCH",
        "DISABLE_PTXAS_OPT",
        "PTXAS_OPTIONS",
    ):
        value = job["runtime"]["environment"].get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    for name in (
        "TRITON_KERNEL_OVERRIDE",
        "TRITON_KERNEL_DUMP",
        "TRITON_OVERRIDE_DIR",
        "LLVM_IR_ENABLE_DUMP",
    ):
        os.environ.pop(name, None)
    sys.path.insert(0, str(Path(options.flagtree).resolve()))
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia import compiler as backend

    for name, record in job["runtime"]["compiler_files"].items():
        path = Path(importlib.import_module(name).__file__).resolve()
        if (
            path != Path(record["path"]).resolve()
            or trace.sha(path) != record["sha256"]
        ):
            raise RuntimeError("Compiler identity changed: " + name)
    if (
        trace.sha(triton.__file__) != job["runtime"]["triton_init_sha256"]
        or trace.sha(os.environ["TRITON_PTXAS_PATH"])
        != job["runtime"]["ptxas"]["sha256"]
    ):
        raise RuntimeError("Runtime/assembler identity changed")
    if (
        trace.sha(job["ir"]) != job["ir_sha256"]
        or trace.sha(job["metadata"]) != job["metadata_sha256"]
    ):
        raise RuntimeError("Replay input changed")
    resume, code = resume_function(Path(backend.__file__).read_text(), vars(backend))
    (folder / "compiler-tail.txt").write_text(code)
    metadata = json.loads(Path(job["metadata"]).read_text())
    target = metadata["target"]
    if target != dict(backend="cuda", arch=90, warp_size=32):
        raise RuntimeError("Unexpected target")
    before = trace.sha(backend.__file__)
    with patch.object(backend.CUDABackend, "make_ttgir", staticmethod(resume)):
        kernel = triton.compile(
            job["ir"],
            target=GPUTarget(**target),
            options=trace.restore_options(metadata),
        )
    if before != trace.sha(backend.__file__):
        raise RuntimeError("Installed compiler changed")
    for name in ("ptx", "ttgir"):
        (folder / (name + ".txt")).write_text(kernel.asm[name])
    (folder / "kernel.cubin").write_bytes(kernel.asm["cubin"])
    record = dict(
        status="compiled",
        gpu_kernel_executed=False,
        ptx_sha256=trace.sha(folder / "ptx.txt"),
        cubin_sha256=trace.sha(folder / "kernel.cubin"),
        shared_bytes=kernel.metadata.shared,
        schedule=inspect(kernel.asm["ttgir"], kernel.asm["ptx"]),
    )
    if job["kind"] == "control" and record["ptx_sha256"] != job["expected_ptx_sha256"]:
        record["status"] = "control_mismatch"
    trace.save(folder / "result.json", record)
    return 0 if record["status"] == "compiled" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--flagtree", default="/opt/flagtree")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--worker-spec")
    options = parser.parse_args()
    if options.worker_spec:
        return child(options)
    report = Path(options.report_dir).resolve()
    report.mkdir(parents=True, exist_ok=True)
    summary = dict(
        status="preparing", gpu_kernel_executed=False, timings={}, jobs=[], decisions=[]
    )
    trace.save(
        report / "sources.json",
        {
            name: dict(text=(ROOT / name).read_text(), sha256=trace.sha(ROOT / name))
            for name in SOURCES
        },
    )
    try:
        if (
            trace.sha(ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py")
            != trace.PRODUCTION
        ):
            raise RuntimeError("Production changed")
        jobs = prepare(Path(options.source_report).resolve(), report)
        for job in jobs:
            folder = Path(job["directory"])
            cmd = [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--worker-spec",
                str(folder / "spec.json"),
            ]
            code = trace.run_job(
                cmd,
                dict(
                    os.environ,
                    PYTHONNOUSERSITE="1",
                    PYTHONUNBUFFERED="1",
                    CUDA_VISIBLE_DEVICES="",
                ),
                folder / "compiler.txt",
                options.timeout,
            )
            result = (
                json.loads((folder / "result.json").read_text())
                if (folder / "result.json").is_file()
                else {}
            )
            summary["jobs"].append(
                dict(name=job["name"], exit_code=code, result=result)
            )
            if code or result.get("status") != "compiled":
                print(
                    "\n".join((folder / "compiler.txt").read_text().splitlines()[-40:]),
                    flush=True,
                )
                raise RuntimeError("Compiler/control failed: " + job["name"])
            result.update(resources((folder / "compiler.txt").read_text()))
            trace.save(folder / "result.json", result)
            print(
                job["name"],
                "early K",
                result["schedule"]["early_K_copies"],
                "K slots",
                result["schedule"]["K_slots"],
                "registers",
                result["registers_per_thread"],
                "shared bytes",
                result["shared_bytes"],
                "spill store/load bytes",
                result["spill_store_bytes"],
                result["spill_load_bytes"],
                flush=True,
            )
            if job["kind"] == "producer":
                control = summary["jobs"][-2]["result"]
                reason = gate(control, result)
                decision = dict(
                    case=job["name"].rsplit("-", 1)[0],
                    status=(
                        "not_promising"
                        if reason
                        else "compile_feasible_pending_accuracy"
                    ),
                    reason=reason,
                    extra_K_slots=result["schedule"]["K_slots"]
                    - control["schedule"]["K_slots"],
                    extra_shared_bytes=result["shared_bytes"] - control["shared_bytes"],
                )
                summary["decisions"].append(decision)
                print("Decision:", decision, flush=True)
        summary["status"] = (
            "compile_feasible_pending_accuracy"
            if any(d["reason"] is None for d in summary["decisions"])
            else "not_promising"
        )
        print("No GPU execution. Score/TopK/pipeline timing: N/A.", flush=True)
        return 0
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        print(summary["error"], flush=True)
        return 1
    finally:
        trace.save(report / "producer-first-summary.json", summary)
        with tarfile.open(report / "producer-artifacts.tar.gz", "w:gz") as archive:
            for folder in sorted(report.iterdir()):
                if folder.is_dir() and folder.name.startswith("perf_"):
                    for path in sorted(folder.iterdir()):
                        if path.is_file():
                            archive.add(
                                path,
                                arcname=str(path.relative_to(report)),
                                recursive=False,
                            )


if __name__ == "__main__":
    sys.exit(main())
