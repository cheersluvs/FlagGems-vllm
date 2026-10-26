# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""R64 Python-source lookahead feasibility; three CPU compilations, no GPU run."""

import argparse
import ast
import contextlib
import importlib
import importlib.util
import json
import os
import re
import sys
import tarfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import qsa_prefill_double_buffer as double  # noqa: E402

trace, producer = double.trace, double.producer
CASE = "perf_n4_q4096_p102400_bfloat16"
OP = "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
EXPLICIT = "tools/qsa_prefill_explicit_prefetch_kernel.py"
PRODUCTION_SHA = "0d7e80e240966565a7c78151c8344f17e0627e73920f5229aadcc0f0fb42f10e"
EXPLICIT_SHA = "7d9a3397135c2428ccb82c80391d477f0d5caf71ef94dafe73681e3ec3ba671d"
HINT_MANAGER_SHA = "0742283d2f4927f0fd09d3770254d04d82b4f4f9b6a34251d1bee043c08c52fb"
SOURCES = (
    OP,
    EXPLICIT,
    "tools/qsa_prefill_source_pipeline.py",
    "tools/test_qsa_prefill_source_pipeline.py",
    "tools/run_qsa_prefill_source_pipeline.sh",
    "docs/qsa_prefill_source_pipeline.md",
    *double.SOURCES,
)


def source(kind):
    path = ROOT / (OP if kind == "production" else EXPLICIT)
    expected = PRODUCTION_SHA if kind == "production" else EXPLICIT_SHA
    if trace.sha(path) != expected:
        raise RuntimeError("Frozen source changed: " + str(path))
    original = path.read_text()
    name = (
        "_qsa_prefill_score_kernel"
        if kind == "production"
        else "_explicit_prefetch_score_kernel"
    )
    fn = next(
        n
        for n in ast.parse(original).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    text = ast.get_source_segment(original, fn)
    text = text.replace("def " + name + "(", "def _qsa_prefill_score_kernel(", 1)
    if kind == "stages2":
        old = "tl.range(first_tile, last_tile, num_stages=1)"
        if text.count(old) != 1:
            raise RuntimeError("Unexpected explicit lookahead iterator")
        changed = text.replace(old, old.replace("=1", "=2"))
        if changed.replace("num_stages=2", "num_stages=1") != text:
            raise RuntimeError("Changed non-stage source")
        text = changed
    elif kind not in ("production", "stages1"):
        raise RuntimeError("Unknown source variant")
    return (
        "import triton\nimport triton.language as tl\n\n"
        '@triton.jit(do_not_specialize=["num_rows", "query_offset", "max_query_len"])\n'
        + text
        + "\n"
    )


def abi(ir, generated, inputs, constants):
    names = next(
        n for n in ast.parse(generated).body if isinstance(n, ast.FunctionDef)
    ).args.args
    names = [n.arg for n in names]
    header = next(
        line
        for line in ir.splitlines()
        if "tt.func public @_qsa_prefill_score_kernel(" in line
    )
    found = list(re.finditer(r"%(\w+): (!tt.ptr<\w+>|i32)", header))
    signature, attrs = {}, {}
    for i, match in enumerate(found):
        name, typ = match.groups()
        if name not in names or name in signature:
            raise RuntimeError("Unknown/duplicate runtime ABI argument")
        signature[name] = "*" + typ[8:-1] if typ.startswith("!tt.ptr<") else typ
        section = header[
            match.end() : (
                found[i + 1].start()
                if i + 1 < len(found)
                else header.index(" attributes ")
            )
        ]
        attribute = re.search(r"tt.divisibility = (\d+) : i32", section)
        if attribute:
            attrs[(names.index(name),)] = [("tt.divisibility", int(attribute[1]))]
    if (
        set(names[:19]) - set(signature) != {"stride_table_page"}
        or len(signature) != 18
    ):
        raise RuntimeError("Unsupported captured ABI specialization")
    tensors = {v["argument"]: v for v in inputs}
    q, cache, table = (tensors[i] for i in (0, 1, 2))
    if (
        q["dtype"] != "torch.bfloat16"
        or cache["dtype"] != q["dtype"]
        or q["shape"][1:] != [4, 128]
    ):
        raise RuntimeError("Unsupported source profile")
    if cache["shape"][2:] != [1, 128] or table["stride"][1] != 1:
        raise RuntimeError("Unsupported cache/table layout")
    fixed = dict(
        stride_table_page=1,
        PAGE_SIZE=cache["shape"][1],
        NUM_HEADS=q["shape"][1],
        HEAD_DIM=q["shape"][2],
        IS_FP8=False,
        **{k: constants[k] for k in ("TILE_R", "BLOCK_N", "K_TILES", "HEAD_GROUP")},
    )
    for name in fixed:
        signature[name] = "constexpr"
    if set(signature) != set(names):
        raise RuntimeError("Incomplete source signature")
    return signature, fixed, attrs


def inspect(ir, ptx):
    try:
        return producer.inspect(ir, ptx)
    except RuntimeError as exc:
        # Keep rejected source compilations observable, including register-load
        # and unrolled forms. Unknown schedules cannot pass the feasibility gate.
        clean = double.runtime.program(ptx)
        return dict(
            schedule_error=str(exc),
            K_slots=None,
            early_K_copies=0,
            dot_sites=len(re.findall(r"\bttng.warp_group_dot\b", ir)),
            waits=re.findall(r".*ttng.warp_group_dot_wait.*", ir),
            mma_sites=len(re.findall(r"\bwgmma.mma_async\.", clean)),
            async_copy_sites=len(
                re.findall(r"\bcp.async.(?:ca|cg).shared.global\b", clean)
            ),
        )


def gate(control, candidate):
    if candidate["schedule"].get("schedule_error"):
        return candidate["schedule"]["schedule_error"]
    reason = producer.gate(control, candidate)
    if reason:
        return reason
    if candidate["schedule"]["K_slots"] != 2:
        return "source does not produce two K slots"
    if candidate["shared_bytes"] > 106496 or candidate["registers_per_thread"] > 217:
        return "exceeds validated ring2 resource footprint"
    if candidate.get("protocol_error"):
        return candidate["protocol_error"]
    if candidate.get("PTX_protocol", {}).get("status") != "passed":
        return "no verified final PTX lifetime protocol"
    return None


def prepare(previous, report):
    summary = json.loads((previous / "double-buffer-runtime-summary.json").read_text())
    if summary["status"] != "passed" or summary["production_modified"]:
        raise RuntimeError("Requires successful ring2 runtime report")
    profile = next(
        v for v in summary["source"]["producer_profiles"] if v["case"] == CASE
    )
    compiled = previous / "compile"
    proof = json.loads((compiled / "double-buffer-summary.json").read_text())
    if proof["status"] != "compile_feasible_pending_accuracy":
        raise RuntimeError("Missing validated compiler report")
    for path, record in json.loads((compiled / "sources.json").read_text()).items():
        if trace.sha(ROOT / path) != record["sha256"]:
            raise RuntimeError("Prior proof source changed")
    captured = compiled / (CASE + "-two")
    record = json.loads((captured / "result.json").read_text())
    if (
        record != profile["candidate"]
        or trace.sha(captured / "ptx.txt") != profile["candidate_ptx_sha256"]
    ):
        raise RuntimeError("Validated candidate changed")
    if trace.sha(captured / "kernel.cubin") != record["cubin_sha256"]:
        raise RuntimeError("Validated binary changed")
    old = json.loads((captured / "spec.json").read_text())
    options = captured / "options.json"
    if trace.sha(options) != old["metadata_sha256"]:
        raise RuntimeError("Captured options changed")
    case = json.loads((previous / (CASE + ".json")).read_text())
    baseline = case["kernels_before"]["baseline"][0]
    if case["status"] != "passed" or baseline != case["kernels_after"]["baseline"][0]:
        raise RuntimeError("Production profile changed during reference run")
    expected = dict(
        num_warps=4,
        num_stages=2,
        num_ctas=1,
        TILE_R=64,
        BLOCK_N=64,
        K_TILES=16,
        HEAD_GROUP=True,
    )
    if baseline["constants"] != expected or baseline["spill_count"] != 0:
        raise RuntimeError("Unsupported fixed launch configuration")
    jobs = []
    for kind in ("production", "stages1", "stages2"):
        folder = report / kind
        folder.mkdir()
        text = source(kind)
        path = folder / "kernel.py"
        path.write_text(text)
        sig, fixed, attrs = abi(
            (captured / "input.ttgir").read_text(), text, case["inputs"], expected
        )
        (folder / "options.json").write_bytes(options.read_bytes())
        job = dict(
            kind=kind,
            directory=str(folder),
            module=str(path),
            module_sha256=trace.sha(path),
            signature=sig,
            constants=fixed,
            attrs=[[list(k), v] for k, v in attrs.items()],
            runtime=old["runtime"],
            metadata=str(folder / "options.json"),
            metadata_sha256=trace.sha(options),
            control_code_sha256=profile["control_code_sha256"],
            production_resources=baseline,
            positive_resources=record,
        )
        trace.save(folder / "spec.json", job)
        jobs.append(job)
    return jobs


@contextlib.contextmanager
def cuda_hints(module):
    """Select the explicit compile target without invoking device discovery.

    Keep the pinned FlagTree HintManager/handler/trigger unchanged. The only
    scoped state change is its otherwise lazily initialized backend selection.
    """
    path = Path(module.__file__).resolve()
    if trace.sha(path) != HINT_MANAGER_SHA:
        raise RuntimeError("Unsupported FlagTree hint-manager source")
    previous = module._global_hint_manager
    if previous is not None and previous.backend_name != "cuda":
        raise RuntimeError("Conflicting existing hint backend")
    try:
        if previous is None:
            module._global_hint_manager = module.HintManager("cuda")
        selected = module._global_hint_manager
        yield dict(
            backend=selected.backend_name,
            handler=type(selected.handler).__module__
            + "."
            + type(selected.handler).__qualname__,
            source=str(path),
            sha256=HINT_MANAGER_SHA,
            selection="explicit CUDA target;native handler;no device discovery",
        )
    finally:
        module._global_hint_manager = previous


def child(args):
    job = json.loads(Path(args.worker_spec).read_text())
    folder = Path(job["directory"])
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        TRITON_ALWAYS_COMPILE="1",
        TRITON_DUMP_PTXAS_LOG="1",
        TRITON_CACHE_DIR=str(folder / "cache"),
        MLIR_ENABLE_DUMP="0",
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
    sys.path.insert(0, str(Path(args.flagtree).resolve()))
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    for name, expected in job["runtime"]["compiler_files"].items():
        path = Path(importlib.import_module(name).__file__).resolve()
        if (
            path != Path(expected["path"]).resolve()
            or trace.sha(path) != expected["sha256"]
        ):
            raise RuntimeError("Compiler identity mismatch: " + name)
    if (
        trace.sha(triton.__file__) != job["runtime"]["triton_init_sha256"]
        or trace.sha(os.environ["TRITON_PTXAS_PATH"])
        != job["runtime"]["ptxas"]["sha256"]
    ):
        raise RuntimeError("Compiler/assembler changed")
    if (
        trace.sha(job["module"]) != job["module_sha256"]
        or trace.sha(job["metadata"]) != job["metadata_sha256"]
    ):
        raise RuntimeError("Source/options changed")
    spec = importlib.util.spec_from_file_location(
        "qsa_source_" + job["kind"], job["module"]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    metadata = json.loads(Path(job["metadata"]).read_text())
    if (
        metadata["target"] != dict(backend="cuda", arch=90, warp_size=32)
        or metadata.get("ir_override") is not None
    ):
        raise RuntimeError("Unexpected target/override")
    src = ASTSource(
        module._qsa_prefill_score_kernel,
        job["signature"],
        job["constants"],
        {tuple(k): [tuple(v) for v in values] for k, values in job["attrs"]},
    )
    from triton.compiler import hint_manager

    with cuda_hints(hint_manager) as hints:
        print("Offline hint backend:", json.dumps(hints), flush=True)
        kernel = triton.compile(
            src,
            target=GPUTarget(**metadata["target"]),
            options=trace.restore_options(metadata),
        )
    for name in ("ttir", "ttgir", "llir", "ptx"):
        (folder / (name + ".txt")).write_text(kernel.asm[name])
    (folder / "kernel.cubin").write_bytes(kernel.asm["cubin"])
    result = dict(
        status="compiled",
        hint_backend=hints,
        gpu_kernel_executed=False,
        ptx_sha256=trace.sha(folder / "ptx.txt"),
        cubin_sha256=trace.sha(folder / "kernel.cubin"),
        code_sha256=double.runtime.code_sha(kernel.asm["ptx"]),
        shared_bytes=kernel.metadata.shared,
        schedule=inspect(kernel.asm["ttgir"], kernel.asm["ptx"]),
    )
    try:
        result["PTX_protocol"] = double.ptx_protocol(kernel.asm["ptx"])
    except RuntimeError as exc:
        result["protocol_error"] = str(exc)
    trace.save(folder / "result.json", result)
    return 0


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
        for job in jobs:
            folder = Path(job["directory"])
            code = trace.run_job(
                [
                    sys.executable,
                    "-I",
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                    "--worker-spec",
                    str(folder / "spec.json"),
                ],
                dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1"),
                folder / "compiler.txt",
                args.timeout,
            )
            if code or not (folder / "result.json").is_file():
                print(
                    "\n".join((folder / "compiler.txt").read_text().splitlines()[-40:]),
                    flush=True,
                )
                raise RuntimeError("Source compilation failed: " + job["kind"])
            result = json.loads((folder / "result.json").read_text())
            result.update(producer.resources((folder / "compiler.txt").read_text()))
            trace.save(folder / "result.json", result)
            summary["jobs"].append(
                dict(kind=job["kind"], exit_code=code, result=result)
            )
            print(
                job["kind"],
                "regs",
                result["registers_per_thread"],
                "shared",
                result["shared_bytes"],
                "K slots",
                result["schedule"]["K_slots"],
                "early K copies",
                result["schedule"]["early_K_copies"],
                "cp.async sites",
                result["schedule"]["async_copy_sites"],
                "MMA",
                result["schedule"]["mma_sites"],
                "spill bytes",
                result["spill_store_bytes"],
                result["spill_load_bytes"],
                flush=True,
            )
            if job["kind"] == "production":
                baseline = job["production_resources"]
                if (
                    result["code_sha256"] != job["control_code_sha256"]
                    or (result["registers_per_thread"], result["shared_bytes"])
                    != (baseline["registers_per_thread"], baseline["shared_bytes"])
                    or result["spill_store_bytes"]
                    or result["spill_load_bytes"]
                ):
                    raise RuntimeError(
                        "Original Python source did not reproduce production;do not interpret candidate"
                    )
            elif job["kind"] == "stages1":
                if (
                    result["registers_per_thread"],
                    result["shared_bytes"],
                    result["schedule"]["async_copy_sites"],
                    result["spill_store_bytes"],
                    result["spill_load_bytes"],
                ) != (215, 81920, 0, 0, 0):
                    raise RuntimeError(
                        "Source stages1 negative control did not reproduce known resources"
                    )
            else:
                reason = gate(summary["jobs"][0]["result"], result)
                summary.update(
                    status=(
                        "not_promising"
                        if reason
                        else "compile_promising_pending_lifetime_and_accuracy"
                    ),
                    reason=reason,
                )
                print(
                    "Decision:",
                    summary["status"],
                    "reason:",
                    reason,
                    "GPU executed:False;no timings",
                    flush=True,
                )
        return 0
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        print(summary["error"], flush=True)
        return 1
    finally:
        trace.save(report / "source-pipeline-summary.json", summary)
        with tarfile.open(
            report / "source-compile-artifacts.tar.gz", "w:gz"
        ) as archive:
            for folder in (report / n for n in ("production", "stages1", "stages2")):
                if folder.is_dir():
                    for path in sorted(folder.iterdir()):
                        if path.is_file():
                            archive.add(
                                path,
                                arcname=str(path.relative_to(report)),
                                recursive=False,
                            )
        print("Compile report:", report, flush=True)


if __name__ == "__main__":
    sys.exit(main())
