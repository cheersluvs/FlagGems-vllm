# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Source-only FP8 paired-dot compilation; no GPU execution or IR override."""

import argparse
import ast
import json
import os
import re
import sys
import tarfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import qsa_prefill_source_pipeline as compiler  # noqa: E402

trace = compiler.trace
NAME = "_qsa_prefill_fp8_store_score_kernel"
CASE = "fp8_tail113_float8_e4m3fn"
IR_SHA = "34341d3e47c9d09f788db6ea308c301a106d60dec25691bbb22c93b310f4bbbd"
OPTIONS_SHA = "1d535376c07db69eaf7a34f0dac9b7fd545ba227bfea5dbb0fa76e7ca2144f42"
PTX_SHA = "e1a00df3db738150cd2baeedacba5bf7cea8f6eb7c0a562023a2cb75921d55b8"
MOVE = "            scores_right = tl.dot(keys, query_right, out_dtype=tl.float32)\n"
ANCHOR = "            scores_left = tl.dot(keys, query_left, out_dtype=tl.float32)\n"
SOURCES = (
    "tools/qsa_prefill_fp8_dot_issue.py",
    "tools/test_qsa_prefill_fp8_dot_issue.py",
    "tools/run_qsa_prefill_fp8_dot_issue_h800.sh",
    "docs/qsa_prefill_fp8_dot_issue.md",
    *compiler.SOURCES,
)


def source(kind):
    path = ROOT / compiler.OP
    if trace.sha(path) != compiler.PRODUCTION_SHA:
        raise RuntimeError("Production source changed")
    text = path.read_text()
    functions = {
        n.name: ast.get_source_segment(text, n)
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef)
    }
    kernel = functions[NAME] + "\n"
    if kernel.count(MOVE) != 1 or kernel.count(ANCHOR) != 1:
        raise RuntimeError("Unexpected FP8 split-query source")
    if kind == "paired":
        kernel = kernel.replace(MOVE, "").replace(ANCHOR, ANCHOR + MOVE)
    elif kind != "production":
        raise ValueError("Unknown source variant")
    return (
        "import triton\nimport triton.language as tl\n\n@triton.jit\n"
        + functions["_fp8_bulk_tail_store"]
        + '\n\n@triton.jit(do_not_specialize=["num_rows", "query_offset", "max_query_len"])\n'
        + kernel
        + "\n_qsa_prefill_score_kernel = "
        + NAME
        + "\n"
    )


def abi(ir):
    names = next(
        n
        for n in ast.parse(source("production")).body
        if isinstance(n, ast.FunctionDef) and n.name == NAME
    ).args.args
    names = [n.arg for n in names]
    header = next(
        line for line in ir.splitlines() if "tt.func public @" + NAME + "(" in line
    )
    found = list(re.finditer(r"%(\w+): (!tt.ptr<\w+>|i32)", header))
    signature, attrs = {}, {}
    for i, match in enumerate(found):
        name, typ = match.groups()
        if name not in names or name in signature:
            raise RuntimeError("Unexpected captured ABI")
        element = typ[8:-1] if typ.startswith("!tt.ptr<") else typ
        types = {"f8E4M3FN": "fp8e4nv", "f32": "fp32", "i32": "i32"}
        if element not in types:
            raise RuntimeError("Unsupported captured type: " + element)
        signature[name] = ("*" if typ.startswith("!tt.ptr<") else "") + types[element]
        end = (
            found[i + 1].start() if i + 1 < len(found) else header.index(" attributes ")
        )
        attr = re.search(r"tt.divisibility = (\d+) : i32", header[match.end() : end])
        if attr:
            attrs[(names.index(name),)] = [("tt.divisibility", int(attr[1]))]
    constants = dict(
        stride_table_page=1,
        num_requests=1,
        PAGE_SIZE=196,
        NUM_HEADS=4,
        HEAD_DIM=128,
        IS_FP8=True,
        TILE_R=64,
        BLOCK_N=64,
        K_TILES=16,
        QUERY_PARTS=2,
    )
    if set(signature) != set(
        n for n in names[:19] if n not in ("stride_table_page", "num_requests")
    ):
        raise RuntimeError("Incomplete runtime ABI")
    if signature["q_ptr"] != "*fp8e4nv" or signature["cache_ptr"] != "*fp8e4nv":
        raise RuntimeError("Expected original FP8 pointer types")
    signature.update({name: "constexpr" for name in constants})
    if set(signature) != set(names):
        raise RuntimeError("Incomplete source signature")
    return signature, constants, attrs


def metrics(ir, ptx):
    clean = compiler.double.runtime.program(ptx)
    return dict(
        dot_epilogue_order=[
            line.strip()
            for line in ir.splitlines()
            if re.search(
                r"\b(?:ttng.warp_group_dot(?:_wait)?|tt.reduce|tt.store|ttg.convert_layout)\b",
                line,
            )
        ],
        dot_sites=len(re.findall(r"\bttng.warp_group_dot\b", ir)),
        dot_wait_sites=len(re.findall(r"\bttng.warp_group_dot_wait\b", ir)),
        mma_sites=len(re.findall(r"\bwgmma.mma_async\.", clean)),
        mma_wait_sites=len(re.findall(r"\bwgmma.wait_group\b", clean)),
        barriers=len(re.findall(r"\bbar.sync\b", clean)),
        async_copy_sites=len(
            re.findall(r"\bcp.async.(?:ca|cg).shared.global\b", clean)
        ),
    )


def gate(a, b):
    if b["spill_store_bytes"] or b["spill_load_bytes"]:
        return "candidate spills"
    if (
        b["registers_per_thread"] > a["registers_per_thread"]
        or b["shared_bytes"] > a["shared_bytes"]
    ):
        return "candidate resource usage increased"
    if any(
        a["instructions"][n] != b["instructions"][n] for n in ("dot_sites", "mma_sites")
    ):
        return "matrix work changed"
    if a["code_sha256"] == b["code_sha256"]:
        return "compiler emitted identical executable PTX"
    return None


def tile32_gate(original, sequential, paired):
    if paired["spill_store_bytes"] or paired["spill_load_bytes"]:
        return "candidate spills"
    if any(paired[k] > original[k] for k in ("registers_per_thread", "shared_bytes")):
        return "candidate exceeds original R64 resource ceiling"
    if any(
        sequential["instructions"][k] != paired["instructions"][k]
        for k in ("dot_sites", "mma_sites")
    ):
        return "same-tile matrix work changed"
    if (
        sequential["instructions"]["dot_sites"] != 2
        or sequential["instructions"]["mma_sites"] == 0
    ):
        return "unrecognized R32 matrix work"
    if paired["code_sha256"] == sequential["code_sha256"]:
        return "compiler emitted identical executable PTX"
    order = paired["instructions"]["dot_epilogue_order"]
    dots = [
        i for i, line in enumerate(order) if re.search(r"\bttng.warp_group_dot\b", line)
    ]
    waits = [i for i, line in enumerate(order) if "ttng.warp_group_dot_wait" in line]
    if (
        len(dots) != 2
        or len(waits) != 2
        or not (dots[0] < dots[1] < waits[0] < waits[1])
    ):
        return "paired issue not retained"
    pendings = [re.search(r"pendings = (\d+)", order[i]) for i in waits]
    if any(v is None for v in pendings) or [v.group(1) for v in pendings] != ["1", "0"]:
        return "unexpected paired wait protocol"
    return None


def prepare(previous, report, tile32=False):
    folder = previous / (CASE + "-replay")
    old = json.loads((folder / "spec.json").read_text())
    ir = folder / (NAME + ".ttir")
    metadata = folder / (NAME + ".json")
    ptx = folder / (NAME + ".ptx")
    if (trace.sha(ir), trace.sha(metadata), trace.sha(ptx)) != (
        IR_SHA,
        OPTIONS_SHA,
        PTX_SHA,
    ):
        raise RuntimeError("Captured original input/options/PTX identity mismatch")
    if old["ir_sha256"] != IR_SHA or old["metadata_sha256"] != OPTIONS_SHA:
        raise RuntimeError("Captured provenance mismatch")
    options = json.loads(metadata.read_text())
    expected = dict(num_warps=4, num_stages=2, num_ctas=1)
    if any(options[k] != v for k, v in expected.items()):
        raise RuntimeError("Launch configuration changed")
    jobs = []
    for kind in (
        ("production", "sequential32", "paired32")
        if tile32
        else ("production", "paired")
    ):
        dest = report / kind
        dest.mkdir()
        module = dest / "kernel.py"
        module.write_text(
            source("paired" if kind.startswith("paired") else "production")
        )
        sig, fixed, attrs = abi(ir.read_text())
        if kind.endswith("32"):
            fixed["TILE_R"] = 32
        (dest / "options.json").write_bytes(metadata.read_bytes())
        job = dict(
            kind=kind,
            directory=str(dest),
            module=str(module),
            module_sha256=trace.sha(module),
            runtime=old["runtime"],
            metadata=str(dest / "options.json"),
            metadata_sha256=OPTIONS_SHA,
            signature=sig,
            constants=fixed,
            attrs=[[list(k), v] for k, v in attrs.items()],
            control_code_sha256=compiler.double.runtime.code_sha(ptx.read_text()),
        )
        trace.save(dest / "spec.json", job)
        jobs.append(job)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--flagtree", default="/opt/flagtree")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--worker-spec")
    parser.add_argument(
        "--tile32",
        action="store_true",
        help="R64 source control plus R32 sequential/paired controls",
    )
    args = parser.parse_args()
    if args.worker_spec:
        return compiler.child(args)
    report = Path(args.report_dir).resolve()
    report.mkdir(parents=True, exist_ok=True)
    summary = dict(
        status="preparing",
        production_modified=False,
        gpu_kernel_executed=False,
        timings={},
        jobs=[],
    )
    try:
        trace.save(
            report / "sources.json",
            {
                n: dict(text=(ROOT / n).read_text(), sha256=trace.sha(ROOT / n))
                for n in dict.fromkeys(SOURCES)
            },
        )
        for job in prepare(Path(args.source_report).resolve(), report, args.tile32):
            dest = Path(job["directory"])
            code = trace.run_job(
                [
                    sys.executable,
                    "-I",
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                    "--worker-spec",
                    str(dest / "spec.json"),
                ],
                dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1"),
                dest / "compiler.txt",
                args.timeout,
            )
            if code:
                print((dest / "compiler.txt").read_text()[-12000:], flush=True)
                raise RuntimeError("Source compile failed: " + job["kind"])
            result = json.loads((dest / "result.json").read_text())
            result.update(
                compiler.producer.resources((dest / "compiler.txt").read_text())
            )
            result["instructions"] = metrics(
                (dest / "ttgir.txt").read_text(), (dest / "ptx.txt").read_text()
            )
            trace.save(dest / "result.json", result)
            summary["jobs"].append(dict(kind=job["kind"], result=result))
            print(job["kind"], json.dumps(result, indent=2), flush=True)
            if job["kind"] == "production":
                if (
                    result["code_sha256"] != job["control_code_sha256"]
                    or result["registers_per_thread"] != 158
                    or result["shared_bytes"] != 57344
                    or result["spill_store_bytes"]
                    or result["spill_load_bytes"]
                ):
                    raise RuntimeError(
                        "Source control did not reproduce original executable/resources"
                    )
            elif not args.tile32:
                reason = gate(summary["jobs"][0]["result"], result)
                summary.update(
                    status=(
                        "not_promising" if reason else "pending_lifetime_and_accuracy"
                    ),
                    reason=reason,
                )
                print("Decision:", summary["status"], "reason:", reason, flush=True)
        if args.tile32:
            original, sequential, paired = [j["result"] for j in summary["jobs"]]
            reason = tile32_gate(original, sequential, paired)
            summary.update(
                status="not_promising" if reason else "pending_lifetime_and_accuracy",
                reason=reason,
                tile32=True,
                resource_delta_vs_sequential={
                    k: paired[k] - sequential[k]
                    for k in ("registers_per_thread", "shared_bytes")
                },
            )
            print(
                "R32 paired minus sequential resources:",
                summary["resource_delta_vs_sequential"],
                flush=True,
            )
            print("Decision:", summary["status"], "reason:", reason, flush=True)
        return 0
    except Exception:
        summary.update(status="failed", error=traceback.format_exc())
        print(summary["error"], flush=True)
        return 1
    finally:
        trace.save(report / "fp8-dot-issue-summary.json", summary)
        with tarfile.open(report / "fp8-dot-issue-artifacts.tar.gz", "w:gz") as archive:
            for kind in ("production", "paired", "sequential32", "paired32"):
                folder = report / kind
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
