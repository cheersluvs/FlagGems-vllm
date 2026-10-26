# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Proof-gated two-slot K-ring compilation; no GPU kernel execution."""

import argparse
import importlib
import json
import os
import re
import sys
import tarfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import qsa_prefill_producer_first as producer  # noqa: E402
from tools import qsa_prefill_producer_runtime as runtime  # noqa: E402

trace = producer.trace
CASES = trace.CASES
SOURCES = (
    "tools/qsa_prefill_double_buffer.py",
    "tools/qsa_prefill_double_buffer_runtime.py",
    "tools/test_qsa_prefill_double_buffer.py",
    "tools/run_qsa_prefill_double_buffer.sh",
    "docs/qsa_prefill_double_buffer.md",
    "tools/qsa_prefill_producer_first.py",
    "tools/qsa_prefill_producer_runtime.py",
    "tools/qsa_prefill_pipeline_trace.py",
)
EXPECTED = dict(
    zip(
        CASES,
        (
            "f97e0e7432987952d66cb4c2f287a4d24de9738a83fb74a23d1658e574c468b2",
            "2e48aab674167c1b46555dd2162da29eff0212bb7f1ddc63b47fe59eca43b7f9",
        ),
    )
)
SSA = r"%[\w.$-]+"


def model(tiles, groups, slots=2, reuse_barrier=True):
    """Abstract per-group copy/wait/CTA-sync/read/wait/reuse-sync protocol.

    A current-copy wait publishes that group's fragment; the pre-MMA barrier
    publishes the whole tile. DMA may overwrite as soon as issued, so issue
    itself must be safe. No hardware latency or weak-memory timing is simulated.
    """
    initial = (
        (0,) * groups,
        tuple((0,) + (-1,) * (slots - 1) for _ in range(groups)),
        tuple((-1,) * slots for _ in range(groups)),
    )
    stack = [initial]
    parents = {initial: None}

    def failure(state, event):
        witness = [event]
        while parents[state] is not None:
            state, previous = parents[state]
            witness.append(previous)
        return dict(safe=False, states=len(parents), witness=list(reversed(witness)))

    while stack:
        state = stack.pop()
        pcs, resident, ready = state
        for group, pc in enumerate(pcs):
            if pc == tiles * 7:
                continue
            iteration, step = divmod(pc, 7)
            position = list(pcs)
            memory = [list(v) for v in resident]
            published = [list(v) for v in ready]
            event = f"g{group}:tile{iteration}:step{step}"
            if step in (2, 5) and (step == 2 or reuse_barrier):
                if group or not all(v == pc for v in pcs):
                    continue
                position = [v + 1 for v in pcs]
                event = f"CTA:tile{iteration}:barrier{step}"
            else:
                position[group] += 1
                slot = iteration % slots
                if step == 0:
                    destination = (iteration + 1) % slots
                    readers = [
                        (v // 7) % slots for v in pcs if v < tiles * 7 and v % 7 == 4
                    ]
                    if destination in readers:
                        return failure(state, event + ":overwrite_active_reader")
                    memory[group][destination] = iteration + 1
                    published[group][destination] = -1
                elif step == 1:
                    if memory[group][slot] != iteration:
                        return failure(state, event + ":current_fragment_overwritten")
                    published[group][slot] = iteration
                elif step == 3 and any(v[slot] != iteration for v in published):
                    return failure(state, event + ":unpublished_current_tile")
            nxt = (
                tuple(position),
                tuple(tuple(v) for v in memory),
                tuple(tuple(v) for v in published),
            )
            if nxt not in parents:
                parents[nxt] = (state, event)
                stack.append(nxt)
    return dict(safe=True, states=len(parents))


def protocol_model():
    rows = [dict(tiles=n, groups=g, **model(n, g)) for g in (1, 2) for n in range(17)]
    if not all(r["safe"] for r in rows):
        raise RuntimeError("Two-slot protocol model failed")
    negative = model(2, 2, reuse_barrier=False)
    if (
        negative["safe"]
        or not model(16, 2, slots=3, reuse_barrier=False)["safe"]
        or model(1, 1, slots=1)["safe"]
    ):
        raise RuntimeError("Protocol negative controls failed")
    return dict(
        status="passed",
        scope="abstract fragment-lifetime model,not hardware numerical validation",
        checks=rows,
        states=sum(v["states"] for v in rows),
        missing_reuse_barrier_counterexample=negative,
    )


def ptx_protocol(ptx):
    lines = runtime.program(ptx).splitlines()
    labels = {
        line[:-1]: i for i, line in enumerate(lines) if re.fullmatch(r"[\w.$]+:", line)
    }
    edges = []
    for i, line in enumerate(lines):
        m = re.fullmatch(r"(@!?%\w+\s+)?bra(?:\.uni)?\s+([\w.$]+);", line)
        if m and m[2] in labels and labels[m[2]] < i:
            edges.append((labels[m[2]], i, m[1]))
    if len(edges) != 1 or not edges[0][2]:
        raise RuntimeError("Unsupported PTX loop CFG")
    start, end, _ = edges[0]
    body = lines[start + 1 : end]
    if any(re.search(r"\bbra(?:\.uni)?\b", v) for v in body):
        raise RuntimeError("Internal branch in reuse protocol")
    mma = [i for i, v in enumerate(body) if re.search(r"\bwgmma.mma_async\.", v)]
    waits = [
        (i, int(m[1]))
        for i, v in enumerate(body)
        if (m := re.fullmatch(r"wgmma.wait_group.sync.aligned (\d+);", v))
    ]
    bars = [i for i, v in enumerate(body) if v == "bar.sync 0;"]
    copies = []
    for i, v in enumerate(body):
        m = re.match(
            r"(?:@!?%\w+ )?cp.async.(?:ca|cg).shared.global \[[^]]+\], \[[^]]+\], (0x[\da-fA-F]+|\d+)(?:,|;)",
            v,
        )
        if m and int(m[1], 0) == 16:
            copies.append(i)
    if not mma or not waits or not copies or max(copies) >= min(mma):
        raise RuntimeError("No early K-copy protocol")
    if (
        waits[-1][1] != 0
        or waits[-1][0] < max(mma)
        or not any(b > waits[-1][0] for b in bars)
    ):
        raise RuntimeError("Missing post-WGMMA complete-CTA reuse barrier")
    copywait = [
        i
        for i, v in enumerate(body)
        if v == "cp.async.wait_group 1;" and max(copies) < i < min(mma)
    ]
    if len(copywait) != 1 or not any(copywait[0] < b < min(mma) for b in bars):
        raise RuntimeError("Missing current-copy wait/pre-MMA CTA barrier")
    tail = lines[end + 1 :]
    if (
        "cp.async.wait_group 0;" not in tail
        or "bar.sync 0;" not in tail[tail.index("cp.async.wait_group 0;") + 1 :]
    ):
        raise RuntimeError("Missing epilogue DMA drain")
    return dict(
        status="passed",
        loop_header_line=start + 1,
        loop_backedge_line=end + 1,
        mma_sites=len(mma),
        post_WGMMA_CTA_barriers=sum(b > waits[-1][0] for b in bars),
        current_copy_wait=1,
        epilogue_drain=True,
    )


def two_slots(text):
    lines = text.splitlines(keepends=True)
    defs = {}
    for i, line in enumerate(lines):
        m = re.match(r"\s*(%[\w.$-]+)(?::\d+)? = (.*)", line)
        if m:
            defs[m[1]] = (i, m[2])
    loops = [i for i, l in enumerate(lines) if re.search(r"\bscf.for\b", l)]
    if len(loops) != 1:
        raise RuntimeError("Requires one captured score loop")
    start = loops[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = next(
        i
        for i in range(start + 1, len(lines))
        if lines[i].startswith(" " * indent + "}")
    )
    allocations = [
        v
        for v, (i, rhs) in defs.items()
        if "ttg.local_alloc" in rhs and "memdesc<3x64x128xbf16" in rhs
    ]
    if len(allocations) != 1 or "memdesc<2x64x128xbf16" in text:
        raise RuntimeError("Ambiguous K ring")
    root = allocations[0]
    uses = [line for line in lines if re.search(re.escape(root) + r"(?![\w.$-])", line)]
    views = {
        v: re.search(r"ttg.memdesc_index " + re.escape(root) + r"\[(%[\w.$-]+)\]", rhs)[
            1
        ]
        for v, (i, rhs) in defs.items()
        if re.search(r"ttg.memdesc_index " + re.escape(root) + r"\[", rhs)
    }
    if len(views) != 3 or len(uses) != 5 or text.count("memdesc<3x64x128xbf16") != 5:
        raise RuntimeError("K allocation escapes expected three views/dealloc")
    phi = dict(
        re.findall(
            r"(%[\w.$-]+) = (%[\w.$-]+)",
            lines[start].split("iter_args(")[1].split(") ->")[0],
        )
    )
    yield_line = next(
        line
        for line in reversed(lines[start + 1 : end])
        if line.strip().startswith("scf.yield")
    )
    yielded = re.findall(SSA, yield_line.split(" : ")[0])

    def constant(value):
        m = re.match(r"arith.constant (-?\d+) : i32", defs[value][1])
        if not m:
            raise RuntimeError("Nonconstant ring phase")
        return int(m[1])

    def slot(index, expected_initial):
        select = re.match(
            r"arith.select (%[\w.$-]+), (%[\w.$-]+), (%[\w.$-]+)", defs[index][1]
        )
        if not select or constant(select[2]) != 0:
            raise RuntimeError("Unsupported wrap select")
        comparison = re.match(
            r"arith.cmpi sge, (%[\w.$-]+), (%[\w.$-]+)", defs[select[1]][1]
        )
        add = re.match(r"arith.addi (%[\w.$-]+), (%[\w.$-]+)", defs[select[3]][1])
        if (
            not comparison
            or not add
            or comparison[1] != select[3]
            or constant(comparison[2]) != 3
            or constant(add[2]) != 1
        ):
            raise RuntimeError("Unsupported modulo counter")
        if (
            add[1] not in phi
            or constant(phi[add[1]]) != expected_initial
            or yielded[list(phi).index(add[1])] != index
        ):
            raise RuntimeError("Counter init/yield differs")
        return dict(
            index=index,
            compare=select[1],
            bound=comparison[2],
            initial=expected_initial,
        )

    copies = [
        (i, v, rhs)
        for v, (i, rhs) in defs.items()
        if "ttg.async_copy_global_to_local" in rhs
        and "tensor<64x128x!tt.ptr<bf16>" in rhs
    ]
    main = [(i, v, rhs) for i, v, rhs in copies if start < i < end]
    prologue = [(i, v, rhs) for i, v, rhs in copies if i < start]
    if len(main) != 1 or len(prologue) != 1 or len(copies) != 2:
        raise RuntimeError("Unsupported K copy history")
    copy_at, copy_token, copy_rhs = main[0]
    destination = re.match(
        r"ttg.async_copy_global_to_local (%[\w.$-]+), (%[\w.$-]+)", copy_rhs
    )[2]
    dots = [
        (i, rhs)
        for v, (i, rhs) in defs.items()
        if start < i < end and re.match(r"ttng.warp_group_dot\b", rhs)
    ]
    if not dots or copy_at >= dots[0][0]:
        raise RuntimeError("Producer copy is not early")
    read = re.match(r"ttng.warp_group_dot (%[\w.$-]+),", dots[0][1])[1]
    if destination not in views or read not in views:
        raise RuntimeError("Indirect K view is unsupported")
    counters = [slot(views[destination], 0), slot(views[read], -1)]
    commits = [
        v
        for v, (i, rhs) in defs.items()
        if re.match(
            r"ttg.async_commit_group tokens " + re.escape(copy_token) + r"\b", rhs
        )
    ]
    if len(commits) != 1:
        raise RuntimeError("Ambiguous copy commit")
    tokens = [v for v in phi if yielded[list(phi).index(v)] == commits[0]]
    if len(tokens) != 1 or not any(
        re.match(r"ttg.async_wait " + re.escape(tokens[0]) + r" \{num = 1 : i32\}", rhs)
        for v, (i, rhs) in defs.items()
        if copy_at < i < dots[0][0]
    ):
        raise RuntimeError("Current-copy wait token changed")
    initial_commit = defs[phi[tokens[0]]][1]
    if not re.match(
        r"ttg.async_commit_group tokens " + re.escape(prologue[0][1]) + r"\b",
        initial_commit,
    ):
        raise RuntimeError("Prologue copy token mismatch")
    prologue_view = re.match(
        r"ttg.async_copy_global_to_local (%[\w.$-]+), (%[\w.$-]+)", prologue[0][2]
    )[2]
    if prologue_view not in views or constant(views[prologue_view]) != 0:
        raise RuntimeError("Initial K slot is not zero")
    symbol = "%qsa_ring_two"
    if symbol in text:
        raise RuntimeError("Reserved proof symbol collision")
    replacements = {}
    for counter in counters:
        i = defs[counter["compare"]][0]
        replacements[i] = lines[i].replace(counter["bound"], symbol)
    if len(replacements) != 2:
        raise RuntimeError("Ring counters unexpectedly merged")
    changed = []
    for i, line in enumerate(lines):
        if i == start:
            changed.append(" " * indent + symbol + " = arith.constant 2 : i32\n")
        changed.append(
            replacements.get(i, line).replace(
                "memdesc<3x64x128xbf16", "memdesc<2x64x128xbf16"
            )
        )
    result = "".join(changed)
    recovered = result.replace(
        " " * indent + symbol + " = arith.constant 2 : i32\n", ""
    ).replace("memdesc<2x64x128xbf16", "memdesc<3x64x128xbf16")
    recovery = recovered.splitlines(keepends=True)
    for i in replacements:
        recovery[i] = lines[i]
    if "".join(recovery) != text:
        raise RuntimeError("Rewrite changed non-ring instructions")
    return result, dict(
        status="proved_under_checked_PTX_protocol",
        ring=3,
        new_ring=2,
        counters=counters,
        local_type_replacements=5,
        global_addresses_masks_and_sync_unchanged=True,
    )


def prepare(source, report):
    summary = json.loads((source / "producer-first-summary.json").read_text())
    if (
        summary["status"] != "compile_feasible_pending_accuracy"
        or summary["gpu_kernel_executed"]
        or len(summary["jobs"]) != 4
    ):
        raise RuntimeError("Requires the validated three-buffer report")
    captured = json.loads((source / "sources.json").read_text())
    if set(captured) != set(producer.SOURCES) or any(
        trace.sha(ROOT / n) != v["sha256"] for n, v in captured.items()
    ):
        raise RuntimeError("Captured producer source identity changed")
    jobs = []
    for case in CASES:
        old = source / (case + "-producer")
        spec = json.loads((old / "spec.json").read_text())
        record = json.loads((old / "result.json").read_text())
        saved = next(j for j in summary["jobs"] if j["name"] == old.name)
        if (
            saved["exit_code"]
            or saved["result"] != record
            or record["ptx_sha256"] != EXPECTED[case]
            or trace.sha(old / "ptx.txt") != EXPECTED[case]
            or trace.sha(old / "kernel.cubin") != record["cubin_sha256"]
        ):
            raise RuntimeError("Captured producer binary changed")
        constants = spec["original_resources"]["constants"]
        if constants["K_TILES"] != 16 or constants["num_warps"] not in (4, 8):
            raise RuntimeError("Unsupported proof scope")
        source_protocol = ptx_protocol((old / "ptx.txt").read_text())
        original = (old / "ttgir.txt").read_text()
        changed, proof = two_slots(original)
        metadata = old / Path(spec["metadata"]).name
        if trace.sha(metadata) != spec["metadata_sha256"]:
            raise RuntimeError("Captured options changed")
        for kind, text in (("control", original), ("two", changed)):
            folder = report / (case + "-" + kind)
            folder.mkdir()
            ir = folder / "input.ttgir"
            ir.write_text(text)
            opts = folder / "options.json"
            opts.write_bytes(metadata.read_bytes())
            job = dict(
                spec,
                name=folder.name,
                kind=kind,
                directory=str(folder),
                ir=str(ir),
                ir_sha256=trace.sha(ir),
                metadata=str(opts),
                metadata_sha256=trace.sha(opts),
                expected_ptx_sha256=EXPECTED[case],
                original_resources=record,
                lifetime_proof=proof,
                original_PTX_protocol=source_protocol,
            )
            trace.save(folder / "spec.json", job)
            jobs.append(job)
    return jobs


def child(options):
    job = json.loads(Path(options.worker_spec).read_text())
    folder = Path(job["directory"])
    os.environ.update(
        TRITON_ALWAYS_COMPILE="1",
        TRITON_DUMP_PTXAS_LOG="1",
        TRITON_CACHE_DIR=str(folder / "cache"),
        MLIR_ENABLE_DUMP="0",
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
        raise RuntimeError("Runtime/assembler identity mismatch")
    if (
        trace.sha(job["ir"]) != job["ir_sha256"]
        or trace.sha(job["metadata"]) != job["metadata_sha256"]
    ):
        raise RuntimeError("Proof-gated input changed")
    metadata = json.loads(Path(job["metadata"]).read_text())
    target = metadata["target"]
    if target != dict(backend="cuda", arch=90, warp_size=32):
        raise RuntimeError("Unsupported target")
    kernel = triton.compile(
        job["ir"], target=GPUTarget(**target), options=trace.restore_options(metadata)
    )
    for name in ("ptx", "ttgir", "llir"):
        if name in kernel.asm:
            (folder / (name + ".txt")).write_text(kernel.asm[name])
    (folder / "kernel.cubin").write_bytes(kernel.asm["cubin"])
    result = dict(
        status="compiled",
        gpu_kernel_executed=False,
        ptx_sha256=trace.sha(folder / "ptx.txt"),
        cubin_sha256=trace.sha(folder / "kernel.cubin"),
        shared_bytes=kernel.metadata.shared,
        schedule=producer.inspect(kernel.asm["ttgir"], kernel.asm["ptx"]),
        PTX_protocol=ptx_protocol(kernel.asm["ptx"]),
    )
    if job["kind"] == "control" and result["ptx_sha256"] != job["expected_ptx_sha256"]:
        result["status"] = "control_mismatch"
    trace.save(folder / "result.json", result)
    return 0 if result["status"] == "compiled" else 2


def gate(control, candidate):
    reason = producer.gate(control, candidate)
    if reason:
        return reason
    if candidate["schedule"]["K_slots"] != 2:
        return "K ring did not reduce to two slots"
    if candidate["shared_bytes"] != control["shared_bytes"] - 16384:
        return "Shared reduction is not exactly one16KiB K tile"
    return None


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
        status="preparing", gpu_kernel_executed=False, jobs=[], decisions=[], timings={}
    )
    try:
        trace.save(
            report / "sources.json",
            {
                n: dict(text=(ROOT / n).read_text(), sha256=trace.sha(ROOT / n))
                for n in SOURCES
            },
        )
        proof = protocol_model()
        trace.save(report / "protocol-model.json", proof)
        print(
            "Protocol model:",
            len(proof["checks"]),
            "checks",
            proof["states"],
            "states;negative controls passed",
            flush=True,
        )
        jobs = prepare(Path(options.source_report).resolve(), report)
        for job in jobs:
            folder = Path(job["directory"])
            command = [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--worker-spec",
                str(folder / "spec.json"),
            ]
            code = trace.run_job(
                command,
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
                raise RuntimeError(
                    "Compilation/protocol/control failed: " + job["name"]
                )
            result.update(producer.resources((folder / "compiler.txt").read_text()))
            trace.save(folder / "result.json", result)
            print(
                job["name"],
                "K slots",
                result["schedule"]["K_slots"],
                "registers",
                result["registers_per_thread"],
                "shared",
                result["shared_bytes"],
                "spill bytes",
                result["spill_store_bytes"],
                result["spill_load_bytes"],
                "reuse barriers",
                result["PTX_protocol"]["post_WGMMA_CTA_barriers"],
                flush=True,
            )
            if job["kind"] == "two":
                reason = gate(summary["jobs"][-2]["result"], result)
                summary["decisions"].append(
                    dict(
                        case=job["name"].rsplit("-", 1)[0],
                        reason=reason,
                        status=(
                            "not_promising"
                            if reason
                            else "compile_feasible_pending_accuracy"
                        ),
                    )
                )
        summary["status"] = (
            "compile_feasible_pending_accuracy"
            if len(summary["decisions"]) == 2
            and all(v["reason"] is None for v in summary["decisions"])
            else "not_promising"
        )
        return 0
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        print(summary["error"], flush=True)
        return 1
    finally:
        trace.save(report / "double-buffer-summary.json", summary)
        with tarfile.open(report / "compile-artifacts.tar.gz", "w:gz") as archive:
            for folder in sorted(report.iterdir()):
                if folder.is_dir() and folder.name.startswith("perf_"):
                    for path in sorted(folder.iterdir()):
                        if path.is_file():
                            archive.add(
                                path,
                                arcname=str(path.relative_to(report)),
                                recursive=False,
                            )
            for name in ("sources.json", "protocol-model.json"):
                if (report / name).is_file():
                    archive.add(report / name, arcname=name, recursive=False)


if __name__ == "__main__":
    sys.exit(main())
