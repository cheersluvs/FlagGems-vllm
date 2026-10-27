# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Measure source-level paired FP8 paths on all10 fixtures,including compact."""

import ast
import copy
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qsa_prefill_fp8_dot_full as full  # noqa: E402

runtime = full.runtime
WORKER, COORDINATE, VERIFY = full.WORKER, full.COORDINATE, full.verify_source
SOURCES = {
    "tools/qsa_prefill_fp8_dot_all.py",
    "tools/test_qsa_prefill_fp8_dot_all.py",
    "tools/run_qsa_prefill_fp8_dot_all_h800.sh",
    "docs/qsa_prefill_fp8_dot_all.md",
}


def target(trace):
    c = trace["constants"]
    if trace["route"] == "compact":
        return c["TILE_R"] == 32 and c["num_warps"] == 4 and c["BLOCK_N"] == 64
    return (
        trace["route"] == "fp8"
        and c["num_warps"] == 4
        and c["num_ctas"] == 1
        and c["BLOCK_N"] == 64
        and c["TILE_R"] >= 32
        and (trace["offset"] == 0 or (c["TILE_R"] == 64 and c["K_TILES"] == 16))
    )


def candidate_traces(traces):
    result = copy.deepcopy(traces)
    for trace in result:
        if target(trace) and trace["route"] == "fp8":
            trace["constants"].update(TILE_R=32, QUERY_PARTS=2)
    return result


def compact_source(paired):
    path = ROOT / runtime.probe.compiler.OP
    if runtime.base.sha(path) != runtime.probe.compiler.PRODUCTION_SHA:
        raise RuntimeError("Compact production source changed")
    text = path.read_text()
    fn = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_qsa_prefill_compact_score_kernel"
    )
    original = ast.get_source_segment(text, fn)
    source = original[: original.index("        scores = tl.dot(query, keys,")]
    anchor = "    column_lanes = tl.arange(0, BLOCK_N)"
    split = """    HALF_R: tl.constexpr = BLOCK_R // 2
    HALF_M: tl.constexpr = HALF_R * HEADS_PADDED
    query_left, query_right = tl.split(
        tl.permute(tl.reshape(query, (2, HALF_M, BLOCK_D)), (1, 2, 0))
    )
    visible_left, visible_right = tl.split(tl.trans(tl.reshape(visible, (2, HALF_R))))
"""
    source = source.replace(anchor, split + anchor)
    left = "        scores_left = tl.dot(query_left, keys, out_dtype=tl.float32)\n"
    right = "        scores_right = tl.dot(query_right, keys, out_dtype=tl.float32)\n"
    source += left + (right if paired else "")
    for side, shift in (("left", "0"), ("right", "HALF_R")):
        if side == "right" and not paired:
            source += right
        source += f"""        scores_{side} = tl.reshape(scores_{side}, (HALF_R, HEADS_PADDED, BLOCK_N))
        real_heads_{side} = tl.arange(0, HEADS_PADDED) < NUM_HEADS
        scores_{side} = tl.where(real_heads_{side}[None, :, None], tl.maximum(scores_{side}, 0.0), 0.0)
        scores_{side} = tl.sum(scores_{side}, axis=1)
        rows_{side} = row_start + {shift} + tl.arange(0, HALF_R)
        tl.store(
            logits_ptr + (rows_{side} - query_offset)[:, None] * logits_width + columns[None, :],
            tl.where(live[None, :], scores_{side}, -float("inf")),
            mask=(rows_{side}[:, None] < request_end)
            & (columns[None, :] < visible_{side}[:, None])
            & (columns[None, :] < logits_width),
        )
"""
    return (
        "import triton\nimport triton.language as tl\n\n"
        '@triton.jit(do_not_specialize=["num_rows", "query_offset", "max_query_len"])\n'
        + source
    )


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dot_shapes(ir):
    shapes = []
    for line in ir.splitlines():
        if re.search(r"\bttng.warp_group_dot\b", line):
            match = re.search(r"-> tensor<(\d+)x(\d+)xf32", line)
            if match is None or "xf8E4M3FN" not in line:
                raise RuntimeError("Unrecognized FP8 matrix operation")
            shapes.append(tuple(map(int, match.groups())))
    return shapes


def compile_gate(control, candidate):
    shapes = dot_shapes(control.asm["ttgir"])
    if len(shapes) != 2 or shapes != dot_shapes(candidate.asm["ttgir"]):
        raise RuntimeError("Paired source changes same-config matrix work")
    for kernel in (control, candidate):
        if not all(hasattr(kernel, k) for k in ("n_regs", "n_spills")):
            kernel._init_handles()
        if (
            not isinstance(kernel.n_regs, int)
            or kernel.n_regs <= 0
            or kernel.n_spills != 0
        ):
            raise RuntimeError("New specialization spills;not launched")
    clean = runtime.fp8.runtime.program(candidate.asm["ptx"])
    waits = re.findall(r"\bwgmma.wait_group.sync.aligned (\d+);", clean)
    if waits not in (["1", "0"], ["0", "0"]):
        raise RuntimeError("Unreviewed final WGMMA wait protocol;not launched")
    return dict(
        dot_shapes=shapes,
        waits=waits,
        overlap_retained=waits == ["1", "0"],
        changed_executable=runtime.fp8.runtime.code_sha(control.asm["ptx"])
        != runtime.fp8.runtime.code_sha(candidate.asm["ptx"]),
        registers=candidate.n_regs,
        shared=candidate.metadata.shared,
        spills=candidate.n_spills,
    )


class SourceJIT:
    def __init__(
        self, original, candidate, names, targets, legacy, control=None, compact=False
    ):
        self.original, self.candidate, self.names = original, candidate, names
        self.control, self.compact = control or original, compact
        self.targets, self.legacy, self.prepared, self.frozen = (
            targets,
            legacy,
            {},
            False,
        )

    def __getitem__(self, grid):
        def launch(*values, **kwargs):
            named = dict(zip(self.names, values), **kwargs)
            key = runtime.base.fixed.signature(named)
            if key not in self.targets:
                return self.original[grid](*values, **kwargs)
            if key not in self.prepared:
                if self.frozen or kwargs.get("ir_override") is not None:
                    raise RuntimeError(
                        "Unprepared specialization or forbidden IR override"
                    )
                if not (
                    named["IS_FP8"]
                    and named["NUM_HEADS"] == 4
                    and named["HEAD_DIM"] == 128
                    and named["TILE_R"] == 32
                    and (self.compact or named["QUERY_PARTS"] == 2)
                ):
                    raise RuntimeError("Unexpected source candidate domain")
                control = self.control.warmup(*values, grid=grid, **kwargs)
                kernel = self.candidate.warmup(*values, grid=grid, **kwargs)
                from tools.qsa_prefill_kernel_resources import compiled_record

                folder = Path(os.environ["QSA_DOT_ALL_REPORT"])
                prefix = (
                    os.environ["QSA_DOT_ALL_CASE"]
                    + "-prelaunch-"
                    + "-".join(map(str, key))
                )
                files = {}
                for side, compiled in (("control", control), ("paired", kernel)):
                    for suffix in ("ptx", "ttgir", "cubin"):
                        data = compiled.asm[suffix]
                        path = folder / (prefix + "-" + side + "." + suffix)
                        path.write_bytes(
                            data if isinstance(data, bytes) else data.encode()
                        )
                        files[side + "." + suffix] = dict(
                            path=path.name, sha256=runtime.base.sha(path)
                        )
                evidence = compile_gate(control, kernel)
                if (
                    named["num_requests"] == 1
                    and named["K_TILES"] == 16
                    and named["num_stages"] == 2
                ):
                    runtime.check_kernel(kernel, self.legacy)
                    evidence["legacy_executable_verified"] = True
                self.prepared[key] = dict(
                    evidence=evidence,
                    candidate=compiled_record(kernel),
                    control=compiled_record(control),
                    files=files,
                )
            return self.candidate[grid](*values, **kwargs)

        return launch


def make_modules(production, traces):
    capture = runtime.proof()["paired32"]
    source = load_source(capture["source"], "qsa_all_fp8_paired")
    targets = {
        (t["rows"], t["offset"], t["width"])
        for t in traces
        if target(t) and t["route"] == "fp8"
    }
    compact_targets = {
        (t["rows"], t["offset"], t["width"])
        for t in traces
        if target(t) and t["route"] == "compact"
    }
    compact_jits = []
    if compact_targets:
        folder = Path(os.environ["QSA_DOT_ALL_REPORT"])
        for paired in (False, True):
            name = "qsa_compact_dot_" + ("paired" if paired else "sequential")
            path = folder / (name + ".py")
            generated = compact_source(paired)
            if path.exists() and path.read_text() != generated:
                raise RuntimeError("Generated compact source changed")
            path.write_text(generated)
            compact_jits.append(
                load_source(path, name)._qsa_prefill_compact_score_kernel
            )
    modules = []
    for kind in ("production", "paired32"):
        chosen = traces if kind == "production" else candidate_traces(traces)
        with patch.object(
            runtime.stage,
            "fixed_traces",
            lambda ts, candidate=False: copy.deepcopy(chosen),
        ):
            module = runtime.stage.make_pair(production, traces)[0]
        module._dot_kind = kind
        if kind == "paired32":
            entry = module._qsa_prefill_fp8_store_score_kernel
            entry.jit = SourceJIT(
                entry.jit,
                source._qsa_prefill_score_kernel,
                entry.arg_names,
                targets,
                capture["result"],
            )
            if compact_targets:
                entry = module._qsa_prefill_compact_score_kernel
                entry.jit = SourceJIT(
                    entry.jit,
                    compact_jits[1],
                    entry.arg_names,
                    compact_targets,
                    capture["result"],
                    control=compact_jits[0],
                    compact=True,
                )
        modules.append(module)
    if modules[0].persistent_topk is not modules[1].persistent_topk:
        raise RuntimeError("TopK callable changed")
    return modules


def freeze_records(modules):
    from tools.qsa_prefill_kernel_resources import compiled_record

    records = {m._dot_kind: runtime.stage.FREEZE((m, m))["baseline"] for m in modules}
    baseline = {tuple(r["chunk"]): r for r in records["production"]}
    entries = {
        "fp8": modules[1]._qsa_prefill_fp8_store_score_kernel,
        "compact": modules[1]._qsa_prefill_compact_score_kernel,
    }
    for row in records["paired32"]:
        key = tuple(row["chunk"])
        old = baseline[key]
        entry = entries.get(row["route"])
        proxy = getattr(entry, "jit", None)
        if isinstance(proxy, SourceJIT) and key in proxy.targets:
            if key not in proxy.prepared:
                raise RuntimeError("Required candidate chunk was not executed")
            receipt = proxy.prepared[key]
            actual = compiled_record(entry.compiled[key][0])
            expected = (
                old["constants"]
                if row["route"] == "compact"
                else dict(old["constants"], TILE_R=32, QUERY_PARTS=2)
            )
            if actual != receipt["candidate"] or row["constants"] != expected:
                raise RuntimeError("Candidate executable or configuration changed")
            row["applied_source_receipt"] = receipt
        elif any(
            row[k] != old[k]
            for k in ("constants", "route", "ptx_sha256", "cubin_sha256")
        ):
            raise RuntimeError("Unchanged route binary changed")
        else:
            row["unchanged_binary_verified"] = True
    for entry in entries.values():
        if isinstance(entry.jit, SourceJIT):
            entry.jit.frozen = True
    return records


def verify_source():
    return dict(
        VERIFY(),
        candidate_scope="all10 FP8 main paths;compact uses compact-preserving split",
        coverage_policy="all10 paths actually execute paired source",
        extended_source_gates=True,
    )


def worker(options):
    os.environ.update(
        QSA_DOT_ALL_REPORT=options.report_dir, QSA_DOT_ALL_CASE=options.worker_case
    )
    code = WORKER(options)
    path = Path(options.report_dir) / (options.worker_case + ".json")
    if path.is_file():
        value = json.loads(path.read_text())
        rows = value.get("kernels_before", {}).get("paired32", [])
        applied = [r for r in rows if "applied_source_receipt" in r]
        compact = any(r["route"] == "compact" for r in rows)
        value["optimization_coverage"] = dict(
            applicable=True,
            applied_chunks=len(applied),
            retained_overlap_chunks=sum(
                r["applied_source_receipt"]["evidence"]["overlap_retained"]
                for r in applied
            ),
            reason=(
                "compact-preserving split/paired source actually executed"
                if compact
                else "paired source actually executed"
            ),
        )
        if value.get("status") == "passed" and not applied:
            value.update(
                status="failed", error="No candidate executed for an applicable fixture"
            )
            code = 1
        path.write_text(json.dumps(value, indent=2) + "\n")
        print(
            "Actual optimization coverage:", value["optimization_coverage"], flush=True
        )
    return code


def coordinate(options):
    code = COORDINATE(options)
    root = Path(options.report_dir)
    old = root / "fp8-dot-runtime-summary.json"
    if old.is_file():
        import statistics

        value = json.loads(old.read_text())
        applicable = []
        for row in value.get("cases", []):
            path = root / (row["case"] + ".json")
            if path.is_file():
                coverage = json.loads(path.read_text()).get("optimization_coverage", {})
                row["optimization_coverage"] = coverage
                if row.get("status") == "passed" and coverage.get("applicable"):
                    applicable.append(row)
        value["applicable_mean"] = (
            dict(
                count=len(applicable),
                **{
                    k: statistics.mean(r["stages"][k]["speedup"] for r in applicable)
                    for k in ("score", "topk", "pipeline")
                },
            )
            if applicable
            else {}
        )
        (root / "fp8-dot-all-summary.json").write_text(
            json.dumps(value, indent=2) + "\n"
        )
        old.unlink()
        print(
            "Actual source-candidate cases mean:",
            value["applicable_mean"],
            flush=True,
        )
    return code


def main():
    with patch.object(full, "WORKER", worker), patch.object(
        full, "COORDINATE", coordinate
    ), patch.object(full, "verify_source", verify_source), patch.object(
        runtime, "make_modules", make_modules
    ), patch.object(
        runtime, "freeze_records", freeze_records
    ), patch.object(
        runtime, "EXTRA", runtime.EXTRA | SOURCES
    ), patch.object(
        full, "__file__", str(Path(__file__).resolve())
    ), patch.object(
        full, "__doc__", __doc__
    ):
        return full.main()


if __name__ == "__main__":
    sys.exit(main())
