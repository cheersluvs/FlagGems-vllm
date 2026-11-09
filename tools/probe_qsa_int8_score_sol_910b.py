# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calibrate component throughput for the current Ascend INT8 scorer."""

import argparse
import ast
import collections
import functools
import gzip
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from probe_qsa_int8_page64_prefix_910b import load_generated_prefix
from probe_qsa_int8_ten_shapes_910b import PLANS, run_child
from profile_qsa_int8_score_910b import check_chunk
from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan

from tools.qsa_int8_ten_workloads_910b import make_inputs

CHUNKS = {"int8_budget_exact": (8192, 128), "int8_long100k": (8176, 112)}
MACS = 64 * 64 * 128
GRID = 20
UNITS = GRID * 16
TILES = UNITS * 8
COPY_BYTES = 64 * 1024**2
SYNC_DUMP_FLAG = "--bishengir-print-ir-after=hivm-graph-sync-solver"


class SyncDumpCompiler:
    """Intercept only the compiler module's final BiSheng call."""

    def __init__(self, report, case):
        self.report = report
        self.case = case
        self.calls = []

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def run(self, command, *args, **kwargs):
        if (
            not isinstance(command, (list, tuple))
            or not command
            or Path(command[0]).name != "bishengir-compile"
            or "--enable-triton-kernel-compile=true" not in command
            or "-o" not in command
        ):
            return subprocess.run(command, *args, **kwargs)
        if SYNC_DUMP_FLAG in command:
            raise RuntimeError(
                "Expected ordinary compiler options before adding IR dump flag"
            )
        captured = [*command, SYNC_DUMP_FLAG]
        record = dict(
            original_command=[str(x) for x in command], added_option=SYNC_DUMP_FLAG
        )
        self.calls.append(record)
        try:
            completed = subprocess.run(captured, *args, **kwargs)
        except subprocess.CalledProcessError as exc:
            self.save(exc, record)
            raise
        self.save(completed, record)
        return completed

    def save(self, process, record):
        record.update(exit_code=process.returncode, streams={})
        for name in ("stdout", "stderr"):
            stream = getattr(process, name)
            if not stream:
                continue
            data = stream.encode() if isinstance(stream, str) else stream
            stem = f"{self.case}-bisheng-{len(self.calls)}-{name}.mlir.gz"
            limit = 24 * 1024**2
            (self.report / stem).write_bytes(gzip.compress(data[:limit]))
            record["streams"][name] = dict(
                file=stem, bytes=len(data), truncated=len(data) > limit
            )
        (self.report / f"{self.case}-bisheng-invocations.json").write_text(
            json.dumps(self.calls, indent=2) + "\n"
        )


def summarize_sync_ir(path):
    after = 0
    active = False
    function = None
    counts = collections.Counter()
    functions = set()
    scalar_workspace_sites = collections.Counter()
    tile_workspace_shapes = collections.defaultdict(list)
    fixpipe_shapes = collections.defaultdict(list)
    table_reads = collections.Counter()
    table_refs = set()
    examples = collections.defaultdict(list)
    with gzip.open(path, "rt", errors="replace") as stream:
        for lineno, line in enumerate(stream, 1):
            if "IR Dump" in line:
                active = "IR Dump After" in line and "graphsyncsolver" in re.sub(
                    r"[^a-z]", "", line.lower()
                )
                if active:
                    after += 1
                    function = None
            if not active:
                continue
            if match := re.search(r"\bfunc\.func\s+(?:private\s+)?@([^\s(]+)", line):
                function = match[1]
                functions.add(function)
                # Table is arg8 after the three hidden workspace/sync arguments.
                table_refs = {"%arg8"}
            if "memref_ext.alloc_workspace" in line and "to memref<1xi32" in line:
                scalar_workspace_sites[function or "unknown"] += 1
            if "memref_ext.alloc_workspace" in line:
                if match := re.search(r"to memref<((?:\d+x)*64x(?:64|128)xi32),", line):
                    tile_workspace_shapes[function or "unknown"].append(match[1])
            if "hivm.hir.fixpipe " in line and "#hivm.address_space<gm>" in line:
                if match := re.search(
                    r"outs\(.*?memref<((?:\d+x)*64x(?:64|128)xi32),", line
                ):
                    fixpipe_shapes[function or "unknown"].append(match[1])
            if match := re.search(
                r"(%[\w.]+) = memref\.(?:reinterpret_cast|subview|cast) (%[\w.]+)\b",
                line,
            ):
                if match[2] in table_refs:
                    table_refs.add(match[1])
            if match := re.search(r"\bmemref\.load (%[\w.]+)\[", line):
                if match[1] in table_refs:
                    table_reads[function or "unknown"] += 1
            for token in re.findall(
                r"\b(?:hivm\.hir\.[\w.]+|memref\.alloca?|memref\.copy|memref_ext\.alloc_workspace)\b",
                line,
            ):
                if token.startswith(("memref.", "memref_ext.")) or re.search(
                    r"sync|flag|barrier|copy|load|store|fixp|mmad|matmul", token
                ):
                    counts[token] += 1
                    if len(examples[token]) < 8:
                        examples[token].append(
                            dict(
                                line=lineno, function=function, text=line.strip()[:1000]
                            )
                        )
    return dict(
        after_solver_dumps=after,
        functions=sorted(functions),
        operations=dict(counts),
        scalar_i32_workspace_sites=dict(scalar_workspace_sites),
        int32_tile_workspace_shapes=dict(tile_workspace_shapes),
        int32_fixpipe_destination_shapes=dict(fixpipe_shapes),
        table_scalar_reads=dict(table_reads),
        examples=dict(examples),
    )


def compile_sync_arm(report, case, name, launch, target, sync_ir):
    from triton.backends.ascend import compiler

    dump = SyncDumpCompiler(report, f"{case}-{name}")
    try:
        with patch.object(compiler, "subprocess", dump):
            compiled = launch(target)
    finally:
        evidence = {
            rec["file"]: summarize_sync_ir(report / rec["file"])
            for call in dump.calls
            for rec in call["streams"].values()
        }
        sync_ir[name] = dict(invocations=dump.calls, evidence=evidence)
    sync_ir[name]["binary_sha256"] = hashlib.sha256(
        bytes(compiled.asm["npubin"])
    ).hexdigest()
    if len(dump.calls) != 1 or not any(
        e["after_solver_dumps"] and e["functions"] for e in evidence.values()
    ):
        raise RuntimeError(f"Missing post-solver IR for {name}")
    return compiled


def traffic(width, visible, tables, bounded):
    """Count source-level work and the unique useful input, ignoring cache lines."""
    if width % 64 or len(visible) % 16:
        raise ValueError("The two calibration chunks must have full output tiles")
    dense_tiles = width // 64
    units = (dense_tiles + 7) // 8
    q_bytes = len(visible) * (4 * (128 + 2 + 2) + 4)
    write_bytes = len(visible) * width * 4
    dots = active_tiles = page_reads = key_rows = 0
    physical_extent = {}
    unique_pages = set()
    for group in range(0, len(visible), 16):
        limit = min(width, max(visible[group : group + 16]))
        useful = (limit + 63) // 64
        issued = useful if bounded else dense_tiles
        active_tiles += useful
        dots += issued
        key_rows += limit
        for tile in range(0, issued, 8):
            end = min(issued, tile + 8)
            page_reads += (end - 1) // 3 - tile // 3 + 1
        for page, start in enumerate(range(0, limit, 192)):
            physical = tables[group][page]
            assert physical >= 0
            unique_pages.add((group, page))
            physical_extent[physical] = max(
                physical_extent.get(physical, 0), min(192, limit - start)
            )
    pairs = sum(min(width, v) for v in visible)
    unique_k_bytes = sum(physical_extent.values()) * 130
    return dict(
        dot_tiles=dots,
        active_dot_tiles=active_tiles,
        fully_invisible_dot_tiles=dots - active_tiles,
        fill_only_tiles=(len(visible) // 16) * dense_tiles - dots,
        executed_macs=dots * MACS,
        useful_macs=pairs * 4 * 128,
        tile_dot_result_logical_bytes=dots * 64 * 64 * 4,
        q_metadata_requested_bytes=q_bytes * units,
        k_scale_requested_bytes=key_rows * 130,
        page_lookup_sites_executed=page_reads,
        score_write_bytes=write_bytes,
        unique_q_metadata_bytes=q_bytes,
        unique_k_scale_bytes=unique_k_bytes,
        # The unbounded kernel masks table reads for wholly invisible pages.
        # Count only the useful table entries in the perfect-reuse byte model.
        perfect_reuse_io_bytes=q_bytes
        + unique_k_bytes
        + 4 * len(unique_pages)
        + write_bytes,
        requested_io_bytes_upper=q_bytes * units
        + key_rows * 130
        + 4 * page_reads
        + write_bytes,
    )


def capture_arms(
    args, result, torch, arms, targets, validate, dot_arms, compile_arm=None
):
    def launch(name, target):
        return arms[name](target)

    names = {}
    result["compiler"] = {}
    for arm in arms:
        launched = (
            launch(arm, targets[arm])
            if compile_arm is None
            else compile_arm(arm, arms[arm], targets[arm])
        )
        if launched is None:
            continue
        if isinstance(launched, tuple):
            names[arm] = dict(
                collections.Counter(k.metadata.kernel_name for k in launched)
            )
            continue
        compiled = launched
        names[arm] = {compiled.metadata.kernel_name: 1}
        result["compiler"][arm] = str(compiled.metadata)
        ttir = compiled.asm["ttir"]
        expected_dots = (
            dot_arms[arm] if isinstance(dot_arms, dict) else int(arm in dot_arms)
        )
        if ttir.count("tt.dot") != expected_dots:
            raise AssertionError(f"Unexpected dot count in {arm}")
        for key in ("ttir", "ttadapter", "bcmlir"):
            asm = compiled.asm.get(key)
            if isinstance(asm, str):
                (Path(args.report) / f"{args.case}-{arm}-{key}.txt").write_text(asm)
        if args.cv_handoff:
            result.setdefault("device_objects", {})[arm] = export_device_object(
                Path(args.report), f"{args.case}-{arm}", compiled
            )
    result["precheck"] = validate()
    result["kernel_timer"] = {}
    samples, _ = measure_prefix_plan(
        args, names, launch, targets, torch, result, expected_hosts=names
    )
    result["postcheck"] = validate()
    med = {
        arm: statistics.median(t for row in rounds for t in row["active_invocation_us"])
        for arm, rounds in samples.items()
    }
    return samples, med


def export_device_object(report, stem, compiled):
    binary = compiled.asm.get("npubin")
    info = {"asm_keys": list(compiled.asm)}
    if not binary:
        return {**info, "status": "binary_unavailable"}
    binary = bytes(binary)
    archive = report / f"{stem}.npubin.gz"
    archive.write_bytes(gzip.compress(binary))
    info.update(
        bytes=len(binary), sha256=hashlib.sha256(binary).hexdigest(), file=archive.name
    )
    ascend = (
        Path(os.environ["ASCEND_HOME_PATH"]) if os.getenv("ASCEND_HOME_PATH") else None
    )
    candidates = []
    if ascend:
        for subdir in ("compiler/ccec_compiler/bin", "ccec_compiler/bin"):
            candidates.append(ascend / subdir / "llvm-objdump")
    if tool := shutil.which("llvm-objdump"):
        candidates.append(Path(tool))
    info["status"] = "disassembly_unavailable"
    info["attempts"] = []
    obj = report / f"{stem}.npubin"
    obj.write_bytes(binary)
    try:
        for tool in dict.fromkeys(candidates):
            if not tool.is_file() or not os.access(tool, os.X_OK):
                continue
            try:
                p = subprocess.run(
                    [str(tool), "-d", "-h", "-t", str(obj)],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=15,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                info["attempts"].append(dict(tool=str(tool), error=str(exc)))
                continue
            output = p.stdout[: 2**21]
            # A successful exit alone does not show that the target ISA was decoded.
            opcodes = [
                m.lower()
                for m in re.findall(
                    r"^\s*[0-9a-fA-F]+:.*\t([a-zA-Z_][a-zA-Z0-9_.]*)", output, re.M
                )
            ]
            attempt = dict(
                tool=str(tool), exit_code=p.returncode, stderr=p.stderr[:4096]
            )
            if output:
                path = report / f"{stem}-objdump-{len(info['attempts']) + 1}.txt"
                path.write_text(output)
                attempt.update(
                    stdout_file=path.name, truncated=len(p.stdout) > len(output)
                )
            info["attempts"].append(attempt)
            if p.returncode == 0 and opcodes:
                info.update(
                    status="instruction_text_exported",
                    disassembly=path.name,
                    truncated=len(p.stdout) > len(output),
                    observed_mnemonics=dict(collections.Counter(opcodes)),
                )
                break
    finally:
        obj.unlink()
    return info


def page_reuse_function(bounded):
    module = ast.parse(
        Path(__file__).with_name("qsa_int8_query_pack4_910b.py").read_text()
    )
    kernel = (
        "_qsa_page64_page_reuse_score"
        if bounded
        else "_qsa_page64_page_reuse_full_score"
    )
    return next(
        n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == kernel
    )


def build_handoff_source(bounded, arm):
    """Use the current tile traversal on both sides of a materialized INT32 edge."""
    fn = page_reuse_function(bounded)
    fn.name = f"_qsa_page64_prefix_{arm}"
    (unit,) = (n for n in fn.body if isinstance(n, ast.For))
    page, *tail = (n for n in unit.body if isinstance(n, ast.For))
    (tile,) = (n for n in page.body if isinstance(n, ast.For))
    assigned = {
        n.targets[0].id: i
        for i, n in enumerate(tile.body)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    edge = "flat_heads[:, None] * WIDTH + columns[None, :]"
    if arm == "producer":
        tile.body = (
            tile.body[: assigned["dot"] + 1]
            + ast.parse(
                "raw_visible = tl.reshape(tl.broadcast_to(visible[:, None], (16, 4)), (64,))\n"
                "raw_live = (columns[None, :] < WIDTH) & (columns[None, :] < raw_visible[:, None])\n"
                f"tl.store(Out + {edge}, tl.where(raw_live, dot, -2147483648),\n"
                "         mask=(flat_heads[:, None] < ROWS * 4) & (columns[None, :] < WIDTH))"
            ).body
        )
        del tile.body[assigned["key_scale"]]
        for loop in tail:
            unit.body.remove(loop)
    elif arm == "consumer":
        fn.args.args.insert(
            next(i for i, x in enumerate(fn.args.args) if x.arg == "ROWS"),
            ast.arg(arg="Raw"),
        )
        tile.body[assigned["dot"]].value = ast.parse(
            f"tl.load(Raw + {edge})", mode="eval"
        ).body
        del tile.body[assigned["keys"]]
    else:
        raise ValueError(arm)
    ast.fix_missing_locations(fn)
    return "import triton\nimport triton.language as tl\n\n" + ast.unparse(fn) + "\n"


def build_group_visible_source(bounded, arm="group_visible", *, request_offset=False):
    fn = page_reuse_function(bounded)
    fn.name = f"_qsa_page64_prefix_{arm}"
    index = next(i for i, arg in enumerate(fn.args.args) if arg.arg == "ROWS")
    fn.args.args.insert(index, ast.arg(arg="GroupVisible"))
    pointer = "GroupVisible + group"
    if request_offset:
        fn.args.args.insert(index + 1, ast.arg(arg="GroupOffset"))
        fn.decorator_list = [
            ast.parse('triton.jit(do_not_specialize=["GroupOffset"])', mode="eval").body
        ]
        pointer = "GroupVisible + GroupOffset + group"
    (unit,) = (n for n in fn.body if isinstance(n, ast.For))
    (maximum,) = (
        n
        for n in unit.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "max_visible"
    )
    maximum.value = ast.parse(f"tl.load({pointer})", mode="eval").body
    ast.fix_missing_locations(fn)
    return "import triton\nimport triton.language as tl\n\n" + ast.unparse(fn) + "\n"


def build_direct_page_source(
    bounded, arm, *, precomputed_visibility, request_offset=False
):
    """Remove only the scalar page-table load mask on fully valid tables."""
    fn = (
        ast.parse(
            build_group_visible_source(bounded, arm, request_offset=request_offset)
        ).body[-1]
        if precomputed_visibility
        else page_reuse_function(bounded)
    )
    fn.name = f"_qsa_page64_prefix_{arm}"
    (physical,) = (
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "physical"
    )
    if (
        not isinstance(physical.value, ast.Call)
        or ast.unparse(physical.value.func) != "tl.load"
        or ast.unparse(physical.value.args[0])
        != "Table + base * COLUMNS + tl.minimum(page, COLUMNS - 1)"
        or {kw.arg for kw in physical.value.keywords} != {"mask", "other"}
    ):
        raise RuntimeError("Unexpected scalar page-table load in score source")
    physical.value.keywords = []
    return "import triton\nimport triton.language as tl\n\n" + ast.unparse(fn) + "\n"


def page_ir_summary(record):
    scalar = collections.Counter()
    reads = collections.Counter()
    tiles = collections.defaultdict(list)
    fixpipe = collections.defaultdict(list)
    for evidence in record["evidence"].values():
        scalar.update(evidence["scalar_i32_workspace_sites"])
        reads.update(evidence["table_scalar_reads"])
        for function, shapes in evidence["int32_tile_workspace_shapes"].items():
            tiles[function.rsplit("_", 1)[-1]].extend(shapes)
        for function, shapes in evidence["int32_fixpipe_destination_shapes"].items():
            fixpipe[function.rsplit("_", 1)[-1]].extend(shapes)
    return dict(
        scalar_i32_workspace_sites=sum(scalar.values()),
        scalar_i32_workspace_sites_by_core={
            f.rsplit("_", 1)[-1]: count for f, count in scalar.items()
        },
        table_scalar_reads={f.rsplit("_", 1)[-1]: count for f, count in reads.items()},
        int32_tile_workspace_shapes={
            core: sorted(shapes) for core, shapes in tiles.items()
        },
        int32_fixpipe_destination_shapes={
            core: sorted(shapes) for core, shapes in fixpipe.items()
        },
    )


def direct_page_inputs(result, inputs, adapter, score):
    route = result["route"]
    first, count = route["first"], route["count"]
    table = inputs["row_table"][first : first + count]
    if count % 16 or route["query_group"] != 16 or not table.shape[1]:
        raise ValueError("Direct page load requires complete query16 groups")
    if bool(((table < 0) | (table >= inputs["cache"].shape[0])).any()):
        raise ValueError(
            "Direct page load requires every clamped table entry to be valid"
        )
    group_visible = (
        inputs["visible"][first : first + count]
        .reshape(-1, 16)
        .amax(1)
        .to(score.device)
    )
    result["table_check"] = dict(
        rows=count, columns=table.shape[1], all_entries_valid=True
    )
    return adapter.bind_query16_score(first, count, score), group_visible


def run_page_id_handoff(args, result, inputs, adapter, score):
    import torch

    route = result["route"]
    first, count = route["first"], route["count"]
    bound, group_visible = direct_page_inputs(result, inputs, adapter, score)
    report = Path(args.report)
    bounded = route["skip_invisible"]
    sources = dict(
        group_visible=build_group_visible_source(bounded),
        direct_page=build_direct_page_source(
            bounded, "direct_page", precomputed_visibility=False
        ),
        direct_page_group_visible=build_direct_page_source(
            bounded, "direct_page_group_visible", precomputed_visibility=True
        ),
    )
    arms = {"current": lambda out: adapter.score_into(first, count, out)}

    def launch_variant(kernel, metadata, out):
        return kernel[(GRID,)](*bound.args[:-1], out, *metadata, **bound.keywords)

    for name, source in sources.items():
        kernel = load_generated_prefix(report, args.case, name, source)
        metadata = (group_visible,) if "group_visible" in name else ()
        arms[name] = functools.partial(launch_variant, kernel, metadata)
    targets = {
        name: score if name == "current" else torch.empty_like(score) for name in arms
    }
    result.update(mode="page_id_handoff", sync_ir={})

    def compile_arm(name, launch, target):
        return compile_sync_arm(
            report, args.case, name, launch, target, result["sync_ir"]
        )

    def validate():
        torch.npu.synchronize()
        reference = score.cpu()
        check_chunk(inputs, reference, first, count)
        for name, target in targets.items():
            if name != "current" and not torch.equal(reference, target.cpu()):
                raise AssertionError(
                    f"FP32 score or visibility mask differs for {name}"
                )
        return dict(full_outputs_bitwise_equal=True, cpu_reference_checked=True)

    samples, medians = capture_arms(
        args, result, torch, arms, targets, validate, set(arms), compile_arm=compile_arm
    )
    evidence = {
        name: page_ir_summary(record) for name, record in result["sync_ir"].items()
    }
    comparisons = {}
    for control, candidate in (
        ("current", "direct_page"),
        ("group_visible", "direct_page_group_visible"),
    ):
        old, new = evidence[control], evidence[candidate]
        sites_removed = (
            old["scalar_i32_workspace_sites"] - new["scalar_i32_workspace_sites"]
        )
        core_sites_removed = {
            core: old["scalar_i32_workspace_sites_by_core"].get(core, 0)
            - new["scalar_i32_workspace_sites_by_core"].get(core, 0)
            for core in ("aic", "aiv")
        }
        saved = [
            a - b
            for old_round, new_round in zip(
                samples[control], samples[candidate], strict=True
            )
            for a, b in zip(
                old_round["active_invocation_us"],
                new_round["active_invocation_us"],
                strict=True,
            )
        ]
        comparisons[candidate] = dict(
            control=control,
            speedup=medians[control] / medians[candidate],
            paired_saved_us=saved,
            median_paired_saved_us=statistics.median(saved),
            scalar_workspace_sites_removed=sites_removed,
            scalar_workspace_sites_removed_by_core=core_sites_removed,
            page_handoff_removed=(
                sites_removed == 2
                and all(count == 1 for count in core_sites_removed.values())
                and {"aic", "aiv"} <= new["table_scalar_reads"].keys()
            ),
            int32_tile_workspace_unchanged=(
                bool(old["int32_tile_workspace_shapes"])
                and old["int32_tile_workspace_shapes"]
                == new["int32_tile_workspace_shapes"]
            ),
        )
    result.update(
        samples=samples, median_us=medians, page_ir=evidence, comparisons=comparisons
    )
    print(
        "SCORE_PAGE_ID_HANDOFF",
        args.case,
        json.dumps(dict(median_us=medians, page_ir=evidence, comparisons=comparisons)),
        flush=True,
    )


def build_paired_tile_source(arm, *, wide):
    base = build_direct_page_source(True, arm, precomputed_visibility=False)
    fn = ast.parse(base).body[-1]
    (unit,) = (node for node in fn.body if isinstance(node, ast.For))
    page = next(node for node in unit.body if isinstance(node, ast.For))
    (tail,) = (node for node in page.body if isinstance(node, ast.For))
    position = page.body.index(tail)
    paired = deepcopy(tail)
    paired.iter = ast.parse("range(tile_begin, pair_end, 2)", mode="eval").body
    if wide:
        for node in paired.body:
            if not isinstance(node, ast.Assign):
                continue
            name = node.targets[0].id
            if name in ("columns", "token"):
                extent = "tl.arange(0, BLOCK_N)"
                expression = ast.unparse(node.value)
                if expression.count(extent) != 1:
                    raise RuntimeError(f"Unexpected tile extent in {name}")
                node.value = ast.parse(
                    expression.replace(extent, "tl.arange(0, 2 * BLOCK_N)"), mode="eval"
                ).body
            elif name == "scores":
                node.value = ast.parse(
                    "tl.sum(tl.reshape(weighted, (16, 4, 2 * BLOCK_N)), axis=1)",
                    mode="eval",
                ).body
    else:
        paired.target.id = "pair"
        two = ast.parse("for i in tl.static_range(0, 2):\n    tile = pair + i").body[0]
        two.body += paired.body
        paired.body = [two]
    tail.iter = ast.parse("range(pair_end, tile_end)", mode="eval").body
    split = ast.parse("pair_end = tile_end - (tile_end - tile_begin) % 2").body[0]
    page.body[position:position] = [split, paired]
    return "import triton\nimport triton.language as tl\n\n" + ast.unparse(fn) + "\n"


def paired_tile_work(width, visible):
    """Count pairs without crossing the page or eight-tile unit boundary."""
    pairs = tails = 0
    for base in range(0, len(visible), 16):
        active = (min(width, max(visible[base : base + 16])) + 63) // 64
        for first_tile in range(0, active, 8):
            end_tile = min(active, first_tile + 8)
            for page in range(first_tile // 3, (end_tile - 1) // 3 + 1):
                count = min(end_tile, page * 3 + 3) - max(first_tile, page * 3)
                pairs += count // 2
                tails += count % 2
    return dict(
        n128_pairs=pairs,
        n64_tails=tails,
        n64_equivalent_tiles=2 * pairs + tails,
        n64_control_transfers=2 * pairs + tails,
        n128_candidate_transfers=pairs + tails,
        logical_result_bytes=(2 * pairs + tails) * 64 * 64 * 4,
    )


def run_tile_handoff(args, result, inputs, adapter, score):
    import torch
    from triton.compiler.errors import MLIRCompilationError

    route = result["route"]
    first, count = route["first"], route["count"]
    if not route["skip_invisible"]:
        raise ValueError("Tile handoff probe requires the bounded score route")
    if adapter.direct_page_routes.get((first, count)) != "bounded":
        raise AssertionError(
            "The current harness must select bounded direct-page scoring"
        )
    bound = adapter.bind_query16_score(first, count, score)
    result["baseline_route"] = "bounded direct page, runtime group visibility"
    visible = inputs["visible"][first : first + count].tolist()
    work = paired_tile_work(result["width"], visible)
    if (
        not work["n128_pairs"]
        or work["n64_equivalent_tiles"] != result["work"]["dot_tiles"]
    ):
        raise AssertionError("Tile pairing changed the issued dot work")
    report = Path(args.report)
    sources = dict(
        paired_n64=build_paired_tile_source("paired_n64", wide=False),
        paired_n128=build_paired_tile_source("paired_n128", wide=True),
    )

    def launch(kernel, out):
        return kernel[(GRID,)](*bound.args[:-1], out, **bound.keywords)

    arms = {"current": lambda out: bound.func(*bound.args[:-1], out, **bound.keywords)}
    arms.update(
        (
            name,
            functools.partial(
                launch, load_generated_prefix(report, args.case, name, source)
            ),
        )
        for name, source in sources.items()
    )
    targets = dict(
        current=score,
        paired_n64=torch.empty_like(score),
        paired_n128=torch.empty_like(score),
    )
    result.update(mode="tile_handoff", sync_ir={}, pair_work=work)

    def compile_arm(name, launch, target):
        try:
            return compile_sync_arm(
                report, args.case, name, launch, target, result["sync_ir"]
            )
        except MLIRCompilationError as exc:
            if name != "paired_n128":
                raise
            diagnostics = [
                line.strip() for line in str(exc).splitlines() if "error:" in line
            ]
            result["wide_compile_rejection"] = dict(
                arm=name, diagnostics=diagnostics[-12:]
            )
            print(
                "SCORE_TILE_WIDE_REJECTED",
                args.case,
                json.dumps(result["wide_compile_rejection"]),
                flush=True,
            )
            return None

    def validate():
        torch.npu.synchronize()
        reference = score.cpu()
        check_chunk(inputs, reference, first, count)
        exact = {}
        for name in ("paired_n64", "paired_n128"):
            if name not in result["compiler"]:
                continue
            actual = targets[name].cpu()
            torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)
            exact[name] = torch.equal(actual, reference)
        return dict(full_outputs_checked=True, bitwise_equal=exact)

    samples, medians = capture_arms(
        args,
        result,
        torch,
        arms,
        targets,
        validate,
        dict(current=1, paired_n64=3, paired_n128=2),
        compile_arm=compile_arm,
    )
    evidence = {
        name: page_ir_summary(record) for name, record in result["sync_ir"].items()
    }
    n64, n128 = evidence["paired_n64"], evidence.get("paired_n128", {})
    narrow = n64["int32_fixpipe_destination_shapes"].get("aic", [])
    wide = n128.get("int32_fixpipe_destination_shapes", {}).get("aic", [])
    verified = (
        "paired_n128" in samples
        and len(narrow) >= 3
        and all(shape.endswith("64x64xi32") for shape in narrow)
        and any(shape.endswith("64x128xi32") for shape in wide)
        and any(shape.endswith("64x64xi32") for shape in wide)
        and any(
            shape.endswith("64x128xi32")
            for shape in n128["int32_tile_workspace_shapes"].get("aiv", [])
        )
    )
    comparisons = {}
    for control, candidate in (
        ("current", "paired_n64"),
        ("current", "paired_n128"),
        ("paired_n64", "paired_n128"),
    ):
        if candidate not in samples:
            continue
        deltas = [
            a - b
            for old, new in zip(samples[control], samples[candidate], strict=True)
            for a, b in zip(
                old["active_invocation_us"], new["active_invocation_us"], strict=True
            )
        ]
        comparisons[f"{control}_vs_{candidate}"] = dict(
            speedup=medians[control] / medians[candidate],
            paired_saved_us=deltas,
            median_paired_saved_us=statistics.median(deltas),
        )
    result.update(
        samples=samples,
        median_us=medians,
        tile_ir=evidence,
        grouped_fixpipe_verified=verified,
        comparisons=comparisons,
    )
    if "paired_n128" not in samples:
        result["exit_code"] = 1
    print(
        "SCORE_TILE_HANDOFF",
        args.case,
        json.dumps(
            dict(
                baseline_route=result["baseline_route"],
                median_us=medians,
                pair_work=work,
                tile_ir=evidence,
                grouped_fixpipe_verified=verified,
                comparisons=comparisons,
            )
        ),
        flush=True,
    )


def run_group_visible(args, result, inputs, adapter, score):
    import torch

    route = result["route"]
    first, count = route["first"], route["count"]
    if count % 16 or route["query_group"] != 16:
        raise ValueError("Group visibility probe requires complete query16 groups")
    report = Path(args.report)
    variant = load_generated_prefix(
        report,
        args.case,
        "group_visible",
        build_group_visible_source(route["skip_invisible"]),
    )
    prepared = []
    # Production route preparation already reads the per-query visibility on CPU.
    visible_cpu = inputs["visible"][first : first + count]
    torch.npu.synchronize()
    for _ in range(8):
        start = time.perf_counter_ns()
        maxima = visible_cpu.reshape(-1, 16).amax(dim=1)
        end = time.perf_counter_ns()
        group_visible = maxima.to(score.device)
        torch.npu.synchronize()
        prepared.append(
            dict(
                cpu_reduce_us=(end - start) / 1000,
                copy_and_sync_us=(time.perf_counter_ns() - end) / 1000,
            )
        )
    result.update(
        mode="group_visible",
        preparation=dict(
            samples=prepared,
            median_cpu_reduce_us=statistics.median(
                x["cpu_reduce_us"] for x in prepared
            ),
            median_copy_and_sync_us=statistics.median(
                x["copy_and_sync_us"] for x in prepared
            ),
            metadata_bytes=group_visible.numel() * group_visible.element_size(),
            scope="CPU reduction from available host visibility, then allocate/copy/synchronize. "
            "Excludes D2H and adapter preparation.",
        ),
        group_max_visible=maxima.tolist(),
    )
    bound = adapter.bind_query16_score(first, count, score)
    original_inputs = bound.args[:-1]
    targets = dict(current=score, group_visible=torch.empty_like(score))
    arms = dict(
        current=lambda out: adapter.score_into(first, count, out),
        group_visible=lambda out: variant[(GRID,)](
            *original_inputs,
            out,
            group_visible,
            **bound.keywords,
        ),
    )
    result["sync_ir"] = {}

    def compile_arm(name, launch, target):
        return compile_sync_arm(
            report, args.case, name, launch, target, result["sync_ir"]
        )

    def validate():
        torch.npu.synchronize()
        current = score.cpu()
        candidate = targets["group_visible"].cpu()
        for output in (current, candidate):
            check_chunk(inputs, output, first, count)
        torch.testing.assert_close(candidate, current, rtol=1e-3, atol=1e-3)
        return dict(
            full_outputs_checked=True, bitwise_equal=torch.equal(candidate, current)
        )

    samples, med = capture_arms(
        args,
        result,
        torch,
        arms,
        targets,
        validate,
        set(arms),
        compile_arm=compile_arm,
    )
    site_counts = {
        arm: sum(
            sum(e["scalar_i32_workspace_sites"].values())
            for e in record["evidence"].values()
        )
        for arm, record in result["sync_ir"].items()
    }
    deltas = [
        baseline - candidate
        for a, b in zip(samples["current"], samples["group_visible"], strict=True)
        for baseline, candidate in zip(
            a["active_invocation_us"], b["active_invocation_us"], strict=True
        )
    ]
    comparison = dict(
        median_us=med,
        score_speedup=med["current"] / med["group_visible"],
        paired_saved_us=dict(samples=deltas, median=statistics.median(deltas)),
        scalar_i32_workspace_sites=site_counts,
        preparation=result["preparation"],
    )
    result.update(samples=samples, **comparison)
    print("SCORE_GROUP_VISIBLE", args.case, json.dumps(comparison), flush=True)


def run_cv_handoff(args, result, inputs, adapter, score):
    import torch

    route = result["route"]
    first, count = route["first"], route["count"]
    bounded = route["skip_invisible"]
    width = result["width"]
    groups, tiles = count // 16, width // 64
    if tiles % 8:
        raise ValueError("Handoff requires full eight-tile units")
    kernel = {
        arm: load_generated_prefix(
            Path(args.report), args.case, arm, build_handoff_source(bounded, arm)
        )
        for arm in ("producer", "consumer")
    }
    raw = torch.full(
        (count * 4, width),
        torch.iinfo(torch.int32).min,
        device=score.device,
        dtype=torch.int32,
    )
    bound = adapter.bind_query16_score(first, count, score)
    original_inputs = bound.args[:-1]
    produce = functools.partial(
        kernel["producer"][(GRID,)], *original_inputs, **bound.keywords
    )

    def consume(out):
        return kernel["consumer"][(GRID,)](*original_inputs, out, raw, **bound.keywords)

    def split(out):
        producer = produce(raw)
        consumer = consume(out)
        return producer, consumer

    targets = dict(
        current=score,
        producer=raw,
        consumer=torch.empty_like(score),
        split=torch.empty_like(score),
    )
    arms = dict(
        current=lambda out: adapter.score_into(first, count, out),
        producer=produce,
        consumer=consume,
        split=split,
    )

    def validate():
        torch.npu.synchronize()
        actual = score.cpu()
        checks = check_chunk(inputs, actual, first, count)
        raw_cpu = raw.cpu()
        raw_checks = []
        for group in sorted({0, groups // 2, groups - 1}):
            base = first + group * 16
            visible = inputs["visible"][base : base + 16].repeat_interleave(4)
            active = (min(width, int(visible.max())) + 63) // 64
            for tile in sorted({0, 1, 2, 3, 7, 8, active - 1, tiles - 1}):
                begin = tile * 64
                x = raw_cpu[group * 64 : (group + 1) * 64, begin : begin + 64]
                live = torch.arange(begin, begin + 64)[None, :] < visible[:, None]
                want = torch.full_like(x, torch.iinfo(torch.int32).min)
                if bool(live.any()):
                    page, offset = divmod(begin, 192)
                    physical = int(inputs["row_table"][base, page])
                    k = inputs["cache"][physical, offset : offset + 64, 0].float()
                    q = inputs["q"][base : base + 16].reshape(64, 128).float()
                    want[live] = (q @ k.T).to(torch.int32)[live]
                bad = x != want
                record = dict(group=group, tile=tile, mismatch=int(bad.sum()))
                if record["mismatch"]:
                    i, j = bad.nonzero()[0].tolist()
                    record.update(
                        head_lane=i,
                        key_lane=j,
                        actual=int(x[i, j]),
                        expected=int(want[i, j]),
                        matches_transposed=torch.equal(x.T, want),
                    )
                raw_checks.append(record)
        result["last_raw_checks"] = raw_checks
        failures = [record for record in raw_checks if record["mismatch"]]
        if failures:
            print("SOL_CV_RAW_MISMATCH", args.case, json.dumps(failures), flush=True)
            raise AssertionError(f"INT32 producer failed {len(failures)} sampled tiles")
        equal = {}
        for arm in ("consumer", "split"):
            candidate = targets[arm].cpu()
            try:
                torch.testing.assert_close(candidate, actual, rtol=1e-3, atol=1e-3)
            except AssertionError as exc:
                raise AssertionError(
                    f"{arm} score mismatch after raw precheck: {exc}"
                ) from exc
            equal[arm] = torch.equal(candidate, actual)
        return dict(
            current=checks,
            full_outputs_checked=True,
            bitwise_equal=equal,
            checked_raw_tiles=len(raw_checks),
        )

    result.update(
        mode="cv_handoff",
        materialized_layout="head_major_with_visibility_sentinel",
        materialized_allocation_bytes=raw.numel() * raw.element_size(),
        materialized_transfer_bytes_each_direction=result["work"][
            "tile_dot_result_logical_bytes"
        ],
        scope="Fused score versus split score with explicit INT32 global-memory handoff. "
        "Split device task sum excludes the launch interval and includes additional traffic and changed scheduling.",
    )
    samples, med = capture_arms(
        args, result, torch, arms, targets, validate, {"current", "producer"}
    )
    deltas = [
        s - c
        for a, b in zip(samples["split"], samples["current"], strict=True)
        for s, c in zip(
            a["active_invocation_us"], b["active_invocation_us"], strict=True
        )
    ]
    comparison = dict(
        median_us=med,
        paired_split_minus_current_us=dict(
            samples_us=deltas, median_us=statistics.median(deltas)
        ),
        split_over_current=med["split"] / med["current"],
        object_status={k: v["status"] for k, v in result["device_objects"].items()},
    )
    result.update(samples=samples, **comparison)
    print("SOL_CV_HANDOFF", args.case, json.dumps(comparison), flush=True)


def run_output_replay(args, result, inputs, adapter, current_score):
    import torch
    import triton

    @triton.jit
    def _sol_output_replay(
        Dot,
        QS,
        KS,
        W,
        Visible,
        Out,
        GROUPS: tl.constexpr,
        WIDTH: tl.constexpr,
        ROW_STRIDE: tl.constexpr,
        TILE_STRIDE: tl.constexpr,
        BOUNDED: tl.constexpr,
        FILL: tl.constexpr,
    ):
        tiles: tl.constexpr = WIDTH // 64
        units: tl.constexpr = tl.cdiv(tiles, 8)
        n = tl.arange(0, 64)
        q = tl.arange(0, 16)
        for unit in range(tl.program_id(0), GROUPS * units, tl.num_programs(0)):
            group = unit // units
            first_tile = unit % units * 8
            heads = group * 64 + n
            qs = tl.load(QS + heads).to(tl.float32)
            w = tl.load(W + heads).to(tl.float32)
            visible = tl.load(Visible + group * 16 + q)
            if BOUNDED:
                max_visible = tl.max(visible, axis=0)
                active = tl.minimum(
                    8,
                    tl.maximum(
                        0, tl.cdiv(tl.minimum(WIDTH, max_visible), 64) - first_tile
                    ),
                )
            else:
                active = 8
            first_page = first_tile // 3
            page_count = tl.where(active > 0, tl.cdiv(first_tile % 3 + active, 3), 0)
            for page in range(first_page, first_page + page_count):
                start = tl.maximum(first_tile, page * 3)
                end = tl.minimum(first_tile + active, page * 3 + 3)
                for tile in range(start, end):
                    t = group * tiles + tile
                    x = tl.load(Dot + t * 4096 + n[:, None] * 64 + n[None, :])
                    ks = tl.load(KS + t * 64 + n).to(tl.float32)
                    value = x.to(tl.float32) * qs[:, None] * ks[None, :]
                    weighted = tl.maximum(value, 0.0) * w[:, None]
                    score = tl.sum(tl.reshape(weighted, (16, 4, 64)), axis=1)
                    live = tile * 64 + n[None, :] < visible[:, None]
                    tl.store(
                        Out
                        + group * 16 * WIDTH
                        + tile * TILE_STRIDE
                        + q[:, None] * ROW_STRIDE
                        + n[None, :],
                        tl.where(live, score, -float("inf")),
                    )
            if BOUNDED and FILL:
                for tile in range(first_tile + active, first_tile + 8):
                    tl.store(
                        Out
                        + group * 16 * WIDTH
                        + tile * TILE_STRIDE
                        + q[:, None] * ROW_STRIDE
                        + n[None, :],
                        tl.full((16, 64), -float("inf"), tl.float32),
                    )

    route = result["route"]
    first, count = route["first"], route["count"]
    bounded = route["skip_invisible"]
    width = result["width"]
    groups, tiles = count // 16, width // 64
    if tiles % 8:
        raise ValueError("Output replay requires complete eight-tile units")
    device = current_score.device
    gen = torch.Generator().manual_seed(20261106)
    dot = torch.randint(
        -1024, 1025, (groups, tiles, 64, 64), generator=gen, dtype=torch.int32
    )
    ks = torch.rand((groups, tiles, 64), generator=gen).half()
    qs, w, visible = (
        inputs[name][first : first + count] for name in ("qs", "weights", "visible")
    )
    device_inputs = [x.to(device) for x in (dot, qs, ks, w, visible)]
    sentinel = 12345.0
    layout = {
        "compact_dense": (64, 1024, True),
        "row_dense": (width, 64, True),
    }
    if bounded:
        layout["compact_scored"] = (64, 1024, False)
    targets = {"current": current_score}
    arms = {"current": lambda out: adapter.score_into(first, count, out)}
    for name, (row_stride, tile_stride, fill) in layout.items():
        shape = (count, width) if name == "row_dense" else (groups, tiles, 16, 64)
        targets[name] = torch.full(shape, sentinel, device=device, dtype=torch.float32)
        arms[name] = functools.partial(
            _sol_output_replay[(GRID,)],
            *device_inputs,
            GROUPS=groups,
            WIDTH=width,
            ROW_STRIDE=row_stride,
            TILE_STRIDE=tile_stride,
            BOUNDED=bounded,
            FILL=fill,
            num_warps=4,
        )

    vis = visible.reshape(groups, 16)
    tile_index = torch.arange(tiles)
    active_tiles = (vis.amax(1).clamp(max=width) + 63) // 64
    fill_mask = tile_index[None, :] >= active_tiles[:, None]
    if int(fill_mask.sum()) * int(bounded) != result["work"]["fill_only_tiles"]:
        raise AssertionError("Replay fill count disagrees with production accounting")

    def validate():
        torch.npu.synchronize()
        checks = {"current": check_chunk(inputs, current_score.cpu(), first, count)}
        dense = targets["compact_dense"].cpu()
        row = targets["row_dense"].cpu().reshape(groups, 16, tiles, 64).transpose(1, 2)
        torch.testing.assert_close(row, dense, rtol=1e-3, atol=1e-3)
        scored = targets["compact_scored"].cpu() if bounded else None
        if bounded:
            if not bool((scored[fill_mask] == sentinel).all()):
                raise AssertionError("No-fill replay wrote skipped tiles")
            torch.testing.assert_close(
                scored[~fill_mask], dense[~fill_mask], rtol=1e-3, atol=1e-3
            )
        columns = torch.arange(width).reshape(tiles, 1, 64)
        for group in range(groups):
            heads = slice(group * 16, (group + 1) * 16)
            scaled = (
                dot[group].float()
                * qs[heads].reshape(64, 1).float()
                * ks[group, :, None, :].float()
            )
            weighted = scaled.relu() * w[heads].reshape(64, 1).float()
            want = weighted.reshape(tiles, 16, 4, 64).sum(dim=2)
            want.masked_fill_(columns >= vis[group, None, :, None], -float("inf"))
            for got in (dense[group], row[group]):
                torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)
        checks.update(
            replay_all_outputs_checked=True,
            row_compact_bitwise_equal=torch.equal(row, dense),
        )
        return checks

    result.update(
        mode="output_replay",
        layout={
            k: dict(row_stride=r, tile_stride=t, fill=f)
            for k, (r, t, f) in layout.items()
        },
        scope="Current score versus AIV replay from materialized INT32. "
        "Replay comparisons capture output layout and fill effects, including compiler rescheduling.",
    )
    samples, med = capture_arms(
        args, result, torch, arms, targets, validate, {"current"}
    )

    def paired(a, b):
        deltas = [
            x - y
            for round_a, round_b in zip(samples[a], samples[b], strict=True)
            for x, y in zip(
                round_a["active_invocation_us"],
                round_b["active_invocation_us"],
                strict=True,
            )
        ]
        return dict(samples_us=deltas, median_us=statistics.median(deltas))

    differences = {
        "row_minus_compact_dense": paired("row_dense", "compact_dense"),
        "current_minus_row_dense": paired("current", "row_dense"),
    }
    if bounded:
        differences["compact_dense_minus_compact_scored"] = paired(
            "compact_dense", "compact_scored"
        )
    comparison = dict(
        median_us=med,
        paired_differences=differences,
        current_over_row_dense=med["current"] / med["row_dense"],
    )
    result.update(samples=samples, **comparison)
    print("SOL_OUTPUT_REPLAY", args.case, json.dumps(comparison), flush=True)


def run_case(args, result):
    global tl

    import torch
    import torch_npu
    import triton
    import triton.language as tl
    from qsa_int8_shared_map_selection_910b_trial import PreparedSharedMapInt8Selection

    root = os.getenv("QSA_ENV_ROOT")
    if not root or not torch.npu.is_available():
        raise ModuleNotFoundError("Pinned NPU environment required")
    if (
        not Path(triton.__file__)
        .resolve()
        .is_relative_to(Path(root).resolve() / "flagtree")
    ):
        raise RuntimeError(f"Unexpected compiler: {triton.__file__}")
    torch.npu.set_device(args.device)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = f"npu:{args.device}"
    result["environment"] = dict(
        device=torch.npu.get_device_name(args.device),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        compiler=triton.__file__,
        opp=os.getenv("ASCEND_OPP_PATH"),
    )
    if "910B" not in result["environment"]["device"]:
        raise NotImplementedError("910B required")

    @triton.jit
    def _sol_copy(X, Y, SIZE: tl.constexpr):
        i = tl.arange(0, 8192)
        for tile in range(tl.program_id(0), SIZE // 8192, tl.num_programs(0)):
            x = tile * 8192 + i
            tl.store(Y + x, tl.load(X + x))

    @triton.jit
    def _sol_cube(Q, K, Out, COUNT: tl.constexpr):
        n = tl.arange(0, 64)
        d = tl.arange(0, 128)
        ij = n[:, None] * 128 + d[None, :]
        for unit in range(tl.program_id(0), COUNT, tl.num_programs(0)):
            q = tl.load(Q + unit * 8192 + ij)
            acc = tl.full((64, 64), 0, tl.int32)
            for tile in range(8):
                k = tl.load(K + (unit * 8 + tile) * 8192 + ij)
                acc = tl.dot(q, tl.trans(k), acc, out_dtype=tl.int32)
            tl.store(Out + unit * 4096 + n[:, None] * 64 + n[None, :], acc)

    @triton.jit
    def _sol_vector(Dot, QS, KS, W, Visible, Out, COUNT: tl.constexpr):
        n = tl.arange(0, 64)
        q = tl.arange(0, 16)
        for unit in range(tl.program_id(0), COUNT, tl.num_programs(0)):
            qs = tl.load(QS + unit * 64 + n).to(tl.float32)
            w = tl.load(W + unit * 64 + n).to(tl.float32)
            visible = tl.load(Visible + unit * 16 + q)
            for i in range(8):
                tile = unit * 8 + i
                x = tl.load(Dot + tile * 4096 + n[:, None] * 64 + n[None, :])
                ks = tl.load(KS + tile * 64 + n).to(tl.float32)
                value = x.to(tl.float32) * qs[:, None] * ks[None, :]
                weighted = tl.maximum(value, 0.0) * w[:, None]
                score = tl.sum(tl.reshape(weighted, (16, 4, 64)), axis=1)
                live = i * 64 + n[None, :] < visible[:, None]
                tl.store(
                    Out + tile * 1024 + q[:, None] * 64 + n[None, :],
                    tl.where(live, score, -float("inf")),
                )

    plan = PLANS[args.case]
    inputs = make_inputs(plan)
    values = {k: v.to(device) for k, v in inputs.items()}
    first, count = CHUNKS[args.case]
    width = plan["logits_width"]
    score = torch.empty((count, width), device=device, dtype=torch.float32)
    direct_page_kernels = None
    if args.tile_handoff:
        arm = "direct_page_bounded"
        direct_page_kernels = {
            True: load_generated_prefix(
                Path(args.report),
                args.case,
                arm,
                build_direct_page_source(True, arm, precomputed_visibility=False),
            )
        }
    adapter = PreparedSharedMapInt8Selection(
        values,
        torch.empty((plan["q_shape"][0], 512), device=device, dtype=torch.int32),
        score,
        torch.empty(1024**2, device=device, dtype=torch.uint8),
        None,
        plan["starts"],
        key_group4=True,
        single_page64=True,
        direct_page_kernels=direct_page_kernels,
        shared_direct_page_proof=args.tile_handoff,
    )
    route = next(
        r for r in adapter.route_checks if (r["first"], r["count"]) == (first, count)
    )
    bounded = args.case == "int8_budget_exact"
    if not route["page_reuse"] or route["skip_invisible"] != bounded:
        raise AssertionError(f"Unexpected score route: {route}")
    work = traffic(
        width,
        inputs["visible"][first : first + count].tolist(),
        inputs["row_table"][first : first + count].tolist(),
        bounded,
    )
    visibility = route["visibility_tiles"]
    if (
        work["active_dot_tiles"] != visibility["useful_tiles"]
        or work["dot_tiles"] + work["fill_only_tiles"] != visibility["issued_tiles"]
    ):
        raise AssertionError(
            f"Work accounting disagrees with the dispatch: {visibility}"
        )
    result.update(route=route, width=width, work=work)
    print("SOL_WORK", args.case, json.dumps(work), flush=True)

    if args.instruction_entry:
        launch = functools.partial(adapter.score_into, first, count, score)
        if args.independent_buffers:
            bound = adapter.bind_query16_score(first, count, score)
            tensors = list(bound.args)
            row_inputs = {0: "q", 2: "qs", 4: "weights", 5: "row_table", 6: "visible"}
            for position, name in row_inputs.items():
                tensors[position] = (
                    inputs[name][first : first + count].clone().to(device)
                )
            if any(t.storage_offset() != 0 or not t.is_contiguous() for t in tensors):
                raise AssertionError(
                    "Instruction entry requires base contiguous allocations"
                )
            result["independent_buffers"] = dict(
                input_shapes=[list(t.shape) for t in tensors],
                allocator_cache_disabled=os.getenv("PYTORCH_NO_NPU_MEMORY_CACHING")
                == "1",
            )
            launch = functools.partial(bound.func, *tensors, **bound.keywords)
        torch.npu.synchronize()
        if args.dump_sync_ir:
            from triton.backends.ascend import compiler

            capture = SyncDumpCompiler(Path(args.report), args.case)
            with patch.object(compiler, "subprocess", capture):
                compiled = launch()
            if len(capture.calls) != 1:
                raise RuntimeError(
                    f"Expected one final BiSheng invocation, captured {len(capture.calls)}"
                )
            result["sync_ir_capture"] = capture.calls[0]
        else:
            compiled = launch()
        binary = bytes(compiled.asm["npubin"])
        result["instruction_entry"] = dict(
            kernel=compiled.metadata.kernel_name,
            binary_sha256=hashlib.sha256(binary).hexdigest(),
            score_launches=1,
        )
        if args.export_instruction_object:
            (Path(args.report) / f"{args.case}.npubin.gz").write_bytes(
                gzip.compress(binary)
            )
        (Path(args.report) / f"{args.case}-launched.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        torch.npu.synchronize()
        if args.verify_output:
            result["checks"] = check_chunk(inputs, score.cpu(), first, count)
        print(
            "SCORE_INSTRUCTION_ENTRY",
            json.dumps(result["instruction_entry"]),
            flush=True,
        )
        return

    if args.cv_handoff:
        run_cv_handoff(args, result, inputs, adapter, score)
        return
    if args.group_visible:
        run_group_visible(args, result, inputs, adapter, score)
        return
    if args.page_id_handoff:
        run_page_id_handoff(args, result, inputs, adapter, score)
        return
    if args.tile_handoff:
        run_tile_handoff(args, result, inputs, adapter, score)
        return
    if args.output_replay:
        run_output_replay(args, result, inputs, adapter, score)
        return

    gen = torch.Generator().manual_seed(20261106)

    def integers(low, high, shape, dtype):
        return torch.randint(low, high, shape, generator=gen, dtype=dtype)

    q = integers(-16, 17, (UNITS, 64, 128), torch.int8)
    k = integers(-16, 17, (TILES, 64, 128), torch.int8)
    dot = integers(-1024, 1025, (TILES, 64, 64), torch.int32)
    qs = torch.rand((UNITS, 64), generator=gen).half()
    ks = torch.rand((TILES, 64), generator=gen).half()
    w = (torch.rand((UNITS, 64), generator=gen) * 2 - 1).half()
    vis = integers(384, 513, (UNITS, 16), torch.int32)
    q_npu, k_npu, dot_npu, qs_npu, ks_npu, w_npu, vis_npu = [
        x.to(device) for x in (q, k, dot, qs, ks, w, vis)
    ]
    copy = integers(-128, 128, (COPY_BYTES,), torch.int8).to(device)
    targets = dict(
        current=score,
        copy=torch.empty_like(copy),
        cube=torch.empty((UNITS, 64, 64), device=device, dtype=torch.int32),
        vector=torch.empty((TILES, 16, 64), device=device, dtype=torch.float32),
    )
    grid = (GRID,)
    arms = {
        "current": lambda out: adapter.score_into(first, count, out),
        "copy": lambda out: _sol_copy[grid](copy, out, COPY_BYTES, num_warps=4),
        "cube": lambda out: _sol_cube[grid](q_npu, k_npu, out, UNITS, num_warps=4),
        "vector": lambda out: _sol_vector[grid](
            dot_npu, qs_npu, ks_npu, w_npu, vis_npu, out, UNITS, num_warps=4
        ),
    }

    def validate():
        torch.npu.synchronize()
        checks = check_chunk(inputs, score.cpu(), first, count)
        if not torch.equal(copy.cpu(), targets["copy"].cpu()):
            raise AssertionError("Streaming copy mismatch")
        for unit in (0, UNITS // 2, UNITS - 1):
            start = unit * 8
            keys = k[start : start + 8].float()
            expected = (q[unit].float() @ keys.transpose(1, 2)).sum(dim=0).int()
            if not torch.equal(targets["cube"][unit].cpu(), expected):
                raise AssertionError(f"Cube mismatch at unit {unit}")
            scaled = (
                dot[start : start + 8].float()
                * qs[unit, :, None].float()
                * ks[start : start + 8, None, :].float()
            )
            want = (
                (scaled.relu() * w[unit, :, None].float()).reshape(8, 16, 4, 64).sum(2)
            )
            live = torch.arange(512).reshape(8, 1, 64) < vis[unit, None, :, None]
            want.masked_fill_(~live, -float("inf"))
            torch.testing.assert_close(
                targets["vector"][start : start + 8].cpu(), want, rtol=1e-3, atol=1e-3
            )
        return checks

    samples, med = capture_arms(
        args, result, torch, arms, targets, validate, {"current", "cube"}
    )
    copy_bytes_us = (2 * COPY_BYTES) / med["copy"]
    macs_us = TILES * MACS / med["cube"]
    vector_tiles_us = TILES / med["vector"]
    refs = dict(
        dense_io_with_perfect_reuse=work["perfect_reuse_io_bytes"] / copy_bytes_us,
        source_io_without_reuse=work["requested_io_bytes_upper"] / copy_bytes_us,
        useful_cube=work["useful_macs"] / macs_us,
        issued_cube=work["executed_macs"] / macs_us,
        issued_vector_replay=work["dot_tiles"] / vector_tiles_us,
    )
    limiting_reference = max(
        ("dense_io_with_perfect_reuse", "issued_cube", "issued_vector_replay"),
        key=refs.__getitem__,
    )
    result.update(
        samples=samples,
        median_us=med,
        copy_gb_s=copy_bytes_us / 1000,
        cube_tmac_s=macs_us / 1e6,
        vector_tiles_per_us=vector_tiles_us,
        component_reference_us=refs,
        limiting_reference=limiting_reference,
        ideal_overlap_reference_us=refs[limiting_reference],
        current_over_ideal_overlap_reference=med["current"] / refs[limiting_reference],
        current_over_component_reference={
            k: med["current"] / v for k, v in refs.items()
        },
        scope="Measured diagnostic references, not physical lower bounds; "
        "copy uses contiguous read plus write, cube accumulates eight tiles per sink, "
        "vector replays materialized INT32 with a compact tile-major store",
    )
    print(
        "SOL_RESULT",
        args.case,
        json.dumps(
            {
                k: result[k]
                for k in (
                    "median_us",
                    "copy_gb_s",
                    "cube_tmac_s",
                    "vector_tiles_per_us",
                    "component_reference_us",
                    "limiting_reference",
                    "ideal_overlap_reference_us",
                    "current_over_ideal_overlap_reference",
                    "current_over_component_reference",
                )
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--kernel-warmup", type=int, default=2)
    parser.add_argument("--kernel-active", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--case", choices=CHUNKS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--output-replay",
        action="store_true",
        help="Compare production-shaped Vector replay with compact and row-strided outputs",
    )
    mode.add_argument(
        "--cv-handoff",
        action="store_true",
        help="Measure the real INT32 producer and consumer",
    )
    mode.add_argument(
        "--group-visible",
        action="store_true",
        help="Probe precomputed maximum visibility per 16 queries",
    )
    mode.add_argument(
        "--page-id-handoff",
        action="store_true",
        help="Probe removing the page-number scalar handoff on fully valid tables",
    )
    mode.add_argument(
        "--tile-handoff",
        action="store_true",
        help="Compare two N64 transfers with one N128 transfer on budget_exact",
    )
    mode.add_argument(
        "--instruction-entry", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--verify-output", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--independent-buffers", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--dump-sync-ir", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--export-instruction-object", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if (args.instruction_entry and not args.case) or (
        (
            args.verify_output
            or args.independent_buffers
            or args.dump_sync_ir
            or args.export_instruction_object
        )
        and not args.instruction_entry
    ):
        parser.error(
            "Instruction entry requires one --case and supports --verify-output"
        )
    if min(args.rounds, args.kernel_warmup, args.kernel_active, args.timeout) < 1:
        parser.error("Positive timing counts required")
    if args.tile_handoff and args.case not in (None, "int8_budget_exact"):
        parser.error("Tile handoff probe requires int8_budget_exact")
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    if args.case:
        if args.group_visible or args.page_id_handoff or args.tile_handoff:
            os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(
                prefix=f"{args.case}-sync-probe-cache-", dir=report
            )
        result = {"case": args.case, "exit_code": 0}
        try:
            run_case(args, result)
        except ModuleNotFoundError:
            result.update(exit_code=2, traceback=traceback.format_exc())
        except Exception:
            result.update(exit_code=1, traceback=traceback.format_exc())
        if "traceback" in result:
            print(result["traceback"], flush=True)
        (report / f"{args.case}.json").write_text(json.dumps(result, indent=2) + "\n")
        return result["exit_code"]

    results = []
    for case in ("int8_budget_exact",) if args.tile_handoff else CHUNKS:
        path = report / f"{case}.json"
        path.unlink(missing_ok=True)
        command = [sys.executable, "-u", __file__, *sys.argv[1:], "--case", case]
        status, timed_out = run_child(
            command, report / f"{case}-worker.txt", args.timeout
        )
        result = json.loads(path.read_text()) if path.exists() else {"case": case}
        if timed_out or status not in (0, 2) or "exit_code" not in result:
            result["exit_code"] = 1
        result.update(worker_status=status, timed_out=timed_out)
        results.append(result)
    codes = {r["exit_code"] for r in results}
    code = 1 if 1 in codes else 2 if 2 in codes else 0
    (report / "summary.json").write_text(
        json.dumps(dict(cases=results, exit_code=code), indent=2) + "\n"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
