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
import traceback
from pathlib import Path

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


def capture_arms(args, result, torch, arms, targets, validate, dot_arms):
    def launch(name, target):
        return arms[name](target)

    names = {}
    result["compiler"] = {}
    for arm in arms:
        launched = launch(arm, targets[arm])
        if isinstance(launched, tuple):
            names[arm] = dict(
                collections.Counter(k.metadata.kernel_name for k in launched)
            )
            continue
        compiled = launched
        names[arm] = {compiled.metadata.kernel_name: 1}
        result["compiler"][arm] = str(compiled.metadata)
        ttir = compiled.asm["ttir"]
        if ttir.count("tt.dot") != int(arm in dot_arms):
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
        args, arms, launch, targets, torch, result, expected_hosts=names
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
                    [str(tool), "-d", str(obj)],
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
            info["attempts"].append(
                dict(tool=str(tool), exit_code=p.returncode, stderr=p.stderr[:4096])
            )
            if p.returncode == 0 and opcodes:
                path = report / f"{stem}-objdump.txt"
                path.write_text(output)
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


def build_handoff_source(bounded, arm):
    """Use the current tile traversal on both sides of a materialized INT32 edge."""
    module = ast.parse(
        Path(__file__).with_name("qsa_int8_query_pack4_910b.py").read_text()
    )
    kernel = (
        "_qsa_page64_page_reuse_score"
        if bounded
        else "_qsa_page64_page_reuse_full_score"
    )
    fn = next(
        n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == kernel
    )
    fn.name = f"_qsa_page64_prefix_{arm}"
    (unit,) = (n for n in fn.body if isinstance(n, ast.For))
    page, *tail = (n for n in unit.body if isinstance(n, ast.For))
    (tile,) = (n for n in page.body if isinstance(n, ast.For))
    assigned = {
        n.targets[0].id: i
        for i, n in enumerate(tile.body)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    edge = "((group * tiles + tile) * 4096 + lanes[:, None] * 64 + tl.arange(0, 64)[None, :])"
    if arm == "producer":
        tile.body = (
            tile.body[: assigned["dot"] + 1]
            + ast.parse(f"tl.store(Out + {edge}, dot)").body
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
        (groups, tiles, 64, 64),
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
        equal = {}
        for arm in ("consumer", "split"):
            candidate = targets[arm].cpu()
            torch.testing.assert_close(candidate, actual, rtol=1e-3, atol=1e-3)
            equal[arm] = torch.equal(candidate, actual)
        seen = []
        for group in (0, groups - 1):
            base = first + group * 16
            visible = min(width, int(inputs["visible"][base : base + 16].max()))
            active = (visible + 63) // 64
            for tile in sorted({0, active - 1, tiles - 1}):
                x = raw[group, tile].cpu()
                if tile >= active:
                    want = torch.full_like(
                        x, torch.iinfo(torch.int32).min if bounded else 0
                    )
                else:
                    begin = tile * 64
                    page, offset = divmod(begin, 192)
                    physical = int(inputs["row_table"][base, page])
                    k = (
                        inputs["cache"][physical, offset : offset + 64, 0]
                        .float()
                        .clone()
                    )
                    k[max(0, visible - begin) :] = 0
                    q = inputs["q"][base : base + 16].reshape(64, 128).float()
                    want = (q @ k.T).to(torch.int32)
                if not torch.equal(x, want):
                    raise AssertionError(
                        f"INT32 handoff mismatch at group {group} tile {tile}"
                    )
                seen.append([group, tile])
        return dict(
            current=checks,
            full_outputs_checked=True,
            bitwise_equal=equal,
            checked_raw_tiles=seen,
        )

    result.update(
        mode="cv_handoff",
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
    adapter = PreparedSharedMapInt8Selection(
        values,
        torch.empty((plan["q_shape"][0], 512), device=device, dtype=torch.int32),
        score,
        torch.empty(1024**2, device=device, dtype=torch.uint8),
        None,
        plan["starts"],
        key_group4=True,
        single_page64=True,
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

    if args.cv_handoff:
        run_cv_handoff(args, result, inputs, adapter, score)
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
    args = parser.parse_args()
    if min(args.rounds, args.kernel_warmup, args.kernel_active, args.timeout) < 1:
        parser.error("Positive timing counts required")
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    if args.case:
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
    for case in CHUNKS:
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
