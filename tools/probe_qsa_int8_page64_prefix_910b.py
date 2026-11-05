# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate and time observable prefixes of the current N64 single-page scorer."""

import argparse
import ast
import bisect
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_qsa_int8_kernel_910b import snapshot  # noqa: E402
from probe_qsa_int8_ten_shapes_910b import PLANS, run_child  # noqa: E402
from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan  # noqa: E402
from qsa_int8_trace_timer_910b import setup_trace  # noqa: E402
from qsa_kernel_benchmark_910b import aggregate  # noqa: E402

from tools.qsa_int8_ten_workloads_910b import make_inputs  # noqa: E402

ARMS = ("loads", "dot", "query_scaled", "dequant", "full", "current")
CASES = ("int8_budget_exact", "int8_long100k")
FROZEN = "0d0b371e406168d706705c904eb382ffc12d4f382d3237fb969c23c9e9424ee6"


PREFETCH_ARMS = ("page_prefetch", "page_prefetch_control")


def build_source(arm):
    """Retain original addressing/loop prefixes and an explicit FP32 output sink."""
    if arm in PREFETCH_ARMS:
        source = Path(__file__).with_name("qsa_int8_query_pack4_910b.py")
        module = ast.parse(source.read_text())
        fn = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_qsa_page64_page_reuse_full_score"
        )
        fn.name = f"_qsa_page64_prefix_{arm}"
        outer = next(node for node in fn.body if isinstance(node, ast.For))
        page = next(node for node in outer.body if isinstance(node, ast.For))
        tile = next(node for node in page.body if isinstance(node, ast.For))
        tile.iter = ast.parse("range(tile_begin, tile_end, 2)", mode="eval").body
        first = tile.body
        dot_index = next(
            i
            for i, node in enumerate(first)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "dot" for t in node.targets)
        )
        second_load = ast.parse(
            """
next_tile = tile + 1
next_columns = next_tile * BLOCK_N + tl.arange(0, BLOCK_N)
next_live = (next_tile < tile_end) & (next_columns < WIDTH) & (next_columns < max_visible)
next_token = physical.to(tl.int64) * PAGE + (next_tile - page * 3) * BLOCK_N + tl.arange(0, BLOCK_N)
next_keys = tl.load(K + next_token[:, None] * 128 + dims[None, :], mask=next_live[:, None], other=0)
"""
        ).body
        second_score = ast.parse(
            """
if next_tile < tile_end:
    next_key_scale = tl.load(KS + next_token, mask=next_live, other=0).to(tl.float32)
"""
        ).body[0]
        score_text = ast.unparse(ast.Module(body=first[dot_index:], type_ignores=[]))
        score_text = re.sub(
            r"\b(columns|keys|key_scale|dot|dequant|weighted|scores|live)\b",
            lambda match: (
                "next_score_live" if match[0] == "live" else f"next_{match[0]}"
            ),
            score_text,
        )
        # Rename local values, leaving the tl.dot intrinsic intact.
        second_score.body.extend(
            ast.parse(score_text.replace("tl.next_dot", "tl.dot")).body
        )
        insert_at = dot_index if arm == "page_prefetch" else len(first)
        tile.body = first[:insert_at] + second_load + first[insert_at:] + [second_score]
        ast.fix_missing_locations(fn)
        return (
            "import triton\nimport triton.language as tl\n\n" + ast.unparse(fn) + "\n"
        )
    if arm in ("page_reuse", "page_reuse_full"):
        module = ast.parse(
            build_source("query16" if arm == "page_reuse_full" else "skip_invisible")
        )
        fn = next(n for n in module.body if isinstance(n, ast.FunctionDef))
        fn.name = f"_qsa_page64_prefix_{arm}"
        outer = next(n for n in fn.body if isinstance(n, ast.For))
        inner = next(n for n in outer.body if isinstance(n, ast.For))
        text = ast.unparse(inner)
        # Preserve eight-tile work ownership, but load one physical page per
        # page segment and reuse it for up to three consecutive N64 tiles.
        text = text.replace(
            (
                "for local_tile in range(KEY_TILES):"
                if arm == "page_reuse_full"
                else "for local_tile in range(active_tiles):"
            ),
            "for tile in range(tile_begin, tile_end):",
        )
        text = text.replace("    tile = first_tile + local_tile\n", "")
        text = text.replace("    page = start // PAGE\n", "")
        text = text.replace(
            "offset = start % PAGE", "offset = (tile - page * 3) * BLOCK_N"
        )
        active = ast.parse(text).body[0]
        active.body = [
            n
            for n in active.body
            if not (
                isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "physical" for t in n.targets
                )
            )
        ]
        page_loop = ast.parse(
            """
first_page = first_tile // 3
page_count = tl.where(active_tiles > 0, tl.cdiv(first_tile % 3 + active_tiles, 3), 0)
for page in range(first_page, first_page + page_count):
    tile_begin = tl.maximum(first_tile, page * 3)
    tile_end = tl.minimum(first_tile + active_tiles, page * 3 + 3)
    physical = tl.load(Table + base*COLUMNS + tl.minimum(page,COLUMNS-1),
                       mask=(page<COLUMNS)&(tile_begin<tile_end),other=0)
"""
        ).body
        if arm == "page_reuse_full":
            page_source = ast.unparse(
                ast.Module(body=page_loop, type_ignores=[])
            ).replace("active_tiles", "KEY_TILES")
            page_source = page_source.replace(
                "(page < COLUMNS) & (tile_begin < tile_end)",
                "(page < COLUMNS) & (tile_begin < tile_end) & (tile_begin * BLOCK_N < WIDTH) & (tile_begin * BLOCK_N < max_visible)",
            )
            page_loop = ast.parse(page_source).body
        page_loop[-1].body.append(active)
        index = outer.body.index(inner)
        outer.body[index : index + 1] = page_loop
        ast.fix_missing_locations(module)
        return ast.unparse(module) + "\n"
    if arm == "skip_invisible":
        module = ast.parse(build_source("query16"))
        fn = next(n for n in module.body if isinstance(n, ast.FunctionDef))
        fn.name = "_qsa_page64_prefix_skip_invisible"
        outer = next(n for n in fn.body if isinstance(n, ast.For))
        inner = next(n for n in outer.body if isinstance(n, ast.For))
        # Bound the dot loop by group visibility. A separate store-only loop
        # fills the remaining tiles, keeping dense logits semantics and cost.
        bound = ast.parse(
            "active_tiles = tl.minimum(KEY_TILES, tl.maximum(0, tl.cdiv(tl.minimum(WIDTH, max_visible), BLOCK_N) - first_tile))"
        ).body[0]
        inner.iter = ast.parse("range(active_tiles)", mode="eval").body
        fill = ast.parse(
            """
for empty_tile in range(active_tiles, KEY_TILES):
    columns = (first_tile + empty_tile) * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(Out + rows[:,None]*WIDTH + columns[None,:], -float('inf'),
             mask=(rows[:,None]<ROWS)&(columns[None,:]<WIDTH))
"""
        ).body[0]
        index = outer.body.index(inner)
        outer.body[index : index + 1] = [bound, inner, fill]
        ast.fix_missing_locations(module)
        return ast.unparse(module) + "\n"
    source = Path(__file__).with_name("qsa_int8_query_pack4_910b.py").read_text()
    module = ast.parse(source)
    fn = next(
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef) and n.name == "_qsa_qk_single_page_score"
    )
    if (
        hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest()
        != FROZEN
    ):
        raise RuntimeError("Single-page scorer drift; review prefix builder")
    fn.name = f"_qsa_page64_prefix_{arm}"
    outer = next(n for n in fn.body if isinstance(n, ast.For))
    inner = next(n for n in outer.body if isinstance(n, ast.For))
    if arm in ("query8", "query16"):
        group = int(arm.removeprefix("query"))
        # Only query grouping changes; four heads/query and K addressing stay fixed.
        text = ast.unparse(fn)
        replacements = {
            "triton.cdiv(ROWS, 4)": f"triton.cdiv(ROWS, {group})",
            "tl.arange(0, 16)": f"tl.arange(0, {group * 4})",
            "tl.arange(0, 4)": f"tl.arange(0, {group})",
            "base = group * 4": f"base = group * {group}",
            "(4, 4, BLOCK_N)": f"({group}, 4, BLOCK_N)",
        }
        for old, new in replacements.items():
            if text.count(old) != 1:
                raise RuntimeError(f"Query sweep source drift: {old}")
            text = text.replace(old, new)
        return "import triton\nimport triton.language as tl\n\n" + text + "\n"
    if arm == "unrolled_full":
        inner.iter = ast.parse("tl.static_range(KEY_TILES)", mode="eval").body
    elif arm == "packed_full":
        position = next(
            i
            for i, n in enumerate(inner.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "keys" for t in n.targets)
        )
        inner.body[position] = ast.parse(
            "keys = tl.load(K + physical.to(tl.int64)*PAGE*128 + dims[:,None]*PAGE"
            " + offset + tl.arange(0,BLOCK_N)[None,:], mask=key_live[None,:], other=0)"
        ).body[0]
        dot = next(
            n
            for n in inner.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "dot" for t in n.targets)
        )
        dot.value = ast.parse(
            "tl.dot(query, keys, out_dtype=tl.int32)", mode="eval"
        ).body
    elif arm in ("raw_i32", "raw_fp32"):
        position = next(
            i
            for i, n in enumerate(inner.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "dot" for t in n.targets)
        )
        value = "dot" if arm == "raw_i32" else "dot.to(tl.float32)"
        invalid = "-2147483648" if arm == "raw_i32" else "-float('inf')"
        tail = ast.parse(
            "raw_visible = tl.reshape(tl.broadcast_to(visible[:,None], (4,4)), (16,))\n"
            "raw_live = (columns[None,:]<WIDTH) & (columns[None,:]<raw_visible[:,None])\n"
            f"tl.store(Out + flat_heads[:,None]*WIDTH + columns[None,:], tl.where(raw_live,{value},{invalid}),"
            " mask=(flat_heads[:,None]<ROWS*4)&(columns[None,:]<WIDTH))"
        ).body
        inner.body = inner.body[: position + 1] + tail
    elif arm != "full":
        cutoff = {
            "loads": "key_scale",
            "dot": "dot",
            "query_scaled": "dot",
            "dequant": "dequant",
        }[arm]
        position = next(
            i
            for i, n in enumerate(inner.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == cutoff for t in n.targets)
        )
        expr = {
            "loads": (
                "(tl.sum(tl.reshape(tl.sum(query.to(tl.float32), axis=1), (4,4)), axis=1)"
                " + tl.sum(tl.reshape(query_scale, (4,4)), axis=1)"
                " + tl.sum(tl.reshape(weights, (4,4)), axis=1))[:,None]"
                " + (tl.sum(keys.to(tl.float32), axis=1) + key_scale)[None,:]"
            ),
            "dot": "tl.sum(tl.reshape(dot.to(tl.float32), (4,4,BLOCK_N)), axis=1)",
            "query_scaled": "tl.sum(tl.reshape(dot.to(tl.float32) * query_scale[:,None], (4,4,BLOCK_N)), axis=1)",
            "dequant": "tl.sum(tl.reshape(dequant, (4,4,BLOCK_N)), axis=1)",
        }[arm]
        tail = ast.parse(f"scores = {expr}").body
        # Preserve the real output layout, visibility mask and padded-row guard.
        inner.body = inner.body[: position + 1] + tail + inner.body[-2:]
    ast.fix_missing_locations(fn)
    return "import triton\nimport triton.language as tl\n\n" + ast.unparse(fn) + "\n"


def prefetch_schedule(ttir):
    """Record the order and SSA use of the two K loads in the paired loop."""
    lines = ttir.splitlines()
    start = max(i for i, line in enumerate(lines) if "scf.for " in line)
    header = lines[start]
    body = lines[start + 1 :]
    loads = [
        (i, line)
        for i, line in enumerate(body)
        if "tt.load" in line and "tensor<64x128x!tt.ptr<i8>>" in line
    ]
    dots = [(i, line) for i, line in enumerate(body) if "tt.dot " in line]
    order = "unconfirmed"
    if len(loads) == len(dots) == 2:
        feeds_dot = []
        for (load_index, load), (dot_index, dot) in zip(loads, dots):
            value = load.split("=", 1)[0].strip()
            transposed = next(
                (
                    line.split("=", 1)[0].strip()
                    for line in body[load_index + 1 : dot_index]
                    if f"tt.trans {value} " in line
                ),
                None,
            )
            feeds_dot.append(transposed is not None and transposed in dot)
        if all(feeds_dot):
            order = "early" if loads[1][0] < dots[0][0] else "late"
    return dict(
        order=order,
        loop_header=header.strip(),
        operations=[
            line.strip()
            for line in body
            if any(
                token in line for token in ("tt.load", "tt.trans", "tt.dot", "tt.store")
            )
        ],
    )


def load_kernels(report, case):
    result = {}
    directory = report / "generated-prefixes"
    directory.mkdir(exist_ok=True)
    for arm in ARMS[:-1]:
        source = build_source(arm)
        path = directory / f"{case}_{arm}.py"
        path.write_text(source)
        (report / f"{case}-{arm}-generated-source.txt").write_text(source)
        name = f"_qsa_page64_{case}_{arm}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        result[arm] = getattr(module, f"_qsa_page64_prefix_{arm}")
    from qsa_int8_query_pack4_910b import _qsa_qk_single_page_score

    result["current"] = _qsa_qk_single_page_score
    return result


def run_case(args):
    result, code = dict(case=args.case, status="failed"), 1
    report = Path(args.report)
    try:
        import torch
        import torch_npu
        import triton
        from qsa_int8_query_pack4_910b import validate_shared_groups
        from qsa_int8_selection_910b_trial import PreparedInt8Selection

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
        result["environment"] = dict(
            python=sys.executable,
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            triton=triton.__version__,
            compiler=triton.__file__,
            device=torch.npu.get_device_name(args.device),
            opp=os.getenv("ASCEND_OPP_PATH"),
            device_isolation="not_verified",
        )
        if "910B" not in result["environment"]["device"]:
            raise NotImplementedError("910B required")
        plan = PLANS[args.case]
        inputs = make_inputs(plan)
        values = {k: v.to(f"npu:{args.device}") for k, v in inputs.items()}
        rows, width, chunk = (
            plan["q_shape"][0],
            plan["logits_width"],
            plan["chunk_rows"],
        )
        first = rows // 2 // chunk * chunk
        count = min(chunk, rows - first)
        if args.page_reuse_full:
            from qsa_int8_shared_map_selection_910b_trial import visibility_tile_policy

            visible_cpu = inputs["visible"].tolist()
            rejected = []
            for begin in range(0, rows, chunk):
                size = min(chunk, rows - begin)
                stats = visibility_tile_policy(
                    visible_cpu[begin : begin + size], width, "conservative"
                )
                if not stats["selected"] and size >= 19:
                    rejected.append((begin, size, stats))
            if not rejected:
                raise ValueError("No non-skip chunk with >=19 rows for this diagnostic")
            if args.page_prefetch:
                first, count, stats = min(
                    rejected, key=lambda entry: abs(entry[0] - first)
                )
            else:
                first, count, stats = rejected[len(rejected) // 2]
            result["non_skip_selection"] = dict(
                first=first, count=count, policy=stats, eligible_chunks=len(rejected)
            )
            print(
                "NON_SKIP_CHUNK", json.dumps(result["non_skip_selection"]), flush=True
            )
        result["shared_map_guard"] = validate_shared_groups(
            inputs["row_table"][first : first + count].tolist(), plan["starts"], first
        )
        result["geometry"] = dict(
            first=first,
            count=count,
            width=width,
            block_n=64,
            key_tiles=8,
            grid_cap=20,
            plan=plan,
        )
        if args.query_pack_sweep or args.skip_invisible:
            table = inputs["row_table"][first : first + count].tolist()
            for group in (8, 16):
                for base in range(0, count, group):
                    owner = bisect.bisect_right(plan["starts"], first + base) - 1
                    for row in range(base, min(base + group, count)):
                        if (
                            table[row] != table[base]
                            or bisect.bisect_right(plan["starts"], first + row) - 1
                            != owner
                        ):
                            raise NotImplementedError(
                                f"Unsafe query{group} shared-map group"
                            )
            result["query_group_guard"] = dict(
                groups=[4, 8, 16], shared_maps_and_request_owners=True
            )
        result["cube_boundary"] = args.cube_boundary
        result["unroll_loop"] = args.unroll_loop
        result["scope"] = (
            "Representative N64/group8 scoring prefixes; observable FP32 sinks and compiler rescheduling included; "
            "differences are not additive operation costs; no TopK/native stage timing"
        )
        if args.cube_boundary or args.hardware_boundary:
            result["scope"] = (
                "Representative N64 Cube boundary diagnostic; packed K excludes separately recorded preparation; "
                "raw INT32/FP32 outputs write 4x bytes; no pure dot/layout/synchronization cost claim"
            )
        if args.unroll_loop:
            result["scope"] = (
                "Representative N64/group8 full-score static inner-loop unroll versus current; "
                "same arithmetic/addressing/outputs; separate counter passes; no full-input/native claim"
            )
        if args.query_pack_sweep:
            result["scope"] = (
                "Representative full-score query grouping 4/8/16; N64/K128/key_tiles8 unchanged; no full-input or native speedup claim"
            )
        scratch = torch.empty(
            (count, width), device=values["q"].device, dtype=torch.float32
        )
        output = torch.empty((rows, 512), device=scratch.device, dtype=torch.int32)
        workspace = torch.empty(
            (1024 * 1024,), device=scratch.device, dtype=torch.uint8
        )
        adapter = PreparedInt8Selection(values, output, scratch, workspace, None)
        packed_cache = None
        if args.cube_boundary:
            result["stage"] = "prepare_packed_k"
            torch.npu.synchronize()
            started = time.perf_counter()
            packed_cache = values["cache"].squeeze(2).transpose(1, 2).contiguous()
            torch.npu.synchronize()
            result["packed_k"] = dict(
                shape=list(packed_cache.shape),
                additional_bytes=packed_cache.numel() * packed_cache.element_size(),
                prepare_wall_ms=(time.perf_counter() - started) * 1000,
                scope="Entire cache transpose+allocation; synchronized wall time, excluded from score timing; not free",
            )
            if not torch.equal(
                packed_cache.transpose(1, 2).contiguous(), values["cache"].squeeze(2)
            ):
                raise AssertionError("Packed cache roundtrip mismatch")
        kernels = load_kernels(report, args.case)
        if args.skip_invisible:
            from qsa_int8_query_pack4_910b import _qsa_page64_query16_score

            kernels["current"] = _qsa_page64_query16_score
            if args.page_prefetch:
                from qsa_int8_query_pack4_910b import _qsa_page64_page_reuse_full_score

                kernels["current"] = _qsa_page64_page_reuse_full_score
            if args.page_reuse:
                from qsa_int8_query_pack4_910b import _qsa_page64_skip_invisible_score

                kernels["current"] = _qsa_page64_skip_invisible_score
            result["scope"] = (
                "Representative query16 N64 visibility-bounded dot loop plus timed dense -inf fill; diagnostic even where production chooses N128; no native/TopK claim"
            )

            def tile_counts(begin, size):
                visible = inputs["visible"][begin : begin + size].tolist()
                total = ((width + 63) // 64 + 7) // 8 * 8
                active = sum(
                    min(total, (min(width, max(visible[b : b + 16])) + 63) // 64)
                    for b in range(0, size, 16)
                )
                issued = ((size + 15) // 16) * total
                return dict(
                    issued=issued,
                    useful=active,
                    skipped=issued - active,
                    skipped_fraction=(issued - active) / issued,
                )

            result["visibility_tiles"] = dict(
                representative=tile_counts(first, count),
                full_input=tile_counts(0, rows),
                scope="Query16 geometry, includes padded key tiles; theoretical skip count, not measured runtime instructions",
            )
            print(
                "VISIBILITY_TILES", json.dumps(result["visibility_tiles"]), flush=True
            )

        if args.page_reuse:
            result["scope"] = (
                "Representative query16 N64 page-segment reuse vs visibility-bounded scorer; same eight-tile ownership and dense fill; not full-input/TopK/native timing"
            )

        if args.page_reuse_full:
            result["scope"] = (
                "Unbounded N64/query16 page reuse vs current unbounded scorer on a chunk rejected by conservative skip policy; every eight-tile unit executes all dots; no native/TopK/full-input speedup claim"
            )

        if args.page_prefetch:
            result["scope"] = (
                "Same full-dot page reuse with early/late second-K loads in two-tile groups; TTIR order is not hardware overlap"
            )

        def launch_at(name, target, start, n, w, visible=None):
            group = (
                16
                if args.skip_invisible
                else int(name.removeprefix("query")) if name.startswith("query") else 4
            )
            return kernels[name][
                (min(20, triton.cdiv(n, group) * triton.cdiv(triton.cdiv(w, 64), 8)),)
            ](
                values["q"][start : start + n],
                packed_cache if name == "packed_full" else values["cache"],
                values["qs"][start : start + n],
                values["ks"],
                values["weights"][start : start + n],
                values["row_table"][start : start + n],
                values["visible"][start : start + n] if visible is None else visible,
                target,
                ROWS=n,
                COLUMNS=values["row_table"].shape[1],
                WIDTH=w,
                PAGE=192,
                BLOCK_N=64,
                KEY_TILES=8,
                num_warps=4,
            )

        def launch(name, target):
            return launch_at(name, target, first, count, width)

        def check(name, target, start, n, w, visible=None):
            actual = target.cpu()
            vis = inputs["visible"][start : start + n] if visible is None else visible
            raw = name in ("raw_i32", "raw_fp32")
            if raw:
                actual = actual.reshape(n, 4, w)
                mask = (torch.arange(w)[None, None, :] < vis[:, None, None]).expand(
                    n, 4, w
                )
            else:
                mask = torch.arange(w)[None, :] < vis[:, None]
            invalid = -2147483648 if name == "raw_i32" else -float("inf")
            if not bool(torch.isfinite(actual[mask]).all()) or not bool(
                (actual[~mask] == invalid).all()
            ):
                raise AssertionError(f"Prefix finite/mask mismatch: {name}")
            maximum = 0.0
            for local in sorted({0, n // 2, n - 1}):
                row = start + local
                length = min(int(vis[local]), w)
                if not length:
                    continue
                table = inputs["row_table"][row].long()
                keys = inputs["cache"][table, :, 0].reshape(-1, 128)[:length].float()
                ks = inputs["ks"][table, :, 0].reshape(-1)[:length].float()
                q = inputs["q"][row].float()
                qs = inputs["qs"][row].float()
                weights = inputs["weights"][row].float()
                dot = q @ keys.T
                if raw:
                    expected = dot.to(torch.int32) if name == "raw_i32" else dot
                    value = actual[local, :, :length]
                    if not torch.equal(value, expected):
                        raise AssertionError(f"Raw dot mismatch: {name}, row={row}")
                    continue
                if name == "loads":
                    expected = (q.sum() + qs.sum() + weights.sum()) + keys.sum(1) + ks
                elif name == "dot":
                    expected = dot.sum(0)
                elif name == "query_scaled":
                    expected = (dot * qs[:, None]).sum(0)
                elif name == "dequant":
                    expected = (dot * qs[:, None] * ks[None, :]).sum(0)
                else:
                    expected = (
                        torch.relu(dot * qs[:, None] * ks[None, :]) * weights[:, None]
                    ).sum(0)
                if not torch.allclose(
                    actual[local, :length], expected, rtol=1e-3, atol=1e-3
                ):
                    raise AssertionError(f"CPU prefix mismatch: {name}, row={row}")
                maximum = max(
                    maximum, (actual[local, :length] - expected).abs().max().item()
                )
            return dict(
                all_masks_checked=True,
                cpu_local_rows=sorted({0, n // 2, n - 1}),
                max_abs_error=maximum,
            )

        result["stage"] = "compile_and_precheck"

        def make_target(name, n, w, guards=0):
            output_rows = n * 4 if name in ("raw_i32", "raw_fp32") else n
            dtype = torch.int32 if name == "raw_i32" else torch.float32
            return torch.full(
                (output_rows + guards, w), 12345, device=scratch.device, dtype=dtype
            )

        targets = {name: make_target(name, count, width) for name in ARMS}
        result["output_bytes"] = {
            name: t.numel() * t.element_size() for name, t in targets.items()
        }
        result["compiler"], result["prechecks"] = {}, {}
        for name in ARMS:
            kernel = launch(name, targets[name])
            result.setdefault("asm_inventory", {})[name] = dict(
                keys=list(kernel.asm), text_artifacts=[]
            )
            for key, value in kernel.asm.items():
                if isinstance(value, str) and key not in (
                    "ttir",
                    "ttadapter",
                    "bcmlir",
                ):
                    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
                    path = report / f"{args.case}-{name}-lower-{safe}.txt"
                    path.write_text(value[: 4 * 1024**2])
                    result["asm_inventory"][name]["text_artifacts"].append(
                        dict(
                            key=key,
                            file=path.name,
                            truncated=len(value) > 4 * 1024**2,
                            static_occurrences={
                                token: len(re.findall(re.escape(token), value, re.I))
                                for token in (
                                    "barrier",
                                    "wait_flag",
                                    "set_flag",
                                    "fixpipe",
                                    "mmad",
                                    "transpose",
                                )
                            },
                        )
                    )
            for ir in ("ttir", "ttadapter", "bcmlir"):
                if isinstance(kernel.asm.get(ir), str):
                    (report / f"{args.case}-{name}-{ir}.txt").write_text(kernel.asm[ir])
            ttir = kernel.asm["ttir"]
            if args.skip_invisible and name in (
                "skip_invisible",
                "page_reuse",
                "page_reuse_full",
            ):
                loop_lines = [
                    line.strip() for line in ttir.splitlines() if "scf.for " in line
                ]
                result["visibility_loop_ir"] = dict(
                    loop_headers=loop_lines,
                    scope="Dynamic dot-loop and store-only fill-loop source exported; runtime skip efficacy judged with timing, not static dot count",
                )
                if len(loop_lines) < (4 if name == "page_reuse" else 3):
                    raise AssertionError("Visibility/fill loops not retained in TTIR")
            if args.page_prefetch and name in PREFETCH_ARMS:
                schedule = prefetch_schedule(ttir)
                result.setdefault("prefetch_schedule", {})[name] = schedule
                print("PREFETCH_SCHEDULE", name, json.dumps(schedule), flush=True)
            dots = len(re.findall(r"\btt\.dot\b", ttir))
            loads = len(re.findall(r"\btt\.load\b", ttir))
            stores = len(re.findall(r"\btt\.store\b", ttir))
            expected_loads = {
                "loads": 7,
                "dot": 4,
                "query_scaled": 5,
                "dequant": 6,
                "full": 7,
                "current": 7,
                "query8": 7,
                "query16": 7,
                "skip_invisible": 7,
                "page_reuse": 7,
                "page_reuse_full": 7,
                "page_prefetch": 9,
                "page_prefetch_control": 9,
                "packed_full": 7,
                "unrolled_full": 28,
                "raw_i32": 4,
                "raw_fp32": 4,
            }
            if (
                dots
                != (
                    8
                    if name == "unrolled_full"
                    else 0 if name == "loads" else 2 if name in PREFETCH_ARMS else 1
                )
                or stores
                != (
                    8
                    if name == "unrolled_full"
                    else (
                        2
                        if name in ("skip_invisible", "page_reuse", *PREFETCH_ARMS)
                        or (name == "current" and args.page_reuse)
                        else 1
                    )
                )
                or (
                    not 20 <= loads <= 28
                    if name == "unrolled_full"
                    else loads != expected_loads[name]
                )
            ):
                raise AssertionError(
                    f"Prefix IR contract mismatch: {name}, dots={dots}, stores={stores}"
                )
            result["compiler"][name] = dict(
                loads=loads,
                dots=dots,
                stores=stores,
                metadata=repr(kernel.metadata),
                npubin_sha256=hashlib.sha256(bytes(kernel.asm["npubin"])).hexdigest(),
                loops=kernel.asm["bcmlir"].count("scf.for"),
                copies=kernel.asm["bcmlir"].count("memref.copy"),
                transpose_lines=[
                    line.strip()
                    for line in kernel.asm["bcmlir"].splitlines()
                    if "linalg.transpose " in line
                ],
                matmul_lines=[
                    line.strip()
                    for line in kernel.asm["bcmlir"].splitlines()
                    if "linalg.matmul " in line
                ],
                reduction_lines=[
                    line.strip()
                    for line in kernel.asm["bcmlir"].splitlines()
                    if "linalg.reduce " in line
                ],
            )
            result["prechecks"][name] = check(name, targets[name], first, count, width)
        reference = torch.empty_like(scratch)
        adapter.score_into(first, count, reference)
        for name in (
            ARMS
            if args.query_pack_sweep or args.skip_invisible
            else (
                ("packed_full", "full", "current")
                if args.cube_boundary
                else (
                    ("unrolled_full", "current")
                    if args.unroll_loop
                    else ("current",) if args.hardware_boundary else ("full", "current")
                )
            )
        ):
            if not torch.allclose(targets[name], reference, rtol=1e-3, atol=1e-3):
                raise AssertionError(f"Full-chunk legacy mismatch: {name}")
        result["full_chunk_vs_legacy"] = True
        # Seven rows exercise partial query groups, partial key blocks and an empty group.
        result["tail_checks"] = []
        for empty in (False, True):
            n, w = 7, 256
            vis = torch.tensor([193, 0, 255, 1, 128, 256, 191], dtype=torch.int32)
            if empty:
                vis[:4] = 0
            if args.query_pack_sweep or args.skip_invisible:
                n = 19
                vis = vis.repeat(3)[:n].clone()
                if empty:
                    vis[:16] = 0
            device_vis = vis.to(scratch.device)
            for name in ARMS:
                guarded = make_target(name, n, w, guards=4)
                output_rows = n * 4 if name in ("raw_i32", "raw_fp32") else n
                launch_at(name, guarded[:output_rows], first, n, w, device_vis)
                checked = check(name, guarded[:output_rows], first, n, w, vis)
                if not bool((guarded[output_rows:] == 12345).all().item()):
                    raise AssertionError(f"Tail guard overwritten: {name}")
                result["tail_checks"].append(
                    dict(arm=name, empty_group=empty, **checked)
                )
        print(
            "PAGE64_PREFIX_PRECHECKS",
            json.dumps(dict(compiler=result["compiler"], checks=result["prechecks"])),
            flush=True,
        )
        setup_trace(
            args,
            dict(
                chunks=[(first, count)], score_into=adapter.score_into, logits=scratch
            ),
            torch,
            result,
        )
        hosts = {
            name: {
                (
                    (
                        (
                            "_qsa_page64_skip_invisible_score"
                            if args.page_reuse
                            else "_qsa_page64_query16_score"
                        )
                        if args.skip_invisible
                        else "_qsa_qk_single_page_score"
                    )
                    if name == "current"
                    else f"_qsa_page64_prefix_{name}"
                ): 1
            }
            for name in ARMS
        }
        if args.page_prefetch:
            hosts["current"] = {"_qsa_page64_page_reuse_full_score": 1}
        if args.page_prefetch and any(
            result["prefetch_schedule"][arm]["order"] != expected
            for arm, expected in zip(PREFETCH_ARMS, ("early", "late"))
        ):
            raise AssertionError("TTIR did not retain distinct early and late K loads")
        result["stage"] = "prefix_timer"
        result["samples"], result["measurement_order"] = measure_prefix_plan(
            args, ARMS, launch, targets, torch, result, expected_hosts=hosts
        )
        result["postchecks"] = {
            name: check(name, targets[name], first, count, width) for name in ARMS
        }
        if "full" in targets and not torch.allclose(
            targets["full"], targets["current"], rtol=1e-3, atol=1e-3
        ):
            raise AssertionError("Last timed full prefix differs from current kernel")
        if args.cube_boundary:
            if not torch.allclose(
                targets["packed_full"], targets["current"], rtol=1e-3, atol=1e-3
            ):
                raise AssertionError(
                    "Last timed packed full scores differ from current"
                )
            live = torch.isfinite(targets["dot"])
            for name in ("raw_i32", "raw_fp32"):
                reduced = targets[name].reshape(count, 4, width).float().sum(1)
                if not torch.equal(reduced[live], targets["dot"][live]):
                    raise AssertionError(f"All-row raw dot checksum mismatch: {name}")
            result["raw_all_row_checksums_vs_dot"] = True
        result["latency"] = {
            name: aggregate(samples) for name, samples in result["samples"].items()
        }
        times = {
            name: v["kernel_us"]["median"] for name, v in result["latency"].items()
        }
        if args.query_pack_sweep or args.skip_invisible:
            result["query_pack_speedup_vs_current"] = {}
            for name in ARMS[:-1]:
                if not torch.allclose(
                    targets[name], targets["current"], rtol=1e-3, atol=1e-3
                ):
                    raise AssertionError(f"Last timed scores differ: {name}")
                result["query_pack_speedup_vs_current"][name] = (
                    times["current"] / times[name]
                )
            print(
                "QUERY_PACK_SPEEDUP_VS_CURRENT",
                json.dumps(result["query_pack_speedup_vs_current"]),
                flush=True,
            )
        if args.unroll_loop:
            if not torch.allclose(
                targets["unrolled_full"], targets["current"], rtol=1e-3, atol=1e-3
            ):
                raise AssertionError("Last timed unrolled scores differ from current")
            result["unroll_speedup_vs_current"] = (
                times["current"] / times["unrolled_full"]
            )
            print(
                "UNROLL_SPEEDUP_VS_CURRENT",
                result["unroll_speedup_vs_current"],
                flush=True,
            )
        if args.cube_boundary:
            result["boundary_comparisons"] = dict(
                packed_full_vs_current=times["current"] / times["packed_full"],
                raw_i32_us=times["raw_i32"],
                raw_fp32_us=times["raw_fp32"],
                dot_sum_us=times["dot"],
                caveat=(
                    "Raw outputs write 4x bytes; modes/layouts/sinks may differ; "
                    "no isolated conversion/sync cost claim"
                ),
            )
            print(
                "CUBE_BOUNDARY_COMPARISONS",
                json.dumps(result["boundary_comparisons"]),
                flush=True,
            )
            print("PACKED_K_PREPARATION", json.dumps(result["packed_k"]), flush=True)
        result["diagnostic_adjacent_deltas_us"] = {
            f"{b}_minus_{a}": times[b] - times[a]
            for a, b in (
                []
                if args.cube_boundary
                or args.hardware_boundary
                or args.unroll_loop
                or args.query_pack_sweep
                or args.skip_invisible
                else zip(ARMS[:4], ARMS[1:5])
            )
        }
        print("arm | kernel_us | relative_to_current", flush=True)
        for name in ARMS:
            print(
                f"{name} | {times[name]:.3f} | {times[name]/times['current']:.4f}",
                flush=True,
            )
        print(
            "DIAGNOSTIC_DELTAS_NOT_ADDITIVE_COSTS",
            json.dumps(result["diagnostic_adjacent_deltas_us"]),
            flush=True,
        )
        result["status"], code = "passed", 0
        if args.hardware_boundary or args.unroll_loop:
            from profile_qsa_int8_score_910b import summarize_tables
            from qsa_timeline_910b import collect_hardware

            result["hardware"] = {}
            for name in ARMS:
                result["hardware"][name] = {}
                kernel_name = (
                    (
                        (
                            "_qsa_page64_skip_invisible_score"
                            if args.page_reuse
                            else "_qsa_page64_query16_score"
                        )
                        if args.skip_invisible
                        else "_qsa_qk_single_page_score"
                    )
                    if name == "current"
                    else f"_qsa_page64_prefix_{name}"
                )
                for metric in args.metrics:
                    result["stage"] = f"hardware_{name}_{metric}"
                    if not hasattr(torch_npu.profiler.AiCMetrics, metric):
                        result["hardware"][name][metric] = dict(
                            status="unsupported_enum"
                        )
                        continue
                    attempts = []
                    for attempt in range(1, 3):
                        launch(name, targets[name])
                        torch.npu.synchronize()
                        options = SimpleNamespace(
                            report=str(report),
                            worker=f"{args.case}-{name}-{metric}-attempt{attempt}",
                            aic_metric=metric,
                            counter_iterations=args.counter_iterations,
                            kernel_substring=kernel_name,
                        )
                        capture = {}
                        collect_hardware(
                            options, lambda: launch(name, targets[name]), torch, capture
                        )
                        entry = capture["score_hardware"]
                        entry["capture_status"] = entry["status"]
                        artifact = report / entry.get("artifact", "no-artifact")
                        if artifact.is_file():
                            with gzip.open(artifact, "rt") as stream:
                                tables = json.load(stream)["tables"]
                            parsed = summarize_tables(
                                tables, metric, kernel_substring=kernel_name
                            )
                            entry["parsed"], entry["status"] = parsed, parsed["status"]
                            if (
                                entry["status"] == "available"
                                and parsed["score_row_count"] != args.counter_iterations
                            ):
                                entry["status"] = "incomplete_counter_rows"
                        else:
                            entry["status"] = "no_counter_archive"
                        entry["postcheck"] = check(
                            name, targets[name], first, count, width
                        )
                        attempts.append(entry)
                        if entry["status"] == "available" or not artifact.is_file():
                            break
                    entry["attempts"] = attempts[:-1]
                    result["hardware"][name][metric] = entry
                    print(
                        "BOUNDARY_HARDWARE",
                        json.dumps(
                            dict(
                                arm=name,
                                metric=metric,
                                status=entry["status"],
                                primary_table=entry.get("parsed", {}).get(
                                    "primary_table"
                                ),
                                statistics=entry.get("parsed", {}).get(
                                    "raw_field_statistics", {}
                                ),
                            )
                        ),
                        flush=True,
                    )
            if not all(
                result["hardware"][name].get("PipeUtilization", {}).get("status")
                == "available"
                for name in ARMS
            ):
                result["status"], code = "partial_capture", 1
            result["stage"] = "completed"
            print(
                "LOWER_ASM_INVENTORY", json.dumps(result["asm_inventory"]), flush=True
            )
    except (ImportError, ModuleNotFoundError):
        result.update(status="prerequisites_missing", traceback=traceback.format_exc())
        code = 2
    except Exception:
        result["status"], code = "failed", 1
        result["traceback"] = traceback.format_exc()
    result["sources"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            Path(__file__).name,
            "qsa_int8_query_pack4_910b.py",
            "qsa_int8_prefix_plan_timer_910b.py",
            "qsa_int8_trace_timer_910b.py",
        )
    }
    result["exit_code"] = code
    (report / f"{args.case}.json").write_text(json.dumps(result, indent=2))
    if code:
        print(json.dumps(result, indent=2), flush=True)
    return code


def main():
    global ARMS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--kernel-warmup", type=int, default=2)
    parser.add_argument("--kernel-active", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--case", choices=PLANS)
    parser.add_argument("--cases", nargs="+", choices=PLANS)
    parser.add_argument(
        "--cube-boundary",
        action="store_true",
        help="Packed K layout and raw INT32/FP32 dot sinks; diagnostic only",
    )
    parser.add_argument(
        "--hardware-boundary",
        action="store_true",
        help="Current full score and raw INT32 output: timing, counters and all available text asm",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=(
            "PipeUtilization",
            "ArithmeticUtilization",
            "Memory",
            "MemoryUB",
            "ResourceConflictRatio",
        ),
        default=["PipeUtilization", "ArithmeticUtilization", "Memory"],
    )
    parser.add_argument("--counter-iterations", type=int, default=2)
    parser.add_argument(
        "--unroll-loop",
        action="store_true",
        help="Static-unroll eight N64 tiles; full scores, timing and counter comparison",
    )
    parser.add_argument(
        "--query-pack-sweep",
        action="store_true",
        help="Full-score query groups 4/8/16: compile, correctness and paired timing",
    )
    parser.add_argument(
        "--skip-invisible",
        action="store_true",
        help="Query16 bounded dot loop with timed dense -inf fill",
    )
    parser.add_argument(
        "--page-reuse",
        action="store_true",
        help="Reuse scalar page-table reads across three N64 tiles; compare visibility-bounded query16",
    )
    parser.add_argument(
        "--page-reuse-full",
        action="store_true",
        help="Page reuse without skipping dots, on a non-skip chunk; compare unbounded query16",
    )
    parser.add_argument(
        "--page-prefetch",
        action="store_true",
        help="Compare full-dot page reuse with one-tile-ahead K loads within each page",
    )
    args = parser.parse_args()
    if args.page_prefetch and not args.page_reuse_full:
        parser.error("--page-prefetch requires --page-reuse-full")
    if (
        sum(
            (
                args.hardware_boundary,
                args.cube_boundary,
                args.unroll_loop,
                args.query_pack_sweep,
                args.skip_invisible,
                args.page_reuse,
                args.page_reuse_full,
            )
        )
        > 1
    ):
        parser.error("Choose one boundary/unroll mode")
    if args.counter_iterations < 1:
        parser.error("Positive counter iterations required")
    if args.hardware_boundary or args.unroll_loop:
        if "PipeUtilization" not in args.metrics:
            parser.error("Hardware mode requires PipeUtilization")
        ARMS = (
            ("unrolled_full", "current") if args.unroll_loop else ("raw_i32", "current")
        )
    if args.query_pack_sweep:
        ARMS = ("query8", "query16", "current")
    if args.page_reuse or args.page_reuse_full:
        args.skip_invisible = True
    if args.skip_invisible:
        ARMS = (
            ("page_reuse", "current")
            if args.page_reuse
            else ("skip_invisible", "current")
        )
    if args.page_reuse_full:
        ARMS = (
            (*PREFETCH_ARMS, "current")
            if args.page_prefetch
            else ("page_reuse_full", "current")
        )
    if args.cube_boundary:
        ARMS = ("dot", "raw_i32", "raw_fp32", "packed_full", "full", "current")
    if min(args.rounds, args.kernel_warmup, args.kernel_active, args.timeout) < 1:
        parser.error("Positive iterations required")
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    if args.case:
        return run_case(args)
    snapshot(report, "before")
    entries = []
    for name in dict.fromkeys(args.cases or CASES):
        path = report / f"{name}.json"
        path.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--case",
            name,
            "--report",
            str(report),
            "--device",
            str(args.device),
            "--rounds",
            str(args.rounds),
            "--kernel-warmup",
            str(args.kernel_warmup),
            "--kernel-active",
            str(args.kernel_active),
        ]
        if args.page_prefetch:
            command.append("--page-prefetch")
        if args.page_reuse_full:
            command.append("--page-reuse-full")
        elif args.page_reuse:
            command.append("--page-reuse")
        elif args.skip_invisible:
            command.append("--skip-invisible")
        if args.query_pack_sweep:
            command.append("--query-pack-sweep")
        if args.cube_boundary:
            command.append("--cube-boundary")
        if args.hardware_boundary or args.unroll_loop:
            command.extend(
                [
                    "--unroll-loop" if args.unroll_loop else "--hardware-boundary",
                    "--counter-iterations",
                    str(args.counter_iterations),
                    "--metrics",
                    *args.metrics,
                ]
            )
        code, timeout = run_child(command, report / f"{name}-console.txt", args.timeout)
        entry = (
            json.loads(path.read_text())
            if path.is_file()
            else dict(case=name, status="crashed")
        )
        entry["subprocess_exit_code"] = code
        if timeout:
            entry["status"] = "timeout"
        elif code and entry["status"] == "passed":
            entry["status"] = "crashed_after_checks"
        entries.append(entry)
    snapshot(report, "after")
    code = (
        0
        if all(e["status"] == "passed" for e in entries)
        else 2 if all(e["status"] == "prerequisites_missing" for e in entries) else 1
    )
    (report / "summary.json").write_text(
        json.dumps(
            dict(
                exit_code=code,
                cases=entries,
                scope="N64 scoring prefix diagnostics only; not native/TopK or full-input performance",
            ),
            indent=2,
        )
    )
    for entry in entries:
        print(entry["case"], entry["status"], flush=True)
        for name, v in entry.get("latency", {}).items():
            print(name, v["kernel_us"], flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
