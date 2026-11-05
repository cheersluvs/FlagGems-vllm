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
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

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


def build_source(arm):
    """Retain original addressing/loop prefixes and an explicit FP32 output sink."""
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
    if arm == "packed_full":
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
        result["cube_boundary"] = args.cube_boundary
        result["scope"] = (
            "Representative N64/group8 scoring prefixes; observable FP32 sinks and compiler rescheduling included; "
            "differences are not additive operation costs; no TopK/native stage timing"
        )
        if args.cube_boundary:
            result["scope"] = (
                "Representative N64 Cube boundary diagnostic; packed K excludes separately recorded preparation; "
                "raw INT32/FP32 outputs write 4x bytes; no pure dot/layout/synchronization cost claim"
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

        def launch_at(name, target, start, n, w, visible=None):
            return kernels[name][
                (min(20, triton.cdiv(n, 4) * triton.cdiv(triton.cdiv(w, 64), 8)),)
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
            for ir in ("ttir", "ttadapter", "bcmlir"):
                if isinstance(kernel.asm.get(ir), str):
                    (report / f"{args.case}-{name}-{ir}.txt").write_text(kernel.asm[ir])
            ttir = kernel.asm["ttir"]
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
                "packed_full": 7,
                "raw_i32": 4,
                "raw_fp32": 4,
            }
            if (
                dots != (0 if name == "loads" else 1)
                or stores != 1
                or loads != expected_loads[name]
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
            ("packed_full", "full", "current")
            if args.cube_boundary
            else ("full", "current")
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
                    "_qsa_qk_single_page_score"
                    if name == "current"
                    else f"_qsa_page64_prefix_{name}"
                ): 1
            }
            for name in ARMS
        }
        result["stage"] = "prefix_timer"
        result["samples"], result["measurement_order"] = measure_prefix_plan(
            args, ARMS, launch, targets, torch, result, expected_hosts=hosts
        )
        result["postchecks"] = {
            name: check(name, targets[name], first, count, width) for name in ARMS
        }
        if not torch.allclose(
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
            for a, b in ([] if args.cube_boundary else zip(ARMS[:4], ARMS[1:5]))
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
    except (ImportError, ModuleNotFoundError):
        result.update(status="prerequisites_missing", traceback=traceback.format_exc())
        code = 2
    except Exception:
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
    args = parser.parse_args()
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
        if args.cube_boundary:
            command.append("--cube-boundary")
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
