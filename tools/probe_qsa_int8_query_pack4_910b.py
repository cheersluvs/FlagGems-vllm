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

"""Compile, validate and time shared-map four-query INT8 scoring."""

import argparse
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_qsa_int8_kernel_910b import snapshot  # noqa: E402
from probe_qsa_int8_score_910b import reference  # noqa: E402
from probe_qsa_int8_ten_shapes_910b import PLANS, run_child  # noqa: E402
from profile_qsa_int8_score_910b import CASES, check_chunk  # noqa: E402
from qsa_int8_full_scale_audit_910b import (  # noqa: E402
    audit_full_scales,
    collect_profiler_logs,
)
from qsa_int8_prefix_builder_910b import load_prefix  # noqa: E402
from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan  # noqa: E402
from qsa_int8_trace_timer_910b import setup_trace  # noqa: E402
from qsa_kernel_benchmark_910b import aggregate  # noqa: E402

from tools.qsa_int8_ten_workloads_910b import make_inputs  # noqa: E402

ARMS = ("integrated_full", "scale_product", "pack4", "pack4_product")


def run_case(args):
    report = Path(args.report)
    arms = (
        ("integrated_full", "pack4", "pack4_transpose", "pack4_rows")
        if args.postprocess
        else ARMS
    )
    store_modes = {"pack4_transpose": 1, "pack4_rows": 2}
    expected_loads = {"pack4_rows": 11}
    result, code = dict(case=args.case, status="failed"), 1
    try:
        import torch
        import torch_npu
        import triton

        root = os.getenv("QSA_ENV_ROOT")
        if not root or not torch.npu.is_available():
            raise ModuleNotFoundError("Pinned NPU environment required")
        if (
            not Path(triton.__file__)
            .resolve()
            .is_relative_to(Path(root).resolve() / "flagtree")
        ):
            raise RuntimeError(f"Wrong compiler: {triton.__file__}")
        torch.npu.set_device(args.device)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        from qsa_int8_query_pack4_910b import (
            _qsa_query_pack4_score,
            validate_shared_groups,
        )
        from qsa_int8_selection_910b_trial import PreparedInt8Selection

        result["environment"] = dict(
            python=sys.executable,
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            triton=triton.__version__,
            compiler=triton.__file__,
            device=torch.npu.get_device_name(args.device),
            device_index=args.device,
            opp=os.getenv("ASCEND_OPP_PATH"),
            device_isolation="not_verified",
        )
        plan = PLANS[args.case]
        inputs = make_inputs(plan)
        values = {
            name: value.to(f"npu:{args.device}") for name, value in inputs.items()
        }
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
        scratch = torch.empty(
            (count, width), device=values["q"].device, dtype=torch.float32
        )
        output = torch.empty((rows, 512), device=scratch.device, dtype=torch.int32)
        workspace = torch.empty(
            (1024 * 1024,), device=scratch.device, dtype=torch.uint8
        )
        adapter = PreparedInt8Selection(values, output, scratch, workspace, None)
        product_kernel = load_prefix(report, args.case, "scale_product")
        targets = {name: torch.empty_like(scratch) for name in arms}
        result["geometry"] = dict(
            first=first,
            count=count,
            width=width,
            plan=plan,
            grid_cap=20,
            block_n=128,
            group_queries=4,
            heads_per_query=4,
            dot_columns=16,
            baseline_work_tiles=count * triton.cdiv(width, 128),
            packed_work_tiles=triton.cdiv(count, 4) * triton.cdiv(width, 128),
        )
        result["scope"] = (
            "Shared-map representative chunks; full scores; query grouping occurs inside the kernel, "
            "with no extra Q/K tensor preparation; not public production API"
        )

        def launch(name, target):
            if name == "integrated_full":
                return adapter.score_into(first, count, target)
            if name == "scale_product":
                return product_kernel[(min(20, count * triton.cdiv(width, 128)),)](
                    values["q"][first : first + count],
                    values["cache"],
                    values["qs"][first : first + count],
                    values["ks"],
                    values["weights"][first : first + count],
                    values["row_table"][first : first + count],
                    values["visible"][first : first + count],
                    target,
                    target,
                    HEADS=4,
                    PAGE=192,
                    COLUMNS=values["row_table"].shape[1],
                    ROWS=count,
                    WIDTH=width,
                    BLOCK_N=128,
                    MASK_RAW_DOT=True,
                    WRITE_RAW_DOT=False,
                    num_warps=4,
                )
            return _qsa_query_pack4_score[
                (min(20, triton.cdiv(count, 4) * triton.cdiv(width, 128)),)
            ](
                values["q"][first : first + count],
                values["cache"],
                values["qs"][first : first + count],
                values["ks"],
                values["weights"][first : first + count],
                values["row_table"][first : first + count],
                values["visible"][first : first + count],
                target,
                ROWS=count,
                COLUMNS=values["row_table"].shape[1],
                WIDTH=width,
                PAGE=192,
                BLOCK_N=128,
                SCALE_PRODUCT=name == "pack4_product",
                STORE_MODE=store_modes.get(name, 0),
                num_warps=4,
            )

        result["stage"] = "compile_and_validate"
        result["compiler"] = {}
        result["prechecks"] = {}
        for name in arms:
            kernel = launch(name, targets[name])
            result["prechecks"][name] = check_chunk(
                inputs, targets[name].cpu(), first, count
            )
            ir = kernel.asm["ttir"]
            dots = len(re.findall(r"\btt\.dot\b", ir))
            stores = len(re.findall(r"\btt\.store\b", ir))
            loads = len(re.findall(r"\btt\.load\b", ir))
            if (dots, stores, loads) != (
                1,
                4 if name == "pack4_rows" else 1,
                expected_loads.get(name, 7),
            ):
                raise AssertionError(
                    f"Unexpected packed-score IR: {name}, {dots,stores,loads}"
                )
            for key in ("ttir", "ttadapter", "bcmlir"):
                if isinstance(kernel.asm.get(key), str):
                    (report / f"{args.case}-{name}-{key}.txt").write_text(
                        kernel.asm[key]
                    )
            reductions = re.findall(
                r"linalg\.reduce ins\([^\n]*tensor<[^>]*x(f32|i32)>",
                kernel.asm["bcmlir"],
            )
            # A packed group also reduces four INT32 visibility values by max.
            if (
                reductions.count("f32") != (4 if name == "pack4_rows" else 1)
                or reductions.count("i32") > 1
                or "mix_mode='mix'" not in repr(kernel.metadata)
            ):
                raise AssertionError(
                    f"Unexpected mode/reductions: {name}, {reductions}"
                )
            binary = kernel.asm.get("npubin")
            result["compiler"][name] = dict(
                dot_count=dots,
                store_count=stores,
                load_count=loads,
                reduction_element_types=reductions,
                store_mode=(
                    store_modes.get(name, 0) if name.startswith("pack4") else None
                ),
                lowering_lines=[
                    line.strip()
                    for line in kernel.asm["bcmlir"].splitlines()
                    if "linalg.reduce " in line or "linalg.transpose " in line
                ],
                metadata=repr(kernel.metadata),
                npubin_sha256=(
                    hashlib.sha256(bytes(binary)).hexdigest()
                    if isinstance(binary, (bytes, bytearray, memoryview))
                    else None
                ),
            )
        print("QUERY_PACK4_LOWERING", json.dumps(result["compiler"]), flush=True)
        baseline = targets["integrated_full"].cpu()

        def check_score(name):
            actual = targets[name].cpu()
            if not torch.allclose(actual, baseline, rtol=1e-3, atol=1e-3):
                raise AssertionError(f"Full chunk mismatch: {name}")
            mask = torch.isfinite(baseline)
            return dict(
                all_scores_checked=True,
                max_abs_error=(actual[mask] - baseline[mask]).abs().max().item(),
                cpu=check_chunk(inputs, actual, first, count),
            )

        for name in arms:
            result["prechecks"][name] = check_score(name)
        result["artifact_comparison"] = audit_full_scales(
            report, args.case, arms, result["compiler"], expected_loads=expected_loads
        )
        (report / f"{args.case}-query-pack4-source.txt").write_text(
            Path(__file__).with_name("qsa_int8_query_pack4_910b.py").read_text()
        )
        print("QUERY_PACK4_GEOMETRY", json.dumps(result["geometry"]), flush=True)
        print(
            "QUERY_PACK4_ARTIFACTS",
            json.dumps(result["artifact_comparison"]),
            flush=True,
        )
        result["stage"] = "tail_and_visibility_checks"
        result["tail_checks"] = []
        for n, visibility_case in (
            (5, "mixed"),
            (6, "mixed"),
            (7, "mixed"),
            (5, "empty_group"),
        ):
            small = dict(inputs)
            for key in ("q", "qs", "weights", "row_table"):
                small[key] = inputs[key][first : first + n]
            small["row_table"] = small["row_table"][:, :2].contiguous().clone()
            small["row_table"][4:] = small["row_table"][4:].flip(1)
            visible_values = [193, 0, 255, 1, 128, 256, 191][:n]
            if visibility_case == "empty_group":
                visible_values = [0, 0, 0, 0, 256]
            small["visible"] = torch.tensor(visible_values, dtype=torch.int32)
            small["weights"] = torch.linspace(-0.5, 1.5, n * 4).reshape(n, 4).half()
            proof = validate_shared_groups(small["row_table"].tolist(), [0, n], 0)
            device_small = {
                key: small[key].to(scratch.device)
                for key in ("q", "qs", "weights", "row_table", "visible")
            }
            refs = {row: reference(small, row)[1] for row in range(n)}
            tail_variants = (
                [(False, mode) for mode in (0, 1, 2)]
                if args.postprocess
                else [(False, 0), (True, 0)]
            )
            for product, store_mode in tail_variants:
                guarded = torch.full(
                    (n + 4, 256), 12345.0, device=scratch.device, dtype=torch.float32
                )
                _qsa_query_pack4_score[(min(20, triton.cdiv(n, 4) * 2),)](
                    device_small["q"],
                    values["cache"],
                    device_small["qs"],
                    values["ks"],
                    device_small["weights"],
                    device_small["row_table"],
                    device_small["visible"],
                    guarded[:n],
                    ROWS=n,
                    COLUMNS=2,
                    WIDTH=256,
                    PAGE=192,
                    BLOCK_N=128,
                    SCALE_PRODUCT=product,
                    STORE_MODE=store_mode,
                    num_warps=4,
                )
                cpu = guarded.cpu()
                if not bool((cpu[n:] == 12345.0).all()):
                    raise AssertionError("Packed tail writes past valid query rows")
                maximum = 0.0
                for row in range(n):
                    expected = refs[row]
                    visible = expected.numel()
                    if not torch.allclose(
                        cpu[row, :visible], expected, rtol=1e-3, atol=1e-3
                    ) or not bool(torch.isneginf(cpu[row, visible:]).all()):
                        raise AssertionError(
                            f"Tail/visibility mismatch: rows={n}, row={row}, product={product}"
                        )
                    if visible:
                        maximum = max(
                            maximum, (cpu[row, :visible] - expected).abs().max().item()
                        )
                result["tail_checks"].append(
                    dict(
                        rows=n,
                        remaining_queries=n % 4,
                        visibility_case=visibility_case,
                        different_maps_across_groups=True,
                        scale_product=product,
                        store_mode=store_mode,
                        all_rows_cpu_checked=True,
                        guard_rows_intact=True,
                        max_abs_error=maximum,
                        shared_maps=proof,
                    )
                )
        result["stage"] = "kernel_timer"

        def score_into(start, n, target):
            return adapter.score_into(start, n, target)

        setup_trace(
            args,
            dict(chunks=[(first, count)], score_into=score_into, logits=scratch),
            torch,
            result,
        )
        expected_hosts = {
            name: {
                (
                    "_qsa_query_pack4_score"
                    if name.startswith("pack4")
                    else "_quant_score_trial"
                ): 1
            }
            for name in arms
        }
        result["samples"], result["measurement_order"] = measure_prefix_plan(
            args, arms, launch, targets, torch, result, expected_hosts=expected_hosts
        )
        result["postchecks"] = {name: check_score(name) for name in arms}
        result["latency"] = {
            name: aggregate(samples) for name, samples in result["samples"].items()
        }
        medians = {
            name: value["kernel_us"]["median"]
            for name, value in result["latency"].items()
        }
        result["speedup_vs_integrated"] = {
            name: medians["integrated_full"] / value for name, value in medians.items()
        }
        result["packing_speedup_matched_formula"] = {
            name: medians[
                "scale_product" if name == "pack4_product" else "integrated_full"
            ]
            / medians[name]
            for name in arms
            if name.startswith("pack4")
        }
        if args.postprocess:
            result["speedup_vs_pack4"] = {
                name: medians["pack4"] / medians[name]
                for name in arms
                if name.startswith("pack4")
            }
            print(
                "POSTPROCESS_SPEEDUP_VS_PACK4",
                json.dumps(result["speedup_vs_pack4"]),
                flush=True,
            )
        print("arm                      chunk_us   speedup_vs_integrated", flush=True)
        for name in arms:
            print(
                f"{name:24} {medians[name]:12.3f} {result['speedup_vs_integrated'][name]:10.4f}",
                flush=True,
            )
        print(
            "PACKING_ONLY_SPEEDUP",
            json.dumps(result["packing_speedup_matched_formula"]),
            flush=True,
        )
        result["status"], result["stage"], code = "passed", "completed", 0
    except ImportError:
        result.update(status="prerequisites_missing", traceback=traceback.format_exc())
        code = 2
    except Exception:
        result["traceback"] = traceback.format_exc()
    result["sources"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            Path(__file__).name,
            "qsa_int8_query_pack4_910b.py",
            "qsa_int8_score_910b_trial.py",
            "qsa_int8_prefix_builder_910b.py",
            "qsa_int8_prefix_plan_timer_910b.py",
            "qsa_int8_stage_trace_timer_910b.py",
            "qsa_int8_full_scale_audit_910b.py",
        )
    }
    result["exit_code"] = code
    (report / f"{args.case}.json").write_text(json.dumps(result, indent=2))
    if code:
        print(json.dumps(result, indent=2), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--kernel-warmup", type=int, default=2)
    parser.add_argument("--kernel-active", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Compare packed reduction/store layouts only",
    )
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cases", nargs="+", choices=CASES)
    args = parser.parse_args()
    if min(args.rounds, args.kernel_warmup, args.kernel_active, args.timeout) < 1:
        parser.error("Positive iterations and timeout required")
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
        if args.postprocess:
            command.append("--postprocess")
        code, timeout = run_child(command, report / f"{name}-console.txt", args.timeout)
        entry = (
            json.loads(path.read_text())
            if path.is_file()
            else dict(case=name, status="crashed")
        )
        entry["subprocess_exit_code"] = code
        entry["profiler_diagnostics"] = collect_profiler_logs(report, name)
        if timeout:
            entry["status"] = "timeout"
        elif code and entry["status"] == "passed":
            entry["status"] = "crashed_after_checks"
        entries.append(entry)
    snapshot(report, "after")
    code = (
        0
        if all(e["status"] == "passed" for e in entries)
        else (2 if all(e["status"] == "prerequisites_missing" for e in entries) else 1)
    )
    (report / "summary.json").write_text(
        json.dumps(
            dict(
                exit_code=code,
                cases=entries,
                scope="Shared-map representative 4-query scoring; not production or full-prefill acceptance",
            ),
            indent=2,
        )
    )
    print(f"INT8 query-pack4 exit={code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
