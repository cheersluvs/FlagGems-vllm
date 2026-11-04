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

"""Initial prepared INT8 selection kernel timings against native fused V1."""

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_qsa_int8_pipeline_910b import check_selection  # noqa: E402
from probe_qsa_int8_score_910b import reference  # noqa: E402
from probe_qsa_int8_ten_shapes_910b import (  # noqa: E402
    PLANS,
    run_child,
    selected_scores,
    verify_structure,
)
from qsa_kernel_benchmark_910b import aggregate, setup  # noqa: E402

from tools.qsa_int8_ten_workloads_910b import make_inputs, sample_rows  # noqa: E402


def run_case(args):
    result, code = dict(case=args.case, status="failed", mode="kernel"), 1
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import torch_npu
        import triton

        root = os.getenv("QSA_ENV_ROOT")
        native = getattr(torch_npu, "npu_quant_lightning_indexer", None)
        if not root or not torch.npu.is_available() or native is None:
            raise ModuleNotFoundError(
                "Pinned NPU environment and native quant V1 required"
            )
        if (
            not Path(triton.__file__)
            .resolve()
            .is_relative_to(Path(root).resolve() / "flagtree")
        ):
            raise RuntimeError(f"Unexpected compiler: {triton.__file__}")
        torch.npu.set_device(args.device)
        if "910B" not in torch.npu.get_device_name(args.device):
            raise RuntimeError("910B required")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        from qsa_int8_score_910b_trial import score
        from qsa_prefill_pipeline_910b_trial import load_topk

        plan = PLANS[args.case]
        result["geometry"] = plan
        result["environment"] = dict(
            python=sys.executable,
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            triton=triton.__version__,
            triton_source=triton.__file__,
            device=torch.npu.get_device_name(args.device),
            ascend_home=os.getenv("ASCEND_HOME_PATH"),
            opp=os.getenv("ASCEND_OPP_PATH"),
            topk_route_requested=os.getenv("FLAGGEMS_ASCEND_PTK_PATH", "auto"),
            device_isolation="not_verified",
        )
        result["measurement_scope"] = (
            "Prepared inputs; sum of device kernel durations, not public production entry"
        )
        result["stage"] = "inputs_and_allocation"
        inputs = make_inputs(plan)
        values = {key: value.to(f"npu:{args.device}") for key, value in inputs.items()}
        rows, width, chunk = (
            plan["q_shape"][0],
            plan["logits_width"],
            plan["chunk_rows"],
        )
        chunks = [(first, min(chunk, rows - first)) for first in range(0, rows, chunk)]
        topk_chunks = [
            (first + local, min(32, count - local))
            for first, count in chunks
            for local in range(0, count, 32)
        ]
        staged_bytes = rows * width * 4
        result["memory"] = dict(
            benchmark_only_full_logits_bytes=staged_bytes,
            bounded_score_scratch_bytes=chunk * width * 4,
            raw_dot_allocation_bytes=0,
        )
        if hasattr(torch.npu, "mem_get_info"):
            try:
                free, total = torch.npu.mem_get_info(args.device)
                result["memory"].update(free_before_staging=int(free), total=int(total))
            except Exception as exc:
                result["memory"]["query_unavailable"] = str(exc)
        needed = staged_bytes + chunk * width * 4 + rows * 512 * 12 + 256 * 1024**2
        if result["memory"].get("free_before_staging", needed) < needed:
            raise MemoryError(
                f"Benchmark staging needs approximately {needed} additional bytes"
            )
        device = values["q"].device
        frozen = torch.empty((rows, width), device=device, dtype=torch.float32)
        scratch = torch.empty((chunk, width), device=device, dtype=torch.float32)
        output = torch.empty((rows, 512), device=device, dtype=torch.int32)
        workspace = torch.empty((1024 * 1024,), device=device, dtype=torch.uint8)
        q_ends = torch.arange(1, rows + 1, device=device, dtype=torch.int32)
        columns = torch.arange(width, device=device)
        topk = load_topk()
        result["topk_identity"] = topk.probe_identity
        result["dispatch"] = dict(
            score_chunk_rows=chunk,
            topk_call_rows=32,
            topk_block=256,
            score_calls=len(chunks),
            topk_calls=len(topk_chunks),
            native_full_calls=1,
            native_chunked_calls=(0 if args.integrated else len(chunks)),
            score_grid=(20 if args.integrated else 8),
            score_block_n=(128 if args.integrated else 32),
            debug_store=False,
        )
        print(
            json.dumps(
                dict(
                    case=args.case, memory=result["memory"], dispatch=result["dispatch"]
                )
            ),
            flush=True,
        )

        adapter = None
        if args.integrated:
            from qsa_int8_selection_910b_trial import PreparedInt8Selection

            adapter = PreparedInt8Selection(values, output, scratch, workspace, topk)
            legacy_adapter = adapter
            if args.shared_map_qk:
                from qsa_int8_shared_map_selection_910b_trial import (
                    PreparedSharedMapInt8Selection,
                )

                adapter = PreparedSharedMapInt8Selection(
                    values, output, scratch, workspace, topk, plan["starts"]
                )
                result["shared_map_dispatch"] = dict(
                    preparation_ms=adapter.preparation_ms,
                    preparation_in_kernel_timing=False,
                    immutable_inputs_required=True,
                    qk_chunks=sum(adapter.shared_routes.values()),
                    fallback_chunks=sum(
                        not value for value in adapter.shared_routes.values()
                    ),
                    routes=adapter.route_checks,
                    calibration="Unchanged single-query scorer; candidate stages timed separately",
                )
                print(
                    "SHARED_MAP_DISPATCH",
                    json.dumps(result["shared_map_dispatch"]),
                    flush=True,
                )

            result["integration"] = dict(
                entry=(
                    "qsa_int8_shared_map_selection_910b_trial.PreparedSharedMapInt8Selection.__call__"
                    if args.shared_map_qk
                    else "qsa_int8_selection_910b_trial.PreparedInt8Selection.__call__"
                ),
                score_grid=20,
                score_block_n=128,
                topk_call_rows=32,
                scope="Prepared INT8 trial adapter; BF16 production unchanged",
            )

        def score_into(first, count, target):
            if adapter is not None:
                return adapter.score_into(first, count, target)
            return score(
                values["q"][first : first + count],
                values["cache"],
                values["qs"][first : first + count],
                values["ks"],
                values["weights"][first : first + count],
                values["row_table"][first : first + count],
                values["visible"][first : first + count],
                target,
                raw_dot=None,
            )

        def score_stage():
            if adapter is not None:
                return adapter.score_all()
            for first, count in chunks:
                score_into(first, count, scratch[:count])

        def topk_stage():
            if adapter is not None:
                return adapter.topk_all(frozen)
            for first, count in topk_chunks:
                topk.persistent_topk(
                    frozen[first : first + count],
                    values["visible"][first : first + count],
                    output[first : first + count],
                    workspace,
                    k=512,
                    max_seq_len=width,
                    block=256,
                )

        def combined_stage():
            if adapter is not None:
                return adapter()
            for first, count in chunks:
                score_into(first, count, scratch[:count])
                for local in range(0, count, 32):
                    end = min(count, local + 32)
                    topk.persistent_topk(
                        scratch[local:end],
                        values["visible"][first + local : first + end],
                        output[first + local : first + end],
                        workspace,
                        k=512,
                        max_seq_len=width,
                        block=256,
                    )

        native_last = dict(full=None, chunked=[])

        def native_call(first, count):
            return native(
                values["q"][first : first + count],
                values["cache"],
                values["weights"][first : first + count],
                values["qs"][first : first + count],
                values["ks"],
                0,
                0,
                actual_seq_lengths_query=q_ends[:count],
                actual_seq_lengths_key=values["visible"][first : first + count],
                block_table=values["row_table"][first : first + count],
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=512,
                sparse_mode=0,
                pre_tokens=2**63 - 1,
                next_tokens=2**63 - 1,
            )

        def native_stage():
            native_last["full"] = None
            native_last["full"] = native_call(0, rows)
            return [native_last["full"]]

        def native_chunked_stage():
            native_last["chunked"] = []
            for first, count in chunks:
                native_last["chunked"].append(native_call(first, count))
            return native_last["chunked"]

        samples = sample_rows(plan)
        references = {row: reference(inputs, row)[1] for row in samples}
        print(
            json.dumps(dict(case=args.case, cpu_reference_rows_ready=samples)),
            flush=True,
        )
        result["cpu_reference_rows"] = samples
        result["correctness"] = {}

        def check_scores():
            maximum = 0.0
            for first, count in chunks:
                selected = [row for row in samples if first <= row < first + count]
                if not selected:
                    continue
                score_into(first, count, scratch[:count])
                for row in selected:
                    actual = scratch[row - first].cpu()
                    expected = references[row]
                    n = expected.numel()
                    if not torch.allclose(actual[:n], expected, rtol=1e-3, atol=1e-3):
                        raise AssertionError(f"Score mismatch at sampled row {row}")
                    if not bool(torch.isneginf(actual[n:]).all()):
                        raise AssertionError(f"Bad score mask at sampled row {row}")
                    if n:
                        maximum = max(
                            maximum, (actual[:n] - expected).abs().max().item()
                        )
            return dict(sampled_rows=samples, max_abs_error=maximum)

        def native_cpu(outputs, expected_chunks=chunks):
            for data, (_, count) in zip(outputs, expected_chunks):
                if data.shape != (count, 1, 512):
                    raise AssertionError(f"Unexpected native output {data.shape}")
            if len(outputs) != len(expected_chunks):
                raise AssertionError("Missing native chunk output")
            return torch.cat([data.cpu().squeeze(1) for data in outputs], dim=0)

        def check_indices(cpu):
            try:
                verify_structure(cpu, inputs["visible"])
            except AssertionError:
                if cpu.shape == (rows, 512):
                    ordered = cpu.sort(dim=1).values
                    bad = ((cpu < -1) | (cpu >= inputs["visible"][:, None])).any(1)
                    bad |= (cpu >= 0).sum(1) != inputs["visible"].clamp(max=512)
                    bad |= (
                        (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)
                    ).any(1)
                    result["failed_index_rows"] = [
                        dict(
                            row=row,
                            visible=int(inputs["visible"][row]),
                            indices=cpu[row].tolist(),
                        )
                        for row in bad.nonzero().flatten()[:8].tolist()
                    ]
                raise
            error = 0.0
            for row in samples:
                check, _ = check_selection(cpu[row : row + 1], [references[row]])
                if not check["checks_passed"]:
                    result["failed_cpu_sample"] = dict(
                        row=row, checks=check, indices=cpu[row].tolist()
                    )
                    raise AssertionError(f"Sampled TopK mismatch row {row}: {check}")
                error = max(error, check["selected_score_max_abs_error"])
            return dict(
                all_rows_structure=True,
                rows=rows,
                sampled_rows=samples,
                selected_score_max_abs_error=error,
            )

        def check_pair(a, b):
            maximum = 0.0
            for first, count in chunks:
                left, right = a[first : first + count], b[first : first + count]
                ac = selected_scores(
                    frozen[first : first + count], left.to(device), left
                )
                bc = selected_scores(
                    frozen[first : first + count], right.to(device), right
                )
                valid = (
                    torch.arange(512)[None, :]
                    < inputs["visible"][first : first + count].clamp(max=512)[:, None]
                )
                if bool(valid.any()):
                    maximum = max(maximum, (ac[valid] - bc[valid]).abs().max().item())
                if not torch.allclose(ac, bc, rtol=1e-3, atol=1e-3):
                    raise AssertionError(
                        f"Paired selected scores differ in chunk {first}; max_abs={maximum}"
                    )
            return dict(
                all_rows_paired=True,
                rows=rows,
                max_abs_error=maximum,
                rtol=1e-3,
                atol=1e-3,
            )

        result["stage"] = "no_debug_preflight"
        compiled = None
        route_artifacts = set()
        full_score_error = 0.0
        for first, count in chunks:
            current = frozen[first : first + count]
            kernel = score_into(first, count, current)
            if args.shared_map_qk:
                reference_target = scratch[:count]
                legacy_adapter.score_into(first, count, reference_target)
                if not torch.allclose(current, reference_target, rtol=1e-3, atol=1e-3):
                    raise AssertionError(f"Full score mismatch at chunk {first}")
                finite = torch.isfinite(reference_target)
                if bool(finite.any().item()):
                    full_score_error = max(
                        full_score_error,
                        (current[finite] - reference_target[finite]).abs().max().item(),
                    )
                route = "qk" if adapter.shared_routes[first, count] else "fallback"
                if route not in route_artifacts:
                    route_artifacts.add(route)
                    for key in ("ttir", "ttadapter", "bcmlir"):
                        if isinstance(kernel.asm.get(key), str):
                            (report / f"{args.case}-{route}-{key}.txt").write_text(
                                kernel.asm[key]
                            )
                    candidate_ir = kernel.asm["ttir"]
                    if len(re.findall(r"\btt\.store\b", candidate_ir)) != 1 or not any(
                        "tt.dot" in line and "xi8>" in line and "xi32>" in line
                        for line in candidate_ir.splitlines()
                    ):
                        raise AssertionError(f"Bad {route} compiled score contract")
            if compiled is None:
                compiled = kernel
            live = columns[None, :] < values["visible"][first : first + count, None]
            if not bool((torch.isfinite(current) | ~live).all().item()):
                raise AssertionError("Nonfinite visible frozen score")
            if not bool(((current == -float("inf")) | live).all().item()):
                raise AssertionError("Frozen score mask mismatch")
            if first == 0 or first + count == rows or (first // chunk + 1) % 32 == 0:
                print(
                    json.dumps(
                        dict(
                            case=args.case,
                            no_debug_preflight_rows=first + count,
                            total_rows=rows,
                        )
                    ),
                    flush=True,
                )
        ir = compiled.asm["ttir"]
        stores = len(re.findall(r"\btt\.store\b", ir))
        integer_dot = any(
            "tt.dot" in line and "xi8>" in line and "xi32>" in line
            for line in ir.splitlines()
        )
        result["no_debug_ir"] = dict(store_count=stores, int8_to_int32_dot=integer_dot)
        (report / f"{args.case}-score-no-debug-ttir.txt").write_text(ir)
        if stores != 1 or not integer_dot:
            raise AssertionError("No-debug INT8 score IR contract failed")
        result["correctness"]["score"] = check_scores()
        if args.shared_map_qk:
            result["correctness"]["all_chunk_scores_vs_legacy"] = dict(
                chunks=len(chunks),
                all_values_compared=True,
                max_abs_error=full_score_error,
            )
        result["stage"] = "native_full_preflight"
        baseline = native_cpu(native_stage(), [(0, rows)])
        result["correctness"]["native_full_preflight"] = check_indices(baseline)
        if not args.integrated:
            result["stage"] = "native_chunked_preflight"
            chunked_cpu = native_cpu(native_chunked_stage())
            result["correctness"]["native_chunked_preflight"] = dict(
                indices=check_indices(chunked_cpu),
                paired=check_pair(chunked_cpu, baseline),
            )
            del chunked_cpu
        result["baseline_policy"] = dict(
            primary="full native V1, one call",
            auxiliary=("omitted" if args.integrated else "chunked native V1"),
            no_fallback=True,
        )
        for name, function in (("topk", topk_stage), ("combined", combined_stage)):
            function()
            current = output.cpu()
            result["correctness"][name] = dict(
                indices=check_indices(current), paired=check_pair(current, baseline)
            )
            print(
                json.dumps(
                    dict(
                        case=args.case,
                        preflight=name,
                        checks=result["correctness"][name],
                    )
                ),
                flush=True,
            )
        jobs = dict(
            score=score_stage,
            topk=topk_stage,
            combined=combined_stage,
            native_fused=native_stage,
            native_chunked=native_chunked_stage,
        )
        if args.shared_map_qk:

            def legacy_score_stage():
                legacy_adapter.score_all()

            jobs["legacy_score"] = legacy_score_stage
        result["stage"] = "kernel_timer"
        if args.integrated:
            jobs.pop("native_chunked")
        timer_setup = setup
        if args.integrated:
            from qsa_int8_stage_trace_timer_910b import setup_stage_trace

            timer_setup = setup_stage_trace
        measure = timer_setup(
            args,
            dict(
                chunks=chunks,
                score_into=(
                    legacy_adapter.score_into if args.shared_map_qk else score_into
                ),
                logits=scratch,
            ),
            torch,
            result,
        )
        result["samples"] = {name: [] for name in jobs}
        result["measurement_order"] = []
        for round_index in range(args.rounds):
            order = list(jobs)
            if (list(PLANS).index(args.case) + round_index) % 2:
                order.reverse()
            result["measurement_order"].append(order)
            for name in order:
                result["stage"] = f"timing_{name}"
                print(
                    json.dumps(
                        dict(case=args.case, round=round_index + 1, measuring=name)
                    ),
                    flush=True,
                )
                timing = measure(jobs[name])
                result["samples"][name].append(timing)
                result["stage"] = f"postcheck_{name}"
                if name in ("score", "legacy_score"):
                    first, count = chunks[-1]
                    if not torch.allclose(
                        scratch[:count].cpu(),
                        frozen[first : first + count].cpu(),
                        rtol=1e-3,
                        atol=1e-3,
                    ):
                        raise AssertionError(
                            "Last timed score chunk differs from preflight"
                        )
                    result["correctness"][f"post_{name}"] = dict(
                        last_timed_chunk_checked=True, sampled=check_scores()
                    )
                elif name in ("native_fused", "native_chunked"):
                    fresh = (
                        native_cpu([native_last["full"]], [(0, rows)])
                        if name == "native_fused"
                        else native_cpu(native_last["chunked"])
                    )
                    result["correctness"][f"post_{name}"] = dict(
                        indices=check_indices(fresh),
                        paired=check_pair(output.cpu(), fresh),
                    )
                else:
                    current = output.cpu()
                    result["correctness"][f"post_{name}"] = dict(
                        indices=check_indices(current),
                        paired=check_pair(current, baseline),
                    )
                print(
                    json.dumps(
                        dict(case=args.case, round=round_index + 1, job=name, **timing)
                    ),
                    flush=True,
                )
        result["latency"] = {
            name: aggregate(values) for name, values in result["samples"].items()
        }
        medians = {
            name: entry["kernel_us"]["median"]
            for name, entry in result["latency"].items()
        }
        result["speedup"] = dict(
            overall=medians["native_fused"] / medians["combined"],
            auxiliary_chunked_native=(
                medians["native_chunked"] / medians["combined"]
                if "native_chunked" in medians
                else None
            ),
            score=None,
            topk=None,
        )
        if args.shared_map_qk:
            result["score_speedup_vs_legacy"] = (
                medians["legacy_score"] / medians["score"]
            )
            print(
                "FULL_INPUT_KERNEL_MS",
                json.dumps({name: value / 1000 for name, value in medians.items()}),
                flush=True,
            )
            print(
                "SCORE_SPEEDUP_VS_LEGACY", result["score_speedup_vs_legacy"], flush=True
            )
            print("OVERALL_SPEEDUP_VS_CANN", result["speedup"]["overall"], flush=True)
        result["native_split"] = dict(
            score_us=None,
            topk_us=None,
            reason="Fused V1 API has no isolated stage entry",
        )
        result["topk_state"] = dict(
            segmerge=dict(topk._SEGMERGE_STATE), persistent=dict(topk._PSEGMERGE_STATE)
        )
        result["stage"], result["status"], code = "completed", "passed", 0
    except ImportError:
        result.update(status="prerequisites_missing", traceback=traceback.format_exc())
        code = 2
    except Exception:
        result["traceback"] = traceback.format_exc()
    for name in (
        Path(__file__).name,
        "qsa_int8_score_910b_trial.py",
        "qsa_int8_ten_workloads_910b.py",
        "qsa_prefill_fp8_workloads.py",
        "qsa_kernel_benchmark_910b.py",
        "probe_qsa_int8_ten_shapes_910b.py",
        "qsa_int8_selection_910b_trial.py",
        "qsa_int8_shared_map_selection_910b_trial.py",
        "qsa_int8_query_pack4_910b.py",
        "qsa_int8_stage_trace_timer_910b.py",
        "qsa_int8_trace_timer_910b.py",
    ):
        result.setdefault("sources", {})[name] = hashlib.sha256(
            (Path(__file__).parent / name).read_bytes()
        ).hexdigest()
    result["exit_code"] = code
    (report / f"{args.case}.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "case",
                    "status",
                    "stage",
                    "latency",
                    "speedup",
                    "traceback",
                )
                if key in result
            },
            indent=2,
        ),
        flush=True,
    )
    return code


def snapshot(report, label):
    try:
        done = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=20
        )
        (report / f"npu-smi-{label}.txt").write_text(done.stdout + done.stderr)
    except Exception as exc:
        (report / f"npu-smi-{label}.txt").write_text(str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", choices=("kernel",), default="kernel")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--kernel-warmup", type=int, default=1)
    parser.add_argument("--kernel-active", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cases", nargs="+", choices=PLANS)
    parser.add_argument("--case", choices=PLANS)
    parser.add_argument(
        "--integrated",
        action="store_true",
        help="Use grid20/N128 INT8 adapter and explicit stage trace timer",
    )
    parser.add_argument(
        "--shared-map-qk",
        action="store_true",
        help="Use guarded QK score adapter; requires --integrated",
    )
    args = parser.parse_args()
    if args.shared_map_qk and not args.integrated:
        parser.error("--shared-map-qk requires --integrated")
    if args.rounds < 1 or args.kernel_active < 1 or args.kernel_warmup < 1:
        parser.error("rounds, warmup and active must be positive")
    if args.case:
        return run_case(args)
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(args.cases or PLANS))
    snapshot(report, "before")
    entries = []
    for name in names:
        print(f"Starting prepared INT8 kernel benchmark: {name}", flush=True)
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
        if args.integrated:
            command.append("--integrated")
        if args.shared_map_qk:
            command.append("--shared-map-qk")
        code, timed_out = run_child(
            command, report / f"{name}-console.txt", args.timeout
        )
        entry = (
            json.loads(path.read_text())
            if path.is_file()
            else dict(case=name, status="crashed")
        )
        if timed_out:
            entry["status"] = "timeout"
        elif code != 0 and entry["status"] == "passed":
            entry["status"] = "crashed_after_checks"
        entry["subprocess_exit_code"] = code
        entries.append(entry)
    snapshot(report, "after")
    passed = [entry for entry in entries if entry["status"] == "passed"]
    status = (
        0
        if len(passed) == len(names)
        else (2 if all(e["status"] == "prerequisites_missing" for e in entries) else 1)
    )
    complete = len(names) == len(PLANS) and len(passed) == len(PLANS)
    summary = dict(
        exit_code=status,
        cases=entries,
        all_ten_passed=complete,
        arithmetic_mean_overall_speedup=(
            statistics.mean(e["speedup"]["overall"] for e in passed)
            if complete
            else None
        ),
        scope=(
            "Integrated prepared INT8 adapter; exploratory trace device sums; not public production API"
            if args.integrated
            else "Initial prepared device-kernel baseline; no public production entry or native stage speedup claim"
        ),
    )
    (report / "summary.json").write_text(json.dumps(summary, indent=2))
    fields = (
        "case",
        "status",
        "score_us",
        "topk_us",
        "combined_us",
        "native_fused_us",
        "native_chunked_us",
        "overall_speedup",
    )
    with (report / "kernel-results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        print(" | ".join(fields), flush=True)
        for entry in entries:
            row = dict(case=entry["case"], status=entry["status"])
            if entry["status"] == "passed":
                row.update(
                    {
                        f"{key}_us": value["kernel_us"]["median"]
                        for key, value in entry["latency"].items()
                    }
                )
                row["overall_speedup"] = entry["speedup"]["overall"]
            writer.writerow(row)
            print(" | ".join(str(row.get(field, "")) for field in fields), flush=True)
    print(
        json.dumps(
            dict(
                all_ten_passed=complete,
                arithmetic_mean_overall_speedup=summary[
                    "arithmetic_mean_overall_speedup"
                ],
                exit_code=status,
            )
        ),
        flush=True,
    )
    return status


if __name__ == "__main__":
    sys.exit(main())
