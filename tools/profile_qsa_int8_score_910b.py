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

"""Capture frozen INT8 score timelines, hardware counters and compiler IR."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_qsa_int8_kernel_910b import snapshot  # noqa: E402
from probe_qsa_int8_score_910b import reference  # noqa: E402
from probe_qsa_int8_ten_shapes_910b import PLANS, run_child  # noqa: E402
from qsa_timeline_910b import collect, collect_hardware  # noqa: E402

from tools.qsa_int8_ten_workloads_910b import make_inputs, sample_rows  # noqa: E402

FROZEN_SCORE_SHA256 = "13718b109b2ffd93164b87c077c9d991b59caaa7f073302f3b7aced74a787334"
CASES = ("int8_budget_exact", "int8_long100k")
METRICS = (
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryUB",
    "ResourceConflictRatio",
)


def validate_score_trace(timeline, report, expected, expected_names=None):
    """Validate this score-only trace, including generic CANN task names."""
    timeline["expected_score_launches"] = expected
    with gzip.open(Path(report) / timeline["trace_file"], "rt") as stream:
        payload = json.load(stream)
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    if expected_names is None:
        expected_names = {"_quant_score_trial": expected}
    host_names = Counter(
        event["name"]
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "cpu_op"
        and event.get("name") in expected_names
    )
    names = timeline.get("kernels", {})
    named_counts = {
        kernel: sum(entry["count"] for name, entry in names.items() if kernel in name)
        for kernel in expected_names
    }
    generic_count = names.get("MIX_AIC", {}).get("count", 0)
    timeline["matched_score_task_names"] = [
        name for name in names if any(kernel in name for kernel in expected_names)
    ]
    timeline["expected_host_kernel_counts"] = expected_names
    timeline["host_score_kernel_counts"] = dict(host_names)
    timeline["host_score_launch_count"] = sum(host_names.values())
    timeline["identity_basis"] = "unconfirmed"
    if timeline.get("task_count") == expected and named_counts == expected_names:
        timeline["identity_basis"] = "named_device_tasks"
    elif (
        timeline.get("task_count") == expected
        and host_names == expected_names
        and generic_count == expected
        and set(names) == {"MIX_AIC"}
    ):
        # collect() scopes this trace to score_all only. This is a count-based
        # validation of that isolated capture, not a per-task name/flow mapping.
        timeline["identity_basis"] = "isolated_score_only_host_and_MIX_AIC_counts"
    timeline["usable"] = timeline["identity_basis"] != "unconfirmed"
    return timeline


def number(value):
    try:
        parsed = float(str(value).strip().rstrip("%").replace(",", ""))
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        return None


def metric_key(name, group):
    key = name.lower()
    if key.endswith("_id") or key.endswith(" id"):
        return False
    tokens = {
        "PipeUtilization": ("mac", "scalar", "mte", "vec_", "fixpipe", "cube"),
        "ArithmeticUtilization": ("_fp", "int8", "int_ratio", "fops", "flops", "cube"),
        "Memory": ("_bw", "bandwidth", "_read_", "_write_"),
        "MemoryUB": ("ub_read", "ub_write", "ub_bw", "ub_ratio"),
        "ResourceConflictRatio": ("conflict", "_cflt", "bank"),
    }[group]
    return any(token in key for token in tokens)


def summarize_tables(tables, group, kernel_substring="_quant_score_trial"):
    """Select one detailed table and keep raw field units; never mix summaries."""
    candidates = []
    for table in tables:
        rows = [
            row
            for row in csv.DictReader(io.StringIO(table["content"]))
            if any(kernel_substring in str(value) for value in row.values())
        ]
        columns = sorted(
            {
                key
                for row in rows
                for key, value in row.items()
                if key and metric_key(key, group) and number(value) is not None
            }
        )
        if rows and columns:
            path = table["path"].lower()
            priority = (
                0
                if "kernel_details" in path
                else (
                    1
                    if "operator_details" in path
                    else (2 if "op_summary" in path else 3)
                )
            )
            candidates.append((priority, table["path"], table, rows, columns))
    if not candidates:
        return dict(status="no_numeric_score_metrics", metric_columns=[])
    _, path, table, rows, columns = min(candidates, key=lambda item: item[:2])
    stats = {}
    for key in columns:
        values = [number(row.get(key)) for row in rows]
        values = [value for value in values if value is not None]
        stats[key] = dict(
            median=statistics.median(values),
            min=min(values),
            max=max(values),
            count=len(values),
        )
    return dict(
        status="available",
        primary_table=path,
        score_row_count=len(rows),
        metric_columns=columns,
        raw_field_statistics=stats,
        rows_preview=rows[:4],
        table_truncated=table.get("truncated", False),
        aggregation="Median of exported score rows, original field units; not whole-device utilization",
    )


def check_chunk(inputs, actual, first, count):
    import torch

    visible = inputs["visible"][first : first + count]
    live = torch.arange(actual.shape[1])[None, :] < visible[:, None]
    if not bool((torch.isfinite(actual) | ~live).all()):
        raise AssertionError("Nonfinite visible profiled score")
    if not bool((torch.isneginf(actual) | live).all()):
        raise AssertionError("Bad invisible profiled score mask")
    maximum = 0.0
    selected = sorted({first, first + count // 2, first + count - 1})
    for row in selected:
        _, expected = reference(inputs, row)
        values = actual[row - first, : expected.numel()]
        if not torch.allclose(values, expected, rtol=1e-3, atol=1e-3):
            raise AssertionError(f"Profiled score mismatch at row {row}")
        if expected.numel():
            maximum = max(maximum, (values - expected).abs().max().item())
    return dict(
        rows=count,
        all_chunk_score_masks_checked=True,
        cpu_reference_rows=selected,
        max_abs_error=maximum,
    )


def run_case(args):
    result, code = (
        dict(
            case=args.case,
            status="failed",
            profile_mode=(
                "page_reuse"
                if args.page_reuse
                else "integrated" if args.integrated else "frozen"
            ),
        ),
        1,
    )
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import torch_npu
        import triton

        root = os.getenv("QSA_ENV_ROOT")
        if not root or not torch.npu.is_available():
            raise ModuleNotFoundError("Pinned 910B environment required")
        if (
            not Path(triton.__file__)
            .resolve()
            .is_relative_to(Path(root).resolve() / "flagtree")
        ):
            raise RuntimeError(f"Unexpected compiler {triton.__file__}")
        torch.npu.set_device(args.device)
        if "910B" not in torch.npu.get_device_name(args.device):
            raise RuntimeError("910B required")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        source = Path(__file__).with_name("qsa_int8_score_910b_trial.py")
        if hashlib.sha256(source.read_bytes()).hexdigest() != FROZEN_SCORE_SHA256:
            raise RuntimeError("Scorer differs from the profiled benchmark baseline")
        from qsa_int8_score_910b_trial import score

        result["environment"] = dict(
            python=sys.executable,
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            triton=triton.__version__,
            triton_source=triton.__file__,
            device_index=args.device,
            device=torch.npu.get_device_name(args.device),
            ascend_home=os.getenv("ASCEND_HOME_PATH"),
            opp=os.getenv("ASCEND_OPP_PATH"),
            device_isolation="not_verified",
        )
        try:
            properties = triton.runtime.driver.active.utils.get_device_properties(
                args.device
            )
            result["device_properties"] = {
                str(k): (
                    v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)
                )
                for k, v in properties.items()
            }
        except Exception:
            result["device_properties_error"] = traceback.format_exc()
        result["stage"] = "inputs"
        plan = PLANS[args.case]
        result["geometry"] = plan
        inputs = make_inputs(plan)
        values = {key: value.to(f"npu:{args.device}") for key, value in inputs.items()}
        rows, width, chunk = (
            plan["q_shape"][0],
            plan["logits_width"],
            plan["chunk_rows"],
        )
        chunks = [(first, min(chunk, rows - first)) for first in range(0, rows, chunk)]
        middle_first = (rows // 2 // chunk) * chunk
        middle_count = min(chunk, rows - middle_first)
        scratch = torch.empty(
            (chunk, width), device=values["q"].device, dtype=torch.float32
        )
        adapter = None
        grid, block_n = (20, 128) if args.integrated else (8, 32)
        if args.integrated:
            from qsa_int8_selection_910b_trial import PreparedInt8Selection

            output = torch.empty(
                (rows, 512), device=values["q"].device, dtype=torch.int32
            )
            workspace = torch.empty(
                (1024 * 1024,), device=values["q"].device, dtype=torch.uint8
            )
            # This capture uses only score_into; no TopK module or kernel is loaded.
            if args.page_reuse:
                from qsa_int8_shared_map_selection_910b_trial import (
                    PreparedSharedMapInt8Selection,
                )

                adapter = PreparedSharedMapInt8Selection(
                    values,
                    output,
                    scratch,
                    workspace,
                    None,
                    plan["starts"],
                    key_group4=True,
                    single_page64=True,
                )
                route = next(
                    entry
                    for entry in adapter.route_checks
                    if (entry["first"], entry["count"]) == (middle_first, middle_count)
                )
                if not route["page_reuse"]:
                    raise AssertionError(
                        "Representative chunk did not select page reuse"
                    )
                result["profile_entry"] = "PreparedSharedMapInt8Selection.score_into"
                result["representative_route"] = route
                result["preparation_ms"] = adapter.preparation_ms
                block_n = 64
            else:
                adapter = PreparedInt8Selection(
                    values, output, scratch, workspace, None
                )
                result["profile_entry"] = "PreparedInt8Selection.score_into"
            result["unused_selection_buffer_bytes"] = (
                output.numel() * 4 + workspace.numel()
            )
        else:
            result["profile_entry"] = "qsa_int8_score_910b_trial.score"
        if args.page_reuse:
            work_tiles = triton.cdiv(middle_count, 16) * triton.cdiv(
                triton.cdiv(width, block_n), 8
            )
        else:
            work_tiles = middle_count * triton.cdiv(width, block_n)
        grid = min(grid, work_tiles)
        result["representative_chunk"] = dict(
            first=middle_first,
            count=middle_count,
            visible_first=int(inputs["visible"][middle_first]),
            visible_last=int(inputs["visible"][middle_first + middle_count - 1]),
            grid_programs=grid,
            block_n=block_n,
            dot_shape=[64, 128, 64] if args.page_reuse else [block_n, 128, 16],
            work_tile_unit=(
                "group of eight N64 tiles" if args.page_reuse else "single score tile"
            ),
            real_heads=4,
            padded_heads=16,
            work_tiles=work_tiles,
            work_tiles_per_program=work_tiles / grid,
            work_tiles_per_program_min=work_tiles // grid,
            work_tiles_per_program_max=triton.cdiv(work_tiles, grid),
            score_scratch_bytes=chunk * width * 4,
            raw_dot_bytes=0,
        )
        print(
            "PROFILE_DISPATCH",
            json.dumps(
                dict(entry=result["profile_entry"], **result["representative_chunk"])
            ),
            flush=True,
        )

        def score_into(first, count):
            if adapter is not None:
                return adapter.score_into(first, count, scratch[:count])
            return score(
                values["q"][first : first + count],
                values["cache"],
                values["qs"][first : first + count],
                values["ks"],
                values["weights"][first : first + count],
                values["row_table"][first : first + count],
                values["visible"][first : first + count],
                scratch[:count],
                raw_dot=None,
            )

        def score_all():
            for first, count in chunks:
                score_into(first, count)

        def score_middle():
            return score_into(middle_first, middle_count)

        result["stage"] = "warmup_and_cpu_reference"
        expected_names = None
        if args.page_reuse:
            kernels = Counter(
                score_into(first, count).metadata.name for first, count in chunks
            )
            expected_names = {
                name: count * args.trace_iterations for name, count in kernels.items()
            }
        else:
            score_all()
        torch.npu.synchronize()
        first, count = chunks[-1]
        result["checks"] = dict(
            last_warmup_chunk=check_chunk(inputs, scratch[:count].cpu(), first, count)
        )
        samples = sample_rows(plan)
        maximum = 0.0
        for first, count in chunks:
            selected = [row for row in samples if first <= row < first + count]
            if not selected:
                continue
            score_into(first, count)
            actual = scratch[:count].cpu()
            for row in selected:
                _, expected = reference(inputs, row)
                n = expected.numel()
                if not torch.allclose(
                    actual[row - first, :n], expected, rtol=1e-3, atol=1e-3
                ):
                    raise AssertionError(f"Warmup reference mismatch row {row}")
                if n:
                    maximum = max(
                        maximum, (actual[row - first, :n] - expected).abs().max().item()
                    )
        result["checks"]["workload_cpu_samples"] = dict(
            rows=samples, max_abs_error=maximum
        )
        compiled = score_middle()
        torch.npu.synchronize()
        result["checks"]["representative_pre"] = check_chunk(
            inputs, scratch[:middle_count].cpu(), middle_first, middle_count
        )
        kernel_name = "_quant_score_trial"
        expected_stores = 1
        if args.page_reuse:
            bounded = route["skip_invisible"]
            kernel_name = (
                "_qsa_page64_page_reuse_score"
                if bounded
                else "_qsa_page64_page_reuse_full_score"
            )
            expected_stores = 2 if bounded else 1
            if compiled.metadata.name != kernel_name:
                raise AssertionError(
                    f"Expected {kernel_name}, got {compiled.metadata.name}"
                )
        result["compiler"] = dict(
            kernel_name=kernel_name,
            asm_keys=list(compiled.asm),
            metadata=repr(getattr(compiled, "metadata", None)),
            artifacts=[],
        )
        for key in ("ttir", "ttadapter", "bcmlir", "source"):
            text = compiled.asm.get(key)
            if not isinstance(text, str):
                continue
            path = report / f"{args.case}-score-{key}.txt"
            path.write_text(text[: 4 * 1024**2])
            result["compiler"]["artifacts"].append(
                dict(
                    file=path.name,
                    truncated=len(text) > 4 * 1024**2,
                    static_token_occurrences={
                        token: len(re.findall(re.escape(token), text, re.IGNORECASE))
                        for token in (
                            "barrier",
                            "wait_flag",
                            "set_flag",
                            "convert_layout",
                            "transpose",
                            "mmad",
                            "tt.dot",
                        )
                    },
                )
            )
        ir = compiled.asm["ttir"]
        if len(re.findall(r"\btt\.store\b", ir)) != expected_stores or not any(
            "tt.dot" in line and "xi8>" in line and "xi32>" in line
            for line in ir.splitlines()
        ):
            raise AssertionError(f"Unexpected INT8 score IR for {kernel_name}")
        if args.wall_samples:
            from benchmark_qsa_prefill_stages_910b import aggregate, sample

            if args.bound_launch_compare:
                result["stage"] = "bound_launch_precheck"
                started = time.perf_counter()
                launches = [
                    adapter.bind_query16_score(first, count, scratch[:count])
                    for first, count in chunks
                ]
                bind_ms = (time.perf_counter() - started) * 1000
                names = Counter()
                for (first, count), launch in zip(chunks, launches):
                    dynamic = score_into(first, count)
                    expected = scratch[:count].clone()
                    bound = launch()
                    if dynamic.metadata.name != bound.metadata.name or bytes(
                        dynamic.asm["npubin"]
                    ) != bytes(bound.asm["npubin"]):
                        raise AssertionError(
                            f"Bound launch changed the binary at row {first}"
                        )
                    if not torch.equal(expected, scratch[:count]):
                        raise AssertionError(
                            f"Bound launch changed scores at row {first}"
                        )
                    names[bound.metadata.name] += 1
                result["bound_launch_identity"] = dict(
                    launch_count=len(launches),
                    kernel_counts=dict(names),
                    binding_ms=bind_ms,
                    all_chunks_exact=True,
                )

            def score_all_bound():
                for launch in launches:
                    launch()

            def capture_unprofiled(reverse=False):
                functions = {"dispatch": score_all}
                if args.bound_launch_compare:
                    functions["bound"] = score_all_bound
                samples = {name: [] for name in functions}
                order = []
                for index in range(args.wall_samples):
                    names = list(functions)
                    if (index + reverse) % 2:
                        names.reverse()
                    order.append(names)
                    for name in names:
                        samples[name].append(
                            sample(functions[name], torch, 1, split_submission=True)
                        )

                def summarize(values):
                    return dict(
                        samples=values,
                        timing=aggregate(
                            values,
                            (
                                "wall_us",
                                "event_us",
                                "submit_us",
                                "sync_wait_us",
                                "event_record_us",
                            ),
                        ),
                    )

                capture = summarize(samples["dispatch"])
                capture["scope"] = (
                    "Host submit includes Python and runtime calls, including any runtime blocking; wait starts after recording the end event"
                )
                if args.bound_launch_compare:
                    capture["bound"] = summarize(samples["bound"])
                    capture["measurement_order"] = order
                return capture

            result["stage"] = "unprofiled_before"
            result["unprofiled"] = dict(before=capture_unprofiled())
        result["stage"] = "timeline"
        result["timeline_attempts"] = []
        for attempt in range(1, 4):
            profile_args = SimpleNamespace(
                report=str(report),
                worker=f"{args.case}-attempt{attempt}",
                trace_iterations=args.trace_iterations,
            )
            result.pop("timeline", None)
            try:
                collect(
                    profile_args,
                    {"score": score_all},
                    torch,
                    result,
                    selected=("score",),
                )
                timeline = result["timeline"]["stages"]["score"]
                validate_score_trace(
                    timeline,
                    report,
                    len(chunks) * args.trace_iterations,
                    expected_names,
                )
                result["timeline_attempts"].append(
                    dict(attempt=attempt, capture=timeline)
                )
                if timeline["usable"]:
                    break
                print("PROFILE_TRACE_REJECTED", attempt, flush=True)
            except Exception:
                result["timeline_attempts"].append(
                    dict(
                        attempt=attempt,
                        error=traceback.format_exc(),
                        capture=result.get("timeline", {}),
                    )
                )
                # Only a preserved, unrecognized device trace is retryable.
                entry = result.get("timeline", {}).get("stages", {}).get("score", {})
                if not entry.get("parse_error") or not entry.get("trace_file"):
                    break
        first, count = chunks[-1]
        if (
            result.get("timeline", {})
            .get("stages", {})
            .get("score", {})
            .get("trace_file")
        ):
            result["checks"]["last_timeline_chunk"] = check_chunk(
                inputs, scratch[:count].cpu(), first, count
            )
        if args.wall_samples:
            result["stage"] = "unprofiled_after"
            result["unprofiled"]["after"] = capture_unprofiled(reverse=True)
            timeline = result.get("timeline", {}).get("stages", {}).get("score", {})
            if timeline.get("usable"):
                result["unprofiled"]["trace_comparison"] = dict(
                    profiled_device_task_sum_us=timeline["device_task_duration_sum_us"],
                    profiled_device_envelope_us=timeline["device_task_envelope_us"],
                    profiled_uncovered_us=timeline["uncovered_envelope_us"],
                    unprofiled_event_minus_profiled_envelope_us={
                        phase: result["unprofiled"][phase]["timing"]["event_us"][
                            "median"
                        ]
                        - timeline["device_task_envelope_us"]
                        for phase in ("before", "after")
                    },
                    scope="Independent runs; event spans include submission gaps and stream boundaries",
                )
            print("UNPROFILED_SCORE", json.dumps(result["unprofiled"]), flush=True)
        result["hardware"] = {}
        for group in args.metrics:
            result["stage"] = f"hardware_{group}"
            if not hasattr(torch_npu.profiler.AiCMetrics, group):
                result["hardware"][group] = dict(status="unsupported_enum")
                continue
            attempts = []
            for attempt in range(1, 4):
                score_middle()
                torch.npu.synchronize()
                options = SimpleNamespace(
                    report=str(report),
                    worker=f"{args.case}-{group}-attempt{attempt}",
                    aic_metric=group,
                    counter_iterations=args.counter_iterations,
                    kernel_substring=kernel_name,
                )
                capture = {}
                collect_hardware(options, score_middle, torch, capture)
                entry = capture["score_hardware"]
                entry["capture_status"] = entry["status"]
                artifact = report / entry.get("artifact", "no-artifact")
                if artifact.is_file():
                    with gzip.open(artifact, "rt") as stream:
                        tables = json.load(stream)["tables"]
                    entry["parsed"] = summarize_tables(
                        tables, group, kernel_substring=kernel_name
                    )
                    entry["status"] = entry["parsed"]["status"]
                    if (
                        entry["status"] == "available"
                        and entry["parsed"]["score_row_count"]
                        != args.counter_iterations
                    ):
                        entry["status"] = "incomplete_counter_rows"
                else:
                    entry["status"] = "no_counter_archive"
                entry["postcheck"] = check_chunk(
                    inputs, scratch[:middle_count].cpu(), middle_first, middle_count
                )
                attempts.append(entry)
                if entry["status"] == "available":
                    break
                # No archive can mean unsupported collection or a runtime error.
                # Retry only an exported archive lacking usable numeric metrics.
                if not artifact.is_file():
                    break
                print(
                    "PROFILE_COUNTER_REJECTED",
                    group,
                    attempt,
                    entry["status"],
                    flush=True,
                )
            result.setdefault("hardware_attempts", {})[group] = attempts
            result["hardware"][group] = entry
            print(
                "INT8_HARDWARE",
                json.dumps(
                    dict(
                        case=args.case,
                        metric=group,
                        status=entry["status"],
                        primary_table=entry.get("parsed", {}).get("primary_table"),
                        medians={
                            key: value["median"]
                            for key, value in entry.get("parsed", {})
                            .get("raw_field_statistics", {})
                            .items()
                        },
                    )
                ),
                flush=True,
            )
        timeline = result.get("timeline", {}).get("stages", {}).get("score", {})
        pipe = result["hardware"].get("PipeUtilization", {})
        if timeline.get("usable") and pipe.get("status") == "available":
            result["status"], code = "captured", 0
        else:
            result["status"] = "partial_capture"
        result["stage"] = "completed"
    except ImportError:
        result.update(status="prerequisites_missing", traceback=traceback.format_exc())
        code = 2
    except Exception:
        result["traceback"] = traceback.format_exc()
    result["scope"] = (
        "Score task timeline plus separate aggregate hardware counters; neither shows "
        "instruction-level overlap"
        if args.page_reuse
        else "Frozen score profiling only; instrumentation changes timing; not benchmark speedup data"
    )
    for name in (
        Path(__file__).name,
        "qsa_timeline_910b.py",
        "benchmark_qsa_prefill_stages_910b.py",
        "qsa_int8_score_910b_trial.py",
        "qsa_int8_ten_workloads_910b.py",
        "qsa_int8_selection_910b_trial.py",
        "qsa_int8_shared_map_selection_910b_trial.py",
        "qsa_int8_query_pack4_910b.py",
    ):
        result.setdefault("sources", {})[name] = hashlib.sha256(
            (Path(__file__).parent / name).read_bytes()
        ).hexdigest()
    result["exit_code"] = code
    (report / f"{args.case}.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                case=args.case,
                status=result["status"],
                stage=result.get("stage"),
                representative_chunk=result.get("representative_chunk"),
                hardware={
                    k: v["status"] for k, v in result.get("hardware", {}).items()
                },
                traceback=result.get("traceback"),
            ),
            indent=2,
        ),
        flush=True,
    )
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--trace-iterations", type=int, default=1)
    parser.add_argument("--counter-iterations", type=int, default=3)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--cases", nargs="+", choices=CASES)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--integrated",
        action="store_true",
        help="Profile integrated grid20/N128 score entry",
    )
    parser.add_argument(
        "--page-reuse",
        action="store_true",
        help="Profile the default prepared N64/query16 page-reuse dispatch",
    )
    parser.add_argument(
        "--wall-samples",
        type=int,
        default=0,
        help="Measure NPU event span, host submission and final sync without a profiler before and after the trace",
    )
    parser.add_argument(
        "--bound-launch-compare",
        action="store_true",
        help="Interleave dynamic and pre-bound query16 launches in each unprofiled block",
    )
    args = parser.parse_args()
    if args.bound_launch_compare and (not args.page_reuse or not args.wall_samples):
        parser.error("--bound-launch-compare requires --page-reuse and --wall-samples")
    if args.wall_samples < 0 or (args.wall_samples and args.trace_iterations != 1):
        parser.error(
            "--wall-samples must be nonnegative and requires one trace iteration when enabled"
        )
    if args.page_reuse and not args.integrated:
        parser.error("--page-reuse requires --integrated")
    args.metrics = list(dict.fromkeys(args.metrics))
    if (
        min(args.trace_iterations, args.counter_iterations) < 1
        or "PipeUtilization" not in args.metrics
    ):
        parser.error("Positive iterations and PipeUtilization are required")
    if args.case:
        return run_case(args)
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    snapshot(report, "before")
    entries = []
    for name in dict.fromkeys(args.cases or CASES):
        print(f"Starting INT8 score profiling: {name}", flush=True)
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
            "--trace-iterations",
            str(args.trace_iterations),
            "--counter-iterations",
            str(args.counter_iterations),
            "--wall-samples",
            str(args.wall_samples),
            "--metrics",
            *args.metrics,
        ]
        if args.integrated:
            command.append("--integrated")
        if args.page_reuse:
            command.append("--page-reuse")
        if args.bound_launch_compare:
            command.append("--bound-launch-compare")
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
        elif code != 0 and entry["status"] == "captured":
            entry["status"] = "crashed_after_capture"
        entry["subprocess_exit_code"] = code
        entries.append(entry)
    snapshot(report, "after")
    status = (
        0
        if all(e["status"] == "captured" for e in entries)
        else (2 if all(e["status"] == "prerequisites_missing" for e in entries) else 1)
    )
    (report / "summary.json").write_text(
        json.dumps(
            dict(
                exit_code=status,
                cases=entries,
                scope="Profiling only; raw artifacts preserved; no latency speedup claim",
            ),
            indent=2,
        )
    )
    captured = sum(entry["status"] == "captured" for entry in entries)
    print(
        f"INT8 score profiling: {captured}/{len(entries)} captured; exit={status}",
        flush=True,
    )
    return status


if __name__ == "__main__":
    sys.exit(main())
