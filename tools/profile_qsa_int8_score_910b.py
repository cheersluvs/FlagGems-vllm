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
import traceback
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


def validate_score_trace(timeline, report, expected):
    """Validate this score-only trace, including generic CANN task names."""
    timeline["expected_score_launches"] = expected
    with gzip.open(Path(report) / timeline["trace_file"], "rt") as stream:
        payload = json.load(stream)
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    host_count = sum(
        event.get("ph") == "X"
        and event.get("cat") == "cpu_op"
        and event.get("name") == "_quant_score_trial"
        for event in events
    )
    names = timeline.get("kernels", {})
    named_count = sum(
        entry["count"] for name, entry in names.items() if "_quant_score_trial" in name
    )
    generic_count = names.get("MIX_AIC", {}).get("count", 0)
    timeline["matched_score_task_names"] = [
        name for name in names if "_quant_score_trial" in name
    ]
    timeline["host_score_launch_count"] = host_count
    timeline["identity_basis"] = "unconfirmed"
    if timeline.get("task_count") == expected and named_count == expected:
        timeline["identity_basis"] = "named_device_tasks"
    elif (
        timeline.get("task_count") == expected
        and host_count == expected
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


def summarize_tables(tables, group):
    """Select one detailed table and keep raw field units; never mix summaries."""
    candidates = []
    for table in tables:
        rows = [
            row
            for row in csv.DictReader(io.StringIO(table["content"]))
            if any("_quant_score_trial" in str(value) for value in row.values())
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
    result, code = dict(case=args.case, status="failed"), 1
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
        result["representative_chunk"] = dict(
            first=middle_first,
            count=middle_count,
            visible_first=int(inputs["visible"][middle_first]),
            visible_last=int(inputs["visible"][middle_first + middle_count - 1]),
            grid_programs=8,
            block_n=32,
            dot_shape=[32, 128, 16],
            real_heads=4,
            padded_heads=16,
            work_tiles=middle_count * ((width + 31) // 32),
            work_tiles_per_program=middle_count * ((width + 31) // 32) // 8,
            score_scratch_bytes=chunk * width * 4,
            raw_dot_bytes=0,
        )

        def score_into(first, count):
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
        result["compiler"] = dict(
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
        if len(re.findall(r"\btt\.store\b", ir)) != 1 or not any(
            "tt.dot" in line and "xi8>" in line and "xi32>" in line
            for line in ir.splitlines()
        ):
            raise AssertionError("Expected frozen no-debug INT8 score IR")
        profile_args = SimpleNamespace(
            report=str(report), worker=args.case, trace_iterations=args.trace_iterations
        )
        result["stage"] = "timeline"
        try:
            collect(
                profile_args, {"score": score_all}, torch, result, selected=("score",)
            )
            timeline = result["timeline"]["stages"]["score"]
            validate_score_trace(timeline, report, len(chunks) * args.trace_iterations)
        except Exception:
            result["timeline_error"] = traceback.format_exc()
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
        result["hardware"] = {}
        for group in args.metrics:
            result["stage"] = f"hardware_{group}"
            if not hasattr(torch_npu.profiler.AiCMetrics, group):
                result["hardware"][group] = dict(status="unsupported_enum")
                continue
            score_middle()
            torch.npu.synchronize()
            options = SimpleNamespace(
                report=str(report),
                worker=f"{args.case}-{group}",
                aic_metric=group,
                counter_iterations=args.counter_iterations,
                kernel_substring="_quant_score_trial",
            )
            capture = {}
            collect_hardware(options, score_middle, torch, capture)
            entry = capture["score_hardware"]
            entry["capture_status"] = entry["status"]
            artifact = report / entry.get("artifact", "no-artifact")
            if artifact.is_file():
                with gzip.open(artifact, "rt") as stream:
                    tables = json.load(stream)["tables"]
                entry["parsed"] = summarize_tables(tables, group)
                entry["status"] = entry["parsed"]["status"]
            else:
                entry["status"] = "no_counter_archive"
            entry["postcheck"] = check_chunk(
                inputs, scratch[:middle_count].cpu(), middle_first, middle_count
            )
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
        "Frozen score profiling only; instrumentation changes timing; not benchmark speedup data"
    )
    for name in (
        Path(__file__).name,
        "qsa_timeline_910b.py",
        "qsa_int8_score_910b_trial.py",
        "qsa_int8_ten_workloads_910b.py",
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
    args = parser.parse_args()
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
        print(f"Starting frozen INT8 score profiling: {name}", flush=True)
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
            "--metrics",
            *args.metrics,
        ]
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
