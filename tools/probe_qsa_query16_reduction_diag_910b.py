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

"""Diagnose second-Cube overhead with observable output sinks and paired timing."""

import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def run_diagnostics(inputs, values, refs, width, timer, report, case, topk, controls):
    import torch
    from diagnose_qsa_structural_mixed_compile_910b import CaptureCompiler
    from probe_qsa_fused_second_cube_910b import prepare_current_int8_score
    from qsa_timeline_910b import collect, collect_hardware
    from triton.backends.ascend import compiler

    report = Path(report)
    path = report / f"{case}-reduction-diagnostics.json"
    result = dict(
        status="running",
        arms={},
        profiles={},
        scope="32-row diagnostic ablations, not full-input acceptance",
        interpretation=(
            "Prefix times include an observable 16x64 output sink. "
            "Different liveness/layout/codegen prevents additive attribution by subtraction. "
            "Task traces/counters do not prove intra-kernel overlap."
        ),
        contracts=dict(
            qk_sink="INT32 QK then FP32 sum over four heads; original visibility and output footprint",
            half_vector=(
                "FP16-rounded positive QK/1024 and FP16-rounded weight*query_scale, "
                "FP32 Vector sum, then 1024*key_scale"
            ),
        ),
    )
    functions = dict(controls)
    outputs = []
    rows = inputs["q"].shape[0]
    mapping = inputs["row_table"][0, : (width + 191) // 192].long()
    keys = inputs["cache"][mapping, :, 0].reshape(-1, 128)[:width].int()
    raw = (inputs["q"].reshape(-1, 128).int() @ keys.T).reshape(rows, 4, width)
    expected_raw = raw.float().sum(1)
    expected_score = torch.full((rows, width), -float("inf"))
    for row, visible in enumerate(inputs["visible"].tolist()):
        expected_raw[row, visible:] = -float("inf")
        expected_score[row, :visible] = refs[row][:visible]
    try:
        for mode in ("qk_sink", "half_vector"):
            output = torch.empty(
                (rows, width), device=values["q"].device, dtype=torch.float32
            )
            outputs.append(output)
            fn, identity = prepare_current_int8_score(
                values,
                values["visible"],
                rows,
                width,
                output,
                report,
                f"{case}_diag_{mode}",
                topk,
                reduction_cube=True,
                reduction_mode=mode,
            )
            entry = dict(identity=identity, samples_ms=[], status="compiling")
            result["arms"][mode] = entry
            path.write_text(json.dumps(result, indent=2))
            capture = CaptureCompiler(report, f"{case}-diag-{mode}")
            with patch.object(compiler, "subprocess", capture):
                compiled = fn()
            torch.npu.synchronize()
            entry["compiler_calls"] = capture.calls
            entry["ir_counts"] = {}
            for call in capture.calls:
                if call.get("input_ir"):
                    data = (report / call["input_ir"]).read_text()
                    entry["ir_counts"][call["input_ir"]] = dict(
                        tt_dot=data.count("tt.dot"),
                        linalg_matmul=data.count("linalg.matmul"),
                    )
            for kind, data in getattr(compiled, "asm", {}).items():
                if isinstance(data, str):
                    (report / f"{case}-diag-{mode}-{kind}.txt").write_text(data)
                    entry["ir_counts"][kind] = dict(
                        tt_dot=data.count("tt.dot"),
                        linalg_matmul=data.count("linalg.matmul"),
                    )
            actual = output.cpu()
            expected = expected_raw if mode == "qk_sink" else expected_score
            close = (
                actual == expected
                if mode == "qk_sink"
                else torch.isclose(actual, expected, rtol=1e-3, atol=1e-3)
            )
            finite = torch.isfinite(expected)
            entry["checks"] = dict(
                mismatches=int((~close).sum()),
                max_abs_error=float((actual[finite] - expected[finite]).abs().max()),
            )
            if not bool(close.all()):
                raise RuntimeError(
                    f"Diagnostic correctness failed: {mode}: {entry['checks']}"
                )
            entry["status"] = "passed"
            functions[mode] = fn
        for name in controls:
            result["arms"][name] = dict(
                status="passed",
                samples_ms=[],
                correctness="full score and TopK checks in parent report",
            )
        for iteration in range(2):
            order = tuple(functions) if iteration == 0 else tuple(reversed(functions))
            for name in order:
                value = float(timer.get_latency(functions[name]))
                if not math.isfinite(value) or value <= 0:
                    raise RuntimeError(f"Invalid diagnostic latency: {name} {value}")
                result["arms"][name]["samples_ms"].append(value)
        for name, entry in result["arms"].items():
            entry["median_us"] = statistics.median(entry["samples_ms"]) * 1000
            print(
                "Q16_REDUCTION_DIAG_TIME " + json.dumps(dict(arm=name, **entry)),
                flush=True,
            )
        # Profile each arm in its own profiler session to avoid name aggregation.
        for name, fn in functions.items():
            entry = {}
            result["profiles"][name] = entry
            args = SimpleNamespace(
                report=str(report),
                worker=f"{case}-reduction-{name}",
                trace_iterations=3,
                counter_iterations=3,
                aic_metric="PipeUtilization",
                kernel_substring="",
            )
            try:
                collect(args, {name: fn}, torch, entry, selected=(name,))
                collect_hardware(args, fn, torch, entry)
                entry["status"] = (
                    "collected"
                    if entry["score_hardware"]["status"] == "exported"
                    else "partial"
                )
            except Exception as exc:
                entry.update(status="failed", error=repr(exc))
            path.write_text(json.dumps(result, indent=2))
        result["status"] = (
            "completed"
            if all(e["status"] == "collected" for e in result["profiles"].values())
            else "partial_profile"
        )
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
