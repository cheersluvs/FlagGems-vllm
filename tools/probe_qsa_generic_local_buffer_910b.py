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

"""Check generic TLE L1 operands before executing or timing a second-Cube probe."""

import json
import math
import re
import statistics
import traceback
from pathlib import Path
from unittest.mock import patch

DUMP_FLAG = "--bishengir-print-ir-after=hivm-graph-sync-solver"


def check_ir(text):
    active = False
    selected = []
    sections = 0
    for number, line in enumerate(text.splitlines(), 1):
        if "IR Dump" in line:
            active = "IR Dump After" in line and "graphsyncsolver" in re.sub(
                r"[^a-z]", "", line.lower()
            )
            sections += int(active)
        if active:
            selected.append((number, line))
    matmuls = [n for n, line in selected if "hivm.hir.mmadL1 " in line]
    fp16_workspace = [
        dict(line=n, text=line.strip())
        for n, line in selected
        if "memref_ext.alloc_workspace" in line
        and "xf16," in line
        and "address_space<gm>" in line
    ]
    fp16_gm_to_cube = [
        dict(line=n, text=line.strip())
        for n, line in selected
        if "hivm.hir.nd2nz " in line
        and re.search(r"ins\(.*?xf16.*?address_space<gm>", line)
    ]
    fp16_cube = [
        n
        for n, line in selected
        if "hivm.hir.mmadL1 " in line
        and "xf16" in line
        and "address_space<cbuf>" in line
    ]
    return dict(
        fp16_cube_lines=fp16_cube,
        passed=bool(fp16_cube)
        and sections >= 2
        and len(matmuls) == 2
        and not fp16_workspace
        and not fp16_gm_to_cube,
        after_solver_sections=sections,
        matmul_lines=matmuls,
        fp16_gm_workspace=fp16_workspace,
        fp16_gm_to_cube=fp16_gm_to_cube,
        scope="Gate for removed FP16 GM staging only; original INT32 QK and final FP32 GM traffic may remain",
    )


def run_probe(inputs, values, refs, width, timer, report, case):
    import torch
    from diagnose_qsa_structural_mixed_compile_910b import CaptureCompiler
    from probe_qsa_fused_second_cube_910b import prepare_current_int8_score
    from probe_qsa_int8_pipeline_910b import check_selection
    from qsa_prefill_pipeline_910b_trial import load_topk
    from triton.backends.ascend import compiler

    class DumpCompiler(CaptureCompiler):
        def run(self, command, *args, **kwargs):
            if (
                isinstance(command, (list, tuple))
                and command
                and Path(command[0]).name == "bishengir-compile"
            ):
                command = [*command, DUMP_FLAG]
            return super().run(command, *args, **kwargs)

    path = report / f"{case}-generic-local-buffer.json"
    result = dict(
        status="running",
        candidate_status="pending",
        checks=[],
        arms={},
        scope="budget_exact shared-map tail32 diagnostic, not acceptance",
        hypothesis="Generic to_buffer(L1)/to_tensor can avoid FP16 GM staging before second Cube",
        compiler_change="Print-only option; time the exact kernels compiled in this process",
        topk_identity=None,
    )
    outputs = []
    functions = {}
    topk_functions = {}
    try:
        topk = load_topk()
        result["topk_identity"] = topk.probe_identity
        for fixture, rows, scan in (
            ("full", inputs["q"].shape[0], width),
            ("tail_visible", 16, 193),
        ):
            visible_cpu = inputs["visible"][:rows].clone().clamp(max=scan)
            if fixture != "full":
                visible_cpu[0], visible_cpu[1] = 0, 1
            visible = visible_cpu.to(values["q"].device)
            output = torch.empty(
                (rows, scan), device=visible.device, dtype=torch.float32
            )
            outputs.append(output)
            candidate, identity = prepare_current_int8_score(
                values,
                visible,
                rows,
                scan,
                output,
                report,
                f"{case}_local_{fixture}",
                topk,
                reduction_cube=True,
                reduction_mode="local_l1",
            )
            capture = DumpCompiler(report, f"{case}-local-{fixture}")
            entry = dict(
                fixture=fixture,
                rows=rows,
                width=scan,
                identity=identity,
                phase="compilation",
            )
            result["checks"].append(entry)
            path.write_text(json.dumps(result, indent=2))
            try:
                with patch.object(compiler, "subprocess", capture):
                    compiled = candidate.qsa_warmup()
            except Exception:
                entry.update(compiler_calls=capture.calls, error=traceback.format_exc())
                result.update(
                    candidate_status="rejected_compile",
                    status="completed_with_candidate_rejection",
                )
                print("LOCAL_BUFFER_REJECTED_COMPILE " + entry["error"], flush=True)
                return result
            entry["compiler_calls"] = capture.calls
            entry["compiled_metadata"] = str(compiled.metadata)
            texts = []
            truncated = False
            for call in capture.calls:
                for stream in ("stdout", "stderr"):
                    if call.get(stream):
                        texts.append(
                            (report / call[stream]["file"]).read_text(errors="replace")
                        )
                        truncated |= call[stream]["truncated"]
            entry["ir"] = check_ir("\n".join(texts))
            entry["ir"]["truncated"] = truncated
            if len(capture.calls) != 1 or truncated or not entry["ir"]["passed"]:
                result.update(
                    candidate_status="rejected_ir",
                    status="completed_with_candidate_rejection",
                )
                print("LOCAL_BUFFER_REJECTED_IR " + json.dumps(entry["ir"]), flush=True)
                return result
            # The first device execution happens only after the same compiled IR passes.
            entry["phase"] = "device_execution"
            path.write_text(json.dumps(result, indent=2))
            candidate()
            actual = output.cpu()
            expected = torch.full((rows, scan), -float("inf"))
            for row, live in enumerate(visible_cpu.tolist()):
                expected[row, :live] = refs[row][:live]
            close = torch.isclose(actual, expected, rtol=1e-3, atol=1e-3)
            entry["mismatches"] = int((~close).sum())
            if not bool(close.all()):
                result.update(
                    candidate_status="rejected_precision",
                    status="completed_with_candidate_rejection",
                )
                return result
            entry["phase"] = "passed"
            print("LOCAL_BUFFER_CHECK " + json.dumps(entry), flush=True)
            if fixture == "full":
                functions["local_l1"] = candidate
                candidate_output, candidate_visible = output, visible
        current_output = torch.empty_like(candidate_output)
        outputs.append(current_output)
        current, identity = prepare_current_int8_score(
            values,
            candidate_visible,
            candidate_output.shape[0],
            width,
            current_output,
            report,
            f"{case}_current",
            topk,
        )
        capture = DumpCompiler(report, f"{case}-current")
        with patch.object(compiler, "subprocess", capture):
            current()
        expected = torch.full_like(current_output.cpu(), -float("inf"))
        for row, ref in enumerate(refs):
            expected[row, : ref.numel()] = ref
        torch.testing.assert_close(current_output.cpu(), expected, rtol=1e-3, atol=1e-3)
        functions = dict(current=current, **functions)
        combined = {}
        result["current_identity"] = identity
        result["current_compiler_calls"] = capture.calls
        for name, fn in functions.items():
            output = current_output if name == "current" else candidate_output
            indices = torch.empty(
                (output.shape[0], 512), device=output.device, dtype=torch.int32
            )
            workspace = torch.empty(1024**2, device=output.device, dtype=torch.uint8)

            def select(output=output, indices=indices, workspace=workspace):
                return topk.persistent_topk(
                    output,
                    candidate_visible,
                    indices,
                    workspace,
                    k=512,
                    max_seq_len=width,
                    block=256,
                )

            select()
            selection, _ = check_selection(indices.cpu(), refs)
            result["arms"][name] = dict(
                selection=selection,
                samples_ms=dict(score=[], topk=[], score_plus_topk=[]),
            )
            if not selection["checks_passed"]:
                if name == "current":
                    raise RuntimeError("Current INT8 selection control failed")
                result.update(
                    candidate_status="rejected_selection",
                    status="completed_with_candidate_rejection",
                )
                return result
            topk_functions[name] = select
            combined[name] = lambda fn=fn, select=select: (fn(), select())
        for iteration in range(2):
            order = tuple(functions) if iteration == 0 else tuple(reversed(functions))
            for name in order:
                for stage, fn in (
                    ("score", functions[name]),
                    ("topk", topk_functions[name]),
                    ("score_plus_topk", combined[name]),
                ):
                    value = float(timer.get_latency(fn))
                    if not math.isfinite(value) or value <= 0:
                        raise RuntimeError("Invalid harness timing")
                    result["arms"][name]["samples_ms"][stage].append(value)
        for name, entry in result["arms"].items():
            entry["median_us"] = {
                stage: statistics.median(times) * 1000
                for stage, times in entry["samples_ms"].items()
            }
            print(
                "LOCAL_BUFFER_TIME " + json.dumps(dict(arm=name, **entry)), flush=True
            )
        result["speedups_over_current"] = {
            stage: result["arms"]["current"]["median_us"][stage]
            / result["arms"]["local_l1"]["median_us"][stage]
            for stage in ("score", "topk", "score_plus_topk")
        }
        result.update(candidate_status="passed", status="completed")
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
        print(
            "LOCAL_BUFFER_RESULT "
            + json.dumps(
                dict(
                    status=result["status"], candidate_status=result["candidate_status"]
                )
            ),
            flush=True,
        )
    return result


def summarize_probe(directory, output, pytest_exit):
    error = None
    record = {}
    try:
        record = json.loads(
            (directory / "int8_budget_exact-generic-local-buffer.json").read_text()
        )
        if pytest_exit or record["status"] not in (
            "completed",
            "completed_with_candidate_rejection",
        ):
            raise ValueError(f"Incomplete probe, pytest exit={pytest_exit}")
        if record["candidate_status"] == "passed":
            if len(record["checks"]) != 2 or not all(
                e["ir"]["passed"] and e["mismatches"] == 0 for e in record["checks"]
            ):
                raise ValueError("Missing IR/precision checks")
            if set(record["arms"]) != {"current", "local_l1"}:
                raise ValueError("Missing paired timing controls")
            for entry in record["arms"].values():
                if set(entry["samples_ms"]) != {"score", "topk", "score_plus_topk"}:
                    raise ValueError("Missing stage timings")
                if not entry["selection"]["checks_passed"]:
                    raise ValueError("Selection failed")
                for times in entry["samples_ms"].values():
                    if len(times) != 2 or not all(
                        math.isfinite(t) and t > 0 for t in times
                    ):
                        raise ValueError("Invalid timings")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        error = str(exc)
    summary = dict(
        exit_code=int(error is not None),
        error=error,
        probe=record,
        candidate_passed=error is None and record.get("candidate_status") == "passed",
        arithmetic_mean_speedup=None,
        scope="single diagnostic chunk; no acceptance mean",
    )
    output.write_text(json.dumps(summary, indent=2))
    print("GENERIC_LOCAL_HANDOFF " + json.dumps(summary), flush=True)
    return summary["exit_code"]
