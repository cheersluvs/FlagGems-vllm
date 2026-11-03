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

"""Execute ten migrated INT8 QSA workloads with bounded correctness staging."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_qsa_int8_pipeline_910b import check_selection  # noqa: E402
from probe_qsa_int8_score_910b import reference  # noqa: E402

from tools.qsa_int8_ten_workloads_910b import (  # noqa: E402
    cases,
    make_inputs,
    sample_rows,
)

PLANS = {case["name"]: case for case in cases()}


def verify_structure(indices, lengths):
    import torch

    if indices.dtype != torch.int32 or indices.shape != (lengths.numel(), 512):
        raise AssertionError("Unexpected index shape/dtype")
    if bool(((indices < -1) | (indices >= lengths[:, None])).any()):
        raise AssertionError("Out-of-range index")
    expected = lengths.clamp(max=512)
    if not torch.equal((indices >= 0).sum(1), expected.long()):
        raise AssertionError("Wrong valid count or -1 padding")
    ordered = indices.sort(dim=1).values
    if bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()):
        raise AssertionError("Duplicate valid indices")
    return ordered


def selected_scores(logits, indices, cpu_indices):
    values = logits.gather(1, indices.clamp_min(0).long()).cpu()
    values[cpu_indices < 0] = -float("inf")
    return values.sort(dim=1, descending=True).values


def run_case(args):
    result, code = dict(case=args.case, status="failed"), 1
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
        )
        result["stage"] = "inputs"
        inputs = make_inputs(plan)
        values = {key: value.to(f"npu:{args.device}") for key, value in inputs.items()}
        rows, width, chunk = (
            plan["q_shape"][0],
            plan["logits_width"],
            plan["chunk_rows"],
        )
        result["execution_policy"] = dict(
            score_chunk_rows=chunk,
            native_call_rows=chunk,
            topk_call_rows=args.topk_call_rows,
        )
        samples = sample_rows(plan)
        result["cpu_reference_rows"] = samples
        topk = load_topk()
        result["topk_identity"] = topk.probe_identity
        logits = torch.empty(
            (chunk, width), device=values["q"].device, dtype=torch.float32
        )
        dots = torch.empty((chunk, width, 4), device=logits.device, dtype=torch.int32)
        output = torch.empty((chunk, 512), device=logits.device, dtype=torch.int32)
        workspace = torch.empty((1024 * 1024,), device=logits.device, dtype=torch.uint8)
        q_ends = torch.arange(1, chunk + 1, device=logits.device, dtype=torch.int32)
        columns = torch.arange(width, device=logits.device)
        digests = dict(candidate=hashlib.sha256(), native=hashlib.sha256())
        checks = dict(
            processed_rows=0,
            identical_index_set_rows=0,
            paired_selected_score_max_abs_error=0.0,
            sampled_score_max_abs_error=0.0,
            cpu_samples=[],
            full_score_masks_checked=False,
            all_rows_index_structure_checked=False,
            score_mask_checked_rows=0,
            index_structure_checked_rows=0,
            sampled_dot_max_abs_error=0,
        )
        result["checks"] = checks
        result["sampled_raw_indices"] = {}
        for first in range(0, rows, chunk):
            count = min(chunk, rows - first)
            last = first + count
            result["chunk"] = [first, last]
            q, qs, weights, table, visible = (
                values[key][first:last]
                for key in ("q", "qs", "weights", "row_table", "visible")
            )
            current = logits[:count]
            result["stage"] = "candidate_score"
            score(
                q,
                values["cache"],
                qs,
                values["ks"],
                weights,
                table,
                visible,
                current,
                dots[:count],
                mask_debug=True,
            )
            result["stage"] = "candidate_topk"
            for offset in range(0, count, args.topk_call_rows):
                end = min(count, offset + args.topk_call_rows)
                topk.persistent_topk(
                    current[offset:end],
                    visible[offset:end],
                    output[offset:end],
                    workspace,
                    k=512,
                    max_seq_len=width,
                    block=256,
                )
            result["stage"] = "native_v1"
            native_output = native(
                q,
                values["cache"],
                weights,
                qs,
                values["ks"],
                0,
                0,
                actual_seq_lengths_query=q_ends[:count],
                actual_seq_lengths_key=visible,
                block_table=table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=512,
                sparse_mode=0,
                pre_tokens=2**63 - 1,
                next_tokens=2**63 - 1,
            )
            torch.npu.synchronize()
            if native_output.shape != (count, 1, 512):
                raise AssertionError(f"Unexpected native output: {native_output.shape}")
            result["stage"] = "all_row_validation"
            candidate_cpu = output[:count].cpu()
            native_cpu = native_output.cpu().squeeze(1)
            lengths = inputs["visible"][first:last]
            a_order = verify_structure(candidate_cpu, lengths)
            b_order = verify_structure(native_cpu, lengths)
            checks["index_structure_checked_rows"] = last
            live = columns[None, :] < visible[:, None]
            if not bool((torch.isfinite(current) | ~live).all().item()):
                raise AssertionError("Nonfinite visible score")
            if not bool(((current == -float("inf")) | live).all().item()):
                raise AssertionError("Invisible score not -inf")
            checks["score_mask_checked_rows"] = last
            a_values = selected_scores(current, output[:count], candidate_cpu)
            b_values = selected_scores(current, native_output.squeeze(1), native_cpu)
            valid = torch.arange(512)[None, :] < lengths.clamp(max=512)[:, None]
            if bool(valid.any()):
                error = (a_values[valid] - b_values[valid]).abs().max().item()
                checks["paired_selected_score_max_abs_error"] = max(
                    checks["paired_selected_score_max_abs_error"], error
                )
            if not torch.allclose(a_values, b_values, rtol=1e-3, atol=1e-3):
                raise AssertionError(
                    "Paired selected-score mismatch; max_abs_error="
                    f"{checks['paired_selected_score_max_abs_error']}"
                )
            checks["identical_index_set_rows"] += int(
                (a_order == b_order).all(1).sum().item()
            )
            for name, indices in (("candidate", candidate_cpu), ("native", native_cpu)):
                digests[name].update(indices.numpy().tobytes())
            result["stage"] = "sampled_cpu_reference"
            for row in (row for row in samples if first <= row < last):
                local = row - first
                expected_dot, expected_score = reference(inputs, row)
                length = expected_score.numel()
                actual = current[local].cpu()
                raw = dots[local].cpu()
                if not torch.equal(raw[:length], expected_dot):
                    checks["sampled_dot_max_abs_error"] = int(
                        (raw[:length].long() - expected_dot.long()).abs().max().item()
                    )
                    raise AssertionError(f"Sample row {row}: integer dot mismatch")
                if bool(raw[length:].any()):
                    raise AssertionError(f"Sample row {row}: nonzero debug padding")
                if not torch.allclose(
                    actual[:length], expected_score, rtol=1e-3, atol=1e-3
                ):
                    raise AssertionError(f"Sample row {row}: full score mismatch")
                if length:
                    error = (actual[:length] - expected_score).abs().max().item()
                    checks["sampled_score_max_abs_error"] = max(
                        checks["sampled_score_max_abs_error"], error
                    )
                row_checks = {}
                for name, indices in (
                    ("candidate", candidate_cpu),
                    ("native", native_cpu),
                ):
                    selected, _ = check_selection(
                        indices[local : local + 1], [expected_score]
                    )
                    row_checks[name] = selected
                    if not selected["checks_passed"]:
                        result["failed_cpu_sample"] = dict(row=row, checks=row_checks)
                        raise AssertionError(
                            f"Sample row {row}: {name} selected scores differ from CPU"
                        )
                checks["cpu_samples"].append(dict(row=row, checks=row_checks))
                result["sampled_raw_indices"][str(row)] = dict(
                    candidate=candidate_cpu[local].tolist(),
                    native=native_cpu[local].tolist(),
                )
            checks["processed_rows"] = last
            if first == 0 or last == rows or (first // chunk + 1) % 32 == 0:
                print(
                    json.dumps(
                        dict(
                            case=args.case,
                            processed=last,
                            total=rows,
                            paired_score_error=checks[
                                "paired_selected_score_max_abs_error"
                            ],
                            cpu_samples_done=len(checks["cpu_samples"]),
                        )
                    ),
                    flush=True,
                )
        if len(checks["cpu_samples"]) != len(samples):
            raise AssertionError("Some CPU sample rows were not checked")
        checks["full_score_masks_checked"] = True
        checks["all_rows_index_structure_checked"] = True
        result["output_stream_sha256"] = {
            name: digest.hexdigest() for name, digest in digests.items()
        }
        result["topk_state"] = dict(
            segmerge=dict(topk._SEGMERGE_STATE),
            persistent=dict(topk._PSEGMERGE_STATE),
            block=256,
        )
        result["stage"], result["status"], code = "completed", "passed", 0
    except ImportError:
        result.update(status="prerequisites_missing", traceback=traceback.format_exc())
        code = 2
    except Exception:
        result["traceback"] = traceback.format_exc()
    for name in (
        Path(__file__).name,
        "qsa_int8_ten_workloads_910b.py",
        "qsa_prefill_fp8_workloads.py",
        "probe_qsa_int8_pipeline_910b.py",
        "probe_qsa_int8_score_910b.py",
        "qsa_int8_score_910b_trial.py",
    ):
        result.setdefault("sources", {})[name] = hashlib.sha256(
            (Path(__file__).parent / name).read_bytes()
        ).hexdigest()
    (report / f"{args.case}.json").write_text(json.dumps(result, indent=2))
    display = {
        key: result[key]
        for key in ("case", "status", "stage", "chunk", "traceback", "topk_state")
        if key in result
    }
    display["checks"] = {
        key: value
        for key, value in result.get("checks", {}).items()
        if key != "cpu_samples"
    }
    display["cpu_samples_done"] = len(result.get("checks", {}).get("cpu_samples", []))
    print(json.dumps(display, indent=2), flush=True)
    return code


def run_child(command, log_path, timeout):
    """Stream the child's log while enforcing a timeout for its process group."""
    deadline = time.monotonic() + timeout
    with log_path.open("wb") as log, log_path.open("rb") as reader:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        timed_out = False
        try:
            while process.poll() is None:
                data = reader.read()
                if data:
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                if time.monotonic() >= deadline:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    timed_out = True
                    break
                time.sleep(0.2)
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
        code = process.wait()
        sys.stdout.buffer.write(reader.read())
        sys.stdout.buffer.flush()
    return code, timed_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--topk-call-rows", type=int, choices=(32, 128), default=32)
    parser.add_argument("--cases", nargs="+", choices=PLANS)
    parser.add_argument("--case", choices=PLANS)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.case:
        return run_case(args)
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(args.cases or PLANS))
    (report / "workload-plan.json").write_text(
        json.dumps(
            [dict(PLANS[name], topk_call_rows=args.topk_call_rows) for name in names],
            indent=2,
        )
    )
    print(
        "case | Q rows | requests | max context | score width | score chunk | TopK call | CPU samples",
        flush=True,
    )
    for name in names:
        p = PLANS[name]
        print(
            f"{name} | {p['q_shape'][0]} | {len(p['query_lengths'])} | {p['batch_max_seq_len']} | "
            f"{p['logits_width']} | {p['chunk_rows']} | {args.topk_call_rows} | {len(sample_rows(p))}",
            flush=True,
        )
    if args.plan_only:
        (report / "summary.json").write_text(
            json.dumps(
                dict(
                    exit_code=0,
                    validation_status="not_run_plan_only",
                    cases=[],
                    scope="Metadata migration only; no device execution or numerical checks",
                ),
                indent=2,
            )
        )
        return 0
    entries = []
    for name in names:
        print(f"Starting full-shape correctness: {name}", flush=True)
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
            "--topk-call-rows",
            str(args.topk_call_rows),
        ]
        code, timed_out = run_child(
            command, report / f"{name}-console.txt", args.timeout
        )
        entry = (
            json.loads(path.read_text())
            if path.is_file()
            else dict(case=name, status="crashed")
        )
        entry.pop("sampled_raw_indices", None)
        if timed_out:
            entry["status"] = "timeout"
        elif code != 0 and entry["status"] == "passed":
            entry["status"] = "crashed_after_checks"
        entry["exit_code"] = code
        entries.append(entry)
    status = (
        0
        if all(e["status"] == "passed" for e in entries)
        else (2 if all(e["status"] == "prerequisites_missing" for e in entries) else 1)
    )
    summary = dict(
        exit_code=status,
        cases=entries,
        scope=(
            "Full logical inputs, bounded chunk execution; all-row pairing and mask checks, "
            "sampled independent CPU reference; no timings"
        ),
    )
    (report / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"INT8 ten shapes: {sum(e['status'] == 'passed' for e in entries)}/{len(names)} passed; exit={status}",
        flush=True,
    )
    return status


if __name__ == "__main__":
    sys.exit(main())
