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

"""Check INT8 scoring plus unchanged persistent TopK against native V1."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from probe_qsa_int8_score_910b import CASES, prepare, reference, validate


def check_selection(output, scores):
    import torch

    if output.shape != (len(scores), 512) or output.dtype != torch.int32:
        raise AssertionError(f"Unexpected index output: {output.shape}, {output.dtype}")
    failures, selected = [], []
    max_abs = max_rel = 0.0
    for row, score in enumerate(scores):
        indices = output[row].long()
        selected.append(None)
        if bool(((indices < -1) | (indices >= score.numel())).any()):
            failures.append(f"row {row}: out-of-range indices")
            continue
        valid = indices[indices >= 0]
        count = min(512, score.numel())
        if valid.numel() != count or valid.unique().numel() != count:
            failures.append(f"row {row}: missing or duplicate valid indices")
            continue
        values = score[valid].sort(descending=True).values
        expected = score.topk(count).values
        selected[row] = values
        if count:
            error = (values - expected).abs()
            max_abs = max(max_abs, error.max().item())
            max_rel = max(
                max_rel, (error / expected.abs().clamp_min(1e-3)).max().item()
            )
        if not torch.allclose(values, expected, rtol=1e-3, atol=1e-3):
            failures.append(f"row {row}: selected scores differ from CPU TopK")
    return (
        dict(
            checked_all_rows=True,
            rows=len(scores),
            checks_passed=not failures,
            failures=failures,
            selected_score_max_abs_error=max_abs,
            selected_score_max_rel_error_with_1e_minus3_floor=max_rel,
            rtol=1e-3,
            atol=1e-3,
            ordering_and_tied_index_identity_required=False,
        ),
        selected,
    )


def compare_selected(left, right):
    import torch

    failures, max_abs = [], 0.0
    for row, (a, b) in enumerate(zip(left, right)):
        if a is None or b is None or a.shape != b.shape:
            failures.append(f"row {row}: invalid selection prevents paired comparison")
            continue
        if a.numel():
            max_abs = max(max_abs, (a - b).abs().max().item())
        if not torch.allclose(a, b, rtol=1e-3, atol=1e-3):
            failures.append(f"row {row}: native/candidate selected scores differ")
    return dict(
        checks_passed=not failures,
        failures=failures,
        selected_score_max_abs_error=max_abs,
        rtol=1e-3,
        atol=1e-3,
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
        native = getattr(torch_npu, "npu_quant_lightning_indexer", None)
        if not root or not torch.npu.is_available() or native is None:
            raise ModuleNotFoundError(
                "Pinned environment, NPU and native quant V1 API required"
            )
        if (
            not Path(triton.__file__)
            .resolve()
            .is_relative_to(Path(root).resolve() / "flagtree")
        ):
            raise RuntimeError(f"Unexpected Triton source: {triton.__file__}")
        torch.npu.set_device(args.device)
        device = torch.npu.get_device_name(args.device)
        if "910B" not in device:
            raise RuntimeError(f"910B required, got {device}")
        result["environment"] = dict(
            python=sys.executable,
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            triton=triton.__version__,
            triton_source=triton.__file__,
            device=device,
            ascend_home=os.getenv("ASCEND_HOME_PATH"),
            opp=os.getenv("ASCEND_OPP_PATH"),
            topk_route_requested=os.getenv("FLAGGEMS_ASCEND_PTK_PATH", "auto"),
        )
        from qsa_int8_score_910b_trial import score
        from qsa_prefill_pipeline_910b_trial import load_topk

        inputs, geometry = prepare(args.case)
        result["geometry"] = geometry
        values = {key: value.to(f"npu:{args.device}") for key, value in inputs.items()}
        rows, heads, _ = values["q"].shape
        width = values["row_table"].shape[1] * geometry["page_size"]
        logits = torch.empty(
            (rows, width), device=values["q"].device, dtype=torch.float32
        )
        dots = torch.empty(
            (rows, width, heads), device=values["q"].device, dtype=torch.int32
        )
        result["stage"] = "candidate_score"
        score(
            values["q"],
            values["cache"],
            values["qs"],
            values["ks"],
            values["weights"],
            values["row_table"],
            values["visible"],
            logits,
            dots,
            mask_debug=True,
        )
        torch.npu.synchronize()
        result["score_checks"] = validate(inputs, logits.cpu(), dots.cpu())
        if not result["score_checks"]["checks_passed"]:
            raise AssertionError("Candidate full-score prerequisite failed")

        result["stage"] = "candidate_persistent_topk"
        topk = load_topk()
        result["topk_identity"] = topk.probe_identity
        output = torch.empty((rows, 512), device=logits.device, dtype=torch.int32)
        workspace = torch.empty((1024 * 1024,), device=logits.device, dtype=torch.uint8)
        topk.persistent_topk(
            logits,
            values["visible"],
            output,
            workspace,
            k=512,
            max_seq_len=width,
            block=256,
        )
        torch.npu.synchronize()
        candidate_cpu = output.cpu()
        result["topk_state"] = dict(
            segmerge=dict(topk._SEGMERGE_STATE),
            persistent=dict(topk._PSEGMERGE_STATE),
            block=256,
        )
        scores = [reference(inputs, row)[1] for row in range(rows)]
        result["candidate_checks"], candidate_selected = check_selection(
            candidate_cpu, scores
        )
        result["raw_indices"] = dict(candidate=candidate_cpu.tolist())

        result["stage"] = "native_quant_v1"
        native_output = native(
            values["q"],
            values["cache"],
            values["weights"],
            values["qs"],
            values["ks"],
            0,
            0,
            actual_seq_lengths_query=values["q_ends"],
            actual_seq_lengths_key=values["visible"],
            block_table=values["row_table"],
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=512,
            sparse_mode=0,
            pre_tokens=2**63 - 1,
            next_tokens=2**63 - 1,
        )
        torch.npu.synchronize()
        if native_output.shape != (rows, 1, 512):
            raise AssertionError(
                f"Unexpected native output shape: {native_output.shape}"
            )
        native_cpu = native_output.cpu().squeeze(1)
        result["raw_indices"]["native"] = native_cpu.tolist()
        result["native_checks"], native_selected = check_selection(native_cpu, scores)
        result["paired_checks"] = compare_selected(candidate_selected, native_selected)
        result["paired_checks"]["identical_index_set_rows"] = sum(
            torch.equal(candidate_cpu[row].sort().values, native_cpu[row].sort().values)
            for row in range(rows)
        )
        result["stage"] = "completed"
        if all(
            result[key]["checks_passed"]
            for key in (
                "score_checks",
                "candidate_checks",
                "native_checks",
                "paired_checks",
            )
        ):
            result["status"], code = "passed", 0
    except ImportError:
        result.update(status="prerequisites_missing", traceback=traceback.format_exc())
        code = 2
    except Exception:
        result["traceback"] = traceback.format_exc()
    for name in (
        Path(__file__).name,
        "qsa_int8_score_910b_trial.py",
        "probe_qsa_int8_score_910b.py",
        "probe_qsa_quant_v1_910b.py",
        "qsa_prefill_pipeline_910b_trial.py",
    ):
        result.setdefault("sources", {})[name] = hashlib.sha256(
            (Path(__file__).parent / name).read_bytes()
        ).hexdigest()
    (report / f"{args.case}.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k != "raw_indices"}, indent=2),
        flush=True,
    )
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    if args.case:
        return run_case(args)
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in CASES:
        print(f"Starting INT8 pipeline probe: {name}", flush=True)
        path = report / f"{name}.json"
        path.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--case",
            name,
            "--report",
            str(report),
            "--device",
            str(args.device),
        ]
        try:
            with (report / f"{name}-console.txt").open("w") as log:
                completed = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout
                )
            entry = (
                json.loads(path.read_text())
                if path.is_file()
                else dict(case=name, status="crashed")
            )
            entry["exit_code"] = completed.returncode
            if completed.returncode != 0 and entry["status"] == "passed":
                entry["status"] = "crashed_after_checks"
        except subprocess.TimeoutExpired:
            entry = dict(case=name, status="timeout", exit_code=1)
        entry.pop("raw_indices", None)
        entries.append(entry)
        print(json.dumps(entry, indent=2), flush=True)
    status = (
        0
        if all(e["status"] == "passed" for e in entries)
        else (2 if all(e["status"] == "prerequisites_missing" for e in entries) else 1)
    )
    summary = dict(
        exit_code=status,
        cases=entries,
        scope="INT8 score plus existing persistent TopK vs native V1; correctness only",
        next_step="If passed, migrate ten NV logical workloads to INT8/page192 before performance measurement",
    )
    (report / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"INT8 pipeline: {sum(e['status'] == 'passed' for e in entries)}/3 passed; exit={status}",
        flush=True,
    )
    return status


if __name__ == "__main__":
    sys.exit(main())
