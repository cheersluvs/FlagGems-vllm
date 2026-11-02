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

"""Validate independent INT8 scores and raw dot products on Ascend 910B."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

from probe_qsa_quant_v1_910b import CASES, fixture


def prepare(name):
    import torch

    inputs, _, geometry = fixture(name)
    gen = torch.Generator().manual_seed(20261103)
    if name == "h4_topk512":
        for key in ("q", "cache"):
            inputs[key] = torch.randint(-128, 128, inputs[key].shape, generator=gen).to(
                torch.int8
            )
    elif name == "h4_compressed_boundary":
        for key in ("qs", "ks"):
            inputs[key] = (
                0.001 + 0.029 * torch.rand(inputs[key].shape, generator=gen)
            ).half()
        inputs["weights"] = (
            0.25 + 1.5 * torch.rand(inputs["weights"].shape, generator=gen)
        ).half()
    else:
        inputs["q"][0].fill_(-128)
        inputs["cache"][inputs["row_table"][0, 0], 0, 0].fill_(-128)
    if name != "h4_select_all":
        inputs["q"][0].fill_(1)
    # Align the forbidden K with row zero, including the negative-extreme case.
    future = geometry["visible_groups"][0]
    page = geometry["page_size"]
    physical = inputs["row_table"][0, future // page]
    inputs["cache"][physical, future % page, 0].fill_(
        -128 if name == "h4_select_all" else 127
    )
    return inputs, geometry


def reference(inputs, row):
    """Reconstruct the exact logical K sequence from the shuffled physical cache."""
    count = int(inputs["visible"][row])
    heads = inputs["q"].shape[1]
    physical = inputs["row_table"][row].long()
    keys = inputs["cache"][physical, :, 0].reshape(-1, 128)[:count].int()
    scales = inputs["ks"][physical, :, 0].reshape(-1)[:count].float()
    dot = keys @ inputs["q"][row].int().T
    assert dot.shape == (count, heads)
    dequant = dot.float() * inputs["qs"][row].float()[None, :] * scales[:, None]
    score = (dequant.relu() * inputs["weights"][row].float()[None, :]).sum(1)
    return dot, score


def validate(inputs, output, dots):
    import torch

    largest_abs = largest_rel = 0.0
    valid_elements = 0
    for row, count in enumerate(inputs["visible"].tolist()):
        expected_dot, expected = reference(inputs, row)
        if not torch.equal(dots[row, :count], expected_dot):
            error = (dots[row, :count].long() - expected_dot.long()).abs().max().item()
            raise AssertionError(
                f"row {row}: INT32 dot mismatch; max_abs_error={error}"
            )
        actual = output[row, :count]
        if count:
            error = (actual - expected).abs()
            largest_abs = max(largest_abs, error.max().item())
            largest_rel = max(
                largest_rel, (error / expected.abs().clamp_min(1e-3)).max().item()
            )
        if not torch.allclose(actual, expected, rtol=1e-3, atol=1e-3):
            raise AssertionError(
                f"row {row}: FP32 score mismatch; max_abs_error={largest_abs}"
            )
        if not bool(torch.isneginf(output[row, count:]).all()):
            raise AssertionError(f"row {row}: invisible scores must be -inf")
        if bool(dots[row, count:].any()):
            raise AssertionError(f"row {row}: invisible debug dots must be zero")
        valid_elements += count
    return dict(
        rows=output.shape[0],
        valid_score_elements=valid_elements,
        int32_dots_exact=True,
        score_max_abs_error=largest_abs,
        score_max_rel_error_with_1e_minus3_floor=largest_rel,
        score_rtol=1e-3,
        score_atol=1e-3,
        invalid_scores_negative_infinity=True,
        invalid_debug_dots_zero=True,
    )


def run_case(args):
    result, code = dict(case=args.case, status="failed"), 1
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import torch_npu
        import triton

        root = os.environ.get("QSA_ENV_ROOT")
        if not root or not torch.npu.is_available():
            raise ModuleNotFoundError("QSA_ENV_ROOT and an available NPU are required")
        compiler_path = Path(triton.__file__).resolve()
        if not compiler_path.is_relative_to(Path(root).resolve() / "flagtree"):
            raise RuntimeError(f"Unexpected Triton compiler source: {compiler_path}")
        torch.npu.set_device(args.device)
        device_name = torch.npu.get_device_name(args.device)
        if "910B" not in device_name:
            raise RuntimeError(f"Expected 910B, got {device_name}")
        result["environment"] = dict(
            python=sys.executable,
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            triton=triton.__version__,
            triton_source=str(compiler_path),
            device=device_name,
            ascend_home=os.getenv("ASCEND_HOME_PATH"),
            opp=os.getenv("ASCEND_OPP_PATH"),
        )
        from qsa_int8_score_910b_trial import score

        inputs, geometry = prepare(args.case)
        result["geometry"] = geometry
        result["input_ranges"] = {
            key: [value.min().item(), value.max().item()]
            for key, value in inputs.items()
            if key in ("q", "cache", "qs", "ks", "weights")
        }
        values = {key: value.to(f"npu:{args.device}") for key, value in inputs.items()}
        width = values["row_table"].shape[1] * geometry["page_size"]
        rows, heads, _ = values["q"].shape
        output = torch.empty(
            (rows, width), device=values["q"].device, dtype=torch.float32
        )
        dots = torch.empty(
            (rows, width, heads), device=values["q"].device, dtype=torch.int32
        )
        compiled = score(
            values["q"],
            values["cache"],
            values["qs"],
            values["ks"],
            values["weights"],
            values["row_table"],
            values["visible"],
            output,
            dots,
        )
        torch.npu.synchronize()
        result["checks"] = validate(inputs, output.cpu(), dots.cpu())
        result["compiler_artifacts"] = []
        try:
            asm = getattr(compiled, "asm", {})
            result["compiler_asm_keys"] = list(asm)
            for key in ("ttir", "ttgir", "llir", "mlir"):
                if isinstance(asm.get(key), str):
                    text = asm[key]
                    path = (
                        report / f"{args.case}-{re.sub('[^a-zA-Z0-9_-]', '_', key)}.txt"
                    )
                    path.write_text(text[: 1024 * 1024])
                    result["compiler_artifacts"].append(
                        dict(file=path.name, truncated=len(text) > 1024 * 1024)
                    )
        except Exception:
            result["compiler_export_warning"] = traceback.format_exc()
        result["status"], code = "passed", 0
    except ImportError:
        result.update(status="prerequisites_missing", traceback=traceback.format_exc())
        code = 2
    except Exception:
        result["traceback"] = traceback.format_exc()
    for filename in (
        "qsa_int8_score_910b_trial.py",
        "probe_qsa_quant_v1_910b.py",
        Path(__file__).name,
    ):
        path = Path(__file__).parent / filename
        result.setdefault("sources", {})[filename] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (report / f"{args.case}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
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
        print(f"Starting INT8 score probe: {name}", flush=True)
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
        scope="Independent debug INT8 score trial only; no TopK integration or timing",
        next_step="If passed, integrate unchanged persistent TopK and compare native V1 selected scores",
    )
    (report / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"INT8 scores: {sum(e['status'] == 'passed' for e in entries)}/3 passed; exit={status}",
        flush=True,
    )
    return status


if __name__ == "__main__":
    sys.exit(main())
