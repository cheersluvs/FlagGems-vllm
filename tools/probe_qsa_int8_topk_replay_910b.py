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

"""Replay budget_exact's first chunk with existing TopK batch/route controls."""

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_qsa_int8_pipeline_910b import check_selection  # noqa: E402
from probe_qsa_int8_score_910b import reference, validate  # noqa: E402
from probe_qsa_int8_ten_shapes_910b import PLANS, run_child  # noqa: E402

from tools.qsa_int8_ten_workloads_910b import make_inputs  # noqa: E402


def duplicate_details(indices, scores):
    result = []
    for row in range(indices.shape[0]):
        valid = indices[row][(indices[row] >= 0) & (indices[row] < scores[row].numel())]
        unique, counts = valid.unique(return_counts=True)
        repeated = counts > 1
        if bool(repeated.any()):
            result.append(
                dict(
                    row=row,
                    visible=scores[row].numel(),
                    duplicate_scores=scores[row][unique[repeated][:8].long()].tolist(),
                    duplicate_indices=unique[repeated][:8].tolist(),
                    multiplicities=counts[repeated][:8].tolist(),
                    duplicate_index_count=int(repeated.sum().item()),
                )
            )
    return result


def run(args):
    result, code = (
        dict(status="failed", workload="int8_budget_exact", replay_rows=[0, 128]),
        1,
    )
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    old_route = os.environ.get("FLAGGEMS_ASCEND_PTK_PATH")
    try:
        import torch
        import torch_npu
        import triton

        root = os.getenv("QSA_ENV_ROOT")
        native = getattr(torch_npu, "npu_quant_lightning_indexer", None)
        if not root or not torch.npu.is_available() or native is None:
            raise ModuleNotFoundError("Pinned 910B environment/native V1 required")
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
        result["environment"] = dict(
            python=sys.executable,
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            triton=triton.__version__,
            triton_source=triton.__file__,
            device=torch.npu.get_device_name(args.device),
            ascend_home=os.getenv("ASCEND_HOME_PATH"),
            opp=os.getenv("ASCEND_OPP_PATH"),
            original_topk_route=old_route,
        )
        import qsa_prefill_pipeline_910b_trial as loader
        from qsa_int8_score_910b_trial import score

        result["stage"] = "full_input_generation"
        plan = PLANS["int8_budget_exact"]
        cpu = make_inputs(plan)
        result["full_input_sha256"] = {
            key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
            for key, value in cpu.items()
        }
        # Generate the full original workload first; slicing earlier changes RNG values.
        small = {
            key: value if key in ("cache", "ks") else value[:128]
            for key, value in cpu.items()
        }
        values = {key: value.to(f"npu:{args.device}") for key, value in small.items()}
        rows, width = 128, plan["logits_width"]
        logits = torch.empty(
            (rows, width), device=values["q"].device, dtype=torch.float32
        )
        dots = torch.empty((rows, width, 4), device=logits.device, dtype=torch.int32)
        result["stage"] = "score_and_independent_cpu_reference"
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
        frozen_cpu = logits.cpu()
        result["frozen_logits_sha256"] = hashlib.sha256(
            frozen_cpu.numpy().tobytes()
        ).hexdigest()
        result["score_checks"] = validate(small, frozen_cpu, dots.cpu())
        if not result["score_checks"]["checks_passed"]:
            raise AssertionError("Full first-chunk score verification failed")
        references = [reference(small, row)[1] for row in range(rows)]
        result["variants"] = []
        for name, batch, route, initialize in (
            ("auto128_original", 128, "auto", False),
            ("auto128_sentinel", 128, "auto", True),
            ("auto32", 32, "auto", True),
            ("radix128", 128, "radix", True),
        ):
            entry = dict(
                name=name,
                call_rows=batch,
                route=route,
                output_initialization="sentinel_-2" if initialize else "empty",
                status="failed",
            )
            result["variants"].append(entry)
            result["stage"] = name
            try:
                os.environ["FLAGGEMS_ASCEND_PTK_PATH"] = route
                loader._TOPK = None
                topk = loader.load_topk()
                entry["topk_identity"] = topk.probe_identity
                output = torch.empty(
                    (rows, 512), device=logits.device, dtype=torch.int32
                )
                if initialize:
                    output.fill_(-2)
                workspace = torch.empty(
                    (1024 * 1024,), device=logits.device, dtype=torch.uint8
                )
                for first in range(0, rows, batch):
                    topk.persistent_topk(
                        logits[first : first + batch],
                        values["visible"][first : first + batch],
                        output[first : first + batch],
                        workspace,
                        k=512,
                        max_seq_len=width,
                        block=256,
                    )
                torch.npu.synchronize()
                indices = output.cpu()
                entry["raw_indices"] = indices.tolist()
                entry["duplicate_rows"] = duplicate_details(indices, references)
                if initialize:
                    entry["unwritten_sentinel_count"] = int(
                        (indices == -2).sum().item()
                    )
                entry["checks"], _ = check_selection(indices, references)
                entry["topk_state"] = dict(
                    segmerge=dict(topk._SEGMERGE_STATE),
                    persistent=dict(topk._PSEGMERGE_STATE),
                )
                entry["status"] = (
                    "passed" if entry["checks"]["checks_passed"] else "rejected"
                )
            except Exception:
                entry["traceback"] = traceback.format_exc()
            display = {
                key: value
                for key, value in entry.items()
                if key not in ("raw_indices", "duplicate_rows")
            }
            display["duplicate_row_count"] = len(entry.get("duplicate_rows", []))
            display["duplicate_examples"] = entry.get("duplicate_rows", [])[:4]
            print(json.dumps(display, indent=2), flush=True)
        result["stage"] = "native_v1"
        native_output = native(
            values["q"],
            values["cache"],
            values["weights"],
            values["qs"],
            values["ks"],
            0,
            0,
            actual_seq_lengths_query=torch.arange(
                1, rows + 1, device=logits.device, dtype=torch.int32
            ),
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
            raise AssertionError(f"Unexpected native shape: {native_output.shape}")
        native_cpu = native_output.cpu().squeeze(1)
        result["native_checks"], _ = check_selection(native_cpu, references)
        result["native_raw_indices"] = native_cpu.tolist()
        result["stage"] = "completed"
        if result["native_checks"]["checks_passed"] and all(
            v["status"] == "passed" for v in result["variants"]
        ):
            result["status"], code = "passed", 0
        else:
            result["status"] = "completed_with_failures"
    except ImportError:
        result.update(status="prerequisites_missing", traceback=traceback.format_exc())
        code = 2
    except Exception:
        result["traceback"] = traceback.format_exc()
    finally:
        if old_route is None:
            os.environ.pop("FLAGGEMS_ASCEND_PTK_PATH", None)
        else:
            os.environ["FLAGGEMS_ASCEND_PTK_PATH"] = old_route
    for name in (
        Path(__file__).name,
        "qsa_int8_ten_workloads_910b.py",
        "qsa_int8_score_910b_trial.py",
        "qsa_prefill_pipeline_910b_trial.py",
        "probe_qsa_int8_pipeline_910b.py",
    ):
        result.setdefault("sources", {})[name] = hashlib.sha256(
            (Path(__file__).parent / name).read_bytes()
        ).hexdigest()
    result["exit_code"] = code
    (report / "replay.json").write_text(json.dumps(result, indent=2))
    result.pop("native_raw_indices", None)
    for variant in result.get("variants", []):
        variant.pop("raw_indices", None)
    (report / "summary.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                status=result["status"],
                native_checks=result.get("native_checks"),
                variants=[
                    dict(
                        name=v["name"],
                        status=v["status"],
                        duplicate_rows=len(v.get("duplicate_rows", [])),
                    )
                    for v in result.get("variants", [])
                ],
                exit_code=code,
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
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return run(args)
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--report",
        str(report),
        "--device",
        str(args.device),
    ]
    code, timed_out = run_child(command, report / "replay-console.txt", args.timeout)
    if timed_out or not (report / "summary.json").is_file():
        (report / "summary.json").write_text(
            json.dumps(
                dict(
                    status="timeout" if timed_out else "crashed",
                    exit_code=1,
                    subprocess_exit_code=code,
                ),
                indent=2,
            )
        )
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
