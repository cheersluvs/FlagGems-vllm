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

"""Probe native QSA group selection on 910B without model/compiler dependencies."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

CASES = {
    **{
        f"visible{n}": dict(lengths=[1], visible=[n], page=192)
        for n in (0, 1, 511, 512, 513, 1537)
    },
    "ragged": dict(lengths=[3, 0, 2, 1], visible=[0, 1, 511, 512, 513, 1537], page=192),
    "page16_control": dict(lengths=[1], visible=[513], page=16),
}
K, HEADS, DIM = 512, 4, 128
RTOL, ATOL = 1e-3, 2e-3


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def worker(args):
    result = dict(status="failed", case=args.worker, python=sys.executable)
    try:
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        import torch
        import torch_npu

        result["runtime"] = {
            name: dict(
                version=str(getattr(module, "__version__", "unknown")),
                source=module.__file__,
            )
            for name, module in (("torch", torch), ("torch_npu", torch_npu))
        }
        if not torch.npu.is_available():
            raise RuntimeError("torch.npu.is_available() is false")
        torch.npu.set_device(args.device)
        result["device"] = dict(
            index=args.device, name=torch.npu.get_device_name(args.device)
        )
        if "910B" not in result["device"]["name"].upper():
            raise RuntimeError(
                "Selected device is not identified as 910B; inspect device name"
            )
        if not hasattr(torch_npu, "npu_lightning_indexer"):
            raise RuntimeError("Installed torch_npu has no npu_lightning_indexer API")
        try:
            result["native_schema"] = str(
                torch.ops.npu.npu_lightning_indexer.default._schema
            )
        except (AttributeError, RuntimeError):
            result["native_schema"] = "not exposed through torch.ops.npu"
        if args.worker == "environment":
            result["status"] = "passed"
            return 0
        config = CASES[args.worker]
        lengths, visible, page = config["lengths"], config["visible"], config["page"]
        rows, requests = sum(lengths), len(lengths)
        capacity = max(K + 1, max(visible))
        columns = math.ceil(capacity / page)
        torch.manual_seed(20261029)
        q_cpu = torch.randn(rows, HEADS, DIM, dtype=torch.bfloat16)
        key_cpu = torch.randn(requests * columns, page, 1, DIM, dtype=torch.bfloat16)
        table_cpu = (
            torch.randperm(requests * columns)
            .to(torch.int32)
            .reshape(requests, columns)
        )
        owner_cpu = torch.repeat_interleave(
            torch.arange(requests), torch.tensor(lengths)
        )
        device = f"npu:{args.device}"
        query, key = q_cpu.to(device), key_cpu.to(device)
        # One TND sequence per row, matching the audited vLLM-Ascend adapter.
        row_table = table_cpu[owner_cpu].contiguous().to(device)
        cu_query = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
        key_lengths = torch.tensor(visible, dtype=torch.int32, device=device)
        weights = torch.full(
            (rows, HEADS), DIM**-0.5, dtype=torch.float32, device=device
        )
        result["inputs"] = dict(
            query=list(query.shape),
            key=list(key.shape),
            page_table=list(row_table.shape),
            request_lengths=lengths,
            visible=visible,
            page_size=page,
            dtype=str(query.dtype),
            sparse_count=K,
            weights=1 / math.sqrt(DIM),
            rtol=RTOL,
            atol=ATOL,
        )
        native = torch_npu.npu_lightning_indexer(
            query,
            key,
            weights,
            actual_seq_lengths_query=cu_query,
            actual_seq_lengths_key=key_lengths,
            block_table=row_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=K,
            sparse_mode=0,
            pre_tokens=2**63 - 1,
            next_tokens=2**63 - 1,
            return_value=False,
        )
        torch.npu.synchronize()
        indices = native[0].cpu()
        result["raw_output"] = dict(shape=list(indices.shape), dtype=str(indices.dtype))
        if indices.shape != (rows, 1, K) or indices.dtype != torch.int32:
            raise AssertionError("Unexpected native group output shape/dtype")
        indices = indices.squeeze(1)
        result["rows"] = []
        errors = []
        for row, count in enumerate(visible):
            raw = indices[row].long()
            # Only invalid/sentinel indices are normalized. Never repair missing
            # valid groups, deduplicate outputs, or run a replacement TopK.
            selected = raw[(raw >= 0) & (raw < count)]
            expected_count = min(K, count)
            info = dict(
                row=row,
                visible=count,
                valid_count=selected.numel(),
                unique_valid=selected.unique().numel(),
                expected_count=expected_count,
                raw_minus_one=int((raw == -1).sum()),
                raw_out_of_range=int(((raw < -1) | (raw >= count)).sum()),
                raw_prefix=raw[:16].tolist(),
            )
            result["rows"].append(info)
            if (
                selected.numel() != expected_count
                or selected.unique().numel() != expected_count
            ):
                errors.append(f"row {row}: valid group count/uniqueness mismatch")
                continue
            if not count:
                continue
            positions = torch.arange(count)
            physical = table_cpu[owner_cpu[row], positions // page].long()
            keys = key_cpu[physical, positions % page, 0].float()
            scores = torch.einsum("hd,kd->hk", q_cpu[row].float(), keys).clamp_min(
                0
            ).sum(0) / math.sqrt(DIM)
            actual = scores[selected].sort().values
            expected = scores.topk(expected_count).values.sort().values
            info["selected_score_max_abs_error"] = float(
                (actual - expected).abs().max()
            )
            if not torch.allclose(actual, expected, rtol=RTOL, atol=ATOL):
                errors.append(f"row {row}: selected scores differ from FP32 reference")
        result["normalization"] = (
            "Mask raw negative/out-of-visibility indices to -1; no deduplication or replacement TopK"
        )
        result["errors"] = errors
        result["status"] = "passed" if not errors else "correctness_failed"
        print(args.worker, result["status"], json.dumps(result["rows"]), flush=True)
        return 0 if not errors else 1
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
        return 2 if args.worker == "environment" else 1
    finally:
        write(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="logical index after ASCEND_RT_VISIBLE_DEVICES",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--worker", choices=("environment", *CASES))
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.device < 0 or args.timeout <= 0:
        parser.error("Invalid device or timeout")
    if args.worker:
        if not args.output:
            parser.error("Worker mode requires --output")
        return worker(args)
    report = Path(args.report).resolve()
    report.mkdir(parents=True, exist_ok=True)
    summary = dict(
        status="running",
        packages={},
        cases=[],
        scope="native BF16 baseline probe; no performance claim",
    )
    for name in (
        "torch",
        "torch-npu",
        "triton",
        "triton-ascend",
        "flagtree",
        "vllm",
        "vllm-ascend",
    ):
        try:
            summary["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            summary["packages"][name] = None
    try:
        inventory = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=30
        )
        (report / "npu-smi.txt").write_text(inventory.stdout + inventory.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        (report / "npu-smi.txt").write_text(str(exc))
    summary["cann_version_files"] = {}
    for root in (
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
        Path("/usr/local/Ascend/driver"),
    ):
        for pattern in ("version.cfg", "version.info", "*/ascend_toolkit_install.info"):
            for file in root.glob(pattern):
                if file.is_file():
                    summary["cann_version_files"][str(file)] = file.read_text(
                        errors="replace"
                    )[:16000]
    summary["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in ("environment", *CASES):
        output = report / f"{name}.json"
        with (report / f"{name}.txt").open("w") as log:
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--report",
                        str(report),
                        "--device",
                        str(args.device),
                        "--worker",
                        name,
                        "--output",
                        str(output),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = 124
        try:
            value = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError):
            value = dict(status="failed", error="worker exited without complete JSON")
        value["exit_code"] = code
        summary["cases"].append(value)
        print(f"{name}: {value['status']}, exit={code}", flush=True)
        print((report / f"{name}.txt").read_text()[-8000:], flush=True)
        write(report / "summary.json", summary)
        if name == "environment" and code:
            summary["status"] = "environment_unavailable"
            write(report / "summary.json", summary)
            return 2
    summary["status"] = (
        "passed" if all(c["exit_code"] == 0 for c in summary["cases"]) else "failed"
    )
    write(report / "summary.json", summary)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
