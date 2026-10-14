# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Validation-only native vLLM adapter; IPC latency must never be benchmarked."""

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


def native_environment(root):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}:{root}"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["VLLM_PLUGINS"] = ""
    env["QSA_REQUIRE_VLLM_REFERENCE"] = "1"
    # Validation only: make an asynchronous error identify the failed launch.
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    for name in (
        "PYTHONHOME",
        "QSA_TRITON_RUNTIME_PATH",
        "QSA_NATIVE_REFERENCE_PYTHON",
        "QSA_NATIVE_REPORT_DIR",
    ):
        env.pop(name, None)
    return env


def tensor_manifest(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
    }


def record_native_outputs(actual, native):
    case = getattr(native, "_qsa_native_case", None)
    if case is None:
        return None
    case = Path(case)
    output = case / "gems-output.pt"
    torch.save(actual.detach().cpu(), output)
    metadata = {
        "gems_output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "sorted_indices_equal": torch.equal(actual.sort().values, native.sort().values),
        "selected_value_check": "not completed",
    }
    (case / "alignment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return case


class NativePrefillReference:
    def __init__(self, source, source_sha256):
        if "benchmark/" in os.environ.get("PYTEST_CURRENT_TEST", ""):
            raise RuntimeError(
                "Isolated native adapter is accuracy-only; IPC must not be timed"
            )
        self.__file__ = str(Path(source).resolve())
        if (
            hashlib.sha256(Path(self.__file__).read_bytes()).hexdigest()
            != source_sha256
        ):
            raise RuntimeError("Native QSA source has the wrong SHA256")
        self.python = os.environ["QSA_NATIVE_REFERENCE_PYTHON"]
        self.root = Path(__file__).resolve().parents[1]
        self.reports = Path(os.environ["QSA_NATIVE_REPORT_DIR"])
        self.reports.mkdir(parents=True, exist_ok=True)
        self.envs = SimpleNamespace(
            VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=int(
                os.environ.get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
            )
        )

    def qsa_select_paged_prefill(self, *args):
        if len(args) != 10:
            raise TypeError("QSA prefill needs ten arguments")
        case = Path(tempfile.mkdtemp(prefix="case-", dir=self.reports))
        packed = []
        for index, value in enumerate(args):
            if isinstance(value, torch.Tensor):
                cpu = torch.empty_strided(
                    value.shape, value.stride(), dtype=value.dtype, device="cpu"
                )
                if index != 8:
                    cpu.copy_(value.detach().cpu())
                else:
                    cpu.fill_(-17)
                packed.append(cpu)
            else:
                packed.append(value)
        device_index = args[0].device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        budget = self.envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
        input_file = case / "input.pt"
        torch.save(
            {"args": packed, "device_index": device_index, "budget_mb": budget},
            input_file,
        )
        manifest = {
            "pytest_test": os.environ.get("PYTEST_CURRENT_TEST"),
            "input_sha256": hashlib.sha256(input_file.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(
                Path(self.__file__).read_bytes()
            ).hexdigest(),
            "device_index": device_index,
            "budget_mb": budget,
            "args": [
                tensor_manifest(value) if isinstance(value, torch.Tensor) else value
                for value in args
            ],
        }
        (case / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        env = native_environment(self.root)
        env["QSA_VLLM_REFERENCE_FILE"] = self.__file__
        timeout = int(os.environ.get("QSA_NATIVE_TIMEOUT_S", "300"))
        if timeout <= 0:
            raise ValueError("QSA_NATIVE_TIMEOUT_S must be positive")
        command = [
            self.python,
            str(self.root / "tools/qsa_native_reference_worker.py"),
            "--case",
            str(case),
        ]
        with (case / "native.txt").open("w") as log:
            try:
                result = subprocess.run(
                    command,
                    cwd=self.root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Native vLLM timed out after {timeout}s; reports: {case}"
                ) from exc
        if result.returncode:
            raise RuntimeError(
                f"Native vLLM failed ({result.returncode}); reports: {case}\n"
                + (case / "native.txt").read_text()[-4000:]
            )
        metadata = json.loads((case / "native.json").read_text())
        if metadata["input_sha256"] != manifest["input_sha256"]:
            raise RuntimeError("Native input identity mismatch")
        if metadata["source_sha256"] != manifest["source_sha256"]:
            raise RuntimeError("Native source identity mismatch")
        if metadata["device_index"] != manifest["device_index"]:
            raise RuntimeError("Native device identity mismatch")
        output_file = case / "output.pt"
        if (
            hashlib.sha256(output_file.read_bytes()).hexdigest()
            != metadata["output_sha256"]
        ):
            raise RuntimeError("Native output identity mismatch")
        output = torch.load(output_file, map_location="cpu", weights_only=True)
        if output.dtype != args[8].dtype or output.shape != args[8].shape:
            raise RuntimeError("Native output shape/dtype mismatch")
        args[8].copy_(output)
        args[8]._qsa_native_case = str(case)
