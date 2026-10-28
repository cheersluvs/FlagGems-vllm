# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Execute the original QSA reference in a clean stock-Triton CUDA process."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def run(case=None):
    import torch
    import triton

    from tests.qsa_select_prefill_utils import REFERENCE_SHA256, vllm_reference

    assert not os.environ.get("QSA_NATIVE_REFERENCE_PYTHON"), "recursive native adapter"
    assert not os.environ.get(
        "QSA_TRITON_RUNTIME_PATH"
    ), "TLE override in native worker"
    metadata = {"python": sys.executable, "prefix": sys.prefix, "sys_path": sys.path}
    for name, module in (("torch", torch), ("triton", triton)):
        source = Path(module.__file__).resolve()
        installed = importlib.metadata.version(name)
        assert source.is_relative_to(Path(sys.prefix).resolve()), (name, source)
        assert module.__version__ == installed, (name, module.__version__, installed)
        metadata[name] = {"version": module.__version__, "source": str(source)}
    payload = None
    if case is not None:
        payload = torch.load(case / "input.pt", map_location="cpu", weights_only=True)
        torch.cuda.set_device(payload["device_index"])
        os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = str(payload["budget_mb"])
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    reference = vllm_reference()
    assert "flaggems_vllm" not in sys.modules, "FlagGems imported in native worker"
    assert importlib.metadata.version("vllm").startswith(
        "0.31.0"
    ), "vLLM 0.31.0 required"
    metadata.update(
        device=torch.cuda.get_device_name(),
        device_index=torch.cuda.current_device(),
        vllm=importlib.metadata.version("vllm"),
        source=str(Path(reference.__file__).resolve()),
        source_sha256=hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
        cuda_launch_blocking=os.environ.get("CUDA_LAUNCH_BLOCKING"),
        ld_library_path=os.environ.get("LD_LIBRARY_PATH"),
        baseline="original pinned vLLM QSA; stock Triton; original native CUDA Top-K dispatch",
        native_extension=importlib.import_module("vllm._C_stable_libtorch").__file__,
    )
    assert (
        Path(metadata["native_extension"])
        .resolve()
        .is_relative_to(Path(sys.prefix).resolve())
    )
    assert metadata["source_sha256"] == REFERENCE_SHA256
    print(json.dumps(metadata, indent=2), flush=True)
    if case is None:
        logits = torch.arange(8192, device="cuda", dtype=torch.float32).repeat(32, 1)
        lengths = torch.full((32,), 8192, device="cuda", dtype=torch.int32)
        out = torch.empty((32, 512), device="cuda", dtype=torch.int32)
        scratch = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)
        torch.ops._C.cooperative_topk(logits, lengths, out, scratch, 512, 8192)
        torch.cuda.synchronize()
        expected = torch.arange(
            8192 - 512, 8192, device="cuda", dtype=torch.int32
        ).expand(32, -1)
        torch.testing.assert_close(out.sort(dim=1).values, expected, rtol=0, atol=0)
        print("PASS: stock-runtime native cooperative Top-K preflight", flush=True)
        return
    args = []
    for index, value in enumerate(payload["args"]):
        if isinstance(value, torch.Tensor):
            device = torch.empty_strided(
                value.shape, value.stride(), dtype=value.dtype, device="cuda"
            )
            if index != 8:
                device.copy_(value)
            else:
                device.fill_(-17)
            assert device.stride() == value.stride()
            args.append(device)
        else:
            args.append(value)
    reference.envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB = payload["budget_mb"]
    torch.cuda.synchronize()
    print("inputs synchronized; original native QSA begins", flush=True)
    reference.qsa_select_paged_prefill(*args)
    torch.cuda.synchronize()
    print("original native QSA synchronized", flush=True)
    output_file = case / "output.pt"
    torch.save(args[8].cpu(), output_file)
    metadata.update(
        input_sha256=hashlib.sha256((case / "input.pt").read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(output_file.read_bytes()).hexdigest(),
    )
    (case / "native.json").write_text(json.dumps(metadata, indent=2) + "\n")


def summarize(report):
    """Keep per-case fingerprints/logs in a top-level JSON, even on failure."""
    cases = []
    for case in sorted((report / "native").glob("case-*")):
        item = {"case": str(case.relative_to(report))}
        for name in ("manifest", "native", "alignment"):
            path = case / (name + ".json")
            if path.exists():
                item[name] = json.loads(path.read_text())
        log = case / "native.txt"
        item["log_tail"] = log.read_text()[-8000:] if log.exists() else ""
        item["native_execution_completed"] = "native" in item
        cases.append(item)
    accuracy = report / "accuracy-full.json"
    native_tests = {}
    if accuracy.exists():
        for name, result in json.loads(accuracy.read_text()).items():
            if "vllm" in name:
                native_tests[name] = result
    result = {
        "scope": "functional comparison; original native dispatch in stock-Triton processes",
        "same_process_compatibility": "not established by this mode",
        "native_executions_completed": sum(
            c["native_execution_completed"] for c in cases
        ),
        "native_executions_attempted": len(cases),
        "native_test_results": native_tests,
        "cases": cases,
    }
    (report / "native-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Native summary:", report / "native-summary.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    else:
        run(args.case)
