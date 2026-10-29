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
ADAPTER_CASES = {
    "empty_batch": dict(lengths=[0, 0], visible=[], page=192, base=137),
    "offset_ragged": dict(
        lengths=[3, 0, 2, 1],
        visible=[0, 1, 511, 512, 513, 1537],
        page=192,
        base=137,
        int64=True,
    ),
    "strided": dict(
        lengths=[2, 0, 1], visible=[513, 1537, 0], page=192, base=53, strided=True
    ),
    "topk1024": dict(lengths=[3], visible=[0, 1023, 1025], page=192, k=1024),
    "topk2048": dict(lengths=[3], visible=[1, 2047, 2049], page=192, k=2048),
    "heads1": dict(lengths=[3], visible=[0, 512, 1537], page=192, heads=1),
    "heads64": dict(lengths=[4], visible=[0, 511, 513, 1537], page=192, heads=64),
    "batch128": dict(
        lengths=[63, 0, 33, 32],
        visible=[8193 - i % 31 for i in range(128)],
        page=192,
        base=211,
    ),
    "all_zero": dict(lengths=[1, 2], visible=[0, 0, 0], page=192),
    "empty_requests": dict(
        lengths=[0, 2, 0, 1, 0], visible=[512, 513, 0], page=192, base=97
    ),
}

SCORE_CASES = {
    "tile_tail": dict(lengths=[1], visible=[37], page=192, width=67),
    "score_ragged": dict(
        lengths=[3, 0, 2, 1],
        visible=[0, 1, 511, 512, 513, 1537],
        page=192,
        base=137,
        width=1551,
        int64=True,
    ),
    "score_chunk": dict(
        lengths=[3, 0, 2, 1],
        visible=[0, 1, 511, 512, 513, 1537],
        page=192,
        base=137,
        width=1551,
        offset=1,
        count=4,
    ),
    "score_strided": dict(
        lengths=[2, 0, 1],
        visible=[513, 1537, 0],
        page=192,
        base=53,
        strided=True,
        width=1547,
    ),
    "score_heads1": ADAPTER_CASES["heads1"],
    "score_heads64": ADAPTER_CASES["heads64"],
    "score_batch128": ADAPTER_CASES["batch128"],
    "score_empty": ADAPTER_CASES["empty_batch"],
    "score_zero": ADAPTER_CASES["all_zero"],
    "score_page196": dict(lengths=[2], visible=[193, 513], page=196, width=517),
    "score_page16": dict(lengths=[2], visible=[17, 513], page=16, width=527),
}

PIPELINE_CASES = {
    "topk_control": dict(
        lengths=[6], visible=[0, 1, 511, 512, 513, 1537], page=192, topk_only=True
    ),
    **ADAPTER_CASES,
    "chunk_reuse": dict(
        lengths=[7, 0, 11],
        visible=[1000 + i * 4 for i in range(18)],
        page=192,
        base=211,
        chunk_rows=5,
    ),
    "radix_control": dict(lengths=[3], visible=[0, 513, 1537], page=192, route="radix"),
    "ties": dict(lengths=[3], visible=[0, 513, 1537], page=192, zero_query=True),
}

K, HEADS, DIM = 512, 4, 128
RTOL, ATOL = 1e-3, 2e-3


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def score_trial(
    config,
    query,
    key,
    table,
    starts,
    visible,
    q_cpu,
    key_cpu,
    table_cpu,
    owner_cpu,
    result,
):
    import torch
    from qsa_prefill_score_910b_trial import score

    rows = query.shape[0]
    offset = config.get("offset", 0)
    count = config.get("count", rows)
    width = config.get("width", table.shape[1] * key.shape[1])
    guard = torch.full((count * width + 32,), 12345.0, device=query.device)
    output = guard[16 : 16 + count * width].view(count, width)
    pointer = output.data_ptr()
    assert score(query, key, table, starts, visible, output, offset=offset) is output
    torch.npu.synchronize()
    assert output.data_ptr() == pointer
    assert (guard[:16] == 12345.0).all() and (guard[-16:] == 12345.0).all()
    actual = output.cpu()
    expected = torch.full((count, width), -torch.inf)
    lengths = config["visible"]
    page = key.shape[1]
    for local in range(count):
        row = local + offset
        live = min(lengths[row], width)
        if not live:
            continue
        columns = torch.arange(live)
        pages = table_cpu[owner_cpu[row], columns // page].long()
        keys = key_cpu[pages, columns % page, 0].float()
        expected[local, :live] = (
            torch.einsum("hd,kd->hk", q_cpu[row].float(), keys).clamp_min(0).sum(0)
        )
    finite = torch.isfinite(expected)
    assert torch.isneginf(actual[~finite]).all()
    error = (actual[finite] - expected[finite]).abs()
    result["score"] = dict(
        shape=list(output.shape),
        offset=offset,
        dtype=str(output.dtype),
        valid_values=int(finite.sum()),
        inactive_values=int((~finite).sum()),
        max_abs_error=float(error.max()) if error.numel() else 0.0,
        rtol=1e-3,
        atol=1e-3,
        canaries_preserved=True,
        input_strides=dict(
            q=list(query.stride()), key=list(key.stride()), table=list(table.stride())
        ),
        query_start_loc=starts.cpu().tolist(),
        scale="unscaled",
        includes_topk=False,
    )
    torch.testing.assert_close(actual[finite], expected[finite], rtol=1e-3, atol=1e-3)
    result["status"] = "passed"
    print("score", json.dumps(result["score"]), flush=True)
    return 0


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
        if args.suite == "score":
            import qsa_prefill_score_910b_trial
            import triton

            result["runtime"]["triton"] = dict(
                version=triton.__version__,
                source=triton.__file__,
                source_sha256=hashlib.sha256(
                    Path(triton.__file__).read_bytes()
                ).hexdigest(),
            )
            result["score_source_sha256"] = hashlib.sha256(
                Path(qsa_prefill_score_910b_trial.__file__).read_bytes()
            ).hexdigest()
        if args.suite in ("pipeline", "production"):
            if args.suite == "production":
                from qsa_prefill_production_910b import load_production, load_topk

                result["production_entry"] = load_production()[3]
                if args.worker == "environment":
                    from qsa_prefill_production_910b import check_rejections

                    result["contract_rejections"] = check_rejections(
                        f"npu:{args.device}"
                    )
                    print(
                        "production_entry",
                        json.dumps(result["production_entry"]),
                        flush=True,
                    )
                    print(
                        "contract_rejections", result["contract_rejections"], flush=True
                    )
            else:
                from qsa_prefill_pipeline_910b_trial import load_topk
            result["existing_topk"] = load_topk().probe_identity
        if args.worker == "environment":
            result["status"] = "passed"
            return 0
        config = CASES[args.worker]
        lengths, visible, page = config["lengths"], config["visible"], config["page"]
        K, HEADS = config.get("k", 512), config.get("heads", 4)
        atol = 1e-3 if args.suite in ("pipeline", "production") else ATOL
        rows, requests = sum(lengths), len(lengths)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        capacity = max(K + 1, max(visible, default=0))
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
            atol=atol,
        )
        if args.suite in ("adapter", "score", "pipeline", "production"):
            from qsa_lightning_baseline_910b import qsa_select_paged_prefill

            base = config.get("base", 0)
            starts_cpu = torch.tensor(
                [base, *[base + sum(lengths[: i + 1]) for i in range(requests)]]
            )
            metadata_dtype = torch.int64 if config.get("int64") else torch.int32
            starts = starts_cpu.to(device=device, dtype=metadata_dtype)
            table = table_cpu.to(device=device, dtype=metadata_dtype)
            if base:
                storage = torch.zeros(
                    (base + rows, HEADS, DIM), device=device, dtype=query.dtype
                )
                storage[base:].copy_(query)
                query = storage[base:]
            if config.get("strided"):
                storage = torch.zeros(
                    (rows, HEADS + 1, DIM), device=device, dtype=query.dtype
                )
                storage[:, :HEADS].copy_(query)
                query = storage[:, :HEADS]
                storage = torch.zeros(
                    (key.shape[0], page + 1, 1, DIM), device=device, dtype=key.dtype
                )
                storage[:, :page].copy_(key)
                key = storage[:, :page]
                storage = torch.full(
                    (requests, columns * 2), -1, device=device, dtype=table.dtype
                )
                storage[:, ::2].copy_(table)
                table = storage[:, ::2]
            if args.suite == "score":
                return score_trial(
                    config,
                    query,
                    key,
                    table,
                    starts,
                    key_lengths,
                    q_cpu,
                    key_cpu,
                    table_cpu,
                    owner_cpu,
                    result,
                )
            guard = torch.full(
                (rows * K + 32,), -123456, device=device, dtype=torch.int32
            )
            output = guard[16 : 16 + rows * K].view(rows, K)
            pointer = output.data_ptr()
            native_calls = []

            def captured_native(*operands, **kwargs):
                native_calls.append(True)
                expected_table = table_cpu[owner_cpu].to(device)
                if not torch.equal(kwargs["block_table"], expected_table):
                    raise AssertionError("Adapter mapped rows to incorrect requests")
                return torch_npu.npu_lightning_indexer(*operands, **kwargs)

            if args.suite in ("pipeline", "production"):
                if args.suite == "production":
                    from qsa_prefill_production_910b import load_topk
                    from qsa_prefill_production_910b import (
                        qsa_select_paged_prefill as pipeline,
                    )
                else:
                    from qsa_prefill_pipeline_910b_trial import load_topk
                    from qsa_prefill_pipeline_910b_trial import (
                        qsa_select_paged_prefill as pipeline,
                    )
                os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = str(
                    config.get("budget_mb", 512)
                )
                os.environ["FLAGGEMS_ASCEND_PTK_PATH"] = config.get("route", "auto")
                topk_module = load_topk()
                if config.get("zero_query"):
                    q_cpu.zero_()
                    query.zero_()
                trace = []
                if config.get("topk_only"):
                    width = columns * page
                    cpu_logits = torch.full((rows, width), -torch.inf)
                    for row, length in enumerate(visible):
                        positions = torch.arange(length)
                        physical = table_cpu[owner_cpu[row], positions // page].long()
                        keys = key_cpu[physical, positions % page, 0].float()
                        cpu_logits[row, :length] = (
                            torch.einsum("hd,kd->hk", q_cpu[row].float(), keys)
                            .clamp_min(0)
                            .sum(0)
                        )
                    logits = cpu_logits.to(device)
                    workspace = torch.empty(
                        (1024 * 1024,), device=device, dtype=torch.uint8
                    )
                    topk_module.persistent_topk(
                        logits, key_lengths, output, workspace, k=K, max_seq_len=width
                    )
                    returned = None
                else:
                    returned = pipeline(
                        query,
                        key,
                        table,
                        starts,
                        key_lengths,
                        K * 4,
                        4,
                        max(lengths),
                        output,
                        capacity * 4,
                        chunk_rows=config.get("chunk_rows", 16),
                        trace=trace,
                    )
                torch.npu.synchronize()
                baseline = torch.empty_like(output)
                qsa_select_paged_prefill(
                    query,
                    key,
                    table,
                    starts,
                    key_lengths,
                    K * 4,
                    4,
                    max(lengths),
                    baseline,
                    capacity * 4,
                    native=captured_native,
                )
                result["pipeline"] = dict(
                    kind=(
                        "topk_only"
                        if config.get("topk_only")
                        else "score_plus_existing_topk"
                    ),
                    existing_topk=topk_module.probe_identity,
                    route_mode=os.environ["FLAGGEMS_ASCEND_PTK_PATH"],
                    segmerge_state=dict(topk_module._SEGMERGE_STATE),
                    persistent_segmerge_state=dict(topk_module._PSEGMERGE_STATE),
                    chunks=trace,
                )
                print("pipeline", json.dumps(result["pipeline"]), flush=True)
                if trace:
                    if args.suite == "production" and config.get("budget_mb") == 0:
                        assert len(trace) == rows and all(t["rows"] == 1 for t in trace)
                    assert len({t["logits_pointer"] for t in trace}) == 1
                    assert len({t["topk_workspace_pointer"] for t in trace}) == 1
                baseline = baseline.cpu()
                result["native_reference_output"] = dict(
                    shape=list(baseline.shape),
                    padding_count=int((baseline == -1).sum()),
                )
            else:
                returned = qsa_select_paged_prefill(
                    query,
                    key,
                    table,
                    starts,
                    key_lengths,
                    K * 4,
                    4,
                    max(lengths),
                    output,
                    capacity * 4,
                    native=captured_native,
                )
            torch.npu.synchronize()
            assert returned is None and output.data_ptr() == pointer
            assert len(native_calls) == (1 if rows else 0)
            assert (guard[:16] == -123456).all() and (guard[-16:] == -123456).all()
            assert (output >= -1).all()
            assert ((output == -1) | (output < key_lengths[:, None])).all()
            result["adapter"] = dict(
                query_start_loc=starts_cpu.tolist(),
                query_stride=list(query.stride()),
                key_stride=list(key.stride()),
                table_stride=list(table.stride()),
                query_storage_offset=query.storage_offset(),
                metadata_dtype=str(table.dtype),
                native_calls=len(native_calls),
                output_pointer_preserved=True,
                canaries_preserved=True,
            )
            # Keep the common group-output verifier for both suites.
            native = (output.unsqueeze(1), None)
        else:
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
        result["raw_output" if args.suite == "native" else "adapter_output"] = dict(
            shape=list(indices.shape), dtype=str(indices.dtype)
        )
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
            if args.suite in ("adapter", "pipeline", "production"):
                info["output_prefix"] = info.pop("raw_prefix")
                info["minus_one"] = info.pop("raw_minus_one")
                info["out_of_range"] = info.pop("raw_out_of_range")
            result["rows"].append(info)
            if (
                selected.numel() != expected_count
                or selected.unique().numel() != expected_count
            ):
                errors.append(f"row {row}: valid group count/uniqueness mismatch")
                continue
            if args.suite in ("pipeline", "production") and not count:
                if not (baseline[row] == -1).all():
                    errors.append(f"row {row}: native zero-visible padding differs")
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
            if args.suite in ("pipeline", "production"):
                reference = baseline[row].cpu().long()
                reference_valid = reference[reference >= 0]
                if (
                    reference_valid.numel() != expected_count
                    or reference_valid.unique().numel() != expected_count
                    or (reference_valid >= count).any()
                ):
                    errors.append(f"row {row}: invalid native baseline groups")
                else:
                    native_values = scores[reference_valid].sort().values
                    info["native_selected_score_max_abs_error"] = float(
                        (actual - native_values).abs().max()
                    )
                    if not torch.allclose(actual, native_values, rtol=RTOL, atol=atol):
                        errors.append(
                            f"row {row}: pipeline differs from native baseline selection"
                        )
            if not torch.allclose(actual, expected, rtol=RTOL, atol=atol):
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
    global CASES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="logical index after ASCEND_RT_VISIBLE_DEVICES",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--suite",
        choices=("native", "adapter", "score", "pipeline", "production"),
        default="native",
    )
    parser.add_argument(
        "--worker",
        choices=(
            "environment",
            "zero_budget",
            *CASES,
            *ADAPTER_CASES,
            *SCORE_CASES,
            *PIPELINE_CASES,
        ),
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.device < 0 or args.timeout <= 0:
        parser.error("Invalid device or timeout")
    if args.suite == "adapter":
        CASES = ADAPTER_CASES
    elif args.suite == "score":
        CASES = SCORE_CASES
    elif args.suite in ("pipeline", "production"):
        CASES = dict(PIPELINE_CASES)
        if args.suite == "production":
            CASES["zero_budget"] = dict(ADAPTER_CASES["offset_ragged"], budget_mb=0)
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
        suite=args.suite,
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
    adapter_path = Path(__file__).with_name("qsa_lightning_baseline_910b.py")
    if args.suite == "adapter":
        summary["adapter_sha256"] = hashlib.sha256(
            adapter_path.read_bytes()
        ).hexdigest()
    if args.suite == "score":
        summary["score_source_sha256"] = hashlib.sha256(
            Path(__file__).with_name("qsa_prefill_score_910b_trial.py").read_bytes()
        ).hexdigest()
        summary["scope"] = (
            "BF16 Triton score compile/correctness probe; no TopK or performance claim"
        )
    if args.suite in ("pipeline", "production"):
        root = Path(__file__).resolve().parents[1]
        summary["source_hashes"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "tools/qsa_prefill_pipeline_910b_trial.py",
                "tools/qsa_prefill_score_910b_trial.py",
                "tools/qsa_lightning_baseline_910b.py",
                "src/flaggems_vllm/runtime/backend/_ascend/ops/persistent_topk.py",
                "src/flaggems_vllm/utils/triton_version_utils.py",
            )
        }
        summary["scope"] = (
            "score + existing Ascend TopK correctness against native LightningIndexer; no performance claim"
        )
    if args.suite == "production":
        root = Path(__file__).resolve().parents[1]
        summary["scope"] = (
            "normal public Ascend entry correctness; no performance claim"
        )
        summary["source_hashes"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "src/flaggems_vllm/__init__.py",
                "src/flaggems_vllm/runtime/backend/_ascend/ops/__init__.py",
                "src/flaggems_vllm/runtime/backend/_ascend/ops/qsa_select_paged_prefill.py",
                "src/flaggems_vllm/runtime/backend/_ascend/ops/persistent_topk.py",
                "tools/qsa_prefill_production_910b.py",
                "tools/qsa_lightning_baseline_910b.py",
            )
        }
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
                        "--suite",
                        args.suite,
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
        if (
            args.suite in ("score", "pipeline", "production")
            and name != "environment"
            and code
        ):
            summary["status"] = "failed"
            names = list(CASES)
            summary["not_run"] = names[names.index(name) + 1 :]
            write(report / "summary.json", summary)
            print(
                "Stopping integration probe at first failure; see per-case diagnostics.",
                flush=True,
            )
            return 1
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
