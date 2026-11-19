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

"""Check installed MetaX QSA against a CPU reference in isolated processes."""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

SOURCE_SHA256 = "5b8bb9de8e5f7f122453811bd055ce752b6aab4f362551678196343a6cea6684"
NATIVE_CASES = [f"{dtype}_{rows}" for dtype in ("bf16", "fp8") for rows in (4, 65)]
DIAGNOSTIC_CASES = [
    "topk_4",
    "topk_65",
    "bf16_r16_s2_65",
    "bf16_r16_s1_65",
    "fp8_r16_s1_65",
    "fp8_input_bf16_dot_65",
]


def score_candidate(module, q, keys, table, starts, visible, max_query_len, stages):
    import torch

    logits = torch.empty((q.shape[0], 768), dtype=torch.float32, device=q.device)
    kernel = module._qsa_mqa_paged_prefill_kernel[
        (table.shape[0], module.triton.cdiv(max_query_len, 16), 1)
    ](
        q,
        keys,
        table,
        starts,
        visible,
        logits,
        *q.stride()[:-1],
        *keys.stride()[:2],
        *table.stride()[:-1],
        *logits.stride()[:-1],
        q.shape[0],
        0,
        table.shape[1],
        PAGE_SIZE=keys.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        TILE_R=16,
        BLOCK_N=64,
        K_TILES=16,
        STAGES=stages,
        num_warps=4,
    )
    return logits, getattr(getattr(kernel, "metadata", None), "shared", None)


def worker(args):
    result = {"case": args.case, "status": "failed"}
    try:
        import torch
        import triton

        importlib.import_module("vllm._custom_ops")
        module = importlib.import_module(
            "vllm_metax.models.qwen4_exp.metax.ops.qsa_indexer"
        )
        source = Path(module.__file__).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        result.update(
            python=sys.executable,
            torch=torch.__version__,
            triton_version=triton.__version__,
            triton_source=triton.__file__,
            source=str(source),
            source_sha256=digest,
        )
        assert digest == SOURCE_SHA256, "Installed QSA differs from pinned baseline"
        compiler = importlib.import_module("triton.compiler.compiler")
        result["compiler"] = {
            "source": compiler.__file__,
            "sha256": hashlib.sha256(Path(compiler.__file__).read_bytes()).hexdigest(),
        }
        torch.cuda.set_device(args.device)
        result["device"] = torch.cuda.get_device_name(args.device)
        result["capability"] = torch.cuda.get_device_capability(args.device)
        rows = int(args.case.rsplit("_", 1)[1])
        topk_only = args.case.startswith("topk_")
        native = args.case in NATIVE_CASES
        dtype = torch.float8_e4m3fn if args.case.startswith("fp8_") else torch.bfloat16
        dot_dtype = (
            torch.bfloat16 if args.case.startswith("fp8_input_bf16_dot") else dtype
        )
        result["dot_dtype"] = None if topk_only else str(dot_dtype)
        torch.manual_seed(1729)
        # Two requests, shuffled physical pages, nonzero query offset base.
        lengths = [rows // 2 + 1, rows - (rows // 2 + 1)]
        q_cpu = torch.randn(rows, 4, 128).to(dtype).float()
        k_cpu = torch.randn(8, 192, 1, 128).to(dtype).float()
        table_cpu = torch.randperm(8, dtype=torch.int64).reshape(2, 4)
        visible_cpu = torch.tensor(
            ([0, 1, 511, 768] * ((rows + 3) // 4))[:rows], dtype=torch.int32
        )
        visible = visible_cpu.to("cuda")
        reference = torch.full((rows, 768), -float("inf"))
        offset = 0
        for request, count in enumerate(lengths):
            keys = k_cpu[table_cpu[request], :, 0, :].reshape(768, 128)
            scores = (
                torch.einsum("rhd,nd->rhn", q_cpu[offset : offset + count], keys)
                .relu()
                .sum(1)
            )
            reference[offset : offset + count] = scores
            offset += count
        mask = torch.arange(768)[None, :] < visible_cpu[:, None]
        reference[~mask] = -float("inf")
        if topk_only:
            reference = torch.arange(768, dtype=torch.float32).expand(rows, -1).clone()
            logits = reference.masked_fill(~mask, float("inf")).to("cuda")
        else:
            q = q_cpu.to(device="cuda", dtype=dot_dtype)
            k = k_cpu.to(device="cuda", dtype=dot_dtype)
            table = table_cpu.to(device="cuda", dtype=torch.int32)
            starts = torch.tensor(
                [7, 7 + lengths[0], 7 + rows], dtype=torch.int32, device="cuda"
            )
            if native:
                logits = module._prefill_logits(
                    q, k, table, starts, visible, max(lengths), 768, 0, rows
                )
            else:
                stages = 2 if "_s2_" in args.case else 1
                result["launch"] = {"tile_r": 16, "stages": stages, "num_warps": 4}
                logits, shared = score_candidate(
                    module, q, k, table, starts, visible, max(lengths), stages
                )
                result["launch"]["shared_bytes"] = shared
            torch.cuda.synchronize()
            actual = logits.float().cpu()
            torch.testing.assert_close(
                actual[mask], reference[mask], atol=0.05, rtol=0.005
            )
            error = (actual[mask] - reference[mask]).abs().max().item()
            result["score"] = {"status": "passed", "max_abs_error": error}
        platform = module.current_platform
        cooperative = (
            rows <= 64
            and logits.stride(0) % 4 == 0
            and platform.has_device_capability(90)
            and not platform.is_device_capability_family(120)
        )
        selected = "cooperative_topk" if cooperative else "persistent_topk"
        result["topk_dispatch"] = selected
        op = getattr(torch.ops._C, selected)
        result["topk_schema"] = str(op._schemas)
        output = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
        if native:
            module.qsa_select_paged_prefill(
                q, k, table, starts, visible, 2048, 4, max(lengths), output, 3072
            )
        else:
            workspace = torch.empty(
                (module._TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device="cuda"
            )
            module._topk(logits, visible, 2048, 4, output, workspace)
        torch.cuda.synchronize()
        indices = output.cpu().long()
        for row in range(rows):
            n = int(visible_cpu[row])
            selected_indices = indices[row][indices[row] >= 0]
            count = min(n, 512)
            assert selected_indices.numel() == count, (row, "selection count")
            assert (indices[row][indices[row] < 0] == -1).all(), "padding"
            assert (selected_indices < n).all(), (row, "index outside visible range")
            assert selected_indices.unique().numel() == count, (row, "duplicates")
            if n > 512:
                # Permit reordered ties; check rank against the CPU score oracle.
                selected_scores = reference[row, selected_indices]
                threshold = reference[row, :n].topk(512).values[-1]
                tolerance = 0 if topk_only else 0.1 + 0.01 * threshold.abs()
                assert selected_scores.min() >= threshold - tolerance, (
                    row,
                    "TopK rank violation",
                )
        result["selection"] = {"status": "passed", "tie_policy": "score threshold"}
        result["status"] = "passed"
    except Exception:
        result["error"] = traceback.format_exc()
        traceback.print_exc()
    Path(args.report).mkdir(parents=True, exist_ok=True)
    (Path(args.report) / f"{args.case}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--case", choices=NATIVE_CASES + DIAGNOSTIC_CASES)
    parser.add_argument("--suite", choices=("native", "diagnostic"), default="native")
    args = parser.parse_args()
    if args.case:
        return worker(args)
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    results = []
    cases = NATIVE_CASES if args.suite == "native" else DIAGNOSTIC_CASES
    for case in cases:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--report",
            str(report.resolve()),
            "--device",
            str(args.device),
            "--case",
            case,
        ]
        print(f"=== {case} ===", flush=True)
        with (report / f"{case}.log").open("w") as log:
            try:
                status = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout
                ).returncode
            except subprocess.TimeoutExpired:
                status = 124
        path = report / f"{case}.json"
        item = json.loads(path.read_text()) if path.exists() else {"case": case}
        item["exit_code"] = status
        results.append(item)
        print(json.dumps(item, indent=2), flush=True)
        if status and "error" not in item:
            print((report / f"{case}.log").read_text()[-6000:], flush=True)
    summary = {
        "status": "passed" if all(x["exit_code"] == 0 for x in results) else "failed",
        "suite": args.suite,
        "scope": "Correctness only; diagnostic launches change native configuration",
        "cases": results,
    }
    (report / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
