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

"""Isolate normal-import TopK writes; no production or compiler modifications."""

import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

CASES = ("isolated_topk", "normal_topk", "isolated_after_import")


def run(args, result):
    import torch
    import torch_npu
    import triton

    if args.worker == "normal_topk":
        from qsa_prefill_production_910b import load_production

        _, _, module, identity = load_production()
        result["production_entry"] = identity
    else:
        if args.worker == "isolated_after_import":
            from qsa_prefill_production_910b import load_production

            result["production_entry"] = load_production()[3]
        from qsa_prefill_pipeline_910b_trial import load_topk

        module = load_topk()
    os.environ["FLAGGEMS_ASCEND_PTK_PATH"] = "auto"
    device = f"npu:{args.device}"
    rows, k, width = 6, 512, 1728
    visible = [0, 1, 511, 512, 513, 1537]
    torch.manual_seed(20261029)
    q_cpu = torch.randn((rows, 4, 128), dtype=torch.bfloat16)
    key_cpu = torch.randn((9, 192, 1, 128), dtype=torch.bfloat16)
    table_cpu = torch.randperm(9).to(torch.int32).reshape(1, 9)
    cpu_logits = torch.full((rows, width), -torch.inf)
    for row, n in enumerate(visible):
        positions = torch.arange(n)
        keys = key_cpu[
            table_cpu[0, positions // 192].long(), positions % 192, 0
        ].float()
        cpu_logits[row, :n] = (
            torch.einsum("hd,kd->hk", q_cpu[row].float(), keys).clamp_min(0).sum(0)
        )
    logits = cpu_logits.to(device)
    lengths = torch.tensor(visible, device=device, dtype=torch.int32)
    stages = []
    result.update(
        diagnostic_only=True,
        loader=module.probe_identity,
        triton=dict(version=triton.__version__, source=triton.__file__),
        cache_directory=os.environ.get("TRITON_CACHE_DIR"),
        initial_route="auto",
        observed_route="radix",
        stages=stages,
    )

    def snapshot(label, output, guard=None, verify=False):
        torch.npu.synchronize()
        values = output.cpu()
        info = dict(
            label=label,
            pointer=output.data_ptr(),
            storage_offset=output.storage_offset(),
            shape=list(output.shape),
            stride=list(output.stride()),
            rows=[],
        )
        if guard is not None:
            cpu_guard = guard.cpu()
            info["canaries_preserved"] = bool(
                (cpu_guard[:16] == -123456).all() and (cpu_guard[-16:] == -123456).all()
            )
        try:
            info["npu_format"] = torch_npu.get_npu_format(output)
        except (AttributeError, RuntimeError):
            info["npu_format"] = None
        for row, n in enumerate(visible):
            data = values[row]
            valid = data[(data >= 0) & (data < n)].long()
            negative = data[data < -1]
            record = dict(
                row=row,
                visible=n,
                below_minus_one=int(negative.numel()),
                sentinel_count=int((data == -123456).sum()),
                out_of_range=int((data >= n).sum()),
                minus_one=int((data == -1).sum()),
                valid_count=int(valid.numel()),
                unique_count=int(valid.unique().numel()),
                prefix=data[:16].tolist(),
                negative_samples=negative[:16].tolist(),
            )
            if verify:
                count = min(n, k)
                passed = (
                    record["below_minus_one"] == 0
                    and record["out_of_range"] == 0
                    and valid.numel() == count
                    and valid.unique().numel() == count
                    and record["minus_one"] == k - count
                )
                if count and passed:
                    actual = cpu_logits[row, valid].sort().values
                    expected = cpu_logits[row, :n].topk(count).values.sort().values
                    record["selected_score_max_abs_error"] = float(
                        (actual - expected).abs().max()
                    )
                    passed = bool(
                        torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)
                    )
                record["correct"] = bool(passed)
            info["rows"].append(record)
        if verify:
            info["correct"] = all(row["correct"] for row in info["rows"]) and info.get(
                "canaries_preserved", True
            )
        stages.append(info)
        print("snapshot", json.dumps(info), flush=True)
        return values

    def buffers(guarded=True):
        if guarded:
            guard = torch.full(
                (rows * k + 32,), -123456, device=device, dtype=torch.int32
            )
            output = guard[16:-16].view(rows, k)
        else:
            guard = None
            output = torch.full((rows, k), -123456, device=device, dtype=torch.int32)
        workspace = torch.empty((1024 * 1024,), device=device, dtype=torch.uint8)
        return guard, output, workspace

    # Run first with no observer or extra synchronization between TopK kernels.
    guard, output, workspace = buffers()
    module.persistent_topk(logits, lengths, output, workspace, k=k, max_seq_len=width)
    before = snapshot(
        "unmodified_auto_guarded_before_native", output, guard, verify=True
    )
    from qsa_lightning_baseline_910b import qsa_select_paged_prefill

    native_output = torch.empty_like(output)
    qsa_select_paged_prefill(
        q_cpu.to(device),
        key_cpu.to(device),
        table_cpu.to(device),
        torch.tensor([0, rows], device=device, dtype=torch.int32),
        lengths,
        2048,
        4,
        rows,
        native_output,
        max(visible) * 4,
    )
    after = snapshot("unmodified_auto_guarded_after_native", output, guard, verify=True)
    result["native_call_preserved_output"] = bool(torch.equal(before, after))
    guard, output, workspace = buffers(False)
    module.persistent_topk(logits, lengths, output, workspace, k=k, max_seq_len=width)
    snapshot("unmodified_auto_base_allocation", output, verify=True)

    result["auto_route_states"] = dict(
        segmerge=dict(module._SEGMERGE_STATE), persistent=dict(module._PSEGMERGE_STATE)
    )
    os.environ["FLAGGEMS_ASCEND_PTK_PATH"] = "radix"
    guard, output, workspace = buffers()
    module.persistent_topk(logits, lengths, output, workspace, k=k, max_seq_len=width)
    snapshot("unmodified_radix_guarded", output, guard, verify=True)

    # Observe the existing radix kernels separately; generated code is unchanged.
    guard, output, workspace = buffers()

    class Observe:
        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __getitem__(self, grid):
            def launch(*operands, **kwargs):
                returned = self.kernel[grid](*operands, **kwargs)
                snapshot(self.name, output, guard)
                return returned

            return launch

    names = (
        "_radix_hist_trivial_all",
        "_radix_hist_clear",
        "_radix_hist_hist",
        "_radix_hist_threshold",
        "_radix_hist_count",
        "_radix_hist_prefix",
        "_radix_hist_write_guar",
        "_radix_hist_write_final",
    )
    result["kernel_keys"] = {
        name: str(getattr(getattr(module, name), "cache_key", "unavailable"))
        for name in names
    }
    with ExitStack() as stack:
        for name in names:
            stack.enter_context(
                patch.object(module, name, Observe(name, getattr(module, name)))
            )
        module.persistent_topk(
            logits, lengths, output, workspace, k=k, max_seq_len=width
        )
    snapshot("observed_radix_final", output, guard, verify=True)
    result["topk_source_sha256"] = hashlib.sha256(
        Path(module.__file__).read_bytes()
    ).hexdigest()
    result["correctness_passed"] = all(
        stage["correct"] for stage in stages if "correct" in stage
    )
    result["status"] = (
        "passed" if result["correctness_passed"] else "correctness_failed"
    )
    print(
        "diagnostic_only",
        json.dumps(
            dict(
                correctness_passed=result["correctness_passed"],
                native_call_preserved_output=result["native_call_preserved_output"],
            )
        ),
        flush=True,
    )
    return 0 if result["correctness_passed"] else 1
