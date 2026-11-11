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


"""Isolate N128 K loading, Cube dot, and cross-core ring transport."""

import json
import traceback


def run_controls(inputs, values, width, report, case):
    import torch
    import triton

    from probe_qsa_int8_mixed_pipeline_910b import (
        _qsa_n128_cube_control, _qsa_n128_handoff_control,
    )
    from qsa_int8_fused_binary_910b_trial import pack_mixed_queries

    device = values["q"].device
    rows = inputs["q"].shape[0]
    padded = triton.cdiv(width, 128) * 128
    logical = torch.zeros((padded, 128), dtype=torch.int8)
    logical[:width] = inputs["cache"][inputs["row_table"][0].long(), :, 0].reshape(-1, 128)[:width]
    dense = logical.to(device)
    packed_q = pack_mixed_queries(values["q"])
    expected_q = torch.zeros((16, 128), dtype=torch.int8)
    expected_q[:8] = inputs["q"][:2].reshape(8, 128)
    starts = sorted({0, 128, 256, 384, padded - 128})
    starts_device = torch.tensor(starts, device=device, dtype=torch.int32)
    expected_k = torch.stack([logical[start:start + 128] for start in starts])
    expected_dot = torch.stack([expected_q.int() @ key.int().T for key in expected_k])
    result = dict(case=case, width=width, rows=rows, starts=starts, arms={})
    manifest_path = report / f"{case}-n128-component-controls.json"

    def save(label, compiled):
        files = []
        for key, text in getattr(compiled, "asm", {}).items():
            if isinstance(text, str):
                name = f"{case}-{label}-{key}.txt"
                (report / name).write_text(text)
                files.append(name)
        return files

    def finish(label, entry):
        result["arms"][label] = entry
        manifest_path.write_text(json.dumps(result, indent=2))
        print(f"QSA N128 COMPONENT {case} {label}: {json.dumps(entry)}", flush=True)

    for is_dense in (True, False):
        for trace in (False, True):
            label = f"cube-{'dense' if is_dense else 'paged'}-{'trace' if trace else 'plain'}"
            entry = {}
            try:
                out = torch.full((len(starts), 16, 128), 0x13579BDF, device=device, dtype=torch.int32)
                q_out = torch.empty((len(starts), 16, 128), device=device, dtype=torch.int8)
                k_out = torch.empty((len(starts), 128, 128), device=device, dtype=torch.int8)
                compiled = _qsa_n128_cube_control[(len(starts),)](
                    packed_q, dense if is_dense else values["cache"].view(-1, 128),
                    values["row_table"][0], starts_device, out, q_out, k_out,
                    192, width, is_dense, trace,
                    enable_auto_bind_sub_block=False, multibuffer=False,
                )
                torch.npu.synchronize()
                entry["artifacts"] = save(label, compiled)
                actual = out.cpu()
                entry.update(dot_mismatches=int((actual != expected_dot).sum()),
                             observed_sample=actual[0, 0, :8].tolist(),
                             expected_sample=expected_dot[0, 0, :8].tolist(),
                             untouched=int((actual == 0x13579BDF).sum()))
                if trace:
                    entry["q_mismatches"] = int((q_out.cpu() != expected_q.expand_as(q_out.cpu())).sum())
                    entry["k_mismatches"] = int((k_out.cpu() != expected_k).sum())
                entry["passed"] = all(entry.get(key, 0) == 0 for key in ("dot_mismatches", "q_mismatches", "k_mismatches", "untouched"))
            except Exception:
                entry.update(passed=False, error=traceback.format_exc())
            finish(label, entry)

    for steps in sorted({1, 2, triton.cdiv(width, 128)}):
        label = f"handoff-only-{steps}"
        entry = {}
        try:
            cores = triton.cdiv(rows, 2)
            ring = torch.full((cores, 2, 16, 128), 0x13579BDF, device=device, dtype=torch.int32)
            out = torch.full((cores, steps, 8, 128), 0x13579BDF, device=device, dtype=torch.int32)
            compiled = _qsa_n128_handoff_control[(cores,)](
                packed_q, values["cache"].view(-1, 128), values["row_table"][0], ring, out,
                values["qs"], values["ks"].view(-1), values["weights"], values["visible"],
                steps, 2, 192, 128, rows, True, True,
                ONLINE_TOP=512, OnlineIndices=out, PACKED_OUTPUT=False,
                KEEP_TILE_STORE=True, UB_SNAPSHOT=True, HANDOFF_STAGE=1, SCAN_WIDTH=width,
                use_bytecode=True, disable_auto_cv_work_space_manage=True,
                disable_auto_inject_block_sync=True, enable_auto_bind_sub_block=False,
                multibuffer=False, sync_solver=False, unit_flag=False,
            )
            torch.npu.synchronize()
            entry["artifacts"] = save(label, compiled)
            raw = out.cpu()
            actual = raw.reshape(cores, steps, 2, 4, 128).permute(0, 2, 3, 1, 4).reshape(rows, 4, steps * 128)
            expected = torch.matmul(inputs["q"].int(), logical[:steps * 128].int().T)
            bad = actual != expected
            entry.update(dot_mismatches=int(bad.sum()), untouched=int((actual == 0x13579BDF).sum()),
                         first_bad=bad.nonzero()[:8].tolist(), observed_sample=actual[0, 0, :8].tolist(),
                         expected_sample=expected[0, 0, :8].tolist(), passed=not bool(bad.any()))
        except Exception:
            entry.update(passed=False, error=traceback.format_exc())
        finish(label, entry)
    return result
