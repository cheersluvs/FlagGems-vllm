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

"""Validate and time the FP32 serial CANN-structure control."""

import json
import math
import statistics
from pathlib import Path


def run_probe(inputs, values, width, full_scores, control, timer, report, case):
    import torch
    from qsa_cann_serial_910b_trial import PreparedCANNSerial
    from probe_qsa_int8_score_910b import reference
    from probe_qsa_int8_pipeline_910b import check_selection
    from probe_qsa_int8_ten_shapes_910b import verify_structure

    result = dict(case=case, status="failed", mode="kernel", scope="32-row serial structural probe",
                  outer_keys=2048, cube_n=64, state_capacity=2048, output_k=512,
                  score_dtype="FP32", state_storage="GM ping-pong, sequential kernel launches",
                  blocks=list(range(0, width, 2048)), checks=[])
    adapter = PreparedCANNSerial(values, width)
    refs = [reference(inputs, row)[1] for row in range(adapter.rows)]
    path = Path(report) / f"{case}-cann-serial.json"

    def check(stage, start, tensor, raw):
        actual = tensor.cpu()
        bad = 0
        if stage == "score":
            dots = raw.cpu()
            for row, ref in enumerate(refs):
                expected = torch.full((2048,), -float("inf"))
                end = min(start + 2048, ref.numel(), width)
                count = max(0, end - start)
                if count:
                    expected[:count] = ref[start:end]
                bad += int((~torch.isclose(actual[row], expected, rtol=1e-3, atol=1e-3)).sum())
                expected_dot = torch.zeros((2048, 4), dtype=torch.int32)
                if count:
                    physical = inputs["row_table"][row].long()
                    keys = inputs["cache"][physical, :, 0].reshape(-1, 128)[start:end].int()
                    expected_dot[:count] = keys @ inputs["q"][row].int().T
                bad += int((dots[row] != expected_dot).sum())
        else:
            scores = actual[:, 0::2]
            indices = actual.view(torch.int32)[:, 1::2].long()
            bad += int((scores[:, :-1] < scores[:, 1:]).sum())
            for row, ref in enumerate(refs):
                if stage == "sort":
                    expected_indices = torch.arange(start, start + 2048)
                    bad += int((torch.sort(indices[row]).values != expected_indices).sum())
                    expected = torch.full((2048,), -float("inf"))
                    valid = (indices[row] >= 0) & (indices[row] < min(ref.numel(), width))
                    expected[valid] = ref[indices[row, valid]]
                    bad += int((~torch.isclose(scores[row], expected, rtol=1e-3, atol=1e-3)).sum())
                else:
                    ref = ref[:min(start + 2048, width)]
                    finite = torch.isfinite(scores[row])
                    count = min(2048, ref.numel())
                    bad += abs(int(finite.sum()) - count)
                    idx = indices[row, finite]
                    in_range = (idx >= 0) & (idx < ref.numel())
                    bad += int((~in_range).sum()) + idx.numel() - idx.unique().numel()
                    if bool(in_range.all()) and idx.numel():
                        bad += int((~torch.isclose(scores[row, finite], ref[idx], rtol=1e-3, atol=1e-3)).sum())
                    if int(finite.sum()) == count:
                        bad += int((~torch.isclose(scores[row, finite], ref.topk(count).values, rtol=1e-3, atol=1e-3)).sum())
            if raw is not None:
                consumed = raw.cpu()
                bad += int((consumed[:, :2].sum(1) < 2048).sum())
        entry = dict(stage=stage, key_start=start, mismatches=bad)
        result["checks"].append(entry)
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
        print("QSA CANN SERIAL CHECK " + json.dumps(entry), flush=True)
        assert bad == 0, (case, entry)

    try:
        adapter.run(check=check)
        indices = adapter.output.cpu()
        verify_structure(indices, inputs["visible"])
        selected, _ = check_selection(indices, refs)
        assert selected["checks_passed"], selected
        actual_values = adapter.output_values.cpu()
        for row, ref in enumerate(refs):
            valid = indices[row] >= 0
            torch.testing.assert_close(actual_values[row, valid], ref[indices[row, valid].long()], rtol=1e-3, atol=1e-3)
            assert torch.isneginf(actual_values[row, ~valid]).all()
        # Validate the independently timed TopK-only arm on frozen full scores too.
        adapter.run(full_source=full_scores)
        verify_structure(adapter.output.cpu(), inputs["visible"])
        checks, _ = check_selection(adapter.output.cpu(), refs)
        assert checks["checks_passed"], checks
        result["correctness_status"] = "passed"
        result["timing_status"] = "running"
        result["timing_attempts"] = []
        functions = dict(score=adapter.score_only, topk=lambda: adapter.run(full_source=full_scores),
                         serial=adapter, production_control=control)
        samples = {name: [] for name in functions}
        result["samples_ms"] = samples
        for iteration in range(2):
            names = tuple(functions) if iteration == 0 else tuple(reversed(functions))
            for name in names:
                for attempt in range(2):
                    value = float(timer.get_latency(functions[name]))
                    valid = math.isfinite(value) and value > 0
                    result["timing_attempts"].append(dict(stage=name, round=iteration,
                        attempt=attempt, valid=valid, value_ms=value if valid else None,
                        invalid_value=None if valid else str(value)))
                    if valid:
                        samples[name].append(value)
                        break
                    print(f"QSA CANN SERIAL INVALID TIME {name}: {value}; attempt={attempt + 1}", flush=True)
                else:
                    result["timing_status"] = "invalid"
                    raise RuntimeError(f"Harness returned invalid latency twice: {case} {name}")
        medians = {name: statistics.median(times) for name, times in samples.items()}
        result.update(status="passed", timing_status="passed", samples_ms=samples, medians_ms=medians,
                      control_over_serial=medians["production_control"] / medians["serial"],
                      spread_pct={name: (max(times)-min(times))/statistics.median(times)*100 for name,times in samples.items()})
        print("QSA CANN SERIAL us " + json.dumps({name: time*1000 for name,time in medians.items()}), flush=True)
    except Exception as exc:
        result["error"] = repr(exc)
        if result.get("timing_status") == "running":
            result["timing_status"] = "failed"
        raise
    finally:
        for name, compiled in adapter.compiled.items():
            for kind, text in getattr(compiled, "asm", {}).items():
                if isinstance(text, str):
                    (Path(report) / f"{case}-cann-serial-{name}-{kind}.txt").write_text(text)
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
