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
import os
import statistics
from pathlib import Path


def run_probe(inputs, values, width, full_scores, control, timer, report, case, *, validate_only=False, query_group=None, mixed_structure=None):
    import torch
    from qsa_cann_serial_910b_trial import PreparedCANNSerial
    from probe_qsa_int8_score_910b import reference
    from probe_qsa_int8_pipeline_910b import check_selection
    from probe_qsa_int8_ten_shapes_910b import verify_structure

    result = dict(case=case, status="failed", mode="kernel", scope=f"{values['q'].shape[0]}-row serial structural probe",
                  outer_keys=2048, cube_n=64, state_capacity=2048, output_k=512,
                  score_dtype="FP32", state_storage="GM ping-pong, sequential kernel launches",
                  blocks=list(range(0, width, 2048)), checks=[])
    fast_score = os.environ.get("QSA_SERIAL_FAST_SCORE") == "1"
    owners = values.get("_request_ids", [0] * values["q"].shape[0])
    group = int(os.environ.get("QSA_SERIAL_QUERY_GROUP", "16")) if query_group is None else query_group
    if not fast_score:
        group = 1
    compare_groups = os.environ.get("QSA_SERIAL_COMPARE_GROUPS") == "1" and not validate_only
    if compare_groups and (not fast_score or group != 4):
        raise ValueError("Paired grouping probe requires shared N64 and query_group=4")
    adapter = PreparedCANNSerial(values, width, query_group=group, request_ids=owners)
    peer = None
    mixed_controls = {}
    active_group = group
    active_implementation = "separated"
    result["query_group"] = group
    result["score_variant"] = "shared_n64" if fast_score else "row_n64"
    result["score_segments"] = adapter.segments
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
        entry = dict(stage=stage, key_start=start, mismatches=bad, query_group=active_group, implementation=active_implementation)
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
        if full_scores is not None:
            adapter.run(full_source=full_scores)
            verify_structure(adapter.output.cpu(), inputs["visible"])
            checks, _ = check_selection(adapter.output.cpu(), refs)
            assert checks["checks_passed"], checks
        if compare_groups:
            peer = PreparedCANNSerial(values, width, query_group=16, request_ids=owners)
            active_group = 16
            peer.run(check=check)
            peer_indices = peer.output.cpu()
            verify_structure(peer_indices, inputs["visible"])
            checks, _ = check_selection(peer_indices, refs)
            assert checks["checks_passed"], checks
            peer_values = peer.output_values.cpu()
            for row, ref in enumerate(refs):
                valid = peer_indices[row] >= 0
                torch.testing.assert_close(peer_values[row, valid], ref[peer_indices[row, valid].long()], rtol=1e-3, atol=1e-3)
                assert torch.isneginf(peer_values[row, ~valid]).all()
            active_group = group
            result["compared_query_groups"] = [4, 16]
            result["peer_score_segments"] = peer.segments
        if os.environ.get("QSA_SERIAL_MIXED_CONTROL") == "1" and group == 4:
            from qsa_cann_mixed_910b_trial import PreparedMixedOuter
            if os.environ.get("QSA_MIXED_STRUCTURE_DIAG") == "1":
                if mixed_structure is None:
                    from diagnose_qsa_structural_mixed_compile_910b import compile_structure_matrix
                    matrix = compile_structure_matrix(values, width, owners, report, case)
                    result["mixed_structure_matrix"] = matrix
                    mixed_structure = matrix["selected"]
                result["mixed_structure"] = mixed_structure
            elif os.environ.get("QSA_MIXED_COMPILE_DIAG") == "1":
                from diagnose_qsa_structural_mixed_compile_910b import compile_matrix
                matrix = compile_matrix(values, width, owners, report, case)
                result["mixed_compile_matrix"] = matrix
                assert all(entry["status"] == "compiled" for entry in matrix), "Mixed compile matrix incomplete; see compiler input IR and logs"
            result["mixed_checks"] = []
            for slots, label in ((1, "mixed_serial"), (2, "mixed_double")):
                active_group = 4
                active_implementation = label
                mixed = PreparedMixedOuter(values, width, slots=slots, request_ids=owners, structure=mixed_structure)
                mixed_controls[label] = mixed
                print(f"QSA MIXED COMPILE {case} {label}", flush=True)
                mixed.run(check=check)
                # Validate the exact non-instrumented version used for timing.
                mixed()
                indices = mixed.output.cpu()
                verify_structure(indices, inputs["visible"])
                checks, _ = check_selection(indices, refs)
                assert checks["checks_passed"], (label, checks)
                actual_values = mixed.output_values.cpu()
                for row, ref in enumerate(refs):
                    valid = indices[row] >= 0
                    torch.testing.assert_close(actual_values[row, valid], ref[indices[row, valid].long()], rtol=1e-3, atol=1e-3)
                    assert torch.isneginf(actual_values[row, ~valid]).all()
                result["mixed_checks"].append(dict(arm=label, slots=slots, status="passed"))
                print(f"QSA MIXED PASS {case} {label}", flush=True)
            active_implementation = "separated"
        result["correctness_status"] = "passed"
        if validate_only:
            result.update(status="passed", timing_status="not_requested")
            return result
        if case == "int8_budget_exact":
            result["boundary_reports"] = run_boundaries(inputs, values, report, mixed_structure=mixed_structure)
        result["timing_status"] = "running"
        result["timing_attempts"] = []
        functions = dict(score=adapter.score_only, topk=lambda: adapter.run(full_source=full_scores),
                         serial=adapter, production_control=control)
        if compare_groups:
            functions.update(peer_score=peer.score_only, peer_serial=peer)
        elif fast_score:
            reference_adapter = PreparedCANNSerial(values, width)
            reference_adapter()
            checks, _ = check_selection(reference_adapter.output.cpu(), refs)
            assert checks["checks_passed"], checks
            functions["reference_serial"] = reference_adapter
        functions.update(mixed_controls)
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
        if mixed_controls:
            result["fusion_speedup"] = medians["serial"] / medians["mixed_serial"]
            result["double_buffer_speedup"] = medians["mixed_serial"] / medians["mixed_double"]
            print(f"QSA MIXED RATIOS {case}: fusion={result['fusion_speedup']:.4f}x; "
                  f"double_buffer={result['double_buffer_speedup']:.4f}x", flush=True)
        if compare_groups:
            result["query4_score_speedup"] = medians["peer_score"] / medians["score"]
            result["query4_serial_speedup"] = medians["peer_serial"] / medians["serial"]
            print(f"QSA QUERY4/16 {case}: score speedup={result['query4_score_speedup']:.4f}x; "
                  f"serial speedup={result['query4_serial_speedup']:.4f}x", flush=True)
        elif fast_score:
            result["score_replacement_serial_speedup"] = medians["reference_serial"] / medians["serial"]
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
        if peer is not None:
            for name, compiled in peer.compiled.items():
                for kind, text in getattr(compiled, "asm", {}).items():
                    if isinstance(text, str):
                        (Path(report) / f"{case}-cann-serial-query16-{name}-{kind}.txt").write_text(text)
        for label, mixed in mixed_controls.items():
            for name, compiled in mixed.compiled.items():
                for kind, text in getattr(compiled, "asm", {}).items():
                    if isinstance(text, str):
                        (Path(report) / f"{case}-{label}-{name}-{kind}.txt").write_text(text)
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def run_boundaries(inputs, values, report, *, mixed_structure=None):
    import torch

    specifications = (
        ("serial_edges_unequal17", 4097,
         [0, 1, 63, 64, 127, 128, 191, 192, 511, 512, 513, 2047, 2048, 2049, 4095, 4096, 4097],
         [3, 5, 9], False),
        ("serial_edges_ties3", 2049, [0, 512, 2049], [1, 2], True),
        ("serial_edges_invisible1", 1, [0], [1], False),
    )
    completed = []
    for name, width, visible, request_lengths, ties in specifications:
        count = len(visible)
        assert sum(request_lengths) == count
        edge = dict(inputs)
        for key in ("q", "qs", "weights"):
            edge[key] = inputs[key][:count].clone()
        if ties:
            edge["weights"].zero_()
        columns = (width + 191) // 192
        mapping = inputs["row_table"][0, :columns]
        tables = []
        request_ids = []
        for request, length in enumerate(request_lengths):
            tables.extend([torch.roll(mapping, request)] * length)
            request_ids.extend([request] * length)
        edge["row_table"] = torch.stack(tables).contiguous()
        edge["visible"] = torch.tensor(visible, dtype=inputs["visible"].dtype)
        target = dict(values)
        for key in ("q", "qs", "weights", "row_table", "visible"):
            target[key] = edge[key].to(values["q"].device)
        target["_request_ids"] = request_ids
        groups = (4, 16) if os.environ.get("QSA_SERIAL_COMPARE_GROUPS") == "1" else (None,)
        for group in groups:
            label = name if group is None else f"{name}_q{group}"
            entry = run_probe(edge, target, width, None, None, None, report, label,
                              validate_only=True, query_group=group, mixed_structure=mixed_structure)
            completed.append(dict(case=label, width=width, rows=count, request_lengths=request_lengths,
                                  query_group=entry["query_group"], tied_scores=ties, status=entry["status"]))
            print("QSA SERIAL BOUNDARY PASS " + json.dumps(completed[-1]), flush=True)
    return completed
