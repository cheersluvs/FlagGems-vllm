# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pair the current and one-copy GroupVisible request paths on 910B."""

import collections
import hashlib
import json
import statistics
import time
from pathlib import Path

from probe_qsa_int8_page64_prefix_910b import load_generated_prefix
from probe_qsa_int8_score_910b import reference
from probe_qsa_int8_score_sol_910b import build_group_visible_source
from probe_qsa_int8_ten_shapes_910b import verify_structure
from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan

from tools.qsa_int8_ten_workloads_910b import sample_rows


def run_group_visible_request(args, result, inputs, values, plan, topk):
    import torch
    from qsa_int8_shared_map_selection_910b_trial import PreparedSharedMapInt8Selection

    report = Path(args.report)
    rows, width, chunk = plan["q_shape"][0], plan["logits_width"], plan["chunk_rows"]
    device = values["q"].device
    score = torch.empty((chunk, width), dtype=torch.float32, device=device)
    alternate = torch.empty_like(score)
    output = torch.empty((rows, 512), dtype=torch.int32, device=device)
    workspace = torch.empty(1024**2, dtype=torch.uint8, device=device)

    kernels = {}
    for bounded in (True, False):
        arm = "group_visible_bounded" if bounded else "group_visible_full"
        kernels[bounded] = load_generated_prefix(
            report,
            args.case,
            arm,
            build_group_visible_source(bounded, arm, request_offset=True),
        )

    def prepare(name):
        return PreparedSharedMapInt8Selection(
            values,
            output,
            score,
            workspace,
            topk,
            plan["starts"],
            key_group4=True,
            single_page64=True,
            group_visible_kernels=kernels if name == "group_visible" else None,
        )

    names = ("current", "group_visible")
    adapters = {name: prepare(name) for name in names}
    current, candidate = (adapters[name] for name in names)
    if current.route_checks != candidate.route_checks:
        raise AssertionError("Candidate changed score dispatch")
    expected_hits = {
        (r["first"], r["count"])
        for r in current.route_checks
        if r["page_reuse"] and r["query_group"] == 16 and r["count"] % 16 == 0
    }
    if set(candidate.group_visible_offsets) != expected_hits:
        raise AssertionError("Unexpected GroupVisible routing")
    if args.case in ("int8_budget_exact", "int8_long100k") and not expected_hits:
        raise AssertionError("Expected candidate hits in the profiled fixtures")
    if args.case == "int8_first_chunk" and expected_hits:
        raise AssertionError("first_chunk must exercise the zero-hit control")
    metadata = candidate.group_visible_metadata
    if (metadata is not None) != bool(expected_hits):
        raise AssertionError("GroupVisible allocation does not match candidate hits")
    packed = metadata.cpu() if metadata is not None else None
    expected_offset = 0
    for (first, count), offset in candidate.group_visible_offsets.items():
        if offset != expected_offset:
            raise AssertionError("Invalid GroupVisible offset")
        expected_offset += count // 16
        torch.testing.assert_close(
            packed[offset:expected_offset],
            inputs["visible"][first : first + count].reshape(-1, 16).amax(1),
            rtol=0,
            atol=0,
        )
    if (packed is not None and packed.numel() != expected_offset) or (
        candidate.group_visible_stats["groups_from_policy"] != expected_offset
    ):
        raise AssertionError(
            "GroupVisible must reuse the policy maxima for every group"
        )

    references = {row: reference(inputs, row)[1] for row in sample_rows(plan)}
    host_names = {name: collections.Counter() for name in names}
    binaries = {name: {} for name in names}
    result["stage"] = "group_visible_full_score_check"
    for first, count in current.chunks:
        a = current.score_into(first, count, score[:count])
        b = candidate.score_into(first, count, alternate[:count])
        for name, compiled in zip(names, (a, b), strict=True):
            host_names[name][compiled.metadata.kernel_name] += 1
            if id(compiled) not in binaries[name]:
                binaries[name][id(compiled)] = (
                    compiled.metadata.kernel_name,
                    hashlib.sha256(bytes(compiled.asm["npubin"])).hexdigest(),
                )
        if not torch.equal(score[:count], alternate[:count]):
            raise AssertionError(f"FP32 score differs at chunk starting {first}")
        for row, expected in references.items():
            if first <= row < first + count:
                actual = score[row - first].cpu()
                n = expected.numel()
                torch.testing.assert_close(actual[:n], expected, rtol=1e-3, atol=1e-3)
                if not bool(torch.isneginf(actual[n:]).all()):
                    raise AssertionError(f"Incorrect visibility mask at row {row}")

    result["stage"] = "group_visible_selection_check"
    expected_indices = current().cpu()
    verify_structure(expected_indices, inputs["visible"])

    def check_indices(actual):
        if not torch.equal(actual.cpu(), expected_indices):
            raise AssertionError("Full request TopK indices differ")

    check_indices(candidate())
    result["correctness"] = dict(
        full_scores_bitwise_equal=True,
        all_topk_indices_bitwise_equal=True,
        cpu_reference_rows=sorted(references),
        chunks=len(current.chunks),
    )
    result["group_visible_prepared"] = candidate.group_visible_stats
    result["score_binaries"] = {
        name: sorted(set(entries.values())) for name, entries in binaries.items()
    }
    result["topk_identity"] = topk.probe_identity

    result["stage"] = "group_visible_score_trace"
    result["kernel_timer"] = {}
    samples, _ = measure_prefix_plan(
        args,
        names,
        lambda name, _: adapters[name].score_all(),
        dict.fromkeys(names),
        torch,
        result,
        expected_hosts=host_names,
    )
    score_medians = {
        name: statistics.median(
            x for round_ in rounds for x in round_["active_invocation_us"]
        )
        for name, rounds in samples.items()
    }
    score_pairs = [
        a - b
        for control, variant in zip(
            samples["current"], samples["group_visible"], strict=True
        )
        for a, b in zip(
            control["active_invocation_us"],
            variant["active_invocation_us"],
            strict=True,
        )
    ]
    result["score_device_us"] = dict(
        medians=score_medians,
        samples=samples,
        paired_saved_us=score_pairs,
        speedup=score_medians["current"] / score_medians["group_visible"],
    )

    result["stage"] = "group_visible_request_wall"
    pairs = []
    for i in range(args.group_visible_request_samples):
        order = names if i % 2 == 0 else names[::-1]
        pair = {"order": order}
        for name in order:
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            prepared = prepare(name)
            actual = prepared()
            torch.npu.synchronize()
            end = time.perf_counter_ns()
            check_indices(actual)
            pair[name] = dict(
                wall_us=(end - start) / 1000,
                preparation_us=prepared.preparation_ms * 1000,
                group_visible=prepared.group_visible_stats,
            )
        pairs.append(pair)
    wall = {
        name: statistics.median(p[name]["wall_us"] for p in pairs) for name in names
    }
    result["request_wall_us"] = dict(
        medians=wall,
        samples=pairs,
        speedup=wall["current"] / wall["group_visible"],
        paired_saved_us=[
            p["current"]["wall_us"] - p["group_visible"]["wall_us"] for p in pairs
        ],
        median_group_visible_prepare_us=statistics.median(
            p["group_visible"]["group_visible"]["prepare_us"] for p in pairs
        ),
        median_group_visible_pack_us=statistics.median(
            p["group_visible"]["group_visible"]["pack_us"] for p in pairs
        ),
        median_group_visible_alloc_copy_us=statistics.median(
            p["group_visible"]["group_visible"]["alloc_copy_us"] for p in pairs
        ),
        scope="Fresh route/metadata preparation, one score+TopK request and final sync "
        "on ready inputs and buffers.",
    )
    result["stage"] = "group_visible_request_completed"
    print(
        "GROUP_VISIBLE_REQUEST",
        args.case,
        json.dumps(
            dict(
                preparation=result["group_visible_prepared"],
                score_device_us={
                    k: v for k, v in result["score_device_us"].items() if k != "samples"
                },
                request_wall_us={
                    k: v for k, v in result["request_wall_us"].items() if k != "samples"
                },
            )
        ),
        flush=True,
    )
