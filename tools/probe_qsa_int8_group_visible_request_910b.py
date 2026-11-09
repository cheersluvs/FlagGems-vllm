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
import gzip
import hashlib
import json
import statistics
import time
from itertools import permutations
from pathlib import Path

from probe_qsa_int8_page64_prefix_910b import load_generated_prefix
from probe_qsa_int8_score_910b import reference
from probe_qsa_int8_score_sol_910b import (
    build_direct_page_source,
    build_group_visible_source,
)
from probe_qsa_int8_ten_shapes_910b import verify_structure
from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan
from qsa_int8_stage_trace_timer_910b import stage_windows
from qsa_timeline_910b import summarize as summarize_timeline

from tools.qsa_int8_ten_workloads_910b import sample_rows


def combined_windows(events, plan, score_hosts, topk_hosts):
    metadata = [e for e in events if e.get("ph") == "M"]
    for job, (window, tasks) in zip(
        plan, stage_windows(events, len(plan)), strict=True
    ):
        expected_score = score_hosts[job["arm"]]
        score, topk = [], []
        for task in tasks:
            name = task["name"]
            if name == "MIX_AIC" or name in expected_score:
                score.append(task["dur"])
            elif name in topk_hosts:
                topk.append(task)
            else:
                raise RuntimeError(f"Unexpected combined device task: {name}")
        counts = collections.Counter(task["name"] for task in topk)
        if len(score) != sum(expected_score.values()) or counts != topk_hosts:
            raise RuntimeError(
                f"Unexpected tasks in {window['name']}: score={len(score)}, TopK={dict(counts)}"
            )
        timeline = summarize_timeline(metadata + tasks)
        yield dict(
            **job,
            score_us=sum(score),
            topk_us=sum(task["dur"] for task in topk),
            device_union_us=timeline["device_task_union_us"],
            device_span_us=timeline["device_task_envelope_us"],
            device_gap_us=timeline["uncovered_envelope_us"],
            profiled_window_us=window["dur"],
        )


def measure_combined_trace(
    args,
    torch,
    result,
    adapters,
    score_hosts,
    check_indices,
    *,
    label="group_visible",
    launch_gap_summary=None,
):
    names = tuple(adapters)
    topk_calls = sum((count + 31) // 32 for _, count in adapters["current"].chunks)
    # long100k's pinned 910B TopK route, verified against host and device events.
    topk_hosts = collections.Counter(
        dict.fromkeys(
            (
                "_radix_hist_trivial_all",
                "_psegmerge_seg_sort",
                "_psegmerge_merge_level",
                "_psegmerge_merge_final",
            ),
            topk_calls,
        )
    )
    result["stage"] = f"{label}_combined_trace"
    result["kernel_timer"] = {}
    measure_prefix_plan(
        args,
        names,
        lambda name, _: adapters[name](),
        dict.fromkeys(names),
        torch,
        result,
        expected_hosts={name: score_hosts[name] + topk_hosts for name in names},
    )
    artifact = result["kernel_timer"]["prefix_plan_artifact"]
    with gzip.open(Path(args.report) / artifact, "rt") as stream:
        data = json.load(stream)
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    samples = list(
        combined_windows(events, result["single_capture_plan"], score_hosts, topk_hosts)
    )
    by_arm = {name: [s for s in samples if s["arm"] == name] for name in names}
    pairs = list(zip(*(by_arm[name] for name in names), strict=True))
    fields = (
        "score_us",
        "topk_us",
        "device_union_us",
        "device_span_us",
        "device_gap_us",
        "profiled_window_us",
    )
    for adapter in adapters.values():
        check_indices(adapter())
    result["combined_trace"] = dict(
        artifact=artifact,
        scope="Prepared score+TopK request. Gaps are uncovered device intervals inside the first-to-last task span.",
        topk_tasks_per_window=dict(topk_hosts),
        medians_us={
            name: {f: statistics.median(s[f] for s in by_arm[name]) for f in fields}
            for name in names
        },
        paired_saved_us={f: [a[f] - b[f] for a, b in pairs] for f in fields},
        samples=samples,
    )
    if launch_gap_summary is not None:
        result["combined_trace"]["launch_gaps"] = launch_gap_summary(
            events, result["single_capture_plan"], topk_calls
        )
    result["stage"] = f"{label}_combined_trace_completed"
    print(
        f"{label.upper()}_COMBINED_TRACE",
        json.dumps(
            {k: v for k, v in result["combined_trace"].items() if k != "samples"}
        ),
        flush=True,
    )


def measure_request_wall(
    names,
    get_adapter,
    synchronize,
    check_indices,
    sample_count,
    *,
    record_group_visible=True,
    record_direct_page=False,
    record_host_phases=False,
):
    """Measure host intervals around one ordinary asynchronous selection call."""
    pairs = []
    orders = tuple(permutations(names))
    launch_phases = {"score_into": "score_submit_us", "select_chunk": "topk_submit_us"}
    for i in range(sample_count):
        order = orders[i % len(orders)]
        pair = {"order": order}
        for name in order:
            synchronize()
            start = time.perf_counter_ns()
            prepared = get_adapter(name)
            ready = time.perf_counter_ns()
            if record_host_phases:
                launch_ns = dict.fromkeys(launch_phases.values(), 0)

                def timed(method, phase):
                    def run(*args, **kwargs):
                        began = time.perf_counter_ns()
                        try:
                            return method(*args, **kwargs)
                        finally:
                            launch_ns[phase] += time.perf_counter_ns() - began

                    return run

                for method, phase in launch_phases.items():
                    setattr(prepared, method, timed(getattr(prepared, method), phase))
            submit_start = time.perf_counter_ns() if record_host_phases else ready
            try:
                actual = prepared()
                submitted = time.perf_counter_ns()
            finally:
                if record_host_phases:
                    for method in launch_phases:
                        delattr(prepared, method)
            synchronize()
            end = time.perf_counter_ns()
            check_indices(actual)
            pair[name] = dict(
                wall_us=(end - start) / 1000,
                preparation_us=(ready - start) / 1000,
                submit_us=(submitted - submit_start) / 1000,
                sync_us=(end - submitted) / 1000,
            )
            if record_group_visible:
                pair[name]["group_visible"] = prepared.group_visible_stats
            if record_direct_page:
                pair[name]["direct_page_guard_us"] = prepared.direct_page_guard_us
                pair[name]["page_proof_us"] = prepared.page_proof_us
            if record_host_phases:
                pair[name].update(
                    prepared.metadata_host_us,
                    **{phase: ns / 1000 for phase, ns in launch_ns.items()},
                    query16_page_proof_us=prepared.page_proof_us,
                    group_metadata_pack_us=prepared.group_visible_stats["pack_us"],
                    group_metadata_upload_us=prepared.group_visible_stats[
                        "alloc_copy_us"
                    ],
                    other_submit_us=(submitted - submit_start - sum(launch_ns.values()))
                    / 1000,
                )
            del prepared
        pairs.append(pair)

    phases = ("preparation_us", "submit_us", "sync_us", "wall_us")
    if record_host_phases:
        phases += (
            "row_table_to_host_us",
            "request_table_expand_us",
            "shared_map_check_us",
            "visible_to_host_us",
            "query16_page_proof_us",
            "group_metadata_pack_us",
            "group_metadata_upload_us",
            *launch_phases.values(),
            "other_submit_us",
        )
    medians = {
        phase: {
            name: statistics.median(p[name][phase] for p in pairs) for name in names
        }
        for phase in phases
    }
    current, candidate = names[0], names[-1]
    saved = {
        phase: [p[current][phase] - p[candidate][phase] for p in pairs]
        for phase in phases
    }
    wall = medians["wall_us"]
    return dict(
        medians=wall,
        phase_medians_us=medians,
        phase_paired_median_saved_us={
            phase: statistics.median(samples) for phase, samples in saved.items()
        },
        samples=pairs,
        speedup=wall[current] / wall[candidate],
        paired_saved_us=saved["wall_us"],
        candidate_wins=sum(delta > 0 for delta in saved["wall_us"]),
    )


def check_group_visible_routes(args, torch, inputs, current, candidate):
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
    return expected_hits


def check_direct_page_edges(torch, values, prepare, width, *, cached):
    """Exercise padding and query-tail dispatch outside the timed fixtures."""
    begin, rows, chunk = 8192, 193, 64
    padded = {
        key: value[begin : begin + rows] if key not in ("cache", "ks") else value
        for key, value in values.items()
    }
    table = padded["row_table"].clone()
    visible = padded["visible"].clone()
    table[:chunk, -1] = -1
    table[chunk : 2 * chunk, -1] = padded["cache"].shape[0]
    # The full-dot chunk still visits the last page, but no live key is there.
    visible[chunk : 3 * chunk] = width - 512
    padded.update(row_table=table, visible=visible)
    device = padded["q"].device
    scratch = torch.empty((chunk, width), dtype=torch.float32, device=device)
    alternate = torch.empty_like(scratch)
    output = torch.empty((rows, 512), dtype=torch.int32, device=device)
    kwargs = dict(
        values=padded, output=output, scratch=scratch, request_starts=[0, rows]
    )
    current = prepare("current", **kwargs)
    candidate = prepare("direct_page", **kwargs)
    expected = {
        (0, chunk): "bounded",
        (chunk, chunk): "invalid_page_id",
        (2 * chunk, chunk): "full",
        (3 * chunk, 1): "partial_query_group",
    }
    if candidate.direct_page_routes != expected:
        raise AssertionError(f"Unexpected edge routes: {candidate.direct_page_routes}")
    if cached:
        scanned = prepare("scanned_direct_page", **kwargs)
        if (
            scanned.direct_page_routes != expected
            or candidate.direct_page_unique_mappings != 3
        ):
            raise AssertionError("Cached guard must distinguish the three mappings")
    for first, count in current.chunks:
        a = current.score_into(first, count, scratch[:count])
        b = candidate.score_into(first, count, alternate[:count])
        kernel = b.metadata.kernel_name
        mode = expected[first, count]
        expected_kernel = (
            f"_qsa_page64_prefix_direct_page_{mode}"
            if mode in ("bounded", "full")
            else a.metadata.kernel_name
        )
        if kernel != expected_kernel:
            raise AssertionError(
                f"Unexpected padded/tail kernel at row {first}: {kernel}"
            )
        if not torch.equal(scratch[:count], alternate[:count]):
            raise AssertionError(f"Padded/tail FP32 score differs at row {first}")
    indices = current().cpu()
    verify_structure(indices, visible.cpu())
    if not torch.equal(indices, candidate().cpu()):
        raise AssertionError("Padded/tail TopK indices differ")
    return [
        dict(first=first, count=count, route=route)
        for (first, count), route in expected.items()
    ]


def run_score_request_probe(args, result, inputs, values, plan, topk):
    import torch
    from qsa_int8_shared_map_selection_910b_trial import PreparedSharedMapInt8Selection

    direct_page = bool(args.direct_page_request_samples)
    cached = args.cache_direct_page_guard
    shared_proof = args.shared_direct_page_proof
    guard_control = (
        "tuple_cached_direct_page" if shared_proof else "scanned_direct_page"
    )
    variant = "direct_page" if direct_page else "group_visible"
    sample_count = (
        args.direct_page_request_samples
        if direct_page
        else args.group_visible_request_samples
    )
    report = Path(args.report)
    rows, width, chunk = plan["q_shape"][0], plan["logits_width"], plan["chunk_rows"]
    device = values["q"].device
    score = torch.empty((chunk, width), dtype=torch.float32, device=device)
    alternate = torch.empty_like(score)
    output = torch.empty((rows, 512), dtype=torch.int32, device=device)
    workspace = torch.empty(1024**2, dtype=torch.uint8, device=device)

    kernels = {}
    for bounded in (True, False):
        arm = f"{variant}_{'bounded' if bounded else 'full'}"
        source = (
            build_direct_page_source(bounded, arm, precomputed_visibility=False)
            if direct_page
            else build_group_visible_source(bounded, arm, request_offset=True)
        )
        kernels[bounded] = load_generated_prefix(report, args.case, arm, source)

    adapter_args = dict(
        values=values,
        output=output,
        scratch=score,
        workspace=workspace,
        topk=topk,
        request_starts=plan["starts"],
        key_group4=True,
        single_page64=True,
    )

    def prepare(name, **overrides):
        route_kernels = (
            {f"{variant}_kernels": kernels}
            if name in (variant, guard_control, "scanned_direct_page")
            else {}
        )
        return PreparedSharedMapInt8Selection(
            **(adapter_args | overrides),
            **route_kernels,
            cache_direct_page_guard=(
                (cached and not shared_proof and name == variant)
                or name == "tuple_cached_direct_page"
            ),
            shared_direct_page_proof=shared_proof and name == variant,
            time_page_proof=shared_proof,
        )

    names = ("current", variant)
    adapters = {name: prepare(name) for name in names}
    current, candidate = (adapters[name] for name in names)
    if current.route_checks != candidate.route_checks:
        raise AssertionError("Candidate changed base score dispatch")
    if direct_page:
        result["stage"] = "direct_page_route_check"
        if candidate.group_visible_metadata is not None:
            raise AssertionError("Direct page probe must not allocate GroupVisible")
        routes = candidate.direct_page_routes
        if routes.keys() != set(current.chunks):
            raise AssertionError("Missing direct-page route decision")
        expected_hits = {
            key for key, mode in routes.items() if mode in ("bounded", "full")
        }
        if args.case in ("int8_budget_exact", "int8_long100k") and not expected_hits:
            raise AssertionError("Expected direct-page hits in profiled fixtures")
        if args.case == "int8_first_chunk" and expected_hits:
            raise AssertionError("first_chunk must exercise the zero-hit control")
        if cached:
            control = prepare(guard_control)
            if (
                control.route_checks != candidate.route_checks
                or control.direct_page_routes != routes
            ):
                raise AssertionError("Page proof changed chunk dispatch")
            unique_mappings = candidate.direct_page_unique_mappings
            if (expected_hits and not unique_mappings) or (
                args.case == "int8_first_chunk" and unique_mappings
            ):
                raise AssertionError(
                    "Unexpected cache activity in the focused fixtures"
                )
            del control
        result["direct_page"] = dict(
            hit_chunks=len(expected_hits),
            counts=dict(collections.Counter(routes.values())),
            routes=[
                dict(first=first, count=count, route=route)
                for (first, count), route in routes.items()
            ],
        )
        if args.case == "int8_budget_exact":
            result["direct_page"]["edge_routes_checked"] = check_direct_page_edges(
                torch, values, prepare, width, cached=cached
            )
        if cached:
            if shared_proof:
                result["direct_page"]["shared_proof"] = dict(
                    routes_equal_to_tuple_cache=True, unique_mappings=unique_mappings
                )
            else:
                result["direct_page"]["cached_guard"] = dict(
                    routes_equal_to_scan=True, unique_mappings=unique_mappings
                )
    else:
        expected_hits = check_group_visible_routes(
            args, torch, inputs, current, candidate
        )

    references = {row: reference(inputs, row)[1] for row in sample_rows(plan)}
    host_names = {name: collections.Counter() for name in names}
    binaries = {name: {} for name in names}
    result["stage"] = f"{variant}_full_score_check"
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
        if direct_page:
            key = (first, count)
            actual = binaries[variant][id(b)]
            if key in expected_hits:
                arm = f"_qsa_page64_prefix_direct_page_{routes[key]}"
                if actual[0] != arm:
                    raise AssertionError(
                        f"Wrong direct-page kernel at row {first}: {actual[0]}"
                    )
            elif actual != binaries["current"][id(a)]:
                raise AssertionError(
                    f"Fallback compiled different score kernel at row {first}"
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

    result["stage"] = f"{variant}_selection_check"
    expected_indices = current().cpu()
    verify_structure(expected_indices, inputs["visible"])

    def check_indices(actual):
        if not torch.equal(actual.cpu(), expected_indices):
            raise AssertionError("Full request TopK indices differ")

    check_indices(candidate())
    result["correctness"] = dict(
        all_topk_indices_bitwise_equal=True,
        cpu_reference_rows=sorted(references),
        chunks=len(current.chunks),
    )
    if direct_page:
        result["correctness"]["full_scores_equal"] = True
    else:
        result["correctness"]["full_scores_bitwise_equal"] = True
        result["group_visible_prepared"] = candidate.group_visible_stats
    result["score_binaries"] = {
        name: sorted(set(entries.values())) for name, entries in binaries.items()
    }
    if not expected_hits and (
        result["score_binaries"]["current"] != result["score_binaries"][variant]
    ):
        raise AssertionError("Zero-hit control compiled different score binaries")
    result["topk_identity"] = topk.probe_identity

    if args.group_visible_combined_trace:
        measure_combined_trace(args, torch, result, adapters, host_names, check_indices)
        return

    result["stage"] = f"{variant}_score_trace"
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
        for control, changed in zip(samples["current"], samples[variant], strict=True)
        for a, b in zip(
            control["active_invocation_us"],
            changed["active_invocation_us"],
            strict=True,
        )
    ]
    result["score_device_us"] = dict(
        medians=score_medians,
        samples=samples,
        paired_saved_us=score_pairs,
        speedup=score_medians["current"] / score_medians[variant],
    )

    result["stage"] = f"{variant}_prepared_request_wall"
    prepared_wall = measure_request_wall(
        names,
        adapters.__getitem__,
        torch.npu.synchronize,
        check_indices,
        sample_count,
        record_group_visible=not direct_page,
    )
    prepared_wall["scope"] = (
        "Reuse an adapter on ready inputs, run score+TopK, then sync."
    )
    result["prepared_request_wall_us"] = prepared_wall

    result["stage"] = f"{variant}_request_wall"
    fresh_wall = measure_request_wall(
        ("current", guard_control, variant) if cached else names,
        prepare,
        torch.npu.synchronize,
        check_indices,
        sample_count,
        record_group_visible=not direct_page,
        record_direct_page=direct_page,
    )
    pairs = fresh_wall["samples"]
    if direct_page:
        fresh_wall["median_direct_page_guard_us"] = statistics.median(
            p[variant]["direct_page_guard_us"] for p in pairs
        )
        if shared_proof:
            fresh_wall["page_proof_medians_us"] = {
                name: statistics.median(p[name]["page_proof_us"] for p in pairs)
                for name in ("current", guard_control, variant)
            }
            fresh_wall["shared_proof_comparisons"] = {}
            for control in ("current", guard_control):
                saved = {
                    phase: [p[control][phase] - p[variant][phase] for p in pairs]
                    for phase in ("page_proof_us", "preparation_us", "wall_us")
                }
                fresh_wall["shared_proof_comparisons"][control] = dict(
                    paired_median_saved_us={
                        phase: statistics.median(samples)
                        for phase, samples in saved.items()
                    },
                    wall_wins=sum(delta > 0 for delta in saved["wall_us"]),
                )
        elif cached:
            saved = {
                phase: [
                    p["scanned_direct_page"][phase] - p[variant][phase] for p in pairs
                ]
                for phase in ("direct_page_guard_us", "preparation_us", "wall_us")
            }
            fresh_wall["cached_vs_scanned"] = dict(
                scanned_guard_us=statistics.median(
                    p["scanned_direct_page"]["direct_page_guard_us"] for p in pairs
                ),
                paired_median_saved_us={
                    phase: statistics.median(samples)
                    for phase, samples in saved.items()
                },
                wall_wins=sum(delta > 0 for delta in saved["wall_us"]),
            )
    else:
        fresh_wall.update(
            {
                f"median_group_visible_{phase}": statistics.median(
                    p[variant]["group_visible"][phase] for p in pairs
                )
                for phase in ("prepare_us", "pack_us", "alloc_copy_us")
            },
        )
    fresh_wall["scope"] = (
        "Fresh route preparation, one score+TopK request and final sync "
        "on ready inputs and buffers."
    )
    result["request_wall_us"] = fresh_wall
    result["stage"] = f"{variant}_request_completed"
    print(
        f"{variant.upper()}_REQUEST",
        args.case,
        json.dumps(
            dict(
                preparation=result[
                    "direct_page" if direct_page else "group_visible_prepared"
                ],
                score_device_us={
                    k: v for k, v in result["score_device_us"].items() if k != "samples"
                },
                prepared_request_wall_us={
                    k: v for k, v in prepared_wall.items() if k != "samples"
                },
                request_wall_us={k: v for k, v in fresh_wall.items() if k != "samples"},
            )
        ),
        flush=True,
    )
