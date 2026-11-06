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

"""Measure request-local reuse of the Ascend persistent TopK scratch."""

import ast
import collections
import copy
import os
import statistics
from bisect import bisect_left
from decimal import Decimal
from pathlib import Path
from types import FunctionType, SimpleNamespace

from probe_qsa_int8_group_visible_request_910b import (
    measure_combined_trace,
    measure_request_wall,
)
from probe_qsa_int8_ten_shapes_910b import verify_structure
from qsa_int8_stage_trace_timer_910b import stage_windows

HOST_HELPER = "_persistent_topk_segmerge_persistent"
BUFFER_NAMES = ("props", "nv_props", "buf_a", "buf_b", "nv_a", "nv_b", "scratch")
ARMS = ("current", "workspace_reuse")


def workspace_functions(topk):
    """Derive the allocator and launcher from the unchanged source helper."""
    source = Path(topk.probe_identity["source"])
    host = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == HOST_HELPER
    )
    allocations = {
        i: node.targets[0].id
        for i, node in enumerate(host.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "torch.empty"
    }
    if tuple(allocations.values()) != BUFFER_NAMES:
        raise RuntimeError("Persistent TopK scratch layout changed")
    names = ", ".join(BUFFER_NAMES)
    allocator = ast.parse("def allocate(logits, k, max_seq_len): pass").body[0]
    allocator.body = copy.deepcopy(host.body[: max(allocations) + 1])
    allocator.body += ast.parse(f"return {names}").body

    first = min(allocations)
    launcher = copy.deepcopy(host)
    launcher.body[first] = ast.copy_location(
        ast.parse(f"{names} = qsa_buffers[num_rows]").body[0], launcher.body[first]
    )
    launcher.body = [
        node
        for i, node in enumerate(launcher.body)
        if i == first or i not in allocations
    ]
    scope = dict(vars(topk))
    tree = ast.Module(body=[allocator, launcher], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), scope)
    return scope["allocate"], scope[HOST_HELPER].__code__


def launch_gap_summary(events, plan, topk_calls):
    """Attribute device gaps up to the next enqueue and audit eliminated allocations."""

    def timestamp(event):
        return Decimal(str(event["ts"]))

    enqueues = sorted(
        (e for e in events if e.get("cat") == "enqueue" and e.get("ph") == "X"),
        key=timestamp,
    )
    enqueue_times = [timestamp(e) for e in enqueues]
    allocations = sorted(
        timestamp(e)
        for e in events
        if e.get("cat") == "cpu_op"
        and e.get("ph") == "X"
        and e["name"] == "empty_tensor"
    )
    samples = []
    for job, (window, tasks) in zip(
        plan, stage_windows(events, len(plan)), strict=True
    ):
        previous_end = None
        trivial_end = None
        before_enqueue = before_sort_enqueue = Decimal(0)
        alloc_count = 0
        streams = {task["args"]["Physic Stream Id"] for task in tasks}
        if len(streams) != 1:
            raise RuntimeError("Expected one stream for request-local workspace reuse")
        start = timestamp(window)
        first = bisect_left(enqueue_times, start)
        end = bisect_left(enqueue_times, start + Decimal(str(window["dur"])))
        if end - first != len(tasks):
            raise RuntimeError("Host enqueue count differs from device task count")
        # Match FIFO order on one stream and validate names. MIX_AIC may not
        # have a HostToDevice end event or the same ID as its host enqueue.
        for task, enqueue in zip(
            sorted(tasks, key=timestamp), enqueues[first:end], strict=True
        ):
            name = enqueue["name"].removeprefix("Enqueue@")
            if not (
                name.startswith("_qsa_")
                if task["name"] == "MIX_AIC"
                else name == task["name"]
            ):
                raise RuntimeError(
                    f"Unexpected task/enqueue association: {task['name']} / {name}"
                )
            start = timestamp(task)
            queued = timestamp(enqueue)
            if queued > start:
                raise RuntimeError("Task timestamp precedes enqueue")
            if previous_end is not None:
                gap = max(Decimal(0), queued - previous_end)
                before_enqueue += gap
                if name == "_psegmerge_seg_sort":
                    before_sort_enqueue += gap
            end = start + Decimal(str(task["dur"]))
            previous_end = end if previous_end is None else max(previous_end, end)
            if name == "_radix_hist_trivial_all":
                trivial_end = queued + Decimal(str(enqueue["dur"]))
            elif name == "_psegmerge_seg_sort":
                if trivial_end is None or queued < trivial_end:
                    raise RuntimeError(
                        "Expected a short-row launch before segment sort"
                    )
                alloc_count += bisect_left(allocations, queued) - bisect_left(
                    allocations, trivial_end
                )
                trivial_end = None
        expected = topk_calls * len(BUFFER_NAMES) if job["arm"] == "current" else 0
        if alloc_count != expected:
            raise RuntimeError(
                f"{job['arm']} scratch allocations: observed {alloc_count}, expected {expected}"
            )
        samples.append(
            dict(
                **job,
                before_enqueue_us=float(before_enqueue),
                before_sort_enqueue_us=float(before_sort_enqueue),
                allocations_between_trivial_and_sort=alloc_count,
            )
        )
    fields = (
        "before_enqueue_us",
        "before_sort_enqueue_us",
        "allocations_between_trivial_and_sort",
    )
    by_arm = {name: [s for s in samples if s["arm"] == name] for name in ARMS}
    return dict(
        medians={
            name: {f: statistics.median(s[f] for s in by_arm[name]) for f in fields}
            for name in ARMS
        },
        paired_saved_us={
            f: [
                a[f] - b[f]
                for a, b in zip(*(by_arm[name] for name in ARMS), strict=True)
            ]
            for f in fields[:2]
        },
        samples=samples,
    )


def run_topk_workspace_probe(args, result, inputs, values, plan, topk):
    import torch
    from qsa_int8_shared_map_selection_910b_trial import PreparedSharedMapInt8Selection

    device = values["q"].device
    rows, width, chunk = plan["q_shape"][0], plan["logits_width"], plan["chunk_rows"]
    score = torch.empty((chunk, width), dtype=torch.float32, device=device)
    output = torch.empty((rows, 512), dtype=torch.int32, device=device)
    workspace = torch.empty(1024**2, dtype=torch.uint8, device=device)
    current = PreparedSharedMapInt8Selection(
        values,
        output,
        score,
        workspace,
        topk,
        plan["starts"],
        key_group4=True,
        single_page64=True,
    )
    batch_counts = collections.Counter(
        min(32, count - offset)
        for _, count in current.chunks
        for offset in range(0, count, 32)
    )
    row_counts = sorted(
        count for count in batch_counts if topk._psegmerge_applicable(count, width, 512)
    )
    expected_reuse = plan.get("workspace_reuse_expected", args.case == "int8_long100k")
    if bool(row_counts) != expected_reuse:
        raise RuntimeError("Unexpected workspace route eligibility")

    result["stage"] = "topk_workspace_correctness"
    output.fill_(-2)
    expected = current().cpu()
    verify_structure(expected, inputs["visible"])
    if row_counts:
        if (
            os.environ.get(topk._SEGMERGE_ENV, "auto").lower() == "radix"
            or not topk._PSEGMERGE_STATE["usable"]
            or not topk._segmerge_probe()
        ):
            raise RuntimeError("Expected persistent TopK route is unavailable")
        allocate, launcher_code = workspace_functions(topk)

    def make_candidate(audit=None):
        if not row_counts:
            return current, {}
        buffers = {count: allocate(score[:count], 512, width) for count in row_counts}
        raw_launcher = FunctionType(
            launcher_code, dict(vars(topk), qsa_buffers=buffers)
        )

        def launch_checked(logits, *operands):
            audit[logits.shape[0]] += 1
            return raw_launcher(logits, *operands)

        launcher = raw_launcher if audit is None else launch_checked
        scope = dict(vars(topk), **{HOST_HELPER: launcher})
        reused_topk = FunctionType(
            topk.persistent_topk.__code__,
            scope,
            argdefs=topk.persistent_topk.__defaults__,
        )
        reused_topk.__kwdefaults__ = topk.persistent_topk.__kwdefaults__
        adapter = copy.copy(current)
        adapter.topk = SimpleNamespace(persistent_topk=reused_topk)
        return adapter, buffers

    def check_indices(actual):
        if not torch.equal(actual.cpu(), expected):
            raise AssertionError("Workspace reuse changed TopK indices")
        if row_counts and not topk._PSEGMERGE_STATE["usable"]:
            raise AssertionError("Persistent TopK route fell back")

    observed = collections.Counter()
    candidate, buffers = make_candidate(audit=observed)
    output.fill_(-2)
    check_indices(candidate())
    if observed != {count: batch_counts[count] for count in row_counts}:
        raise AssertionError(f"Unexpected TopK reuse launch counts: {dict(observed)}")
    result["correctness"] = dict(
        all_topk_indices_bitwise_equal=True, score_unchanged=True
    )
    result["topk_identity"] = topk.probe_identity
    result["workspace"] = dict(
        allocation_names=BUFFER_NAMES,
        route="segmerge_persistent" if row_counts else "existing_fallback",
        calls_by_rows=dict(batch_counts),
        reused_calls_by_rows=dict(observed),
        shapes={
            count: [list(buf.shape) for buf in group]
            for count, group in buffers.items()
        },
        bytes=sum(
            buf.numel() * buf.element_size()
            for group in buffers.values()
            for buf in group
        ),
    )
    # The timed candidate allocates its own buffers once per request. Drop the preflight copy.
    del candidate, buffers

    result["stage"] = "topk_workspace_wall"
    result["request_wall_us"] = measure_request_wall(
        ARMS,
        lambda name: current if name == "current" else make_candidate()[0],
        torch.npu.synchronize,
        check_indices,
        args.topk_workspace_samples,
        record_group_visible=False,
    )
    print(
        "TOPK_WORKSPACE_WALL",
        args.case,
        result["workspace"],
        {k: v for k, v in result["request_wall_us"].items() if k != "samples"},
        flush=True,
    )
    if args.topk_workspace_wall_only:
        result["stage"] = "topk_workspace_wall_completed"
        return

    score_hosts = collections.Counter(
        current.score_into(first, count, score[:count]).metadata.kernel_name
        for first, count in current.chunks
    )
    candidate, _ = make_candidate()
    measure_combined_trace(
        args,
        torch,
        result,
        dict(zip(ARMS, (current, candidate), strict=True)),
        dict.fromkeys(ARMS, score_hosts),
        check_indices,
        label="topk_workspace",
        launch_gap_summary=launch_gap_summary,
    )
