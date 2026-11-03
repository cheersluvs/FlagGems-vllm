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

"""Measure selection stages in synchronized CPU-marked trace windows."""

import collections
import gc
import gzip
import json
from decimal import Decimal
from pathlib import Path

from qsa_int8_trace_timer_910b import IncompleteScoreTrace, setup_trace


def stage_mean(events, active, require_uniform_tasks=True):
    """Sum device durations inside each complete invocation window."""
    pids = {
        e.get("pid")
        for e in events
        if e.get("ph") == "M"
        and e.get("name") == "process_name"
        and e.get("args", {}).get("name") == "Ascend Hardware"
    }
    tasks = [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("pid") in pids
        and isinstance(e.get("dur"), (int, float))
        and e["dur"] > 0
    ]
    windows = [
        e
        for e in events
        if e.get("ph") == "X" and str(e.get("name", "")).startswith("qsa/timed/")
    ]
    windows.sort(key=lambda e: Decimal(str(e["ts"])))
    if len(windows) != active or not tasks:
        raise IncompleteScoreTrace(
            f"Incomplete stage trace: windows={len(windows)}, tasks={len(tasks)}"
        )
    counts, samples, names = [], [], []
    covered = 0
    previous_end = None
    for index, window in enumerate(windows):
        start = Decimal(str(window["ts"]))
        end = start + Decimal(str(window["dur"]))
        if window["name"] != f"qsa/timed/{index}" or (
            previous_end is not None and start < previous_end
        ):
            raise RuntimeError("Overlapping or unexpected invocation windows")
        previous_end = end
        selected = [
            e
            for e in tasks
            if Decimal(str(e["ts"])) >= start
            and Decimal(str(e["ts"])) + Decimal(str(e["dur"])) <= end
        ]
        if not selected:
            raise IncompleteScoreTrace(f"Empty device invocation window: {index}")
        covered += len(selected)
        counts.append(len(selected))
        samples.append(sum(e["dur"] for e in selected))
        names.append(
            dict(collections.Counter(e.get("name", "unknown") for e in selected))
        )
    if covered != len(tasks) or (
        require_uniform_tasks and any(n != names[0] for n in names)
    ):
        raise IncompleteScoreTrace(
            f"Incomplete stage task coverage: covered={covered}, tasks={len(tasks)}, counts={counts}"
        )
    return dict(
        kernel_us=sum(samples) / active,
        active_invocation_us=samples,
        device_task_count=len(tasks),
        tasks_per_invocation=counts,
        device_task_names_per_invocation=(names[0] if require_uniform_tasks else names),
    )


def setup_stage_trace(args, context, torch, result):
    """Calibrate scores, then time arbitrary stages without native splitting."""
    import torch_npu

    setup_trace(args, context, torch, result)
    result["kernel_timer"][
        "backend"
    ] = "CPU+NPU Chrome trace with synchronized invocation windows"
    result["kernel_timer"][
        "scope"
    ] = "Exploratory device-task sums; all stage and native baselines rebuilt together; no host gaps"
    sequence = 0
    api = torch_npu.profiler

    def capture(function):
        nonlocal sequence
        sequence += 1
        for _ in range(args.kernel_warmup):
            function()
        torch.npu.synchronize()
        path = Path(args.report) / f"{args.case}-stage-timer-{sequence:02d}.json"
        with api.profile(
            activities=[api.ProfilerActivity.CPU, api.ProfilerActivity.NPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for index in range(args.kernel_active):
                with torch.profiler.record_function(f"qsa/timed/{index}"):
                    returned = function()
                    torch.npu.synchronize()
            del returned
        prof.export_chrome_trace(str(path))
        del prof
        gc.collect()
        payload = path.read_bytes()
        archive = path.with_suffix(".json.gz")
        with gzip.open(archive, "wb") as stream:
            stream.write(payload)
        path.unlink()
        data = json.loads(payload)
        value = stage_mean(
            data.get("traceEvents", []) if isinstance(data, dict) else data,
            args.kernel_active,
        )
        dispatch = result["dispatch"]
        minimum = {
            "score_stage": dispatch["score_calls"],
            "topk_stage": dispatch["topk_calls"],
            "combined_stage": dispatch["score_calls"] + dispatch["topk_calls"],
            "native_stage": 1,
        }[function.__name__]
        if any(count < minimum for count in value["tasks_per_invocation"]):
            raise IncompleteScoreTrace(
                f"Too few device tasks for {function.__name__}: {value['tasks_per_invocation']}"
            )
        if function.__name__ == "score_stage" and any(
            count != minimum for count in value["tasks_per_invocation"]
        ):
            raise IncompleteScoreTrace("Unexpected score-only task count")
        value["artifact"] = archive.name
        value["stage"] = function.__name__
        return value

    def measure(function):
        for attempt in range(1, 4):
            try:
                return capture(function)
            except IncompleteScoreTrace as exc:
                failure = dict(
                    attempt=attempt,
                    artifact=f"{args.case}-stage-timer-{sequence:02d}.json.gz",
                    reason=str(exc),
                )
                result["kernel_timer"].setdefault("rejected_stage_captures", []).append(
                    failure
                )
                print("STAGE_TRACE_REJECTED", json.dumps(failure), flush=True)
                if attempt == 3:
                    raise

    return measure
