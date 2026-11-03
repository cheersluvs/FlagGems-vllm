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

"""Explicit score-only Chrome trace timer with strict launch-count checks."""

import gzip
import json
import math
from decimal import Decimal
from pathlib import Path


def active_mean(events, expected, active):
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
        and isinstance(e.get("dur"), (float, int))
        and e["dur"] > 0
    ]
    host = [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") == "cpu_op"
        and e.get("name") == "_quant_score_trial"
    ]
    if len(tasks) != expected * active or len(host) != expected * active:
        raise RuntimeError(
            f"Incomplete score trace: device={len(tasks)}, host={len(host)}, expected={expected*active}"
        )
    if any(
        e.get("name") != "MIX_AIC" and "_quant_score_trial" not in e.get("name", "")
        for e in tasks
    ):
        raise RuntimeError("Unexpected device task in score-only capture")
    tasks.sort(key=lambda e: Decimal(str(e["ts"])))
    samples = [
        sum(e["dur"] for e in tasks[i * expected : (i + 1) * expected])
        for i in range(active)
    ]
    value = sum(samples) / active
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"Invalid trace duration: {value}")
    return dict(
        kernel_us=value,
        active_invocation_us=samples,
        device_task_count=len(tasks),
        host_launch_count=len(host),
    )


def setup_trace(args, context, torch, result):
    import torch_npu

    api = torch_npu.profiler
    sequence = 0
    result["kernel_timer"] = dict(
        backend="explicit CPU+NPU Chrome trace",
        warmup=args.kernel_warmup,
        active=args.kernel_active,
        unit="microseconds; sum of score device tasks per invocation",
        host_gaps_included=False,
        scope="Instrumented exploratory kernel timing; rebuild all paired baselines",
        fallback=False,
        raw_traces_preserved=True,
    )

    def measure(function, expected):
        nonlocal sequence
        sequence += 1
        for _ in range(args.kernel_warmup):
            function()
        torch.npu.synchronize()
        path = Path(args.report) / f"{args.case}-timer-{sequence:02d}.json"
        with api.profile(
            activities=[api.ProfilerActivity.CPU, api.ProfilerActivity.NPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for _ in range(args.kernel_active):
                function()
            torch.npu.synchronize()
        prof.export_chrome_trace(str(path))
        payload = path.read_bytes()
        with gzip.open(path.with_suffix(".json.gz"), "wb") as stream:
            stream.write(payload)
        path.unlink()
        data = json.loads(payload)
        parsed = active_mean(
            data.get("traceEvents", []) if isinstance(data, dict) else data,
            expected,
            args.kernel_active,
        )
        parsed["artifact"] = path.with_suffix(".json.gz").name
        print("TRACE_TIMER", json.dumps(parsed), flush=True)
        return parsed

    first, count = context["chunks"][0]

    def one():
        context["score_into"](first, count, context["logits"][:count])

    def triple():
        for _ in range(3):
            one()

    single = measure(one, 1)
    repeated = measure(triple, 3)
    ratio = repeated["kernel_us"] / single["kernel_us"]
    result["kernel_timer"]["invocation_sum_check"] = dict(
        single=single, triple=repeated, ratio=ratio
    )
    if not 2.25 <= ratio <= 3.75:
        raise RuntimeError(f"Trace invocation-sum calibration failed: {ratio}")
    return lambda function: measure(function, len(context["chunks"]))
