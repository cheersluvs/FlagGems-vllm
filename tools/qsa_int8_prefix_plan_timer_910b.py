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

"""Collect interleaved prefix samples in one synchronized trace."""

import gc
import gzip
import json
from pathlib import Path

from qsa_int8_stage_trace_timer_910b import stage_mean
from qsa_int8_trace_timer_910b import IncompleteScoreTrace


def measure_prefix_plan(args, arms, launch, targets, torch, result):
    """Keep every valid sample; retry only incomplete whole captures."""
    import torch_npu

    api = torch_npu.profiler
    plan = []
    orders = []
    for round_index in range(args.rounds):
        order = list(arms) if round_index % 2 == 0 else list(reversed(arms))
        orders.append(order)
        for active in range(args.kernel_active):
            for name in order:
                plan.append(dict(round=round_index, arm=name, active=active))
    result["single_capture_plan"] = plan
    for _ in range(args.kernel_warmup):
        for name in arms:
            launch(name, targets[name])
        torch.npu.synchronize()
    for attempt in range(1, 4):
        path = Path(args.report) / f"{args.case}-prefix-plan-{attempt:02d}.json"
        with api.profile(
            activities=[api.ProfilerActivity.CPU, api.ProfilerActivity.NPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for index, entry in enumerate(plan):
                with torch.profiler.record_function(f"qsa/timed/{index}"):
                    launch(entry["arm"], targets[entry["arm"]])
                    torch.npu.synchronize()
        prof.export_chrome_trace(str(path))
        del prof
        gc.collect()
        payload = path.read_bytes()
        archive = path.with_suffix(".json.gz")
        with gzip.open(archive, "wb") as stream:
            stream.write(payload)
        path.unlink()
        data = json.loads(payload)
        events = data.get("traceEvents", []) if isinstance(data, dict) else data
        try:
            parsed = stage_mean(events, len(plan))
            host = sum(
                e.get("ph") == "X"
                and e.get("cat") == "cpu_op"
                and e.get("name") == "_quant_score_trial"
                for e in events
            )
            if host != len(plan) or any(n != 1 for n in parsed["tasks_per_invocation"]):
                raise IncompleteScoreTrace(
                    f"Expected one score per window: host={host}, expected={len(plan)}, "
                    f"tasks={parsed['tasks_per_invocation']}"
                )
            break
        except IncompleteScoreTrace as exc:
            failure = dict(attempt=attempt, artifact=archive.name, reason=str(exc))
            result.setdefault("rejected_prefix_plans", []).append(failure)
            print("PREFIX_PLAN_REJECTED", json.dumps(failure), flush=True)
            if attempt == 3:
                raise
    samples = {name: [] for name in arms}
    durations = parsed["active_invocation_us"]
    for round_index in range(args.rounds):
        for name in orders[round_index]:
            indices = [
                i
                for i, entry in enumerate(plan)
                if entry["round"] == round_index and entry["arm"] == name
            ]
            active_values = [durations[i] for i in indices]
            value = dict(
                kernel_us=sum(active_values) / len(active_values),
                active_invocation_us=active_values,
                artifact=archive.name,
                window_indices=indices,
                device_task_count=len(indices),
            )
            samples[name].append(value)
            print(
                "PREFIX_TIMING",
                json.dumps(
                    dict(case=args.case, arm=name, round=round_index + 1, **value)
                ),
                flush=True,
            )
    result["kernel_timer"][
        "measurement_backend"
    ] = "Single interleaved CPU+NPU trace; synchronized per-launch windows; raw slow samples retained"
    result["kernel_timer"]["prefix_plan_artifact"] = archive.name
    return samples, orders
