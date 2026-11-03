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

"""Capture CPU/NPU Chrome traces and summarize device-task unions, not gap causes."""

import collections
import gzip
import json
from decimal import Decimal
from pathlib import Path


def summarize(events):
    categories = collections.Counter(str(e.get("cat", "")) for e in events)
    hardware_pids = {
        e.get("pid")
        for e in events
        if e.get("ph") == "M"
        and e.get("name") == "process_name"
        and e.get("args", {}).get("name") == "Ascend Hardware"
    }
    selected = [
        e
        for e in events
        if e.get("ph") == "X"
        and (
            e.get("pid") in hardware_pids
            or str(e.get("cat", "")).lower() in ("kernel", "npu", "npu_kernel")
        )
        and isinstance(e.get("dur"), (int, float))
        and e["dur"] > 0
    ]
    origin = min((Decimal(str(e["ts"])) for e in selected), default=Decimal(0))
    tasks = [dict(e, ts=float(Decimal(str(e["ts"])) - origin)) for e in selected]
    if not tasks:
        raise RuntimeError(
            f"No recognized device tasks; inspect trace categories: {dict(categories)}"
        )
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in tasks)
    merged = []
    gaps = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            if merged:
                gaps.append(
                    dict(
                        start_us=merged[-1][1],
                        end_us=start,
                        duration_us=start - merged[-1][1],
                    )
                )
            merged.append([start, end])
    busy = sum(end - start for start, end in merged)
    envelope = merged[-1][1] - merged[0][0]
    names = {}
    for event in tasks:
        entry = names.setdefault(
            event.get("name", "unknown"), dict(count=0, sum_us=0.0)
        )
        entry["count"] += 1
        entry["sum_us"] += event["dur"]
    return dict(
        task_count=len(tasks),
        timestamp_origin_us=str(origin),
        hardware_pids=sorted(hardware_pids),
        device_task_duration_sum_us=sum(e["dur"] for e in tasks),
        device_task_union_us=busy,
        device_task_envelope_us=envelope,
        uncovered_envelope_us=envelope - busy,
        uncovered_fraction=(envelope - busy) / envelope if envelope else 0,
        categories=dict(categories),
        largest_gaps=sorted(gaps, key=lambda g: g["duration_us"], reverse=True)[:20],
        kernels=names,
        caveat=(
            "Uncovered intervals are not proof of host starvation: dependencies, "
            "unselected activity and other processes can contribute. "
            "Use CPU launch events/flows in the raw trace."
        ),
    )


def collect(args, functions, torch, result, selected=None):
    import torch_npu

    api = torch_npu.profiler
    selected = selected or ("score", "topk", "public_total", "native_fused")
    iterations = getattr(args, "trace_iterations", 3)
    result["timeline"] = dict(
        iterations=iterations,
        stages={},
        units="microseconds",
        source="torch_npu.profiler CPU+NPU, Chrome trace",
    )
    errors = []
    for name in selected:
        path = Path(args.report) / f"{args.worker}-{name}-trace.json"
        torch.npu.synchronize()
        with api.profile(
            activities=[api.ProfilerActivity.CPU, api.ProfilerActivity.NPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            returned = None
            for index in range(iterations):
                with torch.profiler.record_function(f"qsa/{name}/{index}"):
                    returned = functions[name]()
            torch.npu.synchronize()
        del returned
        prof.export_chrome_trace(str(path))
        if not path.is_file():
            raise RuntimeError(f"Profiler failed to export {path}")
        payload = path.read_bytes()
        data = json.loads(payload)
        compressed = path.with_suffix(".json.gz")
        with gzip.open(compressed, "wb") as stream:
            stream.write(payload)
        path.unlink()
        events = data.get("traceEvents", []) if isinstance(data, dict) else data
        entry = dict(
            trace_file=compressed.name, compressed_bytes=compressed.stat().st_size
        )
        result["timeline"]["stages"][name] = entry
        try:
            entry.update(summarize(events))
        except RuntimeError as exc:
            entry["parse_error"] = str(exc)
            entry["categories"] = dict(
                collections.Counter(str(e.get("cat", "")) for e in events)
            )
            errors.append(f"{name}: {exc}")
            continue
        print(
            f"TIMELINE {name}: tasks={entry['task_count']} sum={entry['device_task_duration_sum_us']:.3f} us "
            f"union={entry['device_task_union_us']:.3f} us envelope={entry['device_task_envelope_us']:.3f} us "
            f"uncovered={entry['uncovered_envelope_us']:.3f} us",
            flush=True,
        )

    if errors:
        raise RuntimeError(
            "Traces preserved, but summary parser needs adjustment: "
            + "; ".join(errors)
        )


def collect_hardware(args, score, torch, result):
    """Collect a separate counter pass without changing the timing trace."""
    import csv
    import io
    import traceback

    import torch_npu

    api = torch_npu.profiler
    metric_name = getattr(args, "aic_metric", "PipeUtilization")
    iterations = getattr(args, "counter_iterations", 3)
    kernel_substring = getattr(args, "kernel_substring", "_paged_score_910b")
    root = Path(args.report) / f"{args.worker}-score-metrics"
    entry = dict(
        status="requested",
        metric=metric_name,
        iterations=iterations,
        scope="Separate hardware-counter pass; profiler overhead is not benchmark latency",
    )
    result["score_hardware"] = entry
    try:
        config = api._ExperimentalConfig(
            profiler_level=api.ProfilerLevel.Level1,
            aic_metrics=getattr(api.AiCMetrics, metric_name),
            export_type=api.ExportType.Text,
        )
        torch.npu.synchronize()
        with api.profile(
            activities=[api.ProfilerActivity.CPU, api.ProfilerActivity.NPU],
            experimental_config=config,
            on_trace_ready=api.tensorboard_trace_handler(str(root)),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for index in range(iterations):
                with torch.profiler.record_function(f"qsa/score-counters/{index}"):
                    score()
            torch.npu.synchronize()
            prof.step()
        tables = []
        for path in sorted(root.rglob("*.csv")):
            content = path.read_text(errors="replace")
            fields = next(csv.reader(io.StringIO(content)), [])
            tables.append(
                dict(
                    path=str(path.relative_to(root)),
                    columns=fields,
                    content=content[:2000000],
                    truncated=len(content) > 2000000,
                )
            )
        score_rows = []
        for table in tables:
            for row in csv.DictReader(io.StringIO(table["content"])):
                if any(kernel_substring in str(value) for value in row.values()):
                    score_rows.append(
                        {
                            key: value
                            for key, value in row.items()
                            if key
                            and any(
                                token in key.lower()
                                for token in (
                                    "name",
                                    "duration",
                                    "aic",
                                    "aiv",
                                    "cube",
                                    "scalar",
                                    "mte",
                                    "vector",
                                    "pipe",
                                )
                            )
                        }
                    )
        entry["score_rows_preview"] = score_rows[:8]
        entry["metric_columns"] = sorted(
            {
                key
                for table in tables
                for key in table["columns"]
                if any(
                    token in key.lower()
                    for token in ("aic_", "aiv_", "cube", "scalar", "mte")
                )
            }
        )
        artifact = Path(args.report) / f"{args.worker}-score-hardware-tables.json.gz"
        with gzip.open(artifact, "wt") as stream:
            json.dump(dict(tables=tables), stream)
        entry.update(
            status="exported" if tables else "no_csv_exported",
            artifact=artifact.name,
            tables=[
                dict(path=t["path"], columns=t["columns"], truncated=t["truncated"])
                for t in tables
            ],
        )
    except Exception:
        entry.update(status="failed", error=traceback.format_exc())
    print("SCORE_HARDWARE", json.dumps(entry), flush=True)
