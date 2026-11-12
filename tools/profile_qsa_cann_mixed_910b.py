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

"""Profile the exact mixed kernels after separate harness timing."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def collect_mixed_profiles(controls, report, case, result):
    import torch
    from qsa_timeline_910b import collect, collect_hardware

    report = Path(report)
    profile = dict(status="running", stages={}, benchmark_timing_already_complete=True,
                   overlap_status="unproven",
                   evidence_limit="Task timelines and PipeUtilization counters do not alone prove intra-kernel Cube/Vector overlap")
    result["mixed_profiling"] = profile
    path = report / f"{case}-mixed-profiling.json"
    try:
        for name in ("mixed_serial", "mixed_double"):
            adapter = controls[name]
            binaries = {}
            for launch, kernel in adapter.compiled.items():
                binaries[launch] = dict(metadata=str(kernel.metadata),
                    binary_sha256={kind: hashlib.sha256(data).hexdigest()
                        for kind, data in kernel.asm.items() if isinstance(data, bytes)})
            entry = dict(status="running", slots=adapter.slots, kernel_identity=binaries)
            profile["stages"][name] = entry
            path.write_text(json.dumps(profile, indent=2))
            args = SimpleNamespace(report=str(report), worker=f"{case}-{name}",
                trace_iterations=3, counter_iterations=3, aic_metric="PipeUtilization",
                kernel_substring="_mixed_outer2048")
            capture = {}
            collect(args, {name: adapter}, torch, capture, selected=(name,))
            entry["timeline"] = capture["timeline"]
            collect_hardware(args, adapter, torch, capture)
            entry["hardware"] = capture["score_hardware"]
            entry["status"] = "collected"
            print("QSA MIXED PROFILE " + json.dumps(dict(arm=name,
                hardware_status=entry["hardware"]["status"],
                metric_columns=entry["hardware"].get("metric_columns", []),
                rows=entry["hardware"].get("score_rows_preview", []))), flush=True)
            path.write_text(json.dumps(profile, indent=2))
        profile["status"] = "collected"
        if any(e["hardware"]["status"] != "exported" for e in profile["stages"].values()):
            profile["status"] = "partial_hardware"
        print("QSA MIXED RECHECK paired speedups: " + json.dumps(result["paired_double_buffer_speedups"]), flush=True)
        print("QSA MIXED OVERLAP: unproven; inspect exported timelines and counters", flush=True)
    except Exception as exc:
        profile.update(status="failed", error=repr(exc))
        raise
    finally:
        path.write_text(json.dumps(profile, indent=2))
