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

"""Measure existing packing stages; do not implement or claim cache reuse."""

import json
import math
import os
from pathlib import Path


def traffic(pieces, key_bytes, scale_bytes):
    requests = {}
    valid = padded = 0
    for p in pieces:
        maximum = p["maximum"]
        blocks = p["blocks"]
        if maximum < 0 or blocks != max(1, (maximum + 511) // 512):
            raise ValueError("Invalid packing geometry")
        pages = tuple(p["pages"][: (maximum + 191) // 192])
        if len(pages) != (maximum + 191) // 192:
            raise ValueError("Incomplete physical mapping")
        old = requests.get(p["request"], dict(maximum=0, blocks=0, pages=()))
        overlap = min(len(pages), len(old["pages"]))
        if pages[:overlap] != old["pages"][:overlap]:
            raise ValueError("Request page prefix changed; reuse not established")
        if maximum >= old["maximum"]:
            old["pages"] = pages
        old["maximum"] = max(old["maximum"], maximum)
        old["blocks"] = max(old["blocks"], blocks)
        requests[p["request"]] = old
        valid += maximum
        padded += blocks * 512
    unique = sum(p["maximum"] for p in requests.values())
    unique_padded = sum(p["blocks"] * 512 for p in requests.values())
    return dict(
        pieces=len(pieces),
        requests=len(requests),
        page_prefix_verified=True,
        input_read_bytes=valid * (key_bytes + scale_bytes),
        packed_write_bytes=padded * 130,
        unique_request_prefix_input_bytes=unique * (key_bytes + scale_bytes),
        repeated_input_bytes=(valid - unique) * (key_bytes + scale_bytes),
        repeated_packed_write_bytes=(padded - unique_padded) * 130,
        all_request_max_packed_bytes=unique_padded * 130,
        input_repetition_factor=valid / unique if unique else None,
        scope="Logical element traffic, excluding page-table reads and physical cache/DMA effects",
    )


def probe(context, mixed, raw, benchmark):
    import torch

    pieces = []
    for item in mixed.jobs:
        pieces.append(
            dict(
                request=item["request"],
                maximum=item["maximum"],
                blocks=item["blocks"],
                pages=context["cpu"]["row_table"][item["first"]].tolist(),
            )
        )
    stats = traffic(
        pieces,
        128 * context["values"]["cache"].element_size(),
        context["values"]["ks"].element_size(),
    )
    progress = dict(status="running", traffic=stats, measurements_ms={})
    path = Path(os.environ["QSA_RAW_SERIAL_REPORT"]) / (
        context["plan"]["name"] + "-packing-cost.json"
    )

    def save():
        path.write_text(json.dumps(progress, indent=2))

    save()
    print("PACKING_TRAFFIC", json.dumps(stats), flush=True)

    def time(label, call):
        value = float(benchmark.get_latency(call))
        if not math.isfinite(value) or value <= 0:
            progress.update(
                status="failed", failed_stage=label, invalid_value=repr(value)
            )
            save()
            raise RuntimeError(f"Invalid {label} timing: {value}")
        progress["measurements_ms"][label] = value
        save()
        print("PACKING_COST", label, value, "ms", flush=True)
        return value

    def all_stage(name):
        for job in mixed.jobs:
            job[name]()

    times = dict(
        combined=time("combined", mixed),
        all_qw=time("all_qw", lambda: all_stage("pack_qw")),
        all_kks=time("all_kks", lambda: all_stage("pack_k")),
    )
    sampled = []
    indices = sorted({0, len(mixed.jobs) // 2, len(mixed.jobs) - 1})
    for index in indices:
        job = mixed.jobs[index]
        job["pack_qw"]()
        job["pack_k"]()
        torch.npu.synchronize()
        # Buffers are shared: measure one prepared piece before moving to another.
        sample = dict(
            piece=index,
            first=job["first"],
            rows=job["count"],
            blocks=job["blocks"],
            request=job["request"],
        )
        sample["mixed_only_ms"] = time(f"piece{index}_mixed", job["mixed_only"])
        job["mixed_only"](True)
        torch.npu.synchronize()
        entry, raw_pieces = next(
            (entry, pieces)
            for entry, pieces in context["jobs"]
            if entry["first"] <= job["first"] < entry["first"] + entry["count"]
        )
        raw.score_chunk(raw_pieces)
        torch.npu.synchronize()
        offset = job["first"] - entry["first"]
        expected = context["scores"][offset : offset + job["count"]].cpu()
        torch.testing.assert_close(
            mixed.trace[: job["count"]].cpu(), expected, rtol=0, atol=0
        )
        from qsa_raw_serial_selection_910b_trial import check_rank

        check_rank(
            torch,
            mixed.output[job["first"] : job["first"] + job["count"]].cpu(),
            expected,
            context["cpu"]["visible"][job["first"] : job["first"] + job["count"]],
            rtol=0,
            atol=0,
        )
        mixed.check_guards()
        sample["prepared_trace_and_rank_passed"] = True
        sample["qw_ms"] = time(f"piece{index}_qw", job["pack_qw"])
        sample["kks_ms"] = time(f"piece{index}_kks", job["pack_k"])
        sampled.append(sample)
    result = dict(
        status="passed",
        traffic=stats,
        stage_ms=times,
        samples=sampled,
        scope="Stage diagnostics, non-additive; no CANN or acceptance speedup",
        kks_over_combined=times["all_kks"] / times["combined"],
    )
    print("PACKING_PROBE_RESULT", result, flush=True)
    progress.update(result)
    save()
    return result
