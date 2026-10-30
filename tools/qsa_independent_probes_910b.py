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

"""Independent, test-only launch and radix-granularity probes; no source rewriting."""

import json
import os
from contextlib import nullcontext
from unittest.mock import patch


def score_candidates(torch, triton, device):
    properties = {}
    try:
        raw = triton.runtime.driver.active.utils.get_device_properties(device)
        properties = dict(raw) if isinstance(raw, dict) else {"raw": repr(raw)}
    except Exception as exc:
        properties = {"query_error": str(exc)}
    # Only explicit Cube/AIC counts are trusted; do not interpret Vector counts as Cube counts.
    bound = next(
        (
            int(properties[key])
            for key in ("cube_core_num", "aic_core_num", "num_aicore")
            if type(properties.get(key)) is int and properties[key] > 0
        ),
        None,
    )
    counts = [8] + [
        count
        for count in sorted({16, *([bound] if bound else [])})
        if count > 8 and (bound is None or count <= bound)
    ]
    return [(f"score_p{count}", dict(programs=count)) for count in counts], properties


def run(args, result, context):
    import torch
    import triton
    from benchmark_qsa_prefill_stages_910b import aggregate, sample
    from qsa_timeline_910b import collect

    topk_module = context["topk_module"]
    settings = context["score_settings"]
    if args.probe_kind == "score":
        candidates, props = score_candidates(torch, triton, args.device)
        result["core_properties"] = props
        function = context["score_stage"]
    else:
        candidates = [
            (f"topk_g{g}_b{block}", dict(G=g, BLOCK=block))
            for g, block in ((40, 1024), (8, 1024), (16, 1024), (40, 256), (40, 512))
        ]
        function = context["topk_stage"]
    result["probe"] = dict(
        kind=args.probe_kind,
        variants={},
        production_source_modified=False,
        rounds=args.rounds,
        histogram_route="radix",
        profiler_iterations=3,
    )
    variants = result["probe"]["variants"]
    old_route = os.environ.get("FLAGGEMS_ASCEND_PTK_PATH")
    os.environ["FLAGGEMS_ASCEND_PTK_PATH"] = "radix"

    def configure(params):
        if args.probe_kind == "score":
            settings["programs"] = params["programs"]
            return nullcontext()
        return patch.multiple(topk_module, _G=params["G"], _BLOCK=params["BLOCK"])

    def check():
        if args.probe_kind == "score":
            first, count = context["chunks"][-1]
            last = context["logits"][:count].cpu()
            expected_last = context["reference"][first : first + count]
            mask = torch.isfinite(expected_last)
            torch.testing.assert_close(
                last[mask], expected_last[mask], rtol=1e-3, atol=1e-3
            )
            assert torch.isneginf(last[~mask]).all()
            for first, count in context["chunks"]:
                target = context["frozen"][first : first + count]
                context["score_into"](first, count, target)
            actual, ref = context["frozen"].cpu(), context["reference"]
            finite = torch.isfinite(ref)
            torch.testing.assert_close(
                actual[finite], ref[finite], rtol=1e-3, atol=1e-3
            )
            assert torch.isneginf(actual[~finite]).all()
            context["topk_stage"]()
            selection = context["verify"](context["output"])
            return dict(
                selection=selection,
                max_score_error=float((actual[finite] - ref[finite]).abs().max()),
                masked_scores=True,
            )
        return context["verify"](context["output"])

    try:
        # Compile and validate every candidate before any timed round.
        accepted = []
        for name, params in candidates:
            entry = dict(parameters=params, status="preflight", samples=[])
            variants[name] = entry
            try:
                with configure(params):
                    if args.probe_kind == "topk":
                        context["output"].fill_(-123456)
                    function()
                    entry["correctness"] = check()
                    for _ in range(args.warmup):
                        function()
                    torch.npu.synchronize()
            except AssertionError:
                if args.probe_kind != "topk" or name == candidates[0][0]:
                    raise
                data = context["output"].cpu()
                lengths = context["visible_cpu"].tolist()
                entry["status"] = "rejected_correctness"
                entry["output_diagnostics"] = []
                for row, length in enumerate(lengths):
                    values = data[row]
                    valid = values[(values >= 0) & (values < length)]
                    entry["output_diagnostics"].append(
                        dict(
                            row=row,
                            visible=length,
                            valid_count=valid.numel(),
                            unique_count=valid.unique().numel(),
                            below_minus_one=int((values < -1).sum()),
                            sentinel_count=int((values == -123456).sum()),
                            above_visible=int((values >= length).sum()),
                            prefix=values[:32].tolist(),
                        )
                    )
                print(
                    "REJECTED",
                    name,
                    json.dumps(entry["output_diagnostics"]),
                    flush=True,
                )
                continue
            entry["status"] = "validated"
            accepted.append((name, params))
        result["probe"]["rejected_candidates"] = [
            name
            for name, value in variants.items()
            if value["status"] == "rejected_correctness"
        ]
        candidates = accepted
        for round_index in range(args.rounds):
            order = candidates if round_index % 2 == 0 else list(reversed(candidates))
            for name, params in order:
                with configure(params):
                    value = sample(function, torch, args.iterations)
                    check()
                variants[name]["samples"].append(value)
                print(
                    f"PROBE round={round_index+1} {name}: {json.dumps(value)}",
                    flush=True,
                )
        for name, params in candidates:
            entry = variants[name]
            entry["latency"] = aggregate(entry["samples"])
            capture = {}
            with configure(params):
                collect(args, {name: function}, torch, capture, selected=(name,))
                check()
            entry["timeline"] = capture["timeline"]["stages"][name]
            entry["device_task_us_per_call"] = (
                entry["timeline"]["device_task_union_us"] / 3
            )
            entry["status"] = "passed"
        baseline = variants[candidates[0][0]]
        for name, _ in candidates:
            entry = variants[name]
            entry["device_speedup_vs_baseline"] = (
                baseline["device_task_us_per_call"] / entry["device_task_us_per_call"]
            )
            print(
                f"RESULT {name}: device={entry['device_task_us_per_call']:.3f} us "
                f"ratio={entry['device_speedup_vs_baseline']:.4f} "
                f"event={entry['latency']['event_us']['median']:.3f} us",
                flush=True,
            )
    finally:
        settings["programs"] = 8
        if old_route is None:
            os.environ.pop("FLAGGEMS_ASCEND_PTK_PATH", None)
        else:
            os.environ["FLAGGEMS_ASCEND_PTK_PATH"] = old_route
    result["status"] = "passed"
    return 0
