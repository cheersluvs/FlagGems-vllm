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

"""Compare registered production routing with the prior N32/chunk16 policy."""

import json
import os
from contextlib import nullcontext
from unittest.mock import patch

from qsa_chunk32_regression_910b import CASES as REGRESSION_CASES

CASES = {
    **REGRESSION_CASES,
    "short16_g1024": dict(lengths=[16], groups=1024, timing=True),
    "medium32_g8192": dict(lengths=[16, 16], groups=8192, timing=True),
    "long32_g25600": dict(lengths=[16, 16], groups=25600, timing=True),
    "fallback_heads1": dict(lengths=[3], groups=8192, heads=1, timing=False),
    "fallback_page256": dict(lengths=[3], groups=8192, page=256, timing=False),
}


def run(args, result, context):
    import torch
    from benchmark_qsa_prefill_stages_910b import aggregate, device_inventory, sample

    module, public = context["module"], context["public"]
    config = CASES[args.worker]
    assert public is module.qsa_select_paged_prefill
    raw_score = module._score
    launches = []
    probe = result["production_policy_probe"] = dict(
        scope="Registered public entry; old control only restores N32/chunk16 routing",
        test_only_host_clone=False,
        config=config,
        launches=launches,
        correctness={},
        samples={},
        production_entry=result["production_entry"],
        topk_implementation_changed=False,
    )

    def configure(name):
        return (
            patch.multiple(
                module, _score_block_n=lambda *a: 32, _prefill_chunk_cap=lambda *a: 16
            )
            if name.startswith("old_")
            else nullcontext()
        )

    def plan(cap, budget=512):
        size = min(
            context["rows"], cap, max(1, budget * 1024 * 1024 // (context["width"] * 4))
        )
        return [
            (first, min(size, context["rows"] - first))
            for first in range(0, context["rows"], size)
        ]

    new_cap = module._prefill_chunk_cap(
        context["q"],
        context["cache"],
        context["table"],
        context["starts"],
        context["width"],
        context["k"],
    )
    new_block = module._score_block_n(
        context["q"], context["cache"], context["table"], context["width"]
    )
    eligible = args.worker not in (
        "short16_g1024",
        "fallback_heads1",
        "fallback_page256",
        "chunk_budget_zero",
    )
    assert new_block == (128 if eligible else 32), (args.worker, new_block)
    assert new_cap == (32 if eligible and context["width"] <= 16384 else 16)
    probe["dispatch"] = dict(block_n=new_block, chunk_cap=new_cap)
    buffers = {}
    plans = {"old": plan(16), "production": plan(new_cap)}
    jobs = {}
    for prefix, chunks in plans.items():
        size = max(count for _, count in chunks)
        logits = torch.empty(
            (size, context["width"]), device=context["q"].device, dtype=torch.float32
        )
        buffers[prefix] = logits

        def make_stage(stage, chunks=chunks, logits=logits):
            def function():
                for first, count in chunks:
                    chunk = logits[:count]
                    if stage in ("score", "prepared_pipeline"):
                        context["score_into"](first, count, chunk)
                    if stage in ("topk", "prepared_pipeline"):
                        module.persistent_topk(
                            (
                                context["frozen"][first : first + count]
                                if stage == "topk"
                                else chunk
                            ),
                            context["visible"][first : first + count],
                            context["output"][first : first + count],
                            context["workspace"],
                            k=context["k"],
                            max_seq_len=context["width"],
                            block=module._QSA_TOPK_BLOCK,
                        )

            return function

        for stage in ("score", "topk", "prepared_pipeline"):
            jobs[prefix + "_" + stage] = make_stage(stage)
        jobs[prefix + "_public_total"] = context["public_total"]
    jobs["native_adapter_total"] = context["adapter_total"]
    jobs["native_fused"] = context["native_fused"]
    jobs = {
        prefix + "_" + stage: jobs[prefix + "_" + stage]
        for stage in ("score", "topk", "prepared_pipeline", "public_total")
        for prefix in ("old", "production")
    } | {name: jobs[name] for name in ("native_adapter_total", "native_fused")}
    probe["samples"] = {name: [] for name in jobs}

    def check(name, returned=None):
        if name == "native_fused":
            return context["verify"](returned[0][:, 0])
        if name == "native_adapter_total":
            return context["verify"](context["native_output"])
        if not name.endswith("_score"):
            return context["verify"](context["output"])
        full = torch.empty_like(context["frozen"])
        for first, count in plans[name.split("_")[0]]:
            context["score_into"](first, count, full[first : first + count])
        actual = full.cpu()
        expected = context["reference"]
        finite = torch.isfinite(expected)
        torch.testing.assert_close(
            actual[finite], expected[finite], rtol=1e-3, atol=1e-3
        )
        assert torch.isneginf(actual[~finite]).all()
        return dict(
            max_abs_error=float((actual[finite] - expected[finite]).abs().max()),
            masks_valid=True,
        )

    for name, function in jobs.items():
        with configure(name):
            if name.endswith("_public_total"):

                def observe(*operands, **kwargs):
                    launches.append(
                        dict(
                            variant=name,
                            rows=operands[5].shape[0],
                            offset=kwargs.get("offset", 0),
                            block_n=module._score_block_n(
                                operands[0],
                                operands[1],
                                operands[2],
                                operands[5].shape[1],
                            ),
                        )
                    )
                    return raw_score(*operands, **kwargs)

                with patch.object(module, "_score", observe):
                    returned = function()
            else:
                returned = function()
            probe["correctness"][name] = check(name, returned)
            if config["timing"]:
                for _ in range(args.warmup):
                    function()
                torch.npu.synchronize()
    for prefix in plans:
        expected = plans[prefix]
        actual = [
            (v["offset"], v["rows"])
            for v in launches
            if v["variant"] == prefix + "_public_total"
        ]
        assert actual == expected, (prefix, actual, expected)
    if "budget_mb" in config:
        budget = config["budget_mb"]
        captured = []

        def observe_budget(*operands, **kwargs):
            captured.append((kwargs.get("offset", 0), operands[5].shape[0]))
            return raw_score(*operands, **kwargs)

        with patch.dict(
            os.environ, {"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": str(budget)}
        ), patch.object(module, "_score", observe_budget):
            public(*context["operands"])
        assert captured == plan(new_cap, budget)
        probe["budget_check"] = dict(
            budget_mb=budget,
            chunks=captured,
            correctness=context["verify"](context["output"]),
        )
        print("PRODUCTION_BUDGET", json.dumps(probe["budget_check"]), flush=True)
    print(
        "PRODUCTION_DISPATCH",
        json.dumps(probe["dispatch"]),
        json.dumps(launches),
        flush=True,
    )
    print("PRODUCTION_CORRECTNESS", json.dumps(probe["correctness"]), flush=True)
    if not config["timing"]:
        probe["timing_skipped"] = True
        result["status"] = "passed"
        return 0
    for round_index in range(args.rounds):
        names = list(jobs)
        if round_index % 2:
            names.reverse()
        for name in names:
            with configure(name):
                verify_return = (
                    (lambda value: context["verify"](value[0][:, 0]))
                    if name == "native_fused"
                    else None
                )
                value = sample(jobs[name], torch, args.iterations, verify_return)
                if name != "native_fused":
                    probe["correctness"][name] = check(name)
            probe["samples"][name].append(value)
            print(
                "PRODUCTION_TIMING",
                round_index + 1,
                name,
                json.dumps(value),
                flush=True,
            )
    probe["latency"] = {
        name: aggregate(values) for name, values in probe["samples"].items()
    }
    pairs = [
        ("old_over_production_" + stage, "old_" + stage, "production_" + stage)
        for stage in ("score", "topk", "prepared_pipeline", "public_total")
    ]
    pairs += [
        (
            "native_adapter_over_production",
            "native_adapter_total",
            "production_public_total",
        )
    ]
    probe["speedup"] = {
        label: {
            field: probe["latency"][a][field]["median"]
            / probe["latency"][b][field]["median"]
            for field in ("event_us", "wall_us")
        }
        for label, a, b in pairs
    }
    for name, latency in probe["latency"].items():
        print("PRODUCTION_SUMMARY", name, json.dumps(latency), flush=True)
    print("PRODUCTION_SPEEDUP", json.dumps(probe["speedup"]), flush=True)
    probe["routes"] = dict(
        segmerge=dict(context["topk_module"]._SEGMERGE_STATE),
        persistent=dict(context["topk_module"]._PSEGMERGE_STATE),
    )
    result["npu_smi_after"] = device_inventory()
    result["status"] = "passed"
    return 0
