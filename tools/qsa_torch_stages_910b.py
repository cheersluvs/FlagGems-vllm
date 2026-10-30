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

"""Standalone Torch-NPU stage baselines, not a decomposition of LightningIndexer."""

import json


def run(args, result, c):
    import torch
    from benchmark_qsa_prefill_stages_910b import aggregate, device_inventory, sample
    from qsa_timeline_910b import collect

    q, cache, table = c["q"], c["cache"], c["table"]
    width, dim, heads, k = (c[x] for x in ("width", "dim", "heads", "k"))
    chunks = c["chunks"]
    assert bool(
        (c["visible_cpu"] > k).all()
    ), "This diagnostic requires nontrivial rows"
    plans = []
    for first, count in chunks:
        request = first // c["query_rows"]
        assert (first + count - 1) // c["query_rows"] == request
        query = q[first : first + count].reshape(1, count * heads, dim)
        mask = (
            torch.arange(width, device=q.device)[None, :]
            >= c["visible"][first : first + count, None]
        )
        plans.append((first, count, request, query, mask))
    scratch = torch.empty_like(c["logits"])

    def gather(request):
        return cache[table[request].long()].reshape(c["capacity"], dim)[:width]

    dense = [gather(request) for request in range(c["requests"])]

    def score_one(plan, keys):
        first, count, request, query, mask = plan
        # Explicit FP32 result prevents BF16 rounding of QK before ReLU/reduction.
        product = torch.bmm(
            query, keys.transpose(0, 1).unsqueeze(0), out_dtype=torch.float32
        )
        if product.dtype != torch.float32:
            raise RuntimeError(
                "Torch-NPU BMM did not return FP32; no lower-precision substitute allowed"
            )
        scores = product.clamp_min(0).reshape(count, heads, width).sum(1)
        scratch[:count].copy_(scores.masked_fill(mask, -torch.inf))

    try:
        score_one(plans[0], dense[plans[0][2]])
        torch.npu.synchronize()
    except (TypeError, RuntimeError) as exc:
        result["torch_score_capability_error"] = str(exc)
        raise RuntimeError(
            "Required BF16-input/FP32-output torch.bmm baseline unavailable; no dtype fallback used"
        ) from exc

    def torch_score_paged():
        for plan in plans:
            score_one(plan, gather(plan[2]))

    def torch_score_dense():
        for plan in plans:
            score_one(plan, dense[plan[2]])

    def torch_topk_raw():
        returned = None
        for first, count in chunks:
            returned = torch.topk(
                c["frozen"][first : first + count], k, dim=1, sorted=False
            )
        return returned

    def torch_topk_abi():
        for first, count in chunks:
            indices = torch.topk(
                c["frozen"][first : first + count], k, dim=1, sorted=False
            ).indices
            c["output"][first : first + count].copy_(indices.to(torch.int32))

    def torch_prepared():
        for plan in plans:
            first, count = plan[:2]
            score_one(plan, gather(plan[2]))
            indices = torch.topk(scratch[:count], k, dim=1, sorted=False).indices
            c["output"][first : first + count].copy_(indices.to(torch.int32))

    result["torch_stage_contract"] = dict(
        input_dtype="BF16 Q/K",
        matmul_output="FP32",
        logits="FP32",
        output="int32 groups",
        score_paged=(
            "includes page gather and index conversion, BMM, ReLU, FP32 head sum, "
            "visibility mask and output copy"
        ),
        score_dense="precollected BF16 K; compute-only diagnostic, not the paged operator boundary",
        topk_raw="same frozen logits and chunking; returns values and int64 indices",
        topk_abi="raw topk plus int32 conversion and caller-output copy",
        comparison="standalone Torch-NPU compositions; not internal stages of native LightningIndexer",
    )
    errors = []
    for plan in plans:
        first, count = plan[:2]
        score_one(plan, gather(plan[2]))
        actual = scratch[:count].cpu()
        target = c["reference"][first : first + count]
        finite = torch.isfinite(target)
        torch.testing.assert_close(actual[finite], target[finite], rtol=1e-3, atol=1e-3)
        assert torch.isneginf(actual[~finite]).all()
        errors.append(float((actual[finite] - target[finite]).abs().max()))
    result["torch_score_max_abs_error"] = max(errors)
    for fn in (torch_topk_abi, torch_prepared):
        fn()
        c["verify"](c["output"])
    jobs = dict(
        ours_score=c["score_stage"],
        torch_score_paged=torch_score_paged,
        torch_score_dense=torch_score_dense,
        ours_topk=c["topk_stage"],
        torch_topk_raw=torch_topk_raw,
        torch_topk_abi=torch_topk_abi,
        ours_prepared=c["prepared_pipeline"],
        torch_prepared=torch_prepared,
    )
    samples = {name: [] for name in jobs}
    for fn in jobs.values():
        for _ in range(args.warmup):
            fn()
        torch.npu.synchronize()
    for round_index in range(args.rounds):
        order = list(jobs) if round_index % 2 == 0 else list(reversed(jobs))
        for name in order:
            samples[name].append(sample(jobs[name], torch, args.iterations))
            if name in (
                "ours_topk",
                "torch_topk_abi",
                "ours_prepared",
                "torch_prepared",
            ):
                c["verify"](c["output"])
            print(
                "STAGE",
                round_index + 1,
                name,
                json.dumps(samples[name][-1]),
                flush=True,
            )
    result["torch_stage_latency"] = {name: aggregate(v) for name, v in samples.items()}
    capture = {}
    selected = (
        "ours_score",
        "torch_score_paged",
        "ours_topk",
        "torch_topk_abi",
        "ours_prepared",
        "torch_prepared",
    )
    collect(args, jobs, torch, capture, selected=selected)
    result["torch_stage_timeline"] = capture["timeline"]
    result["torch_stage_ratios"] = {}
    for label, ours, baseline in (
        ("score_paged", "ours_score", "torch_score_paged"),
        ("topk_abi", "ours_topk", "torch_topk_abi"),
        ("prepared", "ours_prepared", "torch_prepared"),
    ):
        stages = capture["timeline"]["stages"]
        ratio = (
            stages[baseline]["device_task_union_us"]
            / stages[ours]["device_task_union_us"]
        )
        result["torch_stage_ratios"][label] = dict(
            device=ratio,
            **{
                field: result["torch_stage_latency"][baseline][field]["median"]
                / result["torch_stage_latency"][ours][field]["median"]
                for field in ("event_us", "wall_us")
            },
        )
        print(
            f"RATIO {label}: {json.dumps(result['torch_stage_ratios'][label])}",
            flush=True,
        )
    for name in selected:
        value = capture["timeline"]["stages"][name]
        print(
            f"DEVICE {name}: {value['device_task_union_us']/3:.3f} us/call", flush=True
        )
    torch_prepared()
    result["torch_prepared_post_check"] = c["verify"](c["output"])
    result["npu_smi_after"] = device_inventory()
    result["status"] = "passed"
    return 0
