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

"""Compare production score with two K-address-only candidates; no production edits."""

import ast
import hashlib
import json
from contextlib import nullcontext
from copy import copy
from pathlib import Path
from unittest.mock import patch


def run(args, result, context):
    import qsa_score_address_kernels_910b as kernels
    import torch
    from benchmark_qsa_prefill_stages_910b import aggregate, sample
    from qsa_timeline_910b import collect, collect_hardware

    module = context["module"]
    source = Path(module.__file__)
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_paged_score_910b"
    )
    digest = hashlib.sha256(
        ast.dump(node, include_attributes=False).encode()
    ).hexdigest()
    assert (
        digest == kernels.PRODUCTION_AST_SHA256
    ), "Production kernel changed; regenerate/review probe"
    cache = context["cache"]
    if cache.shape[1] % 32 or 31 * cache.stride(1) + 127 * cache.stride(3) >= 2**31:
        raise NotImplementedError(
            "Probe requires page divisible by 32 and signed-int32 local K offsets"
        )
    jobs = {
        "production": None,
        "scalar_2d": kernels._paged_score_910b_scalar_2d,
        "scalar_flat": kernels._paged_score_910b_scalar_flat,
    }
    variants = {name: dict(samples=[]) for name in jobs}
    result["address_probe"] = dict(
        scope="K-address representation only; scalar variants also deduplicate the page-table load",
        block_n=32,
        programs=module._score_programs(args.device),
        variants=variants,
        production_ast_sha256=digest,
        production_source_modified=False,
        restriction="Page divisible by 32; int32 local offsets; BF16 diagnostic workloads",
        comparisons=(
            "production->scalar_2d combines scalar base and page-id reuse; "
            "scalar_2d->scalar_flat isolates flattening"
        ),
    )
    function = context["score_stage"]

    def configure(name):
        return (
            nullcontext()
            if jobs[name] is None
            else patch.object(module, "_paged_score_910b", jobs[name])
        )

    def check():
        full = torch.empty_like(context["frozen"])
        for first, count in context["chunks"]:
            context["score_into"](first, count, full[first : first + count])
        actual = full.cpu()
        target = context["reference"]
        finite = torch.isfinite(target)
        torch.testing.assert_close(actual[finite], target[finite], rtol=1e-3, atol=1e-3)
        assert torch.isneginf(actual[~finite]).all()
        torch.testing.assert_close(
            actual[finite], context["frozen"].cpu()[finite], rtol=1e-3, atol=1e-3
        )
        return dict(
            max_abs_error=float((actual[finite] - target[finite]).abs().max()),
            masks_valid=True,
        )

    for name in jobs:
        print("ADDRESS_PREFLIGHT", name, flush=True)
        with configure(name):
            variants[name]["correctness_before"] = check()
            # Additional visibility boundaries, strided K and int64 page table.
            if args.worker.startswith("short"):
                edge = [0, 1, 31, 32, 33, 191, 192, 193]
                lengths = torch.tensor(
                    [edge[i % len(edge)] for i in range(context["rows"])],
                    device=cache.device,
                    dtype=torch.int32,
                )
                storage = torch.empty(
                    (*cache.shape[:-1], cache.shape[-1] * 2),
                    device=cache.device,
                    dtype=cache.dtype,
                )
                strided = storage[..., ::2]
                strided.copy_(cache)
                table64 = context["table"].to(torch.int64)
                target = context["reference"].clone()
                columns = torch.arange(context["width"])
                target[columns[None, :] >= lengths.cpu()[:, None]] = -torch.inf
                full = torch.empty_like(context["frozen"])
                for first, count in context["chunks"]:
                    module._score(
                        context["q"],
                        strided,
                        table64,
                        context["starts"],
                        lengths,
                        full[first : first + count],
                        offset=first,
                        owners=context["owners"],
                    )
                actual = full.cpu()
                finite = torch.isfinite(target)
                torch.testing.assert_close(
                    actual[finite], target[finite], rtol=1e-3, atol=1e-3
                )
                assert torch.isneginf(actual[~finite]).all()
                variants[name]["boundary_strided_int64_check"] = "passed"
            for _ in range(args.warmup):
                function()
            torch.npu.synchronize()
    for round_index in range(args.rounds):
        order = list(jobs)
        if round_index % 2:
            order.reverse()
        for name in order:
            with configure(name):
                measurement = sample(function, torch, args.iterations)
                variants[name]["samples"].append(measurement)
                variants[name]["correctness_after"] = check()
            print(
                "ADDRESS_TIMING",
                name,
                round_index + 1,
                json.dumps(measurement),
                flush=True,
            )
    for name in jobs:
        entry = variants[name]
        entry["latency"] = aggregate(entry["samples"])
        entry["speedup_production_over_candidate"] = {
            field: variants["production"]["latency"][field]["median"]
            / entry["latency"][field]["median"]
            for field in ("event_us", "wall_us")
        }
        print(
            "ADDRESS_SUMMARY",
            name,
            json.dumps(entry["latency"]),
            json.dumps(entry["speedup_production_over_candidate"]),
            flush=True,
        )
        profile_args = copy(args)
        profile_args.worker = f"{args.worker}-{name}"
        with configure(name):
            collect(
                profile_args, {"score": function}, torch, entry, selected=("score",)
            )
            if args.worker.startswith("long"):
                collect_hardware(profile_args, function, torch, entry)
    result["status"] = "passed"
    return 0
