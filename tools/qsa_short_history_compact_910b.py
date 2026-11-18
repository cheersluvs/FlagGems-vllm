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

"""Evaluate short TopK history paths on the complete compact fixture."""

import ast
import json
import math
from collections import Counter
from statistics import mean

import qsa_first_block_seed_probe_910b as first_seed
import qsa_single_block_probe_910b as single_block
from probe_qsa_raw_cube_chain_910b import record_launch_mode
from qsa_history_policy_910b import dispatcher, use_prefix
from qsa_raw_history_merge_910b import ir_evidence as history_ir


def source():
    kernels = []
    for probe, constraint in (
        (single_block, "BLOCKS == 1"),
        (first_seed, "2 <= BLOCKS and BLOCKS <= 3"),
    ):
        kernel = ast.parse(probe.source()).body[0]
        kernel.args.args.append(
            ast.parse("def f(BLOCKS: tl.constexpr): pass").body[0].args.args[0]
        )
        kernel.body[0] = ast.parse(f"tl.static_assert({constraint})").body[0]
        # WIDTH remains the caller's trace stride and logical key extent.
        for loop in ast.walk(kernel):
            if isinstance(loop, ast.For) and loop.target.id in (
                "cube_block",
                "vector_block",
            ):
                loop.iter.args[-1] = ast.Name(id="BLOCKS", ctx=ast.Load())
        kernels.append(ast.unparse(ast.fix_missing_locations(kernel)))
    return "\n" + "\n\n".join(kernels) + "\n"


def short_dispatcher(module, policy):
    fallback = dispatcher(module, policy)

    def choose(blocks):
        if blocks == 1:
            kernel = module.raw_history_single
            evidence, mode = single_block.ir_evidence, "single_direct"
        elif blocks in (2, 3):
            kernel = module.raw_history_firstseed
            evidence, mode = first_seed.ir_evidence, "first_seed"
        else:
            return fallback(blocks)
        if use_prefix(blocks, policy):
            raise ValueError(f"Compact control uses prefix merge for {blocks} blocks")

        def check(compiled):
            if record_launch_mode(compiled, require_cube=False)["mix_mode"] == "aiv":
                return history_ir(compiled, complete=False)
            return evidence(compiled)

        return kernel, module.raw_history_topk_snapshot, check, mode

    return choose


def measure(benchmark, native, snapshot, candidate, path):
    expected_routes = Counter(
        {(1, "single_direct"): 8, (2, "first_seed"): 52, (3, "first_seed"): 54}
    )
    actual_routes = Counter((r["blocks"], r["mode"]) for r in candidate.dispatch_routes)
    if actual_routes != expected_routes or snapshot.routes != candidate.routes:
        raise ValueError(f"Unexpected compact schedule: {actual_routes}")
    calls = {"cann_v1": native, "snapshot": snapshot, "candidate": candidate}
    rounds = []
    result = dict(
        status="running",
        timing="FlagGems-vllm harness get_latency, --mode kernel",
        scope="Full int8_compact; includes every piece, QW/K/KS packing, and mixed execution",
        logical_width=candidate.width,
        pieces_by_blocks=dict(Counter(r["blocks"] for r in candidate.routes)),
        rows_by_blocks={
            blocks: sum(r["count"] for r in candidate.routes if r["blocks"] == blocks)
            for blocks in (1, 2, 3)
        },
        rounds=rounds,
    )

    def save():
        path.write_text(json.dumps(result, indent=2))

    for order in (list(calls), list(reversed(calls))):
        timings = {}
        round_result = dict(order=order, latency_ms=timings)
        rounds.append(round_result)
        for name in order:
            latency = float(benchmark.get_latency(calls[name]))
            if not math.isfinite(latency) or latency <= 0:
                raise RuntimeError(f"Invalid {name} latency: {latency}")
            timings[name] = latency
            save()
            print("SHORT_HISTORY_TIMING", len(rounds), name, latency, "ms", flush=True)
        round_result.update(
            snapshot_over_candidate=timings["snapshot"] / timings["candidate"],
            cann_over_candidate=timings["cann_v1"] / timings["candidate"],
        )
        save()
    result.update(
        status="passed",
        mean_ms={name: mean(r["latency_ms"][name] for r in rounds) for name in calls},
    )
    for label in ("snapshot_over_candidate", "cann_over_candidate"):
        result[label] = mean(r[label] for r in rounds)
    save()
    print("SHORT_HISTORY_COMPACT", json.dumps(result), flush=True)
    return result
