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

"""Validate a test-only N128 score launch through the registered public entry."""

import ast
import hashlib
import inspect
import json
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


def run(args, result, context):
    import torch
    import triton
    from benchmark_qsa_prefill_stages_910b import aggregate, device_inventory, sample

    module = context["module"]
    kernel = module._paged_score_910b
    # Compare the full production AST, including decorator, rather than JIT's stripped source.
    node = next(
        n
        for n in ast.parse(Path(module.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_paged_score_910b"
    )
    digest = hashlib.sha256(
        ast.dump(node, include_attributes=False).encode()
    ).hexdigest()
    assert digest == "8f4c8d07542fe9ac0e83519f31ed95ca68009a68a4fa0ca40ba55eb903b4bad0"
    assert context["q"].shape[1:] == (4, 128) and context["cache"].shape[1] == 192
    programs = module._score_programs(args.device)
    public_launches = []

    class Candidate:
        recording = False

        def __getitem__(self, original_grid):
            def call(*operands, **options):
                assert options["HEADS"] == 4 and options["PAGE"] == 192
                options["BLOCK_N"] = 128
                grid = (
                    min(programs, options["ROWS"] * triton.cdiv(options["WIDTH"], 128)),
                )
                if self.recording:
                    public_launches.append(
                        dict(
                            rows=options["ROWS"],
                            width=options["WIDTH"],
                            offset=options["OFFSET"],
                            block_n=128,
                            grid=grid,
                        )
                    )
                return kernel[grid](*operands, **options)

            return call

    candidate = Candidate()

    def configure(name):
        return (
            patch.object(module, "_paged_score_910b", candidate)
            if name.startswith("n128_")
            else nullcontext()
        )

    jobs = {}
    for prefix in ("n32", "n128"):
        for stage in ("score", "topk", "prepared_pipeline", "public_total"):
            jobs[f"{prefix}_{stage}"] = context["functions"][stage]
    chunk32 = getattr(args, "chunk32_probe", False)
    chunk32_chunks = [
        (first, min(32, context["rows"] - first))
        for first in range(0, context["rows"], 32)
    ]
    candidate_public, host_identity = None, None
    if chunk32:
        candidate_public, host_identity = _public_with_chunk32(
            context["public"], module
        )
        logits32 = torch.empty(
            (min(32, context["rows"]), context["width"]),
            device=context["q"].device,
            dtype=torch.float32,
        )

        def score32():
            for first, count in chunk32_chunks:
                context["score_into"](first, count, logits32[:count])

        def topk32():
            for first, count in chunk32_chunks:
                module.persistent_topk(
                    context["frozen"][first : first + count],
                    context["visible"][first : first + count],
                    context["output"][first : first + count],
                    context["workspace"],
                    k=context["k"],
                    max_seq_len=context["width"],
                    block=module._QSA_TOPK_BLOCK,
                )

        def prepared32():
            for first, count in chunk32_chunks:
                context["score_into"](first, count, logits32[:count])
                module.persistent_topk(
                    logits32[:count],
                    context["visible"][first : first + count],
                    context["output"][first : first + count],
                    context["workspace"],
                    k=context["k"],
                    max_seq_len=context["width"],
                    block=module._QSA_TOPK_BLOCK,
                )

        jobs.update(
            n128_c32_score=score32,
            n128_c32_topk=topk32,
            n128_c32_prepared_pipeline=prepared32,
            n128_c32_public_total=lambda: candidate_public(*context["operands"]),
        )
    jobs["native_fused"] = context["native_fused"]
    jobs["native_adapter_total"] = context["adapter_total"]
    paired = getattr(args, "chunk32_paired", False)
    if paired:
        assert chunk32
        jobs = {
            prefix + "_" + stage: jobs[prefix + "_" + stage]
            for stage in ("score", "topk", "prepared_pipeline", "public_total")
            for prefix in ("n128", "n128_c32")
        }
    probe = result["tile_public_probe"] = dict(
        scope="BF16 H4 D128 page192 diagnostic cases; only candidate score tile/grid differ",
        production_source_modified=False,
        production_ast_sha256=digest,
        public_entry=context["public"].__module__ + "." + context["public"].__name__,
        topk_changed=False,
        public_launches=public_launches,
        correctness={},
        samples={name: [] for name in jobs},
        comparison=(
            "Native adapter/public is the whole-operator comparison; "
            "native fused/prepared uses prebuilt metadata"
        ),
    )
    probe["chunk32_probe"] = chunk32
    if chunk32:
        probe["scope"] = (
            "BF16 H4 D128 page192; compare N32/chunk16, N128/chunk16, N128/chunk32"
        )
        probe["chunk32_host"] = host_identity
        probe["chunk32_scope"] = (
            "Test-only AST clone of public host; only chunk cap16 becomes32. "
            "Budget formula, validation and TopK implementation unchanged. "
            "This is not an integrated production public entry. Prepared stages use the fixed 512MiB benchmark budget."
        )
    probe["paired_only"] = paired
    if paired:
        probe["scope"] = (
            "Independent paired N128/chunk16 vs N128/chunk32; no N32/native timing"
        )
    regression = getattr(args, "chunk32_regression", False)
    timing = True
    if regression:
        from qsa_chunk32_regression_910b import CASES as regression_cases

        probe["regression_case"] = regression_cases[args.worker]
        timing = regression_cases[args.worker]["timing"]
    verify = context["verify"]

    def check_score(use_chunk32=False):
        full = torch.empty_like(context["frozen"])
        chunks = chunk32_chunks if use_chunk32 else context["chunks"]
        for first, count in chunks:
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

    def check_return(name, returned):
        if name == "native_fused":
            return verify(returned[0][:, 0])
        if name == "native_adapter_total":
            return verify(context["native_output"])
        if name.endswith("_score"):
            return check_score(name.startswith("n128_c32_"))
        return verify(context["output"])

    for name, function in jobs.items():
        with configure(name):
            candidate.recording = name.startswith("n128_") and name.endswith(
                "public_total"
            )
            first_launch = len(public_launches)
            returned = function()
            candidate.recording = False
            for launch in public_launches[first_launch:]:
                launch["variant"] = name
            if chunk32 and name == "n128_c32_public_total":
                assert [v["rows"] for v in public_launches[first_launch:]] == [
                    c for _, c in chunk32_chunks
                ]
            probe["correctness"][name] = check_return(name, returned)
            for _ in range(args.warmup if timing else 0):
                function()
            torch.npu.synchronize()
    assert public_launches, "Candidate did not execute score through public entry"
    if regression:
        from qsa_chunk32_regression_910b import check_budget

        check_budget(
            args, context, candidate, candidate_public, configure, verify, probe
        )
        if not timing:
            probe["timing_skipped"] = "Correctness-only case"
            result["status"] = "passed"
            print(
                "CHUNK_REGRESSION_PASSED", args.worker, probe["correctness"], flush=True
            )
            return 0
    if getattr(args, "tile_public_timeline", False):
        _capture_pipeline(
            args, probe, context, module, jobs, configure, check_return, torch
        )
        result["npu_smi_after"] = device_inventory()
        result["status"] = "passed"
        return 0
    for round_index in range(args.rounds):
        order = list(jobs)
        if round_index % 2:
            order.reverse()
        for name in order:
            with configure(name):
                returned_check = (
                    (lambda returned: verify(returned[0][:, 0]))
                    if name == "native_fused"
                    else None
                )
                measurement = sample(jobs[name], torch, args.iterations, returned_check)
                if name != "native_fused":
                    probe["correctness"][name] = check_return(name, None)
            probe["samples"][name].append(measurement)
            print(
                "PUBLIC_TILE_TIMING",
                round_index + 1,
                name,
                json.dumps(measurement),
                flush=True,
            )
    probe["latency"] = {
        name: aggregate(values) for name, values in probe["samples"].items()
    }
    ratios = {}
    pairs = [
        ("n32_over_n128_" + stage, "n32_" + stage, "n128_" + stage)
        for stage in ("score", "topk", "prepared_pipeline", "public_total")
    ]
    pairs += [
        (
            "native_adapter_over_" + prefix,
            "native_adapter_total",
            prefix + "_public_total",
        )
        for prefix in ("n32", "n128")
    ]
    pairs += [
        (
            "native_fused_over_" + prefix + "_prepared",
            "native_fused",
            prefix + "_prepared_pipeline",
        )
        for prefix in ("n32", "n128")
    ]
    if chunk32:
        for stage in ("score", "topk", "prepared_pipeline", "public_total"):
            pairs.append(
                ("chunk16_over_chunk32_" + stage, "n128_" + stage, "n128_c32_" + stage)
            )
        pairs += [
            (
                "native_adapter_over_n128_c32",
                "native_adapter_total",
                "n128_c32_public_total",
            ),
            (
                "n32_over_n128_c32_public_total",
                "n32_public_total",
                "n128_c32_public_total",
            ),
        ]
    for label, numerator, denominator in pairs:
        if numerator not in probe["latency"] or denominator not in probe["latency"]:
            continue
        ratios[label] = {
            field: probe["latency"][numerator][field]["median"]
            / probe["latency"][denominator][field]["median"]
            for field in ("event_us", "wall_us")
        }
    if chunk32:
        probe["paired_round_ratios"] = {
            stage: [
                {
                    field: before[field] / after[field]
                    for field in ("event_us", "wall_us")
                }
                for before, after in zip(
                    probe["samples"]["n128_" + stage],
                    probe["samples"]["n128_c32_" + stage],
                )
            ]
            for stage in ("score", "topk", "prepared_pipeline", "public_total")
        }
        print(
            "CHUNK_PAIRED_ROUND_RATIOS",
            json.dumps(probe["paired_round_ratios"]),
            flush=True,
        )
    probe["speedup"] = ratios
    for name, latency in probe["latency"].items():
        print("PUBLIC_TILE_SUMMARY", name, json.dumps(latency), flush=True)
    print("PUBLIC_TILE_SPEEDUP", json.dumps(ratios), flush=True)
    print("PUBLIC_TILE_LAUNCHES", json.dumps(public_launches), flush=True)
    probe["route_states"] = dict(
        segmerge=dict(context["topk_module"]._SEGMERGE_STATE),
        persistent=dict(context["topk_module"]._PSEGMERGE_STATE),
    )
    result["npu_smi_after"] = device_inventory()
    result["status"] = "passed"
    return 0


def _capture_pipeline(
    args, probe, context, module, jobs, configure, check_return, torch
):
    from qsa_timeline_910b import collect

    probe["capture_only"] = True
    probe["profile_scope"] = (
        "Three consecutive calls per trace; CPU chunk scopes add instrumentation overhead. "
        "Device task sum/3 is distinct from Event/wall latency; no speedup acceptance here. "
        "Uncovered envelope intervals alone do not prove host starvation."
    )
    profiles = probe["profiles"] = {}
    raw_score, raw_topk = module._score, module.persistent_topk

    def score(*operands, **kwargs):
        with torch.profiler.record_function("qsa/score_chunk"):
            return raw_score(*operands, **kwargs)

    def topk(*operands, **kwargs):
        with torch.profiler.record_function("qsa/topk_chunk"):
            return raw_topk(*operands, **kwargs)

    for stage in ("score", "topk", "prepared_pipeline", "public_total"):
        for prefix in ("n32", "n128"):
            name = prefix + "_" + stage
            entry = profiles[name] = {}
            with configure(name), patch.object(module, "_score", score), patch.object(
                module, "persistent_topk", topk
            ):
                collect(args, {name: jobs[name]}, torch, entry, selected=(name,))
                # Validate a fresh call after capture, outside the profiler context.
                probe["correctness"][name] = check_return(name, jobs[name]())
            timeline = entry["timeline"]
            measured = timeline["stages"][name]
            entry["device_us_per_invocation"] = (
                measured["device_task_duration_sum_us"] / timeline["iterations"]
            )
            entry["tasks_per_invocation"] = (
                measured["task_count"] / timeline["iterations"]
            )
            print(
                "PIPELINE_PROFILE",
                name,
                json.dumps(
                    dict(
                        device_us_per_invocation=entry["device_us_per_invocation"],
                        tasks_per_invocation=entry["tasks_per_invocation"],
                        envelope_us=measured["device_task_envelope_us"],
                        uncovered_us=measured["uncovered_envelope_us"],
                        trace_file=measured["trace_file"],
                    )
                ),
                flush=True,
            )
    probe["profile_device_ratios_n32_over_n128"] = {
        stage: profiles["n32_" + stage]["device_us_per_invocation"]
        / profiles["n128_" + stage]["device_us_per_invocation"]
        for stage in ("score", "topk", "prepared_pipeline", "public_total")
    }
    probe["route_states"] = dict(
        segmerge=dict(context["topk_module"]._SEGMERGE_STATE),
        persistent=dict(context["topk_module"]._PSEGMERGE_STATE),
    )
    print(
        "PIPELINE_PROFILE_DEVICE_RATIOS",
        json.dumps(probe["profile_device_ratios_n32_over_n128"]),
        flush=True,
    )
    print("PIPELINE_PROFILE_ROUTES", json.dumps(probe["route_states"]), flush=True)


def _public_with_chunk32(public, module):
    """Clone the production host AST, changing only its literal chunk cap."""
    source = ast.parse(inspect.getsource(public))
    original = deepcopy(source)
    function = source.body[0]
    assignment = next(
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "chunk_rows" for t in n.targets)
    )
    assert isinstance(assignment.value, ast.Call) and isinstance(
        assignment.value.func, ast.Name
    )
    assert assignment.value.func.id == "min" and len(assignment.value.args) == 3
    cap = assignment.value.args[1]
    assert isinstance(cap, ast.Constant) and cap.value == 16
    cap.value = 32
    namespace = dict(vars(module))
    exec(
        compile(ast.fix_missing_locations(source), "<qsa-test-only-chunk32>", "exec"),
        namespace,
    )
    trial = namespace[public.__name__]
    # Round-trip the one change to prove that every other AST node stayed intact.
    cap.value = 16
    assert ast.dump(source, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )
    return trial, dict(
        original_ast_sha256=hashlib.sha256(
            ast.dump(original, include_attributes=False).encode()
        ).hexdigest(),
        change="chunk_rows=min(rows,16,budget_rows) becomes min(rows,32,budget_rows)",
        registered_public_unchanged=True,
        production_source_modified=False,
    )
