# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Focused fixed-tile diagnostics; never imported by production operators."""

import csv
import math
import time

CASES = (
    "fp8_first_chunk_float8_e4m3fn",
    "observed_short_batch3_bfloat16",
)
CONDITIONS = {
    "separate": ("a", "b"),
    "swapped": ("b", "a"),
    "shared_a": ("a", "a"),
    "shared_b": ("b", "b"),
}
# Balanced condition positions and preceding conditions over each four cycles.
CONDITION_ORDERS = (
    ("separate", "swapped", "shared_b", "shared_a"),
    ("swapped", "shared_a", "separate", "shared_b"),
    ("shared_a", "shared_b", "swapped", "separate"),
    ("shared_b", "separate", "shared_a", "swapped"),
)


def schedule(case, cycles):
    if case not in CASES or cycles < 4 or cycles % 4:
        raise ValueError("Query controls require a known case and cycles multiple of4")
    stages = ("score",) if case == CASES[0] else ("score", "topk", "pipeline")
    result = []
    for cycle in range(cycles):
        shift = cycle % len(stages)
        order = stages[shift:] + stages[:shift]
        for condition in CONDITION_ORDERS[cycle % 4]:
            for stage in order:
                result.append(dict(cycle=cycle + 1, condition=condition, stage=stage))
    return result


def fixed_traces(traces, case, *, joint, parts=1):
    if case not in CASES or parts not in (1, 2):
        raise ValueError("Unknown fixed query control")
    if case == CASES[1] and parts != 1:
        raise ValueError("BF16 query controls require full-query mode")
    if len(traces) != 1 or traces[0]["route"] != "rectangular":
        raise RuntimeError("Focused fixture must retain its single rectangular chunk")
    constants = dict(
        num_warps=4,
        num_stages=2,
        num_ctas=1,
        TILE_R=32 if case == CASES[0] else 8,
        BLOCK_N=64,
        K_TILES=4,
    )
    if joint:
        constants["QUERY_PARTS"] = parts
    return [dict(traces[0], constants=constants)]


def assert_choice(module, traces):
    wanted = traces[0]["constants"]
    choices = module._qsa_prefill_score_kernel.fn.configs
    if not any(
        dict(
            c.kwargs,
            num_warps=c.num_warps,
            num_stages=c.num_stages,
            num_ctas=c.num_ctas,
        )
        == wanted
        for c in choices
    ):
        raise RuntimeError("Diagnostic fixed choice absent from existing YAML group")


def freeze(pair):
    from tools import qsa_prefill_query_split_matched as matched

    records = [matched.freeze_and_records((m, m))["baseline"] for m in pair]
    normalized = []
    for side in records:
        normalized.append(
            {
                tuple(r["chunk"]): {
                    k: v for k, v in r["constants"].items() if k != "QUERY_PARTS"
                }
                for r in side
            }
        )
    if normalized[0] != normalized[1]:
        raise RuntimeError("Focused comparison differs in non-mode tile parameters")
    return dict(baseline=records[0], candidate=records[1], same_tile=True)


def layout(chunks):
    result = []
    for c in chunks:
        result.append(
            {
                name: dict(
                    pointer=tensor.data_ptr(),
                    shape=list(tensor.shape),
                    stride=list(tensor.stride()),
                )
                for name, tensor in (
                    ("score", c["score_args"][5]),
                    ("topk_workspace", c["topk_args"][3]),
                    ("output", c["topk_args"][2]),
                )
            }
        )
    return result


def validate_side(stage, fn, module, chunks, args, context):
    """Validate this owner immediately; a following shared-buffer launch may overwrite."""
    b, torch, helpers, device, native, native_chunks = context
    if stage == "score":
        return b.scoring.validate_scores(
            module._prefill_logits, chunks, args, torch, b.support.sample_rows, device
        )
    if stage == "topk":
        return b.validate_topk(native, module, native_chunks, chunks, torch, helpers)
    if stage != "pipeline":
        raise ValueError("Unknown diagnostic stage")
    # Two identical closures/outputs deliberately check one side at a time.
    return b.validate_pipeline(
        dict(baseline=fn, candidate=fn),
        args,
        args,
        native_chunks,
        torch,
        helpers,
    )


def write_table(path, result):
    fields = (
        "cycle",
        "condition",
        "stage",
        "baseline_bank",
        "candidate_bank",
        "baseline_us",
        "candidate_us",
        "speedup",
        "baseline_samples",
        "candidate_samples",
        "paired_ratios",
    )
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        for entry in result["measurements"]:
            if not entry.get("measurement_validated"):
                continue
            row = {k: entry[k] for k in fields[:5]}
            row.update(
                baseline_us=entry["baseline_us"]["median"],
                candidate_us=entry["candidate_us"]["median"],
                speedup=entry["speedup"],
                baseline_samples=entry["baseline_us"]["samples"],
                candidate_samples=entry["candidate_us"]["samples"],
                paired_ratios=entry["paired_ratios"]["samples"],
            )
            writer.writerow(row)


def run(
    options,
    result,
    path,
    *,
    bench,
    torch,
    helpers,
    device,
    native,
    candidate,
    production,
    native_chunks,
    production_chunks,
    candidate_chunks,
    production_args,
    args,
    monitor,
):
    from tools import qsa_prefill_query_split_matched as matched

    case = result["case"]["name"]
    jobs = schedule(case, options.query_control_cycles)
    if candidate.persistent_topk is not production.persistent_topk:
        raise RuntimeError("Control changed production TopK")
    traces = result["cache_granularity"]["production"]
    pair = []
    for source, parts in (
        (candidate if case == CASES[0] else production, 1),
        (candidate, 2 if case == CASES[0] else 1),
    ):
        selected = fixed_traces(traces, case, joint=source is candidate, parts=parts)
        assert_choice(source, selected)
        pair.append(matched.modules(source, source, selected)[0])
    banks = dict(a=production_chunks, b=candidate_chunks)
    owners = dict(a=production_args, b=args)
    context = (bench, torch, helpers, device, native, native_chunks)
    bundles = {
        (index, bank): bench.ab.bind_stages(native, module, native_chunks, chunks)
        for index, module in enumerate(pair)
        for bank, chunks in banks.items()
    }
    diagnostic = result["query_controls"] = dict(
        status="preparing",
        formal_acceptance_eligible=False,
        policy="Fixed tile; balanced condition positions; original paired kernel timer",
        cycles=options.query_control_cycles,
        rounds=options.rounds,
        schedule=jobs,
        buffers={k: layout(v) for k, v in banks.items()},
        same_topk=True,
        measurements=[],
    )
    for name in ("score", "topk_workspace", "output"):
        if (
            diagnostic["buffers"]["a"][0][name]["pointer"]
            == diagnostic["buffers"]["b"][0][name]["pointer"]
        ):
            raise RuntimeError("Independent diagnostic banks unexpectedly alias")
    for bundle in bundles.values():
        for stage in ("score", "topk", "pipeline"):
            bundle[stage]["candidate"]()  # Compile/prepare before any timing.
    device.synchronize()
    frozen = freeze(pair)
    diagnostic["kernels_before"] = frozen
    timer, metadata = bench.scoring.make_timer(
        options.kernel_warmup_ms, options.kernel_rep_ms
    )
    diagnostic["timer"] = {k: v for k, v in metadata.items() if k != "includes_topk"}
    diagnostic["status"] = "running"
    table = path.with_suffix(".controls.tsv")
    for position, job in enumerate(jobs):
        left, right = CONDITIONS[job["condition"]]
        entry = dict(
            job,
            position=position,
            status="validating",
            baseline_bank=left,
            candidate_bank=right,
            baseline_kind=("joint_full_query" if case == CASES[0] else "production"),
            candidate_kind=(
                "joint_split_query" if case == CASES[0] else "joint_full_query"
            ),
            includes_topk=job["stage"] != "score",
            measurement_validated=False,
        )
        diagnostic["measurements"].append(entry)
        functions = {
            side: bundles[(index, bank)][job["stage"]]["candidate"]
            for side, index, bank in (("baseline", 0, left), ("candidate", 1, right))
        }

        def validate():
            return {
                side: validate_side(
                    job["stage"],
                    functions[side],
                    pair[index],
                    banks[bank],
                    owners[bank],
                    context,
                )
                for side, index, bank in (
                    ("baseline", 0, left),
                    ("candidate", 1, right),
                )
            }

        entry["validation_before"] = validate()
        bench.support.write_json(path, result)
        entry["started_monotonic"] = time.monotonic()
        measured, _, order = bench.scoring.measure_scores(
            timer,
            functions["baseline"],
            functions["candidate"],
            options.rounds,
            lambda label: monitor(f"controls:{position}:{label}"),
            bench.support.stats,
        )
        entry.update(measured, measurement_order=order)
        entry["ended_monotonic"] = time.monotonic()
        if not all(
            math.isfinite(v) and v > 0
            for side in ("baseline_us", "candidate_us")
            for v in entry[side]["samples"]
        ):
            raise RuntimeError("Invalid diagnostic timing samples")
        entry["validation_after"] = validate()
        device.synchronize()
        entry.update(status="passed", measurement_validated=True)
        bench.support.write_json(path, result)
        write_table(table, diagnostic)
    diagnostic["kernels_after"] = freeze(pair)
    if diagnostic["kernels_after"] != frozen:
        raise RuntimeError("Fixed diagnostic kernels changed during timing")
    diagnostic["status"] = "passed"
    bench.support.write_json(path, result)
