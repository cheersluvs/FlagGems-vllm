# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Fixed combined-candidate controls with exact-callable self-comparisons."""

import csv
import hashlib
import math
import time
from pathlib import Path

from tools import qsa_prefill_query_controls as common
from tools.qsa_prefill_query_page_trial import PRODUCTION_SHA256

CASE = "observed_short_batch3_bfloat16"
COMBINED_SHA256 = "1ef12d431b65971c5617f0a34f614360008c652a36125d5679da00206b212d7c"
KINDS = ("production", "combined")
COMPARISONS = {
    "production_self": (0, 0),
    "combined_self": (1, 1),
    "production_vs_combined": (0, 1),
}


def schedule(cycles):
    if cycles not in (4, 8, 12, 16):
        raise ValueError("Combined controls require4/8/12/16 cycles")
    result = []
    stages = ("score", "topk", "pipeline")
    comparisons = tuple(COMPARISONS)
    for cycle in range(cycles):
        shift = cycle % len(stages)
        stage_order = stages[shift:] + stages[:shift]
        for ci, condition in enumerate(common.CONDITION_ORDERS[cycle % 4]):
            for si, stage in enumerate(stage_order):
                offset = (cycle + ci + si) % len(comparisons)
                pair_order = comparisons[offset:] + comparisons[:offset]
                if cycle % 2:
                    pair_order = pair_order[::-1]
                for comparison in pair_order:
                    result.append(
                        dict(
                            cycle=cycle + 1,
                            condition=condition,
                            stage=stage,
                            comparison=comparison,
                        )
                    )
    return result


def verify_bindings(result, production, candidate):
    if result["case"]["name"] != CASE or result["candidate_kind"] != "query-joint-page":
        raise RuntimeError(
            "Combined controls require the frozen combined short-batch case"
        )
    values = {}
    for name, module, expected in (
        ("production", production, PRODUCTION_SHA256),
        ("combined", candidate, COMBINED_SHA256),
    ):
        path = Path(module.__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError("Combined self-control source pin changed: " + name)
        values[name] = dict(path=str(path), sha256=digest)
    if (
        candidate.production_baseline is not production
        or candidate.persistent_topk is not production.persistent_topk
    ):
        raise RuntimeError("Combined self-control baseline or TopK binding changed")
    return values


def validate_side(stage, fn, module, chunks, args, context):
    if stage != "pipeline":
        return common.validate_side(stage, fn, module, chunks, args, context)
    bench, torch, helpers, _, _, native_chunks = context
    fn()

    # Read this launch's output immediately. The existing pair checker can
    # then inspect one owner twice without launching anything that overwrites it.
    def prepared():
        pass

    return bench.validate_pipeline(
        dict(baseline=prepared, candidate=prepared),
        args,
        args,
        native_chunks,
        torch,
        helpers,
    )


def write_table(path, result):
    fields = (
        "cycle",
        "position",
        "comparison",
        "condition",
        "stage",
        "baseline_kind",
        "candidate_kind",
        "baseline_bank",
        "candidate_bank",
        "same_module",
        "same_callable",
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
            row = {k: entry[k] for k in fields[:11]}
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

    bindings = verify_bindings(result, production, candidate)
    jobs = schedule(options.combined_self_control_cycles)
    if candidate.persistent_topk is not production.persistent_topk:
        raise RuntimeError("Control changed production TopK")
    traces = result["cache_granularity"]["production"]
    pair = []
    for source, parts in (
        (production, 1),
        (candidate, 1),
    ):
        selected = common.fixed_traces(
            traces, CASE, joint=source is candidate, parts=parts
        )
        common.assert_choice(source, selected)
        pair.append(matched.modules(source, source, selected)[0])
    banks = dict(a=production_chunks, b=candidate_chunks)
    owners = dict(a=production_args, b=args)
    context = (bench, torch, helpers, device, native, native_chunks)
    bundles = {
        (index, bank): bench.ab.bind_stages(native, module, native_chunks, chunks)
        for index, module in enumerate(pair)
        for bank, chunks in banks.items()
    }
    diagnostic = result["combined_self_controls"] = dict(
        status="preparing",
        formal_acceptance_eligible=False,
        policy="Fixed tile; balanced condition positions; original paired kernel timer",
        cycles=options.combined_self_control_cycles,
        rounds=options.rounds,
        schedule=jobs,
        buffers={k: common.layout(v) for k, v in banks.items()},
        same_topk=True,
        source_binding=bindings,
        comparisons={k: list(v) for k, v in COMPARISONS.items()},
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
    frozen = common.freeze(pair)
    diagnostic["kernels_before"] = frozen
    timer, metadata = bench.scoring.make_timer(
        options.kernel_warmup_ms, options.kernel_rep_ms
    )
    diagnostic["timer"] = {k: v for k, v in metadata.items() if k != "includes_topk"}
    diagnostic["status"] = "running"
    table = path.with_suffix(".combined-controls.tsv")
    for position, job in enumerate(jobs):
        left, right = common.CONDITIONS[job["condition"]]
        entry = dict(
            job,
            position=position,
            status="validating",
            baseline_bank=left,
            candidate_bank=right,
            baseline_kind=KINDS[COMPARISONS[job["comparison"]][0]],
            candidate_kind=KINDS[COMPARISONS[job["comparison"]][1]],
            includes_topk=job["stage"] != "score",
            measurement_validated=False,
        )
        diagnostic["measurements"].append(entry)
        left_index, right_index = COMPARISONS[job["comparison"]]
        owners_by_side = (
            ("baseline", left_index, left),
            ("candidate", right_index, right),
        )
        functions = {
            side: bundles[(index, bank)][job["stage"]]["candidate"]
            for side, index, bank in owners_by_side
        }
        entry["same_module"] = left_index == right_index
        entry["same_callable"] = functions["baseline"] is functions["candidate"]
        if entry["same_callable"] != (entry["same_module"] and left == right):
            raise RuntimeError(
                "Self-control does not have the intended callable identity"
            )

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
                for side, index, bank in owners_by_side
            }

        bench.support.write_json(path, result)
        entry["validation_before"] = validate()
        bench.support.write_json(path, result)
        entry["started_monotonic"] = time.monotonic()
        measured, _, order = bench.scoring.measure_scores(
            timer,
            functions["baseline"],
            functions["candidate"],
            options.rounds,
            lambda label: monitor(f"combined_self:{position}:{label}"),
            bench.support.stats,
        )
        entry.update(measured, status="pending_validation", measurement_order=order)
        entry["ended_monotonic"] = time.monotonic()
        if not all(
            math.isfinite(v) and v > 0
            for side in ("baseline_us", "candidate_us")
            for v in entry[side]["samples"]
        ):
            raise RuntimeError("Invalid diagnostic timing samples")
        bench.support.write_json(path, result)
        entry["validation_after"] = validate()
        device.synchronize()
        entry.update(status="passed", measurement_validated=True)
        bench.support.write_json(path, result)
        write_table(table, diagnostic)
    diagnostic["kernels_after"] = common.freeze(pair)
    if diagnostic["kernels_after"] != frozen:
        raise RuntimeError("Fixed diagnostic kernels changed during timing")
    diagnostic["status"] = "passed"
    bench.support.write_json(path, result)
