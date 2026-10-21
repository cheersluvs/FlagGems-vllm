# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU-safe policy/diagnostics and isolated current-production trial factory."""

import hashlib
import math
import statistics
import types
from pathlib import Path

PRODUCTION_SHA256 = "f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c"
MAX_TILE_R = 64


def eligible(requests, max_query_len, rows):
    return (
        requests > 1
        and requests * min(rows, max_query_len) >= 2 * rows
        and requests * ((min(rows, max_query_len) + MAX_TILE_R - 1) // MAX_TILE_R)
        > rows // MAX_TILE_R + requests
    )


def candidate_module(production):
    from flaggems_vllm.utils import libtuner
    from tools import qsa_prefill_packed_grid_kernel as kernel

    if (
        hashlib.sha256(Path(production.__file__).read_bytes()).hexdigest()
        != PRODUCTION_SHA256
    ):
        raise RuntimeError("Packed-grid baseline production source changed")
    module = types.ModuleType(production.__name__ + ".packed_grid_trial")
    module.__dict__.update(vars(production))
    module.__name__ = production.__name__ + ".packed_grid_trial"
    module.__file__ = kernel.__file__
    module.production_baseline = production
    module._packed_eligible = eligible

    def isolate(entry):
        tuner = libtuner(
            configs=list(entry.fn.configs), key=list(entry.fn.keys), strategy="default"
        )(entry.fn.fn)
        cloned = type(entry)(tuner)
        if cloned is entry or any(
            a is b for a, b in zip(cloned.kernel_cache, entry.kernel_cache)
        ):
            raise RuntimeError("Packed-grid trial shares a baseline cache")
        return cloned

    for name in ("_qsa_prefill_score_kernel", "_qsa_prefill_compact_score_kernel"):
        setattr(module, name, isolate(getattr(production, name)))
    module._qsa_prefill_packed_score_kernel = isolate(
        kernel._qsa_prefill_packed_score_kernel
    )
    for name, original in (
        ("_prefill_logits", kernel._prefill_logits),
        ("qsa_select_paged_prefill", production.qsa_select_paged_prefill),
    ):
        cloned = types.FunctionType(
            original.__code__,
            vars(module),
            name,
            original.__defaults__,
            original.__closure__,
        )
        cloned.__kwdefaults__ = original.__kwdefaults__
        setattr(module, name, cloned)
    if module.persistent_topk is not production.persistent_topk:
        raise RuntimeError("Packed-grid trial changed TopK")
    return module


def diagnostics(args, traces, query_lengths):
    requests, maximum = args[2].shape[0], args[7]
    starts = [0]
    for length in query_lengths:
        starts.append(starts[-1] + length)
    if len(query_lengths) != requests or starts[-1] != args[0].shape[0]:
        raise RuntimeError("Packed-grid diagnostic fixture geometry changed")
    result = []
    for t in traces:
        tile = t["constants"]["TILE_R"]
        rectangular = requests * ((min(t["rows"], maximum) + tile - 1) // tile)
        packed = t["rows"] // tile + requests
        useful = sum(
            (
                max(0, min(end, t["offset"] + t["rows"]) - max(begin, t["offset"]))
                + tile
                - 1
            )
            // tile
            for begin, end in zip(starts, starts[1:])
        )
        issued = packed if t["route"] in ("packed", "compact") else rectangular
        key_groups = (
            t["width"] + t["constants"]["BLOCK_N"] * t["constants"]["K_TILES"] - 1
        ) // (t["constants"]["BLOCK_N"] * t["constants"]["K_TILES"])
        result.append(
            dict(
                offset=t["offset"],
                rows=t["rows"],
                route=t["route"],
                TILE_R=tile,
                rectangular_query_slots=rectangular,
                packed_query_slots=packed,
                useful_query_slots=useful,
                issued_query_slots=issued,
                empty_query_slots=issued - useful,
                issued_programs=issued * key_groups,
                scope="Query-boundary occupancy only; visible-key early exits are not counted",
            )
        )
    return result


def acceptance(summary, expected_candidate="packed-grid"):
    """Experimental objective, never labeled formal production acceptance."""
    from tools import qsa_prefill_tuning_round1 as goals

    view = dict(summary, manifest=dict(summary.get("manifest", {})))
    options = dict(view["manifest"].get("options", {}))
    valid_candidate = options.get("candidate") == expected_candidate
    options["candidate"] = "production" if valid_candidate else options.get("candidate")
    view["manifest"]["options"] = options
    result = goals.acceptance(view)
    result.update(
        experimental=True,
        formal_production_acceptance=False,
        candidate=expected_candidate,
    )
    reasons = list(result.get("reasons", []))
    if not valid_candidate:
        reasons.append("not the " + expected_candidate + " trial")
    for c in summary.get("cases", []):
        for name in ("score_vs_production", "pipeline_vs_production"):
            e = c.get("stages", {}).get(name, {})
            a, b, r = (
                e.get("baseline_us", {}).get("median", 0),
                e.get("candidate_us", {}).get("median", 0),
                e.get("speedup", 0),
            )
            if not (
                e.get("status") == "passed"
                and e.get("accepted")
                and e.get("measurement_validated")
                and all(
                    isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                    for v in (a, b, r)
                )
                and math.isclose(r, a / b, rel_tol=1e-9)
            ):
                reasons.append("invalid current-production comparison: " + name)
        if (
            c.get("runtime", {}).get("operator_binding", {}).get("canonical_sha256")
            != PRODUCTION_SHA256
        ):
            reasons.append("baseline source pin mismatch")
    if reasons:
        result.update(measurement_valid=False, target_met=None, reasons=reasons)
        result.pop("stages", None)
    elif result.get("measurement_valid"):
        for name in ("score_vs_production", "pipeline_vs_production"):
            values = [c["stages"][name]["speedup"] for c in summary["cases"]]
            result["stages"][name] = dict(
                count=len(values),
                arithmetic_mean=statistics.mean(values),
                minimum=min(values),
            )
    return result
