# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Isolate FP8 query subtiles against the exact current production operator."""

import hashlib
import math
import types
from pathlib import Path

PRODUCTION_SHA256 = "f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c"


def candidate_module(production):
    from flaggems_vllm.utils import libtuner
    from tools import qsa_prefill_query_split_kernel as kernel

    if (
        hashlib.sha256(Path(production.__file__).read_bytes()).hexdigest()
        != PRODUCTION_SHA256
    ):
        raise RuntimeError("Query-split trial baseline source changed")
    suffix = ".query_split_trial"
    module = types.ModuleType(production.__name__ + suffix)
    module.__dict__.update(vars(production))
    module.__name__ = production.__name__ + suffix
    module.__file__ = kernel.__file__
    module.production_baseline = production
    for name, entry in (
        ("_qsa_prefill_score_kernel", kernel._qsa_prefill_score_kernel),
        (
            "_qsa_prefill_compact_score_kernel",
            production._qsa_prefill_compact_score_kernel,
        ),
    ):
        tuner = libtuner(
            configs=list(entry.fn.configs), key=list(entry.fn.keys), strategy="default"
        )(entry.fn.fn)
        cloned = type(entry)(tuner)
        baseline = getattr(production, name)
        if cloned is baseline or any(
            a is b for a, b in zip(cloned.kernel_cache, baseline.kernel_cache)
        ):
            raise RuntimeError("Query-split trial shares a production cache")
        setattr(module, name, cloned)
    for name in ("_prefill_logits", "qsa_select_paged_prefill"):
        original = getattr(production, name)
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
        raise RuntimeError("Query-split trial changed TopK")
    return module


def acceptance(summary):
    from tools import qsa_prefill_packed_grid as current

    result = current.acceptance(summary, expected_candidate="query-split")
    reasons = list(result.get("reasons", []))
    for case in summary.get("cases", []):
        traces = case.get("cache_granularity", {}).get("production_baseline", [])
        eligible = case.get("case", {}).get("dtype") == "float8_e4m3fn" and any(
            t["route"] == "rectangular" and t["constants"]["TILE_R"] >= 32
            for t in traces
        )
        entry = case.get("stages", {}).get("score_matched", {})
        if eligible:
            a, b, ratio = (
                entry.get("baseline_us", {}).get("median", 0),
                entry.get("candidate_us", {}).get("median", 0),
                entry.get("speedup", 0),
            )
            if not (
                entry.get("status") == "passed"
                and entry.get("accepted")
                and entry.get("measurement_validated")
                and entry.get("baseline_kind") == "production_fixed_config"
                and not entry.get("includes_topk", True)
                and all(
                    isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                    for v in (a, b, ratio)
                )
                and math.isclose(ratio, a / b, rel_tol=1e-9)
                and case.get("query_split", {}).get("matched", {}).get("configs_match")
                is True
            ):
                reasons.append("invalid fixed-config query-split comparison")
    if reasons:
        result.update(measurement_valid=False, target_met=None, reasons=reasons)
        result.pop("stages", None)
    return result
