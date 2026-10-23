# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Round-one old-eight comparison and complete30 acceptance; not production code."""

import hashlib
import math
import statistics
import types
from pathlib import Path

from tools import qsa_prefill_observed_workloads as workloads

PREVIOUS_SOURCE_SHA256 = (
    "6db2918b5b7dc1798244548972458a8c59745f967341a10663e2d704cf35687a"
)
CONFIG_GROUP = "qsa_select_paged_prefill_tuning_round1_score"
PREVIOUS_CONFIG_GROUP = "qsa_select_paged_prefill_reuse_score"
COMPACT_CONFIG_GROUP = "qsa_select_paged_prefill_score"
NEAR_ONE_FLOOR = 0.98
MEAN_TARGET = 1.1


def previous_module(production, rectangular_group=PREVIOUS_CONFIG_GROUP):
    """Same JIT/host/TopK, frozen eight/ten configs, independent standard entries."""
    from flaggems_vllm import runtime
    from flaggems_vllm.utils import libentry, libtuner
    from tools import qsa_prefill_triton_ab as ab
    from tools.qsa_prefill_cache_granularity import round1_source

    if (
        hashlib.sha256(Path(production.__file__).read_bytes()).hexdigest()
        == ab.INTEGRATED_PRODUCTION_SHA256
    ):
        production = ab.pre_store_production()
    try:
        source = round1_source(Path(production.__file__).read_text())
    except RuntimeError as error:
        raise RuntimeError(
            "Round1 must change only the production config-group literal"
        ) from error
    restored = source.replace(CONFIG_GROUP, PREVIOUS_CONFIG_GROUP)
    if (
        source.count(CONFIG_GROUP) != 1
        or hashlib.sha256(restored.encode()).hexdigest() != PREVIOUS_SOURCE_SHA256
    ):
        raise RuntimeError(
            "Round1 must change only the production config-group literal"
        )
    if rectangular_group not in (PREVIOUS_CONFIG_GROUP, CONFIG_GROUP):
        raise ValueError("Unknown rectangular baseline group")
    suffix = ".previous_ten" if rectangular_group == CONFIG_GROUP else ".previous_eight"
    module = types.ModuleType(production.__name__ + suffix)
    module.__dict__.update(vars(production))
    module.__name__ = production.__name__ + suffix
    for name, group, count in (
        (
            "_qsa_prefill_score_kernel",
            rectangular_group,
            10 if rectangular_group == CONFIG_GROUP else 8,
        ),
        ("_qsa_prefill_compact_score_kernel", COMPACT_CONFIG_GROUP, 6),
    ):
        entry = getattr(production, name)
        configs = runtime.get_tuned_config(group)
        if len(configs) != count:
            raise RuntimeError("Previous config group changed: " + group)
        template = entry.fn
        tuner = libtuner(configs=configs, key=list(template.keys), strategy="default")(
            template.fn
        )
        cloned = libentry()(tuner)
        if cloned is entry or any(
            a is b for a, b in zip(cloned.kernel_cache, entry.kernel_cache)
        ):
            raise RuntimeError("Previous tuning baseline shares LibEntry caches")
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
        raise RuntimeError("Round1 must preserve TopK identity")
    return module


def tuning_state(module):
    """Untimed LibTuner evidence, not a per-chunk performance trace."""
    result = {}
    names = ["_qsa_prefill_score_kernel", "_qsa_prefill_compact_score_kernel"]
    if hasattr(module, "_qsa_prefill_fp8_store_score_kernel"):
        names.append("_qsa_prefill_fp8_store_score_kernel")
    for name in names:
        tuner = getattr(module, name).fn
        scans = []
        for config, values in getattr(tuner, "configs_timings", {}).items():
            finite = all(math.isfinite(v) for v in values)
            scans.append(
                dict(
                    config=config.all_kwargs(),
                    latency_quantiles_ms=list(values) if finite else None,
                    status="measured" if finite else "unavailable_or_resource_error",
                )
            )
        result[name] = dict(
            configs=[c.all_kwargs() for c in tuner.configs],
            key_names=list(tuner.keys),
            last_tuner_selection=(
                tuner.best_config.all_kwargs()
                if getattr(tuner, "best_config", None) is not None
                else None
            ),
            last_tuning_scan=scans,
            scan_scope=(
                "Last LibTuner tuning event only, outside acceptance timing. "
                "LibEntry cache hits may reuse a config across chunks. "
                "No scan means no new tuning event, not zero latency."
            ),
        )
    return result


def acceptance(summary):
    """Never average a partial/changed/shared set into a complete30 result."""
    options = summary.get("manifest", {}).get("options", {})
    result = dict(
        applicable=options.get("suite") == "scored-prefill",
        primary_stage="pipeline",
        expected_cases=30,
        aggregation="equal_weight_arithmetic_mean_of_native_pipeline_speedups",
        proposed_near_one_floor=NEAR_ONE_FLOOR,
        arithmetic_mean_target_exclusive=MEAN_TARGET,
        measurement_valid=False,
        target_met=None,
    )
    if not result["applicable"]:
        return result
    expected = {c["name"]: c for c in workloads.cases("scored-prefill")}
    planned = summary.get("manifest", {}).get("cases", [])
    cases = summary.get("cases", [])
    reasons = []
    for label, entries in (
        ("planned", planned),
        ("measured", [c.get("case", {}) for c in cases]),
    ):
        if len(entries) != 30 or {c.get("name") for c in entries} != set(expected):
            reasons.append(label + " set is not exactly the frozen30")
        elif any(c != expected[c["name"]] for c in entries):
            reasons.append(label + " fixture definition changed")
    if (
        summary.get("status") != "passed"
        or summary.get("accuracy", {}).get("status") != "passed"
    ):
        reasons.append("run or full accuracy incomplete")
    if (
        options.get("candidate") != "production"
        or options.get("compiler") != "flagtree"
        or options.get("mode") != "kernel"
    ):
        reasons.append("requires formal production and same-FlagTree kernel profile")
    if (
        options.get("shared_gpu_exploratory")
        or summary.get("acceptance_eligible") is False
    ):
        reasons.append("shared GPU exploratory measurements are ineligible")
    stages = ["score", "topk", "pipeline"]
    if options.get("compare_previous_tuning"):
        stages += ["score_vs_previous_tuning", "pipeline_vs_previous_tuning"]
    for c in cases:
        for name in stages:
            entry = c.get("stages", {}).get(name, {})
            if c.get("status") != "passed" or not all(
                (
                    entry.get("status") == "passed",
                    entry.get("accepted"),
                    entry.get("measurement_validated"),
                )
            ):
                reasons.append(
                    "invalid stage: " + c.get("case", {}).get("name", "?") + ":" + name
                )
                continue
            a = entry.get("baseline_us", {}).get("median", 0)
            b = entry.get("candidate_us", {}).get("median", 0)
            ratio = entry.get("speedup", 0)
            if not all(
                isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                for x in (a, b, ratio)
            ) or not math.isclose(ratio, a / b, rel_tol=1e-9):
                reasons.append("invalid latency/speedup: " + name)
    result["reasons"] = reasons
    if reasons:
        return result
    result["measurement_valid"] = True
    result["stages"] = {}
    for name in stages:
        values = [c["stages"][name]["speedup"] for c in cases]
        result["stages"][name] = dict(
            count=30,
            arithmetic_mean=statistics.mean(values),
            minimum=min(values),
            below_one=[
                c["case"]["name"] for c in cases if c["stages"][name]["speedup"] < 1
            ],
            below_near_one_floor=[
                c["case"]["name"]
                for c in cases
                if c["stages"][name]["speedup"] < NEAR_ONE_FLOOR
            ],
        )
    primary = result["stages"]["pipeline"]
    result["target_met"] = (
        primary["arithmetic_mean"] > MEAN_TARGET
        and primary["minimum"] >= NEAR_ONE_FLOOR
    )
    return result
