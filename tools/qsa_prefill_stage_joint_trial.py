# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Jointly tune the two audited pipeline depths on the exact combined scorer."""

import hashlib
from pathlib import Path

from tools import qsa_prefill_query_page_trial as combined

CONFIG_GROUP = "qsa_select_paged_prefill_stage_joint_score"
KERNEL_SHA256 = "1ef12d431b65971c5617f0a34f614360008c652a36125d5679da00206b212d7c"
PRODUCTION_SHA256 = combined.PRODUCTION_SHA256
BASELINE_KIND = "combined_query_page_stage2"


def validate_values(original, choices):
    if len(original) != 17 or any(c.get("num_stages") != 2 for c in original):
        raise RuntimeError("Stage-joint trial requires the original17 stage2 choices")
    expected = [dict(c) for c in original] + [dict(c, num_stages=1) for c in original]
    if choices != expected:
        raise RuntimeError(
            "Stage-joint choices must be original stage2 plus paired stage1"
        )


def validate_configs(original, candidate):
    validate_values(
        [dict(c.all_kwargs()) for c in original],
        [dict(c.all_kwargs()) for c in candidate],
    )


def candidate_module(production):
    from flaggems_vllm import runtime
    from flaggems_vllm.utils import libtuner
    from tools.qsa_prefill_query_joint_policy import prune_configs

    module = combined.candidate_module(production)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != KERNEL_SHA256:
        raise RuntimeError("Combined score source changed")
    module.__name__ += ".stage_joint"
    entry = module._qsa_prefill_score_kernel
    configs = runtime.get_tuned_config(CONFIG_GROUP)
    validate_configs(entry.fn.configs, configs)
    tuner = libtuner(
        configs=configs,
        key=list(entry.fn.keys),
        strategy="default",
        prune_configs_by={"early_config_prune": prune_configs},
    )(entry.fn.fn)
    module._qsa_prefill_score_kernel = type(entry)(tuner)
    if any(
        a is b
        for a, b in zip(
            module._qsa_prefill_score_kernel.kernel_cache,
            production._qsa_prefill_score_kernel.kernel_cache,
        )
    ):
        raise RuntimeError("Stage-joint trial shares canonical caches")
    module.stage_joint_choices = [dict(c.all_kwargs()) for c in configs]
    return module


def metadata(module, configs):
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != KERNEL_SHA256:
        raise RuntimeError("Stage-joint kernel source changed")
    choices = [
        dict(c.all_kwargs()) for c in module._qsa_prefill_score_kernel.fn.configs
    ]
    if any(c.get("constexprs") not in choices for c in configs):
        raise RuntimeError("Rectangular score compiled outside the joint choice set")
    if choices != module.stage_joint_choices:
        raise RuntimeError("Stage-joint choices changed after preparation")
    return dict(
        config_group=CONFIG_GROUP,
        kernel_sha256=KERNEL_SHA256,
        factory_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        num_stages_choices=[2, 1],
        choices=choices,
        configs=configs,
        same_topk=module.persistent_topk is module.production_baseline.persistent_topk,
    )


def acceptance(summary):
    view = dict(summary, manifest=dict(summary.get("manifest", {})))
    options = dict(view["manifest"].get("options", {}))
    valid = options.get("candidate") == "query-stage-joint"
    if valid:
        options["candidate"] = "query-joint-page"
    view["manifest"]["options"] = options
    result = combined.acceptance(view)
    result["candidate"] = "query-stage-joint"
    reasons = list(result.get("reasons", []))
    if (
        not valid
        or not options.get("compare_previous_tuning")
        or options.get("previous_tuning") != "combined"
    ):
        reasons.append(
            "requires joint-stage trial and same-run combined stage2 baseline"
        )
    for case in summary.get("cases", []):
        m = case.get("stage_joint", {})
        previous = case.get("previous_tuning", {})
        if (
            m.get("config_group") != CONFIG_GROUP
            or m.get("kernel_sha256") != KERNEL_SHA256
            or m.get("num_stages_choices") != [2, 1]
            or m.get("same_topk") is not True
            or m.get("configs") != case.get("production_score_configs")
            or len(m.get("choices", [])) != 34
            or any(
                v.get("constexprs") not in m.get("choices", [])
                for v in case.get("production_score_configs", [])
            )
            or previous.get("baseline_kind") != BASELINE_KIND
            or previous.get("source_equivalent_sha256") != KERNEL_SHA256
            or previous.get("same_topk") is not True
            or previous.get("config_group")
            != "qsa_select_paged_prefill_query_joint_score"
            or any(
                t.get("constants") not in m.get("choices", [])
                for t in case.get("cache_granularity", {}).get("production", [])
                if t.get("route") == "rectangular"
            )
        ):
            reasons.append(
                "invalid joint-stage/combined stage2 source or config metadata"
            )
        try:
            choices = m.get("choices", [])
            validate_values(choices[:17], choices)
        except RuntimeError:
            reasons.append("invalid paired stage2/stage1 choice space")
        for name in ("score_vs_previous_tuning", "pipeline_vs_previous_tuning"):
            entry = case.get("stages", {}).get(name, {})
            if entry.get("baseline_kind") != BASELINE_KIND or entry.get(
                "includes_topk"
            ) != name.startswith("pipeline"):
                reasons.append("invalid combined baseline label: " + name)
    if reasons:
        result.update(measurement_valid=False, target_met=None, reasons=reasons)
        result.pop("stages", None)
    return result


def validate_full_coverage(records):
    from tools import qsa_prefill_accuracy_profiles as profiles

    translated = {
        key.replace(profiles.STAGE_JOINT_FILE, profiles.QUERY_PAGE_FILE): value
        for key, value in records.items()
        if profiles.STAGE_JOINT_FILE + "::" in key
    }
    if len(translated) != sum(
        profiles.STAGE_JOINT_FILE + "::" in key for key in records
    ):
        return "Duplicate joint-stage correctness node"
    return combined.validate_full_coverage(translated)
