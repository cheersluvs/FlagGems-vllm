# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Change only launch pipeline depth on the audited combined score kernel."""

import hashlib
from pathlib import Path

from tools import qsa_prefill_query_page_trial as combined

CONFIG_GROUP = "qsa_select_paged_prefill_loop_stage_score"
KERNEL_SHA256 = "1ef12d431b65971c5617f0a34f614360008c652a36125d5679da00206b212d7c"
PRODUCTION_SHA256 = combined.PRODUCTION_SHA256
BASELINE_KIND = "combined_query_page_stage2"


def validate_configs(original, candidate):
    if len(original) != 17 or len(candidate) != 17:
        raise RuntimeError("Loop-stage trial requires the original17 choices")
    for old, new in zip(original, candidate):
        a, b = dict(old.all_kwargs()), dict(new.all_kwargs())
        if a.pop("num_stages") != 2 or b.pop("num_stages") != 1 or a != b:
            raise RuntimeError("Loop-stage trial may change only num_stages2->1")


def candidate_module(production):
    from flaggems_vllm import runtime
    from flaggems_vllm.utils import libtuner
    from tools.qsa_prefill_query_joint_policy import prune_configs

    module = combined.candidate_module(production)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != KERNEL_SHA256:
        raise RuntimeError("Combined score source changed")
    module.__name__ += ".loop_stage1"
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
        raise RuntimeError("Loop-stage trial shares canonical caches")
    module.loop_stage_choices = [dict(c.all_kwargs()) for c in configs]
    return module


def metadata(module, configs):
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != KERNEL_SHA256:
        raise RuntimeError("Loop-stage kernel source changed")
    choices = [
        dict(c.all_kwargs()) for c in module._qsa_prefill_score_kernel.fn.configs
    ]
    if any(c.get("constexprs", {}).get("num_stages") != 1 for c in configs):
        raise RuntimeError("Rectangular score did not compile with stage1")
    if choices != module.loop_stage_choices:
        raise RuntimeError("Loop-stage choices changed after preparation")
    return dict(
        config_group=CONFIG_GROUP,
        kernel_sha256=KERNEL_SHA256,
        factory_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        num_stages=1,
        choices=choices,
        configs=configs,
        same_topk=module.persistent_topk is module.production_baseline.persistent_topk,
    )


def acceptance(summary):
    view = dict(summary, manifest=dict(summary.get("manifest", {})))
    options = dict(view["manifest"].get("options", {}))
    valid = options.get("candidate") == "query-loop-stage"
    if valid:
        options["candidate"] = "query-joint-page"
    view["manifest"]["options"] = options
    result = combined.acceptance(view)
    result["candidate"] = "query-loop-stage"
    reasons = list(result.get("reasons", []))
    if (
        not valid
        or not options.get("compare_previous_tuning")
        or options.get("previous_tuning") != "combined"
    ):
        reasons.append("requires stage1 trial and same-run combined stage2 baseline")
    for case in summary.get("cases", []):
        m = case.get("loop_stage", {})
        previous = case.get("previous_tuning", {})
        if (
            m.get("config_group") != CONFIG_GROUP
            or m.get("kernel_sha256") != KERNEL_SHA256
            or m.get("num_stages") != 1
            or m.get("same_topk") is not True
            or m.get("configs") != case.get("production_score_configs")
            or len(m.get("choices", [])) != 17
            or any(c.get("num_stages") != 1 for c in m.get("choices", []))
            or previous.get("baseline_kind") != BASELINE_KIND
            or previous.get("source_equivalent_sha256") != KERNEL_SHA256
            or previous.get("same_topk") is not True
            or previous.get("config_group")
            != "qsa_select_paged_prefill_query_joint_score"
            or any(
                t.get("constants", {}).get("num_stages") != 1
                for t in case.get("cache_granularity", {}).get("production", [])
                if t.get("route") == "rectangular"
            )
        ):
            reasons.append("invalid stage1/combined stage2 source or config metadata")
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
        key.replace(profiles.LOOP_STAGE_FILE, profiles.QUERY_PAGE_FILE): value
        for key, value in records.items()
        if profiles.LOOP_STAGE_FILE + "::" in key
    }
    if len(translated) != sum(
        profiles.LOOP_STAGE_FILE + "::" in key for key in records
    ):
        return "Duplicate stage1 correctness node"
    return combined.validate_full_coverage(translated)
