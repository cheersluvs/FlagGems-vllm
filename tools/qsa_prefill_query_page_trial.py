# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Combine BF16 page fetch and joint query modes against the exact production source."""

import hashlib
import types
from pathlib import Path

PRODUCTION_SHA256 = "f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c"


def candidate_module(production):
    from flaggems_vllm.utils import libtuner
    from tools import qsa_prefill_query_page_kernel as kernel
    from tools.qsa_prefill_query_joint_policy import prune_configs

    if (
        hashlib.sha256(Path(production.__file__).read_bytes()).hexdigest()
        != PRODUCTION_SHA256
    ):
        raise RuntimeError("Query-page trial baseline source changed")
    suffix = ".query_page_trial"
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
            configs=list(entry.fn.configs),
            key=list(entry.fn.keys),
            strategy="default",
            prune_configs_by=(
                {"early_config_prune": prune_configs}
                if name == "_qsa_prefill_score_kernel"
                else None
            ),
        )(entry.fn.fn)
        cloned = type(entry)(tuner)
        baseline = getattr(production, name)
        if cloned is baseline or any(
            a is b for a, b in zip(cloned.kernel_cache, baseline.kernel_cache)
        ):
            raise RuntimeError("Query-page trial shares a production cache")
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
        raise RuntimeError("Query-page trial changed TopK")
    return module


def acceptance(summary):
    from tools import qsa_prefill_query_joint_trial as joint

    # Reuse all existing same-run main/fixed/control gates, without relabeling
    # the stored report or permitting this private candidate as production.
    view = dict(summary, manifest=dict(summary.get("manifest", {})))
    options = dict(view["manifest"].get("options", {}))
    valid = options.get("candidate") == "query-joint-page"
    if valid:
        options["candidate"] = "query-joint"
    view["manifest"]["options"] = options
    result = joint.acceptance(view)
    result["candidate"] = "query-joint-page"
    reasons = list(result.get("reasons", []))
    if not valid:
        reasons.append("not the combined query/page trial")
    for case in summary.get("cases", []):
        metadata = case.get("query_page", {})
        if (
            metadata.get("baseline_source_sha256") != PRODUCTION_SHA256
            or metadata.get("page_gate") != "not IS_FP8 and BLOCK_N<=PAGE_SIZE"
            or metadata.get("configs") != case.get("production_score_configs")
            or metadata.get("same_topk") is not True
        ):
            reasons.append("invalid combined query/page metadata")
    if reasons:
        result.update(measurement_valid=False, target_met=None, reasons=reasons)
        result.pop("stages", None)
    return result


def validate_full_coverage(records):
    from tools import qsa_prefill_accuracy_profiles as profiles

    selected = {}
    prefix = profiles.QUERY_PAGE_FILE + "::"
    for node, record in records.items():
        if node.startswith(prefix):
            normalized = node
        elif "/" + prefix in node:
            normalized = node[node.index("/" + prefix) + 1 :]
        else:
            continue
        if normalized in selected:
            return "Duplicate combined correctness node: " + normalized
        selected[normalized] = record
    required = {
        prefix + case
        for case in (*profiles.QUERY_SPLIT_CASES, *profiles.PAGE_TABLE_CASES)
    }
    if required - selected.keys():
        return "Missing forced mode/page boundary combined correctness nodes"
    expected = dict(
        test_query_split_score=4,
        test_query_split_full_entry=2,
        test_page_table_score=6,
        test_page_table_full_entry=2,
        test_combined_rectangular_every_config=8,
        test_combined_compact_every_config=6,
        test_combined_full_entry=12,
    )
    counts = {}
    for node, record in selected.items():
        if record.get("result") != "passed":
            return "Combined correctness node did not pass: " + node
        family = node.split("::", 1)[1].split("[", 1)[0]
        counts[family] = counts.get(family, 0) + 1
    if counts != expected:
        return "Incomplete combined full correctness families: " + repr(counts)
    return None
