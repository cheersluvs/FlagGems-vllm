# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Frozen iteration correctness nodes; full-suite execution stays available."""

PRODUCTION_FILE = "tests/test_qsa_select_paged_prefill.py"
PACKED_FILE = "tests/test_qsa_prefill_packed_grid.py"

PRODUCTION_CASES = (
    "test_visible_counts_and_ties[16]",
    "test_visible_counts_and_ties[64]",
    "test_relu_precedes_head_sum_and_score_is_unscaled",
    "test_native_vllm_topk_dispatch[dtype0-1-4096-1024]",
    "test_native_vllm_topk_dispatch[dtype0-1-8192-2048]",
    "test_native_vllm_topk_dispatch[dtype0-65-8192-512]",
    "test_native_vllm_topk_dispatch[dtype0-65-8192-1024]",
    "test_native_vllm_topk_dispatch[dtype0-65-8192-2048]",
    "test_native_vllm_topk_dispatch[dtype1-1-4096-1024]",
    "test_native_vllm_topk_dispatch[dtype1-1-8192-2048]",
    "test_native_vllm_topk_dispatch[dtype1-65-8192-512]",
    "test_native_vllm_topk_dispatch[dtype1-65-8192-1024]",
    "test_native_vllm_topk_dispatch[dtype1-65-8192-2048]",
    "test_native_vllm_empty_and_select_all[dtype0]",
    "test_native_vllm_empty_and_select_all[dtype1]",
    "test_workspace_reuse_and_cross_request_chunks",
    "test_against_pinned_vllm",
    "test_upstream_prefill_against_vllm[dtype0-False-0]",
    "test_upstream_prefill_against_vllm[dtype0-False-1792]",
    "test_upstream_prefill_against_vllm[dtype0-True-0]",
    "test_upstream_prefill_against_vllm[dtype0-True-1792]",
    "test_upstream_prefill_against_vllm[dtype1-False-0]",
    "test_upstream_prefill_against_vllm[dtype1-False-1792]",
    "test_upstream_prefill_against_vllm[dtype1-True-0]",
    "test_upstream_prefill_against_vllm[dtype1-True-1792]",
    "test_upstream_split_prefill[torch-1]",
    "test_upstream_split_prefill[torch-2]",
    "test_upstream_split_prefill[torch-3]",
    "test_upstream_split_prefill[torch-4]",
    "test_upstream_split_prefill[vllm-1]",
    "test_upstream_split_prefill[vllm-2]",
    "test_upstream_split_prefill[vllm-3]",
    "test_upstream_split_prefill[vllm-4]",
    "test_upstream_no_complete_blocks_dim8[torch]",
    "test_upstream_no_complete_blocks_dim8[vllm]",
    "test_integrated_public_entry_identity",
    "test_integrated_observed_against_native[observed_ragged54_bfloat16]",
    "test_fp8_benchmark_against_native[fp8_compact_float8_e4m3fn]",
    "test_score_cache_separates_runtime_rows[bf16]",
    "test_score_cache_separates_runtime_rows[fp8]",
)

PACKED_CASES = (
    "test_packed_score_boundaries[lengths0-37-64-dtype0]",
    "test_packed_score_boundaries[lengths0-37-64-dtype1]",
    "test_packed_score_boundaries[lengths1-29-36-dtype0]",
    "test_packed_score_boundaries[lengths1-29-36-dtype1]",
    "test_packed_score_boundaries[lengths2-31-1-dtype0]",
    "test_packed_score_boundaries[lengths2-31-1-dtype1]",
    "test_packed_full_entry[dtype0]",
    "test_packed_full_entry[dtype1]",
)


def name(options):
    return getattr(options, "accuracy_profile", None) or "full"


def expected_nodes(candidate):
    if candidate not in ("production", "packed-grid"):
        raise ValueError(
            "Iteration profile supports current production/packed-grid only"
        )
    nodes = tuple(PRODUCTION_FILE + "::" + case for case in PRODUCTION_CASES)
    if candidate == "packed-grid":
        nodes = tuple(PACKED_FILE + "::" + case for case in PACKED_CASES) + nodes
    return nodes


def targets(root, candidate):
    return [
        str(root / node.split("::")[0]) + "::" + node.split("::", 1)[1]
        for node in expected_nodes(candidate)
    ]


def pytest_targets(options, root):
    candidate = getattr(options, "candidate", "production")
    if name(options) == "iteration":
        return targets(root, candidate)
    files = [PRODUCTION_FILE]
    if candidate == "packed-grid":
        files.insert(0, PACKED_FILE)
    if candidate not in ("production", "packed-grid"):
        files.append("tests/test_qsa_prefill_compact_trial.py")
    if candidate in ("key-major", "key-major-reuse", "key-major-evict-first"):
        files.append("tests/test_qsa_prefill_key_major_trial.py")
    if candidate in ("key-major-reuse", "key-major-evict-first"):
        files.append("tests/test_qsa_prefill_reuse_trial.py")
    if candidate == "key-major-evict-first":
        files.append("tests/test_qsa_prefill_eviction_trial.py")
    return [str(root / path) for path in files]


def validate(records, candidate):
    expected = set(expected_nodes(candidate))
    normalized = {}
    for node, record in records.items():
        matches = [key for key in expected if node == key or node.endswith("/" + key)]
        if len(matches) != 1 or matches[0] in normalized:
            return "Unexpected or duplicate iteration correctness node: " + node
        normalized[matches[0]] = record
    missing = expected - normalized.keys()
    if missing:
        return "Missing iteration correctness nodes: " + ", ".join(sorted(missing))
    failed = [
        node for node, record in normalized.items() if record.get("result") != "passed"
    ]
    if failed:
        return "Iteration correctness nodes did not pass: " + ", ".join(failed)
    return None


def annotate_acceptance(result, profile, candidate, accuracy):
    full = profile == "full" and accuracy.get("status") == "passed"
    result.update(
        accuracy_profile=profile,
        full_suite_validated=full,
        formal_acceptance_eligible=(
            full
            and candidate == "production"
            and result.get("measurement_valid", False)
        ),
    )
    return result
