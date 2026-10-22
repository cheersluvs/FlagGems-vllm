# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Frozen iteration correctness nodes; full-suite execution stays available."""

PRODUCTION_FILE = "tests/test_qsa_select_paged_prefill.py"
PACKED_FILE = "tests/test_qsa_prefill_packed_grid.py"
PAGE_TABLE_FILE = "tests/test_qsa_prefill_page_table.py"
PAGE_TABLE_BF16_FILE = "tests/test_qsa_prefill_page_table_bf16.py"
PAGE_TABLE_CANDIDATES = ("page-table", "page-table-bf16")
QUERY_SPLIT_FILE = "tests/test_qsa_prefill_query_split.py"
QUERY_JOINT_FILE = "tests/test_qsa_prefill_query_joint.py"
QUERY_PAGE_FILE = "tests/test_qsa_prefill_query_page.py"
LOOP_STAGE_FILE = "tests/test_qsa_prefill_loop_stage.py"
CURRENT_CANDIDATES = (
    "packed-grid",
    "query-split",
    "query-joint",
    "query-joint-page",
    "query-loop-stage",
    *PAGE_TABLE_CANDIDATES,
)

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


PAGE_TABLE_CASES = (
    "test_page_table_score[page16-bf16]",
    "test_page_table_score[page16-fp8]",
    "test_page_table_score[page64-bf16]",
    "test_page_table_score[page64-fp8]",
    "test_page_table_score[page196-bf16]",
    "test_page_table_score[page196-fp8]",
    "test_page_table_full_entry[bf16]",
    "test_page_table_full_entry[fp8]",
)


QUERY_SPLIT_CASES = (
    "test_query_split_score[bf16-h4d128]",
    "test_query_split_score[fp8-h4d128]",
    "test_query_split_score[fp8-h1d32]",
    "test_query_split_score[fp8-h3d96]",
    "test_query_split_full_entry[bf16]",
    "test_query_split_full_entry[fp8]",
)


def name(options):
    return getattr(options, "accuracy_profile", None) or "full"


def expected_nodes(candidate):
    if candidate not in ("production", *CURRENT_CANDIDATES):
        raise ValueError(
            "Iteration profile supports production/current score trials only"
        )
    nodes = tuple(PRODUCTION_FILE + "::" + case for case in PRODUCTION_CASES)
    if candidate == "packed-grid":
        nodes = tuple(PACKED_FILE + "::" + case for case in PACKED_CASES) + nodes
    if candidate == "query-loop-stage":
        extra = tuple(
            f"test_combined_rectangular_every_config[{heads}-{dim}-dtype{dtype}]"
            for heads, dim in ((4, 128), (3, 24), (5, 96), (1, 8))
            for dtype in (0, 1)
        )
        nodes = (
            tuple(
                LOOP_STAGE_FILE + "::" + case
                for case in (*QUERY_SPLIT_CASES, *PAGE_TABLE_CASES, *extra)
            )
            + nodes
        )
    if candidate == "query-joint-page":
        nodes = (
            tuple(
                QUERY_PAGE_FILE + "::" + case
                for case in (*QUERY_SPLIT_CASES, *PAGE_TABLE_CASES)
            )
            + nodes
        )
    if candidate == "query-joint":
        nodes = (
            tuple(QUERY_JOINT_FILE + "::" + case for case in QUERY_SPLIT_CASES) + nodes
        )
    if candidate == "query-split":
        nodes = (
            tuple(QUERY_SPLIT_FILE + "::" + case for case in QUERY_SPLIT_CASES) + nodes
        )
    if candidate in PAGE_TABLE_CANDIDATES:
        file = (
            PAGE_TABLE_BF16_FILE if candidate == "page-table-bf16" else PAGE_TABLE_FILE
        )
        nodes = tuple(file + "::" + case for case in PAGE_TABLE_CASES) + nodes
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
    if candidate == "query-loop-stage":
        files.insert(0, LOOP_STAGE_FILE)
    if candidate == "query-joint-page":
        files.insert(0, QUERY_PAGE_FILE)
    if candidate == "query-joint":
        files.insert(0, QUERY_JOINT_FILE)
    if candidate == "query-split":
        files.insert(0, QUERY_SPLIT_FILE)
    if candidate == "packed-grid":
        files.insert(0, PACKED_FILE)
    if candidate in PAGE_TABLE_CANDIDATES:
        files.insert(
            0,
            PAGE_TABLE_BF16_FILE if candidate == "page-table-bf16" else PAGE_TABLE_FILE,
        )
    if candidate not in ("production", *CURRENT_CANDIDATES):
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
