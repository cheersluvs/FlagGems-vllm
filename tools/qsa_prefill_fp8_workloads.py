# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Explicit FP8 fixtures: eight recorded envelopes and two dtype variants."""

from tools import qsa_prefill_inference_workloads as inference

UPLOAD_COMMIT = "9937ec6e910db658ee2871ac1743eead0698b7c4"
ARCHIVE_SHA256 = "f2d4ae225613f20b9625dc4e5f0b5279815e11064eb55fbb23c5a4441db05fcc"
SOURCE = (
    f"https://github.com/cheersluvs/FlagGems-vllm/tree/{UPLOAD_COMMIT}/"
    "results/qsa_select_paged_prefill/inference-shapes/20261022-070202"
)
CAPTURE_FILE = "Qwen3.8-Flash-Next-FP8-p32768d1024.txt"


def cases():
    definitions = (
        ("first_chunk", [16384], 16384, 48, 378),
        ("budget_exact", [16384], 32768, 48, 379),
        ("dominant_long", [16331], 33842, 432, 257),
        ("tail112", [16368], 32853, 48, 346),
        ("tail113", [16369], 32851, 48, 348),
        ("two_balanced", [10371, 5958], 32932, 48, 260),
        ("two_skewed", [16330, 54], 32822, 48, 380),
        ("short_tail", [2316], 33843, 48, 382),
    )
    result = []
    for name, queries, maximum, count, line in definitions:
        result.append(
            dict(
                name=f"fp8_{name}_float8_e4m3fn",
                kind="fp8_perf",
                origin="recorded_envelope",
                query_lengths=queries,
                prefix_tokens=[maximum - q for q in queries],
                batch_max_seq_len=maximum,
                recorded_shape=[
                    sum(queries),
                    len(queries),
                    4,
                    128,
                    max(queries),
                    maximum,
                ],
                recorded_count=count,
                capture_line=line,
                query_lengths_synthetic=False,
                layout_provenance=(
                    "captured cache/table extents; assumed contiguous strides and shuffled pages"
                ),
                physical_pages=2712,
                table_columns=88,
                cache_page_stride_factor=1,
                captured_parameters=[
                    sum(queries),
                    4,
                    128,
                    2712,
                    196,
                    1,
                    128,
                    len(queries),
                    88,
                    len(queries) + 1,
                    sum(queries),
                    2048,
                    4,
                    max(queries),
                    sum(queries),
                    512,
                    maximum,
                ],
            )
        )
    for name, queries, maximum, variant_of in (
        (
            "compact",
            [3200] + [112] * 38 + [96] * 15,
            4100,
            "observed_ragged54_bfloat16",
        ),
        ("long100k", [16384], 118784, "perf_n1_q16384_p102400_bfloat16"),
    ):
        result.append(
            dict(
                name=f"fp8_{name}_float8_e4m3fn",
                kind="fp8_perf",
                origin="dtype_variant",
                variant_of=variant_of,
                query_lengths=queries,
                prefix_tokens=[maximum - q for q in queries],
                batch_max_seq_len=maximum,
                query_lengths_synthetic=name == "compact",
            )
        )
    return [
        dict(
            case,
            query_base=0,
            page_size=196,
            heads=4,
            dim=128,
            ratio=4,
            k=512,
            budget_mb=512,
            dtype="float8_e4m3fn",
        )
        for case in result
    ]


def functional_cases():
    """Captured select-all batches; never members of a performance suite."""
    result = []
    for requests, query_length, count, line in ((2048, 8, 48, 381), (451, 2, 48, 384)):
        queries = [query_length] * requests
        rows = sum(queries)
        result.append(
            dict(
                name=f"fp8_select_all_n{requests}_float8_e4m3fn",
                kind="fp8_perf",
                origin="recorded_envelope",
                query_lengths=queries,
                prefix_tokens=[0] * requests,
                batch_max_seq_len=query_length,
                query_lengths_synthetic=False,
                query_base=0,
                page_size=196,
                heads=4,
                dim=128,
                ratio=4,
                k=512,
                budget_mb=512,
                dtype="float8_e4m3fn",
                physical_pages=2712,
                table_columns=88,
                cache_page_stride_factor=1,
                recorded_shape=[rows, requests, 4, 128, query_length, query_length],
                captured_parameters=[
                    rows,
                    4,
                    128,
                    2712,
                    196,
                    1,
                    128,
                    requests,
                    88,
                    requests + 1,
                    rows,
                    2048,
                    4,
                    query_length,
                    rows,
                    512,
                    query_length,
                ],
                recorded_count=count,
                capture_line=line,
            )
        )
    return result


def geometry(case):
    info = inference.geometry(case)
    recorded = case["origin"] == "recorded_envelope"
    info.update(
        source=(
            SOURCE
            if recorded
            else "existing benchmark dtype variant: " + case["variant_of"]
        ),
        provenance=(
            "recorded17-field parameters/capacities; explicit FP8 and synthetic runtime values"
            if recorded
            else "explicit FP8 dtype variant, not a tuple from the FP8 capture"
        ),
        origin=case["origin"],
        actual_indexer_dtype_observed=False,
        synthetic_fields=[
            "dtype",
            "logits_budget",
            "strides",
            "storage_offsets",
            "metadata_dtypes",
            "query_start_loc_values",
            "physical_page_map",
            "tensor_values",
            "request_order",
            "per_request_sequence_lengths",
            "visible_blocks",
        ]
        + (["query_lengths"] if case["query_lengths_synthetic"] else []),
        select_all=min(
            info["page_table_shape"][1] * case["page_size"],
            (info["batch_max_seq_len"] + case["ratio"] - 1) // case["ratio"],
        )
        <= case["k"],
        chunk_provenance=(
            "derived using captured88-column table and assumed512MiB FP32 logits, not traced"
            if recorded
            else "derived using tight table and assumed512MiB FP32 logits, not traced"
        ),
    )
    if recorded:
        info.update(
            upload_commit=UPLOAD_COMMIT,
            archive_sha256=ARCHIVE_SHA256,
            capture_file=CAPTURE_FILE,
            capture_line=case["capture_line"],
            recorded_shape=case["recorded_shape"],
            captured_parameters=case["captured_parameters"],
            query_lengths_provenance="inferred lengths; request order remains synthetic",
            recorded_count=case["recorded_count"],
        )
    else:
        info["synthetic_fields"] += ["compress_ratio", "token_topk", "page_size"]
        info["variant_of"] = case["variant_of"]
    return info
