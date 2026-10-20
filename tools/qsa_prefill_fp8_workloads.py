# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Explicit FP8 fixtures: eight recorded envelopes and two dtype variants."""

from tools import qsa_prefill_inference_workloads as inference

UPLOAD_COMMIT = "7835b9fd94c6f0c75339a9e156d2fc2cb4183186"
ARCHIVE_SHA256 = "857381f56826cf7735b7acf54c158fcf4dc850bffd9347faeb220e60b394e106"
SOURCE = (
    f"https://github.com/cheersluvs/FlagGems-vllm/tree/{UPLOAD_COMMIT}/"
    "results/qsa_select_paged_prefill/inference-shapes/20261020-090859"
)
CAPTURE_FILE = "Qwen3.8-Flash-Next-FP8-p32768d1024.txt"


def cases():
    definitions = (
        ("first_chunk", [16384], 16384, 48, 237),
        ("budget_exact", [16384], 32768, 48, 238),
        ("dominant_long", [16331], 33842, 432, 116),
        ("tail64", [16368], 32852, 48, 205),
        ("tail65", [16369], 32850, 48, 207),
        ("two_balanced", [10375, 5954], 32931, 48, 119),
        ("two_skewed", [16331, 53], 32821, 48, 239),
        ("short_tail", [2319], 33843, 48, 240),
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
            "recorded six-tuple; explicit FP8 choice and synthetic runtime metadata"
            if recorded
            else "explicit FP8 dtype variant, not a tuple from the FP8 capture"
        ),
        origin=case["origin"],
        actual_indexer_dtype_observed=False,
        synthetic_fields=[
            "dtype",
            "compress_ratio",
            "token_topk",
            "logits_budget",
            "page_size",
            "strides",
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
        chunk_provenance="derived using tight page table and 512MiB FP32 logits, not traced",
    )
    if recorded:
        info.update(
            upload_commit=UPLOAD_COMMIT,
            archive_sha256=ARCHIVE_SHA256,
            capture_file=CAPTURE_FILE,
            capture_line=case["capture_line"],
            recorded_shape=case["recorded_shape"],
            recorded_count=case["recorded_count"],
        )
    else:
        info["variant_of"] = case["variant_of"]
    return info
