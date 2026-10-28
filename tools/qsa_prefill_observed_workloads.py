# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Recorded QSA envelopes with explicit synthetic runtime metadata; CPU safe."""

from tools import qsa_prefill_fp8_workloads as fp8
from tools import qsa_prefill_inference_workloads as inference
from tools import qsa_prefill_upstream_workloads as upstream

UPLOAD_COMMIT = "6cbcf092785f266ac39f1f68f110d45b1f519489"
ARCHIVE_SHA256 = "55944bc31b07cff46498c5ee36559f1290955fdefc4ebc7131336ad83a57eb7a"
SOURCE = (
    f"https://github.com/cheersluvs/FlagGems-vllm/tree/{UPLOAD_COMMIT}/"
    "results/qsa_select_paged_prefill/inference-shapes/20261020-014744"
)
SUITES = (
    "upstream-prefill",
    "observed-prefill",
    "fp8-prefill",
    "combined-prefill",
    "scored-prefill",
)
GROUPS = ("upstream_perf", "observed_perf", "fp8_perf")


def observed_cases():
    definitions = (
        ("short96", [96], 4096),
        ("tail16k", [384], 16384),
        ("tail32k", [368], 32781),
        ("short_batch3", [96] * 3, 4101),
        ("ragged54", [3200] + [112] * 38 + [96] * 15, 4100),
        ("unequal2", [12400, 3600], 32769),
        ("chunk_boundary", [16000, 368], 32768),
        ("chunk_tail64", [16000, 368], 32770),
        ("select_all_single", [800], 800),
        ("select_all_batch18", [800] * 18, 1025),
    )
    return [
        dict(
            name=f"observed_{name}_bfloat16",
            kind="observed_perf",
            query_lengths=queries,
            prefix_tokens=[sequence - q for q in queries],
            batch_max_seq_len=sequence,
            recorded_shape=[sum(queries), len(queries), 4, 128, max(queries), sequence],
            query_lengths_synthetic=name == "ragged54",
            query_base=0,
            page_size=196,
            heads=4,
            dim=128,
            ratio=4,
            k=512,
            budget_mb=512,
            dtype="bfloat16",
        )
        for name, queries, sequence in definitions
    ]


def cases(suite="upstream-prefill"):
    if suite not in SUITES:
        raise ValueError(f"Unknown QSA suite: {suite}")
    if suite == "upstream-prefill":
        return upstream.cases()
    if suite == "observed-prefill":
        return observed_cases()
    if suite == "fp8-prefill":
        return fp8.cases()
    combined = upstream.cases() + observed_cases() + fp8.cases()
    if suite == "scored-prefill":
        return [
            case for case in combined if not geometry(case).get("select_all", False)
        ]
    return combined


def geometry(case):
    if case["kind"] == "upstream_perf":
        return upstream.geometry(case)
    if case["kind"] == "fp8_perf":
        return fp8.geometry(case)
    info = inference.geometry(case)
    maximum = info["batch_max_seq_len"]
    bound = min(
        info["page_table_shape"][1] * case["page_size"],
        (maximum + case["ratio"] - 1) // case["ratio"],
    )
    info.update(
        source=SOURCE,
        upload_commit=UPLOAD_COMMIT,
        archive_sha256=ARCHIVE_SHA256,
        provenance="recorded six-tuple; synthetic per-request context and runtime layout",
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
        recorded_shape=case["recorded_shape"],
        select_all=bound <= case["k"],
    )
    return info


def make_inputs(case, helpers, torch, *, device="cuda"):
    if case["kind"] == "fp8_perf" and "physical_pages" in case:
        # Use captured full cache/table capacities with the existing safe remap.
        return inference.make_inputs(case, helpers, torch, device=device)
    if case["kind"] == "upstream_perf":
        return upstream.make_inputs(case, helpers, torch, device=device)
    info = geometry(case)
    visible = [
        (prefix + row + 1) // case["ratio"]
        for length, prefix in zip(case["query_lengths"], case["prefix_tokens"])
        for row in range(length)
    ]
    args = helpers.make_inputs(
        info["query_lengths"],
        info["compressed_lengths"],
        k=case["k"],
        ratio=case["ratio"],
        page_size=case["page_size"],
        query_base=0,
        visible=visible,
        device=device,
        dtype=getattr(torch, case["dtype"]),
        num_heads=case["heads"],
        head_dim=case["dim"],
    )
    return (*args[:9], info["batch_max_seq_len"])
