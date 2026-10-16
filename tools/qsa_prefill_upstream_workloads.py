# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Reconstruct the public PR #54513 prefill grid; no GPU imports here."""

SOURCE_URL = "https://github.com/vllm-project/vllm/pull/54513"
LAYOUTS = ((1, 8192), (1, 16384), (2, 8192), (4, 4096))
PREFIXES = (8192, 51200, 102400)


def cases():
    return [
        dict(
            name=f"perf_n{requests}_q{queries}_p{prefix}_bfloat16",
            kind="upstream_perf",
            requests=requests,
            query_length=queries,
            prefix_tokens=prefix,
            page_size=196,
            heads=4,
            dim=128,
            ratio=4,
            k=512,
            budget_mb=512,
            dtype="bfloat16",
        )
        for requests, queries in LAYOUTS
        for prefix in PREFIXES
    ]


def geometry(case):
    """Keep original-token lengths separate from compressed cache/page extents."""
    requests = case["requests"]
    queries = case["query_length"]
    prefix = case["prefix_tokens"]
    ratio = case["ratio"]
    sequence = prefix + queries
    compressed = (sequence + ratio - 1) // ratio
    pages = (compressed + case["page_size"] - 1) // case["page_size"]
    # Same public workspace policy: width clipped by sequence then aligned to 64.
    width = min(pages * case["page_size"], ((compressed + 63) // 64) * 64)
    rows = requests * queries
    chunk_rows = min(rows, max(1, case["budget_mb"] * 1024**2 // (width * 4)))
    return dict(
        source=SOURCE_URL,
        provenance="published query/prefix/page grid; explicit local dtype/H/D choices",
        prefix_unit="original_tokens_before_query",
        query_lengths=[queries] * requests,
        sequence_lengths=[sequence] * requests,
        compressed_lengths=[compressed] * requests,
        pages_per_request=pages,
        physical_pages=requests * pages,
        q_shape=[rows, case["heads"], case["dim"]],
        cache_shape=[requests * pages, case["page_size"], 1, case["dim"]],
        page_table_shape=[requests, pages],
        visible_first=(prefix + 1) // ratio,
        visible_last=sequence // ratio,
        logits_width=width,
        expected_chunk_rows=[
            min(chunk_rows, rows - offset) for offset in range(0, rows, chunk_rows)
        ],
        full_logits_bytes=rows * width * 4,
        score_workspace_bytes=chunk_rows * width * 4,
    )


def make_inputs(case, helpers, torch):
    """Use the existing randomized page builder with exact causal visibility."""
    info = geometry(case)
    visible = [
        (case["prefix_tokens"] + row + 1) // case["ratio"]
        for _ in range(case["requests"])
        for row in range(case["query_length"])
    ]
    args = helpers.make_inputs(
        info["query_lengths"],
        info["compressed_lengths"],
        k=case["k"],
        ratio=case["ratio"],
        page_size=case["page_size"],
        visible=visible,
        dtype=getattr(torch, case["dtype"]),
        num_heads=case["heads"],
        head_dim=case["dim"],
    )
    # Avoid rounding the original-token bound up to a complete compressed block.
    return (*args[:9], max(info["sequence_lengths"]))
