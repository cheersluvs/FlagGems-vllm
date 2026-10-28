# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Published twelve layouts plus ten source-derived inference probes, not traces."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "qsa_prefill_upstream_workloads",
    Path(__file__).with_name("qsa_prefill_upstream_workloads.py"),
)
upstream = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(upstream)

SOURCE_URL = (
    "https://github.com/vllm-project/vllm/blob/"
    "db9527a46873454610df6dbedf79a36d6bf1a7f6/"
    "vllm/models/qwen4_exp/nvidia/indexer_qsa.py"
)
LAYOUT_SOURCE_URL = (
    "https://github.com/vllm-project/vllm/blob/"
    "db9527a46873454610df6dbedf79a36d6bf1a7f6/"
    "vllm/v1/kv_cache_interface.py#L1423"
)


def inference_cases():
    definitions = [
        ("short128", [128], [8192]),
        ("short512_long", [512], [102400]),
        ("scheduler_tail", [2048], [49152]),
        ("first_chunk", [8192], [0]),
        ("unequal", [4096, 2048, 1024, 1024], [8192, 51200, 102400, 0]),
        ("many_requests", [256] * 32, [8192] * 32),
        ("skewed", [7168] + [64] * 16, [51200] + [8192] * 16),
        ("mixed_decode", [4096], [4096]),
    ]
    result = [
        dict(
            name=f"infer_{name}_bfloat16",
            kind="inference_perf",
            query_lengths=queries,
            prefix_tokens=prefixes,
            page_size=196,
            heads=4,
            dim=128,
            ratio=4,
            k=512,
            budget_mb=512,
            dtype="bfloat16",
            query_base=0,
        )
        for name, queries, prefixes in definitions
    ]
    # One preceding decode row with sequence length 200K; the operator receives
    # only the prefill slice, but keeps the full-batch sequence upper bound.
    result[-1].update(query_base=1, batch_max_seq_len=204800, table_columns=335)
    for original in (result[0], result[-1]):
        result.append(
            dict(
                original,
                name=original["name"].replace("_bfloat16", "_global_strided_bfloat16"),
                table_columns=335,
                physical_pages=4096,
                cache_page_stride_factor=2,
                layout_source=LAYOUT_SOURCE_URL,
                layout_provenance="synthetic two-layer block-outermost view; not a deployment trace",
            )
        )
    return result


def cases():
    return upstream.cases() + inference_cases()


def geometry(case):
    if case["kind"] == "upstream_perf":
        return upstream.geometry(case)
    queries, prefixes = case["query_lengths"], case["prefix_tokens"]
    if len(queries) != len(prefixes) or not queries:
        raise ValueError("Each prefill request needs one query length and prefix")
    if any(q <= 0 for q in queries) or any(p < 0 for p in prefixes):
        raise ValueError("Query lengths must be positive and prefixes nonnegative")
    ratio, page, heads, dim = (case[k] for k in ("ratio", "page_size", "heads", "dim"))
    sequences = [q + p for q, p in zip(queries, prefixes)]
    compressed = [(s + ratio - 1) // ratio for s in sequences]
    pages = [(c + page - 1) // page for c in compressed]
    columns = case.get("table_columns", max(pages))
    physical = case.get("physical_pages", sum(pages))
    maximum = case.get("batch_max_seq_len", max(sequences))
    factor = case.get("cache_page_stride_factor", 1)
    if columns < max(pages) or physical < sum(pages) or maximum < max(sequences):
        raise ValueError("Cache/table/context capacity cannot truncate prefill inputs")
    if factor < 1:
        raise ValueError("Cache page stride factor must be positive")
    width = min(columns * page, ((maximum + ratio - 1) // ratio + 63) // 64 * 64)
    rows = sum(queries)
    chunk = min(rows, max(1, case["budget_mb"] * 1024**2 // (width * 4)))
    base = case["query_base"]
    starts = [base]
    for q in queries:
        starts.append(starts[-1] + q)
    return dict(
        source=SOURCE_URL,
        provenance="source-derived inference probe; lengths/capacities are proposed, not traced",
        layout_source=case.get("layout_source"),
        layout_provenance=case.get(
            "layout_provenance", "compact cache with shuffled pages"
        ),
        prefix_unit="original_tokens_before_query",
        query_lengths=queries,
        prefix_tokens=prefixes,
        sequence_lengths=sequences,
        compressed_lengths=compressed,
        pages_per_request=pages,
        active_physical_pages=sum(pages),
        physical_pages=physical,
        q_shape=[rows, heads, dim],
        cache_shape=[physical, page, 1, dim],
        page_table_shape=[len(queries), columns],
        batch_max_seq_len=maximum,
        query_start_loc=starts,
        visible_first=(prefixes[0] + 1) // ratio,
        visible_last=sequences[-1] // ratio,
        visible_first_per_request=[(p + 1) // ratio for p in prefixes],
        visible_last_per_request=[s // ratio for s in sequences],
        input_layout={
            "0": dict(stride=[heads * dim, dim, 1], storage_offset=base * heads * dim),
            "1": dict(
                stride=[factor * page * dim, dim, dim, 1],
                storage_offset=(factor - 1) * page * dim,
            ),
            "2": dict(stride=[columns, 1], storage_offset=columns if base else 0),
            "3": dict(stride=[1], storage_offset=1 if base else 0),
            "4": dict(stride=[1], storage_offset=base),
            "8": dict(stride=[case["k"], 1], storage_offset=base * case["k"]),
        },
        logits_width=width,
        expected_chunk_rows=[
            min(chunk, rows - offset) for offset in range(0, rows, chunk)
        ],
        full_logits_bytes=rows * width * 4,
        score_workspace_bytes=chunk * width * 4,
    )


def make_inputs(case, helpers, torch, *, device="cuda"):
    if case["kind"] == "upstream_perf":
        return upstream.make_inputs(case, helpers, torch, device=device)
    info = geometry(case)
    visible = [
        (prefix + row + 1) // case["ratio"]
        for queries, prefix in zip(case["query_lengths"], case["prefix_tokens"])
        for row in range(queries)
    ]
    args = list(
        helpers.make_inputs(
            info["query_lengths"],
            info["compressed_lengths"],
            k=case["k"],
            ratio=case["ratio"],
            page_size=case["page_size"],
            query_base=case["query_base"],
            visible=visible,
            device=device,
            dtype=getattr(torch, case["dtype"]),
            num_heads=case["heads"],
            head_dim=case["dim"],
        )
    )
    cache, table = args[1:3]
    if "physical_pages" in case:
        # Model two adjacent layer pages in each physical block. Unused pages
        # are zeroed so fingerprints never depend on uninitialized GPU memory.
        factor = case["cache_page_stride_factor"]
        storage = torch.zeros(
            (info["physical_pages"], factor, case["page_size"], 1, case["dim"]),
            dtype=cache.dtype,
            device=device,
        )
        expanded = storage[:, factor - 1]
        ids = torch.randperm(info["physical_pages"], device=device)[: cache.shape[0]]
        expanded[ids] = cache
        live = table >= 0
        table[live] = ids[table[live].long()].to(table.dtype)
        args[1] = expanded
    if table.shape[1] != info["page_table_shape"][1]:
        expanded = torch.full(
            info["page_table_shape"], -1, dtype=table.dtype, device=device
        )
        expanded[:, : table.shape[1]].copy_(table)
        table = expanded
    args[2] = table
    base = case["query_base"]
    if base:
        # Preserve the view offsets created when the preceding decode rows are
        # sliced off. Starts keep their absolute values, not rebased values.
        for index in (0, 4, 8):
            value = args[index]
            storage = torch.zeros(
                (base + value.shape[0], *value.shape[1:]),
                dtype=value.dtype,
                device=device,
            )
            storage[base:].copy_(value)
            args[index] = storage[base:]
        for index in (2, 3):
            value = args[index]
            storage = torch.zeros(
                (1 + value.shape[0], *value.shape[1:]), dtype=value.dtype, device=device
            )
            storage[1:].copy_(value)
            args[index] = storage[1:]
    args[9] = info["batch_max_seq_len"]
    return tuple(args)
