# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Ten FP8 benchmark envelopes and input builders; no GPU imports.

Eight envelopes use captured cache/table extents; query values, page maps,
visibility and dtype are explicit synthetic inputs. Compact and long100k are
FP8 variants. These definitions preserve the validated H800 workload matrix.
"""

UPLOAD_COMMIT = "9937ec6e910db658ee2871ac1743eead0698b7c4"
ARCHIVE_SHA256 = "f2d4ae225613f20b9625dc4e5f0b5279815e11064eb55fbb23c5a4441db05fcc"
SOURCE = (
    f"https://github.com/cheersluvs/FlagGems-vllm/tree/{UPLOAD_COMMIT}/"
    "results/qsa_select_paged_prefill/inference-shapes/20261022-070202"
)
CAPTURE_FILE = "Qwen3.8-Flash-Next-FP8-p32768d1024.txt"


SOURCE_URL = (
    "https://github.com/vllm-project/vllm/blob/"
    "db9527a46873454610df6dbedf79a36d6bf1a7f6/"
    "vllm/models/qwen4_exp/nvidia/indexer_qsa.py"
)


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


def _base_geometry(case):
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


def geometry(case):
    info = _base_geometry(case)
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


def _paged_inputs(case, helpers, torch, *, device="cuda"):
    info = _base_geometry(case)
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


def make_inputs(case, helpers, torch, *, device="cuda"):
    if case["kind"] == "fp8_perf" and "physical_pages" in case:
        # Use captured full cache/table capacities with the existing safe remap.
        return _paged_inputs(case, helpers, torch, device=device)
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
