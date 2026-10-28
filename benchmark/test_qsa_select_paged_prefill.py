# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H800 QSA prefill: full operator, score stage, and Top-K stage.

The active matrix contains eight recorded FP8 envelopes and two FP8 variants.
The eight captured fixtures use cache [2712,196,1,128] and page table [N,88];
tail112/tail113 replace the former tight-capacity tail64/tail65 fixtures.
The H800 --mode kernel runner reports score, Top-K and the full chunk pipeline;
NV acceptance uses score-only kernel speedup, excluding Top-K. Isolated score
pytest tests retain their single full-row launch. BF16 is excluded from this
benchmark;select-all cases remain functional tests only.

The official baseline is SHA256-pinned and required by the card runner.
The independent Torch baseline includes host metadata reads and is reported
separately; it must not be used as a substitute for the vLLM comparison.
"""

import hashlib
import importlib.util
import os
import sys
from importlib import import_module
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.top_k_per_row_prefill import top_k_per_row_prefill

from . import base

REFERENCE_COMMIT = "7a877ae34dcf4a7d46de08eb16db4699e1e0047c"
REFERENCE_SHA256 = "4d09eb2f409e7cd71c7849c4f3bdd71d722bce6feaccb558f7ae67ac2d489a2b"


def make_inputs(
    query_lengths,
    compressed_lengths,
    *,
    k=512,
    ratio=4,
    page_size=16,
    query_base=0,
    visible=None,
    padded=False,
    device="cuda",
    seed=20261013,
    dtype=torch.bfloat16,
    head_dim=128,
    num_heads=4,
):
    """Build request-major packed queries and shuffled physical pages."""
    torch.manual_seed(seed)
    rows = sum(query_lengths)
    pages_per_req = [
        (length + page_size - 1) // page_size for length in compressed_lengths
    ]
    width = max(pages_per_req)
    num_pages = sum(pages_per_req)
    q = torch.randn(
        (rows, num_heads, head_dim), dtype=torch.bfloat16, device=device
    ).to(dtype)
    cache = torch.randn(
        (num_pages, page_size, 1, head_dim), dtype=torch.bfloat16, device=device
    ).to(dtype)
    page_order = torch.randperm(num_pages, device=device).to(torch.int32)
    table = torch.full(
        (len(query_lengths), width), -1, dtype=torch.int32, device=device
    )
    cursor = 0
    for request, count in enumerate(pages_per_req):
        table[request, :count] = page_order[cursor : cursor + count]
        cursor += count
    offsets = [query_base]
    default_visible = []
    for length, context in zip(query_lengths, compressed_lengths):
        offsets.append(offsets[-1] + length)
        # Simulate the last `length` logical tokens of a causal prefill request.
        default_visible.extend(
            max(0, min(context, (context * ratio - length + row + 1) // ratio))
            for row in range(length)
        )
    starts = torch.tensor(offsets, dtype=torch.int32, device=device)
    visible = torch.tensor(
        default_visible if visible is None else visible,
        dtype=torch.int32,
        device=device,
    )
    if padded:
        q_storage = torch.empty(
            (rows, num_heads + 1, head_dim), dtype=q.dtype, device=device
        )
        q_storage[:, :num_heads].copy_(q)
        q = q_storage[:, :num_heads]
        cache_storage = torch.empty(
            (num_pages, page_size + 1, 1, head_dim), dtype=cache.dtype, device=device
        )
        cache_storage[:, :page_size].copy_(cache)
        cache = cache_storage[:, :page_size]
        table_storage = torch.full(
            (len(query_lengths), width * 2), -1, dtype=table.dtype, device=device
        )
        table_storage[:, ::2].copy_(table)
        table = table_storage[:, ::2]
    output = torch.full((rows, k), -17, dtype=torch.int32, device=device)
    return (
        q,
        cache,
        table,
        starts,
        visible,
        k * ratio,
        ratio,
        max(query_lengths),
        output,
        max(compressed_lengths) * ratio,
    )


def torch_logits(q, cache, table, starts, visible, width=None):
    """Independent FP32 reference, with inactive columns set to -inf."""
    width = table.shape[1] * cache.shape[1] if width is None else width
    logits = torch.full((q.shape[0], width), -torch.inf, device=q.device)
    offsets = (starts - starts[0]).cpu().tolist()
    lengths = visible.cpu().tolist()
    for request in range(table.shape[0]):
        begin, end = offsets[request : request + 2]
        if begin == end:
            continue
        count = max(lengths[begin:end], default=0)
        if count == 0:
            continue
        columns = torch.arange(count, device=q.device)
        pages = table[request, columns // cache.shape[1]].long()
        keys = cache[pages, columns % cache.shape[1], 0].float()
        scores = torch.einsum("rhd,cd->rhc", q[begin:end].float(), keys)
        scores = scores.clamp_min(0).sum(1)
        mask = columns[None, :] < visible[begin:end, None]
        logits[begin:end, :count] = torch.where(mask, scores, -torch.inf)
    return logits


def torch_select(
    q,
    cache,
    table,
    starts,
    visible,
    token_topk,
    ratio,
    max_query_len,
    output,
    max_seq_len,
):
    """Operator-level baseline including score/Top-K/output preparation."""
    del max_query_len
    k = token_topk // ratio
    width = min(table.shape[1] * cache.shape[1], (max_seq_len + ratio - 1) // ratio)
    logits = torch_logits(q, cache, table, starts, visible, width)
    output.fill_(-1)
    if q.shape[0] == 0:
        return
    # Unsorted selection is compatible with vLLM's set-valued Top-K contract.
    selected = torch.topk(logits, min(k, width), dim=1, sorted=False).indices
    selected = torch.where(selected < visible[:, None], selected, -1)
    output[:, : selected.shape[1]].copy_(selected.to(torch.int32))


def vllm_reference():
    """Load only the pinned upstream file; never silently relabel a baseline."""
    path = os.environ.get("QSA_VLLM_REFERENCE_FILE")
    if os.environ.get("QSA_NATIVE_REFERENCE_PYTHON"):
        raise NotImplementedError(
            "Only in-process pinned vLLM comparisons are supported"
        )
    try:
        if path:
            source_path = Path(path)
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != REFERENCE_SHA256:
                raise RuntimeError("QSA_VLLM_REFERENCE_FILE has the wrong SHA256")
            spec = importlib.util.spec_from_file_location("_qsa_vllm_reference", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        else:
            module = import_module("vllm.models.qwen4_exp.nvidia.ops.qsa_indexer")
            source_path = Path(module.__file__)
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != REFERENCE_SHA256:
                raise RuntimeError(
                    "installed vLLM QSA source is not the pinned baseline"
                )
        import_module("vllm._custom_ops")
        for name in ("persistent_topk", "cooperative_topk"):
            if not hasattr(torch.ops._C, name):
                raise RuntimeError(f"vLLM build does not expose {name}")
    except (ImportError, AttributeError, OSError, RuntimeError) as exc:
        message = (
            f"Pinned vLLM QSA baseline unavailable: {exc}. "
            "Set QSA_VLLM_REFERENCE_FILE to the pinned source with a compatible vLLM build."
        )
        if os.environ.get("QSA_REQUIRE_VLLM_REFERENCE") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return module


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


def _benchmark_cases():
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


def _benchmark_base_geometry(case):
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


def _benchmark_geometry(case):
    info = _benchmark_base_geometry(case)
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


def _benchmark_paged_inputs(case, helpers, torch, *, device="cuda"):
    info = _benchmark_base_geometry(case)
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


def _benchmark_inputs(case, helpers, torch, *, device="cuda"):
    if case["kind"] == "fp8_perf" and "physical_pages" in case:
        # Use captured full cache/table capacities with the existing safe remap.
        return _benchmark_paged_inputs(case, helpers, torch, device=device)
    info = _benchmark_geometry(case)
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


op_module = import_module("flaggems_vllm.ops.qsa_select_paged_prefill")
pytestmark = [
    pytest.mark.qsa_select_paged_prefill,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "nvidia"
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (9, 0),
        reason="NVIDIA SM90 required",
    ),
]

# Frozen matrix validated by the H800 --mode kernel runner. Prefix is in
# original tokens before the query, not compressed cache rows.
BENCHMARK_CASES = {(case["name"],): case for case in _benchmark_cases()}


class QSAPrefillBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # All ten FP8 cases exercise both score and TopK.
        self.shapes = list(BENCHMARK_CASES)
        self.shape_desc = (
            "case (query lengths, prefixes and layout in workload manifest) "
            "(10 FP8, H=4, D=128, page=196, block_k=512)"
        )

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        # Each named case has one declared dtype, not a cross-product matrix.
        for shape in self.shapes:
            if getattr(torch, BENCHMARK_CASES[shape]["dtype"]) == dtype:
                yield from self.input_fn(shape, dtype, self.device)


class QSAScoredBenchmark(QSAPrefillBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            shape
            for shape in self.shapes
            if not _benchmark_geometry(BENCHMARK_CASES[shape]).get("select_all")
        ]


def _inputs(shape, dtype, device):
    assert dtype == getattr(torch, BENCHMARK_CASES[shape]["dtype"])
    yield _benchmark_inputs(
        BENCHMARK_CASES[shape], sys.modules[__name__], torch, device=device
    )


def _score_inputs(shape, dtype, device):
    args = next(_inputs(shape, dtype, device))
    width = _benchmark_geometry(BENCHMARK_CASES[shape])["logits_width"]
    workspace = torch.empty((args[0].shape[0] * width,), device=device)
    yield (*args[:5], workspace, args[7], width, 0, args[0].shape[0])


def _topk_inputs(shape, dtype, device):
    args = next(_inputs(shape, dtype, device))
    width = _benchmark_geometry(BENCHMARK_CASES[shape])["logits_width"]
    scores_workspace = torch.empty((args[0].shape[0] * width,), device=device)
    logits = op_module._prefill_logits(
        *args[:5], scores_workspace, args[7], width, 0, args[0].shape[0]
    )
    row_starts = torch.zeros((args[0].shape[0],), device=device, dtype=torch.int32)
    topk_workspace = torch.empty((1024 * 1024,), dtype=torch.uint8, device=device)
    yield logits, args[4], args[5], args[6], args[8], topk_workspace, row_starts


def _gems_topk(logits, visible, token_topk, ratio, output, workspace, starts):
    del starts
    op_module.persistent_topk(
        logits,
        visible,
        output,
        workspace,
        k=token_topk // ratio,
        max_seq_len=logits.shape[1],
    )


def _old_gems_topk(logits, visible, token_topk, ratio, output, workspace, starts):
    del workspace
    top_k_per_row_prefill(
        logits,
        starts,
        visible,
        output,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        token_topk // ratio,
    )


def _run(
    name,
    reference,
    gems,
    input_fn,
    benchmark_cls=QSAPrefillBenchmark,
    dtypes=(torch.float8_e4m3fn,),
):
    # Warm both functions on each input before timing to exclude compilation
    # and autotuning. The common runner then measures operator-level latency.
    def warmed_inputs(shape, dtype, device):
        for args in input_fn(shape, dtype, device):
            reference(*args)
            gems(*args)
            torch.cuda.synchronize()
            yield args

    benchmark_cls(
        op_name=name,
        torch_op=reference,
        gems_op=gems,
        input_fn=warmed_inputs,
        dtypes=list(dtypes),
    ).run()


def test_full_vs_torch():
    _run(
        "qsa_prefill_vs_torch",
        torch_select,
        flaggems_vllm.qsa_select_paged_prefill,
        _inputs,
    )


def test_full_vs_vllm():
    reference = vllm_reference()
    _run(
        "qsa_prefill_vs_vllm",
        reference.qsa_select_paged_prefill,
        flaggems_vllm.qsa_select_paged_prefill,
        _inputs,
    )


def test_scores_vs_vllm():
    reference = vllm_reference()
    _run(
        "qsa_prefill_scores_vs_vllm",
        reference._prefill_logits,
        op_module._prefill_logits,
        _score_inputs,
        benchmark_cls=QSAScoredBenchmark,
    )


def test_topk_vs_vllm():
    reference = vllm_reference()

    def baseline(logits, visible, token_topk, ratio, output, workspace, starts):
        del starts
        reference._topk(logits, visible, token_topk, ratio, output, workspace)

    _run(
        "qsa_prefill_topk_vs_vllm",
        baseline,
        _gems_topk,
        _topk_inputs,
        benchmark_cls=QSAScoredBenchmark,
    )


def test_topk_persistent_vs_prefill():
    # Direct stage comparison in one runtime; baseline here is the old
    # FlagGems prefill implementation, explicitly separate from vLLM results.
    _run(
        "qsa_prefill_topk_persistent_vs_old_prefill",
        _old_gems_topk,
        _gems_topk,
        _topk_inputs,
        benchmark_cls=QSAScoredBenchmark,
    )
