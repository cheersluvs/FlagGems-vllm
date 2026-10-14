# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen inputs/references shared by QSA tests and benchmarks only."""

import hashlib
import importlib.util
import os
import sys
from importlib import import_module
from pathlib import Path

import pytest
import torch

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


def make_upstream_prefill_inputs(
    *, seq_len_slack=0, device="cuda", dtype=torch.bfloat16
):
    """Exact BF16/FP8 inputs from upstream test_qsa_prefill_selection_correctness.

    Source: REFERENCE_COMMIT, tests/models/qwen4_exp/test_qsa_reference.py.
    The oversized page table and page size 24 exercise clipped logits widths
    that need not coincide with the physical page geometry.
    """
    torch.manual_seed(2)
    query_lengths = [3, 33]
    q = torch.randn((36, 4, 128), device=device, dtype=torch.bfloat16).to(dtype)
    cache = torch.randn((128, 24, 1, 128), device=device, dtype=torch.bfloat16).to(
        dtype
    )
    table = torch.randperm(128, device=device, dtype=torch.int32).reshape(2, 64)
    starts = torch.tensor([0, 3, 36], device=device, dtype=torch.int32)
    sequence_lengths = torch.tensor([5120, 4224], device=device, dtype=torch.int32)
    token_to_req = torch.repeat_interleave(
        torch.arange(2, device=device, dtype=torch.int32),
        torch.tensor(query_lengths, device=device),
    )
    positions = torch.cat(
        [
            torch.arange(length - queries, length, device=device, dtype=torch.int32)
            for queries, length in zip(query_lengths, [5120, 4224])
        ]
    )
    visible = torch.minimum(
        (positions + 1) // 4,
        sequence_lengths.index_select(0, token_to_req.long()) // 4,
    )
    return (
        q,
        cache,
        table,
        starts,
        visible,
        2048,
        4,
        33,
        torch.full((36, 512), -17, device=device, dtype=torch.int32),
        5120 + seq_len_slack,
    )


def make_upstream_split_prefill_inputs(decode_query_len, *, device="cuda"):
    """Prefill slice from upstream test_qsa_split_selection_correctness.

    Preserve the preceding decode rows so query_start_loc has its original
    nonzero base. Decode and compressed-index expansion are separate APIs.
    """
    torch.manual_seed(13)
    query_lengths = [decode_query_len, decode_query_len, 33]
    rows = sum(query_lengths)
    q = torch.randn((rows, 4, 128), device=device, dtype=torch.bfloat16)
    cache = torch.randn((120, 16, 1, 128), device=device, dtype=torch.bfloat16)
    table = torch.arange(120, device=device, dtype=torch.int32).view(3, 40)
    starts = torch.tensor(
        [0, decode_query_len, 2 * decode_query_len, rows],
        device=device,
        dtype=torch.int32,
    )
    positions = torch.cat(
        [torch.arange(2560 - queries, 2560, device=device) for queries in query_lengths]
    )
    visible = ((positions + 1) // 4).to(torch.int32)
    prefill = slice(2 * decode_query_len, rows)
    return (
        q[prefill],
        cache,
        table[2:],
        starts[2:],
        visible[prefill],
        2048,
        4,
        33,
        torch.full((33, 512), -17, device=device, dtype=torch.int32),
        2560,
    )


def make_upstream_no_complete_blocks_inputs(*, device="cuda"):
    """Exact selection inputs from the upstream D=8 zero-visible fixture."""
    return (
        torch.zeros((2, 4, 8), device=device, dtype=torch.bfloat16),
        torch.zeros((1, 16, 1, 8), device=device, dtype=torch.bfloat16),
        torch.zeros((1, 1), device=device, dtype=torch.int32),
        torch.tensor([0, 2], device=device, dtype=torch.int32),
        torch.zeros((2,), device=device, dtype=torch.int32),
        2048,
        4,
        2,
        torch.empty((2, 512), device=device, dtype=torch.int32),
        64,
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


def upstream_torch_logits(q, cache, table, starts, visible):
    """Upstream fixture's exact FP32 einsum/reduction, before sqrt(D) scaling.

    Only use with the upstream fixtures whose entire page tables are valid.
    Keep the original synthetic Torch baseline unchanged.
    """
    lengths = starts[1:] - starts[:-1]
    token_to_req = torch.repeat_interleave(
        torch.arange(table.shape[0], device=q.device, dtype=torch.int32), lengths
    )
    pages = table.index_select(0, token_to_req.long()).long()
    keys = cache[pages, :, 0, :].flatten(1, 2)
    scores = torch.einsum("rhd,rnd->rnh", q.float(), keys.float())
    logits = torch.relu(scores).sum(dim=-1)
    columns = torch.arange(keys.shape[1], device=q.device).unsqueeze(0)
    return logits.masked_fill(columns >= visible.unsqueeze(1), -torch.inf)


def assert_selection(logits, visible, output, *, rtol=2e-4, atol=2e-3):
    """Check count, range, uniqueness, padding and the selected score multiset."""
    k = output.shape[1]
    lengths = visible.cpu().tolist()
    for row, length in enumerate(lengths):
        selected = output[row]
        valid = selected[selected >= 0].long()
        count = min(length, k)
        assert valid.numel() == count, (row, length, selected)
        assert (selected >= -1).all(), (row, selected)
        assert (valid < length).all(), (row, valid, length)
        assert valid.unique().numel() == count, (row, valid)
        assert (selected == -1).sum().item() == k - count
        if count:
            actual = logits[row, valid].sort().values
            expected = logits[row, :length].topk(count).values.sort().values
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def assert_upstream_selection(logits, visible, output, *, dtype, head_dim=128):
    """Match upstream BF16 index equality and SM90 FP8 value tolerances.

    The upstream Torch reference divides scores by sqrt(D); production scores
    are unscaled. Scaling here reproduces its FP8 absolute tolerance and BF16
    reference Top-K without changing the production scoring contract.
    """
    fp8 = dtype == torch.float8_e4m3fn
    scale = head_dim**-0.5
    assert_selection(
        logits,
        visible,
        output,
        rtol=1e-3 if fp8 else 2e-4,
        atol=1e-3 / scale if fp8 else 2e-3,
    )
    for row, length in enumerate(visible.cpu().tolist()):
        count = min(length, output.shape[1])
        scores = logits[row, :length] / head_dim**0.5
        if fp8:
            if count:
                selected = output[row][output[row] >= 0].long()
                torch.testing.assert_close(
                    scores[selected].sort().values,
                    scores.topk(count).values.sort().values,
                    rtol=1e-3,
                    atol=1e-3,
                )
        else:
            expected = torch.full_like(output[row], -1)
            if count:
                expected[:count] = scores.topk(count).indices.to(torch.int32)
            torch.testing.assert_close(
                output[row].sort().values, expected.sort().values, rtol=0, atol=0
            )


def vllm_reference():
    """Load only the pinned upstream file; never silently relabel a baseline."""
    path = os.environ.get("QSA_VLLM_REFERENCE_FILE")
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
            "Use tools/run_qsa_prefill_h800.sh with a compatible vLLM build."
        )
        if os.environ.get("QSA_REQUIRE_VLLM_REFERENCE") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return module
