# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
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
):
    """Build request-major packed queries and shuffled physical pages."""
    torch.manual_seed(seed)
    rows = sum(query_lengths)
    pages_per_req = [
        (length + page_size - 1) // page_size for length in compressed_lengths
    ]
    width = max(pages_per_req)
    num_pages = sum(pages_per_req)
    q = torch.randn((rows, 4, 128), dtype=torch.bfloat16, device=device)
    cache = torch.randn(
        (num_pages, page_size, 1, 128), dtype=torch.bfloat16, device=device
    )
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
        q_storage = torch.empty((rows, 5, 128), dtype=q.dtype, device=device)
        q_storage[:, :4].copy_(q)
        q = q_storage[:, :4]
        cache_storage = torch.empty(
            (num_pages, page_size + 1, 1, 128), dtype=cache.dtype, device=device
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
