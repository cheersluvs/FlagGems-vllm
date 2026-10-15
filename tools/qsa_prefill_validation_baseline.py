# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Benchmark-only original QSA validator; never imported by production.

Exact helper ASTs from commit 416083e2ea03dc5695b53f53ec5042dd5fb6fd9a.
Both A/B entries share the current public function bytecode and GPU launchers;
only its validation global differs. No module globals are mutated.
"""

import hashlib
import inspect
import types

import torch

from flaggems_vllm import runtime

BASELINE_COMMIT = "416083e2ea03dc5695b53f53ec5042dd5fb6fd9a"
BASELINE_OP_SHA256 = "1244de0fc2fdb7f67140c3156249d5e387c7ebc0fa026b59cacaa39b6cdc2b51"
VALIDATOR_SOURCE_SHA256 = (
    "327ba01287c2d7f75cf0704cd2ba21a81e1c5b020f48a9a8bf9753866b5af1f1"
)
PUBLIC_ENTRY_SHA256 = "277945267d171b819dc6c282e41368d259ed3831a26faed8d04b13e7f75e508a"


def _check_nonoverlapping_strides(tensor, name):
    extent = 1
    for stride, size in sorted(zip(tensor.stride(), tensor.shape)):
        if size > 1:
            if stride < extent:
                raise NotImplementedError(f"{name} must have nonoverlapping strides")
            extent += (size - 1) * stride


def _overlaps(a, b):
    if not a.numel() or not b.numel():
        return False
    a_start, b_start = a.data_ptr(), b.data_ptr()
    a_end = (
        a_start
        + (1 + sum((size - 1) * stride for size, stride in zip(a.shape, a.stride())))
        * a.element_size()
    )
    b_end = (
        b_start
        + (1 + sum((size - 1) * stride for size, stride in zip(b.shape, b.stride())))
        * b.element_size()
    )
    return a_start < b_end and b_start < a_end


def _validate_inputs(
    q,
    k_cache,
    page_table,
    query_start_loc,
    visible_blocks,
    token_topk,
    compress_ratio,
    max_query_len,
    block_indices,
    max_seq_len,
):
    tensors = (q, k_cache, page_table, query_start_loc, visible_blocks, block_indices)
    if q.device.type != "cuda" or runtime.device.vendor_name != "nvidia":
        raise NotImplementedError("QSA prefill currently requires NVIDIA CUDA SM90")
    if any(t.device != q.device for t in tensors):
        raise ValueError("QSA prefill tensors must be on the same device")
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise NotImplementedError("QSA prefill currently supports SM90 (H800/H100)")
    if q.dtype != k_cache.dtype:
        raise ValueError("Q and the compressed K cache must match dtype")
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise NotImplementedError("QSA prefill supports BF16/FP8 E4M3FN query/cache")
    if q.ndim != 3:
        raise NotImplementedError("QSA prefill query must be [tokens, heads, head_dim]")
    if q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA prefill heads and head_dim must be positive")
    if k_cache.ndim != 4 or k_cache.shape[2:] != (1, q.shape[2]):
        raise NotImplementedError(
            "QSA prefill cache must be [pages, page_size, 1, D] matching Q"
        )
    if q.requires_grad or k_cache.requires_grad:
        raise NotImplementedError("QSA prefill is inference-only")
    for name, value in (
        ("token_topk", token_topk),
        ("compress_ratio", compress_ratio),
        ("max_query_len", max_query_len),
        ("max_seq_len", max_seq_len),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if compress_ratio <= 0 or token_topk <= 0 or token_topk % compress_ratio:
        raise ValueError("token_topk must be positive and divisible by compress_ratio")
    k = token_topk // compress_ratio
    if k not in (512, 1024, 2048):
        raise NotImplementedError("QSA prefill supports block Top-K 512/1024/2048")
    rows = q.shape[0]
    if page_table.ndim != 2:
        raise ValueError("page_table must be [requests, max_pages]")
    if page_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("page_table must use int32/int64")
    if query_start_loc.shape != (page_table.shape[0] + 1,):
        raise ValueError("query_start_loc must contain requests + 1 offsets")
    if query_start_loc.dtype not in (torch.int32, torch.int64):
        raise TypeError("query_start_loc must use int32/int64")
    if visible_blocks.shape != (rows,) or visible_blocks.dtype != torch.int32:
        raise ValueError("visible_blocks must be int32 [tokens]")
    if block_indices.shape != (rows, k) or block_indices.dtype != torch.int32:
        raise ValueError("block_indices must be int32 [tokens, block_topk]")
    if max_query_len < 0 or max_seq_len < 0:
        raise ValueError("maximum lengths must be nonnegative")
    if rows and (
        not all(k_cache.shape[:2])
        or not all(page_table.shape)
        or max_query_len == 0
        or max_seq_len == 0
    ):
        raise ValueError(
            "nonempty queries require positive cache/table/maximum lengths"
        )
    if q.stride(-1) != 1 or k_cache.stride(-1) != 1:
        raise NotImplementedError("QSA query/cache require unit innermost stride")
    for name, tensor in (("q", q), ("cache", k_cache), ("page_table", page_table)):
        _check_nonoverlapping_strides(tensor, name)
    for tensor in (query_start_loc, visible_blocks, block_indices):
        if not tensor.is_contiguous():
            raise NotImplementedError("QSA metadata/output must be contiguous")
    if any(_overlaps(block_indices, tensor) for tensor in tensors[:-1]):
        raise ValueError("block_indices must not overlap any input")
    return rows, k


def baseline_entry(op):
    helpers = (_check_nonoverlapping_strides, _overlaps, _validate_inputs)
    source = "\0".join(inspect.getsource(fn).strip() for fn in helpers)
    if hashlib.sha256(source.encode()).hexdigest() != VALIDATOR_SOURCE_SHA256:
        raise RuntimeError("Frozen validation helpers differ from the pinned baseline")
    original = op.qsa_select_paged_prefill
    source = inspect.getsource(original).strip()
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != PUBLIC_ENTRY_SHA256:
        raise RuntimeError(
            "Validation-only baseline requires the original public entry body"
        )
    namespace = dict(original.__globals__)
    namespace["_validate_inputs"] = _validate_inputs
    for name in ("_prefill_logits", "persistent_topk", "_select_all_kernel"):
        if namespace[name] is not getattr(op, name):
            raise RuntimeError(f"A/B must share the same launcher: {name}")
    return types.FunctionType(
        original.__code__,
        namespace,
        "qsa_full_validation_baseline",
        original.__defaults__,
        original.__closure__,
    )
