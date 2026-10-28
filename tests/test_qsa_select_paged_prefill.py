# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import importlib.util
import os
import sys
from importlib import import_module
from pathlib import Path

import pytest
import torch
import triton

import flaggems_vllm

from . import conftest as cfg

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


def assert_native_selection(logits, visible, actual, native, *, dtype, head_dim):
    """Direct comparison under the existing score/tie tolerance contract.

    Strict BF16 upstream index comparisons remain in their original tests.
    Synthetic ties may choose different equivalent indices; compare their
    selected score multisets directly, including valid counts and -1 padding.
    """
    for row, length in enumerate(visible.cpu().tolist()):
        count = min(length, actual.shape[1])
        a = actual[row][actual[row] >= 0].long()
        b = native[row][native[row] >= 0].long()
        assert a.numel() == b.numel() == count
        assert (actual[row] == -1).sum() == (native[row] == -1).sum()
        scores = logits[row]
        if dtype == torch.float8_e4m3fn:
            scores = scores / head_dim**0.5
        torch.testing.assert_close(
            scores[a].sort().values,
            scores[b].sort().values,
            rtol=1e-3 if dtype == torch.float8_e4m3fn else 2e-4,
            atol=1e-3 if dtype == torch.float8_e4m3fn else 2e-3,
        )


op_module = import_module("flaggems_vllm.ops.qsa_select_paged_prefill")
op = flaggems_vllm.qsa_select_paged_prefill
pytestmark = pytest.mark.qsa_select_paged_prefill


@pytest.fixture
def sm90():
    if (
        flaggems_vllm.vendor_name != "nvidia"
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (9, 0)
    ):
        pytest.skip("NVIDIA SM90 accelerator required")
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


CASES = [
    ((1,), (17,), 512, 16, 0, False),
    ((7, 0, 13), (1536, 17, 2048), 512, 16, 19, True),
    ((33, 19), (1537, 4097), 1024, 64, 61, False),
    ((17,), (4097,), 2048, 16, 0, False),
]
if not cfg.QUICK_MODE:
    CASES += [
        ((129, 67, 31), (8193, 4097, 1025), 512, 64, 97, True),
        ((512,), (16385,), 1024, 16, 0, False),
    ]


@pytest.mark.parametrize("lengths,contexts,k,page_size,base,padded", CASES)
@pytest.mark.parametrize("metadata_dtype", [torch.int32, torch.int64])
def test_matches_torch(
    sm90, lengths, contexts, k, page_size, base, padded, metadata_dtype
):
    args = list(
        make_inputs(
            lengths, contexts, k=k, page_size=page_size, query_base=base, padded=padded
        )
    )
    args[2] = args[2].to(metadata_dtype)
    args[3] = args[3].to(metadata_dtype)
    reference = torch_logits(*args[:5])
    pointer = args[8].data_ptr()
    assert op(*args) is None
    assert args[8].data_ptr() == pointer
    assert_selection(reference, args[4], args[8])


@pytest.mark.parametrize("page_size", [16, 64])
def test_visible_counts_and_ties(sm90, page_size):
    counts = [0, 1, 511, 512, 513, 1024, 1537]
    args = make_inputs((len(counts),), (1537,), page_size=page_size, visible=counts)
    args[0].zero_()  # Equal scores: valid unique selection matters, not tie order.
    op(*args)
    assert_selection(torch_logits(*args[:5]), args[4], args[8], rtol=0, atol=0)


def test_relu_precedes_head_sum_and_score_is_unscaled(sm90):
    args = make_inputs((5,), (1025,), visible=[0, 1, 512, 1024, 1025])
    args[0].zero_()
    args[1].zero_()
    args[0][:, 0, 0] = 2
    args[0][:, 1, 0] = -3
    args[1][..., 0] = 1
    scores = _scores(args, 0, 5)
    mask = torch.arange(scores.shape[1], device="cuda")[None, :] < args[4][:, None]
    torch.testing.assert_close(
        scores[mask], torch.full_like(scores[mask], 2), rtol=0, atol=0
    )


def _scores(args, offset, count, workspace=None):
    width = args[2].shape[1] * args[1].shape[1]
    if workspace is None:
        workspace = torch.full((count * width,), 12345.0, device="cuda")
    return op_module._prefill_logits(
        *args[:5], workspace, args[7], width, offset, count
    )


PARAMETERIZED_SHAPES = [(1, 7), (3, 24), (5, 96), (8, 256)]
if not cfg.QUICK_MODE:
    PARAMETERIZED_SHAPES += [(1, 1), (2, 16), (5, 40), (12, 192), (17, 33), (3, 160)]


@pytest.mark.parametrize("num_heads,head_dim", PARAMETERIZED_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("force_chunk", [False, True])
def test_parameterized_heads_and_dims(
    sm90, monkeypatch, num_heads, head_dim, dtype, force_chunk
):
    monkeypatch.setenv(
        "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "0" if force_chunk else "512"
    )
    args = make_inputs(
        (4, 0, 5),
        (1025, 16, 1537),
        query_base=37,
        padded=True,
        visible=[0, 1, 511, 1025, 512, 513, 800, 1024, 1537],
        dtype=dtype,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    expected = torch_logits(*args[:5])
    original_score = op_module._prefill_logits
    chunks = []

    def checked_score(*score_args, **score_kwargs):
        scores = original_score(*score_args, **score_kwargs)
        width, offset, count = score_args[-3:]
        chunks.append((offset, count))
        mask = (
            torch.arange(width, device=args[0].device)[None, :]
            < args[4][offset : offset + count, None]
        )
        fp8 = dtype == torch.float8_e4m3fn
        torch.testing.assert_close(
            scores[mask],
            expected[offset : offset + count, :width][mask],
            rtol=1e-3 if fp8 else 2e-4,
            atol=1e-3 * head_dim**0.5 if fp8 else 2e-3,
        )
        return scores

    monkeypatch.setattr(op_module, "_prefill_logits", checked_score)
    op(*args)
    assert chunks == ([(row, 1) for row in range(9)] if force_chunk else [(0, 9)])
    if dtype == torch.float8_e4m3fn:
        assert_upstream_selection(
            expected, args[4], args[8], dtype=dtype, head_dim=head_dim
        )
    else:
        # Small D produces exact ties. Check scores/count/unique indices rather
        # than requiring the same choice among tied boundary indices.
        assert_selection(expected, args[4], args[8])


@pytest.mark.parametrize(
    "num_heads,head_dim,dtype",
    [
        (heads, dim, dtype)
        for dtype in (torch.bfloat16, torch.float8_e4m3fn)
        for heads, dim in ((1, 32), (3, 64), (5, 96), (8, 256))
        # Original vLLM BF16 H8/D256 exceeds H800 shared memory. Keep this
        # shape in our independent default/one-row tests above.
        if not (dtype == torch.bfloat16 and (heads, dim) == (8, 256))
    ],
)
def test_parameterized_against_vllm(sm90, num_heads, head_dim, dtype):
    reference = vllm_reference()
    args = make_inputs(
        (3, 7),
        (1025, 1537),
        query_base=19,
        dtype=dtype,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    expected = torch_logits(*args[:5])
    native_args = list(args)
    native_args[8] = torch.empty_like(args[8])
    reference.qsa_select_paged_prefill(*native_args)
    op(*args)
    for output in (native_args[8], args[8]):
        if dtype == torch.float8_e4m3fn:
            assert_upstream_selection(
                expected, args[4], output, dtype=dtype, head_dim=head_dim
            )
        else:
            assert_selection(expected, args[4], output)
    assert_native_selection(
        expected, args[4], args[8], native_args[8], dtype=dtype, head_dim=head_dim
    )


@pytest.mark.parametrize(
    "rows,context,k",
    [
        (1, 4096, 1024),
        (1, 8192, 2048),
        (65, 8192, 512),
        (65, 8192, 1024),
        (65, 8192, 2048),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_native_vllm_topk_dispatch(sm90, monkeypatch, rows, context, k, dtype):
    # One unsplit chunk: <=64 rows uses native cooperative_topk; 65 rows
    # uses native persistent_topk, with both additional supported K values.
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    reference = vllm_reference()
    monkeypatch.setattr(reference.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 512)
    args = make_inputs((rows,), (context,), k=k, dtype=dtype, query_base=53)
    expected = torch_logits(*args[:5])
    native_args = list(args)
    native_args[8] = torch.empty_like(args[8])
    reference.qsa_select_paged_prefill(*native_args)
    op(*args)
    for output in (native_args[8], args[8]):
        if dtype == torch.float8_e4m3fn:
            assert_upstream_selection(expected, args[4], output, dtype=dtype)
        else:
            assert_selection(expected, args[4], output)
    assert_native_selection(
        expected, args[4], args[8], native_args[8], dtype=dtype, head_dim=128
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_native_vllm_empty_and_select_all(sm90, dtype):
    reference = vllm_reference()
    empty = make_inputs((0,), (17,), dtype=dtype, num_heads=3, head_dim=64)
    assert reference.qsa_select_paged_prefill(*empty) is None
    assert op(*empty) is None
    assert empty[8].numel() == 0
    assert_native_selection(
        torch_logits(*empty[:5]), empty[4], empty[8], empty[8], dtype=dtype, head_dim=64
    )
    args = make_inputs(
        (5,), (17,), visible=[0, 1, 3, 16, 17], dtype=dtype, num_heads=3, head_dim=64
    )
    native_args = list(args)
    native_args[8] = torch.empty_like(args[8])
    reference.qsa_select_paged_prefill(*native_args)
    op(*args)
    torch.testing.assert_close(
        args[8].sort().values, native_args[8].sort().values, rtol=0, atol=0
    )
    assert_native_selection(
        torch_logits(*args[:5]),
        args[4],
        args[8],
        native_args[8],
        dtype=dtype,
        head_dim=64,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "num_heads,head_dim", [(4, 8), (4, 128), (1, 7), (3, 24), (2, 40)]
)
def test_every_autotune_candidate(sm90, dtype, num_heads, head_dim):
    args = make_inputs(
        (3, 0, 6),
        (1025, 16, 1537),
        query_base=97,
        dtype=dtype,
        head_dim=head_dim,
        num_heads=num_heads,
    )
    width = args[2].shape[1] * args[1].shape[1]
    expected = torch_logits(*args[:5])
    kernel = op_module._qsa_prefill_score_kernel
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    configs = flaggems_vllm.runtime.get_tuned_config(
        "qsa_select_paged_prefill_reuse_score"
    )
    assert configs
    for config in configs:
        scores = torch.full((args[0].shape[0], width), 12345.0, device="cuda")
        meta = config.kwargs
        grid = (
            args[2].shape[0],
            triton.cdiv(args[7], meta["TILE_R"]),
            triton.cdiv(width, meta["BLOCK_N"] * meta["K_TILES"]),
        )
        kernel[grid](
            *args[:5],
            scores,
            args[0].stride(0),
            args[0].stride(1),
            args[1].stride(0),
            args[1].stride(1),
            args[2].stride(0),
            args[2].stride(1),
            args[0].shape[0],
            0,
            args[7],
            width,
            args[2].shape[0],
            args[1].shape[0],
            args[2].shape[1],
            PAGE_SIZE=args[1].shape[1],
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            IS_FP8=dtype == torch.float8_e4m3fn,
            **meta,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        mask = torch.arange(width, device="cuda")[None, :] < args[4][:, None]
        fp8 = dtype == torch.float8_e4m3fn
        torch.testing.assert_close(
            scores[mask],
            expected[mask],
            rtol=1e-3 if fp8 else 2e-4,
            atol=1e-3 * head_dim**0.5 if fp8 else 2e-3,
        )


def test_workspace_reuse_and_cross_request_chunks(sm90, monkeypatch):
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "1")
    # width 8192 => 32 rows/chunk, splitting both requests and the last chunk.
    args = make_inputs((39, 34), (4096, 8192), query_base=113)
    expected = torch_logits(*args[:5])
    op(*args)
    assert_selection(expected, args[4], args[8])
    # The same buffer may contain larger scores from an earlier request.
    width = args[2].shape[1] * args[1].shape[1]
    workspace = torch.full((32 * width,), 98765.0, device="cuda")
    for offset in range(0, args[0].shape[0], 32):
        count = min(32, args[0].shape[0] - offset)
        scores = _scores(args, offset, count, workspace)
        mask = (
            torch.arange(width, device="cuda")[None, :]
            < args[4][offset : offset + count, None]
        )
        torch.testing.assert_close(
            scores[mask], expected[offset : offset + count][mask], rtol=2e-4, atol=2e-3
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("num_heads,head_dim", [(4, 128), (1, 1), (5, 96)])
def test_empty_and_select_all_do_not_launch_score(
    sm90, monkeypatch, dtype, num_heads, head_dim
):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "score/Top-K must not run for an empty or select-all batch"
        )

    monkeypatch.setattr(op_module, "_prefill_logits", forbidden)
    monkeypatch.setattr(op_module, "persistent_topk", forbidden)
    empty = make_inputs(
        (0,), (17,), dtype=dtype, num_heads=num_heads, head_dim=head_dim
    )
    assert op(*empty) is None
    args = make_inputs(
        (5,),
        (17,),
        visible=[0, 1, 3, 16, 17],
        dtype=dtype,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    op(*args)
    for row, count in enumerate([0, 1, 3, 16, 17]):
        torch.testing.assert_close(
            args[8][row, :count], torch.arange(count, device="cuda", dtype=torch.int32)
        )
        assert (args[8][row, count:] == -1).all()


@pytest.mark.parametrize("bad_budget", ["-1", "invalid"])
def test_bad_workspace_budget(sm90, monkeypatch, bad_budget):
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", bad_budget)
    with pytest.raises(ValueError, match="VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"):
        op(*make_inputs((1,), (1025,)))


def test_persistent_topk_workspace_reused_across_chunks(sm90, monkeypatch):
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "1")
    args = make_inputs((39, 34), (4096, 8192), query_base=113)
    expected = torch_logits(*args[:5])
    original = op_module.persistent_topk
    launches = []

    def checked_topk(logits, lengths, output, workspace, *, k, max_seq_len):
        assert workspace.dtype == torch.uint8
        assert workspace.numel() == 1024 * 1024
        assert max_seq_len == logits.shape[1] == 8192
        assert k == 512
        launches.append((logits.shape[0], workspace.data_ptr()))
        return original(
            logits, lengths, output, workspace, k=k, max_seq_len=max_seq_len
        )

    monkeypatch.setattr(op_module, "persistent_topk", checked_topk)
    op(*args)
    assert [rows for rows, _ in launches] == [32, 32, 9]
    assert len({pointer for _, pointer in launches}) == 1
    assert_selection(expected, args[4], args[8])


@pytest.mark.parametrize(
    "case",
    [
        "dtype",
        "mixed_dtype",
        "heads",
        "head_dim",
        "cache_dim",
        "dim_stride",
        "output_stride",
        "visible_dtype",
        "ratio",
        "k",
        "grad",
        "overlap",
    ],
)
def test_invalid_contract(sm90, case):
    args = list(make_inputs((8,), (1025,)))
    error = NotImplementedError
    if case == "dtype":
        args[0] = args[0].float()
        args[1] = args[1].float()
    elif case == "mixed_dtype":
        args[1] = args[1].to(torch.float8_e4m3fn)
        error = ValueError
    elif case == "heads":
        args[0] = args[0][:, :0]
        error = ValueError
    elif case == "head_dim":
        args[0] = args[0][..., :0]
        args[1] = args[1][..., :0]
        error = ValueError
    elif case == "cache_dim":
        args[1] = args[1][..., :64]
    elif case == "dim_stride":
        args[0] = torch.empty((8, 4, 256), device="cuda", dtype=torch.bfloat16)[
            ..., ::2
        ]
    elif case == "output_stride":
        args[8] = torch.empty((8, 1024), device="cuda", dtype=torch.int32)[:, ::2]
    elif case == "visible_dtype":
        args[4] = args[4].long()
        error = ValueError
    elif case == "ratio":
        args[6] = 0
        error = ValueError
    elif case == "k":
        args[5] = 256 * args[6]
    elif case == "grad":
        args[0].requires_grad_(True)
    elif case == "overlap":
        args[4] = args[8].view(-1)[:8]
        error = ValueError
    with pytest.raises(error):
        op(*args)


def test_cpu_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match="NVIDIA CUDA SM90"):
        op(*make_inputs((1,), (17,), device="cpu"))


@pytest.mark.qsa_select_paged_prefill
def test_top_level_export():
    assert op is op_module.qsa_select_paged_prefill
    assert "qsa_select_paged_prefill" not in dict(flaggems_vllm._FULL_CONFIG)


def test_against_pinned_vllm(sm90, monkeypatch):
    reference = vllm_reference()
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "1")
    # Also handle a vLLM build that has already cached its environment registry.
    monkeypatch.setattr(reference.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 1)
    args = make_inputs((39, 34), (4096, 8192), query_base=113)
    expected = torch_logits(*args[:5])
    vllm_output = torch.empty_like(args[8])
    vllm_args = list(args)
    vllm_args[8] = vllm_output
    reference.qsa_select_paged_prefill(*vllm_args)
    op(*args)
    assert_selection(expected, args[4], vllm_output)
    assert_selection(expected, args[4], args[8])
    assert_native_selection(
        expected,
        args[4],
        args[8],
        vllm_output,
        dtype=args[0].dtype,
        head_dim=args[0].shape[2],
    )


@pytest.mark.parametrize("seq_len_slack", [0, 1792])
@pytest.mark.parametrize("force_chunk", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_upstream_prefill_scenarios(
    sm90, monkeypatch, seq_len_slack, force_chunk, dtype
):
    monkeypatch.setenv(
        "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "0" if force_chunk else "512"
    )
    args = make_upstream_prefill_inputs(seq_len_slack=seq_len_slack, dtype=dtype)
    expected = upstream_torch_logits(*args[:5])
    chunks = []
    score = op_module._prefill_logits

    def checked_score(*score_args, **score_kwargs):
        logits = score(*score_args, **score_kwargs)
        width, offset, count = score_args[-3:]
        chunks.append((width, offset, count))
        mask = (
            torch.arange(width, device=args[0].device)[None, :]
            < args[4][offset : offset + count, None]
        )
        torch.testing.assert_close(
            logits[mask],
            expected[offset : offset + count, :width][mask],
            rtol=1e-3 if dtype == torch.float8_e4m3fn else 2e-4,
            atol=1e-3 * 128**0.5 if dtype == torch.float8_e4m3fn else 2e-3,
        )
        return logits

    monkeypatch.setattr(op_module, "_prefill_logits", checked_score)
    op(*args)
    # slack=0 clips inside a page: 1280 % 24 != 0. With slack=1792 the
    # max_seq_len bound exceeds the 1536-column page-table capacity.
    expected_width = 1280 if seq_len_slack == 0 else 1536
    expected_chunks = (
        [(expected_width, row, 1) for row in range(36)]
        if force_chunk
        else [(expected_width, 0, 36)]
    )
    assert chunks == expected_chunks
    assert_upstream_selection(expected, args[4], args[8], dtype=dtype)


@pytest.mark.parametrize("seq_len_slack", [0, 1792])
@pytest.mark.parametrize("force_chunk", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_upstream_prefill_against_vllm(
    sm90, monkeypatch, seq_len_slack, force_chunk, dtype
):
    reference = vllm_reference()
    budget = 0 if force_chunk else 512
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", str(budget))
    monkeypatch.setattr(reference.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", budget)
    args = make_upstream_prefill_inputs(seq_len_slack=seq_len_slack, dtype=dtype)
    expected = upstream_torch_logits(*args[:5])
    baseline_args = list(args)
    baseline_args[8] = torch.empty_like(args[8])
    reference.qsa_select_paged_prefill(*baseline_args)
    op(*args)
    assert_native_selection(
        expected, args[4], args[8], baseline_args[8], dtype=dtype, head_dim=128
    )
    assert_upstream_selection(expected, args[4], baseline_args[8], dtype=dtype)
    assert_upstream_selection(expected, args[4], args[8], dtype=dtype)
    if dtype == torch.bfloat16:
        torch.testing.assert_close(
            args[8].sort().values, baseline_args[8].sort().values, rtol=0, atol=0
        )


@pytest.mark.parametrize("decode_query_len", [1, 2, 3, 4])
@pytest.mark.parametrize("compare_vllm", [False, True], ids=["torch", "vllm"])
def test_upstream_split_prefill(sm90, decode_query_len, compare_vllm):
    reference = vllm_reference() if compare_vllm else None
    args = make_upstream_split_prefill_inputs(decode_query_len)
    expected = upstream_torch_logits(*args[:5])
    op(*args)
    assert_upstream_selection(expected, args[4], args[8], dtype=torch.bfloat16)
    if reference is not None:
        baseline_args = list(args)
        baseline_args[8] = torch.empty_like(args[8])
        reference.qsa_select_paged_prefill(*baseline_args)
        assert_native_selection(
            expected,
            args[4],
            args[8],
            baseline_args[8],
            dtype=torch.bfloat16,
            head_dim=128,
        )
        assert_upstream_selection(
            expected, args[4], baseline_args[8], dtype=torch.bfloat16
        )
        torch.testing.assert_close(
            args[8].sort().values, baseline_args[8].sort().values, rtol=0, atol=0
        )


@pytest.mark.parametrize("compare_vllm", [False, True], ids=["torch", "vllm"])
def test_upstream_no_complete_blocks_dim8(sm90, compare_vllm):
    reference = vllm_reference() if compare_vllm else None
    args = make_upstream_no_complete_blocks_inputs()
    assert op(*args) is None
    torch.testing.assert_close(args[8], torch.full_like(args[8], -1), rtol=0, atol=0)
    if reference is not None:
        baseline_args = list(args)
        baseline_args[8] = torch.empty_like(args[8])
        reference.qsa_select_paged_prefill(*baseline_args)
        torch.testing.assert_close(args[8], baseline_args[8], rtol=0, atol=0)
        assert_native_selection(
            torch_logits(*args[:5]),
            args[4],
            args[8],
            baseline_args[8],
            dtype=torch.bfloat16,
            head_dim=8,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_dim8_scored_selection(sm90, dtype):
    # D=8 must also work when the score and existing Top-K paths are necessary.
    args = make_inputs((7,), (1025,), dtype=dtype, head_dim=8)
    op(*args)
    assert_selection(
        torch_logits(*args[:5]),
        args[4],
        args[8],
        rtol=1e-3 if dtype == torch.float8_e4m3fn else 2e-4,
        atol=1e-3 * 8**0.5 if dtype == torch.float8_e4m3fn else 2e-3,
    )


def test_noncurrent_cuda_device(sm90):
    if torch.cuda.device_count() < 2 or torch.cuda.get_device_capability(1) != (9, 0):
        pytest.skip("a second SM90 device is needed for device-guard validation")
    previous = torch.cuda.current_device()
    args = make_inputs((5,), (1025,), device="cuda:1")
    op(*args)
    assert torch.cuda.current_device() == previous
    assert_selection(torch_logits(*args[:5]), args[4], args[8])


# Production integration gates deliberately call the official public export.
# Historical trial modules are references only; they never replace this binding.


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_score_cache_separates_runtime_rows(sm90, dtype):
    # Same pointers/strides/dtypes/width, distinct existing num_rows tuner keys.
    # Exercise the real entry miss/hit paths without changing JIT specialization.
    args = make_inputs((81,), (1025,), query_base=97, dtype=dtype, padded=True)
    width = 1088
    expected = torch_logits(*args[:5], width=width)
    workspace = torch.empty((64, width), device=args[0].device, dtype=torch.float32)
    entry = (
        op_module._qsa_prefill_fp8_store_score_kernel
        if dtype == torch.float8_e4m3fn
        else op_module._qsa_prefill_score_kernel
    )
    cache = entry.kernel_cache[torch.cuda.current_device()]
    saved = dict(cache)
    cache.clear()
    keys = []
    try:
        for offset, count in ((0, 64), (64, 17), (0, 64)):
            actual = op_module._prefill_logits(
                *args[:5], workspace, args[7], width, offset, count
            )
            mask = (
                torch.arange(width, device=args[0].device)[None, :]
                < args[4][offset : offset + count, None]
            )
            torch.testing.assert_close(
                actual[mask],
                expected[offset : offset + count][mask],
                rtol=1e-3 if dtype == torch.float8_e4m3fn else 2e-4,
                atol=1e-3 * 128**0.5 if dtype == torch.float8_e4m3fn else 2e-3,
            )
            keys.append(set(cache))
        assert len(keys[0]) == 1
        assert len(keys[1]) == 2 and keys[0] < keys[1]
        assert keys[2] == keys[1]
        assert op is op_module.qsa_select_paged_prefill
    finally:
        cache.clear()
        cache.update(saved)


def test_fp8_store_production_dispatch_and_canaries(sm90, monkeypatch):
    # The real production launcher must choose only eligible buffers. Metadata
    # includes all tail residues, zero-visible rows and chunk offsets.
    for dtype, width, shift, expected_route in (
        (torch.float8_e4m3fn, 1088, 0, "fp8"),
        (torch.float8_e4m3fn, 1088, 1, "original"),
        (torch.float8_e4m3fn, 1089, 0, "original"),
        (torch.bfloat16, 1088, 0, "original"),
    ):
        args = make_inputs(
            (17,),
            (1025,),
            dtype=dtype,
            padded=True,
            visible=[
                0,
                1,
                2,
                3,
                4,
                5,
                511,
                512,
                513,
                514,
                515,
                1021,
                1022,
                1023,
                1024,
                1025,
                1025,
            ],
            query_base=97,
        )
        expected = torch_logits(*args[:5], width=width)
        routes = []

        class Capture:
            def __init__(self, entry, route):
                self.entry, self.route = entry, route

            def __getitem__(self, grid):
                routes.append(self.route)
                return self.entry[grid]

        with monkeypatch.context() as patcher:
            for name, route in (
                ("_qsa_prefill_score_kernel", "original"),
                ("_qsa_prefill_fp8_store_score_kernel", "fp8"),
                ("_qsa_prefill_compact_score_kernel", "compact"),
            ):
                patcher.setattr(
                    op_module, name, Capture(getattr(op_module, name), route)
                )
            for offset, count in ((0, 17), (1, 15), (16, 1)):
                backing = torch.full((count * width + 32,), 12345.0, device="cuda")
                start = 16 + shift
                workspace = backing[start : start + count * width].view(count, width)
                actual = op_module._prefill_logits(
                    *args[:5], workspace, args[7], width, offset, count
                )
                mask = (
                    torch.arange(width, device="cuda")[None, :]
                    < args[4][offset : offset + count, None]
                )
                torch.testing.assert_close(
                    actual[mask],
                    expected[offset : offset + count][mask],
                    rtol=1e-3 if dtype == torch.float8_e4m3fn else 2e-4,
                    atol=1e-3 * 128**0.5 if dtype == torch.float8_e4m3fn else 2e-3,
                )
                assert (actual[~mask] == 12345.0).all()
                assert (backing[:start] == 12345.0).all()
                assert (backing[start + count * width :] == 12345.0).all()
            assert routes == [expected_route] * 3
