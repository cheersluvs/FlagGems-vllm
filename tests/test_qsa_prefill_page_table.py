# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""On-card page-boundary/load-mask and full-entry gates for scalar page IDs."""

import importlib

import pytest
import torch
import triton

from tools import benchmark_qsa_prefill_triton_stages as stages
from tools import qsa_prefill_cache_granularity as capture
from tools import qsa_prefill_page_table_trial as trial
from tools import qsa_prefill_triton_ab as ab

from . import qsa_select_prefill_utils as helpers

pytestmark = pytest.mark.qsa_select_paged_prefill
production = importlib.import_module("flaggems_vllm.ops.qsa_select_paged_prefill")


@pytest.fixture
def sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("NVIDIA SM90 required")
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


@pytest.fixture(scope="module")
def candidate():
    return trial.candidate_module(production)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
@pytest.mark.parametrize(
    "page_size", [16, 64, 196], ids=["page16", "page64", "page196"]
)
def test_page_table_score(sm90, candidate, dtype, page_size):
    args = helpers.make_inputs(
        (81, 81),
        (1025, 1025),
        dtype=dtype,
        page_size=page_size,
        padded=True,
        query_base=97,
    )
    width = 1088
    expected = helpers.torch_logits(*args[:5], width=width)
    strides, layout, constants = production._prefill_score_metadata(*args[:3])
    jit = candidate._qsa_prefill_score_kernel.fn.fn
    # Both block widths from the frozen ten choices; exercise scalar and
    # general branches without exhaustively retuning the whole production set.
    for tile, block, warps in ((8, 64, 4), (32, 128, 8)):
        for offset, rows in ((0, 64), (64, 64), (128, 34)):
            workspace = torch.empty(
                (rows, width), device=args[0].device, dtype=torch.float32
            )
            grid = (
                2,
                triton.cdiv(min(rows, args[7]), tile),
                triton.cdiv(width, block * 4),
            )
            jit[grid](
                *args[:5],
                workspace,
                *strides,
                rows,
                offset,
                args[7],
                width,
                *layout,
                **constants,
                TILE_R=tile,
                BLOCK_N=block,
                K_TILES=4,
                num_warps=warps,
                num_stages=2,
            )
            mask = (
                torch.arange(width, device=args[0].device)[None, :]
                < args[4][offset : offset + rows, None]
            )
            torch.testing.assert_close(
                workspace[mask],
                expected[offset : offset + rows][mask],
                rtol=1e-3 if dtype == torch.float8_e4m3fn else 2e-4,
                atol=1e-3 * 128**0.5 if dtype == torch.float8_e4m3fn else 2e-3,
            )


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_page_table_full_entry(sm90, candidate, dtype, monkeypatch):
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    native = helpers.vllm_reference()
    monkeypatch.setattr(native.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 512)
    args = helpers.make_inputs(
        (80, 80, 0, 0),
        (1025,) * 4,
        dtype=dtype,
        page_size=196,
        padded=True,
        query_base=97,
    )
    # The pinned native kernel accepts stride_table_req but indexes columns
    # with +logical_page. Keep the candidate's strided table and give native
    # an identical logical table with contiguous columns, outside timing.
    assert args[2].stride(1) == 2
    native_table = args[2].contiguous()
    assert native_table.stride(1) == 1
    torch.testing.assert_close(native_table, args[2], rtol=0, atol=0)
    native_args = (
        *args[:2],
        native_table,
        *args[3:8],
        stages.native_output(args[8], torch),
        args[9],
    )
    native_chunks = ab.observe(native, "_topk", native_args)
    expected = helpers.torch_logits(*args[:5])
    stages.validate_reference_selection(
        expected, args[4], native_args[8], args[0], torch, helpers
    )
    candidate_chunks, traces = capture.observe(candidate, args)
    assert any(
        t["route"] == "rectangular" and t["constants"]["BLOCK_N"] <= args[1].shape[1]
        for t in traces
    )
    functions = ab.bind_stages(native, candidate, native_chunks, candidate_chunks)
    assert (
        stages.validate_pipeline(
            functions["pipeline"], native_args, args, native_chunks, torch, helpers
        )
        > 0
    )
    helpers.assert_upstream_selection(expected, args[4], args[8], dtype=dtype)
    assert candidate.persistent_topk is production.persistent_topk
    assert (
        candidate.qsa_select_paged_prefill.__code__
        is production.qsa_select_paged_prefill.__code__
    )
