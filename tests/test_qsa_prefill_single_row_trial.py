# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""On-card gates for the benchmark-only single-row vector candidate."""

from importlib import import_module

import pytest
import torch
import triton

import flaggems_vllm
from tools import qsa_prefill_single_row_trial as trial

from .qsa_select_prefill_utils import (
    assert_selection,
    assert_upstream_selection,
    make_inputs,
    torch_logits,
)

pytestmark = pytest.mark.qsa_select_paged_prefill
op_module = import_module("flaggems_vllm.ops.qsa_select_paged_prefill")


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "heads,dim", [(1, 1), (1, 7), (4, 8), (3, 24), (2, 40), (4, 128), (5, 96), (8, 256)]
)
def test_vector_every_config_and_request_boundary(sm90, dtype, heads, dim):
    args = make_inputs(
        (2, 0, 3),
        (1025, 16, 1537),
        query_base=97,
        visible=[0, 511, 512, 513, 1537],
        padded=True,
        dtype=dtype,
        num_heads=heads,
        head_dim=dim,
    )
    if heads % 2:
        # Cover int64 page IDs/offsets while preserving a strided page table.
        args = list(args)
        table = torch.full(
            (args[2].shape[0], args[2].shape[1] * 2),
            -1,
            dtype=torch.int64,
            device="cuda",
        )
        table[:, ::2].copy_(args[2])
        args[2] = table[:, ::2]
        args[3] = args[3].to(torch.int64)
    expected = torch_logits(*args[:5])
    width = expected.shape[1]
    kernel = trial._qsa_prefill_single_row_score_kernel
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    configs = flaggems_vllm.runtime.get_tuned_config(
        "qsa_select_paged_prefill_single_row_score"
    )
    assert len(configs) == 4
    for config in configs:
        for offset in range(5):
            scores = torch.full((1, width), 12345.0, device="cuda")
            grid = (3, 1, triton.cdiv(width, config.kwargs["BLOCK_N"]))
            kernel[grid](
                *args[:5],
                scores,
                args[0].stride(0),
                args[0].stride(1),
                args[1].stride(0),
                args[1].stride(1),
                args[2].stride(0),
                args[2].stride(1),
                1,
                offset,
                args[7],
                width,
                3,
                args[1].shape[0],
                args[2].shape[1],
                PAGE_SIZE=args[1].shape[1],
                NUM_HEADS=heads,
                HEAD_DIM=dim,
                IS_FP8=dtype == torch.float8_e4m3fn,
                **config.kwargs,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
            live = torch.arange(width, device="cuda") < args[4][offset]
            fp8 = dtype == torch.float8_e4m3fn
            torch.testing.assert_close(
                scores[0, live],
                expected[offset, live],
                rtol=1e-3 if fp8 else 2e-4,
                atol=1e-3 * dim**0.5 if fp8 else 2e-3,
            )
            assert (scores[0, ~live] == 12345.0).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("budget", ["512", "1", "0"])
def test_trial_full_entry_preserves_production_and_selection(
    sm90, monkeypatch, dtype, budget
):
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", budget)
    args = make_inputs((33, 32), (4096, 8192), query_base=113, dtype=dtype)
    expected = torch_logits(*args[:5])
    original = op_module._prefill_logits
    candidate = trial.candidate_module(op_module)
    assert (
        candidate.qsa_select_paged_prefill.__code__
        is op_module.qsa_select_paged_prefill.__code__
    )
    assert candidate.persistent_topk is op_module.persistent_topk
    for entry in (
        op_module.qsa_select_paged_prefill,
        candidate.qsa_select_paged_prefill,
    ):
        assert entry(*args) is None
        if dtype == torch.float8_e4m3fn:
            assert_upstream_selection(
                expected, args[4], args[8], dtype=dtype, head_dim=128
            )
        else:
            assert_selection(expected, args[4], args[8])
    assert op_module._prefill_logits is original
