# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""On-card gates for observed envelopes and compact request scheduling."""

import pytest
import torch
import triton

import flaggems_vllm
from tools import benchmark_qsa_prefill_triton_stages as stages
from tools import qsa_prefill_compact_trial as trial
from tools import qsa_prefill_observed_workloads as workloads
from tools.qsa_prefill_triton_ab import frozen_production

from . import qsa_select_prefill_utils as helpers

pytestmark = pytest.mark.qsa_select_paged_prefill
production = frozen_production()


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


@pytest.mark.parametrize("case", workloads.observed_cases(), ids=lambda c: c["name"])
def test_observed_envelopes_against_native(sm90, monkeypatch, case):
    check_observed_envelope(monkeypatch, case, trial.candidate_module)


def check_observed_envelope(monkeypatch, case, factory):
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", str(case["budget_mb"]))
    native = helpers.vllm_reference()
    monkeypatch.setattr(
        native.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", case["budget_mb"]
    )
    args = workloads.make_inputs(case, helpers, torch)
    baseline_args = (*args[:8], stages.native_output(args[8], torch), args[9])
    candidate = factory(production)
    for module in (production, candidate):
        functions = dict(
            baseline=lambda: native.qsa_select_paged_prefill(*baseline_args),
            candidate=lambda: module.qsa_select_paged_prefill(*args),
        )
        # Independent FP32 sampled rows include every request boundary; native
        # and candidate selections retain the existing score/tie tolerances.
        checked = stages.validate_pipeline(
            functions, baseline_args, args, [], torch, helpers
        )
        assert checked >= len(case["query_lengths"])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("heads,dim", [(4, 128), (3, 24), (5, 96)])
def test_compact_every_config_and_chunk_boundary(sm90, dtype, heads, dim):
    check_score_kernel(dtype, heads, dim, trial._qsa_prefill_compact_score_kernel, True)


def check_score_kernel(
    dtype, heads, dim, kernel, compact_grid, configs=None, width_multiple=1
):
    queries = (0, 7, 0, 33, 1, 65, 0)
    rows = sum(queries)
    visible = [0, 1, 511, 512, 513, 1025, 1537] + [1537] * (rows - 7)
    args = list(
        helpers.make_inputs(
            queries,
            (1537,) * len(queries),
            query_base=97,
            visible=visible,
            padded=True,
            dtype=dtype,
            num_heads=heads,
            head_dim=dim,
        )
    )
    # Preserve a strided page table while covering 64-bit metadata.
    table = torch.full(
        (len(queries), args[2].shape[1] * 2), -1, dtype=torch.int64, device="cuda"
    )
    table[:, ::2].copy_(args[2])
    args[2], args[3] = table[:, ::2], args[3].to(torch.int64)
    expected = helpers.torch_logits(*args[:5])
    width = triton.cdiv(expected.shape[1], width_multiple) * width_multiple
    if width != expected.shape[1]:
        expected = helpers.torch_logits(*args[:5], width=width)
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    if configs is None:
        configs = flaggems_vllm.runtime.get_tuned_config(
            "qsa_select_paged_prefill_score"
        )
        assert len(configs) == 6
    for config in configs:
        for offset, count in (
            (0, rows),
            (1, 1),
            (6, 4),
            (35, 9),
            (40, 3),
            (rows - 1, 1),
        ):
            scores = torch.full((count, width), 12345.0, device="cuda")
            meta = config.kwargs
            kernel[
                (
                    (
                        count // meta["TILE_R"] + len(queries)
                        if compact_grid
                        else len(queries)
                    ),
                    (
                        1
                        if compact_grid
                        else triton.cdiv(min(count, args[7]), meta["TILE_R"])
                    ),
                    triton.cdiv(width, meta["BLOCK_N"] * meta["K_TILES"]),
                )
            ](
                *args[:5],
                scores,
                args[0].stride(0),
                args[0].stride(1),
                args[1].stride(0),
                args[1].stride(1),
                args[2].stride(0),
                args[2].stride(1),
                count,
                offset,
                args[7],
                width,
                len(queries),
                args[1].shape[0],
                args[2].shape[1],
                PAGE_SIZE=args[1].shape[1],
                NUM_HEADS=heads,
                HEAD_DIM=dim,
                IS_FP8=dtype == torch.float8_e4m3fn,
                **meta,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
            mask = (
                torch.arange(width, device="cuda")[None, :]
                < args[4][offset : offset + count, None]
            )
            fp8 = dtype == torch.float8_e4m3fn
            torch.testing.assert_close(
                scores[mask],
                expected[offset : offset + count][mask],
                rtol=1e-3 if fp8 else 2e-4,
                atol=1e-3 * dim**0.5 if fp8 else 2e-3,
            )
            assert (scores[~mask] == 12345.0).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("budget", ["512", "1", "0"])
def test_compact_full_entry(sm90, monkeypatch, dtype, budget):
    check_full_entry(monkeypatch, dtype, budget, trial.candidate_module)


def check_full_entry(monkeypatch, dtype, budget, factory, queries=None, contexts=None):
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", budget)
    args = helpers.make_inputs(
        queries if queries is not None else (33, 0, 1, 1, 1, 1, 1, 12),
        contexts if contexts is not None else (4096, 16) + (8192,) * 6,
        query_base=113,
        dtype=dtype,
    )
    expected = helpers.torch_logits(*args[:5])
    candidate = factory(production)
    original = production._prefill_logits
    assert (
        candidate._qsa_prefill_score_kernel is not production._qsa_prefill_score_kernel
    )
    assert all(
        a is not b
        for a, b in zip(
            candidate._qsa_prefill_score_kernel.kernel_cache,
            production._qsa_prefill_score_kernel.kernel_cache,
        )
    )
    for entry in (
        production.qsa_select_paged_prefill,
        candidate.qsa_select_paged_prefill,
    ):
        before = stages.support.score_configs(production)
        assert entry(*args) is None
        if entry is candidate.qsa_select_paged_prefill:
            assert stages.support.score_configs(production) == before
        if dtype == torch.float8_e4m3fn:
            helpers.assert_upstream_selection(
                expected, args[4], args[8], dtype=dtype, head_dim=128
            )
        else:
            helpers.assert_selection(expected, args[4], args[8])
    assert production._prefill_logits is original
    assert candidate.persistent_topk is production.persistent_topk
