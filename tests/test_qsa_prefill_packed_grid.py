# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""On-card boundary and full-entry gates for the private packed scheduler."""

import importlib

import pytest
import torch
import triton

from tools import benchmark_qsa_prefill_triton_stages as stages
from tools import qsa_prefill_cache_granularity as capture
from tools import qsa_prefill_packed_grid as trial
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "lengths,offset,rows",
    [((81, 81), 37, 64), ((0, 33, 0, 48, 0), 29, 36), ((8, 8, 8, 8), 31, 1)],
)
def test_packed_score_boundaries(sm90, candidate, dtype, lengths, offset, rows):
    args = helpers.make_inputs(
        lengths, (1025,) * len(lengths), dtype=dtype, padded=True, query_base=97
    )
    width = 1088
    expected = helpers.torch_logits(*args[:5], width=width)
    workspace = torch.empty((rows, width), device=args[0].device, dtype=torch.float32)
    strides, layout, constants = production._prefill_score_metadata(*args[:3])
    entry = candidate._qsa_prefill_packed_score_kernel
    entry[
        lambda meta: (
            rows // meta["TILE_R"] + len(lengths),
            1,
            triton.cdiv(width, meta["BLOCK_N"] * meta["K_TILES"]),
        )
    ](
        *args[:5],
        workspace,
        *strides,
        rows,
        offset,
        max(lengths),
        width,
        *layout,
        **constants,
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_packed_full_entry(sm90, candidate, dtype, monkeypatch):
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    native = helpers.vllm_reference()
    monkeypatch.setattr(native.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 512)
    args = helpers.make_inputs(
        (80, 80, 0, 0), (1025,) * 4, dtype=dtype, padded=True, query_base=97
    )
    native_args = (*args[:8], stages.native_output(args[8], torch), args[9])
    native_chunks = ab.observe(native, "_topk", native_args)
    candidate_chunks, traces = capture.observe(candidate, args)
    assert any(t["route"] == "packed" for t in traces)
    functions = ab.bind_stages(native, candidate, native_chunks, candidate_chunks)
    assert (
        stages.validate_pipeline(
            functions["pipeline"], native_args, args, native_chunks, torch, helpers
        )
        > 0
    )
    helpers.assert_upstream_selection(
        helpers.torch_logits(*args[:5]), args[4], args[8], dtype=dtype
    )
    assert candidate.persistent_topk is production.persistent_topk
    assert (
        candidate.qsa_select_paged_prefill.__code__
        is production.qsa_select_paged_prefill.__code__
    )
