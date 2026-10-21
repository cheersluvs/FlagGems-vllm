# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""H800 fixed-tile score and forced split public-entry correctness gates."""

import pytest
import torch
import triton

from flaggems_vllm.utils import libtuner
from tools import qsa_prefill_query_split_trial as trial

from . import qsa_select_prefill_utils as helpers
from . import test_qsa_prefill_page_table as common

pytestmark = pytest.mark.qsa_select_paged_prefill
production = common.production
sm90 = common.sm90


@pytest.fixture(scope="module")
def candidate():
    return trial.candidate_module(production)


@pytest.mark.parametrize(
    "dtype,heads,dim",
    [
        (torch.bfloat16, 4, 128),
        (torch.float8_e4m3fn, 4, 128),
        (torch.float8_e4m3fn, 1, 32),
        (torch.float8_e4m3fn, 3, 96),
    ],
    ids=["bf16-h4d128", "fp8-h4d128", "fp8-h1d32", "fp8-h3d96"],
)
def test_query_split_score(sm90, candidate, dtype, heads, dim):
    args = helpers.make_inputs(
        (81, 81),
        (1025, 1025),
        dtype=dtype,
        page_size=196,
        num_heads=heads,
        head_dim=dim,
        padded=True,
        query_base=97,
    )
    width = 1088
    expected = helpers.torch_logits(*args[:5], width=width)
    strides, layout, constants = production._prefill_score_metadata(*args[:3])
    jit = candidate._qsa_prefill_score_kernel.fn.fn
    # Fixed existing choices cover unchanged tiny tiles and both split widths.
    for tile, block, kt, warps in ((8, 64, 4, 4), (32, 128, 4, 8), (64, 64, 16, 4)):
        for offset, rows in ((0, 64), (64, 64), (128, 34)):
            workspace = torch.empty(
                (rows, width), device=args[0].device, dtype=torch.float32
            )
            grid = (
                2,
                triton.cdiv(min(rows, args[7]), tile),
                triton.cdiv(width, block * kt),
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
                K_TILES=kt,
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
                atol=1e-3 * dim**0.5 if dtype == torch.float8_e4m3fn else 2e-3,
            )


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_query_split_full_entry(sm90, dtype, monkeypatch):
    candidate = trial.candidate_module(production)
    entry = candidate._qsa_prefill_score_kernel
    # Force an existing eligible tile so full API checks cannot silently
    # autotune to the original tiny-tile path on this small correctness fixture.
    configs = [
        c
        for c in entry.fn.configs
        if c.kwargs["TILE_R"] == 64 and c.kwargs["K_TILES"] == 16 and c.num_warps == 4
    ]
    assert len(configs) == 1
    candidate._qsa_prefill_score_kernel = type(entry)(
        libtuner(configs=configs, key=list(entry.fn.keys), strategy="default")(
            entry.fn.fn
        )
    )
    common.test_page_table_full_entry(sm90, candidate, dtype, monkeypatch)
