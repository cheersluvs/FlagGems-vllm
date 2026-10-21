# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Force both full-query and split modes through the existing on-card gates."""

import types

import pytest
import torch

from flaggems_vllm.utils import libtuner
from tools import qsa_prefill_query_joint_trial as trial

from . import test_qsa_prefill_page_table as page
from . import test_qsa_prefill_query_split as common

pytestmark = pytest.mark.qsa_select_paged_prefill
sm90 = common.sm90


@pytest.fixture(scope="module")
def candidate():
    return trial.candidate_module(common.production)


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
    for parts in (1, 2) if dtype == torch.float8_e4m3fn else (1,):
        jit = candidate._qsa_prefill_score_kernel.fn.fn

        class WithMode:
            def __getitem__(self, grid):
                def launch(*a, **kw):
                    return jit[grid](*a, **kw, QUERY_PARTS=parts)

                return launch

        view = types.SimpleNamespace(
            _qsa_prefill_score_kernel=types.SimpleNamespace(
                fn=types.SimpleNamespace(fn=WithMode())
            )
        )
        common.test_query_split_score(sm90, view, dtype, heads, dim)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_query_split_full_entry(sm90, dtype, monkeypatch):
    for parts in (1, 2) if dtype == torch.float8_e4m3fn else (1,):
        candidate = trial.candidate_module(common.production)
        entry = candidate._qsa_prefill_score_kernel
        configs = [
            c
            for c in entry.fn.configs
            if c.kwargs["TILE_R"] == 64
            and c.kwargs["K_TILES"] == 16
            and c.num_warps == 4
            and c.kwargs["QUERY_PARTS"] == parts
        ]
        assert len(configs) == 1
        candidate._qsa_prefill_score_kernel = type(entry)(
            libtuner(configs=configs, key=list(entry.fn.keys), strategy="default")(
                entry.fn.fn
            )
        )
        page.test_page_table_full_entry(sm90, candidate, dtype, monkeypatch)
