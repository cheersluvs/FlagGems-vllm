# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Loop unroll2: both depths, every choice and real private entry."""

import itertools
import types

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.utils import libtuner
from tools import qsa_prefill_loop_unroll_trial as trial

from . import qsa_select_prefill_utils as helpers
from . import test_qsa_prefill_page_table as page
from . import test_qsa_prefill_query_split as common
from .test_qsa_prefill_compact_trial import check_score_kernel

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
    modes = (1, 2) if dtype == torch.float8_e4m3fn else (1,)
    for stages, parts in itertools.product((2, 1), modes):
        jit = candidate._qsa_prefill_score_kernel.fn.fn

        class WithMode:
            def __getitem__(self, grid):
                def launch(*a, **kw):
                    # Force each depth; do not inherit helper stage2 silently.
                    kw["num_stages"] = stages
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
    modes = (1, 2) if dtype == torch.float8_e4m3fn else (1,)
    for stages, parts in itertools.product((2, 1), modes):
        candidate = trial.candidate_module(common.production)
        entry = candidate._qsa_prefill_score_kernel
        configs = [
            c
            for c in entry.fn.configs
            if c.kwargs["TILE_R"] == 64
            and c.kwargs["K_TILES"] == 16
            and c.num_warps == 4
            and c.kwargs["QUERY_PARTS"] == parts
            and c.num_stages == stages
        ]
        assert len(configs) == 1
        candidate._qsa_prefill_score_kernel = type(entry)(
            libtuner(configs=configs, key=list(entry.fn.keys), strategy="default")(
                entry.fn.fn
            )
        )
        page.test_page_table_full_entry(sm90, candidate, dtype, monkeypatch)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
@pytest.mark.parametrize(
    "page_size", [16, 64, 196], ids=["page16", "page64", "page196"]
)
def test_page_table_score(sm90, candidate, dtype, page_size):
    modes = (1, 2) if dtype == torch.float8_e4m3fn else (1,)
    for stages, parts in itertools.product((2, 1), modes):
        jit = candidate._qsa_prefill_score_kernel.fn.fn

        class WithMode:
            def __getitem__(self, grid):
                def launch(*a, **kw):
                    # Force each depth; do not inherit helper stage2 silently.
                    kw["num_stages"] = stages
                    return jit[grid](*a, **kw, QUERY_PARTS=parts)

                return launch

        view = types.SimpleNamespace(
            _qsa_prefill_score_kernel=types.SimpleNamespace(
                fn=types.SimpleNamespace(fn=WithMode())
            )
        )
        page.test_page_table_score(sm90, view, dtype, page_size)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_page_table_full_entry(sm90, candidate, dtype, monkeypatch):
    page.test_page_table_full_entry(sm90, candidate, dtype, monkeypatch)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("heads,dim", [(4, 128), (3, 24), (5, 96), (1, 8)])
def test_combined_rectangular_every_config(sm90, candidate, dtype, heads, dim):
    from tools.qsa_prefill_query_joint_policy import prune_configs

    configs = flaggems_vllm.runtime.get_tuned_config(trial.CONFIG_GROUP)
    configs = prune_configs(configs, dict(IS_FP8=dtype == torch.float8_e4m3fn))
    assert len(configs) == (34 if dtype == torch.float8_e4m3fn else 20)
    assert {c.num_stages for c in configs} == {1, 2}
    check_score_kernel(
        dtype, heads, dim, candidate._qsa_prefill_score_kernel, False, configs=configs
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("heads,dim", [(4, 128), (3, 24), (5, 96)])
def test_combined_compact_every_config(sm90, candidate, dtype, heads, dim):
    check_score_kernel(
        dtype, heads, dim, candidate._qsa_prefill_compact_score_kernel, True
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("budget", ["512", "1", "0"])
@pytest.mark.parametrize("compact", [False, True])
def test_combined_full_entry(sm90, monkeypatch, dtype, budget, compact):
    from tools import qsa_prefill_cache_granularity as capture

    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", budget)
    queries = (33, 0, 1, 1, 1, 1, 1, 12) if compact else (33, 0, 1)
    contexts = (4096, 16) + (8192,) * 6 if compact else (4096, 16, 8192)
    args = helpers.make_inputs(
        queries, contexts, query_base=113, dtype=dtype, padded=True
    )
    expected = helpers.torch_logits(*args[:5])
    module = trial.candidate_module(common.production)
    pointer = args[8].data_ptr()
    assert module.qsa_select_paged_prefill(*args) is None
    assert args[8].data_ptr() == pointer
    _, traces = capture.observe(module, args)
    assert any(t["route"] == "compact" for t in traces) == (compact and budget == "512")
    baseline_args = (*args[:8], torch.empty_like(args[8]), args[9])
    assert common.production.qsa_select_paged_prefill(*baseline_args) is None
    for selected in (args[8], baseline_args[8]):
        if dtype == torch.float8_e4m3fn:
            helpers.assert_upstream_selection(expected, args[4], selected, dtype=dtype)
        else:
            helpers.assert_selection(expected, args[4], selected)
    helpers.assert_native_selection(
        expected, args[4], args[8], baseline_args[8], dtype=dtype, head_dim=128
    )
    assert module.persistent_topk is common.production.persistent_topk
    assert (
        module.qsa_select_paged_prefill.__code__
        is common.production.qsa_select_paged_prefill.__code__
    )
    assert (
        flaggems_vllm.qsa_select_paged_prefill
        is common.production.qsa_select_paged_prefill
    )
