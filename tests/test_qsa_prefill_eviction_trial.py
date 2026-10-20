# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""On-card gates for the K-load hint and baseline-matched configurations."""

import pytest
import torch

from flaggems_vllm import runtime
from tools import qsa_prefill_eviction_trial as trial
from tools import qsa_prefill_observed_workloads as workloads

from . import test_qsa_prefill_compact_trial as checks

pytestmark = pytest.mark.qsa_select_paged_prefill
sm90 = checks.sm90


@pytest.mark.parametrize("case", workloads.observed_cases(), ids=lambda c: c["name"])
def test_eviction_observed_against_native(sm90, monkeypatch, case):
    checks.check_observed_envelope(monkeypatch, case, trial.candidate_module)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("heads,dim", [(4, 128), (3, 24), (5, 96), (1, 8)])
def test_eviction_every_config_and_chunk_boundary(sm90, dtype, heads, dim):
    configs = runtime.get_tuned_config(trial.CONFIG_NAME)
    assert len(configs) == 8
    checks.check_score_kernel(
        dtype,
        heads,
        dim,
        trial._qsa_prefill_evict_first_score_kernel,
        False,
        configs=configs,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("budget", ["512", "1", "0"])
def test_eviction_full_entry(sm90, monkeypatch, dtype, budget):
    checks.check_full_entry(
        monkeypatch,
        dtype,
        budget,
        trial.candidate_module,
        queries=(33, 0, 1),
        contexts=(4096, 16, 8192),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_eviction_matched_baseline_config(sm90, monkeypatch, dtype):
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "0")
    args = checks.helpers.make_inputs(
        (7, 0, 1),
        (4096, 16, 8192),
        query_base=113,
        dtype=dtype,
    )
    expected = checks.helpers.torch_logits(*args[:5])
    candidate = trial.candidate_module(checks.production)
    candidate.retained_baseline.qsa_select_paged_prefill(*args)
    baseline = checks.stages.support.score_configs(candidate.retained_baseline)
    record = trial.freeze_baseline(candidate)
    assert record["status"] == "matched"
    candidate.qsa_select_paged_prefill(*args)
    trial.verify_matched(candidate, record)
    assert checks.stages.support.score_configs(candidate.retained_baseline) == baseline
    assert candidate.persistent_topk is candidate.retained_baseline.persistent_topk
    if dtype == torch.float8_e4m3fn:
        checks.helpers.assert_upstream_selection(
            expected, args[4], args[8], dtype=dtype, head_dim=128
        )
    else:
        checks.helpers.assert_selection(expected, args[4], args[8])
