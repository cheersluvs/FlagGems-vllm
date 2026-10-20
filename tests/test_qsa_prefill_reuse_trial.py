# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""On-card gates for added work-distribution choices; score JIT is unchanged."""

import pytest
import torch

from flaggems_vllm import runtime
from tools import qsa_prefill_observed_workloads as workloads
from tools import qsa_prefill_reuse_trial as trial

from . import test_qsa_prefill_compact_trial as checks

pytestmark = pytest.mark.qsa_select_paged_prefill
sm90 = checks.sm90


@pytest.mark.parametrize("case", workloads.observed_cases(), ids=lambda c: c["name"])
def test_reuse_observed_against_native(sm90, monkeypatch, case):
    checks.check_observed_envelope(monkeypatch, case, trial.candidate_module)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("heads,dim", [(4, 128), (3, 24), (5, 96), (1, 8)])
def test_reuse_new_configs_and_chunk_boundary(sm90, dtype, heads, dim):
    configs = runtime.get_tuned_config("qsa_select_paged_prefill_reuse_score")
    assert len(configs) == 8
    # The unchanged first six are exercised by the existing key-major gate.
    checks.check_score_kernel(
        dtype, heads, dim, trial.score_entry(), False, configs=configs[6:]
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("budget", ["512", "1", "0"])
def test_reuse_full_entry(sm90, monkeypatch, dtype, budget):
    checks.check_full_entry(
        monkeypatch,
        dtype,
        budget,
        trial.candidate_module,
        queries=(33, 0, 1),
        contexts=(4096, 16, 8192),
    )
