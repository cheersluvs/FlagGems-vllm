# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""On-card gates for the transposed-dot experiment using frozen tolerances."""

import pytest
import torch

from tools import qsa_prefill_key_major_trial as trial
from tools import qsa_prefill_observed_workloads as workloads

from . import test_qsa_prefill_compact_trial as checks

pytestmark = pytest.mark.qsa_select_paged_prefill
sm90 = checks.sm90


@pytest.mark.parametrize("case", workloads.observed_cases(), ids=lambda c: c["name"])
def test_key_major_observed_against_native(sm90, monkeypatch, case):
    checks.check_observed_envelope(monkeypatch, case, trial.candidate_module)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("heads,dim", [(4, 128), (3, 24), (5, 96), (1, 8)])
def test_key_major_every_config_and_chunk_boundary(sm90, dtype, heads, dim):
    checks.check_score_kernel(
        dtype, heads, dim, trial._qsa_prefill_key_major_score_kernel, False
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("budget", ["512", "1", "0"])
def test_key_major_full_entry(sm90, monkeypatch, dtype, budget):
    # This ragged envelope fails the compact gate, so full-entry correctness
    # covers the new rectangular kernel even at the largest workspace budget.
    checks.check_full_entry(
        monkeypatch,
        dtype,
        budget,
        trial.candidate_module,
        queries=(33, 0, 1),
        contexts=(4096, 16, 8192),
    )
