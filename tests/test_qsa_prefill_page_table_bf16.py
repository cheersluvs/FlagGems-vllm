# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Same on-card boundary/full-entry gates for the BF16-only page-table trial."""

import pytest
import torch

from tools import qsa_prefill_page_table_trial as trial

from . import test_qsa_prefill_page_table as common

pytestmark = pytest.mark.qsa_select_paged_prefill
sm90 = common.sm90


@pytest.fixture(scope="module")
def candidate():
    return trial.candidate_module(common.production, bf16_only=True)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
@pytest.mark.parametrize(
    "page_size", [16, 64, 196], ids=["page16", "page64", "page196"]
)
def test_page_table_score(sm90, candidate, dtype, page_size):
    common.test_page_table_score(sm90, candidate, dtype, page_size)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_page_table_full_entry(sm90, candidate, dtype, monkeypatch):
    common.test_page_table_full_entry(sm90, candidate, dtype, monkeypatch)
