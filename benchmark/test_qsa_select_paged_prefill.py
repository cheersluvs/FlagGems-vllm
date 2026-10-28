# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA prefill against native vLLM: full operator, score and TopK.

Ten FP8 cases cover eight recorded inference envelopes and two dtype variants.
Fixtures and the pinned native reference are shared with the correctness tests.
NV acceptance uses score-only --mode kernel; full and TopK results are also kept.
The isolated score comparison uses a single full-row launch.
"""

from functools import partial
from importlib import import_module

import pytest
import torch

import flaggems_vllm
from tests.test_qsa_select_paged_prefill import (
    BENCHMARK_CASES,
    _benchmark_geometry,
    make_benchmark_inputs,
    vllm_reference,
)

from . import base

op_module = import_module("flaggems_vllm.ops.qsa_select_paged_prefill")


class QSAPrefillBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(BENCHMARK_CASES)
        self.shape_desc = "FP8 batch case (H=4, D=128, page=196, block_k=512)"

    def set_more_shapes(self):
        return []


def _inputs(shape, dtype, device, *, stage):
    assert dtype == getattr(torch, BENCHMARK_CASES[shape]["dtype"])
    args = make_benchmark_inputs(BENCHMARK_CASES[shape], device=device)
    if stage == "full":
        yield args
        return
    width = _benchmark_geometry(BENCHMARK_CASES[shape])["logits_width"]
    workspace = torch.empty((args[0].shape[0] * width,), device=device)
    score_args = (*args[:5], workspace, args[7], width, 0, args[0].shape[0])
    if stage == "score":
        yield score_args
        return
    logits = op_module._prefill_logits(*score_args)
    topk_workspace = torch.empty((1024 * 1024,), dtype=torch.uint8, device=device)
    yield logits, args[4], args[5], args[6], args[8], topk_workspace


def _gems_topk(logits, visible, token_topk, ratio, output, workspace):
    op_module.persistent_topk(
        logits,
        visible,
        output,
        workspace,
        k=token_topk // ratio,
        max_seq_len=logits.shape[1],
    )


@pytest.mark.qsa_select_paged_prefill
@pytest.mark.parametrize("stage", ["full", "score", "topk"])
def test_qsa_select_paged_prefill(stage):
    reference = vllm_reference()
    baseline, production = {
        "full": (
            reference.qsa_select_paged_prefill,
            flaggems_vllm.qsa_select_paged_prefill,
        ),
        "score": (reference._prefill_logits, op_module._prefill_logits),
        "topk": (reference._topk, _gems_topk),
    }[stage]
    QSAPrefillBenchmark(
        input_fn=partial(_inputs, stage=stage),
        op_name=f"qsa_select_paged_prefill_{stage}",
        torch_op=baseline,
        gems_op=production,
        dtypes=[torch.float8_e4m3fn],
    ).run()


pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="NVIDIA SM90 required",
)
