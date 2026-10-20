# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""H800 QSA prefill: full operator, score stage, and Top-K stage.

The matrix contains 22 BF16 cases plus eight FP8 envelopes and two FP8 variants.
The H800 --mode kernel runner measures the real score chunk sequence without
Top-K; the isolated stage tests below retain their single full-row launch.

The official baseline is SHA256-pinned and required by the card runner.
The independent Torch baseline includes host metadata reads and is reported
separately; it must not be used as a substitute for the vLLM comparison.
"""

from importlib import import_module

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.top_k_per_row_prefill import top_k_per_row_prefill
from tests import qsa_select_prefill_utils as input_helpers
from tests.qsa_select_prefill_utils import torch_select, vllm_reference
from tools import qsa_prefill_observed_workloads as workloads

from . import base

op_module = import_module("flaggems_vllm.ops.qsa_select_paged_prefill")
pytestmark = [
    pytest.mark.qsa_select_paged_prefill,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "nvidia"
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (9, 0),
        reason="NVIDIA SM90 required",
    ),
]

# One shared definition with the H800 --mode kernel runner. Prefix is in
# original tokens before the query, not compressed cache rows.
BENCHMARK_CASES = {
    (case["name"],): case for case in workloads.cases("combined-prefill")
}


class QSAPrefillBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Full-entry measurements include the two select-all fast paths.
        self.shapes = list(BENCHMARK_CASES)
        self.shape_desc = (
            "case (query lengths, prefixes and layout in workload manifest) "
            "(22 BF16 + 10 FP8, H=4, D=128, page=196, block_k=512)"
        )

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        # Each named case has one declared dtype, not a cross-product matrix.
        for shape in self.shapes:
            if getattr(torch, BENCHMARK_CASES[shape]["dtype"]) == dtype:
                yield from self.input_fn(shape, dtype, self.device)


class QSAScoredBenchmark(QSAPrefillBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            shape
            for shape in self.shapes
            if not workloads.geometry(BENCHMARK_CASES[shape]).get("select_all")
        ]


def _inputs(shape, dtype, device):
    assert dtype == getattr(torch, BENCHMARK_CASES[shape]["dtype"])
    yield workloads.make_inputs(
        BENCHMARK_CASES[shape], input_helpers, torch, device=device
    )


def _score_inputs(shape, dtype, device):
    args = next(_inputs(shape, dtype, device))
    width = workloads.geometry(BENCHMARK_CASES[shape])["logits_width"]
    workspace = torch.empty((args[0].shape[0] * width,), device=device)
    yield (*args[:5], workspace, args[7], width, 0, args[0].shape[0])


def _topk_inputs(shape, dtype, device):
    args = next(_inputs(shape, dtype, device))
    width = workloads.geometry(BENCHMARK_CASES[shape])["logits_width"]
    scores_workspace = torch.empty((args[0].shape[0] * width,), device=device)
    logits = op_module._prefill_logits(
        *args[:5], scores_workspace, args[7], width, 0, args[0].shape[0]
    )
    row_starts = torch.zeros((args[0].shape[0],), device=device, dtype=torch.int32)
    topk_workspace = torch.empty((1024 * 1024,), dtype=torch.uint8, device=device)
    yield logits, args[4], args[5], args[6], args[8], topk_workspace, row_starts


def _gems_topk(logits, visible, token_topk, ratio, output, workspace, starts):
    del starts
    op_module.persistent_topk(
        logits,
        visible,
        output,
        workspace,
        k=token_topk // ratio,
        max_seq_len=logits.shape[1],
    )


def _old_gems_topk(logits, visible, token_topk, ratio, output, workspace, starts):
    del workspace
    top_k_per_row_prefill(
        logits,
        starts,
        visible,
        output,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        token_topk // ratio,
    )


def _run(
    name,
    reference,
    gems,
    input_fn,
    benchmark_cls=QSAPrefillBenchmark,
    dtypes=(torch.bfloat16, torch.float8_e4m3fn),
):
    # Warm both functions on each input before timing to exclude compilation
    # and autotuning. The common runner then measures operator-level latency.
    def warmed_inputs(shape, dtype, device):
        for args in input_fn(shape, dtype, device):
            reference(*args)
            gems(*args)
            torch.cuda.synchronize()
            yield args

    benchmark_cls(
        op_name=name,
        torch_op=reference,
        gems_op=gems,
        input_fn=warmed_inputs,
        dtypes=list(dtypes),
    ).run()


def test_full_vs_torch():
    _run(
        "qsa_prefill_vs_torch",
        torch_select,
        flaggems_vllm.qsa_select_paged_prefill,
        _inputs,
    )


def test_full_vs_vllm():
    reference = vllm_reference()
    _run(
        "qsa_prefill_vs_vllm",
        reference.qsa_select_paged_prefill,
        flaggems_vllm.qsa_select_paged_prefill,
        _inputs,
    )


def test_scores_vs_vllm():
    reference = vllm_reference()
    _run(
        "qsa_prefill_scores_vs_vllm",
        reference._prefill_logits,
        op_module._prefill_logits,
        _score_inputs,
        benchmark_cls=QSAScoredBenchmark,
    )


def test_topk_vs_vllm():
    reference = vllm_reference()

    def baseline(logits, visible, token_topk, ratio, output, workspace, starts):
        del starts
        reference._topk(logits, visible, token_topk, ratio, output, workspace)

    _run(
        "qsa_prefill_topk_vs_vllm",
        baseline,
        _gems_topk,
        _topk_inputs,
        benchmark_cls=QSAScoredBenchmark,
    )


def test_topk_persistent_vs_prefill():
    # Direct stage comparison in one runtime; baseline here is the old
    # FlagGems prefill implementation, explicitly separate from vLLM results.
    _run(
        "qsa_prefill_topk_persistent_vs_old_prefill",
        _old_gems_topk,
        _gems_topk,
        _topk_inputs,
        benchmark_cls=QSAScoredBenchmark,
    )
