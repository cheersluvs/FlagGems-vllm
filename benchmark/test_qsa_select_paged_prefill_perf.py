# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""H800 QSA prefill: full operator, score stage, and Top-K stage.

Only the twelve published upstream query/prefix/page layouts are benchmarked.
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
from tools import qsa_prefill_upstream_workloads as upstream_workloads

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
UPSTREAM_CASES = {
    (case["requests"], case["query_length"], case["prefix_tokens"]): case
    for case in upstream_workloads.cases()
}


class QSAPrefillBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Both core and comprehensive use only the published twelve layouts.
        self.shapes = list(UPSTREAM_CASES)
        self.shape_desc = (
            "requests, query_length, prefix_tokens "
            "(BF16, H=4, D=128, page=196, block_k=512)"
        )

    def set_more_shapes(self):
        return []


def _inputs(shape, dtype, device):
    assert dtype == torch.bfloat16
    yield upstream_workloads.make_inputs(
        UPSTREAM_CASES[shape], input_helpers, torch, device=device
    )


def _score_inputs(shape, dtype, device):
    args = next(_inputs(shape, dtype, device))
    width = upstream_workloads.geometry(UPSTREAM_CASES[shape])["logits_width"]
    workspace = torch.empty((args[0].shape[0] * width,), device=device)
    yield (*args[:5], workspace, args[7], width, 0, args[0].shape[0])


def _topk_inputs(shape, dtype, device):
    args = next(_inputs(shape, dtype, device))
    width = upstream_workloads.geometry(UPSTREAM_CASES[shape])["logits_width"]
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
    dtypes=(torch.bfloat16,),
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
    )


def test_topk_vs_vllm():
    reference = vllm_reference()

    def baseline(logits, visible, token_topk, ratio, output, workspace, starts):
        del starts
        reference._topk(logits, visible, token_topk, ratio, output, workspace)

    _run("qsa_prefill_topk_vs_vllm", baseline, _gems_topk, _topk_inputs)


def test_topk_persistent_vs_prefill():
    # Direct stage comparison in one runtime; baseline here is the old
    # FlagGems prefill implementation, explicitly separate from vLLM results.
    _run(
        "qsa_prefill_topk_persistent_vs_old_prefill",
        _old_gems_topk,
        _gems_topk,
        _topk_inputs,
    )
