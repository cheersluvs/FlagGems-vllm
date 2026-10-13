# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""H800 QSA prefill: full operator, score stage, and Top-K stage.

The official baseline is SHA256-pinned and required by the card runner.
The independent Torch baseline includes host metadata reads and is reported
separately; it must not be used as a substitute for the vLLM comparison.
"""

from importlib import import_module

import pytest
import torch

import flaggems_vllm
from tests.qsa_select_prefill_utils import (
    make_inputs,
    make_upstream_no_complete_blocks_inputs,
    make_upstream_prefill_inputs,
    torch_select,
    vllm_reference,
)

from . import base, conftest, consts

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

# (request count, queries per request, compressed context length, page size, K)
CORE_SHAPES = [
    (1, 1, 2048, 16, 512),
    (1, 64, 4096, 16, 512),
    (4, 128, 16384, 64, 512),
    (1, 2048, 65536, 64, 512),
    (1, 64, 4096, 16, 1024),
    (1, 64, 8192, 16, 2048),
]
EXTRA_SHAPES = [
    (1, 7, 1537, 16, 512),
    (4, 16, 8193, 64, 1024),
    (1, 512, 65536, 16, 1024),
    (4, 512, 65536, 64, 2048),
]


class QSAPrefillBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CORE_SHAPES
        if conftest.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes = CORE_SHAPES + EXTRA_SHAPES
        self.shape_desc = (
            "requests, query_length, compressed_context, page_size, block_k"
        )

    def set_more_shapes(self):
        return []


class QSAUpstreamPrefillBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Exact BF16/FP8 upstream prefill fixture, with both max_seq_len bounds.
        self.shapes = [(0,), (1792,)]
        self.shape_desc = "seq_len_slack (Q=36x4x128, cache=128x24x1x128, table=2x64)"

    def set_more_shapes(self):
        return []


def _upstream_inputs(shape, dtype, device):
    yield make_upstream_prefill_inputs(
        seq_len_slack=shape[0], device=device, dtype=dtype
    )


class QSANoCompleteBlocksBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [()]
        self.shape_desc = "upstream zero-visible (Q=2x4x8, cache=1x16x1x8)"

    def set_more_shapes(self):
        return []


def _no_complete_blocks_inputs(shape, dtype, device):
    assert dtype == torch.bfloat16
    yield make_upstream_no_complete_blocks_inputs(device=device)


def _inputs(shape, dtype, device):
    requests, length, context, page, k = shape
    assert dtype == torch.bfloat16
    yield make_inputs(
        (length,) * requests,
        (context,) * requests,
        page_size=page,
        k=k,
        query_base=31,
        device=device,
    )


def _score_inputs(shape, dtype, device):
    args = next(_inputs(shape, dtype, device))
    width = args[2].shape[1] * args[1].shape[1]
    workspace = torch.empty((args[0].shape[0] * width,), device=device)
    yield (*args[:5], workspace, args[7], width, 0, args[0].shape[0])


def _topk_inputs(shape, dtype, device):
    args = next(_inputs(shape, dtype, device))
    width = args[2].shape[1] * args[1].shape[1]
    scores_workspace = torch.empty((args[0].shape[0] * width,), device=device)
    logits = op_module._prefill_logits(
        *args[:5], scores_workspace, args[7], width, 0, args[0].shape[0]
    )
    row_starts = torch.zeros((args[0].shape[0],), device=device, dtype=torch.int32)
    topk_workspace = torch.empty((1024 * 1024,), dtype=torch.uint8, device=device)
    yield logits, args[4], args[5], args[6], args[8], topk_workspace, row_starts


def _gems_topk(logits, visible, token_topk, ratio, output, workspace, starts):
    del workspace
    op_module.top_k_per_row_prefill(
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


@pytest.mark.parametrize("force_chunk", [False, True], ids=["default", "one_row"])
def test_upstream_full_vs_torch(monkeypatch, force_chunk):
    monkeypatch.setenv(
        "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "0" if force_chunk else "512"
    )
    chunk_name = "one_row" if force_chunk else "default"
    _run(
        f"qsa_prefill_upstream_vs_torch_{chunk_name}",
        torch_select,
        flaggems_vllm.qsa_select_paged_prefill,
        _upstream_inputs,
        QSAUpstreamPrefillBenchmark,
        dtypes=(torch.bfloat16, torch.float8_e4m3fn),
    )


@pytest.mark.parametrize("force_chunk", [False, True], ids=["default", "one_row"])
def test_upstream_full_vs_vllm(monkeypatch, force_chunk):
    reference = vllm_reference()
    budget = 0 if force_chunk else 512
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", str(budget))
    monkeypatch.setattr(reference.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", budget)
    chunk_name = "one_row" if force_chunk else "default"
    _run(
        f"qsa_prefill_upstream_vs_vllm_{chunk_name}",
        reference.qsa_select_paged_prefill,
        flaggems_vllm.qsa_select_paged_prefill,
        _upstream_inputs,
        QSAUpstreamPrefillBenchmark,
        dtypes=(torch.bfloat16, torch.float8_e4m3fn),
    )


@pytest.mark.parametrize("baseline", ["torch", "vllm"])
def test_upstream_no_complete_blocks(baseline):
    reference = (
        torch_select
        if baseline == "torch"
        else vllm_reference().qsa_select_paged_prefill
    )
    _run(
        f"qsa_prefill_upstream_no_complete_blocks_vs_{baseline}",
        reference,
        flaggems_vllm.qsa_select_paged_prefill,
        _no_complete_blocks_inputs,
        QSANoCompleteBlocksBenchmark,
    )
