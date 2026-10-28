# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QSA prefill through the public entry point, with a Torch reference.

The ten inference layouts are shared with the correctness tests. NVIDIA uses
FP8; other backends start with BF16 versions of the same layouts. Native vLLM
full/score/TopK comparisons are separate NVIDIA SM90 tests. Torch-reference
ratios include host metadata reads and are not native-vLLM speedup results.
The isolated native score comparison uses a single full-row launch.
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
    torch_select,
    vllm_reference,
)

from . import base


class QSAPrefillBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(BENCHMARK_CASES)
        self.shape_desc = "batch case (H=4, D=128, page=196, block_k=512)"

    def set_more_shapes(self):
        return []


def _inputs(shape, dtype, device, *, stage):
    case = dict(BENCHMARK_CASES[shape], dtype=str(dtype).removeprefix("torch."))
    args = make_benchmark_inputs(case, device=device)
    if stage == "full":
        yield args
        return
    op_module = import_module("flaggems_vllm.ops.qsa_select_paged_prefill")
    width = _benchmark_geometry(case)["logits_width"]
    workspace = torch.empty((args[0].shape[0] * width,), device=device)
    score_args = (*args[:5], workspace, args[7], width, 0, args[0].shape[0])
    if stage == "score":
        yield score_args
        return
    logits = op_module._prefill_logits(*score_args)
    topk_workspace = torch.empty((1024 * 1024,), dtype=torch.uint8, device=device)
    yield logits, args[4], args[5], args[6], args[8], topk_workspace


def _gems_topk(logits, visible, token_topk, ratio, output, workspace, *, topk):
    topk(
        logits,
        visible,
        output,
        workspace,
        k=token_topk // ratio,
        max_seq_len=logits.shape[1],
    )


@pytest.mark.qsa_select_paged_prefill
def test_qsa_select_paged_prefill():
    QSAPrefillBenchmark(
        input_fn=partial(_inputs, stage="full"),
        op_name="qsa_select_paged_prefill_vs_torch",
        torch_op=torch_select,
        gems_op=flaggems_vllm.qsa_select_paged_prefill,
        dtypes=(
            [torch.float8_e4m3fn]
            if flaggems_vllm.vendor_name == "nvidia"
            else [torch.bfloat16]
        ),
    ).run()


@pytest.mark.qsa_select_paged_prefill
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="Native vLLM QSA comparison requires NVIDIA SM90",
)
@pytest.mark.parametrize("stage", ["full", "score", "topk"])
def test_qsa_select_paged_prefill_native_vllm(stage):
    reference = vllm_reference()
    op_module = import_module("flaggems_vllm.ops.qsa_select_paged_prefill")
    baseline, production = {
        "full": (
            reference.qsa_select_paged_prefill,
            flaggems_vllm.qsa_select_paged_prefill,
        ),
        "score": (reference._prefill_logits, op_module._prefill_logits),
        "topk": (
            reference._topk,
            partial(_gems_topk, topk=op_module.persistent_topk),
        ),
    }[stage]
    QSAPrefillBenchmark(
        input_fn=partial(_inputs, stage=stage),
        op_name=f"qsa_select_paged_prefill_{stage}_vs_native_vllm",
        torch_op=baseline,
        gems_op=production,
        dtypes=[torch.float8_e4m3fn],
    ).run()
