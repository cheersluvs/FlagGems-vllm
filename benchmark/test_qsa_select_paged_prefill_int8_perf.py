# Copyright 2026 FlagOS Contributors
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

"""Ascend 910B QSA INT8 selection against the fused CANN V1 indexer."""

import hashlib
import inspect
import json
import math
import os
from functools import partial
from importlib import import_module
from pathlib import Path

import pytest
import torch
import triton

import flaggems_vllm
from tools.qsa_int8_ten_workloads_910b import cases, make_inputs, sample_rows

from . import base, consts
from .conftest import Config

OP_NAME = "qsa_select_paged_prefill_int8_910b"
CASES = {(case["name"],): case for case in cases()}


class QSAPrefillInt8Benchmark(base.GenericBenchmark):
    case_name = None

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(self.case_name,)]
        self.shape_desc = "INT8 selection with H=4, D=128, page=192, K=512"

    def record_shapes(self, *, case_name, **kwargs):
        return (case_name,)


def _call_reference(*, reference, **kwargs):
    return reference()


def _call_candidate(*, candidate, **kwargs):
    return candidate()


@pytest.mark.qsa_select_paged_prefill_int8_910b
@pytest.mark.parametrize("case_name", [shape[0] for shape in CASES])
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend", reason="Ascend 910B required"
)
def test_qsa_select_paged_prefill_int8_kernel(monkeypatch, tmp_path, case_name):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    device_index = int(os.environ.get("QSA_DEVICE", torch.npu.current_device()))
    torch.npu.set_device(device_index)
    if "910B" not in torch.npu.get_device_name(device_index):
        pytest.skip("Ascend 910B required")
    if Config.mode != consts.BenchMode.KERNEL:
        pytest.skip("Run this comparison with --mode kernel")

    native = getattr(torch_npu, "npu_quant_lightning_indexer", None)
    if native is None:
        pytest.fail("CANN V1 torch_npu.npu_quant_lightning_indexer is required")

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))

    from probe_qsa_int8_page64_prefix_910b import load_generated_prefix
    from probe_qsa_int8_pipeline_910b import check_selection
    from probe_qsa_int8_score_910b import reference as cpu_score
    from probe_qsa_int8_score_sol_910b import build_direct_page_source
    from probe_qsa_int8_ten_shapes_910b import verify_structure
    from qsa_int8_shared_map_selection_910b_trial import PreparedSharedMapInt8Selection
    from triton.backends.ascend.testing import do_bench_npu

    topk = import_module("flaggems_vllm.runtime.backend._ascend.ops.persistent_topk")
    if not topk.HAS_TLE_PTK:
        pytest.fail("Ascend persistent TopK TLE is required")
    if (
        Path(topk.__file__).resolve()
        != (
            root / "src/flaggems_vllm/runtime/backend/_ascend/ops/persistent_topk.py"
        ).resolve()
    ):
        pytest.fail(f"Unexpected persistent TopK source: {topk.__file__}")
    compiler = Path(triton.__file__).resolve()
    env_root = os.environ.get("QSA_ENV_ROOT")
    if env_root and not compiler.is_relative_to(Path(env_root).resolve() / "flagtree"):
        pytest.fail(f"Unexpected FlagTree source: {compiler}")

    timer = Path(inspect.getfile(do_bench_npu)).resolve()
    print(
        "QSA_INT8_HARNESS_ENV",
        json.dumps(
            dict(
                device=torch.npu.get_device_name(device_index),
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                triton=triton.__version__,
                triton_source=str(compiler),
                ascend_opp=os.environ.get("ASCEND_OPP_PATH"),
                timer_source=str(timer),
                timer_sha256=hashlib.sha256(timer.read_bytes()).hexdigest(),
                timer_signature=str(inspect.signature(do_bench_npu)),
                topk_source=topk.__file__,
                topk_route=os.environ.get("FLAGGEMS_ASCEND_PTK_PATH", "auto"),
            )
        ),
        flush=True,
    )
    kernels = {}
    for bounded in (True, False):
        arm = "direct_page_bounded" if bounded else "direct_page_full"
        kernels[bounded] = load_generated_prefix(
            tmp_path,
            "harness",
            arm,
            build_direct_page_source(bounded, arm, precomputed_visibility=False),
        )

    def prepare_case(shape, dtype, device):
        assert dtype == torch.int8
        plan = CASES[shape]
        name = plan["name"]
        print(f"QSA_INT8_HARNESS_CASE {name}", flush=True)
        inputs = make_inputs(plan)
        values = {key: value.to(device) for key, value in inputs.items()}
        rows, width, chunk = (
            plan["q_shape"][0],
            plan["logits_width"],
            plan["chunk_rows"],
        )
        output = torch.empty((rows, 512), dtype=torch.int32, device=device)
        scratch = torch.empty((chunk, width), dtype=torch.float32, device=device)
        workspace = torch.empty(1024**2, dtype=torch.uint8, device=device)
        adapter_args = (values, output, scratch, workspace, topk, plan["starts"])
        adapter_kwargs = dict(key_group4=True, single_page64=True)
        control = PreparedSharedMapInt8Selection(*adapter_args, **adapter_kwargs)
        candidate = PreparedSharedMapInt8Selection(
            *adapter_args,
            **adapter_kwargs,
            direct_page_kernels=kernels,
            shared_direct_page_proof=True,
        )
        assert control.route_checks == candidate.route_checks

        native_call = partial(
            native,
            values["q"],
            values["cache"],
            values["weights"],
            values["qs"],
            values["ks"],
            0,
            0,
            actual_seq_lengths_query=torch.arange(
                1, rows + 1, dtype=torch.int32, device=device
            ),
            actual_seq_lengths_key=values["visible"],
            block_table=values["row_table"],
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=512,
            sparse_mode=0,
            pre_tokens=2**63 - 1,
            next_tokens=2**63 - 1,
        )
        torch.npu.synchronize()
        print(f"QSA_INT8_HARNESS_PREFLIGHT_START {name} control", flush=True)
        original = control()
        torch.npu.synchronize()
        original = original.cpu()
        print(f"QSA_INT8_HARNESS_PREFLIGHT_START {name} candidate", flush=True)
        selected = candidate()
        torch.npu.synchronize()
        selected = selected.cpu()
        assert torch.equal(original, selected), f"{name}: candidate indices differ"
        print(f"QSA_INT8_HARNESS_PREFLIGHT_START {name} cann_v1", flush=True)
        native_output = native_call()
        torch.npu.synchronize()
        assert native_output.shape == (rows, 1, 512)
        expected = native_output.cpu().squeeze(1)
        for indices in (expected, selected):
            verify_structure(indices, inputs["visible"])
        samples = sample_rows(plan)
        refs = [cpu_score(inputs, row)[1] for row in samples]
        for label, indices in (("CANN V1", expected), ("candidate", selected)):
            checks, _ = check_selection(indices[samples], refs)
            assert checks["checks_passed"], (name, label, checks)
        print(
            f"QSA_INT8_HARNESS_PREFLIGHT {name} "
            f"direct_chunks={sum(r in ('bounded', 'full') for r in candidate.direct_page_routes.values())} "
            f"total_chunks={len(candidate.chunks)}",
            flush=True,
        )
        del original, selected, native_output, expected, control, refs, inputs

        if name == "int8_budget_exact":
            first, count = candidate.chunks[0]

            def one_score():
                candidate.score_into(first, count, scratch[:count])

            def three_scores():
                for _ in range(3):
                    one_score()

            single = benchmark.get_latency(one_score)
            triple = benchmark.get_latency(three_scores)
            assert math.isfinite(single) and single > 0 and math.isfinite(triple)
            ratio = triple / single
            assert (
                2.25 <= ratio <= 3.75
            ), f"Unexpected harness invocation scaling: {ratio}"
            print(
                f"QSA_INT8_HARNESS_TIMER one_ms={single} three_ms={triple} ratio={ratio}"
            )
        # The harness unpacks dictionaries nested in the input tuple as kwargs.
        yield (dict(case_name=name, reference=native_call, candidate=candidate),)

    benchmark = QSAPrefillInt8Benchmark(
        op_name=OP_NAME,
        case_name=case_name,
        torch_op=_call_reference,
        gems_op=_call_candidate,
        input_fn=prepare_case,
        dtypes=[torch.int8],
        device=f"npu:{device_index}",
    )
    benchmark.run()
