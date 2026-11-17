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

"""Stock whole-call harness for serial raw or mixed two-slot selection."""

import importlib.util
import json
import math
import os
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import triton

from tools.qsa_int8_ten_workloads_910b import cases

from . import consts
from .conftest import Config
from .test_qsa_select_paged_prefill_int8_perf import (
    QSAPrefillInt8Benchmark,
    _call_candidate,
    _call_reference,
)

OP_NAME = (
    "qsa_raw_history_prefix_int8_910b"
    if os.environ.get("QSA_RAW_HISTORY_HARNESS") == "1"
    else (
        "qsa_raw_tle_sort_int8_910b"
        if os.environ.get("QSA_RAW_TLE_SORT_HARNESS") == "1"
        else (
            "qsa_raw_mixed_int8_910b"
            if os.environ.get("QSA_RAW_MIXED_HARNESS") == "1"
            else "qsa_raw_serial_int8_910b"
        )
    )
)


@pytest.mark.qsa_select_paged_prefill_int8_910b
@pytest.mark.parametrize("case_name", [p["name"] for p in cases()])
def test_qsa_raw_serial_kernel(monkeypatch, case_name):
    torch_npu = pytest.importorskip("torch_npu")
    assert Config.mode == consts.BenchMode.KERNEL, "Requires --mode kernel"
    assert torch.npu.is_available()
    assert os.getenv("ASCEND_LAUNCH_BLOCKING") != "1"
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    from probe_qsa_raw_cube_chain_910b import (
        record_launch_mode,
        save_asm,
        verify_compiler_candidate,
    )
    from qsa_cube_query_pack_probe_910b import native_module
    from qsa_finish_routing_probe_910b import boundary
    from qsa_finish_routing_probe_910b import source as finish_source
    from qsa_prefill_pipeline_910b_trial import load_topk
    from qsa_raw_serial_selection_910b_trial import SerialRawSelection, verify_pipeline
    from qsa_ten_score_suite_910b import run_case, runtime_source
    from triton.backends.ascend import compiler

    report = Path(os.environ["QSA_RAW_SERIAL_REPORT"])
    device_index = int(os.environ.get("QSA_DEVICE", "0"))
    torch.npu.set_device(device_index)
    assert "910B" in torch.npu.get_device_name(device_index)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    build = dict(native_builds={})
    save = lambda: (report / (case_name + "-native-build.json")).write_text(
        json.dumps(build, indent=2)
    )
    assert verify_compiler_candidate(
        compiler, triton, report, build
    ), "Requires verified isolated compiler"
    monkeypatch.setenv("FLAGGEMS_ASCEND_PTK_PATH", "auto")
    topk = load_topk()
    assert topk.HAS_TLE_PTK
    native = native_module(report, 16, build, save)
    path = report / (case_name + "-serial-kernels.py")
    mixed_mode = os.environ.get("QSA_RAW_MIXED_HARNESS") == "1"
    fixed_mode = os.environ.get("QSA_RAW_TLE_SORT_HARNESS") == "1"
    history_mode = os.environ.get("QSA_RAW_HISTORY_HARNESS") == "1"
    assert not history_mode or fixed_mode
    assert not fixed_mode or mixed_mode
    source_text = runtime_source() + finish_source()
    if mixed_mode:
        if history_mode:
            from qsa_raw_history_merge_910b import harness_source
        elif fixed_mode:
            from qsa_raw_tle_sort512_910b import harness_source
        else:
            from qsa_raw_mixed_online_910b import harness_source
        from qsa_raw_mixed_online_910b import source as mixed_source

        source_text += mixed_source() + harness_source()
    path.write_text(source_text)
    path.with_suffix(".py.txt").write_text(path.read_text())
    spec = importlib.util.spec_from_file_location("_raw_serial_kernels", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.chain_op = native.chain_op
    if mixed_mode:
        from qsa_raw_mixed_abi_910b import bind_mixed_chain

        module.mixed_chain_op = bind_mixed_chain(report, build, save)
        if fixed_mode:
            from qsa_raw_cube_block512_910b import bind_block512

            module.block512_chain_op = bind_block512(report, build, save)
    build["boundary"] = boundary(
        module, torch, f"npu:{device_index}", variants=("empty1024",)
    )
    build["topk_identity"] = topk.probe_identity
    save()
    plan = next(p for p in cases() if p["name"] == case_name)
    detail = {}

    def integration(context):
        adapter = context["adapter"]
        adapter.topk = topk
        raw = SerialRawSelection(context)
        mixed = None
        generic = None
        extra = {}
        if mixed_mode:
            from qsa_raw_mixed_selection_910b_trial import PreparedRawMixedSelection

            options = {}
            if fixed_mode:
                from qsa_raw_tle_sort512_910b import ir_evidence

                options = dict(
                    kernel_name="raw_tle_fixed_prepared",
                    online_name="raw_tle_fixed_topk",
                    evidence=partial(ir_evidence, fixed=True),
                )
                generic = (
                    None
                    if history_mode
                    else PreparedRawMixedSelection(
                        context,
                        module,
                        report,
                        case_name + "-generic",
                        save_asm,
                        record_launch_mode,
                        kernel_name="raw_tle_generic_prepared",
                        online_name="raw_tle_generic_topk",
                        evidence=partial(ir_evidence, fixed=False),
                    )
                )
                if generic is not None:
                    extra["generic"] = generic
            if history_mode:
                from qsa_history_policy_910b import dispatcher
                from qsa_raw_history_merge_910b import ir_evidence as history_ir

                policy = json.loads(Path(os.environ["QSA_HISTORY_POLICY"]).read_text())
                options = dict(
                    kernel_name="raw_history_prepared_snapshot",
                    online_name="raw_history_topk_snapshot",
                    dispatch=dispatcher(module, policy),
                )
                generic = PreparedRawMixedSelection(
                    context,
                    module,
                    report,
                    case_name + "-snapshot",
                    save_asm,
                    record_launch_mode,
                    kernel_name="raw_history_prepared_snapshot",
                    online_name="raw_history_topk_snapshot",
                    evidence=partial(history_ir, complete=False),
                )
                extra["generic"] = generic
            mixed = PreparedRawMixedSelection(
                context,
                module,
                report,
                case_name,
                save_asm,
                record_launch_mode,
                **options,
            )
            extra["mixed"] = mixed
            detail["mixed_identity"] = dict(
                slots=2,
                kernel="raw_tle_fixed_prepared" if fixed_mode else "raw_mixed_prepared",
                topk="online512_UB_snapshot",
                native=(
                    "block512_query16_cube" if fixed_mode else "unchanged_query16_cube"
                ),
                local_sort="sort32_mrgsort_fixed512" if fixed_mode else "sort_1d_pack",
                source_import=str(Path(module.__file__)),
            )
            detail["mixed_routes"] = mixed.routes
            detail["mixed_compiled_artifacts"] = mixed.artifacts
            if fixed_mode:
                detail["generic_identity"] = dict(
                    kernel="raw_tle_generic_prepared",
                    local_sort="sort_1d_pack",
                    native="block512_query16_cube",
                    same_packing_and_schedule=True,
                )
                detail["generic_compiled_artifacts"] = generic.artifacts
                detail["generic_routes"] = generic.routes
                assert generic.routes == mixed.routes
            if history_mode:
                detail["mixed_identity"]["history"] = "prefix_by_threshold"
                detail["mixed_identity"]["kernel"] = "raw_history_dispatch"
                detail["mixed_identity"]["topk"] = "online512_prefix_dispatch"
                detail["generic_identity"]["kernel"] = "raw_history_prepared_snapshot"
                detail["generic_identity"]["local_sort"] = "sort32_mrgsort_fixed512"
                detail["generic_identity"]["history"] = "snapshot"
                detail["history_policy"] = policy
                detail["history_dispatch"] = mixed.dispatch_routes
        v = context["values"]
        native_call = partial(
            torch_npu.npu_quant_lightning_indexer,
            v["q"],
            v["cache"],
            v["weights"],
            v["qs"],
            v["ks"],
            0,
            0,
            actual_seq_lengths_query=torch.arange(
                1, adapter.rows + 1, dtype=torch.int32, device=adapter.output.device
            ),
            actual_seq_lengths_key=v["visible"],
            block_table=v["row_table"],
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=512,
            sparse_mode=0,
            pre_tokens=2**63 - 1,
            next_tokens=2**63 - 1,
        )
        detail["precheck"] = verify_pipeline(
            torch, context, raw, native_call, "before", extra=extra
        )
        detail["topk_identity"] = topk.probe_identity
        detail["scope"] = (
            "Prepared two-slot mixed candidate; QW/K/KS packing and request splits charged"
            if mixed_mode
            else "Prepared unregistered serial raw candidate; full native CANN V1 baseline; no overlap"
        )

        def prepare(shape, dtype, device):
            assert shape == (case_name,) and dtype == torch.int8
            yield (
                dict(
                    case_name=case_name,
                    route=(
                        "history_prefix_slots2"
                        if history_mode
                        else (
                            "tle_sort512_slots2"
                            if fixed_mode
                            else "online512_slots2" if mixed_mode else "auto"
                        )
                    ),
                    reference=native_call,
                    candidate=mixed if mixed_mode else raw,
                ),
            )

        benchmark = QSAPrefillInt8Benchmark(
            op_name=OP_NAME,
            case_name=case_name,
            torch_op=_call_reference,
            gems_op=_call_candidate,
            input_fn=prepare,
            dtypes=[torch.int8],
            device=f"npu:{device_index}",
        )
        if os.environ.get("QSA_PACKING_PROBE") == "1":
            from qsa_packing_cost_probe_910b import probe

            assert history_mode
            detail["packing_probe"] = probe(context, mixed, raw, benchmark)
            detail["postcheck"] = verify_pipeline(
                torch, context, raw, native_call, "after", extra=extra
            )
            return detail
        benchmark.run()
        # Auxiliary stages use the same stock get_latency. Native has no stage split.
        timings = {}
        stage_calls = [
            ("current_combined", adapter),
            ("current_score", adapter.score_all),
            ("raw_score", raw.score_all),
        ]
        if mixed_mode:
            stage_calls.insert(1, ("serial_combined", raw))
        if fixed_mode:
            stage_calls.insert(1, ("generic_combined", generic))
        for name, call in stage_calls:
            timings[name] = float(benchmark.get_latency(call))
            assert math.isfinite(timings[name]) and timings[name] > 0
        staged = torch.empty(
            (adapter.rows, adapter.width),
            dtype=torch.float32,
            device=adapter.output.device,
        )
        for entry, pieces in context["jobs"]:
            raw.score_chunk(pieces)
            staged[entry["first"] : entry["first"] + entry["count"]].copy_(
                context["scores"][: entry["count"]]
            )
        torch.npu.synchronize()
        timings["raw_topk"] = float(
            benchmark.get_latency(lambda staged=staged: adapter.topk_all(staged))
        )
        assert math.isfinite(timings["raw_topk"]) and timings["raw_topk"] > 0
        detail["isolated_topk_staging_bytes"] = staged.numel() * staged.element_size()
        from qsa_raw_serial_selection_910b_trial import check_rank

        selected = adapter.output.cpu()
        for first, count in adapter.chunks:
            check_rank(
                torch,
                selected[first : first + count],
                staged[first : first + count].cpu(),
                context["cpu"]["visible"][first : first + count],
                rtol=0,
                atol=0,
            )
        if mixed_mode:
            timings["online_topk"] = float(
                benchmark.get_latency(
                    lambda staged=staged: mixed.topk_all_from_staged(staged)
                )
            )
            assert math.isfinite(timings["online_topk"]) and timings["online_topk"] > 0
            torch.npu.synchronize()
            selected = mixed.output.cpu()
            for first, count in adapter.chunks:
                check_rank(
                    torch,
                    selected[first : first + count],
                    staged[first : first + count].cpu(),
                    context["cpu"]["visible"][first : first + count],
                    rtol=0,
                    atol=0,
                )
            mixed.check_guards()
        if fixed_mode:
            timings["generic_online_topk"] = float(
                benchmark.get_latency(
                    lambda staged=staged: generic.topk_all_from_staged(staged)
                )
            )
            assert (
                math.isfinite(timings["generic_online_topk"])
                and timings["generic_online_topk"] > 0
            )
            torch.npu.synchronize()
            selected = generic.output.cpu()
            for first, count in adapter.chunks:
                check_rank(
                    torch,
                    selected[first : first + count],
                    staged[first : first + count].cpu(),
                    context["cpu"]["visible"][first : first + count],
                    rtol=0,
                    atol=0,
                )
            generic.check_guards()
        del staged
        detail["stage_ms"] = timings
        detail["postcheck"] = verify_pipeline(
            torch, context, raw, native_call, "after", extra=extra
        )
        print("RAW_SERIAL_STAGE_MS", case_name, json.dumps(timings), flush=True)
        return detail

    args = SimpleNamespace(
        device=device_index,
        finish_routing_probe=True,
        finish_empty_ten=True,
        finish_empty_pair=False,
        rounds=2,
        warmup=1,
        active=3,
    )
    record = run_case(
        args,
        plan,
        module,
        report,
        save_asm,
        record_launch_mode,
        selection_benchmark=integration,
    )
    assert record["status"] == "passed", record.get("traceback", record)
