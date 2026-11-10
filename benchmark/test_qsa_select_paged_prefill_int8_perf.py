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

import copy
import hashlib
import inspect
import json
import math
import os
import statistics
from functools import partial
from importlib import import_module
from pathlib import Path

import pytest
import torch
import triton

import flaggems_vllm
from tools.qsa_int8_ten_workloads_910b import cases, make_inputs, sample_rows
from tools.summarize_qsa_int8_harness_910b import (
    OP_NAME,
    TOPK_SEGMENT,
    topk_route,
)

from . import base, consts
from .conftest import Config

CASES = {(case["name"],): case for case in cases()}


class QSAPrefillInt8Benchmark(base.GenericBenchmark):
    case_name = None

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(self.case_name,)]
        self.shape_desc = "INT8 selection with H=4, D=128, page=192, K=512"

    def record_shapes(self, *, case_name, route, **kwargs):
        return (case_name, route)


def _call_reference(*, reference, **kwargs):
    return reference()


def _call_candidate(*, candidate, **kwargs):
    return candidate()


class _SynchronizedTopKKernel:
    """Identify the failing launch without relying on asynchronous error labels."""

    def __init__(self, name, kernel):
        self.name = name
        self.kernel = kernel
        self.binaries = set()

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def checked(*args, **kwargs):
            print(
                f"QSA_INT8_HARNESS_DIAG_LAUNCH {self.name} grid={grid} meta={kwargs}",
                flush=True,
            )
            compiled = launch(*args, **kwargs)
            if compiled is not None and id(compiled) not in self.binaries:
                binary = compiled.asm.get("npubin")
                if binary is not None:
                    digest = hashlib.sha256(bytes(binary)).hexdigest()
                    print(
                        f"QSA_INT8_HARNESS_DIAG_BINARY {self.name} "
                        f"compiled={compiled.metadata.kernel_name} sha256={digest}",
                        flush=True,
                    )
                self.binaries.add(id(compiled))
            torch.npu.synchronize()
            return compiled

        return checked


def _preflight_select(name, arm, adapter, diagnose_only):
    print(f"QSA_INT8_HARNESS_PREFLIGHT_START {name} {arm}", flush=True)
    adapter.output.fill_(-2 if arm == "control" else -3)
    torch.npu.synchronize()
    if not diagnose_only:
        output = adapter()
        torch.npu.synchronize()
        return output.cpu()

    for first, count in adapter.chunks:
        target = adapter.scratch[:count]
        print(
            f"QSA_INT8_HARNESS_DIAG {name} {arm} score first={first} rows={count}",
            flush=True,
        )
        adapter.score_into(first, count, target)
        torch.npu.synchronize()
        for local in range(0, count, 32):
            end = min(local + 32, count)
            print(
                f"QSA_INT8_HARNESS_DIAG {name} {arm} topk "
                f"first={first + local} rows={end - local}",
                flush=True,
            )
            adapter.select_chunk(first + local, target[local:end])
            torch.npu.synchronize()
    return adapter.output.cpu()


def _check_preflight_structure(name, arm, indices, lengths, verify_structure):
    try:
        verify_structure(indices, lengths)
    except AssertionError as exc:
        failure = dict(case=name, arm=arm, error=str(exc))
        if indices.shape == (lengths.numel(), 512) and indices.dtype == torch.int32:
            ordered = indices.sort(dim=1).values
            duplicate = (
                (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)
            ).any(1)
            invalid = ((indices < -1) | (indices >= lengths[:, None])).any(1)
            count = (indices >= 0).sum(1) != lengths.clamp(max=512)
            bad = (duplicate | invalid | count).nonzero().flatten()[:8].tolist()
            failure.update(
                duplicate_rows=int(duplicate.sum()),
                invalid_rows=int(invalid.sum()),
                wrong_count_rows=int(count.sum()),
                sentinel_entries=(
                    int((indices == (-2 if arm == "control" else -3)).sum())
                    if arm in ("control", "candidate")
                    else None
                ),
                samples=[
                    dict(
                        row=row,
                        visible=int(lengths[row]),
                        first_indices=indices[row, :12].tolist(),
                    )
                    for row in bad
                ],
            )
        print("QSA_INT8_HARNESS_INVALID_OUTPUT", json.dumps(failure), flush=True)
        return f"{arm}: {exc}"
    return None


def _prefill_topk_adapter(prefill, row_starts):
    def select(logits, lengths, output, workspace, *, k, max_seq_len, block):
        rows = logits.shape[0]
        assert max_seq_len == logits.shape[1] and rows <= row_starts.numel()
        prefill.top_k_per_row_prefill(
            logits, row_starts[:rows], lengths, output, rows, *logits.stride(), k
        )

    return select


def _measure_stages(
    name, route, adapter, expected_indices, device, *, score_control=None
):
    rows, width = adapter.rows, adapter.width
    print(
        f"QSA_INT8_HARNESS_STAGE_INPUT {name} score_bytes={rows * width * 4}",
        flush=True,
    )
    scores = torch.empty((rows, width), dtype=torch.float32, device=device)
    for first, count in adapter.chunks:
        adapter.score_into(first, count, scores[first : first + count])
    torch.npu.synchronize()

    def topk_only():
        adapter.topk_all(scores)

    def check_topk():
        assert torch.equal(
            adapter.output.cpu(), expected_indices
        ), f"{name}: isolated TopK indices differ from the full pipeline"

    adapter.output.fill_(-4)
    topk_only()
    torch.npu.synchronize()
    check_topk()
    for stage, execute in (("score", adapter.score_all), ("topk", topk_only)):
        reference = score_control if stage == "score" else None

        def stage_inputs(shape, dtype, target_device):
            yield (
                dict(
                    case_name=name, route=route, reference=reference, candidate=execute
                ),
            )

        QSAPrefillInt8Benchmark(
            op_name=f"{OP_NAME}_{stage}",
            case_name=name,
            torch_op=_call_reference if reference is not None else None,
            gems_op=_call_candidate,
            input_fn=stage_inputs,
            dtypes=[torch.int8],
            device=device,
        ).run()
        if stage == "topk":
            torch.npu.synchronize()
            check_topk()


def _measure_sorted_topk_ab(name, benchmark, control, lengths, samples, refs,
                            verify_structure, check_selection):
    from flaggems_vllm.runtime.backend._ascend.ops.qsa_select_paged_prefill import (
        _store_topk_indices,
    )
    from flaggems_vllm.runtime.backend._ascend.ops.topk import HAS_TLE, topk

    assert HAS_TLE, "Ascend sorted TLE TopK is required"
    candidate = copy.copy(control)
    candidate.reuse_topk_scratch = False
    k, width = 512, control.width

    def select(first, scores, persistent_scratch=None):
        for local in range(0, scores.shape[0], 32):
            end = min(local + 32, scores.shape[0])
            count = end - local
            output = candidate.output[first + local:first + end]
            visible = candidate.values["visible"][first + local:first + end]
            selected = topk(scores[local:end], k)[1] if width > k else output
            _store_topk_indices[(count,)](selected, visible, output, k, width)

    candidate.select_chunk = select
    scores = torch.empty((control.rows, width), device=control.scratch.device,
                         dtype=torch.float32)
    for first, count in control.chunks:
        control.score_into(first, count, scores[first:first + count])

    results = {}
    for stage, jobs in (
        ("topk", {"previous": lambda: control.topk_all(scores),
                  "sorted": lambda: candidate.topk_all(scores)}),
        ("total", {"previous": control, "sorted": candidate}),
    ):
        for arm, job in jobs.items():
            control.output.fill_(-4)
            job()
            torch.npu.synchronize()
            actual = control.output.cpu()
            verify_structure(actual, lengths)
            checks, _ = check_selection(actual[samples], refs)
            assert checks["checks_passed"], (name, stage, arm, checks)
        timings = {arm: [] for arm in jobs}
        rounds = 4 if os.environ.get("QSA_HARNESS_SORTED_TOPK_SHORT") == "1" else 2
        for round_index in range(rounds):
            order = ("previous", "sorted") if round_index % 2 == 0 else ("sorted", "previous")
            for arm in order:
                elapsed = float(benchmark.get_latency(jobs[arm]))
                assert math.isfinite(elapsed) and elapsed > 0, (name, stage, arm, elapsed)
                timings[arm].append(elapsed)
        medians = {arm: statistics.median(ms) for arm, ms in timings.items()}
        paired = [previous / sorted_ms for previous, sorted_ms in
                  zip(timings["previous"], timings["sorted"], strict=True)]
        results[stage] = dict(samples_ms=timings, medians_ms=medians,
                              speedup=medians["previous"] / medians["sorted"],
                              paired_speedups=paired, sorted_wins=sum(ratio > 1 for ratio in paired))
    print("QSA_INT8_HARNESS_SORTED_TOPK " + json.dumps(dict(case=name, **results)), flush=True)
    return candidate


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
    diagnose_only = os.environ.get("QSA_INT8_HARNESS_PREFLIGHT_ONLY") == "1"
    if not diagnose_only and os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
        pytest.fail("Unset ASCEND_LAUNCH_BLOCKING for kernel timing")

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
    from qsa_prefill_pipeline_910b_trial import load_topk
    from triton.backends.ascend.testing import do_bench_npu

    topk = load_topk()
    policy = os.environ.get("QSA_HARNESS_TOPK_POLICY", "auto")
    assert topk._SEGMERGE_SEG == TOPK_SEGMENT
    route = topk_route(CASES[(case_name,)], policy)
    monkeypatch.setenv(
        "FLAGGEMS_ASCEND_PTK_PATH", "auto" if route == "prefill" else route
    )
    if not topk.HAS_TLE_PTK:
        pytest.fail("Ascend persistent TopK TLE is required")
    if (
        Path(topk.__file__).resolve()
        != (
            root / "src/flaggems_vllm/runtime/backend/_ascend/ops/persistent_topk.py"
        ).resolve()
    ):
        pytest.fail(f"Unexpected persistent TopK source: {topk.__file__}")
    prefill_identity = None
    if route == "prefill":
        prefill = import_module(
            "flaggems_vllm.runtime.backend._ascend.ops.top_k_per_row_prefill"
        )
        if not prefill._TLE_IMPORT_OK:
            pytest.fail("Ascend prefill TopK TLE is required")
        prefill_path = Path(prefill.__file__).resolve()
        expected_path = (
            Path(topk.__file__).with_name("top_k_per_row_prefill.py").resolve()
        )
        if prefill_path != expected_path:
            pytest.fail(f"Unexpected Ascend prefill TopK source: {prefill_path}")
        prefill_identity = dict(
            source=str(prefill_path),
            sha256=hashlib.sha256(prefill_path.read_bytes()).hexdigest(),
        )
        starts = torch.zeros((32,), dtype=torch.int32, device=f"npu:{device_index}")
        monkeypatch.setattr(
            topk, "persistent_topk", _prefill_topk_adapter(prefill, starts)
        )
    if diagnose_only:
        for name in (
            "_radix_hist_trivial_all",
            "_segmerge_seg_sort",
            "_segmerge_merge_unpack",
        ):
            monkeypatch.setattr(
                topk, name, _SynchronizedTopKKernel(name, getattr(topk, name))
            )
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
                topk_identity=topk.probe_identity,
                prefill_topk_identity=prefill_identity,
                topk_policy=policy,
                topk_route=route,
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
        stages = os.environ.get("QSA_HARNESS_STAGES") == "1"
        candidate = PreparedSharedMapInt8Selection(
            *adapter_args,
            **adapter_kwargs,
            allow_short_query16=os.environ.get("QSA_HARNESS_SHORT_SCORE") == "1",
            direct_page_kernels=kernels,
            shared_direct_page_proof=True,
        )
        short_score = candidate.short_query16
        if short_score:
            hits = sum(candidate.query16_routes.values())
            bounded = sum(candidate.skip_invisible_routes.values())
            assert hits and not any(control.query16_routes.values()), name
            print(
                f"QSA_INT8_HARNESS_SHORT_SCORE {name} query16_chunks={hits} "
                f"bounded_chunks={bounded} total_chunks={len(candidate.chunks)}",
                flush=True,
            )
            if not diagnose_only:
                candidate_scores = torch.empty_like(scratch)
                for first, count in candidate.chunks:
                    candidate_scores[:count].fill_(float("nan"))
                    control.score_into(first, count, scratch[:count])
                    candidate.score_into(first, count, candidate_scores[:count])
                    torch.testing.assert_close(
                        candidate_scores[:count], scratch[:count], rtol=1e-3, atol=1e-3
                    )
                del candidate_scores
        else:
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
        original = _preflight_select(name, "control", control, diagnose_only)
        selected = _preflight_select(name, "candidate", candidate, diagnose_only)
        print(f"QSA_INT8_HARNESS_PREFLIGHT_START {name} cann_v1", flush=True)
        native_output = native_call()
        torch.npu.synchronize()
        assert native_output.shape == (rows, 1, 512)
        expected = native_output.cpu().squeeze(1)
        failures = []
        for label, indices in (
            ("cann_v1", expected),
            ("control", original),
            ("candidate", selected),
        ):
            failure = _check_preflight_structure(
                name, label, indices, inputs["visible"], verify_structure
            )
            if failure is not None:
                failures.append(failure)
        assert not failures, f"{name}: {', '.join(failures)}"
        assert torch.equal(original, selected), f"{name}: candidate indices differ"
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
        if not diagnose_only and stages:
            _measure_stages(
                name,
                route,
                candidate,
                selected,
                device,
                score_control=control.score_all if short_score else None,
            )
        if not diagnose_only and os.environ.get("QSA_HARNESS_SORTED_TOPK") == "1":
            assert not stages and not short_score
            assert os.environ.get("QSA_HARNESS_REUSE_TOPK_SCRATCH") != "1"
            candidate = _measure_sorted_topk_ab(
                name, benchmark, candidate, inputs["visible"], samples, refs,
                verify_structure, check_selection,
            )
        del original, selected, native_output, expected, control, refs, inputs

        if (
            not diagnose_only
            and os.environ.get("QSA_HARNESS_REUSE_TOPK_SCRATCH") == "1"
            and route == "auto"
            and topk._psegmerge_applicable(1, width, 512)
        ):
            from probe_qsa_int8_group_visible_request_910b import measure_request_wall
            from probe_qsa_int8_topk_workspace_910b import (
                ARMS,
                verify_workspace_request,
            )

            current = copy.copy(candidate)
            candidate.reuse_topk_scratch = True
            check_indices, audit = verify_workspace_request(
                torch, current, candidate, True, values["visible"].cpu()
            )
            adapters = dict(zip(ARMS, (current, candidate), strict=True))
            wall = measure_request_wall(
                ARMS,
                adapters.__getitem__,
                torch.npu.synchronize,
                check_indices,
                6,
                record_group_visible=False,
            )
            kernel_ms = {}
            for arm, adapter in adapters.items():
                kernel_ms[arm] = benchmark.get_latency(adapter)
                check_indices(output)
            assert all(math.isfinite(ms) and ms > 0 for ms in kernel_ms.values())
            kernel_speedup = kernel_ms[ARMS[0]] / kernel_ms[ARMS[1]]
            print(
                f"TopK scratch {name}: prepared wall {wall['speedup']:.3f}x "
                f"({wall['candidate_wins']}/6 pairs faster), "
                f"whole-pipeline device kernels {kernel_speedup:.3f}x",
                flush=True,
            )
            print(
                "QSA_INT8_HARNESS_TOPK_SCRATCH",
                json.dumps(
                    dict(
                        case=name,
                        audit=audit,
                        prepared_wall=wall,
                        kernel_ms=kernel_ms,
                        kernel_speedup=kernel_speedup,
                    )
                ),
                flush=True,
            )

        if not diagnose_only and name == "int8_budget_exact":
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
        yield (
            dict(
                case_name=name, route=route, reference=native_call, candidate=candidate
            ),
        )

    benchmark = QSAPrefillInt8Benchmark(
        op_name=OP_NAME,
        case_name=case_name,
        torch_op=_call_reference,
        gems_op=_call_candidate,
        input_fn=prepare_case,
        dtypes=[torch.int8],
        device=f"npu:{device_index}",
    )
    if diagnose_only:
        next(prepare_case((case_name,), torch.int8, benchmark.device))
    else:
        benchmark.run()
