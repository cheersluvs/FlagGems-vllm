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
    GROUP_VISIBLE_CASES,
    GROUP_VISIBLE_WALL_CASES,
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


def _check_score_pair(name, current, variant, expected):
    target = torch.empty_like(current.scratch)
    for first, count in variant.chunks:
        target[:count].fill_(float("nan"))
        current.score_into(first, count, current.scratch[:count])
        variant.score_into(first, count, target[:count])
        torch.testing.assert_close(
            target[:count], current.scratch[:count], rtol=0, atol=0
        )
    del target
    actual = _preflight_select(name, "score_trial", variant, False)
    assert torch.equal(actual, expected), "Score trial changed full TopK indices"


def _measure_stage_pairs(benchmark, stages, output, expected, label):
    metrics = {}
    for stage, arms in stages.items():
        samples = [[], []]
        for order in ((0, 1), (1, 0)):
            for i in order:
                ms = benchmark.get_latency(arms[i])
                assert math.isfinite(ms) and ms > 0
                samples[i].append(ms)
                if stage != "score":
                    assert torch.equal(output.cpu(), expected)
        before, after = map(statistics.median, samples)
        metrics[stage] = dict(
            current_ms=before,
            speedup=before / after,
            current_samples_ms=samples[0],
            **{f"{label}_ms": after, f"{label}_samples_ms": samples[1]},
        )
        print(f"{label} {stage}: {before:.3f} → {after:.3f} ms, {before / after:.3f}x")
    return metrics


def _measure_tile_handoff(benchmark, current, wide_kernel, expected):
    variant = copy.copy(current)
    variant.direct_page_kernels = {**current.direct_page_kernels, True: wide_kernel}
    hits = sum(route == "bounded" for route in variant.direct_page_routes.values())
    assert hits, "No guarded bounded chunks for N128"
    _check_score_pair("int8_budget_exact", current, variant, expected)
    metrics = _measure_stage_pairs(
        benchmark,
        {
            "score": (current.score_all, variant.score_all),
            "combined": (current, variant),
        },
        variant.output,
        expected,
        "n128",
    )
    print(
        "QSA_INT8_HARNESS_TILE_HANDOFF",
        json.dumps(dict(hits=hits, total_chunks=len(variant.chunks), metrics=metrics)),
        flush=True,
    )
    return variant


def _measure_group_visible(
    benchmark,
    name,
    current,
    make_adapter,
    expected,
    *,
    diagnose_wall=False,
    compare_request_table=False,
):
    from probe_qsa_int8_group_visible_request_910b import measure_request_wall

    variant = make_adapter("group_visible")
    if compare_request_table:
        assert variant.group_visible_offsets == current.group_visible_offsets
        assert torch.equal(
            variant.group_visible_metadata, current.group_visible_metadata
        )
    assert variant.direct_page_routes == current.direct_page_routes
    assert variant.query16_routes == current.query16_routes
    assert variant.skip_invisible_routes == current.skip_invisible_routes
    routes = {
        key: route
        for key, route in current.direct_page_routes.items()
        if route in ("bounded", "full")
    }
    assert variant.group_visible_offsets.keys() == routes.keys()
    _check_score_pair(name, current, variant, expected)
    metrics = None
    if routes and not diagnose_wall:
        scores = torch.empty(
            (current.rows, current.width),
            dtype=torch.float32,
            device=current.scratch.device,
        )
        for first, count in current.chunks:
            current.score_into(first, count, scores[first : first + count])
        torch.npu.synchronize()
        metrics = _measure_stage_pairs(
            benchmark,
            {
                "score": (current.score_all, variant.score_all),
                "topk": (
                    partial(current.topk_all, scores),
                    partial(variant.topk_all, scores),
                ),
                "combined": (current, variant),
            },
            variant.output,
            expected,
            "group_visible",
        )
        del scores
    elif not routes:
        assert not variant.group_visible_stats["copies"]
        print(
            f"GroupVisible {name}: zero eligible chunks, no metadata upload", flush=True
        )

    def check_indices(actual):
        assert torch.equal(actual.cpu(), expected), "Fresh adapter changed TopK indices"

    wall = measure_request_wall(
        ("current", "group_visible"),
        make_adapter,
        torch.npu.synchronize,
        check_indices,
        12 if diagnose_wall else 6,
    )
    print(f"GroupVisible fresh wall {name}: {wall['speedup']:.3f}x", flush=True)
    host_phases = {}
    if compare_request_table:
        host_phases["metadata_sources"] = {
            "current": "device row-table readback",
            "group_visible": "trusted CPU request table",
        }
    if diagnose_wall:
        assert routes
        host_phases["host_phases"] = measure_request_wall(
            ("current", "group_visible"),
            partial(make_adapter, instrument=True),
            torch.npu.synchronize,
            check_indices,
            6,
            record_host_phases=True,
        )
        print("GroupVisible host phase paired savings (ms, positive is faster)")
        for phase, saved in host_phases["host_phases"][
            "phase_paired_median_saved_us"
        ].items():
            print(f"  {phase.removesuffix('_us')}: {saved / 1000:+.3f}")
    print(
        "QSA_INT8_HARNESS_GROUP_VISIBLE",
        json.dumps(
            dict(
                hits=len(routes),
                bounded_chunks=list(routes.values()).count("bounded"),
                total_chunks=len(variant.chunks),
                metadata=variant.group_visible_stats,
                metrics=metrics,
                fresh_wall=wall,
                **host_phases,
            )
        ),
        flush=True,
    )
    return variant


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
    from probe_qsa_int8_score_sol_910b import (
        build_direct_page_source,
        build_paired_tile_source,
    )
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
        group_visible_mode = os.environ.get("QSA_HARNESS_GROUP_VISIBLE")
        compare_request_table = (
            os.environ.get("QSA_HARNESS_TRUSTED_REQUEST_TABLE") == "1"
        )
        if compare_request_table and (
            group_visible_mode != "wall" or name not in GROUP_VISIBLE_WALL_CASES
        ):
            pytest.fail("Trusted request table trial requires GroupVisible wall mode")
        inputs = make_inputs(plan, include_request_table=compare_request_table)
        request_table = (
            inputs.pop("request_table").tolist() if compare_request_table else None
        )
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
        allow_short_query16 = os.environ.get("QSA_HARNESS_SHORT_SCORE", "1") == "1"
        candidate = PreparedSharedMapInt8Selection(
            *adapter_args,
            **adapter_kwargs,
            allow_short_query16=allow_short_query16,
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
        online_topk = os.environ.get("QSA_HARNESS_ONLINE_TOPK") == "1"
        if os.environ.get("QSA_HARNESS_FUSED_BINARY") == "1" or online_topk:
            if online_topk:
                from qsa_int8_online_topk_910b_trial import PreparedOnlineTopKSelection

                mixed = PreparedOnlineTopKSelection(
                    values, inputs["visible"], plan["starts"], chunk,
                )
                trial = "online_topk"
                hit_chunks = len(mixed.chunks)
            else:
                from qsa_int8_fused_binary_910b_trial import PreparedFusedBinarySelection

                mixed = PreparedFusedBinarySelection(
                    values, inputs["visible"], plan["starts"], chunk,
                )
                trial = "fused_binary"
                hit_chunks = sum(item[3] for item in mixed.chunks)
            assert hit_chunks, f"{name}: mixed candidate did not run"
            print(f"QSA_INT8_HARNESS_PREFLIGHT_START {name} {trial}", flush=True)
            mixed.output.fill_(-3)
            torch.npu.synchronize()
            mixed_output = mixed(synchronize=diagnose_only)
            if mixed_output is None and diagnose_only and os.environ.get("QSA_MERGE53_MODE"):
                pytest.skip("Single-merge diagnostic completed; full fused output not checked")
            torch.npu.synchronize()
            mixed_indices = mixed_output.cpu()
            failure = _check_preflight_structure(
                name, trial, mixed_indices, inputs["visible"], verify_structure,
            )
            assert failure is None, failure
            checks, _ = check_selection(mixed_indices[samples], refs)
            assert checks["checks_passed"], (name, trial, checks)
            print(
                f"QSA INT8 MIXED FULL PASS {name} rows={rows} chunks={len(mixed.chunks)} "
                f"sampled_topk_rows={len(samples)}",
                flush=True,
            )
            if not diagnose_only:
                production_ms = benchmark.get_latency(candidate)
                assert math.isfinite(production_ms) and production_ms > 0
                print(
                    "QSA_INT8_HARNESS_ONLINE_TOPK" if online_topk else "QSA_INT8_HARNESS_FUSED_BINARY",
                    json.dumps({
                        "case": name, "production_ms": production_ms,
                        "online_chunks" if online_topk else "fused_chunks": hit_chunks,
                        "total_chunks": len(mixed.chunks), "sampled_rows": len(samples),
                        "chunk_rows": mixed.chunk_rows if online_topk else chunk,
                        "control_short_query16_enabled": allow_short_query16,
                        "control_short_query16_active": candidate.short_query16,
                        "control_query16_chunks": sum(candidate.query16_routes.values()),
                    }),
                    flush=True,
                )
            yield (dict(case_name=name, route=route, reference=native_call, candidate=mixed),)
            return
        if not diagnose_only and stages:
            _measure_stages(
                name,
                route,
                candidate,
                selected,
                device,
                score_control=(
                    control.score_all
                    if short_score and os.environ.get("QSA_HARNESS_SHORT_SCORE") == "1"
                    else None
                ),
            )
        if (
            not diagnose_only
            and name == "int8_budget_exact"
            and os.environ.get("QSA_HARNESS_TILE_HANDOFF") == "1"
        ):
            arm = "paired_n128"
            wide_kernel = load_generated_prefix(
                tmp_path,
                "harness",
                arm,
                build_paired_tile_source(arm, wide=True),
            )
            candidate = _measure_tile_handoff(
                benchmark, candidate, wide_kernel, selected
            )
        if not diagnose_only and (
            group_visible_mode == "full"
            or (group_visible_mode == "1" and name in GROUP_VISIBLE_CASES)
            or (group_visible_mode == "wall" and name in GROUP_VISIBLE_WALL_CASES)
        ):
            group_kernels = {}
            for bounded in (True, False):
                arm = "group_visible_bounded" if bounded else "group_visible_full"
                group_kernels[bounded] = load_generated_prefix(
                    tmp_path,
                    "harness",
                    arm,
                    build_direct_page_source(
                        bounded, arm, precomputed_visibility=True, request_offset=True
                    ),
                )

            def make_adapter(arm, *, instrument=False):
                return PreparedSharedMapInt8Selection(
                    *adapter_args,
                    **adapter_kwargs,
                    allow_short_query16=short_score,
                    direct_page_kernels=kernels,
                    shared_direct_page_proof=True,
                    group_visible_kernels=(
                        group_kernels
                        if compare_request_table or arm == "group_visible"
                        else None
                    ),
                    group_visible_direct_only=True,
                    time_page_proof=instrument,
                    trusted_request_table=(
                        request_table if arm == "group_visible" else None
                    ),
                )

            candidate = _measure_group_visible(
                benchmark,
                name,
                make_adapter("current") if compare_request_table else candidate,
                make_adapter,
                selected,
                diagnose_wall=group_visible_mode == "wall",
                compare_request_table=compare_request_table,
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


@pytest.mark.qsa_select_paged_prefill_int8_910b
@pytest.mark.parametrize("case_name", ("int8_budget_exact", "int8_compact"))
def test_qsa_select_paged_prefill_int8_chunk_stages(monkeypatch, tmp_path, case_name):
    from types import SimpleNamespace

    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    device_index = int(os.environ.get("QSA_DEVICE", torch.npu.current_device()))
    torch.npu.set_device(device_index)
    if "910B" not in torch.npu.get_device_name(device_index):
        pytest.skip("Ascend 910B required")
    assert Config.mode == consts.BenchMode.KERNEL
    assert os.environ.get("ASCEND_LAUNCH_BLOCKING") != "1"

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    from probe_qsa_int8_page64_prefix_910b import load_generated_prefix
    from probe_qsa_int8_pipeline_910b import check_selection
    from probe_qsa_int8_score_910b import reference as cpu_score
    from probe_qsa_int8_score_sol_910b import build_direct_page_source
    from probe_qsa_int8_ten_shapes_910b import verify_structure
    from qsa_int8_online_topk_910b_trial import PreparedOnlineTopKSelection
    from qsa_int8_shared_map_selection_910b_trial import PreparedSharedMapInt8Selection
    from qsa_prefill_pipeline_910b_trial import load_topk
    from qsa_timeline_910b import collect

    plan = CASES[(case_name,)]
    rows = 32
    assert plan["query_lengths"][0] >= rows
    width = plan["logits_width"]
    inputs = make_inputs(plan)
    for key in ("q", "qs", "weights", "row_table", "visible"):
        inputs[key] = inputs[key][:rows].contiguous()
    device = f"npu:{device_index}"
    values = {key: value.to(device) for key, value in inputs.items()}
    topk = load_topk()
    policy = os.environ.get("QSA_HARNESS_TOPK_POLICY", "prefill_3seg")
    route = topk_route(plan, policy)
    assert route == "auto", "The two chunk controls use the automatic TopK route"
    monkeypatch.setenv("FLAGGEMS_ASCEND_PTK_PATH", "auto")
    kernels = {}
    for bounded in (True, False):
        arm = "chunk_bounded" if bounded else "chunk_full"
        kernels[bounded] = load_generated_prefix(
            tmp_path, "chunk", arm,
            build_direct_page_source(bounded, arm, precomputed_visibility=False),
        )
    output = torch.empty((rows, 512), device=device, dtype=torch.int32)
    scratch = torch.empty((rows, width), device=device, dtype=torch.float32)
    workspace = torch.empty(1024**2, device=device, dtype=torch.uint8)
    current = PreparedSharedMapInt8Selection(
        values, output, scratch, workspace, topk, [0, rows],
        key_group4=True, single_page64=True, direct_page_kernels=kernels,
        shared_direct_page_proof=True, allow_short_query16=True,
    )
    online = PreparedOnlineTopKSelection(
        values, inputs["visible"], [0, rows], rows, scan_width=width,
    )
    assert current.chunks == [(0, rows)] and online.chunks == [(0, rows)]
    if case_name == "int8_compact":
        assert current.short_query16 and current.query16_routes[(0, rows)]

    def score():
        current.score_into(0, rows, scratch)

    def select():
        current.topk_all(scratch)

    def serial():
        score()
        select()

    variants = {"online": online}
    if os.environ.get("QSA_HARNESS_TOPK_GRAIN") == "1":
        for size in (256, 512):
            variants[f"online{size}"] = PreparedOnlineTopKSelection(
                values, inputs["visible"], [0, rows], rows, scan_width=width, update_keys=size,
            )
    if os.environ.get("QSA_HARNESS_HANDOFF_GRAIN") == "1":
        for size in (256, 512):
            variants[f"handoff{size}"] = PreparedOnlineTopKSelection(
                values, inputs["visible"], [0, rows], rows, scan_width=width, handoff_keys=size,
            )
    if os.environ.get("QSA_HARNESS_QUERY_PACK") == "1":
        variants["query4"] = PreparedOnlineTopKSelection(
            values, inputs["visible"], [0, rows], rows, scan_width=width,
            queries_per_program=4,
        )
    if os.environ.get("QSA_HARNESS_MIXED_PREFIX") == "1":
        for stage, label in enumerate(("prefix_score", "prefix_sort", "prefix_merge"), 1):
            variants[label] = PreparedOnlineTopKSelection(
                values, inputs["visible"], [0, rows], rows, scan_width=width,
                prefix_stage=stage,
            )
    handoff_labels = ("qk_only", "qk_handoff", "qk_score")
    if os.environ.get("QSA_HARNESS_SCORE_HANDOFF_PREFIX") == "1":
        for stage, label in enumerate(handoff_labels):
            variants[label] = PreparedOnlineTopKSelection(
                values, inputs["visible"], [0, rows], rows, scan_width=width,
                handoff_stage=stage,
            )
    serial()
    for label, adapter in variants.items():
        print(f"QSA INT8 CHUNK COMPILE {case_name} {label}", flush=True)
        adapter()
        torch.npu.synchronize()
        print(f"QSA INT8 CHUNK EXECUTED {case_name} {label}", flush=True)
    refs = [cpu_score(inputs, row)[1] for row in range(rows)]
    outputs = {"current": output.cpu()}
    outputs.update({
        label: adapter.output.cpu() for label, adapter in variants.items()
        if label not in ("prefix_score", "prefix_sort", *handoff_labels)
    })
    for label, indices in outputs.items():
        verify_structure(indices, inputs["visible"])
        checks, _ = check_selection(indices, refs)
        assert checks["checks_passed"], (case_name, label, checks)
    for label in ("prefix_score", "prefix_sort", "prefix_merge"):
        if label not in variants:
            continue
        adapter = variants[label]
        assert len(adapter.prefix_tiles) == 1
        pairs = adapter.prefix_tiles[0].cpu().reshape(rows, -1, 64, 2)
        tile_scores = pairs[..., 0].contiguous()
        tile_indices = pairs[..., 1].contiguous().view(torch.int32)
        for row, ref in enumerate(refs):
            expected = torch.full((pairs.shape[1] * 64,), -float("inf"))
            expected[:ref.numel()] = ref
            indices = tile_indices[row].to(torch.int64)
            assert torch.equal(torch.sort(indices.reshape(-1)).values, torch.arange(expected.numel()))
            tile_start = torch.arange(pairs.shape[1])[:, None] * 64
            assert ((indices >= tile_start) & (indices < tile_start + 64)).all()
            torch.testing.assert_close(tile_scores[row], expected[indices], rtol=1e-3, atol=1e-3)
            if label != "prefix_score":
                assert (tile_scores[row, :, :-1] >= tile_scores[row, :, 1:]).all()
    if os.environ.get("QSA_HARNESS_SCORE_HANDOFF_PREFIX") == "1":
        assert torch.equal(inputs["row_table"], inputs["row_table"][:1].expand_as(inputs["row_table"]))
        tiles = (width + 63) // 64
        logical_keys = inputs["cache"][inputs["row_table"][0].long(), :, 0].reshape(-1, 128)[:tiles * 64].int()
        expected_dots = torch.stack([
            inputs["q"][row].int() @ logical_keys.T for row in range(rows)
        ]).reshape(rows, 4, tiles, 64).permute(0, 2, 1, 3)
        handoff_checks = []
        for stage, label in enumerate(handoff_labels):
            adapter = variants[label]
            assert len(adapter.handoff_tiles) == 1
            raw = adapter.handoff_tiles[0].cpu()
            assert raw.dtype == (torch.int32 if stage < 2 else torch.float32), (label, raw.dtype)
            observed = raw.reshape(rows // 2, tiles, 2, 4, 64).permute(0, 2, 1, 3, 4).reshape(rows, tiles, 4, 64)
            if stage < 2:
                delta = (observed.long() - expected_dots.long()).abs()
                check = dict(
                    stage=label, dtype=str(raw.dtype), mismatches=int((delta != 0).sum()),
                    max_abs_error=int(delta.max()), nonzero=int(torch.count_nonzero(observed)),
                    observed_sample=observed.reshape(-1)[:8].tolist(),
                    expected_sample=expected_dots.reshape(-1)[:8].tolist(),
                )
            else:
                mismatches = 0
                for row, ref in enumerate(refs):
                    expected = torch.full((tiles * 64,), -float("inf"))
                    expected[:ref.numel()] = ref
                    expected = expected.reshape(tiles, 1, 64).expand(tiles, 4, 64)
                    mismatches += int((~torch.isclose(observed[row], expected, rtol=1e-3, atol=1e-3)).sum())
                check = dict(stage=label, dtype=str(raw.dtype), mismatches=mismatches)
            handoff_checks.append(check)
            print(f"QSA SCORE HANDOFF CHECK {case_name}: {json.dumps(check)}", flush=True)
        check_path = Path(os.environ["QSA_HARNESS_REPORT"]) / f"{case_name}-handoff-checks.json"
        check_path.write_text(json.dumps(handoff_checks, indent=2))
        assert all(check["mismatches"] == 0 for check in handoff_checks), (case_name, handoff_checks)
    staged = scratch.cpu()
    for row, ref in enumerate(refs):
        torch.testing.assert_close(staged[row, :ref.numel()], ref, rtol=1e-3, atol=1e-3)
        assert torch.isneginf(staged[row, ref.numel():]).all()
    functions = {"score": score, "topk": select, "serial": serial, **variants}
    print(
        f"QSA INT8 CHUNK PREFLIGHT PASS {case_name} rows={rows} width={width} "
        f"online_tiles={(width + 63) // 64} "
        f"short_query16={current.short_query16} "
        f"query16_chunks={sum(current.query16_routes.values())} "
        f"current_routes={current.route_checks}",
        flush=True,
    )
    for stage, execute in functions.items():
        def stage_inputs(shape, dtype, target_device):
            yield (dict(case_name=case_name, route=route, candidate=execute),)

        QSAPrefillInt8Benchmark(
            op_name=f"{OP_NAME}_chunk_{stage}", case_name=case_name,
            torch_op=None, gems_op=_call_candidate, input_fn=stage_inputs,
            dtypes=[torch.int8], device=device,
        ).run()
    report = Path(os.environ["QSA_HARNESS_REPORT"])
    profile = {
        "case": case_name, "rows": rows, "width": width,
        "current_short_query16": current.short_query16,
        "current_query16_chunks": sum(current.query16_routes.values()),
    }
    if len(variants) > 1:
        samples = {name: [] for name in variants}
        timer = QSAPrefillInt8Benchmark(
            op_name=f"{OP_NAME}_chunk_online", case_name=case_name,
            torch_op=None, gems_op=_call_candidate, input_fn=stage_inputs,
            dtypes=[torch.int8], device=device,
        )
        for round_index in range(2):
            names = tuple(variants) if round_index == 0 else tuple(reversed(variants))
            for name in names:
                samples[name].append(timer.get_latency(variants[name]))
        profile["paired_kernel_samples_ms"] = samples
        if os.environ.get("QSA_HARNESS_TOPK_GRAIN") == "1":
            profile["update_grains"] = [64, 256, 512]
        if os.environ.get("QSA_HARNESS_HANDOFF_GRAIN") == "1":
            profile["handoff_grains"] = [64, 256, 512]
        if os.environ.get("QSA_HARNESS_QUERY_PACK") == "1":
            profile["query_packs"] = [2, 4]
        if os.environ.get("QSA_HARNESS_MIXED_PREFIX") == "1":
            profile["mixed_prefix"] = ["prefix_score", "prefix_sort", "prefix_merge"]
        if os.environ.get("QSA_HARNESS_SCORE_HANDOFF_PREFIX") == "1":
            profile["score_handoff_prefix"] = list(handoff_labels)
    try:
        collect(
            SimpleNamespace(report=report, worker=case_name), functions, torch, profile,
            selected=tuple(functions),
        )
        profile["profile_status"] = "completed"
    except Exception as exc:
        profile.update(profile_status="failed", profile_error=repr(exc))
    (report / f"{case_name}-chunk-profile.json").write_text(json.dumps(profile, indent=2))
