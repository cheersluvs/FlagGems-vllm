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

"""Compare in-kernel QK plus Vector or second-Cube reduction."""

import hashlib
import json
import math
import os
import statistics
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def _qk_reduce(Q, K, QS, KS, W, Table, Visible, Out,
               ROWS: tl.constexpr, WIDTH: tl.constexpr, COLUMNS: tl.constexpr,
               SECOND_CUBE: tl.constexpr, DOT_ONLY: tl.constexpr = False, TILE_BASE: tl.constexpr = 0,
               LINEAR_Q: tl.constexpr = True, LINEAR_RAW: tl.constexpr = True):
    base = tl.program_id(0) * 4
    start = (tl.program_id(1) + TILE_BASE) * 64
    n = start + tl.arange(0, 64)
    m = tl.arange(0, 16)
    d = tl.arange(0, 128)
    qrow = base + m // 4
    if LINEAR_Q:
        head = base * 4 + m
        q = tl.load(Q + head[:, None] * 128 + d[None, :], mask=head[:, None] < ROWS * 4, other=0)
    else:
        q = tl.load(Q + (qrow[:, None] * 4 + m[:, None] % 4) * 128 + d[None, :],
                    mask=qrow[:, None] < ROWS, other=0)
    page = tl.load(Table + tl.minimum(start // 192, COLUMNS - 1))
    token = page * 192 + start % 192 + tl.arange(0, 64)
    key = tl.load(K + token[:, None] * 128 + d[None, :], mask=n[:, None] < WIDTH, other=0)
    dot = tl.dot(q, tl.trans(key), out_dtype=tl.int32)
    if DOT_ONLY:
        if LINEAR_RAW:
            head = base * 4 + m
            tl.store(Out + head[:, None] * WIDTH + n[None, :], dot,
                     mask=(head[:, None] < ROWS * 4) & (n[None, :] < WIDTH))
        else:
            tl.store(Out + (qrow[:, None] * 4 + m[:, None] % 4) * WIDTH + n[None, :], dot,
                     mask=(qrow[:, None] < ROWS) & (n[None, :] < WIDTH))
    else:
        ks = tl.load(KS + token, mask=n < WIDTH, other=0).to(tl.float32)
        if SECOND_CUBE:
            r = base + tl.arange(0, 16)
            qs = tl.load(QS + r[:, None] * 4 + m[None, :] % 4,
                         mask=r[:, None] < tl.minimum(base + 4, ROWS), other=0).to(tl.float32)
            w = tl.load(W + r[:, None] * 4 + m[None, :] % 4,
                        mask=r[:, None] < tl.minimum(base + 4, ROWS), other=0).to(tl.float32)
            weights = tl.where(tl.arange(0, 16)[:, None] == m[None, :] // 4, qs * w, 0.0).to(tl.float16)
            positive = (tl.maximum(dot.to(tl.float32), 0.0) * (1.0 / 1024.0)).to(tl.float16)
            total = tl.dot(weights, positive, out_dtype=tl.float32)
            visible = tl.load(Visible + r, mask=r < tl.minimum(base + 4, ROWS), other=0)
            score = tl.where(n[None, :] < visible[:, None], total * 1024.0 * ks[None, :], -float("inf"))
            tl.store(Out + r[:, None] * WIDTH + n[None, :], score,
                     mask=(r[:, None] < tl.minimum(base + 4, ROWS)) & (n[None, :] < WIDTH))
        else:
            h = tl.arange(0, 4)
            for local_query in tl.static_range(4):
                row = base + local_query
                products = al.extract_slice(dot, (local_query * 4, 0), (4, 64), (1, 1)).to(tl.float32)
                qs = tl.load(QS + row * 4 + h, mask=row < ROWS, other=0).to(tl.float32)
                w = tl.load(W + row * 4 + h, mask=row < ROWS, other=0).to(tl.float32)
                dequant = products * qs[:, None] * ks[None, :]
                total = tl.sum(tl.maximum(dequant, 0.0) * w[:, None], 0)
                visible = tl.load(Visible + row, mask=row < ROWS, other=0)
                score = tl.where(n < visible, total, -float("inf"))
                tl.store(Out + row * WIDTH + n, score, mask=(row < ROWS) & (n < WIDTH))


def prepare_current_int8_score(values, visible, rows, width, output, report, label, topk, *, reduction_cube=False, reduction_mode="cube"):
    """Bind the frozen integrated INT8 scorer, excluding preparation from timing."""
    from qsa_int8_shared_map_selection_910b_trial import PreparedSharedMapInt8Selection
    from probe_qsa_int8_page64_prefix_910b import load_generated_prefix
    from probe_qsa_int8_score_sol_910b import build_direct_page_source

    kernels = {}
    for bounded in (True, False):
        arm = "chunk_bounded" if bounded else "chunk_full"
        kernels[bounded] = load_generated_prefix(report, label, arm,
            build_direct_page_source(bounded, arm, precomputed_visibility=False))
    target = dict(values)
    for key in ("q", "qs", "weights", "row_table"):
        target[key] = values[key][:rows]
    target["visible"] = visible
    indices = torch.empty((rows, 512), device=output.device, dtype=torch.int32)
    workspace = torch.empty(1024**2, device=output.device, dtype=torch.uint8)
    adapter = PreparedSharedMapInt8Selection(target, indices, output, workspace, topk, [0, rows],
        key_group4=True, single_page64=True, allow_short_query16=True,
        direct_page_kernels=kernels, shared_direct_page_proof=True)
    identity = dict(entry="PreparedSharedMapInt8Selection.score_into", registration="unregistered INT8 integrated trial; public Ascend src entry is BF16",
        flags=dict(key_group4=True, single_page64=True, allow_short_query16=True, shared_direct_page_proof=True),
        routes={name:repr(getattr(adapter, name, None)) for name in
            ("query16_routes", "shared_routes", "skip_invisible_routes", "direct_page_routes")})
    root = Path(__file__).resolve().parent
    identity["source_sha256"] = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
        ("qsa_int8_shared_map_selection_910b_trial.py", "qsa_int8_query_pack4_910b.py", "qsa_int8_selection_910b_trial.py")}
    identity["cube_reduction_applied"] = False
    if reduction_cube:
        if adapter.query16_routes.get((0, rows), False):
            if adapter.direct_page_routes.get((0, rows)) in ("bounded", "full"):
                original = kernels[adapter.skip_invisible_routes[0, rows]]
            elif adapter.skip_invisible_routes.get((0, rows), False):
                original = adapter.visibility_kernel
            else:
                from qsa_int8_query_pack4_910b import _qsa_page64_page_reuse_full_score, _qsa_page64_query16_score
                original = _qsa_page64_page_reuse_full_score if adapter.page_reuse_full else _qsa_page64_query16_score
            from qsa_query16_cube_builder_910b import build_reduction_kernel
            candidate, rewrite = build_reduction_kernel(original, report, label, mode=reduction_mode)
            bound = adapter.bind_query16_score(0, rows, output)
            grid = (min(20, triton.cdiv(rows, 16) * triton.cdiv(triton.cdiv(width, 64), 8)),)
            identity.update(cube_reduction_applied=reduction_mode == "cube", reduction_mode=reduction_mode, rewrite=rewrite, grid=list(grid), launch_kwargs=bound.keywords)
            launch = partial(candidate[grid], *bound.args, **bound.keywords)
            launch.qsa_warmup = partial(candidate.warmup, *bound.args, grid=grid, **bound.keywords)
            return launch, identity
        if rows >= 16:
            raise NotImplementedError("Incremental reduction probe requires actual query16 route")
        identity["fallback_reason"] = "Partial query group: unchanged original fallback; no Cube reduction tested here"
    return partial(adapter.score_into, 0, rows, output), identity


def run_fused_probe(inputs, values, refs, width, timer, report, case):
    from triton.backends.ascend import compiler
    from diagnose_qsa_structural_mixed_compile_910b import CaptureCompiler

    report = Path(report)
    if os.environ.get("QSA_GENERIC_LOCAL_HANDOFF") == "1":
        from probe_qsa_generic_local_buffer_910b import run_probe
        return run_probe(inputs, values, refs, width, timer, report, case)
    path = report / f"{case}-fused-second-cube.json"
    result = dict(status="running", scope="single-launch scorer QK plus reduction; no TopK or full-shape acceptance",
                  query_group=4, cube_n=64, common_gm_qk_input=False,
                  q_load_layout="linear head index", raw_store_layout="linear head index",
                  compiler_gm_intermediate_status="unverified; inspect captured artifacts", arms={})
    funcs = {}
    topk_funcs, combined_funcs, selections = {}, {}, {}
    selection_mode = os.environ.get("QSA_FUSED_TEN_CHUNKS") == "1"
    if selection_mode:
        from qsa_prefill_pipeline_910b_trial import load_topk
        from probe_qsa_int8_pipeline_910b import check_selection
        topk = load_topk()
        result["topk_identity"] = topk.probe_identity
        result["selection_contract"] = "existing selected-score tolerance 1e-3; exact index overlap reported separately"
    try:
        assert bool((inputs["row_table"] == inputs["row_table"][0]).all())
        assert bool(torch.isfinite(inputs["qs"]).all() and (inputs["qs"] > 0).all())
        assert bool(torch.isfinite(inputs["ks"]).all() and (inputs["ks"] > 0).all())
        rows = inputs["q"].shape[0]
        arm_specs = [(False, "qk_vector_fp32"), (True, "qk_second_cube")]
        compare_current = os.environ.get("QSA_COMPARE_CURRENT_INT8") == "1"
        result["production_comparison_requested"] = compare_current
        if compare_current:
            assert selection_mode, "Current INT8 comparison requires selection-mode fixture"
            arm_specs.insert(0, (None, "current_int8_integrated"))
        incremental_cube = os.environ.get("QSA_QUERY16_REDUCTION_CUBE") == "1"
        result["incremental_query16_cube"] = incremental_cube
        if incremental_cube:
            assert compare_current, "Incremental probe requires current INT8 control"
            arm_specs = [(None, "current_int8_integrated"), ("query16", "query16_second_cube")]
        for cube, name in arm_specs:
            entry = dict(status="running", checks=[], samples_ms=[])
            result["arms"][name] = entry
            for fixture, count, scan in (("full", rows, width), ("tail_visible", 3, min(width, 193))):
                visible_cpu = inputs["visible"][:count].clone().clamp(max=scan)
                if fixture == "tail_visible":
                    visible_cpu[0] = 0
                    visible_cpu[1] = min(int(visible_cpu[1]), 191)
                visible = visible_cpu.to(values["q"].device)
                output = torch.empty((count, scan), device=values["q"].device, dtype=torch.float32)
                fn = partial(_qk_reduce[(triton.cdiv(count, 4), triton.cdiv(scan, 64))],
                    values["q"], values["cache"], values["qs"], values["ks"], values["weights"],
                    values["row_table"][0], visible, output,
                    ROWS=count, WIDTH=scan, COLUMNS=values["row_table"].shape[1], SECOND_CUBE=cube, use_bytecode=True)
                if cube is None or cube == "query16":
                    fn, identity = prepare_current_int8_score(values, visible, count, scan, output,
                        report, f"{case}_{name}_{fixture}", topk, reduction_cube=cube == "query16")
                    entry.setdefault("current_identity", {})[fixture] = identity
                capture = CaptureCompiler(report, f"{case}-{name}-{fixture}")
                entry["phase"] = "compilation"
                path.write_text(json.dumps(result, indent=2))
                with patch.object(compiler, "subprocess", capture):
                    if cube is None or cube == "query16":
                        compiled = fn()
                    else:
                        compiled = _qk_reduce.warmup(*fn.args,
                            grid=(triton.cdiv(count, 4), triton.cdiv(scan, 64)), **fn.keywords)
                entry.setdefault("compiler_calls", {})[fixture] = capture.calls
                entry.setdefault("artifacts", {})[fixture] = {}
                for kind, data in getattr(compiled, "asm", {}).items():
                    if isinstance(data, str):
                        filename = f"{case}-{name}-{fixture}-{kind}.txt"
                        (report / filename).write_text(data)
                        entry["artifacts"][fixture][kind] = dict(file=filename,
                            linalg_matmul_count=data.count("linalg.matmul"), triton_dot_count=data.count("tt.dot"))
                    elif isinstance(data, bytes):
                        entry["artifacts"][fixture][kind] = dict(sha256=hashlib.sha256(data).hexdigest())
                entry["phase"] = "device_execution"
                path.write_text(json.dumps(result, indent=2))
                fn()
                actual = output.cpu()
                expected = torch.full((count, scan), -float("inf"))
                for row in range(count):
                    live = min(int(visible_cpu[row]), refs[row].numel(), scan)
                    expected[row, :live] = refs[row][:live]
                finite = torch.isfinite(expected)
                close = torch.isclose(actual, expected, rtol=1e-3, atol=1e-3)
                error = float((actual[finite] - expected[finite]).abs().max()) if bool(finite.any()) else 0.0
                check = dict(fixture=fixture, rows=count, width=scan, mismatches=int((~close).sum()),
                    max_abs_error=error if math.isfinite(error) else None,
                    invalid_output=int((~torch.isfinite(actual[finite])).sum()), compiler_calls=capture.calls)
                if not bool(close.all()):
                    bad = ~close
                    coords = bad.nonzero()[:16]
                    check["mismatches_per_row"] = bad.sum(1).tolist()
                    check["mismatches_per_tile"] = [int(bad[:, start:start + 64].sum()) for start in range(0, scan, 64)]
                    check["examples"] = [dict(row=int(r), key=int(n), actual=float(actual[r,n]) if math.isfinite(float(actual[r,n])) else str(float(actual[r,n])),
                        expected=float(expected[r,n]) if math.isfinite(float(expected[r,n])) else str(float(expected[r,n]))) for r,n in coords]
                entry["checks"].append(check)
                if fixture == "full":
                    funcs[name] = fn
                    if selection_mode and bool(close.all()):
                        indices = torch.empty((count, 512), device=output.device, dtype=torch.int32)
                        workspace = torch.empty(1024**2, device=output.device, dtype=torch.uint8)
                        select = partial(topk.persistent_topk, output, visible, indices, workspace,
                                         k=512, max_seq_len=scan, block=256)
                        select()
                        selected = indices.cpu()
                        selection, _ = check_selection(selected, refs)
                        entry["selection"] = selection
                        selections[name] = selected
                        if name == "qk_second_cube" and "qk_vector_fp32" in selections:
                            baseline_indices = selections["qk_vector_fp32"]
                            entry["exact_index_overlap_counts"] = [len(set(a[a >= 0].tolist()) & set(b[b >= 0].tolist()))
                                for a,b in zip(baseline_indices, selected)]
                        if name != "current_int8_integrated" and "current_int8_integrated" in selections:
                            baseline_indices = selections["current_int8_integrated"]
                            entry["exact_index_overlap_with_current"] = [len(set(a[a >= 0].tolist()) & set(b[b >= 0].tolist()))
                                for a,b in zip(baseline_indices, selected)]
                        topk_funcs[name] = select
                        combined_funcs[name] = lambda score=fn, select=select: (score(), select())
                print("FUSED_SECOND_CUBE_CHECK " + json.dumps(dict(arm=name, **check)), flush=True)
                path.write_text(json.dumps(result, indent=2))
            entry["status"] = "passed" if all(c["mismatches"] == 0 for c in entry["checks"]) else "rejected_precision"
            if selection_mode and entry["status"] == "passed" and not entry["selection"]["checks_passed"]:
                entry["status"] = "rejected_selection"
            if entry["status"] != "passed":
                topk_funcs.pop(name, None)
                combined_funcs.pop(name, None)
                funcs.pop(name, None)
                if name in ("qk_vector_fp32", "current_int8_integrated"):
                    raise RuntimeError("FP32 control mismatch; preserving distributions and refusing second Cube execution")
        baseline_name = "current_int8_integrated" if incremental_cube else "qk_vector_fp32"
        assert result["arms"][baseline_name]["status"] == "passed", "FP32 control mismatch"
        for iteration in range(2):
            names = tuple(funcs) if iteration == 0 else tuple(reversed(funcs))
            for name in names:
                latency = float(timer.get_latency(funcs[name]))
                if not math.isfinite(latency) or latency <= 0:
                    raise RuntimeError(f"Invalid latency: {name} {latency}")
                result["arms"][name]["samples_ms"].append(latency)
        base = statistics.median(result["arms"][baseline_name]["samples_ms"])
        for name in funcs:
            e = result["arms"][name]
            e.update(median_us=statistics.median(e["samples_ms"]) * 1000,
                     speedup_over_scoring_control=base / statistics.median(e["samples_ms"]))
            print("FUSED_SECOND_CUBE_TIME " + json.dumps(dict(arm=name, **e)), flush=True)
        if selection_mode:
            result["component_timing_scope"] = "representative chunk only; no full-input or CANN acceptance ratio"
            for name in funcs:
                result["arms"][name]["component_samples_ms"] = dict(topk=[], score_plus_topk=[])
            for iteration in range(2):
                order = tuple(funcs) if iteration == 0 else tuple(reversed(funcs))
                for name in order:
                    stages = (("topk", topk_funcs[name]), ("score_plus_topk", combined_funcs[name]))
                    for stage, function in (stages if iteration == 0 else tuple(reversed(stages))):
                        value = float(timer.get_latency(function))
                        if not math.isfinite(value) or value <= 0:
                            raise RuntimeError(f"Invalid {name}/{stage} latency: {value}")
                        result["arms"][name]["component_samples_ms"][stage].append(value)
            for name in funcs:
                entry = result["arms"][name]
                entry["component_median_us"] = {stage: statistics.median(times)*1000 for stage,times in entry["component_samples_ms"].items()}
                print("FUSED_SELECTION_TIME " + json.dumps(dict(arm=name, **entry["component_median_us"])), flush=True)
        if compare_current:
            current = result["arms"]["current_int8_integrated"]
            for name in funcs:
                entry = result["arms"][name]
                entry["speedup_over_current_score"] = current["median_us"] / entry["median_us"]
                entry["speedup_over_current_total"] = current["component_median_us"]["score_plus_topk"] / entry["component_median_us"]["score_plus_topk"]
                print("VERSUS_CURRENT_INT8 " + json.dumps(dict(arm=name,
                    score_speedup=entry["speedup_over_current_score"], total_speedup=entry["speedup_over_current_total"])), flush=True)
        if os.environ.get("QSA_QUERY16_REDUCTION_DIAG") == "1":
            from probe_qsa_query16_reduction_diag_910b import run_diagnostics
            result["reduction_diagnostics"] = run_diagnostics(
                inputs, values, refs, width, timer, report, case, topk, funcs)
        result["status"] = "completed" if len(funcs) == len(arm_specs) else ("completed_with_candidate_rejection" if selection_mode else "completed_with_precision_rejection")
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def run_qk_isolation(inputs, values, width, report, case):
    """Check exact raw QK over controlled query/tile launch extents."""
    from triton.backends.ascend import compiler
    from diagnose_qsa_structural_mixed_compile_910b import CaptureCompiler

    report = Path(report)
    path = report / f"{case}-raw-qk-isolation.json"
    result = dict(status="running", scope="raw INT32 QK only; no reduction, TopK or performance claim", controls=[], checks=[],
                  selected_layout=dict(LINEAR_Q=True, LINEAR_RAW=True))
    sentinel = 1515870810
    try:
        rows = inputs["q"].shape[0]
        assert rows >= 4 and rows % 4 == 0
        assert bool((inputs["row_table"] == inputs["row_table"][0]).all())
        mapping = inputs["row_table"][0, :(width + 191) // 192].long()
        keys = inputs["cache"][mapping, :, 0].reshape(-1, 128)[:width].int()
        reference = (inputs["q"].reshape(-1, 128).int() @ keys.T).reshape(rows, 4, width)
        tiles = triton.cdiv(width, 64)
        specs = (("group0_tile0", 0, 4, 0, 1),
                 ("group0_last_tile", 0, 4, tiles - 1, 1),
                 ("last_group_tile0", rows - 4, 4, 0, 1),
                 ("group0_all_tiles", 0, 4, 0, tiles),
                 ("all_groups_tile0", 0, rows, 0, 1),
                 ("full_first", 0, rows, 0, tiles),
                 ("full_repeat", 0, rows, 0, tiles))
        matrix = [(f"layout_q{int(q)}_raw{int(raw)}", 0, 4, 0, 1, q, raw, True)
                  for q, raw in ((False, False), (True, False), (False, True), (True, True))]
        if os.environ.get("QSA_QK_REVERSE_MATRIX") == "1":
            matrix.reverse()
        expanded = matrix + [(*spec, True, True, False) for spec in specs]
        for label, first, count, tile_base, tile_count, linear_q, linear_raw, is_control in expanded:
            if not is_control and next(e for e in result["controls"] if e["LINEAR_Q"] and e["LINEAR_RAW"])["status"] != "passed":
                result["error"] = "Both-linear single-tile control failed; full-grid and reduction tests not attempted"
                break
            entry = dict(LINEAR_Q=linear_q, LINEAR_RAW=linear_raw, fixture=label, first_row=first, rows=count, tile_base=tile_base,
                         tile_count=tile_count, grid=[count // 4, tile_count], phase="compilation")
            result["controls" if is_control else "checks"].append(entry)
            path.write_text(json.dumps(result, indent=2))
            output = torch.full((count, 4, width), sentinel, device=values["q"].device, dtype=torch.int32)
            args = (values["q"][first:first + count], values["cache"], values["qs"][first:first + count], values["ks"],
                    values["weights"][first:first + count], values["row_table"][first], values["visible"][first:first + count], output)
            options = dict(ROWS=count, WIDTH=width, COLUMNS=values["row_table"].shape[1],
                           SECOND_CUBE=False, DOT_ONLY=True, TILE_BASE=tile_base,
                           LINEAR_Q=linear_q, LINEAR_RAW=linear_raw, use_bytecode=True)
            grid = (count // 4, tile_count)
            capture = CaptureCompiler(report, f"{case}-raw-qk-{label}")
            with patch.object(compiler, "subprocess", capture):
                compiled = _qk_reduce.warmup(*args, grid=grid, **options)
            entry["compiler_calls"] = capture.calls
            entry["metadata"] = str(compiled.metadata)
            for kind, data in getattr(compiled, "asm", {}).items():
                if isinstance(data, str):
                    (report / f"{case}-raw-qk-{label}-{kind}.txt").write_text(data)
            entry["phase"] = "device_execution"
            path.write_text(json.dumps(result, indent=2))
            _qk_reduce[grid](*args, **options)
            actual = output.cpu()
            start, end = tile_base * 64, min(width, (tile_base + tile_count) * 64)
            selected = actual[:, :, start:end]
            expected = reference[first:first + count, :, start:end]
            bad = selected != expected
            entry.update(mismatches=int(bad.sum()), mismatches_per_row_head=bad.sum(2).tolist(),
                mismatches_per_tile=[int(bad[:, :, off:off + 64].sum()) for off in range(0, end-start, 64)],
                unwritten_selected=int((selected == sentinel).sum()),
                unexpected_writes=int((actual[:, :, :start] != sentinel).sum()) + int((actual[:, :, end:] != sentinel).sum()))
            entry["examples"] = [dict(row=first + int(r), head=int(h), key=start + int(n),
                actual=int(selected[r,h,n]), expected=int(expected[r,h,n])) for r,h,n in bad.nonzero()[:16]]
            entry["status"] = "passed" if entry["mismatches"] == 0 and entry["unexpected_writes"] == 0 else "failed"
            print("RAW_QK_CHECK " + json.dumps(entry), flush=True)
            path.write_text(json.dumps(result, indent=2))
        result["status"] = "passed" if len(result["checks"]) == len(specs) and all(e["status"] == "passed" for e in result["checks"]) else "failed"
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
