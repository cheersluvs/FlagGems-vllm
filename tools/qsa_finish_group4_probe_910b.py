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

"""Probe four-query output reuse on the frozen fast raw scoring pipeline."""

import collections
import hashlib
import importlib.util
import json
import statistics
import sys
from types import SimpleNamespace
from unittest.mock import patch

from qsa_finish_page64_probe_910b import source as finish_source
from qsa_k_pack_2d_probe_910b import source as k_pack_source
from qsa_qw_pack_2d_probe_910b import source as qw_pack_source
from qsa_raw_vector_profile_910b import hardware_pass, summarize_vector_ir


def source():
    return """import triton
import triton.language as tl

@triton.jit
def raw_finish_group4_part(Raw, KS, Table, Visible, Out, group, tile,
                          ROWS:tl.constexpr, WIDTH:tl.constexpr, RAW_TILES:tl.constexpr):
    r = tl.arange(0,4)
    c = tl.arange(0,64)
    rows = group*4+r
    first = tile*64
    columns = first+c
    visible = tl.load(Visible+rows,mask=rows<ROWS,other=0)
    maximum = tl.max(visible,axis=0)
    physical = tl.load(Table+tile//3,mask=(first<WIDTH)&(first<maximum),other=0)
    key_base = physical.to(tl.int64)*192+tile%3*64
    raw_base = (group*RAW_TILES+tile)*1024
    key_live = (columns<WIDTH)&(columns<maximum)
    ks = tl.load(KS+key_base+c,mask=key_live,other=0).to(tl.float32)
    raw_value = tl.load(Raw+raw_base+r[:,None]*64+c[None,:],
                        mask=(rows[:,None]<ROWS)&key_live[None,:],other=0)
    score = raw_value*ks[None,:]
    live = (columns[None,:]<WIDTH)&(columns[None,:]<visible[:,None])
    tl.store(Out+rows[:,None]*WIDTH+columns[None,:],
             tl.where(live,score,-float("inf")),mask=(rows[:,None]<ROWS)&(columns[None,:]<WIDTH))

@triton.jit
def raw_finish_group4_grid(Raw, KS, Table, Visible, Out, ROWS:tl.constexpr,
                          WIDTH:tl.constexpr, RAW_TILES:tl.constexpr):
    raw_finish_group4_part(Raw,KS,Table,Visible,Out,tl.program_id(0),tl.program_id(1),
                          ROWS,WIDTH,RAW_TILES)

@triton.jit
def raw_finish_group4_capped(Raw, KS, Table, Visible, Out, ROWS:tl.constexpr,
                            WIDTH:tl.constexpr,RAW_TILES:tl.constexpr,CORES:tl.constexpr):
    for unit in range(tl.program_id(0),triton.cdiv(ROWS,4)*triton.cdiv(WIDTH,64),CORES):
        group = unit//triton.cdiv(WIDTH,64)
        tile = unit%triton.cdiv(WIDTH,64)
        raw_finish_group4_part(Raw,KS,Table,Visible,Out,group,tile,ROWS,WIDTH,RAW_TILES)
"""


def boundary_checks(module, device, torch):
    rows, width, tiles = 5, 257, 4
    raw = (torch.arange(2 * tiles * 16 * 64, dtype=torch.float32) / 64).reshape(
        2, tiles, 16, 64
    )
    ks = (torch.arange(3 * 192, dtype=torch.float32) / 4096 + 1 / 64).half()
    table = torch.tensor([2, 0], dtype=torch.int32)
    visible = torch.tensor([0, 1, 63, 192, 193], dtype=torch.int32)
    expected = torch.full((rows, width), -float("inf"), dtype=torch.float32)
    for row in range(rows):
        for col in range(int(visible[row])):
            physical = int(table[col // 192])
            expected[row, col] = (
                raw[row // 4, col // 64, row % 4, col % 64]
                * ks[physical * 192 + col % 192].float()
            )
    inputs = [x.to(device) for x in (raw, ks, table, visible)]
    checks = []
    for arm in ("grid", "capped"):
        out = torch.full(
            (rows, width), float("nan"), dtype=torch.float32, device=device
        )
        if arm == "grid":
            module.raw_finish_group4_grid[(2, (width + 63) // 64)](
                *inputs, out, ROWS=rows, WIDTH=width, RAW_TILES=tiles, num_warps=4
            )
        else:
            cores = min(20, 2 * ((width + 63) // 64))
            module.raw_finish_group4_capped[(cores,)](
                *inputs,
                out,
                ROWS=rows,
                WIDTH=width,
                RAW_TILES=tiles,
                CORES=cores,
                num_warps=4,
            )
        torch.npu.synchronize()
        actual = out.cpu()
        exact = bool(torch.equal(actual, expected))
        check = dict(
            arm=arm,
            rows=rows,
            width=width,
            raw_tiles=tiles,
            visible=visible.tolist(),
            exact=exact,
        )
        if not exact:
            check["mismatch_samples"] = [
                dict(
                    row=r,
                    column=c,
                    actual=str(actual[r, c].item()),
                    expected=str(expected[r, c].item()),
                )
                for r, c in (actual != expected).nonzero()[:8].tolist()
            ]
        checks.append(check)
        print("GROUP4_BOUNDARY_CHECK", json.dumps(check), flush=True)
        if not exact:
            raise AssertionError(f"Group4 boundary check failed: {arm}")
    return checks


def run_probe(
    args,
    name,
    values,
    pq,
    pw,
    pk,
    raw,
    output,
    reference,
    stages,
    kernels,
    record,
    report,
    compiler,
    capture_type,
    save_asm,
    save,
):
    import torch
    from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan
    from qsa_raw_score_chunks_910b import compare_scores

    path = report / "raw-finish-group4-kernels.py"
    path.write_text(
        k_pack_source()
        + "\n"
        + qw_pack_source()
        + "\n"
        + finish_source()
        + "\n"
        + source()
    )
    path.with_suffix(".py.txt").write_bytes(path.read_bytes())
    spec = importlib.util.spec_from_file_location("_qsa_finish_group4", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rows, width = output.shape
    groups = (rows + 3) // 4
    out_tiles = (width + 63) // 64
    tiles = pk.shape[0]
    key_cores = min(20, tiles * 8)
    old_cores = min(20, rows * out_tiles)
    cores = min(20, groups * out_tiles)
    expected_q, expected_w, expected_k, expected_out = [
        x.cpu() for x in (pq, pw, pk, output)
    ]
    targets = {
        arm: torch.full_like(output, float("nan"))
        for arm in ("group4_grid", "group4_capped")
    }
    probe = dict(
        status="running",
        compiler={},
        ir={},
        prechecks={},
        hardware={},
        grid_original=old_cores,
        grid_group4=[groups, out_tiles],
        grid_capped=cores,
        logical_units_original=rows * out_tiles,
        logical_units_group4=groups * out_tiles,
        fixed_sources={
            key: hashlib.sha256(fn().encode()).hexdigest()
            for key, fn in (
                ("qw", qw_pack_source),
                ("k", k_pack_source),
                ("finish", finish_source),
            )
        },
        scope="Four-query finish candidate with frozen fast packing; not formal acceptance",
    )
    record["finish_group4_probe"] = probe
    save()

    def fast_qw():
        return module.raw_qw_2d_grid[(groups,)](
            values["q"], values["qs"], values["weights"], pq, pw, ROWS=rows, num_warps=4
        )

    def fast_k():
        return module.raw_k_2d_capped[(key_cores,)](
            values["cache"],
            values["row_table"],
            pk,
            MAX_VISIBLE=record["max_visible"],
            TILES=tiles,
            CORES=key_cores,
            num_warps=4,
        )

    def original():
        return module.raw_finish_page64_capped[(old_cores,)](
            raw,
            values["ks"],
            values["row_table"],
            values["visible"],
            output,
            ROWS=rows,
            WIDTH=width,
            RAW_TILES=tiles,
            CORES=old_cores,
            num_warps=4,
        )

    def group4_grid():
        return module.raw_finish_group4_grid[(groups, out_tiles)](
            raw,
            values["ks"],
            values["row_table"],
            values["visible"],
            targets["group4_grid"],
            ROWS=rows,
            WIDTH=width,
            RAW_TILES=tiles,
            num_warps=4,
        )

    def group4_capped():
        return module.raw_finish_group4_capped[(cores,)](
            raw,
            values["ks"],
            values["row_table"],
            values["visible"],
            targets["group4_capped"],
            ROWS=rows,
            WIDTH=width,
            RAW_TILES=tiles,
            CORES=cores,
            num_warps=4,
        )

    finishes = dict(
        original=original, group4_grid=group4_grid, group4_capped=group4_capped
    )
    launches = dict(fast_qw=fast_qw, fast_k=fast_k, **finishes)
    selected = {}
    for arm, launch in launches.items():
        capture = capture_type(report, name + "-" + arm)
        try:
            with patch.object(compiler, "subprocess", capture):
                kernel = launch()
            torch.npu.synchronize()
            selected[arm] = kernel
            probe["compiler"][arm] = dict(
                metadata=str(kernel.metadata),
                asm=save_asm(kernel, report, name + "-" + arm),
            )
        finally:
            probe["compiler"].setdefault(arm, {})["compiler_calls"] = capture.calls
            save()
        if arm == "fast_qw":
            probe["qw_exact"] = bool(
                torch.equal(pq.cpu(), expected_q) and torch.equal(pw.cpu(), expected_w)
            )
            if not probe["qw_exact"]:
                raise AssertionError("Frozen QW pack changed")
        elif arm == "fast_k":
            probe["k_exact"] = bool(torch.equal(pk.cpu(), expected_k))
            if not probe["k_exact"]:
                raise AssertionError("Frozen K pack changed")
        else:
            actual = output.cpu() if arm == "original" else targets[arm].cpu()
            check = dict(
                exact_original=bool(torch.equal(actual, expected_out)),
                score=compare_scores(torch, actual, reference.cpu()),
            )
            probe["prechecks"][arm] = check
            print("GROUP4_CORRECTNESS", name, arm, json.dumps(check), flush=True)
            if not check["exact_original"] or not check["score"]["passed"]:
                save()
                raise AssertionError(f"Group4 score mismatch: {arm}")
        save()
    if name == args.score_cases[0]:
        probe["boundary_checks"] = boundary_checks(module, output.device, torch)
        save()
    probe["ir"] = {arm: summarize_vector_ir(selected[arm].asm) for arm in finishes}
    probe["scalar_access_removed"] = {
        arm: bool(ir.get("bcmlir"))
        and ir["bcmlir"]["discrete_access_markers"] == 0
        and ir["bcmlir"]["extracted_access_markers"] == 0
        for arm, ir in probe["ir"].items()
    }
    for arm, ir in probe["ir"].items():
        print("GROUP4_IR", name, arm, json.dumps(ir), flush=True)
    save()

    def original_complete():
        fast_qw()
        fast_k()
        stages["raw_cube"]()
        original()

    def grid_complete():
        fast_qw()
        fast_k()
        stages["raw_cube"]()
        group4_grid()

    def capped_complete():
        fast_qw()
        fast_k()
        stages["raw_cube"]()
        group4_capped()

    complete = dict(
        original_complete=original_complete,
        group4_grid_complete=grid_complete,
        group4_capped_complete=capped_complete,
    )
    arms = dict(current=stages["current"], **finishes, **complete)
    hosts = {arm: {selected[arm].metadata.kernel_name: 1} for arm in finishes}
    hosts["current"] = {kernels["current"].metadata.kernel_name: 1}
    common = collections.Counter(
        {
            selected["fast_qw"].metadata.kernel_name: 1,
            selected["fast_k"].metadata.kernel_name: 1,
            kernels["raw_cube"].metadata.kernel_name: 1,
        }
    )
    for arm in finishes:
        hosts[arm + "_complete"] = dict(common + collections.Counter(hosts[arm]))
    options = SimpleNamespace(
        rounds=args.rounds,
        kernel_warmup=args.warmup,
        kernel_active=args.active,
        report=str(report),
        case=name + "-finish-group4",
    )
    samples, _ = measure_prefix_plan(
        options,
        arms,
        lambda arm, _: arms[arm](),
        dict.fromkeys(arms),
        torch,
        record,
        expected_hosts=hosts,
    )
    med = {
        arm: statistics.median(r["kernel_us"] for r in rounds)
        for arm, rounds in samples.items()
    }
    probe.update(
        samples=samples,
        median_us=med,
        original_finish_over_candidate={
            arm: med["original"] / med[arm] for arm in targets
        },
        original_complete_over_candidate={
            arm: med["original_complete"] / med[arm + "_complete"] for arm in targets
        },
        current_over_candidate_complete={
            arm: med["current"] / med[arm + "_complete"] for arm in targets
        },
    )
    save()
    for arm in finishes:
        probe["hardware"][arm] = hardware_pass(
            args,
            name,
            arm,
            finishes[arm],
            selected[arm].metadata.kernel_name,
            torch,
            report,
        )
        save()
    probe["postchecks"] = {}
    for arm in targets:
        complete[arm + "_complete"]()
        torch.npu.synchronize()
        actual = targets[arm].cpu()
        check = dict(
            exact_original=bool(torch.equal(actual, expected_out)),
            score=compare_scores(torch, actual, reference.cpu()),
            qw_exact=bool(
                torch.equal(pq.cpu(), expected_q) and torch.equal(pw.cpu(), expected_w)
            ),
            k_exact=bool(torch.equal(pk.cpu(), expected_k)),
        )
        probe["postchecks"][arm] = check
        save()
        if (
            not all(check[k] for k in ("exact_original", "qw_exact", "k_exact"))
            or not check["score"]["passed"]
        ):
            raise AssertionError(f"Post-timing group4 mismatch: {arm}")
    probe["hardware_complete"] = all(
        entry[metric]["status"] == "available"
        for entry in probe["hardware"].values()
        for metric in ("PipeUtilization", "Memory")
    )
    probe["status"] = (
        "passed" if probe["hardware_complete"] else "counter_collection_incomplete"
    )
    save()
    print(
        "GROUP4_RESULT",
        json.dumps(
            dict(
                case=name,
                median_us=med,
                scalar_access_removed=probe["scalar_access_removed"],
                finish_speedup=probe["original_finish_over_candidate"],
                raw_pipeline_speedup=probe["original_complete_over_candidate"],
                current_over_candidate_complete=probe[
                    "current_over_candidate_complete"
                ],
                hardware_complete=probe["hardware_complete"],
            )
        ),
        flush=True,
    )
    if not probe["hardware_complete"]:
        raise RuntimeError(
            "Group4 results retained, but hardware counters are incomplete"
        )
