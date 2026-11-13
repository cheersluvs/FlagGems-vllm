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

"""Compare page-aligned raw score finalization with fixed fast key packing."""

import collections
import hashlib
import importlib.util
import json
import statistics
import sys
from types import SimpleNamespace
from unittest.mock import patch

from qsa_k_pack_2d_probe_910b import source as k_pack_source
from qsa_raw_vector_profile_910b import hardware_pass, summarize_vector_ir


def source():
    return """import triton
import triton.language as tl

@triton.jit
def raw_finish_page64_part(Raw, KS, Table, Visible, Out, row, tile,
                          WIDTH:tl.constexpr, RAW_TILES:tl.constexpr):
    c = tl.arange(0,64)
    first = tile*64
    col = first+c
    visible = tl.load(Visible+row)
    physical = tl.load(Table+tile//3,mask=(first<WIDTH)&(first<visible),other=0)
    key_base = physical.to(tl.int64)*192+tile%3*64
    raw_base = ((row//4)*RAW_TILES+tile)*1024+row%4*64
    live = (col<WIDTH)&(col<visible)
    ks = tl.load(KS+key_base+c,mask=live,other=0).to(tl.float32)
    value = tl.load(Raw+raw_base+c,mask=live,other=0)*ks
    tl.store(Out+row*WIDTH+col,tl.where(live,value,-float("inf")),mask=col<WIDTH)

@triton.jit
def raw_finish_page64_grid(Raw, KS, Table, Visible, Out,
                          WIDTH:tl.constexpr, RAW_TILES:tl.constexpr):
    raw_finish_page64_part(Raw,KS,Table,Visible,Out,tl.program_id(0),tl.program_id(1),
                          WIDTH,RAW_TILES)

@triton.jit
def raw_finish_page64_capped(Raw, KS, Table, Visible, Out, ROWS:tl.constexpr,
                            WIDTH:tl.constexpr, RAW_TILES:tl.constexpr,CORES:tl.constexpr):
    for unit in range(tl.program_id(0),ROWS*triton.cdiv(WIDTH,64),CORES):
        row = unit//triton.cdiv(WIDTH,64)
        tile = unit%triton.cdiv(WIDTH,64)
        raw_finish_page64_part(Raw,KS,Table,Visible,Out,row,tile,WIDTH,RAW_TILES)
"""


def run_probe(
    args,
    name,
    values,
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

    path = report / "raw-finish-page64-kernels.py"
    path.write_text(k_pack_source() + "\n" + source())
    path.with_suffix(".py.txt").write_bytes(path.read_bytes())
    spec = importlib.util.spec_from_file_location("_qsa_finish_page64", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rows, width = output.shape
    tiles = pk.shape[0]
    out_tiles = (width + 63) // 64
    cores = min(20, rows * out_tiles)
    key_cores = min(20, tiles * 8)
    expected_keys = pk.cpu()
    expected_output = output.cpu()
    targets = {
        arm: torch.full_like(output, float("nan"))
        for arm in ("page64_grid", "page64_capped")
    }

    def fast_pack():
        return module.raw_k_2d_capped[(key_cores,)](
            values["cache"],
            values["row_table"],
            pk,
            MAX_VISIBLE=record["max_visible"],
            TILES=tiles,
            CORES=key_cores,
            num_warps=4,
        )

    def page64_grid():
        return module.raw_finish_page64_grid[(rows, out_tiles)](
            raw,
            values["ks"],
            values["row_table"],
            values["visible"],
            targets["page64_grid"],
            WIDTH=width,
            RAW_TILES=tiles,
            num_warps=4,
        )

    def page64_capped():
        return module.raw_finish_page64_capped[(cores,)](
            raw,
            values["ks"],
            values["row_table"],
            values["visible"],
            targets["page64_capped"],
            ROWS=rows,
            WIDTH=width,
            RAW_TILES=tiles,
            CORES=cores,
            num_warps=4,
        )

    launchers = dict(
        fast_pack=fast_pack, page64_grid=page64_grid, page64_capped=page64_capped
    )
    finishes = dict(
        original=stages["finish"], page64_grid=page64_grid, page64_capped=page64_capped
    )
    probe = dict(
        status="running",
        compiler={},
        ir={},
        prechecks={},
        hardware={},
        fixed_k_source_sha256=hashlib.sha256(k_pack_source().encode()).hexdigest(),
        grid_original=[rows, (width + 255) // 256],
        grid_page64=[rows, out_tiles],
        grid_capped=cores,
        scope="Finish-only candidate on a frozen fast-K raw pipeline; production unchanged",
    )
    record["finish_probe"] = probe
    save()
    selected = {"original": kernels["finish"]}
    for arm, launch in launchers.items():
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
        if arm == "fast_pack":
            probe["fast_k_exact"] = bool(torch.equal(pk.cpu(), expected_keys))
            save()
            if not probe["fast_k_exact"]:
                raise AssertionError("Frozen fast K packing changed values")
            stages["raw_cube"]()
            torch.npu.synchronize()
            continue
        actual = targets[arm].cpu()
        check = dict(
            exact_original=bool(torch.equal(actual, expected_output)),
            score=compare_scores(torch, actual, reference.cpu()),
        )
        probe["prechecks"][arm] = check
        print("FINISH_CORRECTNESS", name, arm, json.dumps(check), flush=True)
        save()
        if not check["exact_original"] or not check["score"]["passed"]:
            raise AssertionError(f"Finish output mismatch: {arm}")

    probe["ir"] = {
        arm: summarize_vector_ir(kernel.asm) for arm, kernel in selected.items()
    }
    probe["scalar_access_removed"] = {
        arm: bool(ir.get("bcmlir"))
        and ir["bcmlir"]["discrete_access_markers"] == 0
        and ir["bcmlir"]["extracted_access_markers"] == 0
        for arm, ir in probe["ir"].items()
        if arm != "fast_pack"
    }
    for arm, ir in probe["ir"].items():
        print("FINISH_IR", name, arm, json.dumps(ir), flush=True)
    save()

    def original_complete():
        stages["pack_qw"]()
        fast_pack()
        stages["raw_cube"]()
        stages["finish"]()

    def grid_complete():
        stages["pack_qw"]()
        fast_pack()
        stages["raw_cube"]()
        page64_grid()

    def capped_complete():
        stages["pack_qw"]()
        fast_pack()
        stages["raw_cube"]()
        page64_capped()

    complete = dict(
        original_complete=original_complete,
        page64_grid_complete=grid_complete,
        page64_capped_complete=capped_complete,
    )
    arms = dict(current=stages["current"], **finishes, **complete)
    hosts = {arm: {selected[arm].metadata.kernel_name: 1} for arm in finishes}
    hosts["current"] = {kernels["current"].metadata.kernel_name: 1}
    common = collections.Counter(
        {
            kernels["pack_qw"].metadata.kernel_name: 1,
            selected["fast_pack"].metadata.kernel_name: 1,
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
        case=name + "-finish-page64",
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
            exact_original=bool(torch.equal(actual, expected_output)),
            score=compare_scores(torch, actual, reference.cpu()),
            fast_k_exact=bool(torch.equal(pk.cpu(), expected_keys)),
        )
        probe["postchecks"][arm] = check
        if (
            not check["exact_original"]
            or not check["score"]["passed"]
            or not check["fast_k_exact"]
        ):
            save()
            raise AssertionError(f"Post-timing finish/packing mismatch: {arm}")
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
        "FINISH_RESULT",
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
            "Finish results retained, but hardware counters are incomplete"
        )
