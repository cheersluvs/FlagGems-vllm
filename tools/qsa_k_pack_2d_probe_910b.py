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

"""Compare two-dimensional NZ packing with the original flat-index packer."""

import collections
import importlib.util
import json
import statistics
import sys
from types import SimpleNamespace
from unittest.mock import patch

from qsa_raw_vector_profile_910b import hardware_pass, summarize_vector_ir


def source():
    return """import triton
import triton.language as tl

@triton.jit
def raw_k_2d_part(K, Table, PK, part, MAX_VISIBLE:tl.constexpr):
    tile = part//8
    row_start = part%2*32
    dim_start = (part%8//2)*32
    r = tl.arange(0,32)
    c = tl.arange(0,32)
    physical = tl.load(Table+tile//3)
    token = physical.to(tl.int64)*192+tile%3*64+row_start
    global_row = tile*64+row_start+r
    value = tl.load(K+(token+r[:,None])*128+dim_start+c[None,:],
                    mask=global_row[:,None]<MAX_VISIBLE,other=0)
    tl.store(PK+part*1024+r[:,None]*32+c[None,:],value)

@triton.jit
def raw_k_2d_grid(K, Table, PK, MAX_VISIBLE:tl.constexpr):
    raw_k_2d_part(K,Table,PK,tl.program_id(0),MAX_VISIBLE)

@triton.jit
def raw_k_2d_capped(K, Table, PK, MAX_VISIBLE:tl.constexpr,TILES:tl.constexpr,CORES:tl.constexpr):
    for part in range(tl.program_id(0),TILES*8,CORES):
        raw_k_2d_part(K,Table,PK,part,MAX_VISIBLE)
"""


def run_probe(
    args,
    name,
    values,
    pk,
    output,
    reference,
    stages,
    kernels,
    score_with_packed,
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

    path = report / "raw-k-pack-2d-kernels.py"
    path.write_text(source())
    path.with_suffix(".py.txt").write_bytes(path.read_bytes())
    spec = importlib.util.spec_from_file_location("_qsa_k_pack_2d", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tiles = pk.shape[0]
    cores = min(20, tiles * 8)
    maximum = record["max_visible"]
    targets = {arm: torch.full_like(pk, -128) for arm in ("two_d_grid", "two_d_capped")}

    def two_d_grid():
        return module.raw_k_2d_grid[(tiles * 8,)](
            values["cache"],
            values["row_table"],
            targets["two_d_grid"],
            MAX_VISIBLE=maximum,
            num_warps=4,
        )

    def two_d_capped():
        return module.raw_k_2d_capped[(cores,)](
            values["cache"],
            values["row_table"],
            targets["two_d_capped"],
            MAX_VISIBLE=maximum,
            TILES=tiles,
            CORES=cores,
            num_warps=4,
        )

    packers = dict(
        original=stages["pack_k"], two_d_grid=two_d_grid, two_d_capped=two_d_capped
    )
    selected = {"original": kernels["pack_k"]}
    probe = dict(
        status="running",
        grid_original=tiles * 8,
        grid_capped=cores,
        tile_shape=[32, 32],
        values_per_program=1024,
        compiler={},
        ir={},
        prechecks={},
        hardware={},
        scope="Isolated K packing and complete raw-score comparison; production unchanged",
    )
    record["k_pack_probe"] = probe
    save()
    expected = pk.cpu()
    for arm, launch in list(packers.items())[1:]:
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
        actual = targets[arm].cpu()
        matched = bool(torch.equal(actual, expected))
        check = dict(packed_exact=matched, checked_elements=actual.numel())
        probe["prechecks"][arm] = check
        if not matched:
            check["mismatch_samples"] = [
                dict(
                    tile=t,
                    offset=i,
                    actual=int(actual[t, i]),
                    expected=int(expected[t, i]),
                )
                for t, i in (actual != expected).nonzero()[:12].tolist()
            ]
            save()
            raise AssertionError(f"Packed K mismatch: {arm}")
        score_with_packed(targets[arm])
        torch.npu.synchronize()
        check["score"] = compare_scores(torch, output.cpu(), reference.cpu())
        save()
        if not check["score"]["passed"]:
            raise AssertionError(f"Packed candidate changed score: {arm}")
        print("K_PACK_CORRECTNESS", name, arm, json.dumps(check), flush=True)

    probe["ir"] = {
        arm: summarize_vector_ir(kernel.asm) for arm, kernel in selected.items()
    }
    probe["scalar_access_removed"] = {
        arm: bool(evidence.get("bcmlir"))
        and evidence["bcmlir"]["discrete_access_markers"] == 0
        and evidence["bcmlir"]["extracted_access_markers"] == 0
        for arm, evidence in probe["ir"].items()
    }
    for arm, evidence in probe["ir"].items():
        print("K_PACK_IR", name, arm, json.dumps(evidence), flush=True)
    save()

    def original_complete():
        stages["pack_qw"]()
        stages["pack_k"]()
        stages["raw_cube"]()
        stages["finish"]()

    def grid_complete():
        stages["pack_qw"]()
        two_d_grid()
        score_with_packed(targets["two_d_grid"])

    def capped_complete():
        stages["pack_qw"]()
        two_d_capped()
        score_with_packed(targets["two_d_capped"])

    complete = dict(
        original_complete=original_complete,
        two_d_grid_complete=grid_complete,
        two_d_capped_complete=capped_complete,
    )
    arms = dict(current=stages["current"], **packers, **complete)
    hosts = {arm: {kernel.metadata.kernel_name: 1} for arm, kernel in selected.items()}
    hosts["current"] = {kernels["current"].metadata.kernel_name: 1}
    common = collections.Counter(
        {
            kernels[stage].metadata.kernel_name: 1
            for stage in ("pack_qw", "raw_cube", "finish")
        }
    )
    for arm in packers:
        hosts[arm + "_complete"] = dict(common + collections.Counter(hosts[arm]))
    options = SimpleNamespace(
        rounds=args.rounds,
        kernel_warmup=args.warmup,
        kernel_active=args.active,
        report=str(report),
        case=name + "-k-pack-2d",
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
        original_pack_over_candidate={
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
    for arm in packers:
        probe["hardware"][arm] = hardware_pass(
            args,
            name,
            arm,
            packers[arm],
            selected[arm].metadata.kernel_name,
            torch,
            report,
        )
        save()
    probe["postchecks"] = {}
    for arm in targets:
        complete[arm + "_complete"]()
        torch.npu.synchronize()
        check = dict(
            packed_exact=bool(torch.equal(targets[arm].cpu(), expected)),
            score=compare_scores(torch, output.cpu(), reference.cpu()),
        )
        probe["postchecks"][arm] = check
        if not check["packed_exact"] or not check["score"]["passed"]:
            save()
            raise AssertionError(f"Post-timing packing/score mismatch: {arm}")
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
        "K_PACK_RESULT",
        json.dumps(
            dict(
                case=name,
                median_us=med,
                scalar_access_removed=probe["scalar_access_removed"],
                pack_speedup=probe["original_pack_over_candidate"],
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
            "K packing results retained, but hardware counters are incomplete"
        )
