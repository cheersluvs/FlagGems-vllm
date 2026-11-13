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

"""Probe vector-friendly Q/W packing with fixed fast K and output stages."""

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
from qsa_raw_vector_profile_910b import hardware_pass, summarize_vector_ir


def source():
    return """import triton
import triton.language as tl

@triton.jit
def raw_qw_2d_group(Q, QS, W, PQ, PW, group, ROWS:tl.constexpr):
    heads = tl.arange(0,16)
    dims = tl.arange(0,32)
    flat = group*16+heads
    for block in tl.static_range(4):
        query = tl.load(Q+flat[:,None]*128+block*32+dims[None,:],
                        mask=flat[:,None]<ROWS*4,other=0)
        tl.store(PQ+group*2048+block*512+heads[:,None]*32+dims[None,:],query)
    qs = tl.load(QS+flat,mask=flat<ROWS*4,other=0).to(tl.float32)
    weights = tl.load(W+flat,mask=flat<ROWS*4,other=0).to(tl.float32)
    coefficient = (qs*weights*1024.0).to(tl.float16)
    out_rows = tl.arange(0,16)
    live = (out_rows[None,:]<4)&(heads[:,None]//4==out_rows[None,:])
    # Native LoadData2D transposes W.T from L1 into the second Cube's L0A.
    matrix = tl.where(live,coefficient[:,None],tl.full((),0.0,tl.float16))
    tl.store(PW+group*256+heads[:,None]*16+out_rows[None,:],matrix)

@triton.jit
def raw_qw_2d_grid(Q, QS, W, PQ, PW, ROWS:tl.constexpr):
    raw_qw_2d_group(Q,QS,W,PQ,PW,tl.program_id(0),ROWS)

@triton.jit
def raw_qw_2d_capped(Q, QS, W, PQ, PW, ROWS:tl.constexpr,CORES:tl.constexpr):
    for group in range(tl.program_id(0),triton.cdiv(ROWS,4),CORES):
        raw_qw_2d_group(Q,QS,W,PQ,PW,group,ROWS)
"""


def packed_pair_check(torch, q, w, expected_q, expected_w):
    observed_q, observed_w = q.cpu(), w.cpu()
    check = dict(
        q_exact=bool(torch.equal(observed_q, expected_q)),
        w_exact=bool(torch.equal(observed_w, expected_w)),
    )
    for label, observed, expected in (
        ("q", observed_q, expected_q),
        ("w", observed_w, expected_w),
    ):
        if not check[label + "_exact"]:
            check[label + "_mismatch_samples"] = [
                dict(
                    group=g,
                    offset=i,
                    actual=str(observed[g, i].item()),
                    expected=str(expected[g, i].item()),
                )
                for g, i in (observed != expected).nonzero()[:8].tolist()
            ]
    return check


def tail_checks(module, device, torch, pack_nz):
    checks = []
    for rows in (1, 5):
        q = (
            ((torch.arange(rows * 4 * 128, dtype=torch.int32) % 255) - 127)
            .to(torch.int8)
            .reshape(rows, 4, 128)
        )
        qs = (
            (torch.arange(rows * 4, dtype=torch.float32) / 4096 + 1 / 64)
            .half()
            .reshape(rows, 4)
        )
        weights = torch.tensor([0.0, 0.5, -1.0, 1.5], dtype=torch.float16).repeat(
            rows, 1
        )
        q_dev, qs_dev, w_dev = [x.to(device) for x in (q, qs, weights)]
        groups = (rows + 3) // 4
        expected_q = []
        expected_w = []
        for group in range(groups):
            query = torch.zeros((16, 128), dtype=torch.int8)
            matrix = torch.zeros((16, 16), dtype=torch.float16)
            for local in range(min(4, rows - group * 4)):
                row = group * 4 + local
                query[local * 4 : local * 4 + 4] = q[row]
                matrix[local, local * 4 : local * 4 + 4] = (
                    qs[row].float() * weights[row].float() * 1024
                ).half()
            expected_q.append(pack_nz(query, 32))
            expected_w.append(pack_nz(matrix.T.contiguous(), 16))
        expected_q = torch.stack(expected_q)
        expected_w = torch.stack(expected_w)
        for arm in ("grid", "capped"):
            pq = torch.full((groups, 2048), -128, dtype=torch.int8, device=device)
            pw = torch.full(
                (groups, 256), float("nan"), dtype=torch.float16, device=device
            )
            if arm == "grid":
                module.raw_qw_2d_grid[(groups,)](
                    q_dev, qs_dev, w_dev, pq, pw, ROWS=rows, num_warps=4
                )
            else:
                cores = min(20, groups)
                module.raw_qw_2d_capped[(cores,)](
                    q_dev, qs_dev, w_dev, pq, pw, ROWS=rows, CORES=cores, num_warps=4
                )
            torch.npu.synchronize()
            check = packed_pair_check(torch, pq, pw, expected_q, expected_w)
            passed = check["q_exact"] and check["w_exact"]
            checks.append(dict(rows=rows, arm=arm, passed=passed, **check))
            print("QW_TAIL_CHECK", json.dumps(checks[-1]), flush=True)
            if not passed:
                raise AssertionError(
                    f"QW partial-group check failed: rows={rows},arm={arm}"
                )
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
    cube_with_packed,
    record,
    report,
    compiler,
    capture_type,
    save_asm,
    pack_nz,
    save,
):
    import torch
    from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan
    from qsa_raw_score_chunks_910b import compare_scores

    path = report / "raw-qw-pack-2d-kernels.py"
    path.write_text(k_pack_source() + "\n" + finish_source() + "\n" + source())
    path.with_suffix(".py.txt").write_bytes(path.read_bytes())
    spec = importlib.util.spec_from_file_location("_qsa_qw_pack_2d", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rows, width = output.shape
    groups = (rows + 3) // 4
    cores = min(20, groups)
    tiles = pk.shape[0]
    key_cores = min(20, tiles * 8)
    finish_cores = min(20, rows * ((width + 63) // 64))
    expected_q, expected_w, expected_k, expected_out = [
        x.cpu() for x in (pq, pw, pk, output)
    ]
    targets = {
        arm: (torch.full_like(pq, -128), torch.full_like(pw, float("nan")))
        for arm in ("two_d_grid", "two_d_capped")
    }
    probe = dict(
        status="running",
        compiler={},
        ir={},
        prechecks={},
        hardware={},
        grid_original=groups,
        grid_capped=cores,
        fixed_k_source_sha256=hashlib.sha256(k_pack_source().encode()).hexdigest(),
        fixed_finish_source_sha256=hashlib.sha256(finish_source().encode()).hexdigest(),
        scope="Isolated Q/W packing with frozen fast K/page64 finish; not formal acceptance",
    )
    record["qw_pack_probe"] = probe
    save()

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

    def fast_finish():
        return module.raw_finish_page64_capped[(finish_cores,)](
            raw,
            values["ks"],
            values["row_table"],
            values["visible"],
            output,
            ROWS=rows,
            WIDTH=width,
            RAW_TILES=tiles,
            CORES=finish_cores,
            num_warps=4,
        )

    def two_d_grid():
        q, w = targets["two_d_grid"]
        return module.raw_qw_2d_grid[(groups,)](
            values["q"], values["qs"], values["weights"], q, w, ROWS=rows, num_warps=4
        )

    def two_d_capped():
        q, w = targets["two_d_capped"]
        return module.raw_qw_2d_capped[(cores,)](
            values["q"],
            values["qs"],
            values["weights"],
            q,
            w,
            ROWS=rows,
            CORES=cores,
            num_warps=4,
        )

    packers = dict(
        original=stages["pack_qw"], two_d_grid=two_d_grid, two_d_capped=two_d_capped
    )
    selected = {"original": kernels["pack_qw"]}
    launches = dict(
        fast_k=fast_k,
        fast_finish=fast_finish,
        two_d_grid=two_d_grid,
        two_d_capped=two_d_capped,
    )
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
        if arm == "fast_k":
            probe["fixed_k_exact"] = bool(torch.equal(pk.cpu(), expected_k))
            if not probe["fixed_k_exact"]:
                raise AssertionError("Frozen fast K pack changed")
        elif arm == "fast_finish":
            probe["fixed_finish_exact"] = bool(torch.equal(output.cpu(), expected_out))
            if not probe["fixed_finish_exact"]:
                raise AssertionError("Frozen fast finish changed")
        else:
            q, w = targets[arm]
            check = packed_pair_check(torch, q, w, expected_q, expected_w)
            probe["prechecks"][arm] = check
            if not check["q_exact"] or not check["w_exact"]:
                save()
                raise AssertionError(f"Packed Q/W mismatch: {arm}")
            cube_with_packed(q, w)
            fast_finish()
            torch.npu.synchronize()
            actual = output.cpu()
            check.update(
                exact_raw_score=bool(torch.equal(actual, expected_out)),
                score=compare_scores(torch, actual, reference.cpu()),
            )
            print("QW_PACK_CORRECTNESS", name, arm, json.dumps(check), flush=True)
            if not check["exact_raw_score"] or not check["score"]["passed"]:
                save()
                raise AssertionError(f"Packed candidate changed score: {arm}")
        save()

    if name == args.score_cases[0]:
        probe["partial_group_checks"] = tail_checks(
            module, output.device, torch, pack_nz
        )
        save()
    probe["ir"] = {arm: summarize_vector_ir(selected[arm].asm) for arm in packers}
    probe["scalar_access_removed"] = {
        arm: bool(ir.get("bcmlir"))
        and ir["bcmlir"]["discrete_access_markers"] == 0
        and ir["bcmlir"]["extracted_access_markers"] == 0
        for arm, ir in probe["ir"].items()
    }
    for arm, ir in probe["ir"].items():
        print("QW_PACK_IR", name, arm, json.dumps(ir), flush=True)
    save()

    def original_complete():
        stages["pack_qw"]()
        fast_k()
        stages["raw_cube"]()
        fast_finish()

    def grid_complete():
        two_d_grid()
        fast_k()
        cube_with_packed(*targets["two_d_grid"])
        fast_finish()

    def capped_complete():
        two_d_capped()
        fast_k()
        cube_with_packed(*targets["two_d_capped"])
        fast_finish()

    complete = dict(
        original_complete=original_complete,
        two_d_grid_complete=grid_complete,
        two_d_capped_complete=capped_complete,
    )
    arms = dict(current=stages["current"], **packers, **complete)
    hosts = {arm: {selected[arm].metadata.kernel_name: 1} for arm in packers}
    hosts["current"] = {kernels["current"].metadata.kernel_name: 1}
    common = collections.Counter(
        {
            selected["fast_k"].metadata.kernel_name: 1,
            selected["fast_finish"].metadata.kernel_name: 1,
            kernels["raw_cube"].metadata.kernel_name: 1,
        }
    )
    for arm in packers:
        hosts[arm + "_complete"] = dict(common + collections.Counter(hosts[arm]))
    options = SimpleNamespace(
        rounds=args.rounds,
        kernel_warmup=args.warmup,
        kernel_active=args.active,
        report=str(report),
        case=name + "-qw-pack-2d",
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
        q, w = targets[arm]
        actual = output.cpu()
        check = dict(
            **packed_pair_check(torch, q, w, expected_q, expected_w),
            k_exact=bool(torch.equal(pk.cpu(), expected_k)),
            exact_raw_score=bool(torch.equal(actual, expected_out)),
            score=compare_scores(torch, actual, reference.cpu()),
        )
        probe["postchecks"][arm] = check
        save()
        if (
            not all(
                check[k] for k in ("q_exact", "w_exact", "k_exact", "exact_raw_score")
            )
            or not check["score"]["passed"]
        ):
            raise AssertionError(f"Post-timing QW/score mismatch: {arm}")
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
        "QW_PACK_RESULT",
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
        raise RuntimeError("QW results retained, but hardware counters are incomplete")
