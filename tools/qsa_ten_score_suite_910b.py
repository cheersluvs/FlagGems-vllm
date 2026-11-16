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

"""Validate every INT8 workload row and time complete all-chunk scoring sweeps."""

import collections
import gc
import importlib.util
import json
import statistics
import sys
import traceback
from types import SimpleNamespace

from qsa_cube_query_pack_probe_910b import native_module
from qsa_cube_query_pack_probe_910b import source as query_source
from qsa_fused_ks_pack_probe_910b import source as logical_source
from qsa_int8_ten_workloads_910b import cases


def schedule(plan):
    starts = plan["starts"]
    rows = plan["q_shape"][0]
    chunk = plan["chunk_rows"]
    if (
        starts[0] != 0
        or starts[-1] != rows
        or chunk < 1
        or any(b <= a for a, b in zip(starts, starts[1:]))
    ):
        raise ValueError("Invalid workload request boundaries")
    result = []
    for first in range(0, rows, chunk):
        end = min(rows, first + chunk)
        pieces = []
        for request, (a, b) in enumerate(zip(starts, starts[1:])):
            lo, hi = max(a, first), min(b, end)
            if lo < hi:
                pieces.append(
                    dict(first=lo, count=hi - lo, offset=lo - first, request=request)
                )
        result.append(dict(first=first, count=end - first, pieces=pieces))
    return result


def runtime_source():
    # Keep all arithmetic and layouts; avoid a fresh compile for each growing
    # causal key count. Runtime scalars remain bounded by prepared metadata.
    text = logical_source() + "\n" + query_source(16)
    return (
        text.replace("RAW_TILES:tl.constexpr", "RAW_TILES")
        .replace("TILES:tl.constexpr", "TILES")
        .replace("MAX_VISIBLE:tl.constexpr", "MAX_VISIBLE")
    )


def cpu_oracle(torch, cpu, row, width, current, candidate):
    n = min(width, int(cpu["visible"][row]))
    mapping = cpu["row_table"][row, : ((n + 191) // 192)].long()
    keys = cpu["cache"][mapping, :, 0].reshape(-1, 128)[:n].int()
    ks = cpu["ks"][mapping, :, 0].reshape(-1)[:n].float()
    dot = cpu["q"][row].int() @ keys.T
    qs = cpu["qs"][row].float()
    w = cpu["weights"][row].float()
    exact = ((dot.float() * qs[:, None] * ks[None, :]).relu() * w[:, None]).sum(0)
    half = (
        (dot.float() / 1024).relu().half().float()
        * (qs * w * 1024).half().float()[:, None]
    ).sum(0) * ks
    result = dict(
        row=row,
        current_fp32=bool(torch.allclose(current[:n], exact, rtol=1e-3, atol=1e-3)),
        candidate_half=bool(torch.allclose(candidate[:n], half, rtol=1e-3, atol=1e-3)),
    )
    if not all(result[k] for k in ("current_fp32", "candidate_half")):
        raise AssertionError(str(result))
    return result


def run_case(args, plan, module, report, save_asm, record_launch_mode):
    import torch
    from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan
    from qsa_int8_shared_map_selection_910b_trial import PreparedSharedMapInt8Selection
    from qsa_int8_ten_workloads_910b import make_inputs
    from qsa_raw_score_chunks_910b import check_address_ir, compare_scores

    name = plan["name"]
    width = plan["logits_width"]
    cap = plan["chunk_rows"]
    manifest = schedule(plan)
    routing = getattr(args, "finish_routing_probe", False)
    if routing:
        from qsa_finish_routing_probe_910b import VARIANTS

        names = ("current", "logical1024") + VARIANTS
    else:
        names = ("current", "full1024", "logical1024")
    record = dict(
        case=name,
        status="running",
        kernel_timer={},
        schedule=manifest,
        source_mode="Runtime key bounds, frozen arithmetic/layout",
        all_rows=plan["q_shape"][0],
        global_chunks=len(manifest),
        request_pieces=sum(len(c["pieces"]) for c in manifest),
        scope="All input score chunks, no TopK/CANN/harness; candidate request splits fully charged",
    )
    path = report / (name + "-all-score.json")

    def save():
        path.write_text(json.dumps(record, indent=2, allow_nan=False))

    save()
    try:
        cpu = make_inputs(plan)
        if any(
            not bool(torch.isfinite(cpu[k]).all()) or bool((cpu[k] <= 0).any())
            for k in ("qs", "ks")
        ):
            raise ValueError("Finite positive scales required")
        if not bool(
            torch.isfinite(
                (cpu["qs"].float() * cpu["weights"].float() * 1024).half()
            ).all()
        ):
            raise ValueError("Half coefficient overflow")
        device = f"npu:{args.device}"
        v = {k: x.to(device) for k, x in cpu.items()}
        ref = torch.empty((cap, width), dtype=torch.float32, device=device)
        out = torch.empty_like(ref)
        adapter = PreparedSharedMapInt8Selection(
            v,
            torch.empty((plan["q_shape"][0], 512), dtype=torch.int32, device=device),
            ref,
            torch.empty((1024 * 1024,), dtype=torch.uint8, device=device),
            None,
            plan["starts"],
            key_group4=True,
            single_page64=True,
        )
        max_tiles = (width + 63) // 64
        max_groups = (cap + 15) // 16
        pq = torch.empty((max_groups, 8192), dtype=torch.int8, device=device)
        pw = torch.empty((max_groups, 1024), dtype=torch.float16, device=device)
        pk = torch.empty((max_tiles, 8192), dtype=torch.int8, device=device)
        logical = torch.empty((max_tiles * 64,), dtype=torch.float16, device=device)
        raw = torch.empty(
            (max_groups * max_tiles * 1024,), dtype=torch.float32, device=device
        )
        hosts = {a: collections.Counter() for a in names}
        artifacts = {}

        def inspect(kernel, label, rows, cube=False):
            key = (label, str(kernel.metadata))
            if key not in artifacts:
                tag = f"{name}-{label}-n{rows}-{len(artifacts)}"
                evidence = dict(
                    metadata=str(kernel.metadata), asm=save_asm(kernel, report, tag)
                )
                if routing and label.endswith("stage3"):
                    import re

                    evidence["finish_ir_counts"] = {
                        kind: {
                            op: len(
                                re.findall(
                                    r"\b" + re.escape(op) + r"\b",
                                    kernel.asm.get(kind, ""),
                                )
                            )
                            for op in (
                                "tt.load",
                                "tt.store",
                                "scf.if",
                                "vector.gather",
                                "memref.load",
                            )
                        }
                        for kind in ("ttir", "bcmlir")
                        if isinstance(kernel.asm.get(kind), str)
                    }
                    evidence["evidence_scope"] = (
                        "Static IR counts only; not executed DMA or instruction counts"
                    )
                if cube:
                    mode = record_launch_mode(kernel, require_cube=True)
                    if mode["mix_mode"] != "aic" or mode["bs_task_type"] != 20:
                        raise RuntimeError(str(mode))
                    evidence["address_ir"] = check_address_ir(
                        kernel.asm.get("bcmlir", "")
                    )
                    evidence["address_ir"]["byte_strides"].update(
                        query=8192, weight=2048
                    )
                artifacts[key] = evidence

        jobs = []
        for entry in manifest:
            pieces = []
            for piece in entry["pieces"]:
                first, n, offset = piece["first"], piece["count"], piece["offset"]
                maximum = min(width, int(cpu["visible"][first : first + n].max()))
                if maximum < 0:
                    raise ValueError("Negative visibility")
                if not torch.equal(
                    cpu["row_table"][first : first + n],
                    cpu["row_table"][first : first + 1].expand(n, -1),
                ):
                    raise ValueError("Candidate piece crosses page maps")
                table = cpu["row_table"][first, : ((maximum + 191) // 192)]
                if bool(((table < 0) | (table >= cpu["cache"].shape[0])).any()):
                    raise ValueError("Invalid active physical page")
                item = dict(piece, maximum=maximum, tiles=(maximum + 63) // 64)
                if maximum == 0:

                    def execute(mode, first=first, n=n, offset=offset):
                        return [adapter.score_into(first, n, out[offset : offset + n])]

                    item["execute"] = execute
                else:
                    groups = (n + 15) // 16
                    tiles = item["tiles"]
                    cores = min(20, groups * tiles)
                    kcores = min(20, tiles * 8)
                    q, qs, w = [v[k][first : first + n] for k in ("q", "qs", "weights")]
                    bt = v["row_table"][first : first + n]
                    vis = v["visible"][first : first + n]
                    packed_q = pq[:groups]
                    packed_w = pw[:groups]
                    packed_k = pk[:tiles]
                    lks = logical[: tiles * 64]
                    rv = raw[: groups * tiles * 1024].view(groups, tiles, 16, 64)
                    target = out[offset : offset + n]
                    cube_args = (packed_q, packed_k, packed_w, rv)
                    cube_options = dict(
                        GROUPS=groups, TILES=tiles, CORES=cores, QUERIES=16, num_warps=4
                    )
                    warm = module.raw_query16_cube.warmup(
                        *cube_args, grid=(cores,), **cube_options
                    )
                    inspect(warm, "cube", n, True)

                    def execute(
                        mode,
                        q=q,
                        qs=qs,
                        w=w,
                        bt=bt,
                        vis=vis,
                        packed_q=packed_q,
                        packed_w=packed_w,
                        packed_k=packed_k,
                        lks=lks,
                        rv=rv,
                        target=target,
                        n=n,
                        tiles=tiles,
                        groups=groups,
                        cores=cores,
                        kcores=kcores,
                        maximum=maximum,
                        cube_args=cube_args,
                        cube_options=cube_options,
                    ):
                        a = module.raw_query16_qw[(groups,)](
                            q,
                            qs,
                            w,
                            packed_q,
                            packed_w,
                            ROWS=n,
                            QUERIES=16,
                            num_warps=4,
                        )
                        if mode != "full1024":
                            b = module.raw_pack_k_ks[(kcores,)](
                                v["cache"],
                                v["ks"],
                                bt,
                                packed_k,
                                lks,
                                MAX_VISIBLE=maximum,
                                TILES=tiles,
                                CORES=kcores,
                                num_warps=4,
                            )
                        else:
                            b = module.raw_k_2d_capped[(kcores,)](
                                v["cache"],
                                bt,
                                packed_k,
                                MAX_VISIBLE=maximum,
                                TILES=tiles,
                                CORES=kcores,
                                num_warps=4,
                            )
                        c = module.raw_query16_cube[(cores,)](
                            *cube_args, **cube_options
                        )
                        finish = getattr(module, "raw_finish_" + mode)
                        extra = (lks,) if mode != "full1024" else ()
                        fcores = min(20, n * ((width + 1023) // 1024))
                        d = finish[(fcores,)](
                            rv,
                            v["ks"],
                            bt,
                            vis,
                            target,
                            *extra,
                            ROWS=n,
                            WIDTH=width,
                            RAW_TILES=tiles,
                            CORES=fcores,
                            num_warps=4,
                        )
                        return [a, b, c, d]

                    item.update(execute=execute, pk=packed_k, lks=lks)
                pieces.append(item)
            jobs.append((entry, pieces))
        record["zero_visible_pieces"] = sum(
            p["maximum"] == 0 for _, ps in jobs for p in ps
        )

        def validate(phase):
            checked = 0
            max_error = 0.0
            oracles = []
            for index, (entry, pieces) in enumerate(jobs):
                record["active_chunk"] = index
                first, count = entry["first"], entry["count"]
                k = adapter.score_into(first, count, ref[:count])
                torch.npu.synchronize()
                if phase == "before":
                    hosts["current"].update([k.metadata.kernel_name])
                    inspect(k, "current", count)
                reference = ref[:count].cpu()
                for item in pieces:
                    n, offset = item["count"], item["offset"]
                    full = item["execute"]("full1024")
                    torch.npu.synchronize()
                    full_cpu = out[offset : offset + n].cpu()
                    old_k = item["pk"].cpu() if item["maximum"] else None
                    logical_kernels = item["execute"]("logical1024")
                    torch.npu.synchronize()
                    actual = out[offset : offset + n].cpu()
                    if not torch.equal(full_cpu, actual):
                        raise AssertionError(
                            f"{phase}: full/logical mismatch at {item['first']}"
                        )
                    check = compare_scores(
                        torch, actual, reference[offset : offset + n]
                    )
                    if not check["passed"]:
                        raise AssertionError(
                            f"{phase}: current mismatch at {item['first']}: {check}"
                        )
                    max_error = max(max_error, check["max_abs_error"] or 0)
                    if item["maximum"]:
                        if not torch.equal(old_k, item["pk"].cpu()):
                            raise AssertionError("Fused K packing changed")
                        m = item["maximum"]
                        col = torch.arange(m)
                        mapping = cpu["row_table"][item["first"]].long()
                        expected = torch.zeros(
                            (item["tiles"] * 64,), dtype=torch.float16
                        )
                        expected[:m] = cpu["ks"].reshape(-1)[
                            mapping[col // 192] * 192 + col % 192
                        ]
                        if not torch.equal(expected, item["lks"].cpu()):
                            raise AssertionError("Logical KS mismatch")
                    for mode in names:
                        if mode in ("current", "full1024", "logical1024"):
                            continue
                        out[offset : offset + n].fill_(float("nan"))
                        kernels = item["execute"](mode)
                        torch.npu.synchronize()
                        if not torch.equal(actual, out[offset : offset + n].cpu()):
                            raise AssertionError(
                                f"{phase}: {mode} mismatch at {item['first']}"
                            )
                        if phase == "before":
                            hosts[mode].update(k.metadata.kernel_name for k in kernels)
                            for stage, kernel in enumerate(kernels):
                                inspect(kernel, f"{mode}-stage{stage}", n)
                    if phase == "before":
                        for mode, kernels in (
                            ("full1024", full),
                            ("logical1024", logical_kernels),
                        ):
                            if mode in hosts:
                                hosts[mode].update(
                                    k.metadata.kernel_name for k in kernels
                                )
                            for stage, kernel in enumerate(kernels):
                                inspect(kernel, f"{mode}-stage{stage}", n)
                        for row in (0, plan["q_shape"][0] - 1):
                            if item["first"] <= row < item["first"] + n:
                                local = row - item["first"]
                                oracles.append(
                                    cpu_oracle(
                                        torch,
                                        cpu,
                                        row,
                                        width,
                                        reference[offset + local],
                                        actual[local],
                                    )
                                )
                    checked += n
                if index % 16 == 0:
                    save()
                    print(
                        "TEN_SCORE_CHECK", name, phase, index + 1, len(jobs), flush=True
                    )
            if checked != plan["q_shape"][0]:
                raise AssertionError("Incomplete row coverage")
            return dict(
                passed=True,
                rows=checked,
                chunks=len(jobs),
                max_abs_error=max_error,
                cpu_oracles=oracles,
            )

        record["precheck"] = validate("before")
        record["compiled_artifacts"] = list(artifacts.values())
        save()

        def current():
            for entry, _ in jobs:
                adapter.score_into(
                    entry["first"], entry["count"], ref[: entry["count"]]
                )

        def candidate(mode):
            for _, pieces in jobs:
                for item in pieces:
                    item["execute"](mode)

        arms = {
            name: (current if name == "current" else lambda name=name: candidate(name))
            for name in names
        }
        options = SimpleNamespace(
            rounds=args.rounds,
            kernel_warmup=args.warmup,
            kernel_active=args.active,
            report=str(report),
            case=name + "-all-score",
        )
        samples, _ = measure_prefix_plan(
            options,
            arms,
            lambda key, _: arms[key](),
            dict.fromkeys(arms),
            torch,
            record,
            expected_hosts={k: dict(v) for k, v in hosts.items()},
        )
        med = {
            key: statistics.median(x["kernel_us"] for x in rounds)
            for key, rounds in samples.items()
        }
        record.update(
            samples=samples,
            median_us=med,
            current_over_logical=med["current"] / med["logical1024"],
        )
        if routing:
            from qsa_finish_routing_probe_910b import stage_totals

            record["logical_over_candidate"] = {
                name: med["logical1024"] / med[name] for name in VARIANTS
            }
            record["stage_mean_us"] = stage_totals(report, samples)
        else:
            record.update(
                full_over_logical=med["full1024"] / med["logical1024"],
                current_over_full=med["current"] / med["full1024"],
            )
        record["postcheck"] = validate("after")
        record["status"] = "passed"
        print(
            "TEN_SCORE_RESULT",
            json.dumps(
                {
                    k: record[k]
                    for k in (
                        "case",
                        "median_us",
                        "full_over_logical",
                        "current_over_full",
                        "current_over_logical",
                        "logical_over_candidate",
                        "stage_mean_us",
                    )
                    if k in record
                }
            ),
            flush=True,
        )
    except Exception as exc:
        record.update(
            status="failed", error=repr(exc), traceback=traceback.format_exc()
        )
        print("TEN_SCORE_FAILED", name, repr(exc), flush=True)
    finally:
        save()
    return record


def run_suite(
    args, report, result, compiler, capture_type, save_asm, record_launch_mode, save
):
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    build = dict(native_builds={})
    result["ten_score_native"] = build
    native = native_module(report, 16, build, save)
    path = report / "ten-score-runtime-kernels.py"
    routing = getattr(args, "finish_routing_probe", False)
    source = runtime_source()
    if routing:
        from qsa_finish_routing_probe_910b import CASES, VARIANTS, boundary
        from qsa_finish_routing_probe_910b import source as routing_source

        source += routing_source()
    path.write_text(source)
    path.with_suffix(".py.txt").write_bytes(path.read_bytes())
    spec = importlib.util.spec_from_file_location("_qsa_ten_score", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.chain_op = native.chain_op
    if routing:
        result["finish_routing_boundary"] = boundary(
            module, torch, f"npu:{args.device}"
        )
        save()
    entries = []
    plans = [p for p in cases() if not routing or p["name"] in CASES]
    for plan in plans:
        entry = run_case(args, plan, module, report, save_asm, record_launch_mode)
        entries.append(
            {
                k: entry.get(k)
                for k in (
                    "case",
                    "status",
                    "median_us",
                    "full_over_logical",
                    "current_over_full",
                    "current_over_logical",
                    "logical_over_candidate",
                    "stage_mean_us",
                    "error",
                )
            }
        )
        gc.collect()
        torch.npu.empty_cache()
        result["ten_score_suite"] = dict(
            cases=entries, formal_harness_acceptance=False, production_changed=False
        )
        save()
    valid = all(e["status"] == "passed" for e in entries) and len(entries) == len(plans)
    summary = result["ten_score_suite"]
    summary["all_passed"] = valid
    summary["mode"] = "finish_routing_four_cases" if routing else "all_ten_score"
    if valid and routing:
        summary["arithmetic_means"] = {
            name: statistics.mean(e["logical_over_candidate"][name] for e in entries)
            for name in VARIANTS
        }
    elif valid:
        summary["arithmetic_means"] = {
            k: statistics.mean(e[k] for e in entries)
            for k in ("full_over_logical", "current_over_full", "current_over_logical")
        }
    print("\nComplete score sweeps (us; TopK excluded)", flush=True)
    for e in entries:
        print(
            e["case"],
            e["status"],
            e.get("median_us"),
            "logical/candidates" if routing else "full/logical",
            e.get("logical_over_candidate") if routing else e.get("full_over_logical"),
            flush=True,
        )
    print("TEN_SCORE_SUITE", json.dumps(summary), flush=True)
    save()
    if not valid:
        raise RuntimeError("Score suite incomplete; failed cases are not averaged")
    result["outcome"] = (
        "passed_finish_routing_all_score_chunks"
        if routing
        else "passed_all_ten_workloads_all_score_chunks"
    )
