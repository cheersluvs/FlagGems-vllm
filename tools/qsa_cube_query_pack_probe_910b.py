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

"""Compare 4/8/16-query raw Cube groups including preparation and finalization."""

import collections
import importlib.util
import json
import statistics
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qsa_finish_page64_probe_910b import source as finish_source
from qsa_k_pack_2d_probe_910b import source as k_pack_source
from qsa_qw_pack_2d_probe_910b import source as qw_pack_source
from qsa_raw_vector_profile_910b import hardware_pass, summarize_vector_ir

_NATIVE = {}


def geometry(rows, tiles, queries):
    if rows < 1 or tiles < 1 or queries not in (4, 8, 16):
        raise ValueError("Positive rows/tiles and query packing4/8/16 required")
    heads = queries * 4
    groups = (rows + queries - 1) // queries
    return dict(
        queries=queries,
        heads=heads,
        groups=groups,
        tiles=tiles,
        q_bytes=heads * 128,
        w_bytes=heads * 16 * 2,
        raw_bytes=4096,
        cores=min(20, groups * tiles),
        native_calls=groups * tiles,
        raw_payload_bytes=groups * tiles * 4096,
        l0a_bytes=heads * 128,
        l0b_bytes=8192,
        l0c_bytes=heads * 64 * 4,
    )


def source(queries):
    geometry(1, 1, queries)
    # Preserve the verified 16x32 packing granule rather than creating a wider
    # Vector tile along with the larger Cube operand.
    pack = qw_pack_source()
    pack = pack.replace("def raw_qw_2d_group", "def raw_query_pack_group")
    pack = pack[: pack.index("@triton.jit\ndef raw_qw_2d_grid")]
    pack = pack.replace(
        "ROWS:tl.constexpr):", "ROWS:tl.constexpr, QUERIES:tl.constexpr):"
    )
    pack = pack.replace(
        "    heads = tl.arange(0,16)",
        "    heads = tl.arange(0,16)\n    HEADS:tl.constexpr = QUERIES*4",
    )
    body = pack[pack.index("    flat = group*16+heads") :]
    body = body.replace("group*16+heads", "group*HEADS+heads+head_block*16")
    body = body.replace(
        "group*2048+block*512+heads[:,None]*32",
        "group*(HEADS*128)+block*(HEADS*32)+(heads[:,None]+head_block*16)*32",
    )
    body = body.replace("out_rows[None,:]<4", "out_rows[None,:]<QUERIES")
    body = body.replace("heads[:,None]//4", "(heads[:,None]+head_block*16)//4")
    body = body.replace(
        "group*256+heads[:,None]*16",
        "group*(HEADS*16)+(heads[:,None]+head_block*16)*16",
    )
    pack = pack[: pack.index("    flat = group*16+heads")]
    pack += "    for head_block in tl.static_range(QUERIES//4):\n"
    pack += (
        "\n".join("    " + line if line else line for line in body.splitlines()) + "\n"
    )
    pack += """
@triton.jit
def raw_query_pack_qw(Q, QS, W, PQ, PW, ROWS:tl.constexpr, QUERIES:tl.constexpr):
    raw_query_pack_group(Q,QS,W,PQ,PW,tl.program_id(0),ROWS,QUERIES)

@triton.jit
def raw_query_pack_cube(PQ, PK, PW, Raw, GROUPS:tl.constexpr,TILES:tl.constexpr,
                        CORES:tl.constexpr,QUERIES:tl.constexpr):
    with al.scope(core_mode="cube"):
        q_base = tl.cast(PQ,tl.uint64)
        key_base = tl.cast(PK,tl.uint64)
        weight_base = tl.cast(PW,tl.uint64)
        output_base = tl.cast(Raw,tl.uint64)
        for unit in range(tl.program_id(0),GROUPS*TILES,CORES):
            group = unit//TILES
            tile = unit%TILES
            q = q_base+group.to(tl.uint64)*(QUERIES*4*128)
            key = key_base+tile.to(tl.uint64)*8192
            weight = weight_base+group.to(tl.uint64)*(QUERIES*4*16*2)
            output = output_base+unit.to(tl.uint64)*4096
            bits = tl.full((),0x3a800000,tl.int64)
            raw.call(chain_op,[q,key,weight,bits,output],output_indices=[])
"""
    finish = finish_source()
    # Change only the group address mapping; no 2D output sharing is enabled.
    finish = finish.replace("raw_finish_page64", "raw_query_pack_finish")
    finish = finish.replace(
        "RAW_TILES:tl.constexpr", "RAW_TILES:tl.constexpr,QUERIES:tl.constexpr"
    )
    finish = finish.replace("row//4", "row//QUERIES").replace("row%4", "row%QUERIES")
    finish = finish.replace("WIDTH,RAW_TILES)", "WIDTH,RAW_TILES,QUERIES)")
    imports = """import triton.language.extra.cann.extension as al
import triton.experimental.tle.language.raw as raw
"""
    text = k_pack_source() + "\n" + imports + pack + "\n" + finish
    return text.replace("raw_query_pack", f"raw_query{queries}")


def load_module(report, queries, chain_op):
    path = report / f"raw-cube-query{queries}-kernels.py"
    path.write_text(source(queries))
    path.with_suffix(".py.txt").write_bytes(path.read_bytes())
    spec = importlib.util.spec_from_file_location(f"_qsa_raw_query{queries}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.chain_op = chain_op
    return module


def native_module(report, queries, probe, save):
    from probe_qsa_raw_cube_chain_910b import build_bitcode
    from triton.experimental.tle.raw import dialect

    key = (str(report), queries)
    if key not in _NATIVE:
        result = dict(
            custom_symbols={
                name: name + f"_address_query{queries}"
                for name in ("qsa_raw_qk", "qsa_raw_chain")
            }
        )
        probe["native_builds"][str(queries)] = result
        save()
        try:
            bitcode = build_bitcode(
                report,
                result,
                Path(__file__).with_name("qsa_raw_cube_query_pack_910b.cpp"),
                stem=f"query{queries}",
                defines=(f"QSA_QUERY_PACK={queries}",),
            )
            structure = result["bitcode_structure"]
            if (
                not structure["exported_symbols"]
                or not structure["candidate_structure_present"]
            ):
                raise RuntimeError("Larger query chain failed the LLVM structure gate")

            def stub(*args, **kwargs):
                pass

            stub.__name__ = f"qsa_raw_chain_query{queries}"
            chain = dialect(
                name="cann",
                file=str(bitcode),
                format="bitcode",
                extern_func_name=result["custom_symbols"]["qsa_raw_chain"],
                pipeline={"core": "cube", "pipe": "PIPE_M"},
            )(stub)
            result["entry"] = "dialect-cann-bitcode"
            _NATIVE[key] = (load_module(report, queries, chain), result)
        finally:
            save()
    module, result = _NATIVE[key]
    probe["native_builds"][str(queries)] = result
    save()
    return module


def packed_oracle(torch, values, queries, pack_nz):
    rows = values["q"].shape[0]
    groups = (rows + queries - 1) // queries
    heads = queries * 4
    pq, pw = [], []
    for group in range(groups):
        q = torch.zeros((heads, 128), dtype=torch.int8)
        w = torch.zeros((16, heads), dtype=torch.float16)
        for local in range(min(queries, rows - group * queries)):
            row = group * queries + local
            q[local * 4 : local * 4 + 4] = values["q"][row]
            w[local, local * 4 : local * 4 + 4] = (
                values["qs"][row].float() * values["weights"][row].float() * 1024
            ).half()
        pq.append(pack_nz(q, 32))
        pw.append(pack_nz(w.T.contiguous(), 16))
    return torch.stack(pq), torch.stack(pw)


def make_arm(module, queries, values, pk, output, torch):
    rows, width = output.shape
    g = geometry(rows, pk.shape[0], queries)
    pq = torch.empty(
        (g["groups"], g["q_bytes"]), dtype=torch.int8, device=output.device
    )
    pw = torch.empty(
        (g["groups"], g["w_bytes"] // 2), dtype=torch.float16, device=output.device
    )
    raw = torch.full(
        (g["groups"], g["tiles"], 16, 64),
        float("nan"),
        dtype=torch.float32,
        device=output.device,
    )
    pack = getattr(module, f"raw_query{queries}_qw")
    cube = getattr(module, f"raw_query{queries}_cube")
    finish = getattr(module, f"raw_query{queries}_finish_capped")

    def qw_launch():
        return pack[(g["groups"],)](
            values["q"],
            values["qs"],
            values["weights"],
            pq,
            pw,
            ROWS=rows,
            QUERIES=queries,
            num_warps=4,
        )

    cube_options = dict(
        GROUPS=g["groups"],
        TILES=g["tiles"],
        CORES=g["cores"],
        QUERIES=queries,
        num_warps=4,
    )

    def cube_launch():
        return cube[(g["cores"],)](pq, pk, pw, raw, **cube_options)

    def cube_warmup():
        return cube.warmup(pq, pk, pw, raw, grid=(g["cores"],), **cube_options)

    cores = min(20, rows * ((width + 63) // 64))

    def finish_launch():
        return finish[(cores,)](
            raw,
            values["ks"],
            values["row_table"],
            values["visible"],
            output,
            ROWS=rows,
            WIDTH=width,
            RAW_TILES=g["tiles"],
            QUERIES=queries,
            CORES=cores,
            num_warps=4,
        )

    return dict(
        geometry=g,
        pq=pq,
        pw=pw,
        raw=raw,
        output=output,
        qw=qw_launch,
        cube=cube_launch,
        cube_warmup=cube_warmup,
        finish=finish_launch,
    )


def validate_arm(arm, cpu, expected, reference, pack_nz, torch, raw_reference):
    from qsa_raw_score_chunks_910b import compare_scores

    eq, ew = packed_oracle(torch, cpu, arm["geometry"]["queries"], pack_nz)
    raw = arm["raw"].cpu()
    queries = arm["geometry"]["queries"]
    rows = cpu["q"].shape[0]
    useful = torch.stack(
        [raw[row // queries, :, row % queries, :] for row in range(rows)]
    )
    unused_zero = all(
        bool((raw[group, :, min(queries, rows - group * queries) :, :] == 0).all())
        for group in range(raw.shape[0])
    )
    result = dict(
        q_exact=bool(torch.equal(arm["pq"].cpu(), eq)),
        w_exact=bool(torch.equal(arm["pw"].cpu(), ew)),
        raw_finite=bool(torch.isfinite(raw).all()),
        unused_raw_zero=unused_zero,
        useful_raw_comparison=compare_scores(
            torch, useful.reshape(rows, -1), raw_reference.reshape(rows, -1)
        ),
        raw4_comparison=compare_scores(torch, arm["output"].cpu(), expected),
        current_comparison=compare_scores(torch, arm["output"].cpu(), reference),
    )
    result["passed"] = (
        all(result[k] for k in ("q_exact", "w_exact", "raw_finite", "unused_raw_zero"))
        and result["useful_raw_comparison"]["passed"]
        and result["raw4_comparison"]["passed"]
        and result["current_comparison"]["passed"]
    )
    return result


def boundary_checks(module, queries, device, torch, pack_nz):
    from qsa_raw_score_chunks_910b import compare_scores

    rows, width, maximum = 17, 257, 65
    q = (
        ((torch.arange(rows * 4 * 128, dtype=torch.int32) % 17) - 8)
        .to(torch.int8)
        .reshape(rows, 4, 128)
    )
    cache = (
        ((torch.arange(3 * 192 * 128, dtype=torch.int32) % 19) - 9)
        .to(torch.int8)
        .reshape(3, 192, 1, 128)
    )
    qs = (
        torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4) / 4096 + 1 / 64
    ).half()
    weights = torch.tensor([0.0, 0.5, -1.0, 1.5], dtype=torch.float16).repeat(rows, 1)
    ks = (
        (torch.arange(3 * 192, dtype=torch.float32) / 4096 + 1 / 64)
        .half()
        .reshape(3, 192, 1)
    )
    table = torch.tensor([[2, 0]], dtype=torch.int32).repeat(rows, 1)
    visible = torch.tensor([0, 1, 63, 64, 65] * 4, dtype=torch.int32)[:rows]
    cpu = dict(
        q=q,
        cache=cache,
        qs=qs,
        weights=weights,
        ks=ks,
        row_table=table,
        visible=visible,
    )
    values = {k: v.to(device) for k, v in cpu.items()}
    keys = torch.zeros((128, 128), dtype=torch.int8)
    keys[:maximum] = cache[2, :maximum, 0]
    pk = torch.stack([pack_nz(keys[i : i + 64], 32) for i in (0, 64)]).to(device)
    expected = torch.full((rows, width), -float("inf"), dtype=torch.float32)
    raw_reference = []
    for row in range(rows):
        n = int(visible[row])
        dot = q[row].int() @ keys.int().T
        coeff = (qs[row].float() * weights[row].float() * 1024).half().float()
        raw_row = ((dot.float() / 1024).relu().half().float() * coeff[:, None]).sum(0)
        raw_reference.append(raw_row.reshape(2, 64))
        expected[row, :n] = raw_row[:n] * ks[2, :n, 0].float()
    target = torch.full((rows, width), float("nan"), dtype=torch.float32, device=device)
    arm = make_arm(module, queries, values, pk, target, torch)
    arm["qw"]()
    arm["cube"]()
    arm["finish"]()
    torch.npu.synchronize()
    check = validate_arm(
        arm, cpu, expected, expected, pack_nz, torch, torch.stack(raw_reference)
    )
    check.update(
        rows=rows,
        width=width,
        max_visible=maximum,
        queries=queries,
        score=compare_scores(torch, target.cpu(), expected),
    )
    print("QUERY_PACK_BOUNDARY", json.dumps(check), flush=True)
    if not check["passed"]:
        raise AssertionError(
            f"Query{queries} boundary/half-oracle check failed: {check}"
        )
    return check


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
    original_module,
    pack_nz,
):
    import torch
    from probe_qsa_raw_cube_chain_910b import record_launch_mode
    from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan
    from qsa_raw_score_chunks_910b import audit_binary, check_address_ir

    probe = dict(
        status="running",
        native_builds={},
        variants={},
        errors={},
        samples={},
        hardware={},
        scope="Raw score chunks only; no TopK/CANN/formal harness acceptance",
    )
    record["cube_query_pack_probe"] = probe
    save()
    if not hasattr(original_module, "chain_op"):
        raise NotImplementedError("Query packing requires dialect-cann-bitcode")
    expected = output.cpu()
    ref = reference.cpu()
    old_raw = raw.cpu()
    raw_reference = torch.stack(
        [old_raw[row // 4, :, row % 4, :] for row in range(output.shape[0])]
    )
    cpu = {key: value.cpu() for key, value in values.items()}
    expected_pk = pk.cpu()
    baseline = load_module(report, 4, original_module.chain_op)
    key_cores = min(20, pk.shape[0] * 8)

    def fast_k():
        return baseline.raw_k_2d_capped[(key_cores,)](
            values["cache"],
            values["row_table"],
            pk,
            MAX_VISIBLE=record["max_visible"],
            TILES=pk.shape[0],
            CORES=key_cores,
            num_warps=4,
        )

    key_kernel = fast_k()
    torch.npu.synchronize()
    if not torch.equal(pk.cpu(), expected_pk):
        raise AssertionError("Frozen K packing changed")
    models = {}
    selected = {}
    passed = []
    for queries in (4, 8, 16):
        label = f"query{queries}"
        variant = probe["variants"][label] = dict(status="running", compiler={})
        save()
        try:
            module = (
                baseline
                if queries == 4
                else native_module(report, queries, probe, save)
            )
            arm = make_arm(
                module,
                queries,
                values,
                pk,
                torch.full_like(output, float("nan")),
                torch,
            )
            variant["geometry"] = arm["geometry"]
            compiled = {}
            for stage in ("qw", "cube", "finish"):
                capture = capture_type(report, name + "-" + label + "-" + stage)
                try:
                    with patch.object(compiler, "subprocess", capture):
                        kernel = (
                            arm["cube_warmup"]() if stage == "cube" else arm[stage]()
                        )
                    torch.npu.synchronize()
                    compiled[stage] = kernel
                    variant["compiler"][stage] = dict(
                        asm=save_asm(kernel, report, name + "-" + label + "-" + stage),
                        metadata=str(kernel.metadata),
                    )
                finally:
                    variant["compiler"].setdefault(stage, {})[
                        "compiler_calls"
                    ] = capture.calls
                    save()
                if stage == "cube":
                    mode = record_launch_mode(kernel, require_cube=True)
                    variant["launch"] = mode
                    if mode["mix_mode"] != "aic" or mode["bs_task_type"] != 20:
                        raise RuntimeError(f"Invalid raw Cube mode: {mode}")
                    variant["address_ir"] = check_address_ir(
                        kernel.asm.get("bcmlir", "")
                    )
                    variant["address_ir"]["byte_strides"].update(
                        query=arm["geometry"]["q_bytes"],
                        weight=arm["geometry"]["w_bytes"],
                    )
                    arm["cube"]()
                    torch.npu.synchronize()
            check = validate_arm(arm, cpu, expected, ref, pack_nz, torch, raw_reference)
            variant["correctness"] = check
            print("QUERY_PACK_CORRECTNESS", name, label, json.dumps(check), flush=True)
            save()
            if not check["passed"]:
                raise AssertionError(f"{label} packing/score check failed")
            if queries == 4 and not torch.equal(arm["output"].cpu(), expected):
                raise AssertionError("Frozen query4 output changed")
            if name == args.score_cases[0]:
                variant["boundary"] = boundary_checks(
                    module, queries, output.device, torch, pack_nz
                )
            variant["audit"] = audit_binary(
                compiled["cube"], report, name + "-" + label + "-cube"
            )
            variant["vector_ir"] = {
                k: summarize_vector_ir(compiled[k].asm) for k in ("qw", "finish")
            }
            variant["status"] = "correctness_passed"
            models[label] = arm
            selected[label] = compiled
            passed.append(label)
            save()
        except Exception as exc:
            variant.update(status="failed", error=repr(exc))
            probe["errors"][label] = repr(exc)
            (report / f"{name}-{label}-failure.txt").write_text(traceback.format_exc())
            print("QUERY_PACK_REJECTED", name, label, repr(exc), flush=True)
            save()
            if queries == 4:
                raise
    arms = dict(current=stages["current"], pack_k=fast_k)
    hosts = dict(
        current={kernels["current"].metadata.kernel_name: 1},
        pack_k={key_kernel.metadata.kernel_name: 1},
    )
    for label in passed:
        arm = models[label]
        for stage in ("qw", "cube", "finish"):
            key = label + "_" + stage
            arms[key] = arm[stage]
            hosts[key] = {selected[label][stage].metadata.kernel_name: 1}

        def complete(arm=arm):
            arm["qw"]()
            fast_k()
            arm["cube"]()
            arm["finish"]()

        arms[label + "_complete"] = complete
        counter = collections.Counter(hosts["pack_k"])
        for stage in ("qw", "cube", "finish"):
            counter.update(hosts[label + "_" + stage])
        hosts[label + "_complete"] = dict(counter)
    options = SimpleNamespace(
        rounds=args.rounds,
        kernel_warmup=args.warmup,
        kernel_active=args.active,
        report=str(report),
        case=name + "-cube-query-pack",
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
        arm: statistics.median(x["kernel_us"] for x in rounds)
        for arm, rounds in samples.items()
    }
    probe.update(
        samples=samples,
        median_us=med,
        original_complete_over_candidate={
            label: med["query4_complete"] / med[label + "_complete"] for label in passed
        },
        current_over_candidate_complete={
            label: med["current"] / med[label + "_complete"] for label in passed
        },
    )
    print(
        "QUERY_PACK_RESULT",
        name,
        json.dumps(
            {
                k: probe[k]
                for k in (
                    "median_us",
                    "original_complete_over_candidate",
                    "current_over_candidate_complete",
                )
            }
        ),
        flush=True,
    )
    save()
    for label in passed:
        arm = models[label]
        probe["hardware"][label] = hardware_pass(
            args,
            name,
            label + "-cube",
            arm["cube"],
            selected[label]["cube"].metadata.kernel_name,
            torch,
            report,
        )
        save()
        arms[label + "_complete"]()
        torch.npu.synchronize()
        probe["variants"][label]["postcheck"] = validate_arm(
            arm, cpu, expected, ref, pack_nz, torch, raw_reference
        )
        if not probe["variants"][label]["postcheck"]["passed"] or not torch.equal(
            pk.cpu(), expected_pk
        ):
            raise AssertionError(f"Post-timing {label} mismatch")
        probe["variants"][label]["status"] = "passed"
        save()
    probe["hardware_complete"] = all(
        entry[metric]["status"] == "available"
        for entry in probe["hardware"].values()
        for metric in ("PipeUtilization", "Memory")
    )
    probe["status"] = (
        "passed" if not probe["errors"] and probe["hardware_complete"] else "incomplete"
    )
    save()
    if probe["status"] != "passed":
        raise RuntimeError(
            "Query-pack timing retained; one candidate or hardware pass incomplete"
        )
