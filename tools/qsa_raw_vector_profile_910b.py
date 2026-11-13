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

"""Profile unchanged Vector stages against payload-matched copy controls."""

import gzip
import importlib.util
import json
import re
import statistics
import sys
from types import SimpleNamespace
from unittest.mock import patch


def copy_geometry(rows, width, tiles):
    if min(rows, width, tiles) < 1:
        raise ValueError("Nonempty copy geometry required")
    keys, scores = tiles * 8192, rows * width
    return dict(
        key_elements=keys,
        score_elements=scores,
        key_grid=(tiles * 8,),
        key_block=1024,
        score_grid=(rows, (width + 255) // 256),
        score_block=256,
        key_capped_grid=(min(20, (keys + 1023) // 1024),),
        score_capped_grid=(min(20, (scores + 1023) // 1024),),
        key_payload_bytes=keys,
        score_payload_bytes=scores * 4,
    )


def summarize_vector_ir(asm):
    result = {}
    for name in ("ttir", "ttadapter", "bcmlir"):
        text = asm.get(name)
        if not isinstance(text, str):
            continue
        result[name] = dict(
            scf_for=len(re.findall(r"\bscf\.for\b", text)),
            scalar_memref_load=len(re.findall(r"\bmemref\.load\b", text)),
            tensor_insert=len(re.findall(r"\btensor\.insert\s", text)),
            discrete_access_markers=text.count("DiscreteMemAccess"),
            extracted_access_markers=text.count("ExtractedLoadOrStore"),
            evidence_scope="Static pre-native IR counts, not executed instruction counts",
        )
    return result


def copy_source():
    return """import triton
import triton.language as tl

@triton.jit
def raw_copy_k_grid(Src, Dst):
    i = tl.program_id(0)*1024+tl.arange(0,1024)
    tl.store(Dst+i,tl.load(Src+i))

@triton.jit
def raw_copy_out_grid(Src, Dst, WIDTH:tl.constexpr):
    col = tl.program_id(1)*256+tl.arange(0,256)
    i = tl.program_id(0)*WIDTH+col
    value = tl.load(Src+i,mask=col<WIDTH,other=0)
    tl.store(Dst+i,value,mask=col<WIDTH)

@triton.jit
def raw_copy_k_capped(Src, Dst, N:tl.constexpr, CORES:tl.constexpr):
    for block in range(tl.program_id(0),triton.cdiv(N,1024),CORES):
        i = block*1024+tl.arange(0,1024)
        value = tl.load(Src+i,mask=i<N,other=0)
        tl.store(Dst+i,value,mask=i<N)

@triton.jit
def raw_copy_out_capped(Src, Dst, N:tl.constexpr, CORES:tl.constexpr):
    for block in range(tl.program_id(0),triton.cdiv(N,1024),CORES):
        i = block*1024+tl.arange(0,1024)
        value = tl.load(Src+i,mask=i<N,other=0)
        tl.store(Dst+i,value,mask=i<N)
"""


def hardware_pass(args, name, arm, function, kernel_name, torch, report):
    import torch_npu
    from profile_qsa_int8_score_910b import summarize_tables
    from qsa_timeline_910b import collect_hardware

    result = {}
    for metric in ("PipeUtilization", "Memory"):
        if not hasattr(torch_npu.profiler.AiCMetrics, metric):
            result[metric] = dict(status="unsupported_enum")
            continue
        options = SimpleNamespace(
            report=str(report),
            worker=f"{name}-{arm}-{metric}",
            aic_metric=metric,
            counter_iterations=args.counter_iterations,
            kernel_substring=kernel_name,
        )
        function()
        torch.npu.synchronize()
        capture = {}
        collect_hardware(options, function, torch, capture)
        entry = capture["score_hardware"]
        entry["capture_status"] = entry["status"]
        artifact = report / entry.get("artifact", "no-artifact")
        if artifact.is_file():
            with gzip.open(artifact, "rt") as stream:
                tables = json.load(stream)["tables"]
            parsed = summarize_tables(tables, metric, kernel_substring=kernel_name)
            entry["parsed"] = parsed
            entry["status"] = parsed["status"]
            if (
                entry["status"] == "available"
                and parsed["score_row_count"] != args.counter_iterations
            ):
                entry["status"] = "incomplete_counter_rows"
        else:
            entry["status"] = "no_counter_archive"
        result[metric] = entry
        print(
            "VECTOR_COUNTER",
            json.dumps(
                dict(
                    case=name,
                    arm=arm,
                    metric=metric,
                    status=entry["status"],
                    parsed=entry.get("parsed"),
                )
            ),
            flush=True,
        )
    return result


def profile_vectors(
    args,
    name,
    module_stages,
    kernels,
    pk,
    output,
    record,
    report,
    compiler,
    capture_type,
    save_asm,
    save,
):
    import torch
    from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan

    source = report / "raw-vector-copy-kernels.py"
    source.write_text(copy_source())
    source.with_suffix(".py.txt").write_bytes(source.read_bytes())
    spec = importlib.util.spec_from_file_location("_qsa_raw_vector_copies", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    geometry = copy_geometry(output.shape[0], output.shape[1], pk.shape[0])
    targets = {
        "copy_k_grid": torch.full_like(pk, -128),
        "copy_k_capped": torch.full_like(pk, -128),
        "copy_out_grid": torch.full_like(output, float("nan")),
        "copy_out_capped": torch.full_like(output, float("nan")),
    }

    def copy_k_grid():
        return module.raw_copy_k_grid[geometry["key_grid"]](
            pk, targets["copy_k_grid"], num_warps=4
        )

    def copy_out_grid():
        return module.raw_copy_out_grid[geometry["score_grid"]](
            output, targets["copy_out_grid"], WIDTH=output.shape[1], num_warps=4
        )

    def copy_k_capped():
        return module.raw_copy_k_capped[geometry["key_capped_grid"]](
            pk,
            targets["copy_k_capped"],
            N=geometry["key_elements"],
            CORES=geometry["key_capped_grid"][0],
            num_warps=4,
        )

    def copy_out_capped():
        return module.raw_copy_out_capped[geometry["score_capped_grid"]](
            output,
            targets["copy_out_capped"],
            N=geometry["score_elements"],
            CORES=geometry["score_capped_grid"][0],
            num_warps=4,
        )

    controls = dict(
        copy_k_grid=copy_k_grid,
        copy_out_grid=copy_out_grid,
        copy_k_capped=copy_k_capped,
        copy_out_capped=copy_out_capped,
    )
    arms = dict(
        pack_k=module_stages["pack_k"],
        copy_k_grid=copy_k_grid,
        copy_k_capped=copy_k_capped,
        finish=module_stages["finish"],
        copy_out_grid=copy_out_grid,
        copy_out_capped=copy_out_capped,
    )
    selected_kernels = {arm: kernels[arm] for arm in ("pack_k", "finish")}
    profile = dict(
        status="running",
        geometry=geometry,
        ir={},
        hardware={},
        scope="Payload-matched diagnostic copies, not arithmetic-equivalent scoring or formal acceptance",
    )
    record["vector_profile"] = profile
    save()
    for arm, launch in controls.items():
        capture = capture_type(report, name + "-" + arm)
        try:
            with patch.object(compiler, "subprocess", capture):
                kernel = launch()
            torch.npu.synchronize()
            selected_kernels[arm] = kernel
            record["compiler"][arm] = dict(
                metadata=str(kernel.metadata),
                asm=save_asm(kernel, report, name + "-" + arm),
            )
        finally:
            record["compiler"].setdefault(arm, {})["compiler_calls"] = capture.calls
            save()

    def check_copies():
        checks = {
            arm: bool(torch.equal(value, pk if arm.startswith("copy_k_") else output))
            for arm, value in targets.items()
        }
        if not all(checks.values()):
            raise AssertionError(f"Copy control mismatch: {checks}")
        return checks

    profile["precheck"] = check_copies()
    profile["ir"] = {
        arm: summarize_vector_ir(kernel.asm) for arm, kernel in selected_kernels.items()
    }
    for arm, evidence in profile["ir"].items():
        print("VECTOR_IR", name, arm, json.dumps(evidence), flush=True)
    save()
    hosts = {
        arm: {kernel.metadata.kernel_name: 1}
        for arm, kernel in selected_kernels.items()
    }
    options = SimpleNamespace(
        rounds=args.rounds,
        kernel_warmup=args.warmup,
        kernel_active=args.active,
        report=str(report),
        case=name + "-vector",
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
    medians = {
        arm: statistics.median(r["kernel_us"] for r in rounds)
        for arm, rounds in samples.items()
    }
    profile.update(
        samples=samples,
        median_us=medians,
        pack_over_same_grid_copy=medians["pack_k"] / medians["copy_k_grid"],
        finish_over_same_grid_copy=medians["finish"] / medians["copy_out_grid"],
        key_copy_grid_over_capped=medians["copy_k_grid"] / medians["copy_k_capped"],
        out_copy_grid_over_capped=medians["copy_out_grid"] / medians["copy_out_capped"],
    )
    profile["copy_logical_read_write_gbps"] = {
        arm: 2
        * geometry[
            "key_payload_bytes" if arm.startswith("copy_k_") else "score_payload_bytes"
        ]
        / value
        / 1000
        for arm, value in medians.items()
        if arm.startswith("copy_")
    }
    save()
    # Counter passes are deliberately restricted to the two original stages and
    # their matched-grid controls. The capped copies are timing-only controls.
    for arm in ("pack_k", "copy_k_grid", "finish", "copy_out_grid"):
        profile["hardware"][arm] = hardware_pass(
            args,
            name,
            arm,
            arms[arm],
            selected_kernels[arm].metadata.kernel_name,
            torch,
            report,
        )
        save()
    for launch in controls.values():
        launch()
    torch.npu.synchronize()
    profile["postcheck"] = check_copies()
    profile["hardware_complete"] = all(
        profile["hardware"][arm][metric]["status"] == "available"
        for arm in profile["hardware"]
        for metric in ("PipeUtilization", "Memory")
    )
    profile["status"] = (
        "passed" if profile["hardware_complete"] else "counter_collection_incomplete"
    )
    save()
    print(
        "VECTOR_PROFILE_RESULT",
        json.dumps(
            dict(
                case=name,
                median_us=medians,
                hardware_complete=profile["hardware_complete"],
                pack_over_same_grid_copy=profile["pack_over_same_grid_copy"],
                finish_over_same_grid_copy=profile["finish_over_same_grid_copy"],
                key_copy_grid_over_capped=profile["key_copy_grid_over_capped"],
                out_copy_grid_over_capped=profile["out_copy_grid_over_capped"],
            )
        ),
        flush=True,
    )
    if not profile["hardware_complete"]:
        raise RuntimeError(
            "Vector timing passed but numeric hardware counters are incomplete; see vector_profile.hardware"
        )
