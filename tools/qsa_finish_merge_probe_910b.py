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

"""Keep 64-key reads and merge their FP32 results into larger output stores."""

import collections
import importlib.util
import json
import re
import statistics
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

from qsa_finish_prefix_probe_910b import ir_liveness
from qsa_raw_vector_profile_910b import hardware_pass, summarize_vector_ir

WIDTHS = (256, 512, 1024)


def source():
    text = """import triton
import triton.language as tl

@triton.jit
def raw_finish_join1(Raw, KS, Table, row, tile, visible,
                     WIDTH:tl.constexpr,RAW_TILES:tl.constexpr):
    c = tl.arange(0,64)
    first = tile*64
    col = first+c
    segment_live = (first<WIDTH)&(first<visible)
    # Keep even the inactive pointer in bounds if scalar masked loads speculate.
    page = tl.where(segment_live,tile//3,0)
    physical = tl.load(Table+page,mask=segment_live,other=0)
    safe_tile = tl.where(segment_live,tile,0)
    key_base = physical.to(tl.int64)*192+safe_tile%3*64
    raw_base = ((row//16)*RAW_TILES+safe_tile)*1024+row%16*64
    live = (col<WIDTH)&(col<visible)
    ks = tl.load(KS+key_base+c,mask=live,other=0).to(tl.float32)
    value = tl.load(Raw+raw_base+c,mask=live,other=0)*ks
    return tl.where(live,value,-float("inf"))
"""
    for count in (2, 4, 8, 16):
        half = count // 2
        text += f"""
@triton.jit
def raw_finish_join{count}(Raw, KS, Table, row, tile, visible,
                           WIDTH:tl.constexpr,RAW_TILES:tl.constexpr):
    left = raw_finish_join{half}(Raw,KS,Table,row,tile,visible,WIDTH,RAW_TILES)
    right = raw_finish_join{half}(Raw,KS,Table,row,tile+{half},visible,WIDTH,RAW_TILES)
    return tl.cat(left,right,can_reorder=False)
"""
    for width in WIDTHS:
        text += f"""
@triton.jit
def raw_finish_merge{width}(Raw, KS, Table, Visible, Out, ROWS:tl.constexpr,
                            WIDTH:tl.constexpr,RAW_TILES:tl.constexpr,CORES:tl.constexpr):
    for unit in range(tl.program_id(0),ROWS*triton.cdiv(WIDTH,{width}),CORES):
        row = unit//triton.cdiv(WIDTH,{width})
        block = unit%triton.cdiv(WIDTH,{width})
        visible = tl.load(Visible+row)
        value = raw_finish_join{width//64}(Raw,KS,Table,row,block*{width//64},visible,WIDTH,RAW_TILES)
        col = block*{width}+tl.arange(0,{width})
        tl.store(Out+row*WIDTH+col,value,mask=col<WIDTH)
"""
    return text


def store_evidence(text):
    """Record pre-native output views/writes without inventing DMA counts."""
    header = next((line for line in text.splitlines() if "func.func @" in line), "")
    roots = set()
    for segment in re.split(r"(?=%arg\d+:)", header):
        arg = re.match(r"(%arg\d+):", segment)
        if arg and 'loc("Out"' in segment:
            roots.add(arg[1])
    views = []
    writes = []
    for line in text.splitlines():
        match = re.search(
            r"(%[\w.]+)\s*=\s*memref\.(?:reinterpret_cast|subview|cast)\s+(%[\w.]+)",
            line,
        )
        if match and match[2] in roots:
            roots.add(match[1])
            views.append(line.strip())
        if (
            "materialize_in_destination" in line
            or "memref.copy" in line
            or "memref.store" in line
        ):
            destination = re.search(r"in writable\s+(%[\w.]+)", line)
            copy = re.search(r"memref\.(?:copy|store)\s+[^,]+,\s*(%[\w.]+)", line)
            if any(m and m[1] in roots for m in (destination, copy)):
                writes.append(line.strip())
    return dict(
        output_views=views,
        output_write_sites=writes,
        allocation_sites=len(re.findall(r"\bmemref\.alloc\b", text)),
        tensor_insert_slice_sites=len(re.findall(r"\btensor\.insert_slice\b", text)),
        scope="Static pre-native view/write sites, not executed instructions or a DMA count",
    )


def prepare(module, arm, values, torch):
    rows, width = arm["output"].shape
    tiles = arm["raw"].shape[1]
    expected = arm["output"].cpu()
    targets = {
        f"merge{block}": torch.full_like(arm["output"], float("nan"))
        for block in WIDTHS
    }
    launches = {}
    for block in WIDTHS:
        label = f"merge{block}"
        kernel = getattr(module, "raw_finish_" + label)
        cores = min(20, rows * ((width + block - 1) // block))

        def launch(kernel=kernel, cores=cores, label=label):
            return kernel[(cores,)](
                arm["raw"],
                values["ks"],
                values["row_table"],
                values["visible"],
                targets[label],
                ROWS=rows,
                WIDTH=width,
                RAW_TILES=tiles,
                CORES=cores,
                num_warps=4,
            )

        launches[label] = launch
    return dict(targets=targets, launches=launches, expected=expected)


def check_output(torch, actual, expected):
    actual = actual.cpu()
    result = dict(exact=bool(torch.equal(actual, expected)))
    if not result["exact"]:
        result["mismatch_samples"] = [
            dict(
                row=r,
                column=c,
                actual=str(actual[r, c].item()),
                expected=str(expected[r, c].item()),
            )
            for r, c in (actual != expected).nonzero()[:8].tolist()
        ]
    return result


def run_probe(
    args,
    name,
    values,
    cpu,
    arm,
    selected,
    native_module,
    fast_k,
    key_kernel,
    current,
    current_kernel,
    record,
    report,
    compiler,
    capture_type,
    save_asm,
    save,
    pack_nz,
    reference_cpu,
):
    import torch
    from qsa_cube_query_pack_probe_910b import boundary_checks, validate_arm
    from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan

    path = report / "raw-finish-merge-kernels.py"
    path.write_text(source())
    path.with_suffix(".py.txt").write_bytes(path.read_bytes())
    spec = importlib.util.spec_from_file_location("_qsa_finish_merge", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    probe = dict(
        status="running",
        compiler={},
        ir={},
        liveness={},
        stores={},
        prechecks={},
        postchecks={},
        errors={},
        hardware={},
        scope="Output merging on query16 raw score, TopK excluded; not formal acceptance",
    )
    record["finish_merge_probe"] = probe
    save()
    trial = prepare(module, arm, values, torch)
    expected = trial["expected"]
    raw_cpu = arm["raw"].cpu()
    raw_reference = torch.stack(
        [raw_cpu[r // 16, :, r % 16, :] for r in range(expected.shape[0])]
    )
    expected_k = arm["pk"].cpu()
    compiled = {"original": selected["finish"]}
    valid = ["original"]
    for label, launch in trial["launches"].items():
        capture = capture_type(report, name + "-" + label)
        try:
            with patch.object(compiler, "subprocess", capture):
                kernel = launch()
            torch.npu.synchronize()
            compiled[label] = kernel
            probe["compiler"][label] = dict(
                metadata=str(kernel.metadata),
                asm=save_asm(kernel, report, name + "-" + label),
            )
            probe["liveness"][label] = ir_liveness(kernel.asm.get("bcmlir", ""), "full")
            check = check_output(torch, trial["targets"][label], expected)
            probe["prechecks"][label] = check
            print(
                "FINISH_MERGE_CORRECTNESS", name, label, json.dumps(check), flush=True
            )
            save()
            if not check["exact"]:
                raise AssertionError(f"{label} differs from original finish")
            if name == args.score_cases[0]:

                def finish_boundary(
                    boundary_arm, boundary_values, boundary_cpu, label=label
                ):
                    small = prepare(module, boundary_arm, boundary_values, torch)
                    small["launches"][label]()
                    torch.npu.synchronize()
                    check = check_output(
                        torch, small["targets"][label], small["expected"]
                    )
                    print("FINISH_MERGE_BOUNDARY", label, json.dumps(check), flush=True)
                    if not check["exact"]:
                        raise AssertionError(f"{label} boundary mismatch: {check}")
                    return dict(arm=label, **check)

                probe.setdefault("boundary", {})[label] = boundary_checks(
                    native_module,
                    16,
                    arm["output"].device,
                    torch,
                    pack_nz,
                    finish_callback=finish_boundary,
                )
            valid.append(label)
        except Exception as exc:
            probe["errors"][label] = repr(exc)
            (report / f"{name}-{label}-failure.txt").write_text(traceback.format_exc())
            print("FINISH_MERGE_REJECTED", name, label, repr(exc), flush=True)
        finally:
            probe["compiler"].setdefault(label, {})["compiler_calls"] = capture.calls
            save()
    for label in valid:
        probe["ir"][label] = summarize_vector_ir(compiled[label].asm)
        probe["stores"][label] = store_evidence(compiled[label].asm.get("bcmlir", ""))
        if label == "original":
            probe["liveness"][label] = ir_liveness(
                compiled[label].asm.get("bcmlir", ""), "full"
            )
        print(
            "FINISH_MERGE_IR",
            name,
            label,
            json.dumps(dict(ir=probe["ir"][label], stores=probe["stores"][label])),
            flush=True,
        )
    save()
    finishes = {
        "original": arm["finish"],
        **{label: trial["launches"][label] for label in valid if label != "original"},
    }
    complete = {}
    for label, finish in finishes.items():

        def launch(finish=finish):
            arm["qw"]()
            fast_k()
            arm["cube"]()
            finish()

        complete[label] = launch
    launches = dict(current=current, qw=arm["qw"], k=fast_k, cube=arm["cube"])
    hosts = dict(
        current={current_kernel.metadata.kernel_name: 1},
        qw={selected["qw"].metadata.kernel_name: 1},
        k={key_kernel.metadata.kernel_name: 1},
        cube={selected["cube"].metadata.kernel_name: 1},
    )
    common = collections.Counter()
    for label in ("qw", "k", "cube"):
        common.update(hosts[label])
    for label in valid:
        launches[label] = finishes[label]
        launches[label + "_complete"] = complete[label]
        host = {compiled[label].metadata.kernel_name: 1}
        hosts[label] = host
        hosts[label + "_complete"] = dict(common + collections.Counter(host))
    options = SimpleNamespace(
        rounds=args.rounds,
        kernel_warmup=args.warmup,
        kernel_active=args.active,
        report=str(report),
        case=name + "-finish-merge",
    )
    samples, _ = measure_prefix_plan(
        options,
        launches,
        lambda label, _: launches[label](),
        dict.fromkeys(launches),
        torch,
        record,
        expected_hosts=hosts,
    )
    med = {
        label: statistics.median(x["kernel_us"] for x in rounds)
        for label, rounds in samples.items()
    }
    probe.update(
        samples=samples,
        median_us=med,
        original_finish_over_candidate={
            label: med["original"] / med[label] for label in valid
        },
        original_complete_over_candidate={
            label: med["original_complete"] / med[label + "_complete"]
            for label in valid
        },
        current_over_complete={
            label: med["current"] / med[label + "_complete"] for label in valid
        },
    )
    print(
        "FINISH_MERGE_RESULT",
        name,
        json.dumps(
            {
                k: probe[k]
                for k in (
                    "median_us",
                    "original_finish_over_candidate",
                    "original_complete_over_candidate",
                    "current_over_complete",
                )
            }
        ),
        flush=True,
    )
    save()
    best = min(valid, key=lambda label: med[label])
    hardware_labels = (
        valid if name == "int8_long100k" else list(dict.fromkeys(["original", best]))
    )
    probe["requested_hardware_arms"] = hardware_labels
    for label in hardware_labels:
        probe["hardware"][label] = hardware_pass(
            args,
            name,
            label,
            finishes[label],
            compiled[label].metadata.kernel_name,
            torch,
            report,
        )
        save()
    for label in valid:
        complete[label]()
        torch.npu.synchronize()
        target = arm["output"] if label == "original" else trial["targets"][label]
        check = check_output(torch, target, expected)
        probe["postchecks"][label] = check
        if not check["exact"]:
            raise AssertionError(f"Post-timing {label} mismatch")
    complete["original"]()
    torch.npu.synchronize()
    probe["full_postcheck"] = validate_arm(
        arm, cpu, expected, reference_cpu, pack_nz, torch, raw_reference
    )
    probe["k_exact"] = bool(torch.equal(arm["pk"].cpu(), expected_k))
    probe["hardware_complete"] = all(
        entry[metric]["status"] == "available"
        for entry in probe["hardware"].values()
        for metric in ("PipeUtilization", "Memory")
    )
    probe["status"] = (
        "passed"
        if not probe["errors"]
        and probe["full_postcheck"]["passed"]
        and probe["k_exact"]
        and probe["hardware_complete"]
        else "incomplete"
    )
    save()
    print(f"\nFinish merge: {name}; status={probe['status']}", flush=True)
    print(
        f"{'arm':<14} {'finish us':>12} {'complete us':>14} {'old/new complete':>18} {'current/new':>14}",
        flush=True,
    )
    for label in valid:
        print(
            f"{label:<14} {med[label]:>12.3f} {med[label+'_complete']:>14.3f} "
            f"{probe['original_complete_over_candidate'][label]:>18.4f} {probe['current_over_complete'][label]:>14.4f}",
            flush=True,
        )
    print("TopK excluded; score chunks only, not formal harness acceptance", flush=True)
    if probe["status"] != "passed":
        raise RuntimeError(
            "Merge results retained; candidate rejection or incomplete postcheck/counters"
        )
