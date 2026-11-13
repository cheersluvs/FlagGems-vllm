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

"""Locate query16 finish costs with live, full-size diagnostic output sinks."""

import collections
import importlib.util
import json
import re
import statistics
import sys
from types import SimpleNamespace
from unittest.mock import patch

from qsa_raw_vector_profile_910b import hardware_pass, summarize_vector_ir

MODES = dict(fill=0, raw=1, ks=2, sum=3, no_visible=4)
READS = dict(
    fill={"Visible"},
    raw={"Raw", "Visible"},
    ks={"KS", "Table", "Visible"},
    sum={"Raw", "KS", "Table", "Visible"},
    no_visible={"Raw", "KS", "Table"},
    full={"Raw", "KS", "Table", "Visible"},
    copy64={"Dense"},
    copy1024={"Dense"},
)


def source():
    text = """import triton
import triton.language as tl

@triton.jit
def raw_finish_prefix_part(Raw, KS, Table, Visible, Out, row, tile,
                           WIDTH:tl.constexpr,RAW_TILES:tl.constexpr,
                           MAX_VISIBLE:tl.constexpr,MODE:tl.constexpr):
    c = tl.arange(0,64)
    first = tile*64
    col = first+c
    if MODE == 4:
        visible = MAX_VISIBLE
    else:
        visible = tl.load(Visible+row)
    live = (col<WIDTH)&(col<visible)
    if MODE == 0:
        value = tl.full((64,),0.0,tl.float32)
    else:
        if MODE == 2 or MODE == 3 or MODE == 4:
            physical = tl.load(Table+tile//3,mask=(first<WIDTH)&(first<visible),other=0)
            key_base = physical.to(tl.int64)*192+tile%3*64
            ks = tl.load(KS+key_base+c,mask=live,other=0).to(tl.float32)
        if MODE == 1 or MODE == 3 or MODE == 4:
            raw_base = ((row//16)*RAW_TILES+tile)*1024+row%16*64
            value = tl.load(Raw+raw_base+c,mask=live,other=0)
        if MODE == 2:
            value = ks
        elif MODE == 3:
            value = value+ks
        elif MODE == 4:
            value = value*ks
    tl.store(Out+row*WIDTH+col,tl.where(live,value,-float("inf")),mask=col<WIDTH)
"""
    for name, mode in MODES.items():
        text += f"""
@triton.jit
def raw_finish_prefix_{name}(Raw, KS, Table, Visible, Out, ROWS:tl.constexpr,
                             WIDTH:tl.constexpr,RAW_TILES:tl.constexpr,
                             MAX_VISIBLE:tl.constexpr,CORES:tl.constexpr):
    for unit in range(tl.program_id(0),ROWS*triton.cdiv(WIDTH,64),CORES):
        row = unit//triton.cdiv(WIDTH,64)
        tile = unit%triton.cdiv(WIDTH,64)
        raw_finish_prefix_part(Raw,KS,Table,Visible,Out,row,tile,
                               WIDTH,RAW_TILES,MAX_VISIBLE,{mode})
"""
    text += """
@triton.jit
def raw_finish_copy64(Dense, Out, ROWS:tl.constexpr,WIDTH:tl.constexpr,CORES:tl.constexpr):
    c = tl.arange(0,64)
    for unit in range(tl.program_id(0),ROWS*triton.cdiv(WIDTH,64),CORES):
        row = unit//triton.cdiv(WIDTH,64)
        col = unit%triton.cdiv(WIDTH,64)*64+c
        value = tl.load(Dense+row*WIDTH+col,mask=col<WIDTH,other=0)
        tl.store(Out+row*WIDTH+col,value,mask=col<WIDTH)

@triton.jit
def raw_finish_copy1024(Dense, Out, N:tl.constexpr,CORES:tl.constexpr):
    c = tl.arange(0,1024)
    for unit in range(tl.program_id(0),triton.cdiv(N,1024),CORES):
        i = unit*1024+c
        value = tl.load(Dense+i,mask=i<N,other=0)
        tl.store(Out+i,value,mask=i<N)
"""
    return text


def ir_liveness(text, arm):
    """Trace GM roots through pre-native views to visible reads and writes."""
    header = next((line for line in text.splitlines() if "func.func @" in line), "")
    aliases = {}
    for segment in re.split(r"(?=%arg\d+:)", header):
        arg = re.match(r"(%arg\d+):", segment)
        name = re.search(r'loc\("(Raw|KS|Table|Visible|Out|Dense)"', segment)
        if arg and name:
            aliases[arg[1]] = name[1]
    reads, writes = set(), set()
    for line in text.splitlines():
        view = re.search(
            r"(%[\w.]+)\s*=\s*memref\.(?:reinterpret_cast|subview|cast)\s+(%[\w.]+)",
            line,
        )
        if view and view[2] in aliases:
            aliases[view[1]] = aliases[view[2]]
        load = re.search(r"\bmemref\.load\s+(%[\w.]+)", line)
        if load and load[1] in aliases:
            reads.add(aliases[load[1]])
        copy = re.search(r"\bmemref\.copy\s+(%[\w.]+),\s*(%[\w.]+)", line)
        if copy:
            if copy[1] in aliases:
                reads.add(aliases[copy[1]])
            if copy[2] in aliases:
                writes.add(aliases[copy[2]])
        copy = re.search(
            r"\blinalg\.copy\s+ins\((%[\w.]+)[^)]*\)\s+outs\((%[\w.]+)", line
        )
        if copy:
            if copy[1] in aliases:
                reads.add(aliases[copy[1]])
            if copy[2] in aliases:
                writes.add(aliases[copy[2]])
        store = re.search(r"\bmemref\.store\s+[^,]+,\s*(%[\w.]+)", line)
        destination = re.search(r"\bin writable\s+(%[\w.]+)", line)
        fill = re.search(r"linalg\.fill.*\bouts\((%[\w.]+)", line)
        for match in (store, destination, fill):
            if match and match[1] in aliases:
                writes.add(aliases[match[1]])
    expected = READS[arm]
    result = dict(
        reads=sorted(reads),
        writes=sorted(writes),
        expected_reads=sorted(expected),
        passed=reads == expected and "Out" in writes,
        scope="Pre-native memref dataflow, not ISA or measured traffic",
    )
    if not result["passed"]:
        raise RuntimeError(f"{arm} live-input/sink IR gate failed: {result}")
    return result


def signed_differences(medians):
    pairs = dict(
        raw_minus_fill=("raw", "fill"),
        ks_minus_fill=("ks", "fill"),
        sum_minus_raw=("sum", "raw"),
        full_minus_sum=("full", "sum"),
        full_minus_no_visible=("full", "no_visible"),
        full_minus_copy64=("full", "copy64"),
        copy64_minus_copy1024=("copy64", "copy1024"),
    )
    return {key: medians[a] - medians[b] for key, (a, b) in pairs.items()}


def prepare_controls(module, arm, values, cpu, torch):
    rows, width = arm["output"].shape
    raw = arm["raw"].cpu()
    tiles = raw.shape[1]
    maximum = min(width, int(cpu["visible"].max()))
    raw_dense = torch.zeros((rows, width), dtype=torch.float32)
    count = min(width, tiles * 64)
    for row in range(rows):
        raw_dense[row, :count] = raw[row // 16, :, row % 16, :].flatten()[:count]
    columns = torch.arange(width)
    live = columns[None, :] < cpu["visible"][:, None]
    global_live = columns[None, :] < maximum
    ks_dense = torch.zeros((width,), dtype=torch.float32)
    valid = columns[:maximum]
    table = cpu["row_table"][0].long()
    ks_dense[:maximum] = (
        cpu["ks"].reshape(-1)[table[valid // 192] * 192 + valid % 192].float()
    )
    neginf = torch.full_like(raw_dense, -float("inf"))
    dense = arm["output"].cpu()
    expected = dict(
        fill=torch.where(live, torch.zeros_like(raw_dense), neginf),
        raw=torch.where(live, raw_dense, neginf),
        ks=torch.where(live, ks_dense[None, :].expand_as(raw_dense), neginf),
        sum=torch.where(live, raw_dense + ks_dense[None, :], neginf),
        no_visible=torch.where(global_live, raw_dense * ks_dense[None, :], neginf),
        copy64=dense,
        copy1024=dense,
    )
    source_dense = dense.to(arm["output"].device)
    targets = {key: torch.full_like(arm["output"], float("nan")) for key in expected}
    cores = min(20, rows * ((width + 63) // 64))
    launches = {}
    for label in MODES:
        kernel = getattr(module, "raw_finish_prefix_" + label)

        def launch(kernel=kernel, label=label):
            return kernel[(cores,)](
                arm["raw"],
                values["ks"],
                values["row_table"],
                values["visible"],
                targets[label],
                ROWS=rows,
                WIDTH=width,
                RAW_TILES=tiles,
                MAX_VISIBLE=maximum,
                CORES=cores,
                num_warps=4,
            )

        launches[label] = launch

    def copy64():
        return module.raw_finish_copy64[(cores,)](
            source_dense,
            targets["copy64"],
            ROWS=rows,
            WIDTH=width,
            CORES=cores,
            num_warps=4,
        )

    copy_cores = min(20, (rows * width + 1023) // 1024)

    def copy1024():
        return module.raw_finish_copy1024[(copy_cores,)](
            source_dense,
            targets["copy1024"],
            N=rows * width,
            CORES=copy_cores,
            num_warps=4,
        )

    launches.update(copy64=copy64, copy1024=copy1024)
    return dict(
        launches=launches,
        targets=targets,
        expected=expected,
        dense=source_dense,
        geometry=dict(
            rows=rows,
            width=width,
            raw_tiles=tiles,
            max_visible=maximum,
            cores=cores,
            copy1024_cores=copy_cores,
        ),
    )


def check_controls(controls, torch, phase):
    checks = {}
    for label, launch in controls["launches"].items():
        launch()
        torch.npu.synchronize()
        actual = controls["targets"][label].cpu()
        expected = controls["expected"][label]
        exact = bool(torch.equal(actual, expected))
        item = dict(exact=exact)
        if not exact:
            item["mismatch_samples"] = [
                dict(
                    row=r,
                    column=c,
                    actual=str(actual[r, c].item()),
                    expected=str(expected[r, c].item()),
                )
                for r, c in (actual != expected).nonzero()[:8].tolist()
            ]
        checks[label] = item
        print("FINISH_PREFIX_CORRECTNESS", phase, label, json.dumps(item), flush=True)
        if not exact:
            raise AssertionError(f"{phase}/{label} diagnostic output mismatch: {item}")
    return checks


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

    path = report / "raw-finish-prefix-kernels.py"
    path.write_text(source())
    path.with_suffix(".py.txt").write_bytes(path.read_bytes())
    spec = importlib.util.spec_from_file_location("_qsa_finish_prefix", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    probe = dict(
        status="running",
        compiler={},
        liveness={},
        ir={},
        hardware={},
        scope="Conditional controls with full-size sinks, not an additive stage breakdown or formal benchmark",
    )
    record["finish_prefix_probe"] = probe
    save()
    controls = prepare_controls(module, arm, values, cpu, torch)
    probe["geometry"] = controls["geometry"]
    compiled = {}
    probe["liveness"]["full"] = ir_liveness(
        selected["finish"].asm.get("bcmlir", ""), "full"
    )
    probe["ir"]["full"] = summarize_vector_ir(selected["finish"].asm)
    for label, launch in controls["launches"].items():
        capture = capture_type(report, name + "-prefix-" + label)
        try:
            with patch.object(compiler, "subprocess", capture):
                kernel = launch()
            torch.npu.synchronize()
            compiled[label] = kernel
            probe["compiler"][label] = dict(
                metadata=str(kernel.metadata),
                asm=save_asm(kernel, report, name + "-prefix-" + label),
            )
            probe["liveness"][label] = ir_liveness(kernel.asm.get("bcmlir", ""), label)
            probe["ir"][label] = summarize_vector_ir(kernel.asm)
        finally:
            probe["compiler"].setdefault(label, {})["compiler_calls"] = capture.calls
            save()
    probe["prechecks"] = check_controls(controls, torch, name + "-before")
    save()
    if name == args.score_cases[0]:

        def finish_boundary(boundary_arm, boundary_values, boundary_cpu):
            small = prepare_controls(
                module, boundary_arm, boundary_values, boundary_cpu, torch
            )
            return check_controls(small, torch, "boundary-rows17-width257")

        probe["boundary"] = boundary_checks(
            native_module,
            16,
            arm["output"].device,
            torch,
            pack_nz,
            finish_callback=finish_boundary,
        )
        save()
    # Capture references before the current arm overwrites its device output.
    expected = arm["output"].cpu()
    raw_cpu = arm["raw"].cpu()
    raw_reference = torch.stack(
        [raw_cpu[r // 16, :, r % 16, :] for r in range(expected.shape[0])]
    )
    expected_k = arm["pk"].cpu()

    def complete():
        arm["qw"]()
        fast_k()
        arm["cube"]()
        arm["finish"]()

    launches = dict(
        current=current,
        qw=arm["qw"],
        k=fast_k,
        cube=arm["cube"],
        full=arm["finish"],
        **controls["launches"],
        complete=complete,
    )
    hosts = {
        key: {compiled[key].metadata.kernel_name: 1} for key in controls["launches"]
    }
    hosts.update(
        current={current_kernel.metadata.kernel_name: 1},
        k={key_kernel.metadata.kernel_name: 1},
        qw={selected["qw"].metadata.kernel_name: 1},
        cube={selected["cube"].metadata.kernel_name: 1},
        full={selected["finish"].metadata.kernel_name: 1},
    )
    counter = collections.Counter()
    for key in ("qw", "k", "cube", "full"):
        counter.update(hosts[key])
    hosts["complete"] = dict(counter)
    options = SimpleNamespace(
        rounds=args.rounds,
        kernel_warmup=args.warmup,
        kernel_active=args.active,
        report=str(report),
        case=name + "-finish-prefix",
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
        signed_differences_us=signed_differences(med),
        current_over_complete=med["current"] / med["complete"],
    )
    print(
        "FINISH_PREFIX_RESULT",
        json.dumps(
            dict(
                case=name,
                median_us=med,
                signed_differences_us=probe["signed_differences_us"],
                current_over_complete=probe["current_over_complete"],
            )
        ),
        flush=True,
    )
    save()
    hardware_labels = (
        ("fill", "raw", "ks", "full", "copy1024")
        if name == "int8_long100k"
        else ("full",)
    )
    probe["requested_hardware_arms"] = list(hardware_labels)
    for label in hardware_labels:
        kernel = selected["finish"] if label == "full" else compiled[label]
        probe["hardware"][label] = hardware_pass(
            args,
            name,
            "prefix-" + label,
            launches[label],
            kernel.metadata.kernel_name,
            torch,
            report,
        )
        save()
    probe["postchecks"] = check_controls(controls, torch, name + "-after")
    complete()
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
        if probe["full_postcheck"]["passed"]
        and probe["k_exact"]
        and probe["hardware_complete"]
        else "incomplete"
    )
    save()
    print(f"\nFinish diagnostics: {name}; status={probe['status']}", flush=True)
    print(f"{'arm':<26} {'median device us':>18}", flush=True)
    for label, duration in med.items():
        print(f"{label:<26} {duration:>18.3f}", flush=True)
    print("Conditional differences (us); not additive exclusive costs:", flush=True)
    for label, duration in probe["signed_differences_us"].items():
        print(f"{label:<28} {duration:>+12.3f}", flush=True)
    print(
        f"Current score / complete raw score: {probe['current_over_complete']:.4f}x; "
        "TopK excluded, not formal harness acceptance",
        flush=True,
    )
    if probe["status"] != "passed":
        raise RuntimeError("Finish timing retained; postcheck or counters incomplete")
