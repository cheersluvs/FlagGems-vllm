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

"""Audit raw device objects and charge real scoring-chunk packing costs."""

import collections
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

CASES = ("int8_compact", "int8_budget_exact", "int8_long100k")


def nz_offset(row, column, rows, c0):
    """Independent scalar check for the device packing address contract."""
    return (column // c0 * rows + row) * c0 + column % c0


def elf_identity(binary):
    if len(binary) < 64 or binary[:4] != b"\x7fELF":
        return dict(status="not_elf")
    if binary[4:6] != b"\x02\x01":
        return dict(status="unsupported_elf_class_or_endianness")
    return dict(
        status="elf64_little_endian",
        machine=hex(struct.unpack_from("<H", binary, 18)[0]),
        flags=hex(struct.unpack_from("<I", binary, 48)[0]),
    )


def decoded_instructions(text):
    """Exclude raw words, headers, and unknown encodings from ISA evidence."""
    instructions = []
    for line in text.splitlines():
        if not re.match(r"^\s*[0-9a-fA-F]+:\s", line):
            continue
        fields = line.split("\t")
        for field in fields[1:]:
            token = field.strip().split(" ", 1)[0]
            if re.fullmatch(r"[0-9a-fA-F ]+", field.strip()):
                continue
            if re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_.]*", token
            ) and token.lower() not in (
                "unknown",
                "invalid",
            ):
                instructions.append((token.lower(), line.strip()))
                break
    return instructions


def audit_binary(kernel, report, label):
    binary = bytes(kernel.asm["npubin"])
    archive = report / f"{label}-final.bin.gz"
    archive.write_bytes(gzip.compress(binary))
    result = dict(
        bytes=len(binary),
        sha256=hashlib.sha256(binary).hexdigest(),
        archive=archive.name,
        elf=elf_identity(binary),
        attempts=[],
        status="instruction_decode_unavailable",
        relu_encoding_verified=False,
        local_traffic_measured=False,
        overlap_measured=False,
    )
    root = Path(os.environ["ASCEND_HOME_PATH"])
    candidates = []
    if os.getenv("QSA_OBJDUMP"):
        candidates.append(Path(os.environ["QSA_OBJDUMP"]))
    for name in ("llvm-objdump", "cce-objdump"):
        for directory in (
            "bin",
            "compiler/ccec_compiler/bin",
            "ccec_compiler/bin",
            "tools/ccec_compiler/bin",
        ):
            candidates.append(root / directory / name)
        if found := shutil.which(name):
            candidates.append(Path(found))
    obj = report / f"{label}.o"
    obj.write_bytes(binary)
    try:
        for tool in dict.fromkeys(p.resolve() for p in candidates if p.is_file()):
            if not os.access(tool, os.X_OK):
                continue
            # Try the default target first. Only LLVM supports its triple/CPU options.
            options = [["-d", "-h", "-t"]]
            if "llvm" in tool.name:
                options.append(["-d", "--triple=cce", "--mcpu=dav-c220-cube"])
            for flags in options:
                command = [str(tool), *flags, str(obj)]
                entry = dict(command=command)
                try:
                    completed = subprocess.run(command, capture_output=True, timeout=15)
                    output = completed.stdout.decode(errors="replace")
                    stderr = completed.stderr.decode(errors="replace")
                    file = f"{label}-disassembly-{len(result['attempts']) + 1}.txt"
                    (report / file).write_text(
                        output[: 2**21] + "\nSTDERR:\n" + stderr[:8192]
                    )
                    instructions = decoded_instructions(output)
                    entry.update(
                        returncode=completed.returncode,
                        file=file,
                        truncated=len(output) > 2**21,
                    )
                    if completed.returncode == 0 and instructions:
                        result.update(
                            status="instruction_text_exported_needs_review",
                            file=file,
                            mnemonics=dict(
                                collections.Counter(i[0] for i in instructions)
                            ),
                            cube_fix_candidates=[
                                line
                                for op, line in instructions
                                if re.search(r"mad|fix|deq|relu", op)
                            ],
                        )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    entry["error"] = repr(exc)
                result["attempts"].append(entry)
                if result["status"] != "instruction_decode_unavailable":
                    break
            if result["status"] != "instruction_decode_unavailable":
                break
    finally:
        obj.unlink()
    print("FINAL_OBJECT_AUDIT", label, json.dumps(result), flush=True)
    return result


def kernel_source(entry):
    call = (
        "raw.call(chain_op, [q, key, weight, bits, output], output_indices=[])"
        if entry == "dialect-cann-bitcode"
        else 'tle.dsa.ascend.raw("qsa_raw_chain", q, key, weight, bits, output)'
    )
    imports = (
        "import triton.experimental.tle.language.raw as raw"
        if entry == "dialect-cann-bitcode"
        else "import triton.experimental.tle as tle"
    )
    return """import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
__IMPORTS__

@triton.jit
def raw_pack_qw(Q, QS, W, PQ, PW, ROWS:tl.constexpr):
    group = tl.program_id(0)
    i = tl.arange(0,2048)
    head = (i//32)%16
    dim = (i//512)*32+i%32
    flat = group*16+head
    q = tl.load(Q+flat*128+dim, mask=flat<ROWS*4, other=0)
    tl.store(PQ+group*2048+i,q)
    j = tl.arange(0,256)
    out_row = j//16
    in_head = j%16
    h = group*16+in_head
    qs = tl.load(QS+h,mask=h<ROWS*4,other=0).to(tl.float32)
    w = tl.load(W+h,mask=h<ROWS*4,other=0).to(tl.float32)
    coefficient = tl.where((out_row<4)&(in_head//4==out_row),qs*w*1024.0,0.0)
    tl.store(PW+group*256+j,coefficient.to(tl.float16))

@triton.jit
def raw_pack_k(K, Table, PK, MAX_VISIBLE:tl.constexpr):
    part = tl.program_id(0)
    tile = part//8
    i = part%8*1024+tl.arange(0,1024)
    col = tile*64+(i//32)%64
    dim = i//2048*32+i%32
    physical = tl.load(Table+tile//3)
    token = physical.to(tl.int64)*192+col%192
    value = tl.load(K+token*128+dim,mask=col<MAX_VISIBLE,other=0)
    tl.store(PK+tile*8192+i,value)

@triton.jit
def raw_score_tiles(PQ, PK, PW, Raw, GROUPS:tl.constexpr, TILES:tl.constexpr, CORES:tl.constexpr):
    with al.scope(core_mode="cube"):
        for unit in range(tl.program_id(0),GROUPS*TILES,CORES):
            group = unit//TILES
            tile = unit%TILES
            q = tl.cast(PQ+group*2048,tl.uint64)
            key = tl.cast(PK+tile*8192,tl.uint64)
            weight = tl.cast(PW+group*256,tl.uint64)
            output = tl.cast(Raw+unit*1024,tl.uint64)
            bits = tl.full((),0x3a800000,tl.int64)
            __CALL__

@triton.jit
def raw_finish(Raw, KS, Table, Visible, Out, ROWS:tl.constexpr, WIDTH:tl.constexpr,
               TILES:tl.constexpr, BLOCK:tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    visible = tl.load(Visible+row)
    live = (col<WIDTH)&(col<visible)
    physical = tl.load(Table+col//192,mask=live,other=0)
    ks = tl.load(KS+physical.to(tl.int64)*192+col%192,mask=live,other=0).to(tl.float32)
    offset = ((row//4)*TILES+col//64)*1024+(row%4)*64+col%64
    score = tl.load(Raw+offset,mask=live,other=0)*ks
    tl.store(Out+row*WIDTH+col,tl.where(live,score,-float("inf")),mask=col<WIDTH)
""".replace(
        "__IMPORTS__", imports
    ).replace(
        "__CALL__", call
    )


def load_module(report, entry, bindings):
    source = report / "raw-score-chunk-kernels.py"
    source.write_text(kernel_source(entry))
    # The uploader deliberately includes text artifacts only.
    source.with_suffix(".py.txt").write_bytes(source.read_bytes())
    spec = importlib.util.spec_from_file_location("_qsa_raw_score_chunks", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if bindings:
        module.chain_op = bindings["qsa_raw_chain"]
    return module


def compare_scores(torch, actual, expected):
    mask_ok = torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    errors = (actual[finite] - expected[finite]).abs()
    nonfinite_errors = int((~torch.isfinite(errors)).sum())
    return dict(
        passed=mask_ok and bool(torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)),
        masks_equal=mask_ok,
        max_abs_error=(
            None
            if nonfinite_errors
            else (float(errors.max()) if errors.numel() else 0.0)
        ),
        nonfinite_error_count=nonfinite_errors,
        mismatches=int((~torch.isclose(actual, expected, rtol=1e-3, atol=1e-3)).sum()),
        rtol=1e-3,
        atol=1e-3,
    )


def check_packing(torch, cpu, packed, max_visible, pack_nz):
    rows = cpu["q"].shape[0]
    groups = (rows + 3) // 4
    q, w, k = [value.cpu() for value in packed]
    for group in range(groups):
        query = torch.zeros((16, 128), dtype=torch.int8)
        live = min(4, rows - group * 4)
        query[: live * 4] = cpu["q"][group * 4 : group * 4 + live].reshape(-1, 128)
        if not torch.equal(q[group], pack_nz(query, 32)):
            raise AssertionError("Q NZ packing mismatch")
        weight = torch.zeros((16, 16), dtype=torch.float16)
        for row in range(live):
            source = group * 4 + row
            weight[row, row * 4 : row * 4 + 4] = (
                cpu["qs"][source].float() * cpu["weights"][source].float() * 1024
            ).half()
        if not torch.equal(w[group], pack_nz(weight, 16)):
            raise AssertionError("Weight NZ packing mismatch")
    tiles = (max_visible + 63) // 64
    mapping = cpu["row_table"][0]
    for tile in sorted({0, min(2, tiles - 1), min(3, tiles - 1), tiles - 1}):
        keys = torch.zeros((64, 128), dtype=torch.int8)
        for i in range(min(64, max_visible - tile * 64)):
            column = tile * 64 + i
            physical = int(mapping[column // 192])
            keys[i] = cpu["cache"][physical, column % 192, 0]
        if not torch.equal(k[tile], pack_nz(keys, 32)):
            raise AssertionError(f"K NZ packing mismatch: tile {tile}")
    return dict(
        status="passed",
        all_query_weight_groups=groups,
        key_tiles_checked=sorted({0, min(2, tiles - 1), min(3, tiles - 1), tiles - 1}),
    )


def check_cpu(torch, cpu, current, candidate):
    checks = []
    for row in sorted({0, 1, cpu["q"].shape[0] - 1}):
        count = int(cpu["visible"][row])
        mapping = cpu["row_table"][row].long()
        keys = cpu["cache"][mapping, :, 0].reshape(-1, 128)[:count].int()
        ks = cpu["ks"][mapping, :, 0].reshape(-1)[:count].float()
        dot = cpu["q"][row].int() @ keys.T
        qs = cpu["qs"][row].float()
        weights = cpu["weights"][row].float()
        expected = (
            (dot.float() * qs[:, None] * ks[None, :]).relu() * weights[:, None]
        ).sum(0)
        half_dot = (dot.float() / 1024).relu().half().float()
        coefficient = (qs * weights * 1024).half().float()
        oracle = (half_dot * coefficient[:, None]).sum(0) * ks
        baseline_ok = bool(
            torch.allclose(current[row, :count], expected, rtol=1e-3, atol=1e-3)
        )
        candidate_ok = bool(
            torch.allclose(candidate[row, :count], oracle, rtol=1e-3, atol=1e-3)
        )
        checks.append(
            dict(
                row=row,
                current_vs_fp32=baseline_ok,
                candidate_vs_half_oracle=candidate_ok,
            )
        )
    return dict(
        passed=all(
            c["current_vs_fp32"] and c["candidate_vs_half_oracle"] for c in checks
        ),
        rows=checks,
    )


def run_case(
    args,
    name,
    module,
    compiler,
    capture_type,
    report,
    save_asm,
    record_launch_mode,
    pack_nz,
    entry,
):
    import torch
    import triton
    from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan
    from qsa_int8_shared_map_selection_910b_trial import PreparedSharedMapInt8Selection

    from tools.qsa_int8_ten_workloads_910b import cases, make_inputs

    plan = next(c for c in cases() if c["name"] == name)
    started = time.perf_counter()
    cpu = make_inputs(plan)
    rows = min(plan["chunk_rows"], plan["query_lengths"][0])
    for key in ("q", "qs", "weights", "visible", "row_table"):
        cpu[key] = cpu[key][:rows].contiguous()
    if not torch.equal(
        cpu["row_table"], cpu["row_table"][:1].expand_as(cpu["row_table"])
    ):
        raise NotImplementedError(
            "Representative chunk must have one shared page table"
        )
    width = plan["logits_width"]
    max_visible = min(width, int(cpu["visible"].max()))
    if not 0 < max_visible <= cpu["row_table"].shape[1] * 192:
        raise ValueError("Invalid visible key bound")
    needed_pages = (max_visible + 191) // 192
    mapping = cpu["row_table"][0, :needed_pages]
    if bool(((mapping < 0) | (mapping >= cpu["cache"].shape[0])).any()):
        raise ValueError("Invalid physical page ID")
    if any(
        not bool(torch.isfinite(cpu[key]).all()) or bool((cpu[key] <= 0).any())
        for key in ("qs", "ks")
    ):
        raise NotImplementedError("Finite positive scales required")
    coefficient = (cpu["qs"].float() * cpu["weights"].float() * 1024).half()
    if not bool(torch.isfinite(coefficient).all()):
        raise NotImplementedError("FP16 coefficient overflow")
    values = {key: value.to(f"npu:{args.device}") for key, value in cpu.items()}
    rows, groups, tiles = rows, (rows + 3) // 4, (max_visible + 63) // 64
    output = torch.empty((rows, width), dtype=torch.float32, device=values["q"].device)
    reference = torch.empty_like(output)
    pq = torch.empty((groups, 2048), dtype=torch.int8, device=output.device)
    pw = torch.empty((groups, 256), dtype=torch.float16, device=output.device)
    pk = torch.empty((tiles, 8192), dtype=torch.int8, device=output.device)
    raw = torch.empty(
        (groups, tiles, 16, 64), dtype=torch.float32, device=output.device
    )
    adapter = PreparedSharedMapInt8Selection(
        values,
        torch.empty((rows, 512), dtype=torch.int32, device=output.device),
        reference,
        torch.empty((1024 * 1024,), dtype=torch.uint8, device=output.device),
        None,
        [0, rows],
        key_group4=True,
        single_page64=True,
    )
    record = dict(
        status="running",
        case=name,
        rows=rows,
        width=width,
        max_visible=max_visible,
        query_groups=groups,
        key_tiles=tiles,
        route=adapter.route_checks,
        fixture_and_preparation_ms=(time.perf_counter() - started) * 1000,
        extra_scratch_bytes=sum(
            x.numel() * x.element_size() for x in (pq, pw, pk, raw)
        ),
        kernel_timer={},
        compiler={},
        entry=entry,
        scope="Representative first chunk, actual integrated score dispatch; no TopK/CANN/formal acceptance",
        current_sources={
            filename: hashlib.sha256(
                Path(__file__).with_name(filename).read_bytes()
            ).hexdigest()
            for filename in (
                "qsa_int8_shared_map_selection_910b_trial.py",
                "qsa_int8_query_pack4_910b.py",
                "qsa_int8_selection_910b_trial.py",
            )
        },
    )
    path = report / f"{name}-raw-score.json"

    def save():
        path.write_text(json.dumps(record, indent=2, allow_nan=False))

    save()
    v = values
    cores = min(20, groups * tiles)

    def current():
        return adapter.score_into(0, rows, reference)

    def pack_qw():
        return module.raw_pack_qw[(groups,)](
            v["q"], v["qs"], v["weights"], pq, pw, ROWS=rows, num_warps=4
        )

    def pack_k():
        return module.raw_pack_k[(tiles * 8,)](
            v["cache"], v["row_table"], pk, MAX_VISIBLE=max_visible, num_warps=4
        )

    raw_args = (pq, pk, pw, raw)
    raw_options = dict(GROUPS=groups, TILES=tiles, CORES=cores, num_warps=4)

    def raw_cube():
        return module.raw_score_tiles[(cores,)](*raw_args, **raw_options)

    def finish():
        return module.raw_finish[(rows, triton.cdiv(width, 256))](
            raw,
            v["ks"],
            v["row_table"],
            v["visible"],
            output,
            ROWS=rows,
            WIDTH=width,
            TILES=tiles,
            BLOCK=256,
            num_warps=4,
        )

    stages = dict(
        current=current,
        pack_qw=pack_qw,
        pack_k=pack_k,
        raw_cube=raw_cube,
        finish=finish,
    )
    kernels = {}
    try:
        for arm, launch in stages.items():
            capture = capture_type(report, f"{name}-{arm}")
            try:
                with patch.object(compiler, "subprocess", capture):
                    kernel = (
                        module.raw_score_tiles.warmup(
                            *raw_args, grid=(cores,), **raw_options
                        )
                        if arm == "raw_cube"
                        else launch()
                    )
                torch.npu.synchronize()
                kernels[arm] = kernel
                record["compiler"][arm] = dict(
                    launch=record_launch_mode(kernel, require_cube=arm == "raw_cube"),
                    asm=save_asm(kernel, report, f"{name}-{arm}"),
                )
            finally:
                record["compiler"].setdefault(arm, {})["compiler_calls"] = capture.calls
                save()
            if arm == "raw_cube":
                metadata = record["compiler"][arm]["launch"]
                if metadata["mix_mode"] != "aic" or metadata["bs_task_type"] != 20:
                    raise RuntimeError(
                        f"Unexpected raw tile-loop launch metadata: {metadata}"
                    )
                raw_cube()
                torch.npu.synchronize()
        record["native_audit"] = audit_binary(
            kernels["raw_cube"], report, name + "-raw"
        )
        record["packing"] = check_packing(
            torch, cpu, (pq, pw, pk), max_visible, pack_nz
        )
        record["correctness"] = compare_scores(torch, output.cpu(), reference.cpu())
        record["cpu_oracles"] = check_cpu(torch, cpu, reference.cpu(), output.cpu())
        print(
            "CHUNK_CORRECTNESS",
            name,
            json.dumps(record["correctness"]),
            json.dumps(record["cpu_oracles"]),
            flush=True,
        )
        save()
        if not record["correctness"]["passed"] or not record["cpu_oracles"]["passed"]:
            raise AssertionError(
                "Raw score chunk differs from current/oracle; candidate rejected without timing"
            )

        def prepared():
            raw_cube()
            finish()

        def complete():
            pack_qw()
            pack_k()
            prepared()

        launchers = {**stages, "prepared": prepared, "complete": complete}
        hosts = {
            arm: {kernel.metadata.kernel_name: 1} for arm, kernel in kernels.items()
        }
        hosts["prepared"] = dict(
            collections.Counter(hosts["raw_cube"])
            + collections.Counter(hosts["finish"])
        )
        hosts["complete"] = dict(
            collections.Counter(hosts["prepared"])
            + collections.Counter(hosts["pack_qw"])
            + collections.Counter(hosts["pack_k"])
        )
        params = SimpleNamespace(
            rounds=args.rounds,
            kernel_warmup=args.warmup,
            kernel_active=args.active,
            report=str(report),
            case=name + "-raw",
        )
        samples, _ = measure_prefix_plan(
            params,
            launchers,
            lambda arm, _: launchers[arm](),
            dict.fromkeys(launchers),
            torch,
            record,
            expected_hosts=hosts,
        )
        med = {
            arm: statistics.median(row["kernel_us"] for row in rounds)
            for arm, rounds in samples.items()
        }
        record.update(
            samples=samples,
            median_us=med,
            current_over_complete=med["current"] / med["complete"],
            current_over_prepared=med["current"] / med["prepared"],
        )
        complete()
        current()
        torch.npu.synchronize()
        record["postcheck"] = compare_scores(torch, output.cpu(), reference.cpu())
        if not record["postcheck"]["passed"]:
            raise AssertionError("Post-timing correctness failed")
        record["status"] = "passed"
        print(
            "CHUNK_RESULT",
            json.dumps(
                dict(
                    case=name,
                    median_us=med,
                    current_over_complete=record["current_over_complete"],
                    current_over_prepared=record["current_over_prepared"],
                )
            ),
            flush=True,
        )
    except Exception as exc:
        record.update(
            status="failed", error=repr(exc), traceback=traceback.format_exc()
        )
        print("CHUNK_FAILED", name, record["error"], flush=True)
    finally:
        save()
    return record


def run_chunks(
    args,
    report,
    result,
    bindings,
    compiler,
    capture_type,
    save_asm,
    record_launch_mode,
    pack_nz,
):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    module = load_module(report, result["entry"], bindings)
    records = []
    for name in CASES:
        try:
            record = run_case(
                args,
                name,
                module,
                compiler,
                capture_type,
                report,
                save_asm,
                record_launch_mode,
                pack_nz,
                result["entry"],
            )
        except Exception as exc:
            record = dict(
                case=name,
                status="failed",
                error=repr(exc),
                traceback=traceback.format_exc(),
            )
            (report / f"{name}-raw-score.json").write_text(json.dumps(record, indent=2))
            print("CHUNK_FAILED", name, repr(exc), flush=True)
        records.append(record)
    result["score_chunks"] = dict(
        cases=[
            dict(
                case=r["case"],
                status=r["status"],
                current_over_complete=r.get("current_over_complete"),
                current_over_prepared=r.get("current_over_prepared"),
                error=r.get("error"),
            )
            for r in records
        ],
        all_passed=all(r["status"] == "passed" for r in records),
        formal_harness_acceptance=False,
        production_changed=False,
    )
    if not result["score_chunks"]["all_passed"]:
        raise RuntimeError(
            "At least one raw score chunk rejected; see per-case diagnostics"
        )
    result["outcome"] = "passed_micro_and_representative_score_chunks"
