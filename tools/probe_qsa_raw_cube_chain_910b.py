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

"""Check custom-op ABI, two-Cube local handoff, correctness and kernel timing."""

import argparse
import gzip
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

M, K, N = 16, 128, 64


class MissingPrerequisite(RuntimeError):
    pass


def write_json(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def command(args, report, label, timeout=180):
    (report / f"{label}-command.json").write_text(json.dumps(args, indent=2))
    print(f"BUILD {label}: {args[0]}", flush=True)
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        for field in ("stdout", "stderr"):
            (report / f"{label}-{field}.txt").write_bytes(getattr(exc, field) or b"")
        raise
    for field in ("stdout", "stderr"):
        data = getattr(result, field)
        (report / f"{label}-{field}.txt").write_bytes(data)
    if result.returncode:
        raise RuntimeError(
            f"{label} exit={result.returncode}: "
            + result.stderr.decode(errors="replace")[-6000:]
        )
    print(f"BUILD {label}: passed", flush=True)
    return result.stdout.decode(errors="replace")


def tool(name, ccec=None):
    override = os.environ.get(name.upper().replace("-", "_"))
    candidates = [override, shutil.which(name)]
    if ccec:
        candidates.insert(1, str(Path(ccec).parent / name))
        candidates.insert(2, str(Path(ccec).resolve().parent / name))
    if name == "ccec" and os.environ.get("TRITON_NPU_COMPILER_PATH"):
        candidates.append(str(Path(os.environ["TRITON_NPU_COMPILER_PATH"]) / name))
    return next((v for v in candidates if v and Path(v).is_file()), None)


def include_dirs():
    root = Path(
        os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    )
    candidates = [
        root / "compiler/tikcpp/tikcfw",
        root / "include/ascendc",
        root / "include",
        root / "tools/tikcpp/tikcfw",
    ]
    extra = os.environ.get("QSA_CANN_INCLUDES", "")
    candidates.extend(Path(v) for v in extra.split(os.pathsep) if v)
    headers = []
    for base in candidates:
        if not base.is_dir():
            continue
        for relative in (
            "kernel_operator.h",
            "interface/kernel_operator.h",
            "basic_api/kernel_operator.h",
        ):
            header = base / relative
            if header.is_file():
                headers.append(header)
    if not headers:
        raise MissingPrerequisite(
            f"kernel_operator.h missing below {root}; set QSA_CANN_INCLUDES to its include directories"
        )
    dirs = set()
    for header in headers:
        base = header.parent
        if base.name in ("interface", "basic_api"):
            base = base.parent
        dirs.update((base, base.parent, header.parent))
        dirs.update(
            base / name for name in ("interface", "impl", "basic_api", "adv_api")
        )
    dirs.update(Path(v) for v in extra.split(os.pathsep) if v)
    return sorted(str(p.resolve()) for p in dirs if p.is_dir())


def inspect_bitcode(text, symbols=("qsa_raw_qk", "qsa_raw_chain")):
    calls = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"\b(?:call|invoke)\b", line)
    ]
    patterns = dict(
        matrix_calls=r"(?:mmad|mma[._]|[._]mad[._]|matmul)",
        int8_matrix_calls=r"@llvm\.hivm\.MAD\.s8(?:[.(])",
        half_matrix_calls=r"@llvm\.hivm\.MAD\.f162f32(?:[.(])",
        cc_to_cbuf=r"(?:cc|l0c).*to.*(?:cbuf|l1)",
        prequant=r"(?:set[._]?deq|set[._]?fpc|pre[._]?quant|set[._]quant[._]pre)",
    )
    found = {
        name: [line for line in calls if re.search(pattern, line, re.I)]
        for name, pattern in patterns.items()
    }
    direct = [
        line
        for line in found["cc_to_cbuf"]
        if re.search(r"FIX\.L0C\.TO\.L1\.s322f16", line, re.I)
        and "addrspace(2)" in line
        and "addrspace(5)" in line
    ]
    # Attribute constants and final opcode interpretation still need human review;
    # do not infer a fused ReLU solely from the numerical reference.
    return dict(
        exported_symbols=all(
            re.search(rf"^define[^\n]*@_mlir_ciface_{re.escape(name)}\(", text, re.M)
            is not None
            for name in symbols
        ),
        **found,
        direct_int32_to_half_l1=direct,
        candidate_structure_present=(
            bool(found["int8_matrix_calls"])
            and bool(found["half_matrix_calls"])
            and bool(direct)
            and bool(found["prequant"])
        ),
        relu_mode="Requested by C++ DataCopyCO12DstParams.reluPre=1; retain LLVM arguments for review",
        scope="Custom bitcode structure, not a hardware traffic or overlap measurement",
    )


def build_bitcode(report, result):
    ccec = tool("ccec")
    dis = tool("llvm-dis", ccec)
    linker = tool("llvm-link", ccec)
    if not ccec:
        raise MissingPrerequisite("Need ccec")
    includes = include_dirs()
    source = Path(__file__).with_name("qsa_raw_cube_chain_910b.cpp")
    result["toolchain"] = dict(
        ccec=ccec,
        llvm_dis=dis,
        llvm_link=linker,
        includes=includes,
        bitcode_mode="ordinary-AIC-only",
        mix_bitcode_merged=False,
    )
    result["toolchain"]["version"] = command(
        [ccec, "--version"], report, "ccec-version", 30
    )
    result["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (report / "custom-source.cpp.txt").write_bytes(source.read_bytes())
    common = [
        ccec,
        "-O2",
        "-x",
        "cce",
        "--cce-auto-sync=off",
        "--cce-aicore-only",
        "--cce-generic-addrspace=off",
        "-mllvm",
        "-disable-llvm-optzns",
        "--cce-aicore-arch=dav-c220-cube",
        "-std=c++17",
    ]
    for directory in includes:
        common.extend(["-I", directory])
    # The pointer-only wrapper has one AIC custom op and no AIV scope. Use
    # ordinary AIC bitcode directly. Native kernel_operator.h adds CANN globals
    # (e.g. g_tilingKey); merging plain/mix copies duplicates those definitions.
    # Do not strip or rename the globals to hide that ABI issue.
    output = report / "custom-plain.bc"
    command(
        [*common, str(source), "-emit-llvm", "-c", "-o", str(output)],
        report,
        "ccec-plain",
    )
    result["bitcode_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    # Preserve the successful native build even if text emission or a later
    # Triton lowering/link fails. Only files at the report root are uploaded.
    with gzip.open(str(output) + ".gz", "wb") as stream:
        stream.write(output.read_bytes())
    llvm = report / "custom-plain.ll.txt"
    try:
        if dis:
            result["toolchain"]["llvm_text_method"] = "llvm-dis"
            command([dis, str(output), "-o", str(llvm)], report, "llvm-dis")
        elif linker:
            # llvm-link -S can serialize a single existing bitcode module as
            # LLVM text. There is no second module and no duplicate definitions;
            # the final kernel still consumes the original unchanged .bc file.
            result["toolchain"]["llvm_text_method"] = "llvm-link-S-single-module"
            command(
                [linker, "-S", str(output), "-o", str(llvm)], report, "llvm-link-text"
            )
        else:
            raise MissingPrerequisite(
                "No llvm-dis or llvm-link for bitcode text inspection"
            )
        symbols = tuple(result.get("custom_symbols", {}).values()) or (
            "qsa_raw_qk",
            "qsa_raw_chain",
        )
        result["bitcode_structure"] = inspect_bitcode(llvm.read_text(), symbols)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        # Artifact inspection must not suppress the kernel compile/device
        # diagnostics. Missing structural evidence still fails the final gate.
        result["bitcode_structure"] = dict(
            status="unavailable",
            error=repr(exc),
            exported_symbols=False,
            candidate_structure_present=False,
        )
        print(
            f"LLVM inspection unavailable: {exc}; continuing device diagnostics",
            flush=True,
        )
    return output


def bind_ops(bitcode, result, abi="address"):
    import triton.language.extra.cann.extension as al

    suffix = "_address" if abi == "address" else ""
    symbols = {name: name + suffix for name in ("qsa_raw_qk", "qsa_raw_chain")}
    result["custom_symbols"] = symbols
    try:
        import triton.experimental.tle.language.raw as raw
        from triton.experimental.tle.raw import dialect
        from triton.experimental.tle.raw.cann import CANNJITFunction

        if not callable(getattr(raw, "call", None)):
            raise ImportError("tle language raw.call absent")
    except ImportError as exc:
        result["object_entry"] = dict(available=False, error=repr(exc))
        result["entry"] = "process-local-registry-bitcode"
        result["entry_scope"] = (
            "Same bitcode, no installed package edits; object/source-JIT entry not proven"
        )

        def register(name):
            def initialize(self, q, key, weight, scale_bits, dst):
                self.symbol, self.bitcode = symbols[name], str(bitcode)

            cls = type(
                name,
                (),
                dict(
                    core=al.CORE.CUBE,
                    pipe=al.PIPE.PIPE_M,
                    mode=al.MODE.SIMD,
                    __init__=initialize,
                ),
            )
            al.register_custom_op(cls)

        for name in ("qsa_raw_qk", "qsa_raw_chain"):
            register(name)
        return {}
    result["object_entry"] = dict(
        available=True,
        runtime=inspect.getfile(CANNJITFunction),
        signature=str(inspect.signature(CANNJITFunction)),
    )
    result["entry"] = "dialect-cann-bitcode"
    result["entry_scope"] = (
        "Operator-owned C++ compiled above; object bitcode path, not built-in source-JIT validation"
    )
    bound = {}
    for name in ("qsa_raw_qk", "qsa_raw_chain"):

        def stub(*args, **kwargs):
            pass

        stub.__name__ = name
        bound[name] = dialect(
            name="cann",
            file=str(bitcode),
            format="bitcode",
            extern_func_name=symbols[name],
            pipeline={"core": "cube", "pipe": "PIPE_M"},
        )(stub)
    return bound


def wrapper_source(entry, abi="address"):
    if entry == "dialect-cann-bitcode":
        imports = "import triton.experimental.tle.language.raw as raw"
        chain = "raw.call(chain_op, [q, key, weight, bits, output], output_indices=[])"
        qk = "raw.call(qk_op, [q, key, weight, bits, output], output_indices=[])"
    else:
        imports = "import triton.experimental.tle as tle"
        chain = 'tle.dsa.ascend.raw("qsa_raw_chain", q, key, weight, bits, output)'
        qk = 'tle.dsa.ascend.raw("qsa_raw_qk", q, key, weight, bits, output)'
    if abi == "address":
        views = """q = tl.cast(Q, tl.uint64)
        key = tl.cast(Key, tl.uint64)
        weight = tl.cast(Weight, tl.uint64)
        output = tl.cast(Out, tl.uint64)"""
    else:
        views = """q = tl.make_block_ptr(Q, (2048,), (1,), (0,), (2048,), (0,))
        key = tl.make_block_ptr(Key, (8192,), (1,), (0,), (8192,), (0,))
        weight = tl.make_block_ptr(Weight, (256,), (1,), (0,), (256,), (0,))
        output = tl.make_block_ptr(Out, (1024,), (1,), (0,), (1024,), (0,))"""
    return f"""
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
{imports}

@triton.jit
def custom(Q, Key, Weight, Out, SCALE_BITS:tl.constexpr, CHAIN:tl.constexpr):
    # All five arguments are external-function operands. There is no
    # tensor-valued DPS result: the custom op has no outs/results. Leave
    # no_side_effect unset so its opaque memory writes remain conservative.
    with al.scope(core_mode="cube"):
        # Scalar addresses or static views, with no surrounding tile loads.
        {views}
        bits = tl.full((), SCALE_BITS, tl.int64)
        if CHAIN:
            {chain}
        else:
            {qk}

@triton.jit
def reference(Q, Key, Weight, Out, SCALE:tl.constexpr):
    m = tl.arange(0,16)
    k = tl.arange(0,128)
    n = tl.arange(0,64)
    q = tl.load(Q + m[:,None]*128 + k[None,:])
    key = tl.load(Key + n[None,:]*128 + k[:,None])
    dot = tl.dot(q, key, out_dtype=tl.int32)
    positive = tl.maximum(dot.to(tl.float32)*SCALE, 0).to(tl.float16)
    weight = tl.load(Weight + m[:,None]*16 + m[None,:])
    out = tl.dot(weight, positive, out_dtype=tl.float32)
    tl.store(Out + m[:,None]*64 + n[None,:], out)

# Independent smoke kernels; neither is called from the custom wrapper.
@triton.jit
def vector_smoke(X, Out):
    i = tl.arange(0,128)
    x = tl.load(X+i)
    tl.store(Out+i, x*1.5+0.25)

@triton.jit
def dot_vector_smoke(Q, Key, Out):
    m = tl.arange(0,16)
    k = tl.arange(0,128)
    n = tl.arange(0,64)
    q = tl.load(Q+m[:,None]*128+k[None,:])
    key = tl.load(Key+n[None,:]*128+k[:,None])
    dot = tl.dot(q,key,out_dtype=tl.int32)
    result = tl.exp(dot.to(tl.float32)/8192.0)
    tl.store(Out+m[:,None]*64+n[None,:],result)
"""


def pack_nz(value, c0):
    rows, columns = value.shape
    if rows % 16 or columns % c0:
        raise ValueError("NZ pack requires aligned rows and columns")
    return (
        value.reshape(rows // 16, 16, columns // c0, c0)
        .permute(2, 0, 1, 3)
        .contiguous()
        .flatten()
    )


def fixtures(torch):
    for seed in (13, 71, 109):
        gen = torch.Generator().manual_seed(seed)
        q = torch.randint(-8, 9, (M, K), dtype=torch.int8, generator=gen)
        key = torch.randint(-8, 9, (N, K), dtype=torch.int8, generator=gen)
        q[0].fill_(8)
        q[0, 0] = 7
        q[1] = -q[0]
        q[2].zero_()
        key[0].fill_(4)
        key[0, 0] = 5
        key[1] = -key[0]
        dot = q.int() @ key.int().T
        assert int(dot[0, 0]) == 4099 and int(dot[1, 0]) == -4099
        yield seed, q, key, dot


def save_asm(kernel, report, label):
    inventory = []
    for name, data in kernel.asm.items():
        if isinstance(data, str):
            (report / f"{label}-{name}.txt").write_text(data)
            inventory.append(dict(kind=name, bytes=len(data)))
        elif isinstance(data, (bytes, bytearray)):
            with gzip.open(report / f"{label}-{name}.bin.gz", "wb") as stream:
                stream.write(data)
            inventory.append(dict(kind=name, bytes=len(data), binary=True))
    return inventory


def record_launch_mode(kernel, require_cube=True):
    metadata = kernel.metadata
    mode = (
        metadata.get("mix_mode")
        if isinstance(metadata, dict)
        else getattr(metadata, "mix_mode", None)
    )
    task = (
        metadata.get("bs_task_type")
        if isinstance(metadata, dict)
        else getattr(metadata, "bs_task_type", None)
    )
    entry = dict(mix_mode=mode, bs_task_type=task, metadata=str(metadata))
    if require_cube and mode == "aiv":
        raise RuntimeError(
            "Cube custom-op wrapper still reports mix_mode=aiv; refusing device execution. "
            + str(entry)
        )
    return entry


def collect_runtime_sources(compiler, report):
    driver = __import__("triton.backends.ascend.driver", fromlist=["__file__"])
    paths = [
        Path(compiler.__file__),
        Path(driver.__file__),
        Path(driver.__file__).with_name("npu_utils.cpp"),
    ]
    records = []
    for path in paths:
        if not path.is_file():
            records.append(dict(source=str(path), available=False))
            continue
        data = path.read_bytes()
        target = "installed-" + path.name + ".txt"
        (report / target).write_bytes(data[: 512 * 1024])
        records.append(
            dict(
                source=str(path),
                file=target,
                sha256=hashlib.sha256(data).hexdigest(),
                bytes=len(data),
                truncated=len(data) > 512 * 1024,
            )
        )
    return records


def verify_compiler_candidate(compiler, triton, report, result):
    path = os.environ.get("QSA_CORE_MODE_FIX_MANIFEST")
    result["compiler_variant"] = "original"
    if not path:
        return False
    manifest = json.loads(Path(path).read_text())
    if manifest.get("status") != "prepared":
        raise RuntimeError("Compiler candidate was not prepared successfully")
    root = Path(manifest["candidate_root"]).resolve()
    actual = Path(compiler.__file__).resolve()
    if not actual.is_relative_to(root) or not Path(
        triton.__file__
    ).resolve().is_relative_to(root):
        raise RuntimeError(
            "Compiler/Triton import escaped the independent candidate copy"
        )
    if (
        hashlib.sha256(actual.read_bytes()).hexdigest()
        != manifest["patched_compiler_sha256"]
    ):
        raise RuntimeError("Imported compiler does not match the candidate hash")
    original = Path(manifest["source_compiler"])
    if (
        hashlib.sha256(original.read_bytes()).hexdigest()
        != manifest["source_compiler_sha256"]
    ):
        raise RuntimeError(
            "Original installed compiler no longer matches its preparation snapshot"
        )
    helper = getattr(compiler, manifest["helper"], None)
    if not callable(helper) or Path(helper.__code__.co_filename).resolve() != actual:
        raise RuntimeError(
            "Reconciliation helper did not load from candidate compiler source"
        )
    for task, mode in (
        (10, "aiv"),
        (20, "aic"),
        (31, "mix"),
        (32, "mix"),
        (41, "mix"),
        (42, "mix"),
    ):
        values = dict(mix_mode="old", bs_task_type=task, workspace_size=4096)
        helper(values)
        if values != dict(mix_mode=mode, bs_task_type=task, workspace_size=4096):
            raise RuntimeError(
                "Imported candidate failed task/ratio preservation check"
            )
    result["compiler_variant"] = "isolated-final-task-mode-candidate"
    result["compiler_candidate"] = manifest
    print(f"VERIFIED_COMPILER_CANDIDATE {actual}", flush=True)
    return True


def candidate_smokes(module, device, report, result, save, compiler, capture_type):
    import torch

    x = torch.arange(128, dtype=torch.float32) / 16
    _, q, key, dot = next(fixtures(torch))
    vector_out = torch.full((128,), float("nan"), device=device)
    dot_out = torch.full((M, N), float("nan"), device=device)
    cases = [
        (
            "vector",
            module.vector_smoke,
            (x.to(device), vector_out),
            vector_out,
            x * 1.5 + 0.25,
        ),
        (
            "dot-vector",
            module.dot_vector_smoke,
            (q.to(device), key.to(device), dot_out),
            dot_out,
            torch.exp(dot.float() / 8192.0),
        ),
    ]
    result["candidate_smokes"] = []
    for name, function, inputs, output, expected in cases:
        label = "candidate-smoke-" + name
        result["phase"] = label
        entry = dict(name=name, status="compiling")
        result["candidate_smokes"].append(entry)
        capture = capture_type(report, label)
        try:
            with patch.object(compiler, "subprocess", capture):
                kernel = function.warmup(*inputs, grid=(1,))
            launch = record_launch_mode(kernel, require_cube=name != "vector")
            kind = int(launch.get("bs_task_type") or 0) // 10
            expected_mode = {1: "aiv", 2: "aic", 3: "mix", 4: "mix"}.get(kind)
            if launch["mix_mode"] != expected_mode or name == "vector" and kind != 1:
                raise RuntimeError(f"Smoke core kind/mode inconsistent: {launch}")
            entry.update(
                launch=launch, asm=save_asm(kernel, report, label), status="compiled"
            )
            save()
            function[(1,)](*inputs)
            torch.npu.synchronize()
            actual = output.cpu()
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0 if name == "vector" else 1e-3,
                atol=0 if name == "vector" else 1e-3,
            )
            entry.update(
                status="passed", max_abs_error=float((actual - expected).abs().max())
            )
            print(
                f"CANDIDATE_SMOKE {name}: passed mode={launch['mix_mode']} "
                f"task={launch['bs_task_type']} "
                f"max_abs_error={entry['max_abs_error']:.6g}",
                flush=True,
            )
        finally:
            entry["compiler_calls"] = capture.calls
            save()


def preflight_core_options(
    module, tensors, report, result, save, compiler, capture_type
):
    import torch

    q, key, weight = tensors
    outputs = {
        False: torch.empty((M, N), device=q.device, dtype=torch.int32),
        True: torch.empty((M, N), device=q.device, dtype=torch.float32),
    }
    choices = [("default", {})]
    fields = getattr(getattr(compiler, "NPUOptions", None), "__dataclass_fields__", {})
    if "mix_mode" in fields:
        choices.append(("explicit-aic", {"mix_mode": "aic"}))
    result["core_option_matrix"] = []
    selected = None
    for option_name, launch_options in choices:
        complete = []
        for chain in (False, True):
            label = f"core-{option_name}-{'chain' if chain else 'qk'}"
            result["phase"] = label
            capture = capture_type(report, label)
            entry = dict(
                label=label,
                requested_options=launch_options,
                status="failed",
                device_executed=False,
            )
            result["core_option_matrix"].append(entry)
            save()
            try:
                with patch.object(compiler, "subprocess", capture):
                    kernel = module.custom.warmup(
                        q,
                        key,
                        weight,
                        outputs[chain],
                        0x3F800000,
                        chain,
                        grid=(1,),
                        **launch_options,
                    )
                entry.update(
                    status="compiled",
                    asm=save_asm(kernel, report, label),
                    launch=record_launch_mode(kernel, require_cube=False),
                )
                task_kind = int(entry["launch"].get("bs_task_type") or 0) // 10
                entry["core_coherent"] = entry["launch"]["mix_mode"] in (
                    "aic",
                    "mix",
                ) and task_kind in (2, 3, 4)
                complete.append(entry["core_coherent"])
            except Exception as exc:
                entry["error"] = repr(exc)
                (report / f"{label}-failure.txt").write_text(traceback.format_exc())
                complete.append(False)
            finally:
                entry["compiler_calls"] = capture.calls
                save()
            launch = entry.get("launch", {})
            print(
                f"CORE_OPTION {label}: status={entry['status']} "
                f"mix_mode={launch.get('mix_mode')} "
                f"bs_task_type={launch.get('bs_task_type')} "
                f"coherent={entry.get('core_coherent',False)}",
                flush=True,
            )
        if all(complete) and selected is None:
            selected = launch_options
    if selected is None:
        result["outcome"] = "no_coherent_public_core_configuration"
        save()
        raise RuntimeError(
            "Neither default nor the available public AIC option compiled both "
            "kernels with coherent Cube launch metadata; no target kernel executed. "
            "See core_option_matrix and installed runtime sources"
        )
    result["launch_options"] = selected
    save()
    return selected


def execute(args, report, result, save):
    import torch
    import torch_npu  # noqa: F401
    import triton
    from diagnose_qsa_structural_mixed_compile_910b import CaptureCompiler
    from triton.backends.ascend import compiler
    from triton.backends.ascend.testing import do_bench_npu

    result["installed_runtime_sources"] = collect_runtime_sources(compiler, report)
    is_compiler_candidate = verify_compiler_candidate(compiler, triton, report, result)
    trace = dict(checked=False, flags=[])
    result["compiler_trace"] = trace

    class RawCompilerCapture(CaptureCompiler):
        def run(self, command_args, *positional, **kwargs):
            if (
                isinstance(command_args, (list, tuple))
                and command_args
                and Path(command_args[0]).name == "bishengir-compile"
            ):
                if not trace["checked"]:
                    trace["checked"] = True
                    try:
                        help_result = subprocess.run(
                            [command_args[0], "--help-hidden"],
                            env=kwargs.get("env"),
                            capture_output=True,
                            timeout=20,
                        )
                        help_text = (help_result.stdout + help_result.stderr).decode(
                            errors="replace"
                        )
                        (report / "bishengir-help.txt").write_text(
                            help_text[: 1024 * 1024]
                        )
                        for when in ("before", "after"):
                            for prefix in ("bishengir", "mlir"):
                                option = f"--{prefix}-print-ir-{when}"
                                if re.search(
                                    re.escape(option) + r"(?:[=\s])", help_text
                                ):
                                    trace["flags"].append(option + "=hivm-plan-memory")
                                    break
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        trace["help_error"] = repr(exc)
                    print("COMPILER_TRACE " + json.dumps(trace), flush=True)
                command_args = [*command_args, *trace["flags"]]
            return super().run(command_args, *positional, **kwargs)

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    result["environment"] = dict(
        torch=torch.__version__,
        triton=triton.__version__,
        triton_source=triton.__file__,
        device=torch.npu.get_device_name(args.device),
        ascend_home=os.environ.get("ASCEND_HOME_PATH"),
        opp=os.environ.get("ASCEND_OPP_PATH"),
    )
    if "910B" not in result["environment"]["device"].upper():
        raise MissingPrerequisite(
            "This fixed dav-c220 probe requires an Ascend910B device"
        )
    bindings = bind_ops(report / "custom-plain.bc", result, args.abi)
    result["call_abi"] = dict(
        representation=args.abi,
        operands="Q,K,W,scale,destination order; scalar uint64 addresses or rank-one GM views",
        static_elements=dict(q=2048, key=8192, weight=256, destination=1024),
        core_scope="cube",
        dps_outputs=0,
        results=0,
        destination="Fifth ordinary operand; C++ writes GM through the explicitly selected ABI",
        side_effects=(
            "Opaque custom op; no no_side_effect annotation; "
            "inspect retained op and validate destination writes"
        ),
    )
    print("ENVIRONMENT " + json.dumps(result["environment"]), flush=True)
    print(f"CUSTOM_ENTRY {result['entry']}", flush=True)
    print(f"CALL_ABI {args.abi}: {result['custom_symbols']}", flush=True)
    result["phase"] = "compile_operator_owned_bitcode"
    save()
    build_bitcode(report, result)
    structure = result["bitcode_structure"]
    print(
        "BITCODE_STRUCTURE "
        + json.dumps(
            dict(
                exported_symbols=structure["exported_symbols"],
                candidate_structure_present=structure["candidate_structure_present"],
                int8_mad_static_sites=len(structure.get("int8_matrix_calls", [])),
                half_mad_static_sites=len(structure.get("half_matrix_calls", [])),
                direct_l1_static_sites=len(
                    structure.get("direct_int32_to_half_l1", [])
                ),
                scope="Static call sites, not executed instruction counts",
            )
        ),
        flush=True,
    )
    save()
    source = report / "raw-kernels.py"
    source.write_text(wrapper_source(result["entry"], args.abi))
    (report / "raw-kernels-source.txt").write_bytes(source.read_bytes())
    spec = importlib.util.spec_from_file_location("_qsa_raw_chain", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if bindings:
        module.qk_op, module.chain_op = (
            bindings["qsa_raw_qk"],
            bindings["qsa_raw_chain"],
        )
    signature = inspect.signature(do_bench_npu)
    options = {
        key: value
        for key, value in dict(warmup=args.warmup, active=args.active).items()
        if key in signature.parameters
    }
    if len(options) != 2:
        raise MissingPrerequisite(
            "do_bench_npu needs explicit warmup/active parameters"
        )
    if "clear_l2_cache" in signature.parameters:
        options["clear_l2_cache"] = False
    result["timer"] = dict(
        name="triton.backends.ascend.testing.do_bench_npu",
        signature=str(signature),
        options=options,
        scope="Single-program diagnostic; prepared NZ weights/inputs; not QSA acceptance",
    )
    result["checks"], result["timings"] = [], []
    if is_compiler_candidate:
        candidate_smokes(
            module, device, report, result, save, compiler, RawCompilerCapture
        )
    _, preflight_q, preflight_key, _ = next(fixtures(torch))
    preflight_tensors = [
        pack_nz(preflight_q, 32).to(device),
        pack_nz(preflight_key, 32).to(device),
        pack_nz(torch.eye(M, dtype=torch.float16), 16).to(device),
    ]
    launch_options = preflight_core_options(
        module, preflight_tensors, report, result, save, compiler, RawCompilerCapture
    )
    for seed, q, key, dot in fixtures(torch):
        q_nz, key_nz = [pack_nz(t, 32).to(device) for t in (q, key)]
        weights = {"identity": torch.eye(M, dtype=torch.float16)}
        grouped = torch.zeros((M, M), dtype=torch.float16)
        for row in range(4):
            grouped[row, row * 4 : row * 4 + 4] = torch.tensor(
                [0.25, 0.5, 1, 1.5]
            ).half()
        weights["grouped"] = grouped
        for kind, weight in weights.items():
            # L1 weights store W.T; the CANN LoadData2D transpose forms W in L0A.
            w_nz = pack_nz(weight.T.contiguous(), 16).to(device)
            if kind == "identity":
                label = f"qk-seed{seed}"
                result["phase"] = label
                print(f"COMPILE {label}", flush=True)
                capture = RawCompilerCapture(report, label)
                raw = torch.full((M, N), 2**30, device=device, dtype=torch.int32)
                try:
                    with patch.object(compiler, "subprocess", capture):
                        kernel = module.custom.warmup(
                            q_nz,
                            key_nz,
                            w_nz,
                            raw,
                            0x3F800000,
                            False,
                            grid=(1,),
                            **launch_options,
                        )
                finally:
                    result["last_compiler_calls"] = capture.calls
                    save()
                result["checks"].append(
                    dict(
                        label=label,
                        status="compiled",
                        compiler_calls=capture.calls,
                        asm=save_asm(kernel, report, label),
                        launch=record_launch_mode(kernel),
                    )
                )
                save()
                module.custom[(1,)](
                    q_nz, key_nz, w_nz, raw, 0x3F800000, False, **launch_options
                )
                torch.npu.synchronize()
                torch.testing.assert_close(raw.cpu(), dot, rtol=0, atol=0)
                result["checks"][-1]["status"] = "passed"
                print(
                    f"QK seed={seed}: exact INT32 passed, min={int(dot.min())}, max={int(dot.max())}",
                    flush=True,
                )
                save()
            for scale_bits, scale in ((0x3F800000, 1.0), (0x3A800000, 1 / 1024)):
                label = f"chain-{kind}-seed{seed}-scale{scale_bits:x}"
                result["phase"] = label
                print(f"COMPILE {label}", flush=True)
                output = torch.full((M, N), float("nan"), device=device)
                reference_output = torch.empty_like(output)
                expected = (
                    weight.float() @ (dot.float() * scale).clamp_min(0).half().float()
                )
                capture = RawCompilerCapture(report, label)
                try:
                    with patch.object(compiler, "subprocess", capture):
                        compiled = module.custom.warmup(
                            q_nz,
                            key_nz,
                            w_nz,
                            output,
                            scale_bits,
                            True,
                            grid=(1,),
                            **launch_options,
                        )
                finally:
                    result["last_compiler_calls"] = capture.calls
                    save()
                result["checks"].append(
                    dict(
                        label=label,
                        status="compiled",
                        compiler_calls=capture.calls,
                        asm=save_asm(compiled, report, label),
                        launch=record_launch_mode(compiled),
                    )
                )
                save()
                custom = lambda: module.custom[(1,)](
                    q_nz, key_nz, w_nz, output, scale_bits, True, **launch_options
                )
                custom()
                torch.npu.synchronize()
                actual = output.cpu()
                torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
                check = result["checks"][-1]
                check.update(
                    status="passed",
                    max_abs_error=float((actual - expected).abs().max()),
                )
                print(
                    f"CHAIN {label}: passed max_abs_error={check['max_abs_error']:.6g}",
                    flush=True,
                )
                save()
                if seed != 13:
                    continue
                logical_q, logical_key, logical_weight = [
                    t.to(device) for t in (q, key, weight)
                ]
                reference = lambda: module.reference[(1,)](
                    logical_q, logical_key, logical_weight, reference_output, scale
                )
                ref_kernel = module.reference.warmup(
                    logical_q,
                    logical_key,
                    logical_weight,
                    reference_output,
                    scale,
                    grid=(1,),
                )
                save_asm(ref_kernel, report, f"reference-{kind}-{scale_bits:x}")
                reference()
                torch.npu.synchronize()
                torch.testing.assert_close(
                    reference_output.cpu(), expected, rtol=1e-3, atol=1e-3
                )
                samples = dict(custom=[], reference=[])
                for round_index in range(args.rounds):
                    for name in (
                        ("custom", "reference")
                        if round_index % 2 == 0
                        else ("reference", "custom")
                    ):
                        value = (
                            float(
                                do_bench_npu(
                                    custom if name == "custom" else reference, **options
                                )
                            )
                            * 1000
                        )
                        if not math.isfinite(value) or value <= 0:
                            raise RuntimeError(f"Invalid kernel_us={value}")
                        samples[name].append(value)
                        print(
                            f"TIME {label} round={round_index + 1} {name}={value:.3f} us",
                            flush=True,
                        )
                ratio = statistics.median(samples["reference"]) / statistics.median(
                    samples["custom"]
                )
                result["timings"].append(
                    dict(label=label, kernel_us=samples, reference_over_custom=ratio)
                )
                print(
                    f"RATIO {label}: reference/custom={ratio:.4f}x (micro probe)",
                    flush=True,
                )
                save()
    structural = result["bitcode_structure"]
    result["numeric_chain_passed"] = True
    result["local_chain_supported"] = bool(
        structural["exported_symbols"] and structural["candidate_structure_present"]
    )
    result["pipeline_proven"] = False
    result["pending_evidence"] = (
        "Review CC-to-CBUF call arguments for DEQF16/ReLU and final generated code; no traffic or overlap measurement"
    )
    result["outcome"] = (
        "passed_basic_structural_and_numerical_checks"
        if result["local_chain_supported"]
        else "numerically_passed_structure_needs_review"
    )
    if not result["local_chain_supported"]:
        raise RuntimeError(
            "Numerical checks passed; LLVM did not establish the requested local dataflow"
        )
    if args.score_chunks:
        from qsa_raw_score_chunks_910b import run_chunks

        result["phase"] = "representative-score-chunks"
        save()
        run_chunks(
            args,
            report,
            result,
            bindings,
            compiler,
            RawCompilerCapture,
            save_asm,
            record_launch_mode,
            pack_nz,
        )
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--abi", default="address", choices=("address", "memref"))
    parser.add_argument("--rounds", default=2, type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--warmup", default=3, type=int)
    parser.add_argument("--active", default=10, type=int)
    parser.add_argument(
        "--score-chunks",
        action="store_true",
        help="Audit final objects and compare three real INT8 chunks including device packing",
    )
    args = parser.parse_args()
    if args.score_chunks and args.abi != "address":
        parser.error("--score-chunks requires the verified scalar-address ABI")
    if args.warmup < 1 or args.active < 1:
        parser.error("warmup and active must be positive")
    args.report.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="running",
        phase="prerequisites",
        pipeline_proven=False,
        scope="Operator-owned native custom-op feasibility, no production changes",
    )
    path = args.report / "raw-cube-chain.json"
    save = lambda: write_json(path, result)
    save()
    status = 0
    try:
        execute(args, args.report, result, save)
        result["status"] = "passed"
    except (MissingPrerequisite, ImportError) as exc:
        result.update(status="prerequisites_missing", error=repr(exc))
        status = 2
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        (args.report / "failure-traceback.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        status = 1
    finally:
        result["exit_code"] = status
        save()
        summary = {
            key: result[key]
            for key in (
                "status",
                "phase",
                "entry",
                "compiler_variant",
                "score_chunks",
                "outcome",
                "error",
                "numeric_chain_passed",
                "local_chain_supported",
                "pipeline_proven",
                "pending_evidence",
                "exit_code",
            )
            if key in result
        }
        summary["report"] = str(path)
        print("RAW_CUBE_SUMMARY " + json.dumps(summary, allow_nan=False), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
