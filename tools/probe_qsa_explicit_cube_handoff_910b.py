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

"""Probe explicit Cube buffers; require fused INT32/ReLU/F16 Fixpipe into L1."""

import importlib.util
import json
import math
import re
import statistics
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

DUMP_FLAG = "--bishengir-print-ir-after=hivm-graph-sync-solver"
KERNEL_SOURCE = """
import triton
import triton.language as tl
import triton.experimental.tle as tle
import triton.language.extra.cann.extension as al

@triton.jit
def explicit(A, B, W, Out, M:tl.constexpr, K:tl.constexpr, N:tl.constexpr,
             MODE:tl.constexpr):
    a = tle.dsa.alloc([M,K], tl.int8, tle.dsa.ascend.L1)
    b = tle.dsa.alloc([K,N], tl.int8, tle.dsa.ascend.L1)
    sa = tle.dsa.alloc([M,K], tl.int8, tle.dsa.ascend.L0A)
    sb = tle.dsa.alloc([K,N], tl.int8, tle.dsa.ascend.L0B)
    acc = tle.dsa.alloc([M,N], tl.int32, tle.dsa.ascend.L0C)
    ap = tl.make_block_ptr(A, (M,K), (K,1), (0,0), (M,K), (1,0))
    bp = tl.make_block_ptr(B, (K,N), (N,1), (0,0), (K,N), (1,0))
    tle.dsa.copy(ap, a, [M,K])
    tle.dsa.copy(bp, b, [K,N])
    if MODE == 0:
        dst = tle.dsa.alloc([M,N], tl.int32, al.ascend_address_space.GM)
        tle.dsa.tile_cube_launch(a,b,acc,sa,sb,dst,init=True)
        tle.dsa.tile_cube_wait()
        op = tl.make_block_ptr(Out, (M,N), (N,1), (0,0), (M,N), (1,0))
        tle.dsa.copy(dst, op, [M,N])
    else:
        middle = tle.dsa.alloc([M,N], tl.float16, tle.dsa.ascend.L1)
        tle.dsa.tile_cube_launch(a,b,acc,sa,sb,middle,init=True)
        tle.dsa.tile_cube_wait()
        if MODE == 2:
            positive = tl.maximum(tle.dsa.to_tensor(middle), 0.0).to(tl.float16)
            middle = tle.dsa.to_buffer(positive, space=tle.dsa.ascend.L1)
        weight = tle.dsa.alloc([M,M], tl.float16, tle.dsa.ascend.L1)
        wp = tl.make_block_ptr(W, (M,M), (M,1), (0,0), (M,M), (1,0))
        tle.dsa.copy(wp, weight, [M,M])
        wa = tle.dsa.alloc([M,M], tl.float16, tle.dsa.ascend.L0A)
        wb = tle.dsa.alloc([M,N], tl.float16, tle.dsa.ascend.L0B)
        acc2 = tle.dsa.alloc([M,N], tl.float32, tle.dsa.ascend.L0C)
        dst2 = tle.dsa.alloc([M,N], tl.float32, al.ascend_address_space.GM)
        tle.dsa.tile_cube_launch(weight,middle,acc2,wa,wb,dst2,init=True)
        tle.dsa.tile_cube_wait()
        op = tl.make_block_ptr(Out, (M,N), (N,1), (0,0), (M,N), (1,0))
        tle.dsa.copy(dst2, op, [M,N])

@triton.jit
def reference(A, B, W, Out, M:tl.constexpr, K:tl.constexpr, N:tl.constexpr):
    m=tl.arange(0,M)
    k=tl.arange(0,K)
    n=tl.arange(0,N)
    dot=tl.dot(tl.load(A+m[:,None]*K+k[None,:]),
               tl.load(B+k[:,None]*N+n[None,:]),out_dtype=tl.int32)
    positive=tl.maximum(dot.to(tl.float32),0.0).to(tl.float16)
    w=tl.load(W+m[:,None]*M+m[None,:])
    result=tl.dot(w,positive,out_dtype=tl.float32)
    tl.store(Out+m[:,None]*N+n[None,:],result)
"""


def inspect_ir(text, mode):
    active = False
    lines = []
    sections = 0
    for number, line in enumerate(text.splitlines(), 1):
        if "IR Dump" in line:
            active = "IR Dump After" in line and "graphsyncsolver" in re.sub(
                r"[^a-z]", "", line.lower()
            )
            sections += int(active)
        if active:
            lines.append((number, line))
    mmads = [
        n for n, l in lines if re.search(r"hivm\.hir\.(?:mmadL1|mmad|matmul)\s", l)
    ]
    fix = [dict(line=n, text=l.strip()) for n, l in lines if "hivm.hir.fixpipe " in l]
    direct = [
        e
        for e in fix
        if re.search(
            r"ins\(.*?xi32.*?address_space<cc>.*?outs\(.*?xf16.*?address_space<cbuf>",
            e["text"],
        )
    ]
    relu = [
        e
        for e in direct
        if re.search(
            r"(?:relu\w*\s*=\s*(?:true|1\b|[^,}]*normal_relu))", e["text"], re.I
        )
    ]
    gm_intermediates = []
    for n, l in lines:
        if (
            not ("alloc_workspace" in l or "memref.alloc" in l)
            or "address_space<gm>" not in l
        ):
            continue
        for shape in re.findall(
            r"memref<((?:[0-9]+x)+)(?:i32|f16),[^>]*address_space<gm>", l
        ):
            elements = math.prod(int(v) for v in shape.rstrip("x").split("x"))
            if elements >= 16:
                gm_intermediates.append(dict(line=n, text=l.strip()))
    return dict(
        passed=sections > 0
        and len(mmads) == (1 if mode == 0 else 2)
        and (mode == 0 or bool(direct) and not gm_intermediates)
        and (mode != 2 or bool(relu)),
        sections=sections,
        matmul_lines=mmads,
        fixpipe=fix,
        direct_int32_to_half_l1=direct,
        fused_relu=relu,
        gm_intermediates=gm_intermediates,
        scope="Static structural gate, not a traffic measurement; scalar workspace and final FP32 output allowed",
    )


def run_probe(report, device, timer):
    from functools import partial

    import torch
    from diagnose_qsa_structural_mixed_compile_910b import CaptureCompiler
    from triton.backends.ascend import compiler

    class DumpCompiler(CaptureCompiler):
        def run(self, command, *args, **kwargs):
            if (
                isinstance(command, (tuple, list))
                and command
                and Path(command[0]).name == "bishengir-compile"
            ):
                command = [*command, DUMP_FLAG]
            return super().run(command, *args, **kwargs)

    path = report / "explicit-cube-handoff.json"
    result = dict(
        status="running",
        outcome="pending",
        stages=[],
        timings={},
        scope="One program M16/K128/N64; explicit low-level API feasibility, no full operator acceptance",
        numerical_contract=(
            "Small signed INT8 values; QK exact INT32; FP16 cast/ReLU then half weights "
            "with FP32 accumulation, rtol/atol1e-3; does not reproduce CANN DEQF16 scale flag"
        ),
    )
    try:
        source = report / "explicit-cube-kernels.py"
        source.write_text(KERNEL_SOURCE)
        (report / "explicit-cube-kernels-source.txt").write_text(KERNEL_SOURCE)
        spec = importlib.util.spec_from_file_location("_qsa_explicit_cube", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        import inspect

        import triton
        import triton.experimental.tle as tle
        import triton.language.extra.cann.extension as al

        result["api"] = {}
        for name in (
            "tile_cube_launch",
            "tile_cube_wait",
            "alloc",
            "copy",
            "to_tensor",
            "to_buffer",
        ):
            value = getattr(tle.dsa, name, None)
            result["api"][name] = dict(
                present=value is not None,
                signature=str(inspect.signature(value)) if value is not None else None,
            )
        result["api"]["GM"] = dict(present=hasattr(al.ascend_address_space, "GM"))
        result["environment"] = dict(
            torch=torch.__version__,
            triton=triton.__version__,
            triton_source=triton.__file__,
            ascend_home=__import__("os").environ.get("ASCEND_HOME_PATH"),
        )
        if not all(e["present"] for e in result["api"].values()):
            result.update(status="completed", outcome="missing_explicit_api")
            return result
        gen = torch.Generator().manual_seed(20261112)
        a = torch.randint(-8, 9, (16, 128), generator=gen, dtype=torch.int8)
        b = torch.randint(-8, 9, (128, 64), generator=gen, dtype=torch.int8)
        # Include +/-4099: finite FP16 results with observable rounding and ReLU.
        a[0].fill_(8)
        a[0, 0] = 7
        a[1] = -a[0]
        b[:, 0].fill_(4)
        b[0, 0] = 5
        b[:, 1] = -b[:, 0]
        raw = a.int() @ b.int()
        assert int(raw[0, 0]) == 4099 and int(raw[1, 0]) == -4099
        assert bool((raw.half().float() != raw.float()).any())
        assert bool((raw < 0).any() and (raw > 0).any())
        w = torch.zeros((16, 16), dtype=torch.float16)
        for row in range(4):
            w[row, row * 4 : row * 4 + 4] = torch.tensor([0.25, 0.5, 1.0, 1.5]).half()
        values = [t.to(device) for t in (a, b, w)]
        result["device"] = torch.npu.get_device_name(torch.npu.current_device())
        result["source_triton"] = __import__("triton").__file__
        result["inputs"] = dict(
            raw_min=int(raw.min()),
            raw_max=int(raw.max()),
            negative_elements=int((raw < 0).sum()),
        )
        compiled_functions = {}
        for mode, label in (
            (0, "launch_control"),
            (1, "direct_half_l1"),
            (2, "relu_half_l1"),
        ):
            output = torch.empty(
                (16, 64),
                device=device,
                dtype=torch.int32 if mode == 0 else torch.float32,
            )
            args = (*values, output)
            kwargs = dict(M=16, K=128, N=64, MODE=mode)
            capture = DumpCompiler(report, label)
            entry = dict(mode=mode, label=label, phase="compilation")
            result["stages"].append(entry)
            path.write_text(json.dumps(result, indent=2))
            try:
                with patch.object(compiler, "subprocess", capture):
                    kernel = module.explicit.warmup(*args, grid=(1,), **kwargs)
            except Exception:
                entry.update(
                    status="compile_failed",
                    error=traceback.format_exc(),
                    compiler_calls=capture.calls,
                )
                result.update(
                    status="completed",
                    outcome=(
                        "blocked_launch_control" if mode == 0 else "rejected_compile"
                    ),
                )
                return result
            entry.update(compiler_calls=capture.calls, metadata=str(kernel.metadata))
            texts = []
            truncated = False
            for call in capture.calls:
                for stream in ("stdout", "stderr"):
                    if call.get(stream):
                        texts.append(
                            (report / call[stream]["file"]).read_text(errors="replace")
                        )
                        truncated |= call[stream]["truncated"]
            entry["ir"] = inspect_ir("\n".join(texts), mode)
            if truncated or len(capture.calls) != 1 or not entry["ir"]["passed"]:
                entry["status"] = "ir_not_proven"
                result.update(
                    status="completed",
                    outcome="blocked_launch_control" if mode == 0 else "rejected_ir",
                )
                return result
            entry["phase"] = "execution"
            path.write_text(json.dumps(result, indent=2))
            fn = partial(module.explicit[(1,)], *args, **kwargs)
            fn()
            actual = output.cpu()
            expected = (
                raw
                if mode == 0
                else w.float()
                @ (raw.half().float().relu() if mode == 2 else raw.half().float())
            )
            if mode == 0:
                close = actual == expected
            else:
                close = torch.isclose(actual, expected, rtol=1e-3, atol=1e-3)
            entry.update(
                mismatches=int((~close).sum()),
                max_abs_error=float((actual.float() - expected.float()).abs().max()),
            )
            if not bool(close.all()):
                entry["status"] = "precision_failed"
                result.update(
                    status="completed",
                    outcome=(
                        "blocked_launch_control" if mode == 0 else "rejected_precision"
                    ),
                )
                return result
            entry.update(status="passed", phase="complete")
            compiled_functions[label] = fn
            print("EXPLICIT_CUBE_CHECK " + json.dumps(entry), flush=True)
        # Only the fully proven ReLU/L1 chain reaches paired micro timing.
        ref_out = torch.empty((16, 64), device=device, dtype=torch.float32)
        reference = partial(module.reference[(1,)], *values, ref_out, M=16, K=128, N=64)
        capture = DumpCompiler(report, "tensor-reference")
        with patch.object(compiler, "subprocess", capture):
            reference()
        result["reference_compiler_calls"] = capture.calls
        torch.testing.assert_close(
            ref_out.cpu(), w.float() @ raw.half().float().relu(), rtol=1e-3, atol=1e-3
        )
        funcs = dict(reference=reference, explicit=compiled_functions["relu_half_l1"])
        samples = {name: [] for name in funcs}
        for iteration in range(2):
            for name in (tuple(funcs) if iteration == 0 else tuple(reversed(funcs))):
                ms = float(timer.get_latency(funcs[name]))
                if not math.isfinite(ms) or ms <= 0:
                    raise RuntimeError("Invalid micro timing")
                samples[name].append(ms)
        result["timings"] = {
            name: dict(samples_ms=times, median_us=statistics.median(times) * 1000)
            for name, times in samples.items()
        }
        result.update(
            status="completed",
            outcome="passed",
            speedup=statistics.median(samples["reference"])
            / statistics.median(samples["explicit"]),
        )
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
        print("EXPLICIT_CUBE_RESULT " + json.dumps(result), flush=True)
    return result


def summarize_probe(directory, output, pytest_exit):
    error = None
    record = {}
    try:
        record = json.loads((directory / "explicit-cube-handoff.json").read_text())
        if pytest_exit or record["status"] != "completed":
            raise ValueError(f"Incomplete probe, pytest exit={pytest_exit}")
        if record["outcome"] == "passed":
            if len(record["stages"]) != 3 or not all(
                e["status"] == "passed" and e["ir"]["passed"] for e in record["stages"]
            ):
                raise ValueError("Incomplete stage gates")
            if set(record["timings"]) != {"reference", "explicit"}:
                raise ValueError("Missing paired micro timing")
            for entry in record["timings"].values():
                if len(entry["samples_ms"]) != 2 or not all(
                    math.isfinite(t) and t > 0 for t in entry["samples_ms"]
                ):
                    raise ValueError("Invalid timing")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        error = str(exc)
    result = dict(
        exit_code=int(error is not None),
        error=error,
        probe=record,
        pipeline_proven=error is None and record.get("outcome") == "passed",
        arithmetic_mean_speedup=None,
    )
    output.write_text(json.dumps(result, indent=2))
    print("EXPLICIT_CUBE_SUMMARY " + json.dumps(result), flush=True)
    return result["exit_code"]
