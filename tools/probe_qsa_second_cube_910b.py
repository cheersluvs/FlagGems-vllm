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

"""Isolate post-QK reduction, including conversion and operand construction."""

import json
import math
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_reduce(D, QS, KS, W, Visible, Out, ROWS: tl.constexpr, WIDTH: tl.constexpr, LATE: tl.constexpr):
    row = tl.program_id(0)
    n = tl.program_id(1) * 64 + tl.arange(0, 64)
    h = tl.arange(0, 4)
    d = tl.load(D + (row * 4 + h[:, None]) * WIDTH + n[None, :], mask=n[None, :] < WIDTH, other=0).to(tl.float32)
    qs = tl.load(QS + row * 4 + h).to(tl.float32)
    w = tl.load(W + row * 4 + h).to(tl.float32)
    ks = tl.load(KS + n, mask=n < WIDTH, other=0).to(tl.float32)
    if LATE:
        score = tl.sum(tl.maximum(d, 0.0) * (qs * w)[:, None], 0) * ks
    else:
        score = tl.sum(tl.maximum(d * qs[:, None] * ks[None, :], 0.0) * w[:, None], 0)
    visible = tl.load(Visible + row)
    tl.store(Out + row * WIDTH + n, tl.where(n < visible, score, -float("inf")), mask=n < WIDTH)


@triton.jit
def _cube_reduce(D, QS, KS, W, Visible, Out, ROWS: tl.constexpr, WIDTH: tl.constexpr):
    base = tl.program_id(0) * 4
    n = tl.program_id(1) * 64 + tl.arange(0, 64)
    m = tl.arange(0, 16)
    row = base + m // 4
    d = tl.load(D + (row[:, None] * 4 + m[:, None] % 4) * WIDTH + n[None, :],
                mask=(row[:, None] < ROWS) & (n[None, :] < WIDTH), other=0).to(tl.float32)
    positive = (tl.maximum(d, 0.0) * (1.0 / 1024.0)).to(tl.float16)
    r = base + tl.arange(0, 16)
    qs = tl.load(QS + r[:, None] * 4 + m[None, :] % 4, mask=r[:, None] < tl.minimum(base + 4, ROWS), other=0).to(tl.float32)
    w = tl.load(W + r[:, None] * 4 + m[None, :] % 4, mask=r[:, None] < tl.minimum(base + 4, ROWS), other=0).to(tl.float32)
    weights = tl.where(tl.arange(0, 16)[:, None] == m[None, :] // 4, qs * w, 0.0).to(tl.float16)
    total = tl.dot(weights, positive, out_dtype=tl.float32)
    ks = tl.load(KS + n, mask=n < WIDTH, other=0).to(tl.float32)
    visible = tl.load(Visible + r, mask=r < tl.minimum(base + 4, ROWS), other=0)
    score = tl.where(n[None, :] < visible[:, None], total * 1024.0 * ks[None, :], -float("inf"))
    tl.store(Out + r[:, None] * WIDTH + n[None, :], score,
             mask=(r[:, None] < tl.minimum(base + 4, ROWS)) & (n[None, :] < WIDTH))


def run_reduction_probe(inputs, values, refs, width, timer, report, case):
    report = Path(report)
    path = report / f"{case}-second-cube-reduction.json"
    result = dict(status="running", scope="post-QK reduction only; not full scorer or acceptance",
        qk_preparation_timed=False, conversion_and_weight_construction_timed=True,
        cube_operands="FP16 positive(dot/1024), FP16(query_scale*weight); FP32 accumulation and rescale",
        cann_equivalence="inspired by two-Cube structure; not a bitwise reproduction of CANN Fixpipe", arms={})
    try:
        rows = inputs["q"].shape[0]
        mapping = inputs["row_table"][0]
        assert bool((inputs["row_table"] == mapping).all()), "Shared key mapping required for this probe"
        assert bool(torch.isfinite(inputs["qs"]).all() and (inputs["qs"] > 0).all())
        columns = (width + 191) // 192
        keys = inputs["cache"][mapping[:columns].long(), :, 0].reshape(-1, 128)[:width].int()
        ks = inputs["ks"].reshape(-1)[(mapping[:columns].long()[:, None] * 192 + torch.arange(192)).flatten()[:width]]
        assert bool(torch.isfinite(ks).all() and (ks > 0).all())
        dots = (inputs["q"].reshape(-1, 128).int() @ keys.T).reshape(rows, 4, width).contiguous().to(values["q"].device)
        resolved_ks = ks.contiguous().to(values["q"].device)
        expected = torch.full((rows, width), -float("inf"))
        for row, ref in enumerate(refs):
            expected[row, :min(width, ref.numel())] = ref[:width]
        funcs = {}
        for name, late in (("vector_fp32", False), ("vector_late_key", True), ("cube_fp16", None)):
            output = torch.empty((rows, width), device=values["q"].device, dtype=torch.float32)
            args = (dots, values["qs"], resolved_ks, values["weights"], values["visible"], output)
            if late is None:
                from functools import partial
                fn = partial(_cube_reduce[(triton.cdiv(rows, 4), triton.cdiv(width, 64))], *args, ROWS=rows, WIDTH=width, use_bytecode=True)
            else:
                from functools import partial
                fn = partial(_vector_reduce[(rows, triton.cdiv(width, 64))], *args, ROWS=rows, WIDTH=width, LATE=late, use_bytecode=True)
            entry = dict(status="compiling", samples_ms=[])
            result["arms"][name] = entry
            path.write_text(json.dumps(result, indent=2))
            try:
                compiled = fn()
                actual = output.cpu()
                finite = torch.isfinite(expected)
                close = torch.isclose(actual, expected, rtol=1e-3, atol=1e-3)
                entry.update(mismatches=int((~close).sum()),
                    max_abs_error=float((actual[finite] - expected[finite]).abs().max()) if bool(finite.any()) else 0.0,
                    invalid_output=int((~torch.isfinite(actual[finite])).sum()))
                if not math.isfinite(entry["max_abs_error"]):
                    entry["max_abs_error"] = None
                entry["status"] = "passed" if bool(close.all()) else "rejected_precision"
                for kind, text in compiled.asm.items():
                    if isinstance(text, str):
                        (report / f"{case}-second-cube-{name}-{kind}.txt").write_text(text)
            except Exception as exc:
                entry.update(status="failed", error=repr(exc))
                raise
            if entry["status"] == "passed":
                funcs[name] = fn
            print("SECOND_CUBE_CHECK " + json.dumps(dict(arm=name, **entry)), flush=True)
            path.write_text(json.dumps(result, indent=2))
        assert result["arms"]["vector_fp32"]["status"] == "passed", "FP32 control mismatch"
        for iteration in range(2):
            names = tuple(funcs) if iteration == 0 else tuple(reversed(funcs))
            for name in names:
                value = float(timer.get_latency(funcs[name]))
                if not math.isfinite(value) or value <= 0:
                    raise RuntimeError(f"Invalid kernel latency: {name} {value}")
                result["arms"][name]["samples_ms"].append(value)
        baseline = statistics.median(result["arms"]["vector_fp32"]["samples_ms"])
        for name in funcs:
            entry = result["arms"][name]
            entry["median_us"] = statistics.median(entry["samples_ms"]) * 1000
            entry["speedup_over_vector"] = baseline * 1000 / entry["median_us"]
            print("SECOND_CUBE_TIME " + json.dumps(dict(arm=name, **entry)), flush=True)
        result["status"] = "completed" if len(funcs) == 3 else "completed_with_precision_rejection"
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
