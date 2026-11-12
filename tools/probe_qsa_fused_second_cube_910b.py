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

"""Compare in-kernel QK plus Vector or second-Cube reduction."""

import hashlib
import json
import math
import statistics
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_reduce(Q, K, QS, KS, W, Table, Visible, Out,
               ROWS: tl.constexpr, WIDTH: tl.constexpr, COLUMNS: tl.constexpr,
               SECOND_CUBE: tl.constexpr):
    base = tl.program_id(0) * 4
    start = tl.program_id(1) * 64
    n = start + tl.arange(0, 64)
    m = tl.arange(0, 16)
    d = tl.arange(0, 128)
    qrow = base + m // 4
    q = tl.load(Q + (qrow[:, None] * 4 + m[:, None] % 4) * 128 + d[None, :],
                mask=qrow[:, None] < ROWS, other=0)
    page = tl.load(Table + tl.minimum(start // 192, COLUMNS - 1))
    token = page * 192 + start % 192 + tl.arange(0, 64)
    key = tl.load(K + token[:, None] * 128 + d[None, :], mask=n[:, None] < WIDTH, other=0)
    dot = tl.dot(q, tl.trans(key), out_dtype=tl.int32)
    ks = tl.load(KS + token, mask=n < WIDTH, other=0).to(tl.float32)
    if SECOND_CUBE:
        r = base + tl.arange(0, 16)
        qs = tl.load(QS + r[:, None] * 4 + m[None, :] % 4,
                     mask=r[:, None] < tl.minimum(base + 4, ROWS), other=0).to(tl.float32)
        w = tl.load(W + r[:, None] * 4 + m[None, :] % 4,
                    mask=r[:, None] < tl.minimum(base + 4, ROWS), other=0).to(tl.float32)
        weights = tl.where(tl.arange(0, 16)[:, None] == m[None, :] // 4, qs * w, 0.0).to(tl.float16)
        positive = (tl.maximum(dot.to(tl.float32), 0.0) * (1.0 / 1024.0)).to(tl.float16)
        total = tl.dot(weights, positive, out_dtype=tl.float32)
        visible = tl.load(Visible + r, mask=r < tl.minimum(base + 4, ROWS), other=0)
        score = tl.where(n[None, :] < visible[:, None], total * 1024.0 * ks[None, :], -float("inf"))
        tl.store(Out + r[:, None] * WIDTH + n[None, :], score,
                 mask=(r[:, None] < tl.minimum(base + 4, ROWS)) & (n[None, :] < WIDTH))
    else:
        r = base + tl.arange(0, 4)
        h = tl.arange(0, 4)
        qs = tl.load(QS + r[:, None] * 4 + h[None, :], mask=r[:, None] < ROWS, other=0).to(tl.float32)
        w = tl.load(W + r[:, None] * 4 + h[None, :], mask=r[:, None] < ROWS, other=0).to(tl.float32)
        products = tl.reshape(dot.to(tl.float32), (4, 4, 64))
        dequant = products * qs[:, :, None] * ks[None, None, :]
        total = tl.sum(tl.maximum(dequant, 0.0) * w[:, :, None], 1)
        visible = tl.load(Visible + r, mask=r < ROWS, other=0)
        score = tl.where(n[None, :] < visible[:, None], total, -float("inf"))
        tl.store(Out + r[:, None] * WIDTH + n[None, :], score, mask=(r[:, None] < ROWS) & (n[None, :] < WIDTH))


def run_fused_probe(inputs, values, refs, width, timer, report, case):
    from triton.backends.ascend import compiler
    from diagnose_qsa_structural_mixed_compile_910b import CaptureCompiler

    report = Path(report)
    path = report / f"{case}-fused-second-cube.json"
    result = dict(status="running", scope="single-launch scorer QK plus reduction; no TopK or full-shape acceptance",
                  query_group=4, cube_n=64, common_gm_qk_input=False,
                  compiler_gm_intermediate_status="unverified; inspect captured artifacts", arms={})
    funcs = {}
    try:
        assert bool((inputs["row_table"] == inputs["row_table"][0]).all())
        assert bool(torch.isfinite(inputs["qs"]).all() and (inputs["qs"] > 0).all())
        assert bool(torch.isfinite(inputs["ks"]).all() and (inputs["ks"] > 0).all())
        rows = inputs["q"].shape[0]
        for cube, name in ((False, "qk_vector_fp32"), (True, "qk_second_cube")):
            entry = dict(status="running", checks=[], samples_ms=[])
            result["arms"][name] = entry
            for fixture, count, scan in (("full", rows, width), ("tail_visible", 3, min(width, 193))):
                visible_cpu = inputs["visible"][:count].clone().clamp(max=scan)
                if fixture == "tail_visible":
                    visible_cpu[0] = 0
                    visible_cpu[1] = min(int(visible_cpu[1]), 191)
                visible = visible_cpu.to(values["q"].device)
                output = torch.empty((count, scan), device=values["q"].device, dtype=torch.float32)
                fn = partial(_qk_reduce[(triton.cdiv(count, 4), triton.cdiv(scan, 64))],
                    values["q"], values["cache"], values["qs"], values["ks"], values["weights"],
                    values["row_table"][0], visible, output,
                    ROWS=count, WIDTH=scan, COLUMNS=values["row_table"].shape[1], SECOND_CUBE=cube, use_bytecode=True)
                capture = CaptureCompiler(report, f"{case}-{name}-{fixture}")
                with patch.object(compiler, "subprocess", capture):
                    compiled = fn()
                actual = output.cpu()
                expected = torch.full((count, scan), -float("inf"))
                for row in range(count):
                    live = min(int(visible_cpu[row]), refs[row].numel(), scan)
                    expected[row, :live] = refs[row][:live]
                finite = torch.isfinite(expected)
                close = torch.isclose(actual, expected, rtol=1e-3, atol=1e-3)
                error = float((actual[finite] - expected[finite]).abs().max()) if bool(finite.any()) else 0.0
                check = dict(fixture=fixture, rows=count, width=scan, mismatches=int((~close).sum()),
                    max_abs_error=error if math.isfinite(error) else None,
                    invalid_output=int((~torch.isfinite(actual[finite])).sum()), compiler_calls=capture.calls)
                entry["checks"].append(check)
                entry.setdefault("artifacts", {})[fixture] = {}
                for kind, data in compiled.asm.items():
                    if isinstance(data, str):
                        filename = f"{case}-{name}-{fixture}-{kind}.txt"
                        (report / filename).write_text(data)
                        entry["artifacts"][fixture][kind] = dict(file=filename,
                            linalg_matmul_count=data.count("linalg.matmul"), triton_dot_count=data.count("tt.dot"))
                    elif isinstance(data, bytes):
                        entry["artifacts"][fixture][kind] = dict(sha256=hashlib.sha256(data).hexdigest())
                if fixture == "full":
                    funcs[name] = fn
                print("FUSED_SECOND_CUBE_CHECK " + json.dumps(dict(arm=name, **check)), flush=True)
                path.write_text(json.dumps(result, indent=2))
            entry["status"] = "passed" if all(c["mismatches"] == 0 for c in entry["checks"]) else "rejected_precision"
            if entry["status"] != "passed":
                funcs.pop(name, None)
        assert result["arms"]["qk_vector_fp32"]["status"] == "passed", "Fused FP32 control mismatch"
        for iteration in range(2):
            names = tuple(funcs) if iteration == 0 else tuple(reversed(funcs))
            for name in names:
                latency = float(timer.get_latency(funcs[name]))
                if not math.isfinite(latency) or latency <= 0:
                    raise RuntimeError(f"Invalid latency: {name} {latency}")
                result["arms"][name]["samples_ms"].append(latency)
        base = statistics.median(result["arms"]["qk_vector_fp32"]["samples_ms"])
        for name in funcs:
            e = result["arms"][name]
            e.update(median_us=statistics.median(e["samples_ms"]) * 1000,
                     speedup_over_fused_vector=base / statistics.median(e["samples_ms"]))
            print("FUSED_SECOND_CUBE_TIME " + json.dumps(dict(arm=name, **e)), flush=True)
        result["status"] = "completed" if len(funcs) == 2 else "completed_with_precision_rejection"
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
