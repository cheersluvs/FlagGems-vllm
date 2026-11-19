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

"""Measure exposed local sort latency with exact pairs captured from the control."""

import ast
import json
import math
import re
from functools import partial
from statistics import mean


def source(text):
    fn = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "raw_history_prepared_paired_reuse_local32"
    )
    fn.name = "raw_paired_presort_diagnostic"
    trace = next(i for i, arg in enumerate(fn.args.args) if arg.arg == "Trace")
    fn.args.args.insert(trace + 1, ast.arg(arg="Staged"))
    fn.args.args.append(
        ast.arg(arg="ARM", annotation=ast.parse("tl.constexpr", mode="eval").body)
    )
    loop = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.For) and n.target.id == "update_query"
    )
    start = next(
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "local32"
    )
    sort = loop.body[start : start + 3]
    assert [ast.literal_eval(n.value.args[0]) for n in sort] == [
        "sort32",
        "mrgsort",
        "mrgsort",
    ]
    code = ast.parse(
        """
pair_addr = Staged + (update_row.to(tl.int64) * BLOCKS + vector_block) * 1024 + word
if ARM != 3:
    pass
if ARM == 1:
    tl.debug_barrier()
    tl.store(pair_addr, pairs.to(tl.int32, bitcast=True))
elif ARM == 2:
    loaded = tl.load(pair_addr)
    pairs = tl.where(word < (visible % 1024), pairs.to(tl.int32, bitcast=True), loaded).to(tl.float32, bitcast=True)
elif ARM == 3:
    pairs = tl.load(pair_addr).to(tl.float32, bitcast=True)
"""
    ).body
    code[1].body = sort
    loop.body[start : start + 3] = code
    return "\n" + ast.unparse(ast.fix_missing_locations(fn)) + "\n"


def run(torch, module, mixed, benchmark, report):
    # TRACE=True in all four diagnostic arms keeps the ring read, key scale,
    # mask and score computation observable even in the no-sort variant.
    if any(r["mode"] != "paired_reuse_local32" for r in mixed.dispatch_routes):
        raise ValueError("Expected out=local32 on every piece")
    parts = []
    sizes = [r["groups"] * 16 * r["blocks"] * 1024 for r in mixed.routes]
    storage = torch.empty(sum(sizes), dtype=torch.int32, device=mixed.output.device)
    for job, route, staged in zip(
        mixed.jobs, mixed.routes, storage.split(sizes), strict=True
    ):
        parts.append((job, (route["groups"],), (*job["mixed_operands"], staged)))
    modes = dict(scored_trace=0, write_pairs=1, sort_and_read=2, replay=3)
    status = dict(
        status="running",
        rows=mixed.rows,
        pieces=len(parts),
        pair_bytes_per_full_read_or_write=storage.numel() * storage.element_size(),
        checks={},
        rounds=[],
        compiled={},
    )
    path = report / "int8_long100k-presort-replay.json"

    def save():
        path.write_text(json.dumps(status, indent=2))

    def execute(name, inspect=False):
        for j, (job, grid, operands) in enumerate(parts):
            job["pack_qw"]()
            job["pack_k"]()
            k = module.raw_paired_presort_diagnostic[grid](
                *operands, **job["mixed_meta"], TRACE=True, ARM=modes[name]
            )
            if inspect and j == 0:
                ir = k.asm["bcmlir"]
                if isinstance(ir, bytes):
                    ir = ir.decode()
                (report / (name + "-mixed-bcmlir.txt")).write_text(ir)
                ttir = k.asm["ttir"]
                if isinstance(ttir, bytes):
                    ttir = ttir.decode()
                (report / (name + "-mixed-ttir.txt")).write_text(ttir)
                loads = re.findall(r"tt\.load [^\n]*tensor<1024x!tt.ptr<i32>>", ttir)
                stores = re.findall(r"tt\.store [^\n]*tensor<1024x!tt.ptr<i32>>", ttir)
                custom = [line for line in ir.splitlines() if "hivm.hir.custom" in line]
                counts = {
                    op: sum('"' + op + '"' in line for line in custom)
                    for op in ("sort32", "mrgsort", "merge_exhaust_sort4")
                }
                counts.update(
                    pair_loads=len(loads),
                    pair_stores=len(stores),
                    ring_read=bool(
                        re.search(r"tt\.load [^\n]*tensor<8x64x!tt.ptr<f32>>", ttir)
                    ),
                    score_write=len(
                        re.findall(r"tt\.store [^\n]*tensor<512x!tt.ptr<f32>>", ttir)
                    )
                    >= 2,
                    cube_call=any(
                        "qsa_raw_block512_mixed_query16" in line for line in custom
                    ),
                    sync_set="sync_block_set" in ir,
                    sync_wait="sync_block_wait" in ir,
                )
                status["compiled"][name] = counts
                save()
                if (
                    bool(loads) != (name in ("sort_and_read", "replay"))
                    or bool(stores) != (name == "write_pairs")
                    or not all(
                        counts[k]
                        for k in (
                            "cube_call",
                            "sync_set",
                            "sync_wait",
                            "ring_read",
                            "score_write",
                        )
                    )
                ):
                    raise RuntimeError(f"{name}: diagnostic I/O mismatch: {counts}")
                expected_sort = int(name != "replay")
                if (
                    counts["sort32"],
                    counts["mrgsort"],
                    counts["merge_exhaust_sort4"],
                ) != (expected_sort, 2 * expected_sort, 2):
                    raise RuntimeError(f"{name}: unexpected mixed IR: {counts}")
        return mixed.output

    save()
    reference = mixed().cpu()
    calls = {"normal": mixed, **{n: partial(execute, n) for n in modes}}

    def check(phase):
        # The writer initializes every staged pair before either reader runs.
        for name in modes:
            mixed.output.fill_(-3)
            mixed.reset_guards()
            actual = execute(name, inspect=phase == "before").cpu()
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            mixed.check_guards()
            status["checks"][phase + "_" + name] = True
            save()

    check("before")
    for order in (list(calls), list(reversed(calls))):
        ms = {}
        status["rounds"].append(dict(order=order, latency_ms=ms))
        for name in order:
            ms[name] = float(benchmark.get_latency(calls[name]))
            if not math.isfinite(ms[name]) or ms[name] <= 0:
                raise ValueError(f"Invalid {name} timing: {ms[name]}")
            save()
            print("PRESORT_TIMING", name, ms[name], "ms", flush=True)
        deltas = dict(
            score_trace_ms=ms["scored_trace"] - ms["normal"],
            pair_write_ms=ms["write_pairs"] - ms["scored_trace"],
            pair_read_and_select_ms=ms["sort_and_read"] - ms["scored_trace"],
            net_replay_saving_ms=ms["scored_trace"] - ms["replay"],
            matched_sort_and_select_delta_ms=ms["sort_and_read"] - ms["replay"],
        )
        status["rounds"][-1]["deltas_ms"] = deltas
        print("PRESORT_DELTAS", json.dumps(deltas), flush=True)
    check("after")
    status.update(
        status="passed",
        mean_ms={
            name: mean(r["latency_ms"][name] for r in status["rounds"])
            for name in calls
        },
    )
    save()
    print("PRESORT_RESULT", json.dumps(status), flush=True)
    return status
