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

"""Three-shape BF16 stage baseline, using unchanged production and native ops."""

import json
import math
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path

CASES = {
    "short16_g1024": (1, 16, 1024),
    "medium32_g8192": (2, 16, 8192),
    "long32_g25600": (2, 16, 25600),
}


def profiler_inventory():
    paths = [shutil.which("msprof")]
    base = Path("/usr/local/Ascend/ascend-toolkit/latest/tools/profiler")
    paths.extend(str(p) for p in (base / "bin/msprof", base / "msprof") if p.is_file())
    tool = next((path for path in paths if path), None)
    result = dict(msprof=tool, collected=False)
    if tool:
        try:
            proc = subprocess.run(
                [tool, "--help"], capture_output=True, text=True, timeout=10
            )
            result.update(
                help_exit_code=proc.returncode, help=(proc.stdout + proc.stderr)[:16000]
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["error"] = str(exc)
    return result


def device_inventory():
    try:
        proc = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=10
        )
        return dict(exit_code=proc.returncode, text=(proc.stdout + proc.stderr)[:32000])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(error=str(exc))


def sample(fn, torch, iterations, verify_return=None):
    torch.npu.synchronize()
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    # Prime lazy event handles before measuring the host interval.
    start.record()
    end.record()
    end.synchronize()
    begin = time.perf_counter()
    start.record()
    returned = None
    for _ in range(iterations):
        returned = fn()
    end.record()
    end.synchronize()
    wall = (time.perf_counter() - begin) * 1e6 / iterations
    device = start.elapsed_time(end) * 1000 / iterations
    if not math.isfinite(wall) or not math.isfinite(device) or wall <= 0 or device <= 0:
        raise RuntimeError(f"Invalid timing: wall={wall}, device={device}")
    # Checks run after both time measurements, never inside the timed interval.
    if verify_return is not None:
        verify_return(returned)
    # Keep returned native tensors alive through the final event synchronization.
    del returned
    return dict(wall_us=wall, event_us=device)


def aggregate(samples):
    result = {}
    for field in ("wall_us", "event_us"):
        values = [s[field] for s in samples]
        median = statistics.median(values)
        result[field] = dict(
            median=median,
            min=min(values),
            max=max(values),
            range_over_median=(max(values) - min(values)) / median,
        )
    return result


def run(args, result):
    import torch
    import torch_npu
    import triton
    from qsa_lightning_baseline_910b import qsa_select_paged_prefill as adapter
    from qsa_prefill_production_910b import load_production

    public, module, topk_module, identity = load_production()
    result["npu_smi_before"] = device_inventory()
    result["production_entry"] = identity
    os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = "512"
    os.environ["FLAGGEMS_ASCEND_PTK_PATH"] = "auto"
    requests, query_rows, groups = CASES[args.worker]
    rows, heads, dim, page, k, ratio = requests * query_rows, 4, 128, 192, 512, 4
    columns = triton.cdiv(groups, page)
    capacity = columns * page
    width = min(capacity, triton.cdiv(groups, 64) * 64)
    device = f"npu:{args.device}"
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(20261029)
    q_cpu = torch.randn((rows, heads, dim), dtype=torch.bfloat16)
    key_cpu = torch.randn((requests * columns, page, 1, dim), dtype=torch.bfloat16)
    table_cpu = (
        torch.randperm(requests * columns).to(torch.int32).reshape(requests, columns)
    )
    starts_cpu = torch.arange(requests + 1, dtype=torch.int32) * query_rows + 137
    visible_cpu = torch.tensor(
        [
            groups - query_rows // ratio + (row + 1) // ratio
            for _ in range(requests)
            for row in range(query_rows)
        ],
        dtype=torch.int32,
    )
    q, cache, table, starts, visible = (
        t.to(device) for t in (q_cpu, key_cpu, table_cpu, starts_cpu, visible_cpu)
    )
    output = torch.empty((rows, k), device=device, dtype=torch.int32)
    native_output = torch.empty_like(output)
    operands = (
        q,
        cache,
        table,
        starts,
        visible,
        k * ratio,
        ratio,
        query_rows,
        output,
        groups * ratio,
    )
    native_operands = (*operands[:8], native_output, operands[9])
    owners = module._build_owners(q, table, starts)
    assert owners.cpu().tolist() == [row // query_rows for row in range(rows)]
    chunk_rows = min(rows, 16)
    logits = torch.empty((chunk_rows, width), device=device, dtype=torch.float32)
    frozen = torch.empty((rows, width), device=device, dtype=torch.float32)
    workspace = torch.empty((1024 * 1024,), device=device, dtype=torch.uint8)
    chunks = [
        (first, min(chunk_rows, rows - first)) for first in range(0, rows, chunk_rows)
    ]

    score_settings = dict(programs=8)

    def score_into(first, count, target):
        return module._score(
            q,
            cache,
            table,
            starts,
            visible,
            target,
            offset=first,
            owners=owners,
            programs=score_settings["programs"],
        )

    for first, count in chunks:
        score_into(first, count, frozen[first : first + count])
    reference = torch.full((rows, width), -torch.inf)
    for row, n in enumerate(visible_cpu.tolist()):
        positions = torch.arange(n)
        keys = key_cpu[
            table_cpu[row // query_rows, positions // page].long(), positions % page, 0
        ].float()
        reference[row, :n] = (
            torch.einsum("hd,kd->hk", q_cpu[row].float(), keys).clamp_min(0).sum(0)
        )
    actual = frozen.cpu()
    finite = torch.isfinite(reference)
    torch.testing.assert_close(actual[finite], reference[finite], rtol=1e-3, atol=1e-3)
    assert torch.isneginf(actual[~finite]).all()
    result["score_correctness"] = dict(
        max_abs_error=float((actual[finite] - reference[finite]).abs().max()),
        valid_scores=int(finite.sum()),
        masked_scores=int((~finite).sum()),
        rtol=1e-3,
        atol=1e-3,
    )

    def verify(indices):
        indices = indices.cpu()
        max_error = 0.0
        for row, n in enumerate(visible_cpu.tolist()):
            selected = indices[row].long()
            assert (
                (selected >= 0) & (selected < n)
            ).all() and selected.unique().numel() == k
            values = reference[row, selected].sort().values
            expected = reference[row, :n].topk(k).values.sort().values
            torch.testing.assert_close(values, expected, rtol=1e-3, atol=1e-3)
            max_error = max(max_error, float((values - expected).abs().max()))
        return dict(
            rows=rows, max_selected_score_error=max_error, ranges_and_uniqueness=True
        )

    row_table = table_cpu.repeat_interleave(query_rows, dim=0).to(device)
    cu_query = torch.arange(1, rows + 1, device=device, dtype=torch.int32)
    weights = torch.full((rows, heads), dim**-0.5, device=device, dtype=torch.float32)

    def native_fused():
        return torch_npu.npu_lightning_indexer(
            q,
            cache,
            weights,
            actual_seq_lengths_query=cu_query,
            actual_seq_lengths_key=visible,
            block_table=row_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=k,
            sparse_mode=0,
            pre_tokens=2**63 - 1,
            next_tokens=2**63 - 1,
            return_value=False,
        )

    def score_stage():
        for first, count in chunks:
            score_into(first, count, logits[:count])

    def topk_stage():
        for first, count in chunks:
            module.persistent_topk(
                frozen[first : first + count],
                visible[first : first + count],
                output[first : first + count],
                workspace,
                k=k,
                max_seq_len=width,
            )

    def prepared_pipeline():
        for first, count in chunks:
            score_into(first, count, logits[:count])
            module.persistent_topk(
                logits[:count],
                visible[first : first + count],
                output[first : first + count],
                workspace,
                k=k,
                max_seq_len=width,
            )

    def public_total():
        return public(*operands)

    def adapter_total():
        return adapter(*native_operands)

    def owner_stage():
        return module._build_owners(q, table, starts)

    functions = dict(
        score=score_stage,
        topk=topk_stage,
        prepared_pipeline=prepared_pipeline,
        native_fused=native_fused,
        public_total=public_total,
        native_adapter_total=adapter_total,
        request_mapping=owner_stage,
    )
    result["selection_correctness"] = {}
    for name in ("topk", "prepared_pipeline", "public_total", "native_adapter_total"):
        functions[name]()
        result["selection_correctness"][name] = verify(
            native_output if name == "native_adapter_total" else output
        )
    raw, _ = native_fused()
    assert raw.shape == (rows, 1, k)
    result["selection_correctness"]["native_fused"] = verify(raw[:, 0])
    result["shape"] = dict(
        query=list(q.shape),
        cache=list(cache.shape),
        page_table=list(table.shape),
        request_query_lengths=[query_rows] * requests,
        visible_min=int(visible_cpu.min()),
        visible_max=int(visible_cpu.max()),
        logits_width=width,
        page=page,
        group_topk=k,
        token_topk=k * ratio,
        compress_ratio=ratio,
        dtype="bfloat16",
        chunks=chunks,
    )
    result["timing_policy"] = dict(
        rounds=args.rounds,
        warmup=args.warmup,
        iterations=args.iterations,
        units="microseconds per invocation",
        order="alternating forward/reverse each round",
        event_scope="stream elapsed including possible host launch gaps; not sum of kernel durations",
        wall_scope="synchronized wall time including event instrumentation, amortized over iterations",
        prepared_scope="prebuilt metadata and reusable workspaces; direct score+TopK timing",
        public_scope="unchanged public API, includes request mapping and allocations",
        native_scope="fused kernel with prebuilt row table and weights; no separate score/TopK latency",
        adapter_scope="unchanged native adapter includes metadata construction and output normalization",
        stage_note=(
            "isolated TopK reads frozen scores; isolated stages are not additive "
            "or a kernel-mode acceptance metric"
        ),
        workload_scope="three structural diagnostic cases, not a replacement for the repository benchmark suite",
    )
    result["samples"] = {name: [] for name in functions}
    print("shape", json.dumps(result["shape"]), flush=True)
    print("score_correctness", json.dumps(result["score_correctness"]), flush=True)
    print("correctness", json.dumps(result["selection_correctness"]), flush=True)
    if getattr(args, "probe_kind", None):
        from qsa_independent_probes_910b import run as probe

        return probe(args, result, locals())
    for fn in functions.values():
        for _ in range(args.warmup):
            fn()
        torch.npu.synchronize()
    if getattr(args, "timeline", False):
        from qsa_timeline_910b import collect

        result["timing_policy"]["capture_only"] = True
        collect(args, functions, torch, result)
        public_total()
        verify(output)
        result["npu_smi_after"] = device_inventory()
        result["status"] = "passed"
        return 0
    for round_index in range(args.rounds):
        order = list(functions)
        if round_index % 2:
            order.reverse()
        for name in order:
            check_return = (
                (lambda returned: verify(returned[0][:, 0]))
                if name == "native_fused"
                else None
            )
            value = sample(functions[name], torch, args.iterations, check_return)
            if name in ("topk", "prepared_pipeline", "public_total"):
                verify(output)
            elif name == "native_adapter_total":
                verify(native_output)
            elif name == "score":
                first, count = chunks[-1]
                last = logits[:count].cpu()
                target = reference[first : first + count]
                mask = torch.isfinite(target)
                torch.testing.assert_close(
                    last[mask], target[mask], rtol=1e-3, atol=1e-3
                )
                assert torch.isneginf(last[~mask]).all()
            result["samples"][name].append(value)
            print(
                f"round={round_index+1} {name}: event={value['event_us']:.3f} us wall={value['wall_us']:.3f} us",
                flush=True,
            )
        public_total()
        verify(output)
    result["latency"] = {
        name: aggregate(values) for name, values in result["samples"].items()
    }
    result["speedup"] = {}
    for label, numerator, denominator in (
        ("prepared_native_over_gems", "native_fused", "prepared_pipeline"),
        ("public_adapter_over_gems", "native_adapter_total", "public_total"),
    ):
        result["speedup"][label] = {
            field: result["latency"][numerator][field]["median"]
            / result["latency"][denominator][field]["median"]
            for field in ("event_us", "wall_us")
        }
    result["route_states"] = dict(
        segmerge=dict(topk_module._SEGMERGE_STATE),
        persistent=dict(topk_module._PSEGMERGE_STATE),
    )
    print(
        "SUMMARY stage                       event_us      wall_us      event_range/median",
        flush=True,
    )
    for name, value in result["latency"].items():
        print(
            f"{name:30} {value['event_us']['median']:12.3f} "
            f"{value['wall_us']['median']:12.3f} "
            f"{value['event_us']['range_over_median']:12.2%}",
            flush=True,
        )
    print("SPEEDUP", json.dumps(result["speedup"]), flush=True)
    print("ROUTES", json.dumps(result["route_states"]), flush=True)
    result["npu_smi_after"] = device_inventory()
    result["status"] = "passed"
    return 0
