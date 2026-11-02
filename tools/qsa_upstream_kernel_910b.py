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

"""Upstream 12-query/prefix grid on the validated BF16/page192 Ascend domain."""

import json
import os

from qsa_prefill_upstream_workloads import SOURCE_URL, cases, geometry

CASES = {c["name"]: dict(c, page_size=192) for c in cases()}


def run(args, result):
    import torch
    import torch_npu
    from benchmark_qsa_prefill_stages_910b import device_inventory
    from qsa_kernel_benchmark_910b import aggregate, setup
    from qsa_lightning_baseline_910b import qsa_select_paged_prefill as adapter
    from qsa_prefill_production_910b import load_production

    public, module, topk_module, identity = load_production()
    case = CASES[args.worker]
    g = geometry(case)
    result["production_entry"] = identity
    result["npu_smi_before"] = device_inventory()
    os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = "512"
    os.environ["FLAGGEMS_ASCEND_PTK_PATH"] = "auto"
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(20261029)
    rows, heads, dim = g["q_shape"]
    width = g["logits_width"]
    n = case["requests"]
    queries = case["query_length"]
    page = 192
    k = 512
    ratio = 4
    device = f"npu:{args.device}"
    q_cpu = torch.randn(g["q_shape"], dtype=torch.bfloat16)
    key_cpu = torch.randn(g["cache_shape"], dtype=torch.bfloat16)
    table_cpu = (
        torch.randperm(g["physical_pages"])
        .to(torch.int32)
        .reshape(g["page_table_shape"])
    )
    starts_cpu = torch.arange(n + 1, dtype=torch.int32) * queries + 137
    visible_cpu = torch.tensor(
        [
            (case["prefix_tokens"] + i + 1) // ratio
            for _ in range(n)
            for i in range(queries)
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
        queries,
        output,
        case["prefix_tokens"] + queries,
    )
    owners = module._build_owners(q, table, starts)
    assert owners.cpu().tolist() == [i // queries for i in range(rows)]
    cap = module._prefill_chunk_cap(q, cache, table, starts, width, k)
    chunk_rows = min(rows, cap, max(1, 512 * 1024**2 // (width * 4)))
    chunks = [
        (first, min(chunk_rows, rows - first)) for first in range(0, rows, chunk_rows)
    ]
    logits = torch.empty((chunk_rows, width), device=device, dtype=torch.float32)
    # Benchmark-only staging for isolated TopK. Production still uses bounded chunks.
    frozen = torch.empty((rows, width), device=device, dtype=torch.float32)
    workspace = torch.empty((1024 * 1024,), device=device, dtype=torch.uint8)

    def score_into(first, count, target):
        return module._score(
            q, cache, table, starts, visible, target, offset=first, owners=owners
        )

    print(
        "UPSTREAM_PREPARATION",
        args.worker,
        "rows",
        rows,
        "width",
        width,
        "production_chunks",
        len(chunks),
        "staging_MiB",
        rows * width * 4 / 1024**2,
        flush=True,
    )
    # OFFSET is constexpr in the current production kernel; each first offset
    # may need its own initial JIT compilation. Keep the production path intact.
    for first, count in chunks:
        score_into(first, count, frozen[first : first + count])
    sample_rows = sorted(
        {0, 15, 16, 31, 32, rows // 2, rows - 1}
        | {r * queries for r in range(n)}
        | {(r + 1) * queries - 1 for r in range(n)}
    )
    references = {}
    maximum = 0.0
    for row in sample_rows:
        length = int(visible_cpu[row])
        positions = torch.arange(length)
        keys = key_cpu[
            table_cpu[row // queries, positions // page].long(), positions % page, 0
        ].float()
        expected = (
            torch.einsum("hd,kd->hk", q_cpu[row].float(), keys).clamp_min(0).sum(0)
        )
        actual = frozen[row].cpu()
        torch.testing.assert_close(actual[:length], expected, rtol=1e-3, atol=1e-3)
        assert torch.isneginf(actual[length:]).all()
        maximum = max(maximum, float((actual[:length] - expected).abs().max()))
        references[row] = expected

    def verify(indices):
        actual = indices.cpu()
        assert actual.shape == (rows, k) and actual.dtype == torch.int32
        assert ((actual >= 0) & (actual < visible_cpu[:, None])).all()
        ordered = actual.sort(dim=1).values
        assert (ordered[:, 1:] != ordered[:, :-1]).all()
        error = 0.0
        for row in sample_rows:
            selected = references[row][actual[row].long()].sort().values
            expected = references[row].topk(k).values.sort().values
            torch.testing.assert_close(selected, expected, rtol=1e-3, atol=1e-3)
            error = max(error, float((selected - expected).abs().max()))
        return dict(
            all_rows_range_and_uniqueness=True,
            rows=rows,
            cpu_topk_rows=sample_rows,
            max_selected_score_error=error,
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
                block=module._QSA_TOPK_BLOCK,
            )

    def prepared():
        for first, count in chunks:
            score_into(first, count, logits[:count])
            module.persistent_topk(
                logits[:count],
                visible[first : first + count],
                output[first : first + count],
                workspace,
                k=k,
                max_seq_len=width,
                block=module._QSA_TOPK_BLOCK,
            )

    def production():
        return public(*operands)

    def native_adapter():
        return adapter(*operands[:8], native_output, operands[9])

    row_table = table_cpu.repeat_interleave(queries, dim=0).to(device)
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

    jobs = dict(
        score=score_stage,
        topk=topk_stage,
        prepared_pipeline=prepared,
        public_total=production,
        native_adapter_total=native_adapter,
        native_fused=native_fused,
    )
    image_baseline = getattr(args, "image_triton_baseline", False)
    image_identity, image_output = None, None
    if image_baseline:
        from qsa_image_triton_baseline_910b import build

        positions = torch.tensor(
            [case["prefix_tokens"] + i for _ in range(n) for i in range(queries)],
            device=device,
            dtype=torch.int64,
        )
        sequence_lengths = torch.full(
            (n,), case["prefix_tokens"] + queries, device=device, dtype=torch.int32
        )
        image_jobs, image_output, image_identity = build(
            args,
            dict(
                q=q,
                cache=cache,
                table=table,
                owners=owners,
                positions=positions,
                sequence_lengths=sequence_lengths,
                rows=rows,
                k=k,
                starts=starts,
                max_seq_len=case["prefix_tokens"] + queries,
                visible_cpu=visible_cpu,
                references=references,
            ),
        )
        jobs = {
            name: jobs[name]
            for name in ("score", "topk", "prepared_pipeline", "public_total")
        }
        jobs.update(image_jobs)
    probe = result["upstream12_kernel"] = dict(
        source=SOURCE_URL,
        upstream_case=case,
        geometry=g,
        scope="Upstream query/prefix matrix; BF16 H4 D128 K512 ratio4; page196 rebuilt as page192 for Ascend",
        production_registered_entry=True,
        production_source_modified=False,
        dispatch=dict(
            block_n=module._score_block_n(q, cache, table, width),
            chunk_rows=chunk_rows,
            num_chunks=len(chunks),
        ),
        validation_scope="All output rows: range/uniqueness; representative rows only: CPU FP32 scores/TopK",
        sampled_score_error=maximum,
        cpu_reference_rows=sample_rows,
        benchmark_only_full_score_staging_bytes=rows * width * 4,
        correctness={},
        samples={name: [] for name in jobs},
    )

    probe["baseline"] = (
        "image_triton_torch_topk" if image_baseline else "CANN_LightningIndexer"
    )
    if image_baseline:
        probe["image_source"] = image_identity

    def check(name, value):
        if name.startswith("image_"):
            if name == "image_score":
                actual = value[0][-1].cpu()
                expected = references[rows - 1] / (dim**0.5)
                torch.testing.assert_close(
                    actual[: len(expected)], expected, rtol=1e-3, atol=1e-3
                )
                assert torch.isneginf(actual[len(expected) :]).all()
                return dict(initial_sampled_rows=sample_rows, postcheck_last_row=True)
            return verify(image_output)
        if name == "native_fused":
            return verify(value[0][:, 0])
        if name == "native_adapter_total":
            return verify(native_output)
        if name == "score":
            # Recheck all sampled score rows through the production score entry.
            for row in sample_rows:
                first = row // chunk_rows * chunk_rows
                count = min(chunk_rows, rows - first)
                score_into(first, count, logits[:count])
                length = len(references[row])
                actual = logits[row - first].cpu()
                torch.testing.assert_close(
                    actual[:length], references[row], rtol=1e-3, atol=1e-3
                )
                assert torch.isneginf(actual[length:]).all()
            return dict(sampled_scores_valid=True, rows=sample_rows)
        return verify(output)

    for name, function in jobs.items():
        value = function()
        torch.npu.synchronize()
        probe["correctness"][name] = check(name, value)
    print("UPSTREAM_SHAPE", json.dumps(probe["geometry"]), flush=True)
    print("UPSTREAM_DISPATCH", json.dumps(probe["dispatch"]), flush=True)
    print("UPSTREAM_VALIDATION", json.dumps(probe["correctness"]), flush=True)
    measure = setup(
        args, dict(chunks=chunks, score_into=score_into, logits=logits), torch, probe
    )
    for round_index in range(args.rounds):
        names = list(jobs)
        if round_index % 2:
            names.reverse()
        for name in names:
            value = measure(jobs[name])
            probe["samples"][name].append(value)
            probe["correctness"][name] = check(name, jobs[name]())
            print(
                "UPSTREAM_KERNEL_TIMING",
                round_index + 1,
                name,
                json.dumps(value),
                flush=True,
            )
    probe["latency"] = {
        name: aggregate(samples) for name, samples in probe["samples"].items()
    }
    if image_baseline:
        probe["speedup"] = {
            label: probe["latency"][a]["kernel_us"]["median"]
            / probe["latency"][b]["kernel_us"]["median"]
            for label, a, b in (
                ("image_score_over_production_score", "image_score", "score"),
                ("image_topk_over_production_topk", "image_topk", "topk"),
                (
                    "image_prepared_over_production_prepared",
                    "image_groups_prepared",
                    "prepared_pipeline",
                ),
                (
                    "image_adapter_over_production_public",
                    "image_groups_adapter",
                    "public_total",
                ),
            )
        }
    else:
        probe["speedup"] = dict(
            native_adapter_over_public=probe["latency"]["native_adapter_total"][
                "kernel_us"
            ]["median"]
            / probe["latency"]["public_total"]["kernel_us"]["median"],
            native_fused_over_prepared=probe["latency"]["native_fused"]["kernel_us"][
                "median"
            ]
            / probe["latency"]["prepared_pipeline"]["kernel_us"]["median"],
        )
    for name, value in probe["latency"].items():
        print("UPSTREAM_KERNEL_SUMMARY", name, json.dumps(value), flush=True)
    print("UPSTREAM_KERNEL_SPEEDUP", json.dumps(probe["speedup"]), flush=True)
    result["npu_smi_after"] = device_inventory()
    result["status"] = "passed"
    return 0
