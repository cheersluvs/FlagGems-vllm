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

"""Torch-free migration of ten frozen NV workloads to INT8/page192."""

from tools.qsa_prefill_fp8_workloads import cases as nv_cases


def cases():
    result = []
    for nv in nv_cases():
        if (nv["heads"], nv["dim"], nv["ratio"], nv["k"], nv["page_size"]) != (
            4,
            128,
            4,
            512,
            196,
        ):
            raise ValueError("Frozen NV workload contract changed; review migration")
        queries, prefixes = nv["query_lengths"], nv["prefix_tokens"]
        compressed = [(q + p + 3) // 4 for q, p in zip(queries, prefixes)]
        pages = [(length + 191) // 192 for length in compressed]
        columns = max(pages)
        width = min(columns * 192, ((nv["batch_max_seq_len"] + 3) // 4 + 63) // 64 * 64)
        rows = sum(queries)
        chunk = min(128, max(1, 64 * 1024**2 // (width * 4 * (nv["heads"] + 1))))
        starts = [0]
        for length in queries:
            starts.append(starts[-1] + length)
        result.append(
            dict(
                name=nv["name"]
                .replace("fp8_", "int8_", 1)
                .removesuffix("_float8_e4m3fn"),
                nv_case=nv["name"],
                origin=nv["origin"],
                query_lengths=list(queries),
                prefix_tokens=list(prefixes),
                starts=starts,
                batch_max_seq_len=nv["batch_max_seq_len"],
                heads=nv["heads"],
                dim=nv["dim"],
                ratio=nv["ratio"],
                k=nv["k"],
                dtype="int8",
                page_size=192,
                pages_per_request=pages,
                physical_pages=sum(pages),
                table_columns=columns,
                q_shape=[rows, 4, 128],
                cache_shape=[sum(pages), 192, 1, 128],
                logits_width=width,
                chunk_rows=chunk,
                score_and_dot_scratch_bytes=chunk * width * 4 * 5,
                nv_original_page_size=nv["page_size"],
                nv_original_physical_pages=nv.get("physical_pages"),
                nv_original_table_columns=nv.get("table_columns"),
                provenance="NV logical workload migration; synthetic INT8 tensors and minimal repacked storage",
                query_lengths_synthetic=nv.get("query_lengths_synthetic", False),
            )
        )
    return result


def sample_rows(case):
    rows, chunk = case["q_shape"][0], case["chunk_rows"]
    picks = {
        0,
        rows // 2,
        rows - 1,
        min(chunk - 1, rows - 1),
        min(chunk, rows - 1),
        (rows - 1) // chunk * chunk,
    }
    requests = len(case["query_lengths"])
    selected = (
        [0]
        if requests == 1
        else sorted(
            {
                round(i * (requests - 1) / (min(6, requests) - 1))
                for i in range(min(6, requests))
            }
        )
    )
    for request in selected:
        first, last = case["starts"][request : request + 2]
        picks.update((first, last - 1))
    for request in range(min(2, requests)):
        first, last = case["starts"][request : request + 2]
        crossing = (
            first + case["k"] * case["ratio"] - case["prefix_tokens"][request] - 1
        )
        picks.update(
            row for row in (crossing - 1, crossing, crossing + 1) if first <= row < last
        )
    assert len(picks) <= 24
    return sorted(picks)


def make_inputs(case):
    import torch

    gen = torch.Generator().manual_seed(20261103)

    def quantize(shape):
        values = torch.randn(shape, generator=gen)
        scale = (values.abs().amax(-1).clamp_min(1e-6) / 127).half()
        return (values / scale.float()[..., None]).round().clamp(-127, 127).to(
            torch.int8
        ), scale

    q, qs = quantize(case["q_shape"])
    cache, ks = quantize(case["cache_shape"])
    table = torch.full(
        (len(case["query_lengths"]), case["table_columns"]), -1, dtype=torch.int32
    )
    physical = torch.randperm(case["physical_pages"], generator=gen).to(torch.int32)
    cursor = 0
    for request, pages in enumerate(case["pages_per_request"]):
        table[request, :pages] = physical[cursor : cursor + pages]
        cursor += pages
    owners = torch.repeat_interleave(
        torch.arange(len(case["query_lengths"])), torch.tensor(case["query_lengths"])
    )
    visible = torch.tensor(
        [
            (prefix + local + 1) // case["ratio"]
            for length, prefix in zip(case["query_lengths"], case["prefix_tokens"])
            for local in range(length)
        ],
        dtype=torch.int32,
    )
    return dict(
        q=q,
        cache=cache,
        qs=qs,
        ks=ks,
        weights=torch.ones(q.shape[:2], dtype=torch.float16),
        row_table=table[owners].contiguous(),
        visible=visible,
    )
