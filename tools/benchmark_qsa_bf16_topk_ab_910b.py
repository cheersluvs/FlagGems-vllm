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


"""Pair old and new public BF16 QSA TopK paths with the Ascend kernel timer."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import statistics
import sys
from pathlib import Path

from qsa_kernel_benchmark_910b import setup
from qsa_prefill_production_910b import load_production

CASES = {"short16_g1024": (16, 1024), "medium32_g8192": (32, 8192),
         "long32_g25600": (32, 25600)}
OLD_SHA256 = "19d48cfafaa321599b932d89f89390c97ca43db640fac3704e6d880d734bc0a6"


def run(args, torch, old, new, public, name, result):
    import triton

    rows, width = CASES[name]
    device = f"npu:{args.device}"
    k, page = 512, 192
    pages = triton.cdiv(width, page)
    requests = 1 if rows == 16 else 2
    gen = torch.Generator().manual_seed(20261110)
    q = torch.randn((rows, 4, 128), generator=gen, dtype=torch.bfloat16).to(device)
    cache = torch.randn((pages * requests, page, 1, 128), generator=gen,
                        dtype=torch.bfloat16).to(device)
    table = torch.randperm(pages * requests, generator=gen, dtype=torch.int32)
    table = table.reshape(requests, pages).to(device)
    starts = (torch.arange(requests + 1, dtype=torch.int32) * (rows // requests)).to(device)
    lengths = (width - torch.arange(rows, dtype=torch.int32)).to(device)
    outputs = {arm: torch.empty((rows, k), device=device, dtype=torch.int32)
               for arm in ("old", "new")}

    cap = new._prefill_chunk_cap(q, cache, table, starts, width, k)
    chunk_rows = min(rows, cap, max(1, 512 * 1024**2 // (width * 4)))
    chunks = [(i, min(chunk_rows, rows - i)) for i in range(0, rows, chunk_rows)]
    frozen = torch.empty((rows, width), device=device, dtype=torch.float32)
    owners = new._build_owners(q, table, starts)

    def score_into(first, count, target):
        new._score(q, cache, table, starts, lengths, target, offset=first, owners=owners)

    for first, count in chunks:
        score_into(first, count, frozen[first:first + count])

    def topk_old():
        workspace = torch.empty((1024 * 1024,), device=device, dtype=torch.uint8)
        kwargs = {}
        if rows >= 2 * chunk_rows and old._psegmerge_applicable(chunk_rows, width, k):
            if old._is_910b(args.device):
                kwargs["persistent_scratch"] = old._PersistentTopKScratch()
        for first, count in chunks:
            end = first + count
            old.persistent_topk(frozen[first:end], lengths[first:end], outputs["old"][first:end],
                                workspace, k=k, max_seq_len=width,
                                block=old._QSA_TOPK_BLOCK, **kwargs)

    def topk_new():
        for first, count in chunks:
            end = first + count
            selected = new.topk(frozen[first:end], k)[1]
            new._store_topk_indices[(count,)](
                selected, lengths[first:end], outputs["new"][first:end], k, width
            )

    def whole(fn, arm):
        return fn(q, cache, table, starts, lengths, 4 * k, 4, rows // requests,
                  outputs[arm], 4 * width)

    jobs = {"topk": {"old": topk_old, "new": topk_new},
            "public_total": {"old": lambda: whole(old.qsa_select_paged_prefill, "old"),
                             "new": lambda: whole(public, "new")}}
    cpu_logits = frozen.cpu()
    cpu_lengths = lengths.cpu()
    expected = cpu_logits.topk(k, dim=1).values
    result.update(rows=rows, width=width, chunks=chunks, stages={})
    for stage, arms in jobs.items():
        for arm, call in arms.items():
            call()
            indices = outputs[arm].cpu().long()
            assert ((indices >= 0) & (indices < cpu_lengths[:, None])).all(), (stage, arm)
            ordered = indices.sort(dim=1).values
            assert (ordered[:, 1:] != ordered[:, :-1]).all(), (stage, arm)
            selected = cpu_logits.gather(1, indices).sort(dim=1, descending=True).values
            torch.testing.assert_close(selected, expected, rtol=1e-3, atol=1e-3)

    measure = setup(args, dict(chunks=chunks, score_into=score_into, logits=frozen), torch, result)
    for stage, arms in jobs.items():
        samples = {arm: [] for arm in arms}
        for round_index in range(args.rounds):
            for arm in (("old", "new") if round_index % 2 == 0 else ("new", "old")):
                samples[arm].append(measure(arms[arm])["kernel_us"])
        medians = {arm: statistics.median(values) for arm, values in samples.items()}
        result["stages"][stage] = dict(samples_us=samples, medians_us=medians,
                                       speedup=medians["old"] / medians["new"])
    print(name, json.dumps(result["stages"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--old-source", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("kernel",))
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--rounds", default=2, type=int)
    parser.add_argument("--kernel-warmup", default=2, type=int)
    parser.add_argument("--kernel-active", default=10, type=int)
    args = parser.parse_args()
    if min(args.rounds, args.kernel_active) < 1:
        parser.error("rounds and kernel-active must be positive")

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    assert "910B" in torch.npu.get_device_name(args.device)
    os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = "512"
    os.environ["FLAGGEMS_ASCEND_PTK_PATH"] = "auto"
    public, new, persistent, identity = load_production()
    if hashlib.sha256(args.old_source.read_bytes()).hexdigest() != OLD_SHA256:
        raise ValueError("Unexpected old production source")
    spec = importlib.util.spec_from_file_location("qsa_old_production_ab", args.old_source)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    assert old.persistent_topk is persistent.persistent_topk
    for name in ("_score", "_paged_score_910b", "_prefill_chunk_cap", "_build_owners"):
        a, b = getattr(old, name), getattr(new, name)
        assert inspect.getsource(getattr(a, "fn", a)) == inspect.getsource(getattr(b, "fn", b)), name
    report = Path(args.report) / "bf16-topk-ab.json"
    result = dict(mode=args.mode, new=identity, old_source_sha256=OLD_SHA256, cases={})
    for name in CASES:
        entry = result["cases"][name] = {}
        try:
            run(args, torch, old, new, public, name, entry)
        finally:
            report.write_text(json.dumps(result, indent=2))
    result["arithmetic_mean_speedup"] = {
        stage: statistics.mean(case["stages"][stage]["speedup"] for case in result["cases"].values())
        for stage in ("topk", "public_total")
    }
    report.write_text(json.dumps(result, indent=2))
    print("ARITHMETIC_MEAN_SPEEDUP", result["arithmetic_mean_speedup"], flush=True)


if __name__ == "__main__":
    main()
