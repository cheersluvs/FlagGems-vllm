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

"""Check native INT8 QuantLightningIndexer V1 before adopting a 910B baseline."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

CASES = {
    "h4_topk512": ([2], [16384]),
    "h4_compressed_boundary": ([9, 5], [2044, 4093]),
    "h4_select_all": ([3], [28]),
}


def fixture(name):
    import torch

    lengths, prefixes = CASES[name]
    generator = torch.Generator().manual_seed(20261102)
    ratio = 4
    page = 192
    heads, dim, topk = 4, 128, 512
    rows = sum(lengths)
    visible, owners = [], []
    for request, (length, prefix) in enumerate(zip(lengths, prefixes)):
        for local in range(length):
            count = (prefix + local + 1) // ratio
            visible.append(count)
            owners.append(request)
    columns = (max(visible) + page - 1) // page + 1
    pages = columns * len(lengths)
    table = torch.randperm(pages, generator=generator).reshape(len(lengths), columns)
    q = torch.randint(-8, 9, (rows, heads, dim), generator=generator).to(torch.int8)
    cache = torch.randint(-8, 9, (pages, page, 1, dim), generator=generator).to(
        torch.int8
    )
    # A forbidden key would win for row zero if visibility were ignored.
    q[0].fill_(1)
    future = visible[0]
    cache[table[owners[0], future // page], future % page, 0].fill_(127)
    # Powers of two keep this prerequisite probe free of scale-rounding ambiguity.
    qs = torch.full((rows, heads), 1 / 64, dtype=torch.float16)
    qs[:, 1::2] = 1 / 128
    ks = torch.full((pages, page, 1), 1 / 64, dtype=torch.float16)
    ks[:, 1::2] = 1 / 128
    weights = torch.ones((rows, heads), dtype=torch.float16)
    row_table = table[owners].to(torch.int32).contiguous()
    inputs = dict(
        q=q,
        cache=cache,
        weights=weights,
        qs=qs,
        ks=ks,
        row_table=row_table,
        q_ends=torch.arange(1, rows + 1, dtype=torch.int32),
        visible=torch.tensor(visible, dtype=torch.int32),
    )
    scores = []
    for row, count in enumerate(visible):
        physical = table[owners[row]]
        key = cache[physical, :, 0].reshape(-1, dim)[:count].to(torch.int32)
        scale = ks[physical, :, 0].reshape(-1)[:count].float()
        dot = q[row].to(torch.int32) @ key.T
        score = dot.float() * qs[row].float()[:, None] * scale[None, :]
        scores.append((score.relu() * weights[row].float()[:, None]).sum(0))
    return (
        inputs,
        scores,
        dict(
            heads=heads,
            dim=dim,
            page_size=page,
            group_topk=topk,
            compress_ratio=ratio,
            query_lengths=lengths,
            prefix_tokens=prefixes,
            visible_groups=visible,
            q_shape=list(q.shape),
            cache_shape=list(cache.shape),
            adaptation="One native batch per query; sparse_mode=0; exact visible group lengths",
            compression_boundary="floor((prefix_tokens + local_query + 1) / 4)",
        ),
    )


def check_output(output, scores):
    import torch

    if output.dtype != torch.int32 or output.shape != (len(scores), 1, 512):
        raise AssertionError(f"Unexpected output: {output.shape}, {output.dtype}")
    largest_error = 0.0
    for row, score in enumerate(scores):
        indices = output[row, 0].long()
        if bool(((indices < -1) | (indices >= score.numel())).any()):
            raise AssertionError(f"row {row}: invalid index outside [-1, visible)")
        selected = indices[indices >= 0]
        count = min(512, score.numel())
        if selected.numel() != count or selected.unique().numel() != count:
            raise AssertionError(f"row {row}: missing or duplicate valid indices")
        actual = score[selected].sort(descending=True).values
        expected = score.topk(count).values
        error = (actual - expected).abs().max().item() if count else 0.0
        largest_error = max(largest_error, error)
        if not torch.allclose(actual, expected, rtol=0, atol=1e-3):
            raise AssertionError(f"row {row}: selected-score max error {error}")
    return dict(rows=len(scores), selected_score_max_abs_error=largest_error)


def run_case(args):
    result = dict(case=args.case, status="failed")
    code = 1
    try:
        import torch
        import torch_npu

        result["environment"] = dict(
            python=sys.executable,
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            torch_npu_source=torch_npu.__file__,
            ascend_home=os.getenv("ASCEND_HOME_PATH"),
            opp=os.getenv("ASCEND_OPP_PATH"),
        )
        native = getattr(torch_npu, "npu_quant_lightning_indexer", None)
        if native is None or not torch.npu.is_available():
            result.update(
                status="prerequisites_missing",
                reason="Native V1 API or NPU unavailable",
            )
            code = 2
        else:
            torch.npu.set_device(args.device)
            result["environment"]["device"] = torch.npu.get_device_name(args.device)
            if "910B" not in result["environment"]["device"]:
                raise RuntimeError("This probe requires Ascend 910B")
            try:
                result["native_schema"] = str(
                    torch.ops.npu.npu_quant_lightning_indexer.default._schema
                )
            except AttributeError:
                result["native_schema"] = str(native)
            inputs, scores, geometry = fixture(args.case)
            result["geometry"] = geometry
            values = {
                key: value.to(f"npu:{args.device}") for key, value in inputs.items()
            }
            output = native(
                values["q"],
                values["cache"],
                values["weights"],
                values["qs"],
                values["ks"],
                0,
                0,
                actual_seq_lengths_query=values["q_ends"],
                actual_seq_lengths_key=values["visible"],
                block_table=values["row_table"],
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=512,
                sparse_mode=0,
                pre_tokens=2**63 - 1,
                next_tokens=2**63 - 1,
            )
            torch.npu.synchronize()
            output = output.cpu()
            # Keep raw indices before validation; no output repair or replacement TopK.
            result["raw_indices"] = output.tolist()
            result["checks"] = check_output(output, scores)
            result["status"] = "passed"
            code = 0
    except (ImportError, ModuleNotFoundError):
        result.update(status="prerequisites_missing", traceback=traceback.format_exc())
        code = 2
    except Exception:
        result["traceback"] = traceback.format_exc()
    Path(args.report).mkdir(parents=True, exist_ok=True)
    (Path(args.report) / f"{args.case}.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k != "raw_indices"}, indent=2),
        flush=True,
    )
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cpu-self-test", action="store_true")
    args = parser.parse_args()
    if args.cpu_self_test:
        import torch

        for name in CASES:
            _, scores, geometry = fixture(name)
            output = torch.full((len(scores), 1, 512), -1, dtype=torch.int32)
            for row, score in enumerate(scores):
                count = min(512, score.numel())
                output[row, 0, :count] = score.topk(count).indices.to(torch.int32)
            print(name, geometry["visible_groups"], check_output(output, scores))
            broken = output.clone()
            broken[0, 0, 1] = broken[0, 0, 0]
            try:
                check_output(broken, scores)
            except AssertionError:
                continue
            raise AssertionError("Validator accepted duplicate indices")
        return 0
    if args.case:
        return run_case(args)
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in CASES:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--case",
            name,
            "--report",
            str(report),
            "--device",
            str(args.device),
        ]
        try:
            with (report / f"{name}-console.txt").open("w") as log:
                completed = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout
                )
            code = completed.returncode
            path = report / f"{name}.json"
            entry = (
                json.loads(path.read_text())
                if path.is_file()
                else dict(case=name, status="crashed")
            )
            entry.pop("raw_indices", None)
            entry["exit_code"] = code
            if code != 0 and entry["status"] == "passed":
                entry["status"] = "crashed_after_checks"
        except subprocess.TimeoutExpired:
            entry = dict(case=name, status="timeout", exit_code=1)
        entries.append(entry)
        print(json.dumps(entry, indent=2), flush=True)
    status = (
        0
        if all(e["status"] == "passed" for e in entries)
        else (2 if all(e["status"] == "prerequisites_missing" for e in entries) else 1)
    )
    summary = dict(
        exit_code=status,
        cases=entries,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Native V1 correctness prerequisite only; no production INT8 support or performance claim",
        baseline_candidate=status == 0,
        next_step="If passed: implement matching operator INT8 path before paired benchmarks",
    )
    (report / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"Quant V1: {sum(e['status'] == 'passed' for e in entries)}/{len(entries)} passed; exit={status}",
        flush=True,
    )
    return status


if __name__ == "__main__":
    sys.exit(main())
