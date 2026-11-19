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

"""Compare a single 512-key local sort with mask compaction on 910B."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.experimental.tle.language.dsa.ascend import custom_ops

from qsa_timeline_910b import collect


@triton.jit
def _full(score, ids):
    a = tle.dsa.ascend.raw(
        "sort32", score, ids.to(tl.uint32), 16, out=tl.zeros((1024,), tl.float32)
    )
    b = tle.dsa.ascend.raw(
        "mrgsort",
        a,
        0,
        32,
        64,
        96,
        32,
        32,
        32,
        32,
        False,
        15,
        4,
        out=tl.zeros((1024,), tl.float32),
    )
    return tle.dsa.ascend.raw(
        "mrgsort", b, 0, 128, 256, 384, 128, 128, 128, 128, False, 15, 1, out=a
    )


@triton.jit
def _mask_sort_tile(
    Scores, Cutoffs, Output, COMPACT: tl.constexpr, PER_CORE: tl.constexpr
):
    lane = tl.arange(0, 512)
    word = tl.arange(0, 1024)
    empty = tl.where(
        word % 2 == 0,
        -float("inf"),
        tl.full((), -1, tl.int32).to(tl.float32, bitcast=True),
    )
    for i in range(PER_CORE):
        row = tl.program_id(0) * PER_CORE + i
        score = tl.load(Scores + row * 512 + lane)
        ids = row * 512 + lane
        if COMPACT:
            cutoff = tl.load(Cutoffs + row)
            keep = (score >= cutoff) & (score != -float("inf"))
            count = tl.sum(keep.to(tl.int32), 0)
            if count == 0:
                pairs = empty
            elif count <= 32:
                # Each group of 64 lanes uses two bit words at the head of a 32-byte block.
                bits = 1 << tl.arange(0, 32)
                words = tl.sum(
                    tl.reshape(keep.to(tl.int32), (8, 2, 32)) * bits[None, None, :], 2
                )
                lo, hi = tl.split(words)
                col = tl.arange(0, 8)[None, :]
                pattern = lo[:, None] * (col == 0).to(tl.int32) + hi[:, None] * (
                    col == 1
                ).to(tl.int32)
                pattern = tl.reshape(pattern, (64,)).to(tl.uint32, bitcast=True)
                vals, nv = tle.dsa.ascend.raw(
                    "gather_mask_custom_pattern",
                    score,
                    pattern,
                    True,
                    64,
                    1,
                    8,
                    8,
                    1,
                    out=[tl.zeros((512,), tl.float32), tl.zeros((1,), tl.int64)],
                )
                indexes, ni = tle.dsa.ascend.raw(
                    "gather_mask_custom_pattern",
                    ids,
                    pattern,
                    True,
                    64,
                    1,
                    8,
                    8,
                    1,
                    out=[tl.zeros((512,), tl.int32), tl.zeros((1,), tl.int64)],
                )
                tl.debug_barrier()
                valid = tl.arange(0, 32) < count
                sv = tl.where(
                    valid, al.extract_slice(vals, (0,), (32,), (1,)), -float("inf")
                )
                si = tl.where(valid, al.extract_slice(indexes, (0,), (32,), (1,)), -1)
                small = tle.dsa.ascend.raw(
                    "sort32", sv, si.to(tl.uint32), 1, out=tl.zeros((64,), tl.float32)
                )
                tl.debug_barrier()
                pairs = tle.dsa.insert_slice(empty, small, (0,), (64,), (1,))
            else:
                pairs = _full(score, ids)
        else:
            pairs = _full(score, ids)
        tl.debug_barrier()
        tl.store(Output + row * 1024 + word, pairs)
        tl.debug_barrier()


def inputs(counts):
    # Permute every row independently. The lowest surviving value ties the cutoff.
    gen = torch.Generator().manual_seed(910)
    rows = counts.numel()
    permutation = torch.stack([torch.randperm(512, generator=gen) for _ in range(rows)])
    cutoff = (torch.arange(rows) % 13 - 6).float()
    scores = (permutation - (512 - counts[:, None])).float() + cutoff[:, None]
    return scores, cutoff


def verify(words, scores, cutoff, compact):
    rows = scores.shape[0]
    values, ids = words[:, ::2].contiguous(), words[:, 1::2].contiguous().view(
        torch.int32
    )
    keep = (scores >= cutoff[:, None]) & scores.isfinite()
    sparse = (keep.sum(1) <= 32)[:, None] & compact
    oracle = (
        scores.masked_fill(sparse & ~keep, -float("inf")).sort(descending=True).values
    )
    torch.testing.assert_close(values, oracle, rtol=0, atol=0)
    valid = values.isfinite()
    local = ids.long() - torch.arange(rows)[:, None] * 512
    assert ((local[valid] >= 0) & (local[valid] < 512)).all()
    torch.testing.assert_close(
        scores.gather(1, local.clamp(0, 511))[valid], values[valid], rtol=0, atol=0
    )
    ordered = local.masked_fill(~valid, -1).sort().values
    assert not ((ordered[:, 1:] >= 0) & (ordered[:, 1:] == ordered[:, :-1])).any()
    assert (ids[sparse & ~valid] == -1).all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-sort32-tile", action="store_true", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    name = torch.npu.get_device_name(args.device)
    assert "910B" in name
    if not hasattr(custom_ops, "gather_mask_custom_pattern"):
        raise RuntimeError("Installed TLE lacks gather_mask_custom_pattern")
    rows, per_core = 1024, 32
    result = dict(
        status="running",
        device=name,
        triton=triton.__file__,
        rows=rows,
        keys_per_block=512,
        blocks_per_program=per_core,
        cases={},
    )
    path = args.report / "raw-cube-chain.json"

    def save():
        path.write_text(json.dumps(result, indent=2))

    save()
    for label, counts in [
        ("edges", torch.tensor([0, 1, 31, 32, 33, 128, 511, 512]).repeat(rows // 8)),
        *((str(n), torch.full((rows,), n)) for n in (19, 32, 33)),
    ]:
        cpu, cutoff = inputs(counts)
        if label == "edges":
            cpu[::8] = -float("inf")
            # Exactly 32 values tied at the cutoff, spanning all eight 64-lane repeats.
            cpu[3::8] = cutoff[3::8, None] - 1
            for lane in (0, 31, 32, 63):
                cpu[3::8, lane::64] = cutoff[3::8, None]
        src, threshold = cpu.to(device), cutoff.to(device)
        buffers = {
            arm: torch.empty((rows, 1024), device=device) for arm in ("full", "mask")
        }

        def launch(arm):
            return _mask_sort_tile[(rows // per_core,)](
                src, threshold, buffers[arm], arm == "mask", per_core, num_warps=4
            )

        entry = result["cases"][label] = dict(correct=False, rounds=[])
        calls = {arm: (lambda arm=arm: launch(arm)) for arm in buffers}
        for arm, call in calls.items():
            compiled = call()
            torch.npu.synchronize()
            if label == "edges":
                ir = compiled.asm["bcmlir"]
                if isinstance(ir, bytes):
                    ir = ir.decode()
                (args.report / (arm + "-bcmlir.txt")).write_text(ir)
                calls_in_ir = [
                    line for line in ir.splitlines() if "hivm.hir.custom" in line
                ]
                assert sum(
                    '"gather_mask_custom_pattern"' in line for line in calls_in_ir
                ) == (2 if arm == "mask" else 0)
            verify(buffers[arm].cpu(), cpu, cutoff, arm == "mask")
        entry["correct"] = True
        save()
        if label == "edges":
            continue
        for r, order in enumerate((tuple(calls), tuple(reversed(calls))), 1):
            timing = {}
            collect(
                SimpleNamespace(
                    report=args.report, worker=f"{label}-r{r}", trace_iterations=5
                ),
                calls,
                torch,
                timing,
                selected=order,
            )
            stages = timing["timeline"]["stages"]
            if any(stage["task_count"] != 5 for stage in stages.values()):
                raise RuntimeError(f"Expected one device task per launch: {stages}")
            us = {arm: stages[arm]["device_task_duration_sum_us"] / 5 for arm in order}
            sample = dict(
                device_us_per_launch=us, full_over_mask=us["full"] / us["mask"]
            )
            entry["rounds"].append(sample)
            save()
            print("MASK_SORT32_TILE", label, r, json.dumps(sample), flush=True)
    result["status"] = "passed"
    save()


if __name__ == "__main__":
    main()
