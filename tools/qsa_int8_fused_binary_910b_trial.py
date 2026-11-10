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


"""Compare one- and two-slot Cube-to-Vector score and TopK scheduling."""
"""Prepared mixed Cube/Vector candidate for the full INT8 harness inputs."""

import hashlib
import os
from functools import partial
from pathlib import Path

import torch
import triton

from probe_qsa_int8_mixed_pipeline_910b import (
    CUBE_TILE,
    PAGE,
    TOPK,
    _qsa_int8_handoff_copy,
    _qsa_int8_merge_sorted_tiles,
    _qsa_int8_merge_two_levels,
    _qsa_int8_probe_raw_pair512,
)


def _check_packed(label, packed, limits):
    scores = packed[..., ::2].contiguous()
    ids = packed[..., 1::2].contiguous().view(torch.int32)
    finite = torch.isfinite(scores)
    invalid = finite & ((ids < 0) | (ids >= limits[:, None, None]))
    unsorted = scores[..., :-1] < scores[..., 1:]
    nonfinite = torch.isnan(scores) | torch.isposinf(scores)
    print(
        f"{label} shape={tuple(packed.shape)} invalid_ids={int(invalid.sum())} "
        f"ascending_pairs={int(unsorted.sum())} nan_or_posinf={int(nonfinite.sum())}",
        flush=True,
    )
    assert not (invalid.any() or unsorted.any() or nonfinite.any()), label
    return scores, ids


def _diagnose_merge(mode, merge, visible, snapshot):
    source, target, lists, groups, in_width, out_width = merge.args
    assert (lists, groups, in_width, out_width) == (5, 3, 512, 512)
    assert source.shape == (128, 5, 1024)
    limits = visible.cpu()
    if mode == "capture":
        packed = source.cpu()
        temporary = snapshot.with_suffix(".tmp")
        torch.save({"packed": packed, "visible": limits}, temporary)
        temporary.replace(snapshot)
    elif mode == "real":
        saved = torch.load(snapshot, map_location="cpu", weights_only=True)
        packed = saved["packed"]
        assert torch.equal(saved["visible"], limits)
    else:
        limits = torch.full_like(limits, 5 * 512)
        ids = torch.arange(5 * 512, dtype=torch.int32).view(512, 5).T.contiguous()
        pairs = torch.stack((-ids.float(), ids.view(torch.float32)), dim=-1)
        packed = pairs.reshape(1, 5, 1024).expand_as(source).contiguous()

    assert packed.shape == source.shape and packed.dtype == torch.float32
    scores, ids = _check_packed(f"MERGE53 INPUT mode={mode}", packed, limits)
    if mode == "capture":
        print(f"MERGE53 CAPTURED {snapshot}", flush=True)
        return

    source.copy_(packed)
    target.fill_(float("nan"))
    torch.npu.synchronize()
    print(f"MERGE53 LAUNCH mode={mode} grid=384 IN=512 OUT=512", flush=True)
    merge()
    torch.npu.synchronize()
    print(f"MERGE53 RETURNED mode={mode}", flush=True)

    _check_merge_output(scores, ids, target.cpu(), fan_in=2)
    print(f"MERGE53 PASS mode={mode}", flush=True)


def _check_merge_output(scores, ids, result, fan_in, *, locate=False):
    out_scores = result[..., ::2].contiguous()
    out_ids = result[..., 1::2].contiguous().view(torch.int32)
    groups, out_width = out_scores.shape[1:]
    bad_groups = []
    for group in range(groups):
        start = group * fan_in
        values = scores[:, start : start + fan_in].flatten(1)
        indices = ids[:, start : start + fan_in].flatten(1)
        expected = values.topk(min(out_width, values.shape[1]), dim=1).values
        if expected.shape[1] < out_width:
            expected = torch.nn.functional.pad(
                expected, (0, out_width - expected.shape[1]), value=-float("inf")
            )
        actual = out_scores[:, group]
        if not locate:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        valid = torch.isfinite(actual)
        # Match each finite score/index pair to a source candidate, allowing ties.
        keys, order = indices.masked_fill(~torch.isfinite(values), 2**31 - 1).sort(dim=1)
        returned = out_ids[:, group].contiguous()
        found = torch.searchsorted(keys, returned).clamp(max=keys.shape[1] - 1)
        bad_id = valid & (keys.gather(1, found) != returned)
        bad_pair = valid & (values.gather(1, order.gather(1, found)) != actual)
        ordered = returned.masked_fill(~valid, -1).sort(dim=1).values
        duplicate = (ordered[:, 1:] >= 0) & (ordered[:, 1:] == ordered[:, :-1])
        if locate:
            bad_groups.append(
                ((actual != expected) | bad_id | bad_pair).any(1) | duplicate.any(1)
            )
        else:
            assert not (bad_id.any() or bad_pair.any() or duplicate.any())
    if locate:
        return torch.stack(bad_groups, dim=1)


def _probe_pair512(mode, device):
    ids = torch.arange(1024, dtype=torch.int32).reshape(512, 2).T.contiguous()
    scores = -ids.float()
    live = (512, 512) if mode == "pair_full" else (320, 384)
    for i, count in enumerate(live):
        scores[i, count:] = -float("inf")
    packed = torch.stack((scores, ids.view(torch.float32)), dim=-1).contiguous()
    raw = torch.full((2048,), float("nan"), dtype=torch.float32, device=device)
    prefix = torch.full((1024,), float("nan"), dtype=torch.float32, device=device)
    counts = torch.full((4,), -1, dtype=torch.int32, device=device)
    source = packed.to(device)
    torch.npu.synchronize()
    print(f"PAIR512 LAUNCH mode={mode} valid={live} raw_words=2048 grid=1", flush=True)
    _qsa_int8_probe_raw_pair512[(1,)](
        source, raw, prefix, counts, use_bytecode=True, enable_ubuf_saving=True,
        enable_auto_bind_sub_block=False, multibuffer=False,
    )
    torch.npu.synchronize()
    raw, prefix, counts = raw.cpu(), prefix.cpu(), counts.cpu()
    raw_pairs, prefix_pairs = raw.view(-1, 2), prefix.view(-1, 2)
    raw_ids = raw_pairs[:, 1].contiguous().view(torch.int32)
    prefix_ids = prefix_pairs[:, 1].contiguous().view(torch.int32)
    expected, order = scores.flatten().topk(512)
    expected_ids = ids.flatten()[order]
    raw_bad = (raw_pairs[:512, 0] != expected) | (raw_ids[:512] != expected_ids)
    prefix_bad = (prefix_pairs[:, 0] != expected) | (prefix_ids != expected_ids)
    slice_bad = (raw[:1024].view(torch.int32) != prefix.view(torch.int32)).sum()
    print(
        f"PAIR512 RESULT mode={mode} consumed={counts.tolist()} "
        f"raw_bad={int(raw_bad.sum())} prefix_bad={int(prefix_bad.sum())} "
        f"slice_word_diff={int(slice_bad)}",
        flush=True,
    )
    for at in (0, 504, 512):
        print(
            f"PAIR512 RAW at={at} scores={raw_pairs[at:at+8, 0].tolist()} "
            f"ids={raw_ids[at:at+8].tolist()}",
            flush=True,
        )
    assert ((counts[:2] >= 0) & (counts[:2] <= 512)).all(), "invalid consumption"
    assert int(counts[:2].sum()) >= 512, "raw merge produced fewer than 512 candidates"
    assert not raw_bad.any(), "raw Top-512 differs from CPU reference"
    assert not prefix_bad.any() and int(slice_bad) == 0, "Top-512 slice mismatch"
    print(f"PAIR512 PASS mode={mode}", flush=True)


def _replay_179(mode, merge, snapshot):
    source, target, lists, groups, in_width, out_width = merge.args
    assert (lists, groups, in_width, out_width) == (17, 9, 512, 512)
    saved = torch.load(snapshot, map_location="cpu", weights_only=True)
    packed = saved["packed"]
    assert packed.shape == source.shape and packed.dtype == source.dtype
    scores, ids = _check_packed(f"MERGE179 INPUT mode={mode}", packed, saved["visible"])
    digest = hashlib.sha256(packed.numpy().tobytes()).hexdigest()
    source.copy_(packed)
    target.fill_(float("nan"))
    torch.npu.synchronize()
    full = mode in ("full_179", "locate_179", "grid_179", "split_179")
    words = 2048 if full else 1024
    grid = source.shape[0] * groups
    force_pair = False
    if mode == "grid_179":
        grid = int(os.environ["QSA_MERGE179_GRID"])
        assert 1 <= grid <= source.shape[0] * groups
        force_pair = os.environ.get("QSA_MERGE179_FORCE_PAIR") == "1"
        assert not force_pair or grid == 1, "force-pair probe is only safe for task 0"
        sentinel = target[0, 0, :1].cpu().view(torch.int32)
    print(
        f"MERGE179 LAUNCH mode={mode} input_sha256={digest} raw_words={words} "
        f"grid={grid} lists={lists} groups={groups} output_words=1024 force_pair={force_pair}",
        flush=True,
    )
    if mode == "split_179":
        pair_groups = lists // 2
        pair_grid = source.shape[0] * pair_groups
        print(f"MERGE179 SPLIT pair_grid={pair_grid} tail_rows={source.shape[0]}", flush=True)
        compiled = _qsa_int8_merge_sorted_tiles[(pair_grid,)](
            *merge.args, FULL_RAW_OUTPUT=True, FORCE_PAIR=True,
            PAIR_GROUPS=pair_groups, **merge.keywords,
        )
        torch.npu.synchronize()
        pair_result = target.cpu()
        assert torch.isnan(pair_result[:, -1, ::2]).all(), "pair kernel wrote the tail"
        target.view(torch.int32)[:, -1].copy_(source.view(torch.int32)[:, -1])
        torch.npu.synchronize()
        result = target.cpu()
        tail_equal = torch.equal(result.view(torch.int32)[:, -1], packed.view(torch.int32)[:, -1])
        pairs_equal = torch.equal(
            result.view(torch.int32)[:, :-1], pair_result.view(torch.int32)[:, :-1],
        )
        ttir = None if compiled is None else compiled.asm.get("ttir")
        branch_count = ttir.count("scf.if") if isinstance(ttir, str) else None
        print(
            f"MERGE179 SPLIT COPY tail_exact={tail_equal} pairs_unchanged={pairs_equal} "
            f"pair_scf_if_count={branch_count}",
            flush=True,
        )
        assert tail_equal and pairs_equal
        bad = _check_merge_output(scores, ids, result, fan_in=2, locate=True)
        print(
            f"MERGE179 SPLIT RESULT failed={int(bad.sum())} failed_by_group={bad.sum(0).tolist()}",
            flush=True,
        )
        assert not bad.any(), "split replay failed CPU comparison"
        _check_packed("MERGE179 SPLIT OUTPUT", result, saved["visible"])
        print("MERGE179 SPLIT PASS", flush=True)
        return
    if mode == "grid_179":
        compiled = _qsa_int8_merge_sorted_tiles[(grid,)](
            *merge.args, FULL_RAW_OUTPUT=True, FORCE_PAIR=force_pair, **merge.keywords
        )
    elif full:
        merge(FULL_RAW_OUTPUT=True)
    else:
        merge()
    torch.npu.synchronize()
    result = target.cpu()
    if mode == "grid_179":
        bad = _check_merge_output(scores, ids, result, fan_in=2, locate=True)
        active = torch.arange(bad.numel()).view_as(bad) < grid
        failed = bad & active
        inactive = result[~active].view(torch.int32)
        # Every inactive word should retain the same NaN bit pattern.
        inactive_written = int((inactive != sentinel).any(dim=1).sum())
        binary = None if compiled is None else compiled.asm.get("npubin")
        binary_sha = "unavailable" if binary is None else hashlib.sha256(bytes(binary)).hexdigest()
        ttir = None if compiled is None else compiled.asm.get("ttir")
        if isinstance(ttir, str):
            print(
                f"MERGE179 TTIR sha256={hashlib.sha256(ttir.encode()).hexdigest()} "
                f"scf_if_count={ttir.count('scf.if')}",
                flush=True,
            )
        if grid == 1:
            sample = result[0, 0, :16].view(8, 2)
            print(
                f"MERGE179 TASK0 scores={sample[:, 0].tolist()} "
                f"ids={sample[:, 1].contiguous().view(torch.int32).tolist()}",
                flush=True,
            )
        print(
            f"MERGE179 GRID RESULT grid={grid} failed={int(failed.sum())} "
            f"launched_by_group={active.sum(0).tolist()} "
            f"failed_by_group={failed.sum(0).tolist()} "
            f"inactive_written={inactive_written} force_pair={force_pair} "
            f"npubin_sha256={binary_sha}",
            flush=True,
        )
        if failed.any():
            row, group = failed.nonzero()[0].tolist()
            print(f"MERGE179 GRID FIRST_BAD row={row} group={group}", flush=True)
        assert not failed.any() and inactive_written == 0, f"grid={grid} replay failed"
        print(f"MERGE179 GRID PASS grid={grid}", flush=True)
        return
    if mode == "locate_179":
        bad = _check_merge_output(scores, ids, result, fan_in=2, locate=True)
        failures = bad.nonzero()
        print(f"MERGE179 LOCATE failed_tasks={len(failures)}", flush=True)
        assert len(failures), "batch replay did not reproduce a bad task"
        row, group = failures[0].tolist()
        pair = packed[row:row + 1, group * 2:group * 2 + 2].contiguous()
        limit = saved["visible"][row:row + 1].clone()
        path = Path(os.environ["QSA_MERGE179_PAIR"])
        temporary = path.with_suffix(".tmp")
        torch.save(
            {"packed": pair, "visible": limit, "row": row, "group": group,
             "batch_output": result[row:row + 1, group:group + 1].clone()},
            temporary,
        )
        temporary.replace(path)
        pair_sha = hashlib.sha256(pair.numpy().tobytes()).hexdigest()
        print(
            f"MERGE179 EXTRACT row={row} group={group} lists={pair.shape[1]} "
            f"pair_sha256={pair_sha} path={path}",
            flush=True,
        )
        return
    _check_packed(f"MERGE179 OUTPUT mode={mode}", result, saved["visible"])
    _check_merge_output(scores, ids, result, fan_in=2)
    print(f"MERGE179 PASS mode={mode}", flush=True)


def _replay_actual_pair(device, launch_kwargs):
    path = Path(os.environ["QSA_MERGE179_PAIR"])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    packed, limits = saved["packed"], saved["visible"]
    assert packed.dtype == torch.float32 and packed.shape in ((1, 1, 1024), (1, 2, 1024))
    scores, ids = _check_packed("MERGE179 ACTUAL INPUT", packed, limits)
    pair_sha = hashlib.sha256(packed.numpy().tobytes()).hexdigest()
    print(
        f"MERGE179 ACTUAL LAUNCH row={saved['row']} group={saved['group']} "
        f"lists={packed.shape[1]} pair_sha256={pair_sha} grid=1",
        flush=True,
    )
    source = packed.to(device)
    target = torch.full((1, 1, 1024), float("nan"), dtype=torch.float32, device=device)
    torch.npu.synchronize()
    _qsa_int8_merge_sorted_tiles[(1,)](
        source, target, packed.shape[1], 1, 512, 512,
        FULL_RAW_OUTPUT=True, **launch_kwargs,
    )
    torch.npu.synchronize()
    actual = target.cpu()
    difference = (actual.view(torch.int32) != saved["batch_output"].view(torch.int32))
    bad = _check_merge_output(scores, ids, actual, fan_in=2, locate=True)
    print(
        f"MERGE179 ACTUAL RESULT valid={not bool(bad.any())} "
        f"words_differing_from_batch={int(difference.sum())}",
        flush=True,
    )
    assert not bad.any(), "actual single-task output differs from CPU reference"
    print("MERGE179 ACTUAL PASS", flush=True)


def _trace_prefix(producer, merges, visible, capture_179=None):
    limits = visible.cpu()
    print("MERGE53 TRACE LAUNCH producer", flush=True)
    producer()
    torch.npu.synchronize()
    scores, ids = _check_packed("MERGE53 TRACE producer", merges[0].args[0].cpu(), limits)
    start = torch.arange(scores.shape[1]) * scores.shape[2]
    width = scores.shape[2]
    live = torch.isfinite(scores)
    assert torch.equal(
        live.sum(-1), (limits[:, None] - start[None, :]).clamp(0, width)
    ), "producer finite count differs from visibility"
    assert not (
        live & ((ids < start[None, :, None]) | (ids >= start[None, :, None] + width))
    ).any(), "producer candidate outside its tile"
    ordered = ids.masked_fill(~live, -1).sort(dim=-1).values
    assert not (
        (ordered[..., 1:] >= 0) & (ordered[..., 1:] == ordered[..., :-1])
    ).any(), "producer repeated a live index"
    print("MERGE53 TRACE PASS producer", flush=True)
    for level, merge in enumerate(merges[:4]):
        if capture_179 is not None and level == 2:
            packed = merge.args[0].cpu()
            temporary = capture_179.with_suffix(".tmp")
            torch.save({"packed": packed, "visible": limits}, temporary)
            temporary.replace(capture_179)
            print(f"MERGE179 CAPTURED {capture_179}", flush=True)
            return
        target = merge.args[1]
        label = f"MERGE53 TRACE merge={level}"
        print(f"{label} LAUNCH target={tuple(target.shape)}", flush=True)
        merge()
        torch.npu.synchronize()
        packed = target.cpu()
        out_scores, out_ids = _check_packed(label, packed, limits)
        _check_merge_output(scores, ids, packed, fan_in=4 if level == 0 else 2)
        scores, ids = out_scores, out_ids
        print(f"{label} PASS", flush=True)
    print("MERGE53 TRACE COMPLETE (merge=4 not launched)", flush=True)


class PreparedFusedBinarySelection:
    def __init__(self, values, host_visible, starts, chunk_rows):
        device = values["q"].device
        rows = values["q"].shape[0]
        self.output = torch.empty((rows, TOPK), device=device, dtype=torch.int32)
        self.chunks = []
        self.visible = host_visible
        rank = torch.arange(TOPK, device=device, dtype=torch.int32)
        key = values["cache"].view(-1, 128)
        key_scale = values["ks"].view(-1)
        scratch = {}

        def buffer(shape, dtype):
            return torch.empty(shape, device=device, dtype=dtype)

        for begin, end in zip(starts[:-1], starts[1:], strict=True):
            # Both rows in each Cube program use the same request page table.
            # Start a new chunk at every request boundary.
            lists = max(1, triton.cdiv(int(host_visible[end - 1]), CUBE_TILE))
            for first in range(begin, end, chunk_rows):
                last = min(first + chunk_rows, end)
                count = last - first
                cores = triton.cdiv(count, 2)
                packed_q = torch.zeros((cores, 16, 128), dtype=torch.int8, device=device)
                # Use a padded row dimension so an odd final row never reads the next request.
                q = values["q"][first:last]
                pair_count = count // 2
                if pair_count:
                    packed_q[:pair_count, :8].copy_(q[:2 * pair_count].view(pair_count, 8, 128))
                if count % 2:
                    packed_q[-1, :4].copy_(q[-1])

                layout = count, lists
                if layout not in scratch:
                    ring = buffer((cores, 2, 16, CUBE_TILE), torch.int32)
                    proposals = buffer((count, lists, 2 * CUBE_TILE), torch.float32)
                    levels = []
                    groups, span = lists, CUBE_TILE
                    while groups > 2 or groups * span < TOPK:
                        groups = triton.cdiv(groups, 2)
                        span = min(2 * span, TOPK)
                        levels.append(buffer((count, groups, 2 * span), torch.float32))
                    scratch[layout] = (
                        ring, proposals, levels, buffer((count, TOPK), torch.float32),
                        buffer((count, TOPK), torch.int64),
                    )
                ring, proposals, levels, scores, positions = scratch[layout]
                visible = values["visible"][first:last]
                producer = partial(
                    _qsa_int8_handoff_copy[(cores,)], packed_q, key,
                    values["row_table"][begin], ring, proposals,
                    values["qs"][first:last], key_scale, values["weights"][first:last], visible,
                    lists, 2, PAGE, CUBE_TILE, count, True, True,
                    use_bytecode=True, disable_auto_cv_work_space_manage=True,
                    disable_auto_inject_block_sync=True, enable_auto_bind_sub_block=False,
                    multibuffer=False, sync_solver=False, unit_flag=False,
                )
                merges = []
                source = proposals
                # With fewer than three lists the two-level prefix does not exist.
                fused = len(levels) >= 2 and lists > 4
                for level, target in enumerate(levels):
                    if fused and level == 0:
                        continue
                    if fused and level == 1:
                        kernel = _qsa_int8_merge_two_levels
                        widths = (CUBE_TILE,)
                    else:
                        kernel = _qsa_int8_merge_sorted_tiles
                        widths = (source.shape[2] // 2, target.shape[2] // 2)
                    merges.append(partial(
                        kernel[(count * target.shape[1],)], source, target,
                        source.shape[1], target.shape[1], *widths,
                        use_bytecode=True, enable_ubuf_saving=True,
                        enable_auto_bind_sub_block=False, multibuffer=False,
                    ))
                    source = target
                self.chunks.append((
                    first, count, lists, fused, producer, merges, source.view(count, -1, 2),
                    scores, positions, rank >= visible[:, None], self.output[first:last],
                ))

    def __call__(self, *, synchronize=False):
        mode = os.environ.get("QSA_MERGE53_MODE")
        if mode:
            assert synchronize and mode in (
                "capture", "real", "synthetic", "trace",
                "capture_179", "original_179", "full_179", "pair_full", "pair_padded",
                "locate_179", "actual_pair_179", "grid_179", "split_179",
            )
            if mode not in ("trace", "pair_full", "pair_padded", "actual_pair_179"):
                snapshot = Path(os.environ["QSA_MERGE53_SNAPSHOT"])
        for first, count, lists, fused, producer, merges, pairs, scores, positions, padding, output in self.chunks:
            if mode:
                assert (first, count, lists, fused) == (0, 128, 133, True)
                if mode in ("pair_full", "pair_padded"):
                    _probe_pair512(mode, output.device)
                    return None
                if mode == "trace":
                    _trace_prefix(producer, merges, self.visible[:count])
                    return None
                if mode == "capture_179":
                    _trace_prefix(producer, merges, self.visible[:count], snapshot)
                    return None
                if mode in ("original_179", "full_179", "locate_179", "grid_179", "split_179"):
                    _replay_179(mode, merges[2], snapshot)
                    return None
                if mode == "actual_pair_179":
                    _replay_actual_pair(output.device, merges[2].keywords)
                    return None
                if mode != "capture":
                    _diagnose_merge(mode, merges[4], self.visible[:count], snapshot)
                    return None
            if synchronize:
                print(f"mixed first={first} rows={count} tiles={lists} fused={fused}", flush=True)
            producer()
            if synchronize:
                torch.npu.synchronize()
            for level, merge in enumerate(merges):
                if mode == "capture" and level == 4:
                    _diagnose_merge(mode, merge, self.visible[:count], snapshot)
                    return None
                if synchronize:
                    print(f"mixed first={first} merge={level}", flush=True)
                merge()
                if synchronize:
                    torch.npu.synchronize()
            candidates = pairs[..., 0].contiguous()
            indices = pairs[..., 1].contiguous().view(torch.int32)
            torch.topk(candidates, TOPK, dim=-1, sorted=True, out=(scores, positions))
            torch.gather(indices, 1, positions, out=output)
            output.masked_fill_(padding, -1)
            if synchronize:
                torch.npu.synchronize()
        return self.output
