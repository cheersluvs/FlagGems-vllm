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

import argparse
import gzip
import json
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.experimental.tle.language.dsa.ascend.custom_ops import SORT_IMPL_BASE

pipe = al.PIPE
CASES = {"short": (35, 769), "long": (39, 8193)}
PAGE = 192
TOPK = 512
ROWS_PER_CORE = 2
CUBE_TILE = 64


@triton.jit
def _qsa_int8_mixed_ring(
    Q, K, QS, KS, Weight, Table, Visible, Workspace, Proposals, DebugDot,
    ROWS: tl.constexpr, WIDTH: tl.constexpr, SLOTS: tl.constexpr,
    PAGE_SIZE: tl.constexpr, TOP: tl.constexpr, CHUNK: tl.constexpr, SCORE_ONLY: tl.constexpr, PRODUCER_ONLY: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.arange(0, 16)
    n = tl.arange(0, CHUNK)
    lane = tl.arange(0, TOP)
    pair_lane = tl.arange(0, TOP * 2)
    base_row = pid * 2
    d = tl.arange(0, 128)
    q = tl.load(Q + (pid * 16 + m[:, None]) * 128 + d[None, :])
    if not PRODUCER_ONLY:
        for ring_index in tl.static_range(SLOTS):
            al.sync_block_set("vector", "cube", ring_index, pipe.PIPE_MTE3, pipe.PIPE_FIX)
    SUBTILES_PER_TILE: tl.constexpr = TOP // CHUNK
    # Keep the final tile full in the workspace. Vector masks its logical tail.
    for flat in range(tl.cdiv(WIDTH, TOP) * SUBTILES_PER_TILE):
        tile = flat // SUBTILES_PER_TILE
        sub = flat % SUBTILES_PER_TILE
        slot = tile % SLOTS
        work = Workspace + (pid * SLOTS + slot) * 16 * TOP
        if not PRODUCER_ONLY:
            if sub == 0:
                al.sync_block_wait("vector", "cube", slot, pipe.PIPE_MTE3, pipe.PIPE_FIX)
        start = flat * CHUNK
        cube_page = tl.load(Table + start // PAGE_SIZE, mask=start < WIDTH, other=0)
        cube_offset = cube_page * PAGE_SIZE + start % PAGE_SIZE
        k = tl.load(K + (cube_offset + n[:, None]) * 128 + d[None, :])
        cube_dot = tl.dot(q, tl.trans(k), out_dtype=tl.int32)
        tl.store(work + sub * 16 * CHUNK + m[:, None] * CHUNK + n[None, :], cube_dot)

        if not PRODUCER_ONLY:
            if sub == SUBTILES_PER_TILE - 1:
                al.sync_block_set("cube", "vector", slot, pipe.PIPE_FIX, pipe.PIPE_MTE2)

                al.sync_block_wait("cube", "vector", slot, pipe.PIPE_FIX, pipe.PIPE_MTE2)
                column = tile * TOP + lane
                page = tl.load(Table + column // PAGE_SIZE, mask=column < WIDTH, other=0)
                token = page * PAGE_SIZE + column % PAGE_SIZE
                key_scale = tl.load(KS + token, mask=column < WIDTH, other=0).to(tl.float32)
                row_in_core = al.sub_vec_id()
                row = base_row + row_in_core
                if row < ROWS:
                    visible = tl.load(Visible + row)
                    h = tl.arange(0, 4)
                    dot = tl.load(
                        work + (lane[None, :] // CHUNK) * 16 * CHUNK
                        + (row_in_core * 4 + h[:, None]) * CHUNK + lane[None, :] % CHUNK
                    )
                    qs = tl.load(QS + row * 4 + h).to(tl.float32)
                    weight = tl.load(Weight + row * 4 + h).to(tl.float32)
                    dequant = dot.to(tl.float32) * qs[:, None] * key_scale[None, :]
                    score = tl.sum(tl.maximum(dequant, 0.0) * weight[:, None], 0)
                    score = tl.where((column < visible) & (column < WIDTH), score, -float("inf"))
                    offset = row * tl.cdiv(WIDTH, TOP) + tile
                    if SCORE_ONLY:
                        tl.store(Proposals + offset * TOP + lane, score)
                        tl.store(DebugDot + (offset * 4 + h[:, None]) * TOP + lane[None, :], dot)
                    else:
                        pairs = tle.dsa.ascend.raw(
                            "sort_1d_pack", score, tl.zeros((4 * TOP,), tl.float32),
                            True, TOP, tile * TOP, SORT_IMPL_BASE,
                            out=tl.zeros((2 * TOP,), tl.float32),
                        )
                        tl.store(Proposals + offset * 2 * TOP + pair_lane, pairs)
                al.sync_block_set("vector", "cube", slot, pipe.PIPE_MTE3, pipe.PIPE_FIX)
    if not PRODUCER_ONLY:
        for ring_index in tl.static_range(SLOTS):
            al.sync_block_wait("vector", "cube", ring_index, pipe.PIPE_MTE3, pipe.PIPE_FIX)


@triton.jit
def _qsa_int8_vector_merge(
    Proposals, Out, Visible, Indices, Values,
    INPUT_LISTS: tl.constexpr, OUTPUT_LISTS: tl.constexpr,
    TOP: tl.constexpr, FINAL: tl.constexpr,
):
    task = tl.program_id(0)
    row = task // OUTPUT_LISTS
    group = task % OUTPUT_LISTS
    first = group * 4
    lanes = tl.arange(0, 8 * TOP)
    ways = tl.minimum(4, INPUT_LISTS - first)
    packed = tl.load(
        Proposals + (row * INPUT_LISTS + first) * 2 * TOP + lanes,
        mask=lanes < ways * 2 * TOP, other=0,
    )
    pairs = al.extract_slice(packed, (0,), (2 * TOP,), (1,))
    if ways > 1:
        merged, _ = tle.dsa.ascend.raw(
            "merge_exhaust_sort4", packed, ways,
            0, TOP, 2 * TOP, 3 * TOP, TOP, TOP,
            tl.where(ways > 2, TOP, 0), tl.where(ways > 3, TOP, 0),
            out=[tl.zeros((8 * TOP,), tl.float32), tl.zeros((4,), tl.int32)],
        )
        pairs = al.extract_slice(merged, (0,), (2 * TOP,), (1,))
    if FINAL:
        values, indices = tle.dsa.ascend.raw(
            "unpack_sort", pairs, TOP,
            out=[tl.zeros((TOP,), tl.float32), tl.zeros((TOP,), tl.int32)],
        )
        lane = tl.arange(0, TOP)
        live = lane < tl.minimum(tl.load(Visible + row), TOP)
        tl.store(Indices + row * TOP + lane, tl.where(live, indices, -1))
        tl.store(Values + row * TOP + lane, tl.where(live, values, -float("inf")))
    else:
        tl.store(Out + task * 2 * TOP + tl.arange(0, 2 * TOP), pairs)

@triton.jit
def _qsa_int8_cube_tile_control(
    Q, K, Table, Starts, DotOut, QOut, KOut,
    PAGE_SIZE: tl.constexpr, CHUNK: tl.constexpr,
    PAGED: tl.constexpr, TRACE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = tl.load(Starts + pid)
    m = tl.arange(0, 16)
    d = tl.arange(0, 128)
    n = tl.arange(0, CHUNK)
    q = tl.load(Q + m[:, None] * 128 + d[None, :])
    if PAGED:
        page = tl.load(Table + start // PAGE_SIZE)
        offset = page * PAGE_SIZE + start % PAGE_SIZE
    else:
        offset = start
    k = tl.load(K + (offset + n[:, None]) * 128 + d[None, :])
    dot = tl.dot(q, tl.trans(k), out_dtype=tl.int32)
    tl.store(DotOut + (pid * 16 + m[:, None]) * CHUNK + n[None, :], dot)
    if TRACE:
        tl.store(QOut + (pid * 16 + m[:, None]) * 128 + d[None, :], q)
        tl.store(KOut + (pid * CHUNK + n[:, None]) * 128 + d[None, :], k)


@triton.jit
def _qsa_int8_cube_flat_control(
    Q, K, Table, DotOut,
    SUBTILES: tl.constexpr, STORAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr, CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.arange(0, 16)
    d = tl.arange(0, 128)
    n = tl.arange(0, CHUNK)
    q = tl.load(Q + (pid * 16 + m[:, None]) * 128 + d[None, :])
    for sub in range(SUBTILES):
        start = sub * CHUNK
        page = tl.load(Table + start // PAGE_SIZE)
        offset = page * PAGE_SIZE + start % PAGE_SIZE
        k = tl.load(K + (offset + n[:, None]) * 128 + d[None, :])
        dot = tl.dot(q, tl.trans(k), out_dtype=tl.int32)
        slot = sub % STORAGE
        tl.store(DotOut + ((pid * STORAGE + slot) * 16 + m[:, None]) * CHUNK + n[None, :], dot)


def fixture(case):
    rows, width = CASES[case]
    pages = triton.cdiv(width, PAGE)
    gen = torch.Generator().manual_seed(20261109)
    q = torch.randint(-128, 128, (rows, 4, 128), dtype=torch.int8, generator=gen)
    k = torch.randint(-128, 128, (pages, PAGE, 128), dtype=torch.int8, generator=gen)
    qs = (0.001 + 0.02 * torch.rand((rows, 4), generator=gen)).half()
    ks = (0.001 + 0.02 * torch.rand((pages, PAGE), generator=gen)).half()
    weights = (torch.rand((rows, 4), generator=gen) - 0.25).half()
    table = torch.randperm(pages, generator=gen, dtype=torch.int32)
    visible = torch.linspace(width // 2, width, rows).to(torch.int32)
    visible[:5] = torch.tensor([0, 191, 193, 511, 513])
    return q, k, qs, ks, weights, table, visible


def check(cpu, indices, values):
    q, k, qs, ks, weights, table, visible = cpu
    keys = k[table.long()].reshape(-1, 128).int()
    scales = ks[table.long()].flatten().float()
    indices, values = indices.cpu(), values.cpu()
    max_error = 0.0
    for row, count in enumerate(visible.tolist()):
        take = min(count, TOPK)
        assert (indices[row, take:] == -1).all(), (row, "index padding")
        assert torch.isneginf(values[row, take:]).all(), (row, "value padding")
        if not take:
            continue
        dot = keys[:count] @ q[row].int().T
        score = ((dot.float() * qs[row].float() * scales[:count, None]).relu()
                 * weights[row].float()).sum(1)
        selected = indices[row, :take].long()
        assert (selected >= 0).all() and (selected < count).all(), (row, "index bounds")
        assert selected.unique().numel() == take, (row, "duplicate indices")
        got = values[row, :take]
        expected = score.topk(take).values
        torch.testing.assert_close(got, score[selected], rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(got.sort(descending=True).values, expected, rtol=1e-3, atol=1e-3)
        max_error = max(max_error, (got - score[selected]).abs().max().item())
    return {"rows": len(visible), "max_score_abs_error": max_error}




def check_workspace(cpu, workspace, width, slots):
    q, k, _, _, _, table, _ = cpu
    keys = k[table.long()].reshape(-1, 128).int()
    tiles = triton.cdiv(width, TOPK)
    ring = workspace.cpu().flatten()[:workspace.shape[0] * slots * 16 * TOPK]
    ring = ring.reshape(-1, slots, TOPK // CUBE_TILE, 16, CUBE_TILE)
    result = {"checked_subtiles": 0, "mismatches": 0, "sentinel_values": 0, "zero_values": 0, "subtile_mismatches": {}}
    for tile in range(max(tiles - slots, 0), tiles):
        for sub in range(TOPK // CUBE_TILE):
            start = tile * TOPK + sub * CUBE_TILE
            count = min(CUBE_TILE, width - start)
            if count <= 0:
                continue
            for row in range(q.shape[0]):
                core, local = divmod(row, ROWS_PER_CORE)
                got = ring[core, tile % slots, sub, local * 4:local * 4 + 4, :count].T
                expected = keys[start:start + count] @ q[row].int().T
                result["checked_subtiles"] += 1
                result["sentinel_values"] += (got == -2147483647).sum().item()
                result["zero_values"] += (got == 0).sum().item()
                if not torch.equal(got, expected):
                    result["mismatches"] += 1
                    key = str(sub)
                    counts = result["subtile_mismatches"]
                    counts[key] = counts.get(key, 0) + 1
                    result.setdefault("first_failure", {
                        "row": row, "tile": tile, "subtile": sub,
                        "got": got[:4].tolist(), "expected": expected[:4].tolist(),
                    })
    return result

def check_producer(cpu, raw, proposals, debug_dot):
    q, k, qs, ks, weights, table, visible = cpu
    keys = k[table.long()].reshape(-1, 128).int()
    scales = ks[table.long()].flatten().float()
    raw = raw.cpu()
    proposals = proposals.cpu()
    debug_dot = debug_dot.cpu()
    results = {name: {"failures": 0} for name in ("dot_handoff", "score", "proposals")}
    for row, count in enumerate(visible.tolist()):
        for tile in range(raw.shape[1]):
            begin = tile * TOPK
            end = min(begin + TOPK, count)
            take = max(end - begin, 0)
            expected = torch.full((TOPK,), -float("inf"))
            dot = keys[begin:begin + take] @ q[row].int().T
            received_dot = debug_dot[row, tile, :, :take].T
            dot_ok = torch.equal(received_dot, dot)
            if take:
                expected[:take] = (
                    (dot.float() * qs[row].float() * scales[begin:end, None]).relu()
                    * weights[row].float()
                ).sum(1)
            actual = raw[row, tile]
            pairs = proposals[row, tile]
            values = pairs[0::2]
            indices = pairs[1::2].contiguous().view(torch.int32).long()
            valid = indices[:take] - begin
            score_ok = torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)
            index_ok = bool(((valid >= 0) & (valid < take)).all()) and valid.unique().numel() == take
            values_ok = index_ok and torch.allclose(
                values[:take], expected[valid], rtol=1e-3, atol=1e-3
            )
            sorted_ok = torch.allclose(
                values, expected.sort(descending=True).values, rtol=1e-3, atol=1e-3
            )
            for name, ok in (("dot_handoff", dot_ok), ("score", score_ok), ("proposals", values_ok and sorted_ok)):
                if ok:
                    continue
                entry = results[name]
                entry["failures"] += 1
                entry.setdefault("first_failure", {
                    "row": row, "tile": tile, "visible_in_tile": take,
                    "dot_head": received_dot[:4].tolist(),
                    "expected_dot_head": dot[:4].tolist(),
                    "raw_head": [float(x) if torch.isfinite(x) else str(x.item()) for x in actual[:8]],
                    "proposal_head": [float(x) if torch.isfinite(x) else str(x.item()) for x in values[:8]],
                    "index_head": indices[:8].tolist(),
                    "expected_head": [float(x) if torch.isfinite(x) else str(x.item()) for x in expected[:8]],
                    "index_ok": index_ok, "values_ok": bool(values_ok), "sorted_ok": bool(sorted_ok),
                })
    return results

def save_artifacts(kernel, report, label):
    files = []
    for key, value in kernel.asm.items():
        if isinstance(value, str):
            name = f"{label}-{key}.txt"
            (report / name).write_text(value)
        elif isinstance(value, (bytes, bytearray)) and key == "npubin":
            name = f"{label}.npubin.gz"
            (report / name).write_bytes(gzip.compress(value))
        else:
            continue
        files.append(name)
    return files


class CompileCapture:
    def __init__(self, report, label):
        self.report = report
        self.label = label
        self.attempt = 0

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def run(self, command, *args, **kwargs):
        if not isinstance(command, (tuple, list)) or not command:
            return subprocess.run(command, *args, **kwargs)
        if Path(command[0]).name != "bishengir-compile":
            return subprocess.run(command, *args, **kwargs)
        self.attempt += 1
        try:
            return subprocess.run(command, *args, **kwargs)
        except subprocess.CalledProcessError as exc:
            stem = f"{self.label}-bisheng-{self.attempt}"
            (self.report / f"{stem}-command.json").write_text(
                json.dumps({"argv": list(map(str, command)), "exit_code": exc.returncode}, indent=2)
            )
            for arg in command[1:]:
                path = Path(arg)
                if path.suffix == ".mlir" and path.is_file():
                    (self.report / f"{stem}-input.mlir.gz").write_bytes(
                        gzip.compress(path.read_bytes(), compresslevel=3)
                    )
                    break
            for name in ("stdout", "stderr"):
                data = getattr(exc, name)
                if data:
                    if isinstance(data, str):
                        data = data.encode()
                    (self.report / f"{stem}-{name}.txt").write_bytes(data)
            raise


def cube_controls(args, cpu, inputs, packed_q, workspace):
    from triton.backends.ascend import compiler

    _, keys, _, _, _, table, _ = cpu
    logical = keys[table.long()].reshape(-1, 128).contiguous()
    rows, width = CASES[args.worker]
    device = workspace.device
    starts = sorted({0, 64, 192, 512, (width - 1) // CUBE_TILE * CUBE_TILE})
    starts_tensor = torch.tensor(starts, dtype=torch.int32, device=device)
    dense = logical.to(device)
    dots = torch.empty((len(starts), 16, CUBE_TILE), device=device, dtype=torch.int32)
    q_out = torch.empty((len(starts), 16, 128), device=device, dtype=torch.int8)
    k_out = torch.empty((len(starts), CUBE_TILE, 128), device=device, dtype=torch.int8)
    report = Path(args.report)
    result = {"case": args.worker, "rows": rows, "width": width, "controls": {}}
    for paged in (False, True):
        for trace in (False, True):
            label = f"{args.worker}-cube-{'paged' if paged else 'dense'}-{'trace' if trace else 'dot'}"
            entry = {}
            result["controls"][label] = entry
            try:
                with patch.object(compiler, "subprocess", CompileCapture(report, label)):
                    kernel = _qsa_int8_cube_tile_control[(len(starts),)](
                        inputs[0], inputs[1] if paged else dense, inputs[5], starts_tensor,
                        dots, q_out, k_out, PAGE, CUBE_TILE, paged, trace,
                        enable_auto_bind_sub_block=False, multibuffer=False,
                    )
                torch.npu.synchronize()
                entry["artifacts"] = save_artifacts(kernel, report, label)
                got = dots.cpu()
                expected_k = torch.stack([logical[start:start + CUBE_TILE] for start in starts])
                expected_q = packed_q[0]
                expected = torch.stack([expected_q.int() @ tile.int().T for tile in expected_k])
                entry["dot_matches_by_start"] = {
                    str(start): bool(torch.equal(got[i], expected[i]))
                    for i, start in enumerate(starts)
                }
                if trace:
                    entry["q_matches"] = torch.equal(q_out.cpu(), expected_q.expand_as(q_out))
                    entry["k_matches"] = torch.equal(k_out.cpu(), expected_k)
            except Exception as exc:
                entry["error"] = str(exc)
            print(json.dumps({label: entry}, indent=2), flush=True)

    subtiles = triton.cdiv(width, CUBE_TILE)
    expected_q = packed_q.int()
    for storage in (subtiles, TOPK // CUBE_TILE, 2 * TOPK // CUBE_TILE):
        label = f"{args.worker}-cube-flat-storage{storage}"
        entry = {}
        result["controls"][label] = entry
        try:
            output = torch.full(
                (packed_q.shape[0], storage, 16, CUBE_TILE),
                -2147483647, dtype=torch.int32, device=device,
            )
            with patch.object(compiler, "subprocess", CompileCapture(report, label)):
                kernel = _qsa_int8_cube_flat_control[(packed_q.shape[0],)](
                    inputs[0], inputs[1], inputs[5], output,
                    subtiles, storage, PAGE, CUBE_TILE,
                    enable_auto_bind_sub_block=False, multibuffer=False,
                )
            torch.npu.synchronize()
            entry["artifacts"] = save_artifacts(kernel, report, label)
            got = output.cpu()
            checks = {}
            for sub in range(max(0, subtiles - storage), subtiles):
                start = sub * CUBE_TILE
                tile = logical[start:start + CUBE_TILE].int()
                expected = torch.stack([q @ tile.T for q in expected_q])
                actual = got[:, sub % storage]
                checks[str(start)] = bool(torch.equal(actual, expected))
                if not checks[str(start)] and "first_mismatch" not in entry:
                    first = (actual != expected).nonzero()[0].tolist()
                    pos = tuple(first)
                    entry["first_mismatch"] = dict(
                        start=start, position=first,
                        actual=int(actual[pos]), expected=int(expected[pos]),
                    )
            entry["dot_matches_by_start"] = checks
        except Exception as exc:
            entry["error"] = str(exc)
        print(json.dumps({label: entry}, indent=2), flush=True)
    (report / f"{args.worker}-cube-controls.json").write_text(json.dumps(result, indent=2))
    if any(
        "error" in entry or entry.get("mismatches", 0)
        or not all(entry.get("dot_matches_by_start", {}).values())
        or entry.get("q_matches") is False or entry.get("k_matches") is False
        for entry in result["controls"].values()
    ):
        raise SystemExit(1)


def worker(args):
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    assert "910B" in torch.npu.get_device_name(args.device)
    rows, width = CASES[args.worker]
    cores = triton.cdiv(rows, ROWS_PER_CORE)
    cpu = fixture(args.worker)
    q, *rest = cpu
    padded_q = torch.zeros((cores * ROWS_PER_CORE, 4, 128), dtype=torch.int8)
    padded_q[:rows].copy_(q)
    packed_q = torch.zeros((cores, 16, 128), dtype=torch.int8)
    packed_q[:, :ROWS_PER_CORE * 4].copy_(padded_q.reshape(cores, ROWS_PER_CORE * 4, 128))
    inputs = tuple(t.to(device) for t in (packed_q, *rest))
    workspace = torch.empty((cores, 2, 16, TOPK), device=device, dtype=torch.int32)
    lists = triton.cdiv(width, TOPK)
    state = torch.empty((rows, lists, 2 * TOPK), device=device, dtype=torch.float32)
    raw = torch.empty((rows, lists, TOPK), device=device, dtype=torch.float32)
    debug_dot = torch.empty((rows, lists, 4, TOPK), device=device, dtype=torch.int32)
    levels = [state]
    while lists > 1:
        lists = triton.cdiv(lists, 4)
        levels.append(torch.empty((rows, lists, 2 * TOPK), device=device, dtype=torch.float32))
    indices = torch.empty((rows, TOPK), device=device, dtype=torch.int32)
    values = torch.empty_like(indices, dtype=torch.float32)

    def launch(slots, compile_only=False, score_only=False, producer_only=False):
        if compile_only:
            invoke = lambda *a, **kw: _qsa_int8_mixed_ring.warmup(*a, grid=(cores,), **kw)
        else:
            invoke = _qsa_int8_mixed_ring[(cores,)]
        return invoke(
            *inputs, workspace, raw if score_only else state, debug_dot,
            rows, width, slots, PAGE, TOPK, CUBE_TILE, score_only, producer_only,
            disable_auto_cv_work_space_manage=True,
            disable_auto_inject_block_sync=True,
            enable_auto_bind_sub_block=False,
            use_bytecode=True,
            enable_ubuf_saving=True,
            unit_flag=False,
            multibuffer=False,
            sync_solver=False,
        )

    if args.cube_controls:
        cube_controls(args, cpu, inputs, packed_q, workspace)
        return

    if args.launch_only:
        launch(args.slots)
        torch.npu.synchronize()
        return

    report = Path(args.report)
    from triton.backends.ascend import compiler

    preflight = {}
    for slots, arm in ((1, "serial"), (2, "double_buffer")):
        label = f"{args.worker}-{arm}-proposals"
        print(f"{label}: compile only", flush=True)
        with patch.object(compiler, "subprocess", CompileCapture(report, label)):
            try:
                kernel = launch(slots, compile_only=True)
                preflight[arm] = dict(compiled=True, artifacts=save_artifacts(kernel, report, label))
            except Exception as exc:
                preflight[arm] = dict(compiled=False, error=str(exc))
                print(f"{label}: {exc}", flush=True)
    (report / f"{args.worker}-compile.json").write_text(json.dumps(preflight, indent=2))
    if not all(outcome["compiled"] for outcome in preflight.values()):
        raise SystemExit(1)

    def merge():
        kernels = []
        for current, target in zip(levels, levels[1:]):
            kernels.append(_qsa_int8_vector_merge[(rows * target.shape[1],)](
                current, target, inputs[-1], indices, values,
                current.shape[1], target.shape[1], TOPK, target.shape[1] == 1,
                use_bytecode=True, enable_auto_bind_sub_block=False,
                enable_ubuf_saving=True, multibuffer=False,
            ))
        return kernels

    def pipeline(slots):
        launch(slots)
        return merge()

    functions = {"serial": lambda: pipeline(1), "double_buffer": lambda: pipeline(2)}
    proposal_functions = {"serial": lambda: launch(1), "double_buffer": lambda: launch(2)}
    result = {"case": args.worker, "rows": rows, "width": width,
              "rows_per_core": ROWS_PER_CORE, "merge": "separate_vector_tree", "arms": {}}
    summary = report / f"{args.worker}-summary.json"
    for slots, name in ((1, "serial"), (2, "double_buffer")):
        label = f"{args.worker}-{name}-producer"
        print(f"{label}: checking raw scores and packed proposals independently", flush=True)
        arm = {}
        result["arms"][name] = arm
        try:
            with patch.object(compiler, "subprocess", CompileCapture(report, label)):
                workspace.fill_(-2147483647)
                kernel = launch(slots, score_only=True)
                arm["score_artifacts"] = save_artifacts(kernel, report, f"{label}-score")
                torch.npu.synchronize()
                arm["mixed_workspace"] = check_workspace(cpu, workspace, width, slots)
                launch(slots)
            torch.npu.synchronize()
            arm["producer"] = check_producer(cpu, raw, state, debug_dot)
        except Exception as exc:
            arm["producer_error"] = str(exc)
        if "producer_error" in arm or arm["producer"]["dot_handoff"]["failures"]:
            try:
                with patch.object(compiler, "subprocess", CompileCapture(report, f"{label}-cube")):
                    workspace.fill_(-2147483647)
                    kernel = launch(slots, producer_only=True)
                    arm["cube_artifacts"] = save_artifacts(kernel, report, f"{label}-cube")
                torch.npu.synchronize()
                arm["cube_only_workspace"] = check_workspace(cpu, workspace, width, slots)
            except Exception as exc:
                arm["cube_only_error"] = str(exc)
        print(json.dumps({name: arm}, indent=2), flush=True)
        summary.write_text(json.dumps(result, indent=2))

    producer_ok = all(
        "producer" in arm and not any(stage["failures"] for stage in arm["producer"].values())
        for arm in result["arms"].values()
    )
    if not producer_ok:
        raise SystemExit(1)

    for name, fn in functions.items():
        arm = result["arms"][name]
        try:
            with patch.object(compiler, "subprocess", CompileCapture(report, f"{args.worker}-{name}-merge")):
                kernels = fn()
            arm["merge_artifacts"] = [
                file for i, kernel in enumerate(kernels)
                for file in save_artifacts(kernel, report, f"{args.worker}-{name}-merge{i}")
            ]
            torch.npu.synchronize()
            arm["correctness"] = check(cpu, indices, values)
        except Exception as exc:
            arm["merge_error"] = str(exc)
            print(f"{args.worker}/{name} merge: {exc}", flush=True)

    pipeline_ok = all("correctness" in arm for arm in result["arms"].values())
    stages = [("proposals", proposal_functions)]
    if pipeline_ok:
        stages.append(("pipeline", functions))
    for stage, funcs in stages:
        samples = {name: [] for name in funcs}
        for round_index in range(6):
            order = tuple(funcs) if round_index % 2 == 0 else tuple(reversed(funcs))
            for name in order:
                start = torch.npu.Event(enable_timing=True)
                stop = torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(8):
                    funcs[name]()
                stop.record()
                stop.synchronize()
                samples[name].append(start.elapsed_time(stop) * 1000 / 8)
        medians = {name: statistics.median(times) for name, times in samples.items()}
        result[stage] = dict(
            event_samples_us=samples, event_medians_us=medians,
            serial_over_double=medians["serial"] / medians["double_buffer"],
        )
    result["hardware_overlap_verified"] = False
    summary = report / f"{args.worker}-summary.json"
    summary.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if args.task_trace:
        from qsa_timeline_910b import collect

        try:
            trace_functions = functions if pipeline_ok else proposal_functions
            collect(SimpleNamespace(report=report, worker=args.worker), trace_functions, torch, result,
                    selected=tuple(trace_functions))
        except Exception as exc:
            result["task_trace_error"] = str(exc)
        summary.write_text(json.dumps(result, indent=2))
    if not pipeline_ok:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--slots", type=int, choices=(1, 2), default=2)
    parser.add_argument("--launch-only", action="store_true")
    parser.add_argument("--cube-controls", action="store_true")
    parser.add_argument("--instruction-timeline", action="store_true")
    parser.add_argument("--task-trace", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return

    from probe_qsa_int8_score_instructions_910b import collect, find_simulator
    from probe_qsa_int8_ten_shapes_910b import run_child

    report = Path(args.report).resolve()
    base = [sys.executable, "-u", __file__, "--report", str(report), "--device", str(args.device)]
    statuses = {}
    for case in (("short",) if args.cube_controls else CASES):
        log = report / f"{case}-worker.txt"
        command = base + ["--worker", case]
        if args.cube_controls:
            command.append("--cube-controls")
        if args.task_trace:
            command.append("--task-trace")
        code, timed_out = run_child(command, log, args.timeout)
        statuses[case] = dict(exit_code=code, timed_out=timed_out)

    timeline_status = "skipped_worker_failure" if args.instruction_timeline else "not_requested"
    result = {"cases": statuses, "instruction_timeline": {"status": timeline_status}}
    long_summary = report / "long-summary.json"
    long_producer_ok = long_summary.is_file() and "proposals" in json.loads(long_summary.read_text())
    if args.instruction_timeline and long_producer_ok:
        profiler, checks = find_simulator(report)
        timeline = {"status": "unavailable", "discovery": checks}
        if profiler:
            timeline["status"] = "attempted"
            for slots in (1, 2):
                label = f"long-slots{slots}"
                capture = report / label
                capture.mkdir()
                log = report / f"{label}-simulator.txt"
                command = profiler + [
                    "--soc-version=Ascend910B4", "--core-id=0",
                    "--kernel-name=_qsa_int8_mixed_ring", "--launch-count=1",
                    "--timeout=1", f"--output={capture}",
                ] + base + ["--worker", "long", "--launch-only", "--slots", str(slots)]
                code, timed_out = run_child(command, log, min(args.timeout, 120))
                timeline[label] = dict(
                    exit_code=code, timed_out=timed_out,
                    artifacts=collect(label, capture, report),
                )
        result["instruction_timeline"] = timeline
    (report / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if any(item["exit_code"] != 0 or item["timed_out"] for item in statuses.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
