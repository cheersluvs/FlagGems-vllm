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
    Q, K, QS, KS, Weight, Table, Visible, Workspace, State, Indices, Values,
    ROWS: tl.constexpr, WIDTH: tl.constexpr, SLOTS: tl.constexpr,
    PAGE_SIZE: tl.constexpr, TOP: tl.constexpr, CHUNK: tl.constexpr, CUT: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.arange(0, 16)
    d = tl.arange(0, 128)
    n = tl.arange(0, CHUNK)
    lane = tl.arange(0, TOP)
    pair_lane = tl.arange(0, TOP * 2)
    merge_lane = tl.arange(0, TOP * 8)
    base_row = pid * 2
    q = tl.load(
        Q + (base_row * 4 + m[:, None]) * 128 + d[None, :],
        mask=(m[:, None] < 8) & (base_row + m[:, None] // 4 < ROWS), other=0,
    )
    for ring_index in tl.static_range(SLOTS):
        al.sync_block_set("vector", "cube", ring_index, pipe.PIPE_MTE3, pipe.PIPE_FIX)
    for tile in range(tl.cdiv(WIDTH, TOP)):
        slot = tile % SLOTS
        work = Workspace + (pid * SLOTS + slot) * 16 * TOP
        al.sync_block_wait("vector", "cube", slot, pipe.PIPE_MTE3, pipe.PIPE_FIX)
        for sub in range(TOP // CHUNK):
            column = tile * TOP + sub * CHUNK + n
            page = tl.load(Table + column // PAGE_SIZE, mask=column < WIDTH, other=0)
            token = page * PAGE_SIZE + column % PAGE_SIZE
            k = tl.load(
                K + token[:, None] * 128 + d[None, :],
                mask=column[:, None] < WIDTH, other=0,
            )
            dot = tl.dot(q, tl.trans(k), out_dtype=tl.int32)
            tl.store(work + m[:, None] * TOP + sub * CHUNK + n[None, :], dot)
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
            dot = tl.load(work + (row_in_core * 4 + h[:, None]) * TOP + lane[None, :])
            qs = tl.load(QS + row * 4 + h).to(tl.float32)
            weight = tl.load(Weight + row * 4 + h).to(tl.float32)
            dequant = dot.to(tl.float32) * qs[:, None] * key_scale[None, :]
            score = tl.sum(tl.maximum(dequant, 0.0) * weight[:, None], 0)
            score = tl.where((column < visible) & (column < WIDTH), score, -float("inf"))
            if CUT == 0:
                tl.store(State + row * 4 * TOP + lane, score)
            else:
                pairs = tl.zeros((2 * TOP,), tl.float32)
                pairs = tle.dsa.ascend.raw(
                    "sort_1d_pack", score, tl.zeros((4 * TOP,), tl.float32),
                    True, TOP, tile * TOP, SORT_IMPL_BASE, out=pairs,
                )
                state = State + row * 4 * TOP
                if CUT == 1:
                    tl.store(state + pair_lane, pairs)
                else:
                    if tile > 0:
                        if CUT == 4 or CUT == 5:
                            previous = tl.load(state + pair_lane)
                            active = tl.join(previous, pairs)
                            padded = tl.join(active, tl.zeros((2 * TOP, 2), tl.float32))
                            packed = tl.reshape(tl.permute(padded, (2, 1, 0)), (8 * TOP,))
                        else:
                            tl.store(state + 2 * TOP + pair_lane, pairs)
                            packed = tl.load(state + merge_lane, mask=merge_lane < 4 * TOP, other=0)
                        merged, _ = tle.dsa.ascend.raw(
                            "merge_exhaust_sort4", packed, 2,
                            0, TOP, 2 * TOP, 3 * TOP, TOP, TOP, 0, 0,
                            out=[tl.zeros((8 * TOP,), tl.float32), tl.zeros((4,), tl.int32)],
                        )
                        # Each input has TOP entries including -inf padding. Exhausting
                        # either list therefore guarantees at least TOP merged entries.
                        pairs = al.extract_slice(merged, (0,), (2 * TOP,), (1,))
                    tl.store(state + pair_lane, pairs)
                    if CUT == 2 or CUT == 4:
                        if tile == tl.cdiv(WIDTH, TOP) - 1:
                            values, indices = tle.dsa.ascend.raw(
                                "unpack_sort", pairs, TOP,
                                out=[tl.zeros((TOP,), tl.float32), tl.zeros((TOP,), tl.int32)],
                            )
                            live = lane < tl.minimum(visible, TOP)
                            tl.store(Indices + row * TOP + lane, tl.where(live, indices, -1))
                            tl.store(Values + row * TOP + lane, tl.where(live, values, -float("inf")))
        al.sync_block_set("vector", "cube", slot, pipe.PIPE_MTE3, pipe.PIPE_FIX)
    for ring_index in tl.static_range(SLOTS):
        al.sync_block_wait("vector", "cube", ring_index, pipe.PIPE_MTE3, pipe.PIPE_FIX)


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


def worker(args):
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    assert "910B" in torch.npu.get_device_name(args.device)
    rows, width = CASES[args.worker]
    cores = triton.cdiv(rows, ROWS_PER_CORE)
    cpu = fixture(args.worker)
    inputs = tuple(t.to(device) for t in cpu)
    workspace = torch.empty((cores, 2, 16, TOPK), device=device, dtype=torch.int32)
    state = torch.empty((rows, 4 * TOPK), device=device, dtype=torch.float32)
    indices = torch.empty((rows, TOPK), device=device, dtype=torch.int32)
    values = torch.empty_like(indices, dtype=torch.float32)

    def launch(slots, cut=4, compile_only=False):
        if compile_only:
            invoke = lambda *a, **kw: _qsa_int8_mixed_ring.warmup(*a, grid=(cores,), **kw)
        else:
            invoke = _qsa_int8_mixed_ring[(cores,)]
        return invoke(
            *inputs, workspace, state, indices, values,
            rows, width, slots, PAGE, TOPK, CUBE_TILE, cut,
            disable_auto_cv_work_space_manage=True,
            disable_auto_inject_block_sync=True,
            enable_auto_bind_sub_block=False,
            use_bytecode=True,
            enable_ubuf_saving=True,
            unit_flag=False,
            multibuffer=False,
            sync_solver=False,
        )

    if args.launch_only:
        launch(args.slots)
        torch.npu.synchronize()
        return

    report = Path(args.report)
    from triton.backends.ascend import compiler

    preflight = {}
    for slots, arm in ((1, "serial"), (2, "double_buffer")):
        outcomes = {}
        preflight[arm] = outcomes
        for cut, stage in ((4, "full"), (3, "merge_gm_roundtrip"), (5, "merge_direct_pairs")):
            if stage != "full" and outcomes["full"]["compiled"]:
                break
            label = f"{args.worker}-{arm}-{stage}"
            print(f"{label}: compile only", flush=True)
            with patch.object(compiler, "subprocess", CompileCapture(report, label)):
                try:
                    kernel = launch(slots, cut, compile_only=True)
                    outcomes[stage] = dict(compiled=True, artifacts=save_artifacts(kernel, report, label))
                except Exception as exc:
                    outcomes[stage] = dict(compiled=False, error=str(exc))
                    print(f"{label}: {exc}", flush=True)
    (report / f"{args.worker}-compile.json").write_text(json.dumps(preflight, indent=2))
    if not all(outcomes["full"]["compiled"] for outcomes in preflight.values()):
        raise SystemExit(1)

    functions = {"serial": lambda: launch(1), "double_buffer": lambda: launch(2)}
    result = {"case": args.worker, "rows": rows, "width": width, "rows_per_core": ROWS_PER_CORE, "merge_input": "direct_pairs", "arms": {}}
    for name, fn in functions.items():
        print(f"{args.worker}/{name}: compiling and launching", flush=True)
        kernel = fn()
        artifacts = save_artifacts(kernel, report, f"{args.worker}-{name}")
        torch.npu.synchronize()
        result["arms"][name] = dict(artifacts=artifacts, correctness=check(cpu, indices, values))
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        result["arms"][name]["repeat_correctness"] = check(cpu, indices, values)

    samples = {name: [] for name in functions}
    for round_index in range(6):
        order = tuple(functions) if round_index % 2 == 0 else tuple(reversed(functions))
        for name in order:
            start = torch.npu.Event(enable_timing=True)
            stop = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(8):
                functions[name]()
            stop.record()
            stop.synchronize()
            samples[name].append(start.elapsed_time(stop) * 1000 / 8)
    medians = {name: statistics.median(times) for name, times in samples.items()}
    result.update(
        event_samples_us=samples, event_medians_us=medians,
        serial_over_double=medians["serial"] / medians["double_buffer"],
        hardware_overlap_verified=False,
    )
    summary = report / f"{args.worker}-summary.json"
    summary.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if args.task_trace:
        from qsa_timeline_910b import collect

        try:
            collect(SimpleNamespace(report=report, worker=args.worker), functions, torch, result,
                    selected=tuple(functions))
        except Exception as exc:
            result["task_trace_error"] = str(exc)
        summary.write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--slots", type=int, choices=(1, 2), default=2)
    parser.add_argument("--launch-only", action="store_true")
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
    for case in CASES:
        log = report / f"{case}-worker.txt"
        command = base + ["--worker", case]
        if args.task_trace:
            command.append("--task-trace")
        code, timed_out = run_child(command, log, args.timeout)
        statuses[case] = dict(exit_code=code, timed_out=timed_out)

    timeline_status = "skipped_worker_failure" if args.instruction_timeline else "not_requested"
    result = {"cases": statuses, "instruction_timeline": {"status": timeline_status}}
    if args.instruction_timeline and all(item["exit_code"] == 0 for item in statuses.values()):
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
