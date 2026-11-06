# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calibrate component throughput for the current Ascend INT8 scorer."""

import argparse
import json
import os
import statistics
import sys
import traceback
from pathlib import Path

from profile_qsa_int8_score_910b import check_chunk
from probe_qsa_int8_ten_shapes_910b import PLANS, run_child
from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan

from tools.qsa_int8_ten_workloads_910b import make_inputs

CHUNKS = {"int8_budget_exact": (8192, 128), "int8_long100k": (8176, 112)}
MACS = 64 * 64 * 128
GRID = 20
UNITS = GRID * 16
TILES = UNITS * 8
COPY_BYTES = 64 * 1024**2


def traffic(width, visible, tables, bounded):
    """Count source-level work and the unique useful input, ignoring cache lines."""
    if width % 64 or len(visible) % 16:
        raise ValueError("The two calibration chunks must have full output tiles")
    dense_tiles = width // 64
    units = (dense_tiles + 7) // 8
    q_bytes = len(visible) * (4 * (128 + 2 + 2) + 4)
    write_bytes = len(visible) * width * 4
    dots = active_tiles = page_reads = key_rows = 0
    physical_extent = {}
    unique_pages = set()
    for group in range(0, len(visible), 16):
        limit = min(width, max(visible[group : group + 16]))
        useful = (limit + 63) // 64
        issued = useful if bounded else dense_tiles
        active_tiles += useful
        dots += issued
        key_rows += limit
        for tile in range(0, issued, 8):
            end = min(issued, tile + 8)
            page_reads += (end - 1) // 3 - tile // 3 + 1
        for page, start in enumerate(range(0, limit, 192)):
            physical = tables[group][page]
            assert physical >= 0
            unique_pages.add((group, page))
            physical_extent[physical] = max(
                physical_extent.get(physical, 0), min(192, limit - start)
            )
    pairs = sum(min(width, v) for v in visible)
    unique_k_bytes = sum(physical_extent.values()) * 130
    return dict(
        dot_tiles=dots,
        active_dot_tiles=active_tiles,
        fully_invisible_dot_tiles=dots - active_tiles,
        fill_only_tiles=(len(visible) // 16) * dense_tiles - dots,
        executed_macs=dots * MACS,
        useful_macs=pairs * 4 * 128,
        tile_dot_result_logical_bytes=dots * 64 * 64 * 4,
        q_metadata_requested_bytes=q_bytes * units,
        k_scale_requested_bytes=key_rows * 130,
        page_lookup_sites_executed=page_reads,
        score_write_bytes=write_bytes,
        unique_q_metadata_bytes=q_bytes,
        unique_k_scale_bytes=unique_k_bytes,
        # The unbounded kernel masks table reads for wholly invisible pages.
        # Count only the useful table entries in the perfect-reuse byte model.
        perfect_reuse_io_bytes=q_bytes
        + unique_k_bytes
        + 4 * len(unique_pages)
        + write_bytes,
        requested_io_bytes_upper=q_bytes * units
        + key_rows * 130
        + 4 * page_reads
        + write_bytes,
    )


def run_case(args, result):
    global tl

    import torch
    import torch_npu
    import triton
    import triton.language as tl

    from qsa_int8_shared_map_selection_910b_trial import PreparedSharedMapInt8Selection

    root = os.getenv("QSA_ENV_ROOT")
    if not root or not torch.npu.is_available():
        raise ModuleNotFoundError("Pinned NPU environment required")
    if (
        not Path(triton.__file__)
        .resolve()
        .is_relative_to(Path(root).resolve() / "flagtree")
    ):
        raise RuntimeError(f"Unexpected compiler: {triton.__file__}")
    torch.npu.set_device(args.device)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = f"npu:{args.device}"
    result["environment"] = dict(
        device=torch.npu.get_device_name(args.device),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        compiler=triton.__file__,
        opp=os.getenv("ASCEND_OPP_PATH"),
    )
    if "910B" not in result["environment"]["device"]:
        raise NotImplementedError("910B required")

    @triton.jit
    def _sol_copy(X, Y, SIZE: tl.constexpr):
        i = tl.arange(0, 8192)
        for tile in range(tl.program_id(0), SIZE // 8192, tl.num_programs(0)):
            x = tile * 8192 + i
            tl.store(Y + x, tl.load(X + x))

    @triton.jit
    def _sol_cube(Q, K, Out, COUNT: tl.constexpr):
        n = tl.arange(0, 64)
        d = tl.arange(0, 128)
        ij = n[:, None] * 128 + d[None, :]
        for unit in range(tl.program_id(0), COUNT, tl.num_programs(0)):
            q = tl.load(Q + unit * 8192 + ij)
            acc = tl.full((64, 64), 0, tl.int32)
            for tile in range(8):
                k = tl.load(K + (unit * 8 + tile) * 8192 + ij)
                acc = tl.dot(q, tl.trans(k), acc, out_dtype=tl.int32)
            tl.store(Out + unit * 4096 + n[:, None] * 64 + n[None, :], acc)

    @triton.jit
    def _sol_vector(Dot, QS, KS, W, Visible, Out, COUNT: tl.constexpr):
        n = tl.arange(0, 64)
        q = tl.arange(0, 16)
        for unit in range(tl.program_id(0), COUNT, tl.num_programs(0)):
            qs = tl.load(QS + unit * 64 + n).to(tl.float32)
            w = tl.load(W + unit * 64 + n).to(tl.float32)
            visible = tl.load(Visible + unit * 16 + q)
            for i in range(8):
                tile = unit * 8 + i
                x = tl.load(Dot + tile * 4096 + n[:, None] * 64 + n[None, :])
                ks = tl.load(KS + tile * 64 + n).to(tl.float32)
                value = x.to(tl.float32) * qs[:, None] * ks[None, :]
                weighted = tl.maximum(value, 0.0) * w[:, None]
                score = tl.sum(tl.reshape(weighted, (16, 4, 64)), axis=1)
                live = i * 64 + n[None, :] < visible[:, None]
                tl.store(
                    Out + tile * 1024 + q[:, None] * 64 + n[None, :],
                    tl.where(live, score, -float("inf")),
                )

    plan = PLANS[args.case]
    inputs = make_inputs(plan)
    values = {k: v.to(device) for k, v in inputs.items()}
    first, count = CHUNKS[args.case]
    width = plan["logits_width"]
    score = torch.empty((count, width), device=device, dtype=torch.float32)
    adapter = PreparedSharedMapInt8Selection(
        values,
        torch.empty((plan["q_shape"][0], 512), device=device, dtype=torch.int32),
        score,
        torch.empty(1024**2, device=device, dtype=torch.uint8),
        None,
        plan["starts"],
        key_group4=True,
        single_page64=True,
    )
    route = next(
        r for r in adapter.route_checks if (r["first"], r["count"]) == (first, count)
    )
    bounded = args.case == "int8_budget_exact"
    if not route["page_reuse"] or route["skip_invisible"] != bounded:
        raise AssertionError(f"Unexpected score route: {route}")
    work = traffic(
        width,
        inputs["visible"][first : first + count].tolist(),
        inputs["row_table"][first : first + count].tolist(),
        bounded,
    )
    visibility = route["visibility_tiles"]
    if (
        work["active_dot_tiles"] != visibility["useful_tiles"]
        or work["dot_tiles"] + work["fill_only_tiles"] != visibility["issued_tiles"]
    ):
        raise AssertionError(
            f"Work accounting disagrees with the dispatch: {visibility}"
        )
    result.update(route=route, width=width, work=work)
    print("SOL_WORK", args.case, json.dumps(work), flush=True)

    gen = torch.Generator().manual_seed(20261106)

    def integers(low, high, shape, dtype):
        return torch.randint(low, high, shape, generator=gen, dtype=dtype)

    q = integers(-16, 17, (UNITS, 64, 128), torch.int8)
    k = integers(-16, 17, (TILES, 64, 128), torch.int8)
    dot = integers(-1024, 1025, (TILES, 64, 64), torch.int32)
    qs = torch.rand((UNITS, 64), generator=gen).half()
    ks = torch.rand((TILES, 64), generator=gen).half()
    w = (torch.rand((UNITS, 64), generator=gen) * 2 - 1).half()
    vis = integers(384, 513, (UNITS, 16), torch.int32)
    q_npu, k_npu, dot_npu, qs_npu, ks_npu, w_npu, vis_npu = [
        x.to(device) for x in (q, k, dot, qs, ks, w, vis)
    ]
    copy = integers(-128, 128, (COPY_BYTES,), torch.int8).to(device)
    targets = dict(
        current=score,
        copy=torch.empty_like(copy),
        cube=torch.empty((UNITS, 64, 64), device=device, dtype=torch.int32),
        vector=torch.empty((TILES, 16, 64), device=device, dtype=torch.float32),
    )
    grid = (GRID,)
    arms = {
        "current": lambda out: adapter.score_into(first, count, out),
        "copy": lambda out: _sol_copy[grid](copy, out, COPY_BYTES, num_warps=4),
        "cube": lambda out: _sol_cube[grid](q_npu, k_npu, out, UNITS, num_warps=4),
        "vector": lambda out: _sol_vector[grid](
            dot_npu, qs_npu, ks_npu, w_npu, vis_npu, out, UNITS, num_warps=4
        ),
    }

    def launch(name, target):
        return arms[name](target)

    def validate():
        torch.npu.synchronize()
        checks = check_chunk(inputs, score.cpu(), first, count)
        if not torch.equal(copy.cpu(), targets["copy"].cpu()):
            raise AssertionError("Streaming copy mismatch")
        for unit in (0, UNITS // 2, UNITS - 1):
            start = unit * 8
            keys = k[start : start + 8].float()
            expected = (q[unit].float() @ keys.transpose(1, 2)).sum(dim=0).int()
            if not torch.equal(targets["cube"][unit].cpu(), expected):
                raise AssertionError(f"Cube mismatch at unit {unit}")
            scaled = (
                dot[start : start + 8].float()
                * qs[unit, :, None].float()
                * ks[start : start + 8, None, :].float()
            )
            want = (
                (scaled.relu() * w[unit, :, None].float()).reshape(8, 16, 4, 64).sum(2)
            )
            live = torch.arange(512).reshape(8, 1, 64) < vis[unit, None, :, None]
            want.masked_fill_(~live, -float("inf"))
            torch.testing.assert_close(
                targets["vector"][start : start + 8].cpu(), want, rtol=1e-3, atol=1e-3
            )
        return checks

    names = {}
    result["compiler"] = {}
    for arm in arms:
        compiled = launch(arm, targets[arm])
        names[arm] = {compiled.metadata.kernel_name: 1}
        result["compiler"][arm] = str(compiled.metadata)
        ttir = compiled.asm["ttir"]
        if ttir.count("tt.dot") != int(arm in ("current", "cube")):
            raise AssertionError(f"Unexpected dot count in {arm}")
        for key in ("ttir", "ttadapter", "bcmlir"):
            asm = compiled.asm.get(key)
            if isinstance(asm, str):
                (Path(args.report) / f"{args.case}-{arm}-{key}.txt").write_text(asm)
    result["precheck"] = validate()
    result["kernel_timer"] = {}
    samples, _ = measure_prefix_plan(
        args, arms, launch, targets, torch, result, expected_hosts=names
    )
    result["postcheck"] = validate()
    med = {
        arm: statistics.median(t for row in rounds for t in row["active_invocation_us"])
        for arm, rounds in samples.items()
    }
    copy_bytes_us = (2 * COPY_BYTES) / med["copy"]
    macs_us = TILES * MACS / med["cube"]
    vector_tiles_us = TILES / med["vector"]
    refs = dict(
        dense_io_with_perfect_reuse=work["perfect_reuse_io_bytes"] / copy_bytes_us,
        source_io_without_reuse=work["requested_io_bytes_upper"] / copy_bytes_us,
        useful_cube=work["useful_macs"] / macs_us,
        issued_cube=work["executed_macs"] / macs_us,
        issued_vector_replay=work["dot_tiles"] / vector_tiles_us,
    )
    limiting_reference = max(
        ("dense_io_with_perfect_reuse", "issued_cube", "issued_vector_replay"),
        key=refs.__getitem__,
    )
    result.update(
        samples=samples,
        median_us=med,
        copy_gb_s=copy_bytes_us / 1000,
        cube_tmac_s=macs_us / 1e6,
        vector_tiles_per_us=vector_tiles_us,
        component_reference_us=refs,
        limiting_reference=limiting_reference,
        ideal_overlap_reference_us=refs[limiting_reference],
        current_over_ideal_overlap_reference=med["current"] / refs[limiting_reference],
        current_over_component_reference={
            k: med["current"] / v for k, v in refs.items()
        },
        scope="Measured diagnostic references, not physical lower bounds; "
        "copy uses contiguous read plus write, cube accumulates eight tiles per sink, "
        "vector replays materialized INT32 with a compact tile-major store",
    )
    print(
        "SOL_RESULT",
        args.case,
        json.dumps(
            {
                k: result[k]
                for k in (
                    "median_us",
                    "copy_gb_s",
                    "cube_tmac_s",
                    "vector_tiles_per_us",
                    "component_reference_us",
                    "limiting_reference",
                    "ideal_overlap_reference_us",
                    "current_over_ideal_overlap_reference",
                    "current_over_component_reference",
                )
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--kernel-warmup", type=int, default=2)
    parser.add_argument("--kernel-active", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--case", choices=CHUNKS)
    args = parser.parse_args()
    if min(args.rounds, args.kernel_warmup, args.kernel_active, args.timeout) < 1:
        parser.error("Positive timing counts required")
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    if args.case:
        result = {"case": args.case, "exit_code": 0}
        try:
            run_case(args, result)
        except ModuleNotFoundError:
            result.update(exit_code=2, traceback=traceback.format_exc())
        except Exception:
            result.update(exit_code=1, traceback=traceback.format_exc())
        if "traceback" in result:
            print(result["traceback"], flush=True)
        (report / f"{args.case}.json").write_text(json.dumps(result, indent=2) + "\n")
        return result["exit_code"]

    results = []
    for case in CHUNKS:
        path = report / f"{case}.json"
        path.unlink(missing_ok=True)
        command = [sys.executable, "-u", __file__, *sys.argv[1:], "--case", case]
        status, timed_out = run_child(
            command, report / f"{case}-worker.txt", args.timeout
        )
        result = json.loads(path.read_text()) if path.exists() else {"case": case}
        if timed_out or status not in (0, 2) or "exit_code" not in result:
            result["exit_code"] = 1
        result.update(worker_status=status, timed_out=timed_out)
        results.append(result)
    codes = {r["exit_code"] for r in results}
    code = 1 if 1 in codes else 2 if 2 in codes else 0
    (report / "summary.json").write_text(
        json.dumps(dict(cases=results, exit_code=code), indent=2) + "\n"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
