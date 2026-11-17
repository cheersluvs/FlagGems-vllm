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

"""Gate raw custom-op mixed composability before any full QSA integration."""

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-mixed-online", action="store_true", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, choices=(1, 2), default=2)
    parser.add_argument("--boundaries", action="store_true")
    parser.add_argument("--cube-block512", action="store_true")
    parser.add_argument("--critical-path", action="store_true")
    args = parser.parse_args()
    if args.critical_path and not args.cube_block512:
        parser.error("--critical-path requires --cube-block512")
    if args.cube_block512 and not args.boundaries:
        parser.error("--cube-block512 requires --boundaries")
    report = args.report
    report.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="running",
        phase="imports",
        production_changed=False,
        overlap_proven=False,
        pipeline_proven=False,
        scope="16-row prepared mixed raw+online TopK feasibility; no full-input/CANN mean",
        native_builds={},
        cases=[],
    )

    def save():
        (report / "raw-cube-chain.json").write_text(json.dumps(result, indent=2))

    save()
    try:
        import torch
        import torch_npu
        import triton
        from probe_qsa_raw_cube_chain_910b import (
            record_launch_mode,
            save_asm,
            verify_compiler_candidate,
        )
        from qsa_cube_query_pack_probe_910b import native_module
        from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan
        from qsa_raw_mixed_online_910b import (
            boundary_cases,
            boundary_geometry,
            boundary_source,
            geometry,
            source,
        )
        from qsa_raw_serial_selection_910b_trial import check_rank
        from qsa_raw_vector_profile_910b import hardware_pass
        from qsa_ten_score_suite_910b import runtime_source
        from triton.backends.ascend import compiler

        assert verify_compiler_candidate(compiler, triton, report, result)
        torch.npu.set_device(args.device)
        assert "910B" in torch.npu.get_device_name(args.device)
        torch.set_num_threads(4)
        device = f"npu:{args.device}"
        result["environment"] = dict(
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            triton=triton.__version__,
            source=triton.__file__,
            device=torch.npu.get_device_name(args.device),
        )
        result["phase"] = "native_build"
        save()
        native = native_module(report, 16, result, save)
        path = report / "raw-mixed-online-kernels.py"
        text = (
            runtime_source() + source() + (boundary_source() if args.boundaries else "")
        )
        if args.cube_block512:
            from qsa_raw_cube_block512_910b import source as block512_source

            text += block512_source()
        if args.critical_path:
            from qsa_raw_critical_path_910b import source as critical_source

            text += critical_source()
        path.write_text(text)
        path.with_suffix(".py.txt").write_text(path.read_text())
        spec = importlib.util.spec_from_file_location("_qsa_raw_mixed_online", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.chain_op = native.chain_op
        from qsa_raw_mixed_abi_910b import bind_mixed_chain

        result["phase"] = "mixed_abi_build"
        save()
        module.mixed_chain_op = bind_mixed_chain(report, result, save)
        if args.cube_block512:
            from qsa_raw_cube_block512_910b import bind_block512

            result["phase"] = "block512_native_build"
            save()
            module.block512_chain_op = bind_block512(report, result, save)
        if args.critical_path:
            from qsa_raw_critical_path_910b import bind_tick

            result["phase"] = "replay_tick_build"
            save()
            module.replay_tick_op = bind_tick(report, result, save)
        fixtures = (
            boundary_cases() if args.boundaries else ((16, 512), (16, 1024), (16, 4096))
        )
        if args.boundaries:
            result["scope"] = (
                "Prepared multi-group/tail mixed raw+online TopK feasibility; no full-input/CANN mean"
            )
        result["mode"] = (
            "query_and_key_boundaries"
            if args.boundaries
            else "single_group_whole_updates"
        )
        if args.cube_block512:
            fixtures = ((16, 512), (17, 8256), (112, 29696))
            result["scope"] = (
                "Prepared two-slot baseline64/block512 pair; no full-input/CANN acceptance mean"
            )
            result["mode"] = "cube_block512_reuse"
        if args.critical_path:
            result["scope"] = (
                "Prepared mixed critical-path ablations; no additive phase timing or acceptance ratio"
            )
            result["mode"] = "critical_path"
            result["limitations"] = [
                "Light consumer retains scaling/masking plus checksum reduction and GM stores",
                "Replay preserves control events but bypasses ring payload writes; "
                "full raw-buffer cache footprint differs",
                "No inference of exact internal phase duration or proven overlap",
            ]
        for rows, width in fixtures:
            case_name = f"r{rows}-w{width}" if args.boundaries else str(width)
            shape = boundary_geometry(rows, width, 1)
            groups, tiles = shape["groups"], shape["tiles"]
            entry = dict(
                rows=rows, width=width, status="running", kernel_timer={}, arms={}
            )
            result["cases"].append(entry)
            result["phase"] = f"fixture-{case_name}"
            save()
            gen = torch.Generator().manual_seed(13 + width)
            pages = (width + 191) // 192
            q = torch.randint(-8, 9, (rows, 4, 128), generator=gen, dtype=torch.int8)
            if rows > 3:
                q[3].zero_()
            cache = torch.randint(
                -8, 9, (pages, 192, 1, 128), generator=gen, dtype=torch.int8
            )
            table = torch.arange(pages - 1, -1, -1, dtype=torch.int32).repeat(rows, 1)
            qs = (torch.rand((rows, 4), generator=gen) / 64 + 1 / 64).half()
            ks = (torch.rand((pages, 192, 1), generator=gen) / 64 + 1 / 64).half()
            weights = torch.ones((rows, 4), dtype=torch.float16)
            visible = torch.tensor(
                (
                    [0, 1, 63, 64, 65, 511, 512, 513, width - 64, width]
                    * ((rows + 9) // 10)
                )[:rows],
                dtype=torch.int32,
            ).clamp(max=width)
            if args.boundaries:
                visible[-1] = width
            values = [x.to(device) for x in (q, cache, table, qs, ks, weights, visible)]
            qd, kd, td, qsd, ksd, wd, vd = values
            pq = torch.empty((groups, 8192), dtype=torch.int8, device=device)
            pw = torch.empty((groups, 1024), dtype=torch.float16, device=device)
            pk = torch.empty((tiles, 8192), dtype=torch.int8, device=device)
            lks = torch.empty(
                (shape["padded_width"],), dtype=torch.float16, device=device
            )
            raw = torch.empty(
                (groups, tiles, 16, 64), dtype=torch.float32, device=device
            )
            expected_d = torch.empty((rows, width), dtype=torch.float32, device=device)
            module.raw_query16_qw[(groups,)](
                qd, qsd, wd, pq, pw, ROWS=rows, QUERIES=16, num_warps=4
            )
            pack_k = (
                module.raw_pack_k_ks_padded if args.boundaries else module.raw_pack_k_ks
            )
            pack_k[(20,)](
                kd,
                ksd,
                td,
                pk,
                lks,
                MAX_VISIBLE=width,
                TILES=tiles,
                CORES=20,
                num_warps=4,
            )
            ref = module.raw_query16_cube.warmup(
                pq,
                pk,
                pw,
                raw,
                GROUPS=groups,
                TILES=tiles,
                CORES=min(20, groups * tiles),
                QUERIES=16,
                num_warps=4,
                grid=(min(20, groups * tiles),),
            )
            ref_mode = record_launch_mode(ref)
            assert (
                ref_mode["mix_mode"] == "aic" and ref_mode["bs_task_type"] == 20
            ), ref_mode
            module.raw_query16_cube[(min(20, groups * tiles),)](
                pq,
                pk,
                pw,
                raw,
                GROUPS=groups,
                TILES=tiles,
                CORES=min(20, groups * tiles),
                QUERIES=16,
                num_warps=4,
            )
            module.raw_finish_logical1024[(20,)](
                raw,
                ksd,
                td,
                vd,
                expected_d,
                lks,
                ROWS=rows,
                WIDTH=width,
                RAW_TILES=tiles,
                CORES=20,
                num_warps=4,
            )
            torch.npu.synchronize()
            expected = expected_d.cpu()
            # Independent CPU half-chain oracle, including NZ and weight packing.
            keys = cache[table[0].long(), :, 0].reshape(-1, 128)[:width].int()
            scales = ks[table[0].long(), :, 0].reshape(-1)[:width].float()
            for row in range(rows):
                dot = q[row].int() @ keys.T
                oracle = (
                    (dot.float() / 1024).relu().half().float()
                    * (qs[row].float() * 1024).half().float()[:, None]
                ).sum(0) * scales
                n = int(visible[row])
                torch.testing.assert_close(
                    expected[row, :n], oracle[:n], rtol=1e-3, atol=1e-3
                )
            packed_keys = torch.zeros((shape["padded_width"], 128), dtype=torch.int8)
            packed_keys[:width] = keys.to(torch.int8)
            expected_pk = (
                packed_keys.view(tiles, 2, 32, 4, 32)
                .permute(0, 3, 1, 2, 4)
                .contiguous()
                .view(tiles, 8192)
            )
            expected_ks = torch.zeros((shape["padded_width"],), dtype=torch.float16)
            expected_ks[:width] = scales.half()
            torch.testing.assert_close(pk.cpu(), expected_pk, rtol=0, atol=0)
            torch.testing.assert_close(lks.cpu(), expected_ks, rtol=0, atol=0)
            entry["packing_check"] = dict(
                k_exact=True,
                ks_exact=True,
                padded_width=shape["padded_width"],
                table_columns=pages,
            )
            arms = {}
            hosts = {}
            # Compile every arm and both trace settings before the first target launch.
            configurations = (
                ((2, "baseline64"), (2, "block512"))
                if args.cube_block512
                else ((1, "slots1"), (2, "slots2"))
            )
            if args.critical_path:
                configurations = ((2, "full"), (2, "light"), (2, "replay"))
            for slots, label in configurations:
                g = (
                    boundary_geometry(rows, width, slots)
                    if args.boundaries
                    else geometry(width, slots)
                )
                ring = torch.empty(
                    (g["ring_elements"],), dtype=torch.float32, device=device
                )
                guards = []

                def guarded(columns, dtype, padded_columns):
                    logical = rows * columns
                    capacity = shape["padded_rows"] * padded_columns
                    storage = torch.empty(
                        (capacity + 2048,), dtype=dtype, device=device
                    )
                    guards.append((storage, logical))
                    return storage[1024 : 1024 + logical].view(rows, columns)

                idx = guarded(512, torch.int32, 512)
                val = guarded(512, torch.float32, 512)
                trace = guarded(width, torch.float32, shape["padded_width"])
                operands = (pq, pk, pw, lks, vd, ring, idx, val, trace)
                audit = None
                if args.critical_path:
                    audit = guarded(shape["steps"], torch.float32, shape["steps"])
                    operands += (raw, audit)
                kernel = (
                    module.raw_mixed_boundaries
                    if args.boundaries
                    else module.raw_mixed_online
                )
                if label == "block512":
                    kernel = module.raw_mixed_block512
                if args.critical_path:
                    kernel = getattr(module, "raw_critical_" + label)
                meta = dict(WIDTH=width, SLOTS=slots, num_warps=4)
                if args.boundaries:
                    meta["ROWS"] = rows
                binaries = {}
                for traced in (False, True):
                    result["phase"] = f"compile-{case_name}-{label}-trace{traced}"
                    save()
                    compiled = kernel.warmup(
                        *operands, TRACE=traced, grid=(groups,), **meta
                    )
                    mode = record_launch_mode(compiled)
                    if mode["mix_mode"] != "mix" or mode["bs_task_type"] // 10 not in (
                        3,
                        4,
                    ):
                        raise RuntimeError(f"Expected mixed Cube/Vector binary: {mode}")
                    binaries[str(traced)] = dict(
                        launch=mode,
                        asm=save_asm(
                            compiled,
                            report,
                            f"mixed-{case_name}-{label}-trace{int(traced)}",
                        ),
                    )
                    if not traced:
                        hosts[label] = {compiled.metadata.kernel_name: 1}

                def launch(
                    traced=False,
                    operands=operands,
                    kernel=kernel,
                    meta=meta,
                    groups=groups,
                ):
                    return kernel[(groups,)](*operands, TRACE=traced, **meta)

                arms[label] = dict(
                    launch=launch,
                    indices=idx,
                    values=val,
                    trace=trace,
                    ring=ring,
                    guards=guards,
                    audit=audit,
                )
                entry["arms"][label] = dict(geometry=g, binaries=binaries)

            def validate(phase):
                checks = {}
                for label, arm in arms.items():
                    result["phase"] = f"launch-{case_name}-{label}-{phase}"
                    save()
                    for storage, _ in arm["guards"]:
                        storage.fill_(
                            -3 if storage.dtype == torch.int32 else float("nan")
                        )
                    arm["ring"].fill_(float("nan"))
                    arm["indices"].fill_(-3)
                    arm["values"].fill_(float("nan"))
                    arm["trace"].fill_(float("nan"))
                    arm["launch"](True)
                    torch.npu.synchronize()
                    torch.testing.assert_close(
                        arm["trace"].cpu(), expected, rtol=0, atol=0
                    )
                    for traced in (True, False):
                        if not traced:
                            arm["ring"].fill_(float("nan"))
                            arm["indices"].fill_(-3)
                            arm["values"].fill_(float("nan"))
                            arm["launch"](False)
                            torch.npu.synchronize()
                        for storage, logical in arm["guards"]:
                            for region in (storage[:1024], storage[1024 + logical :]):
                                cpu_guard = region.cpu()
                                assert (
                                    bool((cpu_guard == -3).all())
                                    if storage.dtype == torch.int32
                                    else bool(torch.isnan(cpu_guard).all())
                                ), "Output guard overwritten"
                        if args.critical_path and label == "light":
                            padded = torch.zeros(
                                (rows, shape["padded_width"]), dtype=torch.float32
                            )
                            padded[:, :width] = torch.where(
                                torch.isfinite(expected), expected, 0
                            )
                            torch.testing.assert_close(
                                arm["audit"].cpu(),
                                padded.view(rows, shape["steps"], 512).sum(2),
                                rtol=1e-3,
                                atol=1e-3,
                            )
                            assert bool((arm["indices"].cpu() == -3).all())
                            assert bool(torch.isnan(arm["values"].cpu()).all())
                        else:
                            indices = arm["indices"].cpu()
                            check_rank(
                                torch, indices, expected, visible, rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                arm["values"].cpu(),
                                expected.topk(512, dim=1).values,
                                rtol=0,
                                atol=0,
                            )
                            if args.critical_path and label == "replay":
                                ticks = (
                                    arm["audit"]
                                    .cpu()
                                    .reshape(-1)
                                    .view(torch.int32)[:groups]
                                )
                                assert bool(
                                    (ticks == shape["steps"] - 1).all()
                                ), "Replay AIC progress missing"
                    checks[label] = dict(
                        passed=True,
                        rows=rows,
                        steps=shape["steps"],
                        output_guards_unchanged=True,
                        trace_scores_exact=True,
                        selected_scores_exact=not (
                            args.critical_path and label == "light"
                        ),
                        checksum_checked=bool(args.critical_path and label == "light"),
                        replay_progress_checked=bool(
                            args.critical_path and label == "replay"
                        ),
                    )
                    print(
                        "RAW_MIXED_CHECK",
                        width,
                        phase,
                        label,
                        checks[label],
                        flush=True,
                    )
                return checks

            entry["precheck"] = validate("before")
            save()
            options = SimpleNamespace(
                rounds=args.rounds,
                kernel_active=3,
                kernel_warmup=1,
                report=str(report),
                case=f"raw-mixed-{case_name}",
                prefix_guard_ms=5,
            )
            result["phase"] = f"timing-{case_name}"
            save()
            samples, _ = measure_prefix_plan(
                options,
                arms,
                lambda name, _: arms[name]["launch"](),
                dict.fromkeys(arms),
                torch,
                entry,
                expected_hosts=hosts,
            )
            med = {
                name: sum(s["kernel_us"] for s in arr) / len(arr)
                for name, arr in samples.items()
            }
            numerator, denominator = (
                ("full", "light")
                if args.critical_path
                else (
                    ("baseline64", "block512")
                    if args.cube_block512
                    else ("slots1", "slots2")
                )
            )
            ratio = med[numerator] / med[denominator]
            entry.update(samples=samples, mean_us=med)
            entry[
                "call64_over_block512" if args.cube_block512 else "serial_over_buffered"
            ] = ratio
            if args.critical_path:
                entry["ablation_ratios"] = dict(
                    full_over_light=med["full"] / med["light"],
                    full_over_replay=med["full"] / med["replay"],
                    scope="Diagnostic work removal; not internal stage times",
                )
                entry.pop("call64_over_block512", None)
            if args.cube_block512 and not args.critical_path:
                entry["expected_native_calls"] = dict(
                    baseline64=groups * shape["steps"] * 8,
                    block512=groups * shape["steps"],
                    scope="Source schedule count; not measured hardware instruction count",
                )
            entry["postcheck"] = validate("after")
            if width == 4096 and not args.boundaries:
                entry["hardware"] = {}
                for label, arm in arms.items():
                    result["phase"] = f"hardware-{label}"
                    save()
                    try:
                        entry["hardware"][label] = hardware_pass(
                            SimpleNamespace(counter_iterations=1),
                            f"raw-mixed-{width}",
                            label,
                            arm["launch"],
                            next(iter(hosts[label])),
                            torch,
                            report,
                        )
                    except Exception as exc:
                        entry["hardware"][label] = dict(
                            status="collection_failed", error=repr(exc)
                        )
                        print("RAW_MIXED_COUNTER_FAILED", label, repr(exc), flush=True)
                        break
            entry["status"] = "passed"
            save()
            print(
                "RAW_MIXED_RESULT",
                case_name,
                json.dumps(
                    dict(
                        mean_us=med,
                        speedup=ratio,
                        overlap_proven=False,
                        ablation_ratios=entry.get("ablation_ratios"),
                    )
                ),
                flush=True,
            )
        result.update(
            status="passed",
            phase="completed",
            outcome="mixed_numerical_and_pair_timing_passed_overlap_unproven",
        )
        save()
        return 0
    except Exception as exc:
        result.update(
            status="failed", error=repr(exc), traceback=traceback.format_exc()
        )
        save()
        print("RAW_MIXED_FAILED", result["phase"], repr(exc), flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
