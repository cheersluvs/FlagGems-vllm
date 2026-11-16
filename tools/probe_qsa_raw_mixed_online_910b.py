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
    args = parser.parse_args()
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
        from qsa_raw_mixed_online_910b import geometry, source
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
        path.write_text(runtime_source() + source())
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
        for width in (512, 1024, 4096):
            entry = dict(width=width, status="running", kernel_timer={}, arms={})
            result["cases"].append(entry)
            result["phase"] = f"fixture-{width}"
            save()
            gen = torch.Generator().manual_seed(13 + width)
            pages = (width + 191) // 192
            q = torch.randint(-8, 9, (16, 4, 128), generator=gen, dtype=torch.int8)
            q[3].zero_()
            cache = torch.randint(
                -8, 9, (pages, 192, 1, 128), generator=gen, dtype=torch.int8
            )
            table = torch.arange(pages - 1, -1, -1, dtype=torch.int32).repeat(16, 1)
            qs = (torch.rand((16, 4), generator=gen) / 64 + 1 / 64).half()
            ks = (torch.rand((pages, 192, 1), generator=gen) / 64 + 1 / 64).half()
            weights = torch.ones((16, 4), dtype=torch.float16)
            visible = torch.tensor(
                ([0, 1, 63, 64, 65, 511, 512, 513, width - 64, width] * 2)[:16],
                dtype=torch.int32,
            ).clamp(max=width)
            values = [x.to(device) for x in (q, cache, table, qs, ks, weights, visible)]
            qd, kd, td, qsd, ksd, wd, vd = values
            tiles = width // 64
            pq = torch.empty((1, 8192), dtype=torch.int8, device=device)
            pw = torch.empty((1, 1024), dtype=torch.float16, device=device)
            pk = torch.empty((tiles, 8192), dtype=torch.int8, device=device)
            lks = torch.empty((width,), dtype=torch.float16, device=device)
            raw = torch.empty((1, tiles, 16, 64), dtype=torch.float32, device=device)
            expected_d = torch.empty((16, width), dtype=torch.float32, device=device)
            module.raw_query16_qw[(1,)](
                qd, qsd, wd, pq, pw, ROWS=16, QUERIES=16, num_warps=4
            )
            module.raw_pack_k_ks[(20,)](
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
                GROUPS=1,
                TILES=tiles,
                CORES=min(20, tiles),
                QUERIES=16,
                num_warps=4,
                grid=(min(20, tiles),),
            )
            ref_mode = record_launch_mode(ref)
            assert (
                ref_mode["mix_mode"] == "aic" and ref_mode["bs_task_type"] == 20
            ), ref_mode
            module.raw_query16_cube[(min(20, tiles),)](
                pq,
                pk,
                pw,
                raw,
                GROUPS=1,
                TILES=tiles,
                CORES=min(20, tiles),
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
                ROWS=16,
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
            for row in range(16):
                dot = q[row].int() @ keys.T
                oracle = (
                    (dot.float() / 1024).relu().half().float()
                    * (qs[row].float() * 1024).half().float()[:, None]
                ).sum(0) * scales
                n = int(visible[row])
                torch.testing.assert_close(
                    expected[row, :n], oracle[:n], rtol=1e-3, atol=1e-3
                )
            arms = {}
            hosts = {}
            # Compile every arm and both trace settings before the first target launch.
            for slots in (1, 2):
                label = f"slots{slots}"
                g = geometry(width, slots)
                ring = torch.empty(
                    (g["ring_elements"],), dtype=torch.float32, device=device
                )
                idx = torch.empty((16, 512), dtype=torch.int32, device=device)
                val = torch.empty((16, 512), dtype=torch.float32, device=device)
                trace = torch.empty((16, width), dtype=torch.float32, device=device)
                operands = (pq, pk, pw, lks, vd, ring, idx, val, trace)
                binaries = {}
                for traced in (False, True):
                    result["phase"] = f"compile-{width}-{label}-trace{traced}"
                    save()
                    compiled = module.raw_mixed_online.warmup(
                        *operands,
                        WIDTH=width,
                        SLOTS=slots,
                        TRACE=traced,
                        num_warps=4,
                        grid=(1,),
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
                            f"mixed-w{width}-{label}-trace{int(traced)}",
                        ),
                    )
                    if not traced:
                        hosts[label] = {compiled.metadata.kernel_name: 1}

                def launch(traced=False, operands=operands, slots=slots):
                    return module.raw_mixed_online[(1,)](
                        *operands, WIDTH=width, SLOTS=slots, TRACE=traced, num_warps=4
                    )

                arms[label] = dict(
                    launch=launch, indices=idx, values=val, trace=trace, ring=ring
                )
                entry["arms"][label] = dict(geometry=g, binaries=binaries)

            def validate(phase):
                checks = {}
                for label, arm in arms.items():
                    result["phase"] = f"launch-{width}-{label}-{phase}"
                    save()
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
                        indices = arm["indices"].cpu()
                        check_rank(torch, indices, expected, visible, rtol=0, atol=0)
                        torch.testing.assert_close(
                            arm["values"].cpu(),
                            expected.topk(512, dim=1).values,
                            rtol=0,
                            atol=0,
                        )
                    checks[label] = dict(
                        passed=True,
                        rows=16,
                        steps=width // 512,
                        trace_scores_exact=True,
                        selected_scores_exact=True,
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
                case=f"raw-mixed-{width}",
                prefix_guard_ms=5,
            )
            result["phase"] = f"timing-{width}"
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
            entry.update(
                samples=samples,
                mean_us=med,
                serial_over_buffered=med["slots1"] / med["slots2"],
            )
            entry["postcheck"] = validate("after")
            if width == 4096:
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
                width,
                json.dumps(
                    dict(
                        mean_us=med,
                        speedup=entry["serial_over_buffered"],
                        overlap_proven=False,
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
