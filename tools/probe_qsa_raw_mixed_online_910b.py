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
    parser.add_argument("--topk-prefix", action="store_true")
    parser.add_argument("--sort-buffers", action="store_true")
    parser.add_argument("--tle-sort512", action="store_true")
    parser.add_argument("--history-full-merge", action="store_true")
    parser.add_argument("--history-boundary", action="store_true")
    parser.add_argument("--query-batch", type=int, choices=(2, 4))
    parser.add_argument("--history-direct", action="store_true")
    parser.add_argument("--single-block", action="store_true")
    parser.add_argument("--first-block-seed", action="store_true")
    paired = parser.add_mutually_exclusive_group()
    paired.add_argument("--paired-vector-profile", action="store_true")
    paired.add_argument("--paired-sort-barrier", action="store_true")
    paired.add_argument("--paired-sort32-barrier", action="store_true")
    paired.add_argument("--paired-ub-audit", action="store_true")
    paired.add_argument("--paired-final-mrgsort-ub", action="store_true")
    paired.add_argument("--paired-final-mrgsort-poison-check", action="store_true")
    paired.add_argument("--paired-final-mrgsort-reuse", action="store_true")
    args = parser.parse_args()
    final_mrgsort_probe = (
        args.paired_final_mrgsort_ub
        or args.paired_final_mrgsort_poison_check
        or args.paired_final_mrgsort_reuse
    )
    sort_barrier_probe = args.paired_sort_barrier or args.paired_sort32_barrier
    paired_probe = (
        args.paired_vector_profile
        or sort_barrier_probe
        or args.paired_ub_audit
        or final_mrgsort_probe
    )
    removed_barrier = (
        "no_sort32_barrier" if args.paired_sort32_barrier else "no_merge_barrier"
    )
    if paired_probe and (
        not args.cube_block512
        or args.history_full_merge
        or args.critical_path
        or args.tle_sort512
        or args.sort_buffers
    ):
        parser.error("Paired probes require --cube-block512 without other variants")
    if args.first_block_seed and (
        not args.history_full_merge
        or args.history_boundary
        or args.query_batch
        or args.history_direct
        or args.single_block
    ):
        parser.error(
            "--first-block-seed requires --history-full-merge without other history variants"
        )
    if args.single_block and (
        not args.history_full_merge
        or args.history_boundary
        or args.query_batch
        or args.history_direct
    ):
        parser.error(
            "--single-block requires --history-full-merge without other history variants"
        )
    if args.history_direct and (
        not args.history_full_merge or args.history_boundary or args.query_batch
    ):
        parser.error(
            "--history-direct requires the history merge probe as its only variant"
        )
    if args.query_batch and (not args.history_full_merge or args.history_boundary):
        parser.error(
            "--query-batch requires --history-full-merge without --history-boundary"
        )
    if args.history_boundary and not args.history_full_merge:
        parser.error("--history-boundary requires --history-full-merge")
    if args.history_full_merge and (
        not args.cube_block512
        or args.critical_path
        or args.topk_prefix
        or args.sort_buffers
        or args.tle_sort512
    ):
        parser.error(
            "--history-full-merge requires --cube-block512 without other diagnostic variants"
        )
    if args.tle_sort512 and (
        not args.cube_block512
        or args.critical_path
        or args.topk_prefix
        or args.sort_buffers
    ):
        parser.error(
            "--tle-sort512 requires --cube-block512 without other diagnostic variants"
        )
    if args.sort_buffers and (
        not args.cube_block512 or args.critical_path or args.topk_prefix
    ):
        parser.error(
            "--sort-buffers requires --cube-block512 without critical/prefix modes"
        )
    if args.topk_prefix and not args.critical_path:
        parser.error("--topk-prefix requires --critical-path")
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
        if args.topk_prefix:
            from qsa_raw_topk_prefix_910b import source as prefix_source

            text += prefix_source()
        if args.sort_buffers:
            from qsa_raw_sort_buffers_910b import source as buffers_source

            text += buffers_source()
        if args.tle_sort512:
            from qsa_raw_tle_sort512_910b import source as tle_sort_source

            text += tle_sort_source()
        if args.history_full_merge:
            from qsa_raw_history_merge_910b import source as history_source

            text += history_source()
        if args.query_batch:
            from qsa_query_batch_probe_910b import source as batch_source

            text += batch_source()
        if args.history_direct:
            from qsa_history_direct_probe_910b import source as direct_source

            text += direct_source()
        if args.single_block:
            from qsa_single_block_probe_910b import source as single_source

            text += single_source()
        if args.first_block_seed:
            from qsa_first_block_seed_probe_910b import source as first_seed_source

            text += first_seed_source()
        if args.paired_vector_profile:
            from qsa_paired_history_910b import vector_profile_ir, vector_profile_source

            text += vector_profile_source()
        if sort_barrier_probe or args.paired_ub_audit:
            from qsa_paired_history_910b import sort_barrier_ir, sort_barrier_source

            text += sort_barrier_source(
                sort32_boundary=args.paired_sort32_barrier or args.paired_ub_audit
            )
        if final_mrgsort_probe:
            from qsa_final_mrgsort_ub_910b import (
                compile_capture,
                gate_output_init,
                replay_output_init,
                source as final_mrgsort_source,
            )

            text += final_mrgsort_source(reuse_local32=args.paired_final_mrgsort_reuse)
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
        if args.paired_ub_audit:
            from qsa_paired_ub_audit_910b import run_audit

            result.update(
                phase="paired_ub_audit",
                mode="paired_ub_audit",
                scope="Compiler-only audit of local sort output and paired history buffers",
            )
            save()
            audit = run_audit(torch, device, module, compiler, report)
            passed = audit["status"] == "captured"
            result.update(
                status="passed" if passed else "failed",
                phase="completed" if passed else "paired_ub_audit",
                audit="paired-ub-audit.json",
            )
            save()
            return int(not passed)
        if args.critical_path:
            from qsa_raw_critical_path_910b import TICK_STRIDE_WORDS, bind_tick

            result["phase"] = "replay_tick_build"
            save()
            module.replay_tick_op = bind_tick(report, result, save)
        if args.history_full_merge and not (args.single_block or args.first_block_seed):
            from qsa_raw_history_merge_910b import transition_checks

            result["phase"] = "history_transitions"
            save()
            result["history_transitions"] = transition_checks(
                module, torch, device, report, save_asm, record_launch_mode
            )
            save()
        if args.sort_buffers or args.tle_sort512:
            from qsa_raw_sort_buffers_910b import capture_installed

            result["installed_sort_audit"] = capture_installed(report)
            save()
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
        if args.topk_prefix:
            result["mode"] = "topk_prefix"
            result["scope"] = (
                "Sort/merge fixed-history diagnostic; no internal phase subtraction or acceptance ratio"
            )
            result["seed_contract"] = dict(
                dtype="float32",
                count=512,
                high=0.5,
                low=-0.5,
                indices="packed int32 0..511",
                history="fixed per row",
            )
            result["limitations"] = [
                "Sort/merge export rank-weighted checksums; export cost included",
                "Merge uses fixed synthetic nonempty history; full uses evolving UB history",
                "Snapshot/fallback work remains inside merge helper; commit cost not isolated",
            ]
        if args.sort_buffers:
            result["mode"] = "sort_buffer_lifetime"
            result["scope"] = (
                "Same full mixed kernel and sort algorithm; explicit buffer lifetime micro A/B"
            )
            result["limitations"] = [
                "Zero-filled operands observed in pre-device IR; hardware fill savings not assumed",
                "Poison validation is empirical for these fixtures,not a general proof",
                "No production change or CANN acceptance ratio",
            ]
        if args.tle_sort512:
            result["mode"] = "tle_sort512"
            result["scope"] = (
                "Same full mixed kernel;generic local sort versus fixed TLE primitive network"
            )
            result["limitations"] = [
                "Prepared micro comparison,not full-input or CANN acceptance",
                "Equal-score index order may differ;ranked scores and index validity checked",
            ]
        if args.history_full_merge:
            result["mode"] = "history_merge_strategies"
            result["scope"] = (
                "Fixed local sort with evolving history;snapshot,prefix-only and complete merge"
            )
            result["limitations"] = [
                "Prepared micro pair,not full-input/CANN acceptance",
                "Complete merge emits all1024 proposals;removes coupled snapshot/count/fallback work",
                "Outer persistent state commit and final unpack unchanged",
            ]
        if args.history_boundary:
            from qsa_history_policy_910b import boundary_cases as history_boundary_cases

            fixtures = history_boundary_cases()
            result["mode"] = "history_boundary_tuning"
        if args.query_batch:
            fixtures = ((16, 512), (17, 8256), (112, 29696))
            result["mode"] = "local_query_batch"
            result["scope"] = (
                "Prepared prefix mixed kernel; local sort batch1 versus selected batch; no CANN mean"
            )
            result["query_batch"] = args.query_batch
            result["limitations"] = [
                "History merge/state commits remain per query; Cube,ring and events unchanged",
                "Prepared micro timings exclude packing and are not acceptance results",
            ]
        if args.history_direct:
            fixtures = ((17, 8256), (112, 29696))
            result["mode"] = "history_direct_destination"
            result["scope"] = "Prepared prefix mixed kernel vs direct UB destination"
            result["limits"] = dict(packing_included=False, cann_acceptance=False)
        if args.single_block:
            fixtures = ((16, 512), (17, 512), (128, 512))
            result["mode"] = "single_block_topk"
            result["scope"] = (
                "Prepared mixed kernel, snapshot versus direct unpack of local sort"
            )
            result["limitations"] = [
                "WIDTH=512 only, padded visibility and row tails included",
                "Prepared timings exclude packing and full-input CANN comparison",
            ]
        if args.first_block_seed:
            fixtures = ((17, 769), (80, 1024), (17, 1281), (112, 1536))
            result["mode"] = "first_block_seed_topk"
            result["scope"] = (
                "Prepared mixed kernel, snapshot versus first-block history seed"
            )
            result["limitations"] = [
                "Peel the first Vector block, preserve Cube, ring and events",
                "Prepared timings exclude packing and full-input CANN comparison",
            ]
        if paired_probe:
            fixtures = ((128, 8192), (112, 29696))
        if args.paired_vector_profile:
            result["mode"] = "paired_vector_profile"
            result["scope"] = "Synthetic causal chunks with two slots and 512-key handoff"
            result["limitations"] = [
                "All arms include a block checksum and its write",
                "Full minus sort-only includes history setup, state updates and final export",
                "The differences are changes in mixed kernel time under overlap",
            ]
        if sort_barrier_probe:
            result["mode"] = (
                "paired_sort32_barrier"
                if args.paired_sort32_barrier
                else "paired_sort_barrier"
            )
            boundary = (
                "from sort32 to the first mrgsort"
                if args.paired_sort32_barrier
                else "between mrgsort calls"
            )
            result["scope"] = (
                f"Paired-history mixed kernel with or without the barrier {boundary}"
            )
            result["limitations"] = [
                "Barrier evidence is from pre-device backend IR",
                "Timings include the full mixed kernel on two causal chunks",
            ]
        if final_mrgsort_probe:
            result.update(
                mode=(
                    "paired_final_mrgsort_poison_check"
                    if args.paired_final_mrgsort_poison_check
                    else "paired_final_mrgsort_ub"
                ),
                scope="Complete mixed history kernel; only final local sort output storage varies",
                limitations=[
                    "Exact score/index words recorded with TRACE=True before history update",
                    "Copy evidence is from post-sync-solver IR",
                    (
                        "Correctness-only comparison of zero and poison-filled destinations"
                        if args.paired_final_mrgsort_poison_check
                        else "Timed binaries exclude the pair trace"
                    ),
                ],
            )
            if args.paired_final_mrgsort_reuse:
                result["mode"] = "paired_final_mrgsort_reuse"
                result["scope"] = "Final local mrgsort with a zero template or out=local32"
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
            if paired_probe:
                visible = torch.arange(width - rows + 1, width + 1, dtype=torch.int32)
            if final_mrgsort_probe:
                visible[:8] = torch.tensor([0, 1, 64, 65, 511, 512, 513, 1023])
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
            seed = None
            seed_values = None
            if args.topk_prefix:
                seed_values = torch.linspace(0.5, -0.5, 512, dtype=torch.float32)
                seed_cpu = torch.empty((rows, 1024), dtype=torch.float32)
                seed_cpu[:, 0::2] = seed_values
                seed_cpu[:, 1::2] = torch.arange(512, dtype=torch.int32).view(
                    torch.float32
                )
                seed = seed_cpu.to(device)
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
            if args.topk_prefix:
                configurations = ((2, "full"), (2, "sort"), (2, "merge"))
            if args.sort_buffers:
                configurations = ((2, "poison"), (2, "zero"), (2, "uninit"))
            if args.tle_sort512:
                configurations = ((2, "generic"), (2, "fixed"))
            if args.history_full_merge:
                configurations = ((2, "snapshot"), (2, "prefix"), (2, "complete"))
            if args.history_boundary:
                configurations = ((2, "snapshot"), (2, "prefix"))
            if args.query_batch:
                configurations = ((2, "prefix"), (2, "batch" + str(args.query_batch)))
            if args.history_direct:
                configurations = ((2, "prefix"), (2, "direct"))
            if args.single_block:
                configurations = ((2, "snapshot"), (2, "single"))
            if args.first_block_seed:
                configurations = ((2, "snapshot"), (2, "firstseed"))
            if args.paired_vector_profile:
                configurations = ((2, "full"), (2, "sort_only"), (2, "score_only"))
            if sort_barrier_probe:
                configurations = ((2, "full"), (2, removed_barrier))
            if final_mrgsort_probe:
                configurations = ((2, "full"), (2, "poison"))
                if args.paired_final_mrgsort_reuse:
                    configurations = ((2, "full"), (2, "reuse_local32"))
                elif not args.paired_final_mrgsort_poison_check:
                    configurations += ((2, "uninit"),)
            trace_options = (
                (True,) if args.paired_final_mrgsort_poison_check else (False, True)
            )
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
                    audit_columns = (
                        TICK_STRIDE_WORDS if label == "replay" else shape["steps"]
                    )
                    audit = guarded(audit_columns, torch.float32, audit_columns)
                    operands += (raw, audit)
                if args.topk_prefix:
                    operands += (seed,)
                kernel = (
                    module.raw_mixed_boundaries
                    if args.boundaries
                    else module.raw_mixed_online
                )
                if args.sort_buffers:
                    kernel = getattr(module, "raw_sort_" + label)
                if args.tle_sort512:
                    kernel = getattr(module, "raw_tle_sort_" + label)
                if args.history_full_merge:
                    kernel = getattr(module, "raw_history_" + label)
                if label == "block512":
                    kernel = module.raw_mixed_block512
                if args.critical_path:
                    kernel = getattr(
                        module,
                        ("raw_topk_" if args.topk_prefix else "raw_critical_") + label,
                    )
                meta = dict(WIDTH=width, SLOTS=slots, num_warps=4)
                if args.boundaries:
                    meta["ROWS"] = rows
                if args.paired_vector_profile:
                    kernel = getattr(module, "raw_paired_vector_" + label)
                    audit = guarded(shape["steps"], torch.float32, shape["steps"])
                    operands += (audit,)
                    meta["BLOCKS"] = shape["steps"]
                if sort_barrier_probe:
                    kernel_names = (
                        (
                            "raw_history_prepared_paired_no_merge_barrier",
                            "raw_history_prepared_paired_no_sort_barriers",
                        )
                        if args.paired_sort32_barrier
                        else (
                            "raw_history_prepared_paired",
                            "raw_history_prepared_paired_no_merge_barrier",
                        )
                    )
                    kernel = getattr(module, kernel_names[int(label != "full")])
                    meta["BLOCKS"] = shape["steps"]
                pair_trace = None
                if final_mrgsort_probe:
                    kernel = getattr(module, "raw_final_mrgsort_" + label)
                    meta["BLOCKS"] = shape["steps"]
                    pair_words = shape["steps"] * 1024
                    pair_trace = guarded(pair_words, torch.int32, pair_words)
                    operands += (pair_trace,)
                binaries = {}
                for traced in trace_options:
                    result["phase"] = f"compile-{case_name}-{label}-trace{traced}"
                    save()
                    compiler_call = None
                    if final_mrgsort_probe and traced == (label == "poison"):
                        capture_label = f"mrgsort-{case_name}-{label}-trace{int(traced)}"
                        compiled, compiler_call = compile_capture(
                            compiler,
                            report,
                            capture_label,
                            lambda: kernel.warmup(
                                *operands, TRACE=traced, grid=(groups,), **meta
                            ),
                        )
                    else:
                        compiled = kernel.warmup(
                            *operands, TRACE=traced, grid=(groups,), **meta
                        )
                    mode = record_launch_mode(compiled)
                    if mode["mix_mode"] != "mix" or mode["bs_task_type"] // 10 not in (
                        3,
                        4,
                    ):
                        raise RuntimeError(f"Expected mixed Cube/Vector binary: {mode}")
                    if args.sort_buffers:
                        from qsa_raw_sort_buffers_910b import sort_ir_evidence

                        entry.setdefault("sort_ir", {}).setdefault(label, {})[
                            str(traced)
                        ] = sort_ir_evidence(compiled)
                    if args.tle_sort512:
                        from qsa_raw_tle_sort512_910b import ir_evidence

                        entry.setdefault("primitive_ir", {}).setdefault(label, {})[
                            str(traced)
                        ] = ir_evidence(compiled, label == "fixed")
                    if args.history_full_merge:
                        from qsa_raw_history_merge_910b import ir_evidence as history_ir

                        if args.query_batch:
                            entry.setdefault("query_batch_compilation", {}).setdefault(
                                label, {}
                            )[str(traced)] = dict(
                                launch=mode,
                                asm=save_asm(
                                    compiled,
                                    report,
                                    f"batch-proof-{case_name}-{label}-trace{int(traced)}",
                                ),
                            )
                            save()
                        if args.query_batch and label != "prefix":
                            from qsa_query_batch_probe_910b import (
                                ir_evidence as batch_ir,
                            )

                            evidence = batch_ir(compiled, int(label[-1]))
                        elif args.history_direct:
                            from qsa_history_direct_probe_910b import (
                                ir_evidence as direct_ir,
                            )

                            asm = save_asm(
                                compiled,
                                report,
                                f"history-direct-{case_name}-{label}-trace{int(traced)}",
                            )
                            entry.setdefault(
                                "history_direct_compilation", {}
                            ).setdefault(label, {})[str(traced)] = dict(
                                launch=mode, asm=asm
                            )
                            save()
                            evidence = direct_ir(compiled, direct=label == "direct")
                        elif args.single_block and label == "single":
                            from qsa_single_block_probe_910b import (
                                ir_evidence as single_ir,
                            )

                            entry["single_compilation_" + str(traced)] = dict(
                                launch=mode,
                                asm=save_asm(
                                    compiled,
                                    report,
                                    f"single-{case_name}-trace{int(traced)}",
                                ),
                            )
                            save()
                            evidence = single_ir(compiled)
                        elif args.first_block_seed and label == "firstseed":
                            from qsa_first_block_seed_probe_910b import (
                                ir_evidence as first_seed_ir,
                            )

                            entry.setdefault("first_seed_compilation", {})[
                                str(traced)
                            ] = dict(
                                launch=mode,
                                asm=save_asm(
                                    compiled,
                                    report,
                                    f"first-seed-{case_name}-trace{int(traced)}",
                                ),
                            )
                            save()
                            evidence = first_seed_ir(compiled)
                        else:
                            evidence = history_ir(
                                compiled,
                                label == "complete",
                                expect_no_gather=label != "snapshot",
                            )
                        entry.setdefault("history_ir", {}).setdefault(label, {})[
                            str(traced)
                        ] = evidence
                    if args.paired_vector_profile:
                        entry.setdefault("vector_ir", {}).setdefault(label, {})[
                            str(traced)
                        ] = vector_profile_ir(compiled, label)
                    binaries[str(traced)] = dict(
                        launch=mode,
                        asm=save_asm(
                            compiled,
                            report,
                            f"mixed-{case_name}-{label}-trace{int(traced)}",
                        ),
                    )
                    if sort_barrier_probe:
                        entry["arms"][label] = dict(geometry=g, binaries=binaries)
                    if sort_barrier_probe:
                        entry.setdefault("barrier_ir", {}).setdefault(label, {})[
                            str(traced)
                        ] = sort_barrier_ir(
                            compiled, label, sort32_boundary=args.paired_sort32_barrier
                        )
                    if compiler_call is not None:
                        evidence = replay_output_init(
                            compiler, report, capture_label, compiler_call
                        )
                        entry.setdefault("mrgsort_init", {})[label] = evidence
                        save()
                        gate_output_init(evidence, label)
                        print(
                            "FINAL_MRGSORT_INIT",
                            case_name,
                            label,
                            json.dumps(dict(
                                copies=len(evidence["copies"]),
                                fills=len(evidence["fills"]),
                                reuses_sort32_buffer=evidence["reuses_sort32_buffer"],
                                aliases_final_input=evidence["aliases_final_input"],
                            )),
                            flush=True,
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
                    pair_trace=pair_trace,
                )
                entry["arms"][label] = dict(geometry=g, binaries=binaries)

            def validate(phase):
                checks = {}
                reference_pairs = None
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
                    if final_mrgsort_probe:
                        bits = arm["pair_trace"].cpu()
                        if bool((bits == -3).any()):
                            raise AssertionError("Final mrgsort pair trace has unwritten words")
                        if label == "full":
                            reference_pairs = bits
                        elif not torch.equal(bits, reference_pairs):
                            raise AssertionError(f"Final mrgsort pair words differ for {label}")
                    for traced in reversed(trace_options):
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
                        if args.paired_vector_profile:
                            tiles_cpu = torch.full(
                                (rows, shape["padded_width"]), -float("inf")
                            )
                            tiles_cpu[:, :width] = expected
                            tiles_cpu = tiles_cpu.view(rows, shape["steps"], 512)
                            if label != "score_only":
                                tiles_cpu = tiles_cpu.sort(dim=2, descending=True).values
                            weighted = (
                                torch.where(torch.isfinite(tiles_cpu), tiles_cpu, 0)
                                * torch.arange(1, 513, dtype=torch.float32)
                            ).sum(2)
                            torch.testing.assert_close(
                                arm["audit"].cpu(), weighted, rtol=1e-3, atol=1e-3
                            )
                            if label != "full":
                                assert bool((arm["indices"].cpu() == -3).all())
                                assert bool(torch.isnan(arm["values"].cpu()).all())
                        if args.topk_prefix and label in ("sort", "merge"):
                            padded = torch.full(
                                (rows, shape["padded_width"]), -float("inf")
                            )
                            padded[:, :width] = expected
                            tiles_cpu = padded.view(rows, shape["steps"], 512)
                            if label == "merge":
                                history = seed_values.view(1, 1, 512).expand(
                                    rows, shape["steps"], 512
                                )
                                tiles_cpu = torch.cat((history, tiles_cpu), dim=2)
                            selected_cpu = tiles_cpu.sort(
                                dim=2, descending=True
                            ).values[:, :, :512]
                            weighted = (
                                torch.where(
                                    torch.isfinite(selected_cpu), selected_cpu, 0
                                )
                                * torch.arange(1, 513, dtype=torch.float32)
                            ).sum(2)
                            torch.testing.assert_close(
                                arm["audit"].cpu(), weighted, rtol=1e-3, atol=1e-3
                            )
                            assert bool((arm["indices"].cpu() == -3).all())
                            assert bool(torch.isnan(arm["values"].cpu()).all())
                        elif args.critical_path and label == "light":
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
                        elif not args.paired_vector_profile or label == "full":
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
                                    .view(torch.int32)[::TICK_STRIDE_WORDS][:groups]
                                )
                                observed = dict(
                                    phase=phase,
                                    traced=traced,
                                    groups=groups,
                                    stride_bytes=TICK_STRIDE_WORDS * 4,
                                    expected=shape["steps"] - 1,
                                    observed=ticks.tolist(),
                                )
                                entry.setdefault("replay_progress", []).append(observed)
                                print(
                                    "REPLAY_PROGRESS", json.dumps(observed), flush=True
                                )
                                save()
                                assert bool(
                                    (ticks == shape["steps"] - 1).all()
                                ), "Replay AIC progress missing: " + repr(observed)
                    checks[label] = dict(
                        passed=True,
                        rows=rows,
                        steps=shape["steps"],
                        output_guards_unchanged=True,
                        trace_scores_exact=True,
                        selected_scores_exact=not (
                            (args.paired_vector_profile and label != "full")
                            or (args.critical_path and label == "light")
                            or args.topk_prefix
                            and label in ("sort", "merge")
                        ),
                        checksum_checked=bool(
                            args.paired_vector_profile
                            or (args.critical_path and label == "light")
                            or args.topk_prefix
                            and label in ("sort", "merge")
                        ),
                        replay_progress_checked=bool(
                            args.critical_path and label == "replay"
                        ),
                    )
                    if final_mrgsort_probe:
                        checks[label]["pair_words_match_control"] = bits.numel()
                    print(
                        "RAW_MIXED_CHECK",
                        width,
                        phase,
                        label,
                        checks[label],
                        flush=True,
                    )
                return checks

            entry["precheck"] = validate(
                "poison-check" if args.paired_final_mrgsort_poison_check else "before"
            )
            if args.paired_final_mrgsort_poison_check:
                entry["status"] = "passed"
                save()
                print(
                    "FINAL_MRGSORT_POISON_CHECK",
                    case_name,
                    json.dumps(entry["precheck"]),
                    flush=True,
                )
                continue
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
            timed_arms = {name: arm for name, arm in arms.items() if name != "poison"}
            timed_hosts = {name: hosts[name] for name in timed_arms}
            samples, _ = measure_prefix_plan(
                options,
                timed_arms,
                lambda name, _: arms[name]["launch"](),
                dict.fromkeys(timed_arms),
                torch,
                entry,
                expected_hosts=timed_hosts,
            )
            med = {
                name: sum(s["kernel_us"] for s in arr) / len(arr)
                for name, arr in samples.items()
            }
            numerator, denominator = (
                ("snapshot", "prefix")
                if args.history_boundary
                else (
                    ("snapshot", "complete")
                    if args.history_full_merge
                    else (
                        ("generic", "fixed")
                        if args.tle_sort512
                        else (
                            ("zero", "uninit")
                            if args.sort_buffers
                            else (
                                ("full", "sort")
                                if args.topk_prefix
                                else (
                                    ("full", "light")
                                    if args.critical_path
                                    else (
                                        ("baseline64", "block512")
                                        if args.cube_block512
                                        else ("slots1", "slots2")
                                    )
                                )
                            )
                        )
                    )
                )
            )
            if args.query_batch:
                numerator, denominator = "prefix", "batch" + str(args.query_batch)
            if args.history_direct:
                numerator, denominator = "prefix", "direct"
            if args.single_block:
                numerator, denominator = "snapshot", "single"
            if args.first_block_seed:
                numerator, denominator = "snapshot", "firstseed"
            if args.paired_vector_profile:
                numerator, denominator = "full", "sort_only"
                entry["vector_deltas_us"] = dict(
                    history_and_export=med["full"] - med["sort_only"],
                    sort_and_audit_difference=med["sort_only"] - med["score_only"],
                )
            if sort_barrier_probe:
                numerator, denominator = "full", removed_barrier
                entry["saved_us"] = med[numerator] - med[denominator]
            if final_mrgsort_probe:
                numerator = "full"
                denominator = "reuse_local32" if args.paired_final_mrgsort_reuse else "uninit"
                entry["saved_us"] = med[numerator] - med[denominator]
            ratio = med[numerator] / med[denominator]
            entry.update(samples=samples, mean_us=med)
            entry[
                "call64_over_block512" if args.cube_block512 else "serial_over_buffered"
            ] = ratio
            if args.history_full_merge and not (
                args.query_batch
                or args.history_direct
                or args.single_block
                or args.first_block_seed
            ):
                entry["snapshot_over_prefix"] = med["snapshot"] / med["prefix"]
                entry["history_ratios"] = dict(
                    snapshot_over_prefix=entry["snapshot_over_prefix"]
                )
                if not args.history_boundary:
                    entry["snapshot_over_complete"] = ratio
                    entry["history_ratios"]["snapshot_over_complete"] = ratio
                entry.pop("call64_over_block512", None)
            if args.query_batch:
                entry["query_batch_ratios"] = {
                    denominator: med["prefix"] / med[denominator]
                }
                entry.pop("call64_over_block512", None)
            if args.history_direct:
                entry["prefix_over_direct"] = ratio
                entry.pop("call64_over_block512", None)
            if args.single_block:
                entry["snapshot_over_single"] = ratio
                entry.pop("call64_over_block512", None)
            if args.first_block_seed:
                entry["snapshot_over_first_seed"] = ratio
                entry.pop("call64_over_block512", None)
            if args.tle_sort512:
                entry["generic_over_fixed"] = ratio
                entry.pop("call64_over_block512", None)
            if args.sort_buffers:
                entry["zero_over_uninit"] = ratio
                entry.pop("call64_over_block512", None)
            if args.topk_prefix:
                entry["prefix_ratios"] = dict(
                    full_over_sort=med["full"] / med["sort"],
                    full_over_merge=med["full"] / med["merge"],
                    scope="Fixed-history diagnostic; not phase durations",
                )
                entry.pop("call64_over_block512", None)
            elif args.critical_path:
                entry["ablation_ratios"] = dict(
                    full_over_light=med["full"] / med["light"],
                    full_over_replay=med["full"] / med["replay"],
                    scope="Diagnostic work removal; not internal stage times",
                )
                entry.pop("call64_over_block512", None)
            if paired_probe:
                entry.pop("call64_over_block512", None)
            if (
                args.cube_block512
                and not paired_probe
                and not args.critical_path
                and not args.sort_buffers
                and not args.tle_sort512
                and not args.history_full_merge
            ):
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
                        prefix_ratios=entry.get("prefix_ratios"),
                        history_ratios=entry.get("history_ratios"),
                        vector_deltas_us=entry.get("vector_deltas_us"),
                        saved_us=entry.get("saved_us"),
                    )
                ),
                flush=True,
            )
        result.update(
            status="passed",
            phase="completed",
            outcome=(
                "final_mrgsort_poison_overwrite_check_passed"
                if args.paired_final_mrgsort_poison_check
                else "mixed_numerical_and_pair_timing_passed_overlap_unproven"
            ),
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
