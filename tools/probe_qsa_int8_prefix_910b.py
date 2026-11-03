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

"""Time validated score prefixes, not wrong-answer middle-stage deletions."""

import argparse
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_qsa_int8_kernel_910b import snapshot  # noqa: E402
from probe_qsa_int8_score_910b import reference  # noqa: E402
from probe_qsa_int8_ten_shapes_910b import PLANS, run_child  # noqa: E402
from profile_qsa_int8_score_910b import CASES  # noqa: E402
from qsa_int8_prefix_builder_910b import PREFIXES, load_prefix  # noqa: E402
from qsa_int8_trace_timer_910b import setup_trace  # noqa: E402
from qsa_kernel_benchmark_910b import aggregate  # noqa: E402

from tools.qsa_int8_ten_workloads_910b import make_inputs  # noqa: E402


def run_case(args):
    prefixes = (
        ("dot_fp32", "query_scaled", "dequant", "full")
        if args.scale_only
        else (("dot", "dot_fp32", "dequant", "full") if args.tail_only else PREFIXES)
    )
    result, code = dict(case=args.case, status="failed"), 1
    report = Path(args.report)
    try:
        import torch
        import torch_npu
        import triton

        root = os.getenv("QSA_ENV_ROOT")
        if not root or not torch.npu.is_available():
            raise ModuleNotFoundError("Pinned NPU environment required")
        if (
            not Path(triton.__file__)
            .resolve()
            .is_relative_to(Path(root).resolve() / "flagtree")
        ):
            raise RuntimeError(f"Wrong compiler: {triton.__file__}")
        torch.npu.set_device(args.device)
        from qsa_int8_selection_910b_trial import PreparedInt8Selection

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        result["environment"] = dict(
            python=sys.executable,
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            triton=triton.__version__,
            compiler=triton.__file__,
            device=torch.npu.get_device_name(args.device),
            device_index=args.device,
            opp=os.getenv("ASCEND_OPP_PATH"),
            device_isolation="not_verified",
        )
        plan = PLANS[args.case]
        inputs = make_inputs(plan)
        values = {k: v.to(f"npu:{args.device}") for k, v in inputs.items()}
        rows, width, chunk = (
            plan["q_shape"][0],
            plan["logits_width"],
            plan["chunk_rows"],
        )
        first = rows // 2 // chunk * chunk
        count = min(chunk, rows - first)
        scratch = torch.empty(
            (chunk, width), device=values["q"].device, dtype=torch.float32
        )
        output = torch.empty((rows, 512), device=scratch.device, dtype=torch.int32)
        workspace = torch.empty(
            (1024 * 1024,), device=scratch.device, dtype=torch.uint8
        )
        adapter = PreparedInt8Selection(values, output, scratch, workspace, None)
        result["geometry"] = dict(
            plan=plan,
            first=first,
            count=count,
            width=width,
            grid=20,
            block_n=128,
            heads=4,
            padded_heads=16,
            logical_output_slots=count * width,
        )
        result["scope"] = (
            "Representative score chunk, not complete workload; checksum sink overhead included; "
            "diagnostic prefix differences are not additive operation costs"
        )
        result["stage"] = "prefix_generation"
        kernels = {name: load_prefix(report, args.case, name) for name in prefixes}

        def launch(name, target=None):
            target = scratch[:count] if target is None else target
            if name == "integrated_full":
                return adapter.score_into(first, count, target)
            return kernels[name][(min(20, count * triton.cdiv(width, 128)),)](
                values["q"][first : first + count],
                values["cache"],
                values["qs"][first : first + count],
                values["ks"],
                values["weights"][first : first + count],
                values["row_table"][first : first + count],
                values["visible"][first : first + count],
                target,
                target,
                HEADS=4,
                PAGE=192,
                COLUMNS=values["row_table"].shape[1],
                ROWS=count,
                WIDTH=width,
                BLOCK_N=128,
                MASK_RAW_DOT=True,
                WRITE_RAW_DOT=False,
                num_warps=4,
            )

        selected = sorted({first, first + count // 2, first + count - 1})
        references = {}
        for row in selected:
            n = int(inputs["visible"][row])
            physical = inputs["row_table"][row].long()
            keys = inputs["cache"][physical, :, 0].reshape(-1, 128)[:n].int()
            scales = inputs["ks"][physical, :, 0].reshape(-1)[:n].float()
            dot, full = reference(inputs, row)
            token = physical[torch.arange(n) // 192] * 192 + torch.arange(n) % 192
            references[row] = dict(
                mask_store=torch.zeros(n),
                index=token.float(),
                k_load=keys.sum(1).float(),
                q_load=(keys.sum(1) + inputs["q"][row].int().sum()).float(),
                dot=dot.sum(1).float(),
                dot_fp32=dot.float().sum(1),
                query_scaled=(dot.float() * inputs["qs"][row].float()[None, :]).sum(1),
                dequant=(
                    dot.float() * inputs["qs"][row].float()[None, :] * scales[:, None]
                ).sum(1),
                full=full,
                integrated_full=full,
            )

        def check(name, target=None):
            actual = (scratch[:count] if target is None else target).cpu()
            live = (
                torch.arange(width)[None, :]
                < inputs["visible"][first : first + count, None]
            )
            if not bool((torch.isfinite(actual) | ~live).all()) or not bool(
                (torch.isneginf(actual) | live).all()
            ):
                raise AssertionError(f"Bad mask/finite check: {name}")
            maximum = 0.0
            for row in selected:
                expected = references[row][name]
                observed = actual[row - first, : expected.numel()]
                exact = name in (
                    "mask_store",
                    "index",
                    "k_load",
                    "q_load",
                    "dot",
                    "dot_fp32",
                )
                if not (
                    torch.equal(observed, expected)
                    if exact
                    else torch.allclose(observed, expected, rtol=1e-3, atol=1e-3)
                ):
                    raise AssertionError(f"Invalid prefix checksum: {name}, row={row}")
                if expected.numel():
                    maximum = max(maximum, (observed - expected).abs().max().item())
            return dict(all_chunk_masks=True, cpu_rows=selected, max_abs_error=maximum)

        arms = ["integrated_full", *prefixes]
        result["selected_prefixes"] = list(prefixes)
        result["stage"] = "compile_and_validate"
        result["prechecks"], result["compiler"] = {}, {}
        for name in arms:
            compiled = launch(name)
            result["prechecks"][name] = check(name)
            ir = compiled.asm["ttir"]
            dots = len(re.findall(r"\btt\.dot\b", ir))
            stores = len(re.findall(r"\btt\.store\b", ir))
            if stores != 1 or dots != (
                0 if name in ("mask_store", "index", "k_load", "q_load") else 1
            ):
                raise AssertionError(
                    f"Unexpected prefix IR: {name}, dots={dots}, stores={stores}"
                )
            for key in ("ttir", "ttadapter", "bcmlir"):
                if isinstance(compiled.asm.get(key), str):
                    (report / f"{args.case}-{name}-{key}.txt").write_text(
                        compiled.asm[key]
                    )
            result["compiler"][name] = dict(
                dot_count=dots,
                store_count=stores,
                load_count=len(re.findall(r"\btt\.load\b", ir)),
                metadata=repr(compiled.metadata),
            )
        if args.tail_only or args.scale_only:
            for name in arms:
                meta = result["compiler"][name]["metadata"]
                lowered = (report / f"{args.case}-{name}-bcmlir.txt").read_text()
                reductions = re.findall(
                    r"linalg\.reduce ins\([^\n]*tensor<[^>]*x(f32|i32)>", lowered
                )
                expected_type = "i32" if name == "dot" else "f32"
                result["compiler"][name]["reduction_element_types"] = reductions
                result["compiler"][name]["mixed_mode"] = "mix_mode='mix'" in meta
                if not result["compiler"][name]["mixed_mode"] or reductions != [
                    expected_type
                ]:
                    raise AssertionError(
                        f"Lowering control mismatch: {name}, reductions={reductions}"
                    )
            if args.scale_only:
                expected_loads = dict(
                    integrated_full=7, dot_fp32=4, query_scaled=5, dequant=6, full=7
                )
                for name in arms:
                    if result["compiler"][name]["load_count"] != expected_loads[name]:
                        raise AssertionError(
                            f"Unexpected scale load count: {name}, {result['compiler'][name]['load_count']}"
                        )
            print(
                "LOWERING_CONTROL",
                json.dumps(
                    {
                        name: dict(
                            mixed_mode=True,
                            loads=result["compiler"][name]["load_count"],
                            reductions=result["compiler"][name][
                                "reduction_element_types"
                            ],
                        )
                        for name in arms
                    }
                ),
                flush=True,
            )
        result["stage"] = "kernel_timer"

        def score_into(start, n, target):
            return adapter.score_into(start, n, target)

        measure = setup_trace(
            args,
            dict(chunks=[(first, count)], score_into=score_into, logits=scratch),
            torch,
            result,
        )
        result["samples"] = {name: [] for name in arms}
        result["postchecks"] = {}
        result["measurement_order"] = []
        if args.single_capture:
            from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan

            targets = {name: torch.empty_like(scratch[:count]) for name in arms}
            result["separate_prefix_output_bytes"] = len(arms) * count * width * 4
            result["samples"], result["measurement_order"] = measure_prefix_plan(
                args, arms, launch, targets, torch, result
            )
            for name in arms:
                # Each arm owns a separate output: these are the last timed
                # results, not a fresh post-timing recomputation.
                result["postchecks"][name] = check(name, targets[name])
        else:
            for round_index in range(args.rounds):
                order = arms if round_index % 2 == 0 else list(reversed(arms))
                result["measurement_order"].append(order)
                for name in order:
                    sample = measure(lambda: launch(name))
                    result["samples"][name].append(sample)
                    result["postchecks"][name] = check(name)
                    print(
                        "PREFIX_TIMING",
                        json.dumps(
                            dict(
                                case=args.case,
                                arm=name,
                                round=round_index + 1,
                                **sample,
                            )
                        ),
                        flush=True,
                    )
        result["timings"] = {
            name: aggregate(samples) for name, samples in result["samples"].items()
        }
        medians = {
            name: value["kernel_us"]["median"]
            for name, value in result["timings"].items()
        }
        ratio = medians["full"] / medians["integrated_full"]
        result["generated_full_over_integrated"] = ratio
        result["prefix_differences_us"] = {
            name: medians[name] - medians[prefixes[i - 1]]
            for i, name in enumerate(prefixes)
            if i
        }
        print(
            "arm             chunk_us   ns_per_output_slot   delta_from_prior_prefix_us",
            flush=True,
        )
        for name in arms:
            delta = result["prefix_differences_us"].get(name)
            print(
                f"{name:16} {medians[name]:10.3f} {medians[name]*1000/(count*width):12.4f} {str(delta):>15}",
                flush=True,
            )
        result["attribution_control_passed"] = 0.9 <= ratio <= 1.1
        if not result["attribution_control_passed"]:
            raise RuntimeError(f"Generated full control differs in timing: {ratio}")
        result["status"], result["stage"], code = "passed", "completed", 0
    except ImportError:
        result.update(status="prerequisites_missing", traceback=traceback.format_exc())
        code = 2
    except Exception:
        result["traceback"] = traceback.format_exc()
    result["sources"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            Path(__file__).name,
            "qsa_int8_prefix_builder_910b.py",
            "qsa_int8_score_910b_trial.py",
            "qsa_int8_selection_910b_trial.py",
            "qsa_int8_trace_timer_910b.py",
            "qsa_int8_prefix_plan_timer_910b.py",
            "qsa_int8_stage_trace_timer_910b.py",
        )
    }
    result["exit_code"] = code
    (report / f"{args.case}.json").write_text(json.dumps(result, indent=2))
    if code:
        print(json.dumps(result, indent=2), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--kernel-warmup", type=int, default=1)
    parser.add_argument("--kernel-active", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cases", nargs="+", choices=CASES)
    parser.add_argument(
        "--tail-only",
        action="store_true",
        help="Compare INT32/FP32 dot sinks, dequant and full, with lowering gates",
    )
    parser.add_argument(
        "--scale-only",
        action="store_true",
        help="Compare query-scale and key-scale prefixes with FP32 reduction",
    )
    parser.add_argument(
        "--single-capture",
        action="store_true",
        help="Interleave scale prefixes within one trace",
    )
    args = parser.parse_args()
    if args.single_capture and not args.scale_only:
        parser.error("single-capture requires scale-only")
    if args.tail_only and args.scale_only:
        parser.error("Choose only one prefix subset")
    if min(args.rounds, args.kernel_warmup, args.kernel_active, args.timeout) < 1:
        parser.error("Positive iterations and timeout required")
    report = Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    if args.case:
        return run_case(args)
    snapshot(report, "before")
    entries = []
    for name in dict.fromkeys(args.cases or CASES):
        path = report / f"{name}.json"
        path.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--case",
            name,
            "--report",
            str(report),
            "--device",
            str(args.device),
            "--rounds",
            str(args.rounds),
            "--kernel-warmup",
            str(args.kernel_warmup),
            "--kernel-active",
            str(args.kernel_active),
        ]
        if args.tail_only:
            command.append("--tail-only")
        if args.scale_only:
            command.append("--scale-only")
        if args.single_capture:
            command.append("--single-capture")
        code, timeout = run_child(command, report / f"{name}-console.txt", args.timeout)
        entry = (
            json.loads(path.read_text())
            if path.is_file()
            else dict(case=name, status="crashed")
        )
        entry["subprocess_exit_code"] = code
        if timeout:
            entry["status"] = "timeout"
        elif code and entry["status"] == "passed":
            entry["status"] = "crashed_after_checks"
        entries.append(entry)
    snapshot(report, "after")
    code = (
        0
        if all(e["status"] == "passed" for e in entries)
        else (2 if all(e["status"] == "prerequisites_missing" for e in entries) else 1)
    )
    (report / "summary.json").write_text(
        json.dumps(
            dict(
                exit_code=code,
                cases=entries,
                scope="Validated prefix diagnostics; no stage speedup or production acceptance claim",
            ),
            indent=2,
        )
    )
    print(f"INT8 prefix probe exit={code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
