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

"""Compile and time full key-scale layouts including device preparation."""

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
from probe_qsa_int8_ten_shapes_910b import PLANS, run_child  # noqa: E402
from profile_qsa_int8_score_910b import CASES, check_chunk  # noqa: E402
from qsa_int8_full_scale_audit_910b import (  # noqa: E402
    audit_full_scales,
    collect_profiler_logs,
)
from qsa_int8_prefix_builder_910b import load_prefix  # noqa: E402
from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan  # noqa: E402
from qsa_int8_trace_timer_910b import setup_trace  # noqa: E402
from qsa_kernel_benchmark_910b import aggregate  # noqa: E402

from tools.qsa_int8_ten_workloads_910b import make_inputs  # noqa: E402

SCORES = ("integrated_full", "scale_product", "expanded_key", "expanded_product")
JOBS = (*SCORES, "pack_only", "pack_plus_key", "pack_plus_product")


def run_case(args):
    report = Path(args.report)
    result, code = dict(case=args.case, status="failed"), 1
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
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        from qsa_int8_pack_key_scale_910b import pack_key_scale
        from qsa_int8_selection_910b_trial import PreparedInt8Selection

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
        values = {
            name: value.to(f"npu:{args.device}") for name, value in inputs.items()
        }
        rows, width, chunk = (
            plan["q_shape"][0],
            plan["logits_width"],
            plan["chunk_rows"],
        )
        first = rows // 2 // chunk * chunk
        count = min(chunk, rows - first)
        scratch = torch.empty(
            (count, width), device=values["q"].device, dtype=torch.float32
        )
        output = torch.empty((rows, 512), device=scratch.device, dtype=torch.int32)
        workspace = torch.empty(
            (1024 * 1024,), device=scratch.device, dtype=torch.uint8
        )
        adapter = PreparedInt8Selection(values, output, scratch, workspace, None)
        expanded = torch.empty(
            (*values["ks"].shape[:2], 16), device=scratch.device, dtype=torch.float16
        )
        result["geometry"] = dict(
            first=first, count=count, width=width, grid=20, block_n=128, plan=plan
        )
        result["memory"] = dict(
            original_scale_bytes=values["ks"].numel() * 2,
            expanded_scale_bytes=expanded.numel() * 2,
            expansion_factor=16,
            fp16_bytes_per_token_row=32,
        )
        result["scope"] = (
            "Representative complete score kernels; preparation covers all physical-cache scales "
            "each combined invocation; exploratory device sums excluding host gaps"
        )
        result["stage"] = "pack_compile_validate"
        packed_kernel = pack_key_scale(values["ks"], expanded)
        expected = inputs["ks"].reshape(-1, 1).expand(-1, 16)

        def check_pack():
            if not torch.equal(expanded.cpu().reshape(-1, 16), expected):
                raise AssertionError("Expanded FP16 scales differ")
            return dict(all_values_exact=True, rows=expected.shape[0], columns=16)

        result["pack_precheck"] = check_pack()
        for key in ("ttir", "ttadapter", "bcmlir"):
            if isinstance(packed_kernel.asm.get(key), str):
                (report / f"{args.case}-pack-{key}.txt").write_text(
                    packed_kernel.asm[key]
                )
        result["pack_compiler"] = dict(metadata=repr(packed_kernel.metadata))
        (report / f"{args.case}-pack-source.txt").write_text(
            Path(__file__).with_name("qsa_int8_pack_key_scale_910b.py").read_text()
        )
        kernels = {
            name: load_prefix(report, args.case, name)
            for name in SCORES
            if name != "integrated_full"
        }
        targets = {
            name: torch.empty_like(scratch) for name in JOBS if name != "pack_only"
        }
        result["memory"]["score_output_bytes"] = len(targets) * count * width * 4
        targets["pack_only"] = None

        def score(name, target):
            if name == "integrated_full":
                return adapter.score_into(first, count, target)
            scales = expanded if name.startswith("expanded") else values["ks"]
            return kernels[name][(min(20, count * triton.cdiv(width, 128)),)](
                values["q"][first : first + count],
                values["cache"],
                values["qs"][first : first + count],
                scales,
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

        def launch(name, target):
            if name == "pack_only":
                return pack_key_scale(values["ks"], expanded)
            if name.startswith("pack_plus_"):
                pack_key_scale(values["ks"], expanded)
                return score(
                    "expanded_key" if name == "pack_plus_key" else "expanded_product",
                    target,
                )
            return score(name, target)

        result["stage"] = "score_compile_validate"
        result["compiler"] = {}
        result["prechecks"] = {}
        for name in SCORES:
            kernel = score(name, targets[name])
            result["prechecks"][name] = check_chunk(
                inputs, targets[name].cpu(), first, count
            )
            ir = kernel.asm["ttir"]
            dots = len(re.findall(r"\btt\.dot\b", ir))
            stores = len(re.findall(r"\btt\.store\b", ir))
            loads = len(re.findall(r"\btt\.load\b", ir))
            if (dots, stores, loads) != (1, 1, 7):
                raise AssertionError(
                    f"Unexpected full-score IR: {name}, {dots,stores,loads}"
                )
            for key in ("ttir", "ttadapter", "bcmlir"):
                if isinstance(kernel.asm.get(key), str):
                    (report / f"{args.case}-{name}-{key}.txt").write_text(
                        kernel.asm[key]
                    )
            lowered = kernel.asm["bcmlir"]
            types = re.findall(
                r"linalg\.reduce ins\([^\n]*tensor<[^>]*x(f32|i32)>", lowered
            )
            if types != ["f32"] or "mix_mode='mix'" not in repr(kernel.metadata):
                raise AssertionError(f"Unexpected mode/reduction: {name}")
            binary = kernel.asm.get("npubin")
            result["compiler"][name] = dict(
                dot_count=dots,
                store_count=stores,
                load_count=loads,
                reduction_element_types=types,
                metadata=repr(kernel.metadata),
                npubin_sha256=(
                    hashlib.sha256(bytes(binary)).hexdigest()
                    if isinstance(binary, (bytes, bytearray, memoryview))
                    else None
                ),
            )
        baseline = targets["integrated_full"].cpu()

        def check_score(name):
            actual = targets[name].cpu()
            if not torch.allclose(actual, baseline, rtol=1e-3, atol=1e-3):
                raise AssertionError(f"Complete chunk mismatch: {name}")
            mask = torch.isfinite(baseline)
            return dict(
                all_scores_checked=True,
                max_abs_error=(actual[mask] - baseline[mask]).abs().max().item(),
                cpu=check_chunk(inputs, actual, first, count),
            )

        for name in JOBS:
            if name == "pack_only":
                continue
            if name.startswith("pack_plus_"):
                launch(name, targets[name])
            result["prechecks"][name] = check_score(name)
        result["artifact_comparison"] = audit_full_scales(
            report, args.case, SCORES, result["compiler"]
        )
        print("LAYOUT_MEMORY", json.dumps(result["memory"]), flush=True)
        print("LAYOUT_ARTIFACTS", json.dumps(result["artifact_comparison"]), flush=True)
        result["stage"] = "kernel_timer"

        def score_into(start, n, target):
            return adapter.score_into(start, n, target)

        setup_trace(
            args,
            dict(chunks=[(first, count)], score_into=score_into, logits=scratch),
            torch,
            result,
        )
        expected_hosts = {name: {"_quant_score_trial": 1} for name in SCORES}
        expected_hosts["pack_only"] = {"_qsa_pack_key_scale": 1}
        for name in ("pack_plus_key", "pack_plus_product"):
            expected_hosts[name] = {"_qsa_pack_key_scale": 1, "_quant_score_trial": 1}
        result["samples"], result["measurement_order"] = measure_prefix_plan(
            args, JOBS, launch, targets, torch, result, expected_hosts=expected_hosts
        )
        result["postchecks"] = {
            name: check_score(name) for name in JOBS if name != "pack_only"
        }
        result["pack_postcheck"] = check_pack()
        result["latency"] = {
            name: aggregate(samples) for name, samples in result["samples"].items()
        }
        medians = {
            name: data["kernel_us"]["median"]
            for name, data in result["latency"].items()
        }
        result["speedup_vs_integrated"] = {
            name: medians["integrated_full"] / value
            for name, value in medians.items()
            if name != "pack_only"
        }
        result["layout_only_speedup_vs_matched_formula"] = {
            "expanded_key": medians["integrated_full"] / medians["expanded_key"],
            "expanded_product": medians["scale_product"] / medians["expanded_product"],
        }
        print("job                      chunk_us   speedup_vs_integrated", flush=True)
        for name in JOBS:
            print(
                f"{name:25} {medians[name]:12.3f} {str(result['speedup_vs_integrated'].get(name,'n/a')):>12}",
                flush=True,
            )
        print(
            "LAYOUT_ONLY_SPEEDUP",
            json.dumps(result["layout_only_speedup_vs_matched_formula"]),
            flush=True,
        )
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
            "qsa_int8_pack_key_scale_910b.py",
            "qsa_int8_prefix_builder_910b.py",
            "qsa_int8_score_910b_trial.py",
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
    parser.add_argument("--kernel-warmup", type=int, default=2)
    parser.add_argument("--kernel-active", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cases", nargs="+", choices=CASES)
    args = parser.parse_args()
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
        code, timeout = run_child(command, report / f"{name}-console.txt", args.timeout)
        entry = (
            json.loads(path.read_text())
            if path.is_file()
            else dict(case=name, status="crashed")
        )
        entry["subprocess_exit_code"] = code
        entry["profiler_diagnostics"] = collect_profiler_logs(report, name)
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
                scope="Complete-score layout diagnostics, all-cache packing cost charged per combined "
                "invocation; not full-prefill acceptance",
            ),
            indent=2,
        )
    )
    print(f"INT8 scale-layout exit={code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
