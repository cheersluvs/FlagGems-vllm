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

"""Compare independent INT8 score grid and tile probes on Ascend 910B."""

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
from profile_qsa_int8_score_910b import (  # noqa: E402
    CASES,
    FROZEN_SCORE_SHA256,
    check_chunk,
)
from qsa_kernel_benchmark_910b import aggregate, setup  # noqa: E402

from tools.qsa_int8_ten_workloads_910b import make_inputs, sample_rows  # noqa: E402

VARIANTS = {"g8_n32": (8, 32), "g20_n32": (20, 32), "g8_n128": (8, 128)}


def run_case(args):
    report = Path(args.report)
    result, code = dict(case=args.case, status="failed"), 1
    try:
        import torch
        import torch_npu
        import triton

        root = os.getenv("QSA_ENV_ROOT")
        if not root or not torch.npu.is_available():
            raise ModuleNotFoundError("Pinned 910B environment required")
        if (
            not Path(triton.__file__)
            .resolve()
            .is_relative_to(Path(root).resolve() / "flagtree")
        ):
            raise RuntimeError(f"Unexpected compiler: {triton.__file__}")
        torch.npu.set_device(args.device)
        if "910B" not in torch.npu.get_device_name(args.device):
            raise RuntimeError("910B required")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        source = Path(__file__).with_name("qsa_int8_score_910b_trial.py")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != FROZEN_SCORE_SHA256:
            raise RuntimeError("Frozen scorer source changed")
        from qsa_int8_score_910b_trial import _quant_score_trial, score

        result["environment"] = dict(
            python=sys.executable,
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            triton=triton.__version__,
            triton_source=triton.__file__,
            device_index=args.device,
            device=torch.npu.get_device_name(args.device),
            opp=os.getenv("ASCEND_OPP_PATH"),
            device_isolation="not_verified",
            scorer_sha256=digest,
        )
        result["stage"] = "inputs"
        plan = PLANS[args.case]
        result["geometry"] = plan
        inputs = make_inputs(plan)
        values = {key: value.to(f"npu:{args.device}") for key, value in inputs.items()}
        rows, width, chunk = (
            plan["q_shape"][0],
            plan["logits_width"],
            plan["chunk_rows"],
        )
        chunks = [(first, min(chunk, rows - first)) for first in range(0, rows, chunk)]
        scratch = torch.empty(
            (chunk, width), device=values["q"].device, dtype=torch.float32
        )
        control = torch.empty_like(scratch)
        columns = torch.arange(width, device=scratch.device)[None, :]
        samples = sample_rows(plan)
        refs = {row: reference(inputs, row)[1] for row in samples}
        result["memory"] = dict(
            score_scratch_bytes=2 * chunk * width * 4, raw_dot_bytes=0
        )
        result["variants"] = {
            name: dict(grid=grid, block_n=n) for name, (grid, n) in VARIANTS.items()
        }
        result["scope"] = (
            "Score-only prepared inputs; candidate speedup vs frozen INT8 score, not native fused or whole operator"
        )

        def launch(name, first, count, target):
            q = values["q"][first : first + count]
            qs = values["qs"][first : first + count]
            weights = values["weights"][first : first + count]
            table = values["row_table"][first : first + count]
            visible = values["visible"][first : first + count]
            if name == "g8_n32":
                return score(
                    q,
                    values["cache"],
                    qs,
                    values["ks"],
                    weights,
                    table,
                    visible,
                    target,
                    raw_dot=None,
                )
            grid, n = VARIANTS[name]
            return _quant_score_trial[(min(grid, count * triton.cdiv(width, n)),)](
                q,
                values["cache"],
                qs,
                values["ks"],
                weights,
                table,
                visible,
                target,
                target,
                HEADS=q.shape[1],
                PAGE=plan["page_size"],
                COLUMNS=table.shape[1],
                ROWS=count,
                WIDTH=width,
                BLOCK_N=n,
                MASK_RAW_DOT=True,
                WRITE_RAW_DOT=False,
                num_warps=4,
            )

        def score_into(first, count, target):
            return launch("g8_n32", first, count, target)

        def job(name):
            for first, count in chunks:
                launch(name, first, count, scratch[:count])

        result["stage"] = "compile_and_ir"
        result["compiler"] = {}
        for name in VARIANTS:
            kernel = launch(name, *chunks[0], scratch[: chunks[0][1]])
            ir = kernel.asm["ttir"]
            stores = len(re.findall(r"\btt\.store\b", ir))
            integer_dot = any(
                "tt.dot" in line and "xi8>" in line and "xi32>" in line
                for line in ir.splitlines()
            )
            if stores != 1 or not integer_dot:
                raise AssertionError(f"Unexpected score IR: {name}")
            (report / f"{args.case}-{name}-ttir.txt").write_text(ir)
            result["compiler"][name] = dict(
                store_count=stores, int8_dot=integer_dot, metadata=repr(kernel.metadata)
            )
        torch.npu.synchronize()
        result["stage"] = "full_score_preflight"
        result["correctness"] = {
            name: dict(rows_checked=0, cpu_rows=[], cpu_max_abs_error=0.0)
            for name in VARIANTS
        }
        for first, count in chunks:
            launch("g8_n32", first, count, control[:count])
            live = columns < values["visible"][first : first + count, None]
            selected = [row for row in samples if first <= row < first + count]
            for name in VARIANTS:
                if name == "g8_n32":
                    current = control[:count]
                else:
                    launch(name, first, count, scratch[:count])
                    current = scratch[:count]
                if not bool((torch.isfinite(current) | ~live).all().item()):
                    raise AssertionError(f"Nonfinite visible score: {name}, {first}")
                if not bool((torch.isneginf(current) | live).all().item()):
                    raise AssertionError(f"Invalid score mask: {name}, {first}")
                if not torch.allclose(current, control[:count], rtol=1e-3, atol=1e-3):
                    raise AssertionError(
                        f"Candidate differs from frozen score: {name}, {first}"
                    )
                check = result["correctness"][name]
                check["rows_checked"] += count
                for row in selected:
                    actual = current[row - first, : refs[row].numel()].cpu()
                    if not torch.allclose(actual, refs[row], rtol=1e-3, atol=1e-3):
                        raise AssertionError(
                            f"Independent CPU reference: {name}, {row}"
                        )
                    check["cpu_rows"].append(row)
                    if actual.numel():
                        check["cpu_max_abs_error"] = max(
                            check["cpu_max_abs_error"],
                            (actual - refs[row]).abs().max().item(),
                        )
            if first == 0 or first + count == rows or (first // chunk + 1) % 32 == 0:
                print("SCORE_CHECK", args.case, first + count, "/", rows, flush=True)
        result["stage"] = "kernel_timer"
        measure = setup(
            args,
            dict(chunks=chunks, score_into=score_into, logits=scratch),
            torch,
            result,
        )
        result["samples"] = {name: [] for name in VARIANTS}
        result["measurement_order"] = []
        for round_index in range(args.rounds):
            order = list(VARIANTS)
            if round_index % 2:
                order.reverse()
            result["measurement_order"].append(order)
            for name in order:
                sample = measure(lambda: job(name))
                result["samples"][name].append(sample)
                print(
                    "SCORE_TIMING",
                    json.dumps(
                        dict(
                            case=args.case,
                            round=round_index + 1,
                            variant=name,
                            score_ms=sample["kernel_us"] / 1000,
                        )
                    ),
                    flush=True,
                )
        result["timings"] = {
            name: aggregate(data) for name, data in result["samples"].items()
        }
        base = result["timings"]["g8_n32"]["kernel_us"]["median"]
        result["speedup_vs_frozen_score"] = {
            name: base / data["kernel_us"]["median"]
            for name, data in result["timings"].items()
        }
        result["paired_round_speedups"] = {
            name: [
                a["kernel_us"] / b["kernel_us"]
                for a, b in zip(result["samples"]["g8_n32"], data)
            ]
            for name, data in result["samples"].items()
        }
        result["stage"] = "postcheck"
        first, count = chunks[len(chunks) // 2]
        result["postchecks"] = {}
        for name in VARIANTS:
            launch(name, first, count, scratch[:count])
            result["postchecks"][name] = check_chunk(
                inputs, scratch[:count].cpu(), first, count
            )
        result["status"], result["stage"], code = "passed", "completed", 0
        print("variant         score_ms   speedup_vs_frozen_score", flush=True)
        for name in VARIANTS:
            print(
                f"{name:16} "
                f"{result['timings'][name]['kernel_us']['median']/1000:10.3f} "
                f"{result['speedup_vs_frozen_score'][name]:10.4f}",
                flush=True,
            )
    except ImportError:
        result.update(status="prerequisites_missing", traceback=traceback.format_exc())
        code = 2
    except Exception:
        result["traceback"] = traceback.format_exc()
    result["probe_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
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
    parser.add_argument("--kernel-active", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cases", nargs="+", choices=CASES)
    args = parser.parse_args()
    if min(args.rounds, args.kernel_warmup, args.kernel_active, args.timeout) < 1:
        parser.error("Positive iterations and timeout required")
    Path(args.report).mkdir(parents=True, exist_ok=True)
    if args.case:
        return run_case(args)
    report = Path(args.report)
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
        if timeout:
            entry["status"] = "timeout"
        elif code and entry["status"] == "passed":
            entry["status"] = "crashed_after_completion"
        entries.append(entry)
    snapshot(report, "after")
    code = (
        0
        if all(e["status"] == "passed" for e in entries)
        else (2 if all(e["status"] == "prerequisites_missing" for e in entries) else 1)
    )
    (report / "summary.json").write_text(
        json.dumps(dict(exit_code=code, cases=entries), indent=2)
    )
    print(f"INT8 grid/tile probe: exit={code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
