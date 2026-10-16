# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Stock Triton native-vLLM vs production-score + top_k_per_row_prefill A/B.

Default: 22 scenarios, kernel timer, separate score/Top-K/sequential pipeline.
Native Top-K remains vLLM CUDA; only the candidate substitutes repo non-TLE Top-K.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Also works under python -I: never add an external compiler search directory.
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from tools import benchmark_qsa_prefill_stages as support  # noqa: E402
from tools import qsa_prefill_inference_workloads as workloads  # noqa: E402
from tools import qsa_prefill_score_benchmark as scoring  # noqa: E402
from tools import qsa_prefill_triton_ab as ab  # noqa: E402

SOURCES = sorted(
    set(support.SOURCES)
    | {
        "tools/benchmark_qsa_prefill_triton_stages.py",
        "tools/qsa_prefill_triton_ab.py",
        "tools/run_qsa_prefill_triton_ab_h800.sh",
        "src/flaggems_vllm/ops/top_k_per_row_prefill.py",
        "src/flaggems_vllm/utils/triton_version_utils.py",
    }
)


def configure_environment():
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "QSA_TRITON_RUNTIME_PATH",
        "QSA_NATIVE_REFERENCE_PYTHON",
        "QSA_NATIVE_REPORT_DIR",
        "QSA_NATIVE_RESULT_PATH",
        "FLAGGEMS_DB_URL",
    ):
        os.environ.pop(key, None)
    os.environ.update(
        DNN_VENDOR="nvidia",
        VLLM_PLUGINS="",
        FLAGGEMS_FORCE_TLE="0",
        QSA_REQUIRE_VLLM_REFERENCE="1",
    )
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") != "0":
        raise RuntimeError("Unset CUDA_LAUNCH_BLOCKING before benchmarking")
    if not sys.flags.isolated:
        raise RuntimeError("Run this script with the selected virtualenv Python and -I")


def runtime(reference_path):
    configure_environment()
    import torch
    import triton

    if torch.__version__ != importlib.metadata.version("torch"):
        raise RuntimeError("Torch module and distribution versions differ")
    stock = ab.stock_runtime(torch, triton, sys.prefix)
    if triton.__version__ != "3.6.0":
        raise RuntimeError("This experiment pins upstream Triton 3.6.0")
    import flaggems_vllm
    from tests import qsa_select_prefill_utils as helpers

    op = importlib.import_module("flaggems_vllm.ops.qsa_select_paged_prefill")
    old_topk = importlib.import_module("flaggems_vllm.ops.top_k_per_row_prefill")
    expected = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
    if (
        Path(op.__file__).resolve() != expected
        or Path(old_topk.__file__).resolve()
        != ROOT / "src/flaggems_vllm/ops/top_k_per_row_prefill.py"
        or old_topk.HAS_TLE
    ):
        raise RuntimeError("Expected local production source and non-TLE Top-K")
    if not importlib.metadata.version("vllm").startswith("0.31.0"):
        raise RuntimeError(
            "Select the existing vLLM 0.31.0 virtualenv with native CUDA extensions"
        )
    os.environ["QSA_VLLM_REFERENCE_FILE"] = str(reference_path)
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != ab.REFERENCE_SHA256:
        raise RuntimeError("Pinned original native vLLM source SHA256 mismatch")
    native = helpers.vllm_reference()
    utils = importlib.import_module("vllm.triton_utils")
    if any(
        module.triton is not triton for module in (native, utils, op, old_topk)
    ) or any(module.tl is not triton.language for module in (native, op, old_topk)):
        raise RuntimeError(
            "Native vLLM and production do not share the same Triton modules"
        )
    stock_after = ab.stock_runtime(torch, triton, sys.prefix)
    if stock != stock_after:
        raise RuntimeError("Compiler identity changed during imports")
    if flaggems_vllm.vendor_name != "nvidia" or torch.cuda.get_device_capability() != (
        9,
        0,
    ):
        raise RuntimeError("NVIDIA SM90 required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    info = support.runtime_info(torch, triton, flaggems_vllm.runtime)
    info["stock_wheel"] = stock
    info["native_topk_libraries"] = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("vllm.") and path and path.endswith(".so"):
            file = Path(path).resolve()
            if not file.is_relative_to(Path(sys.prefix).resolve()):
                raise RuntimeError(
                    f"Native vLLM extension outside selected virtualenv: {file}"
                )
            info["native_topk_libraries"][name] = dict(
                path=str(file), sha256=hashlib.sha256(file.read_bytes()).hexdigest()
            )
    if not info["native_topk_libraries"]:
        raise RuntimeError("Native vLLM CUDA libraries were not loaded")
    uuid = info["uuid"]
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    if uuid != os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Actual device differs from the selected GPU UUID")
    adapter = ab.PrefillTopK(old_topk.top_k_per_row_prefill, torch)
    candidate = ab.candidate_module(op, adapter)
    return (
        torch,
        native,
        candidate,
        adapter,
        helpers,
        flaggems_vllm.runtime.torch_device_fn,
        info,
    )


def native_output(output, torch):
    # Preserve the inference slice's output storage offset as well as shape
    # and strides. full_like() would rebase mixed-batch views to offset zero.
    offset = output.storage_offset()
    storage = torch.full(
        (offset + output.numel(),), -17, dtype=output.dtype, device=output.device
    )
    return storage[offset:].view(output.shape)


def preflight_smoke(torch, native, candidate, helpers):
    checks = []
    os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = "512"
    native.envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB = 512
    for rows in (8, 128):
        case = dict(workloads.inference_cases()[0], query_lengths=[rows])
        args = workloads.make_inputs(case, helpers, torch)
        native_args = (*args[:8], native_output(args[8], torch), args[9])
        functions = dict(
            baseline=lambda: native.qsa_select_paged_prefill(*native_args),
            candidate=lambda: candidate.qsa_select_paged_prefill(*args),
        )
        checked = validate_pipeline(functions, native_args, args, [], torch, helpers)
        torch.cuda.synchronize()
        checks.append(
            dict(
                rows=rows,
                status="passed",
                sampled_rows=checked,
                native_topk="cooperative_topk" if rows <= 64 else "persistent_topk",
            )
        )
    return checks


def verify_geometry(case, args, chunks):
    expected = workloads.geometry(case)
    offset = 0
    signature = []
    for rows in expected["expected_chunk_rows"]:
        signature.append((offset, rows, expected["logits_width"]))
        offset += rows
    if ab.signatures(chunks) != signature:
        raise RuntimeError("Public pipeline differs from canonical scoring chunks")
    for index, key in ((0, "q_shape"), (1, "cache_shape"), (2, "page_table_shape")):
        if list(args[index].shape) != expected[key]:
            raise RuntimeError(f"Unexpected input shape: {index}")
    if case["kind"] == "inference_perf":
        for index, layout in expected["input_layout"].items():
            value = args[int(index)]
            if (
                list(value.stride()) != layout["stride"]
                or value.storage_offset() != layout["storage_offset"]
            ):
                raise RuntimeError(f"Unexpected input view layout: {index}")
        if (
            args[3].cpu().tolist() != expected["query_start_loc"]
            or args[9] != expected["batch_max_seq_len"]
        ):
            raise RuntimeError("Unexpected mixed-batch metadata")
    return expected


def validate_topk(native, candidate, native_chunks, candidate_chunks, torch, helpers):
    checked = 0
    for a, b in zip(native_chunks, candidate_chunks):
        common = a["common_logits"]
        native._topk(common, *a["topk_args"][1:], **a["topk_kwargs"])
        candidate.persistent_topk(common, *b["topk_args"][1:], **b["topk_kwargs"])
        # Native args: logits, visible, token_topk, ratio, output, workspace.
        # Candidate args: logits, visible, output, workspace.
        visible, left, right = a["topk_args"][1], a["topk_args"][4], b["topk_args"][2]
        starts = (a["score_args"][3] - a["score_args"][3][0]).cpu().tolist()
        for row in scoring.request_sample_rows(
            a["offset"], a["rows"], starts, support.sample_rows
        ):
            logits = common[row : row + 1]
            lengths = visible[row : row + 1]
            actual, expected = right[row : row + 1], left[row : row + 1]
            helpers.assert_selection(logits, lengths, actual)
            helpers.assert_selection(logits, lengths, expected)
            helpers.assert_native_selection(
                logits, lengths, actual, expected, dtype=torch.bfloat16, head_dim=128
            )
            checked += 1
    return checked


def validate_pipeline(functions, native_args, args, chunks, torch, helpers):
    functions["baseline"]()
    functions["candidate"]()
    q, cache, table, starts, visible = args[:5]
    offsets = (starts - starts[0]).cpu().tolist()
    rows = set(scoring.request_sample_rows(0, q.shape[0], offsets, support.sample_rows))
    for chunk in chunks:
        rows.update((chunk["offset"], chunk["offset"] + chunk["rows"] - 1))
    values = visible.cpu().tolist()
    for row in sorted(rows):
        request = next(
            i for i, (a, b) in enumerate(zip(offsets, offsets[1:])) if a <= row < b
        )
        columns = torch.arange(values[row], device=q.device)
        pages = table[request, columns // cache.shape[1]].long()
        keys = cache[pages, columns % cache.shape[1], 0].float()
        reference = (q[row].float() @ keys.T).clamp_min(0).sum(0)[None, :]
        lengths = visible[row : row + 1]
        left, right = native_args[8][row : row + 1], args[8][row : row + 1]
        helpers.assert_selection(reference, lengths, left)
        helpers.assert_selection(reference, lengths, right)
        helpers.assert_native_selection(
            reference, lengths, right, left, dtype=torch.bfloat16, head_dim=128
        )
    return len(rows)


def common_fingerprints(chunks, torch):
    # Each tensor gets its own argument zero: fingerprint() otherwise treats
    # argument eight as an output and intentionally omits its content hash.
    return [support.fingerprint((c["common_logits"],), torch)[0] for c in chunks]


def worker(options, case, path):
    result = dict(case=case, status="running", stages={})
    support.write_json(path, result)
    torch, native, candidate, adapter, helpers, device, info = runtime(
        Path(options.reference_file)
    )
    result["runtime"] = info
    os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = str(case["budget_mb"])
    native.envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB = case["budget_mb"]
    args = workloads.make_inputs(case, helpers, torch)
    native_args = (*args[:8], native_output(args[8], torch), args[9])
    result["inputs"] = support.fingerprint(args, torch)
    result["native_output_layout"] = dict(
        shape=list(native_args[8].shape),
        stride=list(native_args[8].stride()),
        storage_offset=native_args[8].storage_offset(),
    )
    native_chunks = ab.observe(native, "_topk", native_args)
    candidate_chunks = ab.observe(candidate, "persistent_topk", args)
    result["workload"] = verify_geometry(case, args, candidate_chunks)
    verify_geometry(case, native_args, native_chunks)
    if ab.signatures(native_chunks) != ab.signatures(candidate_chunks):
        raise RuntimeError("Baseline/candidate scoring geometry differs")
    for c in candidate_chunks:
        c.pop("common_logits")  # Both Top-K stages use only native frozen scores.
    adapter.frozen = True
    result["chunks"] = [
        dict(
            offset=a["offset"],
            rows=a["rows"],
            width=a["width"],
            native_workspace_shape=list(a["score_args"][5].shape),
            candidate_workspace_shape=list(b["score_args"][5].shape),
        )
        for a, b in zip(native_chunks, candidate_chunks)
    ]
    result["common_logits_before"] = common_fingerprints(native_chunks, torch)
    functions = ab.bind_stages(native, candidate, native_chunks, candidate_chunks)
    for stage in functions.values():
        for fn in stage.values():
            fn()  # Compile/autotune and allocate operator-internal scratch untimed.
    device.synchronize()
    timer, timer_info = scoring.make_timer(
        options.kernel_warmup_ms, options.kernel_rep_ms
    )
    result["timer"] = {k: v for k, v in timer_info.items() if k != "includes_topk"}
    result["production_score_configs"] = support.score_configs(candidate)
    result["topk_backends"] = dict(
        baseline="vLLM native CUDA _C dispatcher",
        candidate="repo top_k_per_row_prefill, HAS_TLE=False",
    )
    guard = support.gpu_guard_module()

    def validate(name):
        if name == "score":
            return {
                side: scoring.validate_scores(
                    module._prefill_logits,
                    chunks,
                    a,
                    torch,
                    support.sample_rows,
                    device,
                )
                for side, module, chunks, a in (
                    ("baseline", native, native_chunks, native_args),
                    ("candidate", candidate, candidate_chunks, args),
                )
            }
        if name == "topk":
            return dict(
                sample_rows=validate_topk(
                    native, candidate, native_chunks, candidate_chunks, torch, helpers
                )
            )
        return dict(
            sample_rows=validate_pipeline(
                functions[name], native_args, args, native_chunks, torch, helpers
            )
        )

    for name, stage in functions.items():
        entry = result["stages"][name] = dict(
            status="validating", accepted=False, includes_topk=name != "score"
        )
        entry["validation_before"] = validate(name)
        support.write_json(path, result)
        measured, observations, order = scoring.measure_scores(
            timer,
            stage["baseline"],
            stage["candidate"],
            options.rounds,
            lambda label: support.guarded_telemetry(
                info["uuid"], name + ":" + label, result, path, guard
            ),
            support.stats,
        )
        entry.update(
            measured,
            status="pending_validation",
            accepted=False,
            measurement_order=order,
        )
        entry["validation_after"] = validate(name)
        device.synchronize()
        entry.update(status="passed", accepted=True)
        support.write_json(path, result)
    result["common_logits_after"] = common_fingerprints(native_chunks, torch)
    if result["common_logits_before"] != result["common_logits_after"]:
        raise RuntimeError("Common Top-K logits were modified during this run")
    result.update(status="passed", peak_allocated_bytes=device.max_memory_allocated())
    support.write_json(path, result)


def write_table(summary, directory):
    lines = [
        "case\tstage\tstatus\tvllm_us\tgems_prefill_topk_us\tspeedup\tpaired_ratio_median"
    ]
    for case in summary["cases"]:
        for name in ("score", "topk", "pipeline"):
            entry = case.get("stages", {}).get(name, {})
            accepted = case["status"] == "passed" and entry.get("accepted")
            values = [
                case["case"]["name"],
                name,
                "passed" if accepted else case["status"],
            ]
            if accepted:
                values += [
                    entry["baseline_us"]["median"],
                    entry["candidate_us"]["median"],
                    entry["speedup"],
                    entry["paired_ratios"]["median"],
                ]
            else:
                values += [""] * 4
            lines.append("\t".join(map(str, values)))
    (directory / "triton-stage-ab.tsv").write_text("\n".join(lines) + "\n")


def child_command(options, directory, flag):
    return [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--suite",
        options.suite,
        "--report-dir",
        str(directory),
        "--reference-file",
        options.reference_file,
        "--rounds",
        str(options.rounds),
        "--kernel-warmup-ms",
        str(options.kernel_warmup_ms),
        "--kernel-rep-ms",
        str(options.kernel_rep_ms),
        flag,
    ]


def runtime_identity(info):
    from tools.qsa_prefill_score_resume import fingerprint

    return dict(
        fingerprint(info),
        stock_wheel=info.get("stock_wheel"),
        native_topk_libraries=info.get("native_topk_libraries"),
        vllm_version=info.get("versions", {}).get("vllm"),
    )


def child_environment(directory, label):
    env = os.environ.copy()
    env.pop("FLAGGEMS_DB_URL", None)
    env.update(
        TRITON_CACHE_DIR=str(directory / "cache" / label / "triton"),
        FLAGGEMS_CACHE_DIR=str(directory / "cache" / label / "gems"),
    )
    return env


def parent(options, selected, directory):
    summary = dict(
        schema=1,
        status="running",
        cases=[],
        manifest=dict(
            options=vars(options),
            cases=selected,
            git_head=support.command(["git", "rev-parse", "HEAD"]),
            git_status=support.command(["git", "status", "--porcelain"]),
            policy=(
                "Stock Triton 3.6.0; native CUDA vLLM Top-K vs repo non-TLE Top-K; "
                "preallocated outer workspaces; actual sequential pipeline; no production source edits"
            ),
        ),
    )
    path = directory / "stage-summary.json"
    support.write_json(path, summary)
    try:
        configure_environment()
        uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        guard = support.gpu_guard_module()
        if not guard.UUID_PATTERN.fullmatch(uuid):
            raise ValueError("CUDA_VISIBLE_DEVICES must select one full GPU UUID")
        reference = directory / "qsa_indexer_vllm.py"
        if options.reference_file:
            content = Path(options.reference_file).read_bytes()
        else:
            with urllib.request.urlopen(ab.REFERENCE_URL, timeout=60) as response:
                content = response.read()
        if hashlib.sha256(content).hexdigest() != ab.REFERENCE_SHA256:
            raise RuntimeError("Original native vLLM download/source SHA256 mismatch")
        reference.write_bytes(content)
        options.reference_file = str(reference)
        summary["manifest"]["reference"] = dict(
            commit=ab.REFERENCE_COMMIT, sha256=ab.REFERENCE_SHA256, url=ab.REFERENCE_URL
        )
        sources = {
            name: dict(
                sha256=hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                text=(ROOT / name).read_text(),
            )
            for name in SOURCES
        }
        sources["pinned_vllm/qsa_indexer.py"] = dict(
            sha256=ab.REFERENCE_SHA256, text=content.decode()
        )
        support.write_json(directory / "sources.json", sources)
        summary["manifest"]["source_hashes"] = {
            k: v["sha256"] for k, v in sources.items()
        }
        support.write_json(directory / "manifest.json", summary["manifest"])
        if not support.idle_gpu_gate(
            summary, directory, os.environ, "before_preflight", guard
        ):
            summary["status"] = "environment_rejected"
            return 2
        with (directory / "preflight.txt").open("w") as log:
            proc = subprocess.run(
                child_command(options, directory, "--preflight"),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=options.timeout,
                env=child_environment(directory, "preflight"),
            )
        preflight = (
            json.loads((directory / "preflight.json").read_text())
            if (directory / "preflight.json").is_file()
            else {}
        )
        summary["preflight"] = preflight
        if proc.returncode or preflight.get("status") != "passed":
            raise RuntimeError(
                "Stock-Triton/native-extension preflight failed; see preflight.txt/json"
            )
        for case in selected:
            if not support.idle_gpu_gate(
                summary, directory, os.environ, "before_" + case["name"], guard
            ):
                summary["status"] = "environment_rejected"
                return 2
            case_path = directory / (case["name"] + ".json")
            command = child_command(options, directory, "--worker") + [case["name"]]
            error = None
            with (directory / (case["name"] + ".txt")).open("w") as log:
                try:
                    proc = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=options.timeout,
                        env=child_environment(directory, case["name"]),
                    )
                    if proc.returncode:
                        error = f"worker exit={proc.returncode}"
                except subprocess.TimeoutExpired:
                    error = "worker timeout"
            record = (
                json.loads(case_path.read_text())
                if case_path.is_file()
                else dict(case=case, stages={})
            )
            if error or record.get("status") != "passed":
                record.update(
                    status="failed", worker_error=error or "incomplete worker"
                )
                support.write_json(case_path, record)
            elif runtime_identity(record["runtime"]) != runtime_identity(
                preflight["runtime"]
            ):
                record.update(
                    status="failed",
                    worker_error="compiler/GPU/native-extension identity changed",
                )
                support.write_json(case_path, record)
            summary["cases"].append(record)
            summary["progress"] = dict(
                completed=len(summary["cases"]), active=len(selected)
            )
            support.write_json(path, summary)
            write_table(summary, directory)
            print(case["name"], record["status"], flush=True)
        summary["status"] = (
            "passed"
            if all(c["status"] == "passed" for c in summary["cases"])
            else "failed"
        )
        return 0 if summary["status"] == "passed" else 1
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        print(summary["error"], file=sys.stderr)
        return 1
    finally:
        support.write_json(path, summary)
        write_table(summary, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("kernel",), default="kernel")
    parser.add_argument(
        "--suite",
        choices=("inference-prefill", "upstream-prefill"),
        default="inference-prefill",
    )
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--report-dir")
    parser.add_argument("--reference-file")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--kernel-warmup-ms", type=float, default=1000)
    parser.add_argument("--kernel-rep-ms", type=float, default=100)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--preflight", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    options = parser.parse_args()
    all_cases = (
        workloads.cases()
        if options.suite == "inference-prefill"
        else workloads.upstream.cases()
    )
    if options.list_cases:
        print(json.dumps(all_cases, indent=2))
        return 0
    if (
        min(
            options.rounds,
            options.kernel_warmup_ms,
            options.kernel_rep_ms,
            options.timeout,
        )
        <= 0
    ):
        parser.error("Round count and durations must be positive")
    definitions = {c["name"]: c for c in all_cases}
    if options.cases and (
        len(set(options.cases)) != len(options.cases)
        or set(options.cases) - definitions.keys()
    ):
        parser.error("Unknown or duplicate case names")
    directory = (
        Path(options.report_dir).resolve()
        if options.report_dir
        else Path(tempfile.mkdtemp(prefix="qsa-triton-ab-h800."))
    )
    if options.worker or options.preflight:
        if not options.report_dir or not options.reference_file:
            parser.error("Child requires report directory and pinned reference file")
        if options.worker and options.worker not in definitions:
            parser.error("Unknown worker case")
        path = directory / (
            options.worker + ".json" if options.worker else "preflight.json"
        )
        try:
            if options.preflight:
                torch, native, candidate, _, helpers, _, info = runtime(
                    Path(options.reference_file)
                )
                smoke = preflight_smoke(torch, native, candidate, helpers)
                support.write_json(
                    path, dict(status="passed", runtime=info, smoke_checks=smoke)
                )
            else:
                worker(options, definitions[options.worker], path)
            return 0
        except BaseException:
            result = json.loads(path.read_text()) if path.is_file() else {}
            result.update(status="failed", error=traceback.format_exc())
            support.write_json(path, result)
            print(result["error"], file=sys.stderr)
            return 1
    directory.mkdir(parents=True, exist_ok=True)
    if any(p.name != "runner.txt" for p in directory.iterdir()):
        parser.error(
            "Report directory must be fresh; existing reports are not overwritten"
        )
    print("Reports:", directory, flush=True)
    selected = [c for c in all_cases if not options.cases or c["name"] in options.cases]
    return parent(options, selected, directory)


if __name__ == "__main__":
    sys.exit(main())
