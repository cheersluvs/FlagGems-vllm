# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Same-compiler native-vLLM vs production QSA prefill stage A/B.

Twenty-two BF16 cases plus ten explicit FP8 fixtures, with separate stage timers.
Stock Triton uses non-TLE Top-K; FlagTree also supports a private scheduling trial.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Both profiles run under -I; only FlagTree explicitly adds its selected directory.
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from tools import benchmark_qsa_prefill_stages as support  # noqa: E402
from tools import qsa_prefill_accuracy_profiles as accuracy_profiles  # noqa: E402
from tools import qsa_prefill_cache_granularity as cache_trial  # noqa: E402
from tools import qsa_prefill_loop_stage_trial as loop_stage  # noqa: E402
from tools import qsa_prefill_loop_unroll_trial as loop_unroll  # noqa: E402
from tools import qsa_prefill_observed_workloads as workloads  # noqa: E402
from tools import qsa_prefill_packed_grid as packed  # noqa: E402
from tools import qsa_prefill_page_table_trial as page_table  # noqa: E402
from tools import qsa_prefill_query_joint_trial as query_joint  # noqa: E402
from tools import qsa_prefill_query_page_trial as query_page  # noqa: E402
from tools import qsa_prefill_query_split_trial as query_split  # noqa: E402
from tools import qsa_prefill_score_benchmark as scoring  # noqa: E402
from tools import qsa_prefill_shared_gpu as shared  # noqa: E402
from tools import qsa_prefill_stage_joint_trial as stage_joint  # noqa: E402
from tools import qsa_prefill_triton_ab as ab  # noqa: E402
from tools import qsa_prefill_tuning_round1 as round1  # noqa: E402

SOURCES = sorted(
    set(support.SOURCES)
    | {
        "tools/benchmark_qsa_prefill_triton_stages.py",
        "tools/qsa_prefill_triton_ab.py",
        "tools/qsa_prefill_pre_store_production.py",
        "tools/qsa_prefill_pre_head_group_production.py",
        "tools/qsa_prefill_pre_paired_production.py",
        "tools/test_qsa_prefill_fp8_paired_production.py",
        "tools/run_qsa_prefill_fp8_paired_production_h800.sh",
        "docs/qsa_prefill_fp8_paired_production.md",
        "tools/test_qsa_prefill_head_group_production.py",
        "tools/run_qsa_prefill_head_group_production_h800.sh",
        "docs/qsa_prefill_head_group_production.md",
        "tools/test_qsa_prefill_fp8_store_production.py",
        "tools/run_qsa_prefill_fp8_store_production_h800.sh",
        "docs/qsa_prefill_fp8_store_production.md",
        "tools/qsa_prefill_tuning_round1.py",
        "tools/test_qsa_prefill_tuning_round1.py",
        "tools/qsa_prefill_cache_granularity.py",
        "tools/test_qsa_prefill_cache_granularity.py",
        "docs/qsa_prefill_cache_granularity.md",
        "tools/qsa_prefill_profiling.py",
        "tools/qsa_prefill_report.py",
        "tools/test_qsa_prefill_report.py",
        "tools/test_qsa_prefill_profiling.py",
        "tools/run_qsa_prefill_profiling_h800.sh",
        "docs/qsa_prefill_profiling_workflow.md",
        "tools/qsa_prefill_packed_grid.py",
        "tools/qsa_prefill_packed_grid_kernel.py",
        "tools/test_qsa_prefill_packed_grid.py",
        "tests/test_qsa_prefill_packed_grid.py",
        "docs/qsa_prefill_packed_grid.md",
        "tools/qsa_prefill_accuracy_profiles.py",
        "tools/test_qsa_prefill_accuracy_profiles.py",
        "tools/test_qsa_prefill_benchmark_only.py",
        "docs/qsa_prefill_benchmark_only.md",
        "docs/qsa_prefill_accuracy_profiles.md",
        "tools/qsa_prefill_combined_controls.py",
        "tools/test_qsa_prefill_combined_controls.py",
        "tools/run_qsa_prefill_combined_controls_h800.sh",
        "docs/qsa_prefill_combined_controls.md",
        "tools/qsa_prefill_loop_stage_trial.py",
        "tools/test_qsa_prefill_loop_stage.py",
        "tests/test_qsa_prefill_loop_stage.py",
        "tools/run_qsa_prefill_loop_stage_h800.sh",
        "docs/qsa_prefill_loop_stage_trial.md",
        "tools/qsa_prefill_loop_unroll_kernel.py",
        "tools/qsa_prefill_loop_unroll_trial.py",
        "tools/test_qsa_prefill_loop_unroll.py",
        "tests/test_qsa_prefill_loop_unroll.py",
        "tools/run_qsa_prefill_loop_unroll_h800.sh",
        "docs/qsa_prefill_loop_unroll_trial.md",
        "tools/qsa_prefill_stage_joint_trial.py",
        "tools/test_qsa_prefill_stage_joint.py",
        "tests/test_qsa_prefill_stage_joint.py",
        "tools/run_qsa_prefill_stage_joint_h800.sh",
        "docs/qsa_prefill_stage_joint_trial.md",
        "tools/qsa_prefill_query_controls.py",
        "tools/test_qsa_prefill_query_controls.py",
        "tools/run_qsa_prefill_query_controls_h800.sh",
        "docs/qsa_prefill_query_controls.md",
        "tools/qsa_prefill_query_page_kernel.py",
        "tools/qsa_prefill_query_page_trial.py",
        "tools/test_qsa_prefill_query_page.py",
        "tests/test_qsa_prefill_query_page.py",
        "tools/run_qsa_prefill_query_page_h800.sh",
        "docs/qsa_prefill_query_page_trial.md",
        "tools/qsa_prefill_query_joint_kernel.py",
        "tools/qsa_prefill_query_joint_trial.py",
        "tools/qsa_prefill_query_joint_policy.py",
        "tools/test_qsa_prefill_query_joint.py",
        "tests/test_qsa_prefill_query_joint.py",
        "docs/qsa_prefill_query_joint_trial.md",
        "tools/qsa_prefill_query_split_kernel.py",
        "tools/qsa_prefill_query_split_trial.py",
        "tools/qsa_prefill_query_split_matched.py",
        "tools/test_qsa_prefill_query_split.py",
        "tests/test_qsa_prefill_query_split.py",
        "docs/qsa_prefill_query_split_trial.md",
        "tools/qsa_prefill_page_table_kernel.py",
        "tools/qsa_prefill_page_table_bf16_kernel.py",
        "tools/test_qsa_prefill_page_table_bf16.py",
        "tests/test_qsa_prefill_page_table_bf16.py",
        "docs/qsa_prefill_page_table_bf16_trial.md",
        "tools/qsa_prefill_page_table_trial.py",
        "tools/test_qsa_prefill_page_table.py",
        "tests/test_qsa_prefill_page_table.py",
        "docs/qsa_prefill_page_table_trial.md",
        "tools/qsa_prefill_frozen_production.py",
        "tools/test_qsa_prefill_production_integration.py",
        "tools/run_qsa_prefill_triton_ab_h800.sh",
        "src/flaggems_vllm/ops/top_k_per_row_prefill.py",
        "src/flaggems_vllm/utils/triton_version_utils.py",
        "tools/qsa_prefill_flagtree_ab.py",
        "tools/test_qsa_prefill_flagtree_ab.py",
        "tools/qsa_prefill_shared_gpu.py",
        "tools/run_qsa_prefill_flagtree_ab_h800.sh",
        "tools/with_cuda_compat134.py",
        "tools/test_cuda_compat134.py",
        "tools/validate_qsa_ptxas_h800.py",
        "tools/diagnose_qsa_native_score_h800.py",
        "tools/diagnose_qsa_native_decode_h800.py",
        "tools/qsa_prefill_observed_workloads.py",
        "tools/qsa_prefill_fp8_workloads.py",
        "tools/test_qsa_prefill_fp8_workloads.py",
        "docs/qsa_prefill_fp8_shape_manifest.json",
        "docs/qsa_prefill_tuning_round1.md",
        "tools/qsa_prefill_compact_trial.py",
        "tools/qsa_prefill_key_major_trial.py",
        "tools/qsa_prefill_reuse_trial.py",
        "tools/qsa_prefill_eviction_trial.py",
        "tools/qsa_prefill_kernel_resources.py",
        "tests/test_qsa_prefill_compact_trial.py",
        "tests/test_qsa_prefill_key_major_trial.py",
        "tests/test_qsa_prefill_reuse_trial.py",
        "tests/test_qsa_prefill_eviction_trial.py",
        "benchmark/test_qsa_select_paged_prefill_perf.py",
    }
)

SCORE_STAGES = (
    "score",
    "score_vs_production",
    "score_vs_key_major",
    "score_vs_reuse",
    "score_vs_previous_tuning",
    "score_matched",
)
STAGE_ORDER = (
    "score",
    "topk",
    "pipeline",
    "score_vs_production",
    "score_vs_key_major",
    "score_vs_reuse",
    "score_vs_previous_tuning",
    "pipeline_vs_previous_tuning",
    "pipeline_vs_production",
    "score_matched",
    "pipeline_buffer_swap",
)
KEY_MAJOR_CANDIDATES = ("key-major", "key-major-reuse", "key-major-evict-first")


def is_flagtree(options):
    return getattr(options, "compiler", "stock") == "flagtree"


def configure_environment(options=None):
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
    if is_flagtree(options):
        from tools import qsa_prefill_flagtree_ab as profile

        profile.configure(options.flagtree)


def check_stock_environment(torch, triton, diagnostic_path=None, options=None):
    environment = dict(
        python=sys.executable,
        prefix=sys.prefix,
        isolated=bool(sys.flags.isolated),
        torch_version=torch.__version__,
        torch_source=torch.__file__,
        triton_version=triton.__version__,
        triton_source=triton.__file__,
    )
    print("Selected environment:", json.dumps(environment), flush=True)
    if diagnostic_path is not None:
        result = (
            json.loads(diagnostic_path.read_text())
            if diagnostic_path.is_file()
            else dict(status="checking_environment")
        )
        result["environment"] = environment
        support.write_json(diagnostic_path, result)
    if torch.__version__ != importlib.metadata.version("torch"):
        raise RuntimeError("Torch module and distribution versions differ")
    if is_flagtree(options):
        from tools import qsa_prefill_flagtree_ab as profile

        return profile.compiler_runtime(
            torch, triton, options.flagtree, options.report_dir
        )
    return ab.stock_runtime(torch, triton, sys.prefix)


def runtime(reference_path, diagnostic_path=None, options=None):
    configure_environment(options)
    import torch
    import triton

    stock = check_stock_environment(torch, triton, diagnostic_path, options)
    import flaggems_vllm
    from tests import qsa_select_prefill_utils as helpers

    op = importlib.import_module("flaggems_vllm.ops.qsa_select_paged_prefill")
    topk_name = "persistent_topk" if is_flagtree(options) else "top_k_per_row_prefill"
    old_topk = importlib.import_module("flaggems_vllm.ops." + topk_name)
    expected = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
    if (
        Path(op.__file__).resolve() != expected
        or Path(old_topk.__file__).resolve()
        != ROOT / ("src/flaggems_vllm/ops/" + topk_name + ".py")
        or old_topk.HAS_TLE != is_flagtree(options)
    ):
        raise RuntimeError(
            "Expected local production source and selected Top-K backend"
        )
    if (
        hashlib.sha256(expected.read_bytes()).hexdigest()
        != ab.INTEGRATED_PRODUCTION_SHA256
    ):
        raise RuntimeError("Production source changed; review and repin the experiment")
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
    stock_after = check_stock_environment(torch, triton, options=options)
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
    info["compiler_profile"] = "flagtree" if is_flagtree(options) else "stock"
    info["flagtree_runtime" if is_flagtree(options) else "stock_wheel"] = stock
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
    if is_flagtree(options):
        if op.persistent_topk is not old_topk.persistent_topk:
            raise RuntimeError("Production persistent_topk binding changed")
        adapter, candidate = None, op
    else:
        adapter = ab.PrefillTopK(old_topk.top_k_per_row_prefill, torch)
        candidate = ab.candidate_module(op, adapter)
    if getattr(options, "candidate", "production") == "compact-ragged":
        from tools import qsa_prefill_compact_trial as trial

        candidate = trial.candidate_module(ab.frozen_production())
    elif getattr(options, "candidate", "production") == "query-loop-unroll":
        candidate = loop_unroll.candidate_module(op)
    elif getattr(options, "candidate", "production") == "query-stage-joint":
        candidate = stage_joint.candidate_module(op)
    elif getattr(options, "candidate", "production") == "query-loop-stage":
        candidate = loop_stage.candidate_module(op)
    elif getattr(options, "candidate", "production") == "query-joint-page":
        candidate = query_page.candidate_module(op)
    elif getattr(options, "candidate", "production") == "query-joint":
        candidate = query_joint.candidate_module(op)
    elif getattr(options, "candidate", "production") == "query-split":
        candidate = query_split.candidate_module(op)
    elif getattr(options, "candidate", "production") == "packed-grid":
        candidate = packed.candidate_module(op)
    elif (
        getattr(options, "candidate", "production")
        in accuracy_profiles.PAGE_TABLE_CANDIDATES
    ):
        candidate = page_table.candidate_module(
            op, bf16_only=options.candidate == "page-table-bf16"
        )
    elif getattr(options, "candidate", "production") == "key-major":
        from tools import qsa_prefill_key_major_trial as trial

        candidate = trial.candidate_module(ab.frozen_production())
    elif getattr(options, "candidate", "production") == "key-major-reuse":
        from tools import qsa_prefill_reuse_trial as trial

        candidate = trial.candidate_module(ab.frozen_production())
    elif getattr(options, "candidate", "production") == "key-major-evict-first":
        from tools import qsa_prefill_eviction_trial as trial

        candidate = trial.candidate_module(ab.frozen_production())
    info["operator_binding"] = operator_binding(
        op,
        candidate,
        flaggems_vllm.qsa_select_paged_prefill,
        require_production=is_flagtree(options)
        and getattr(options, "candidate", "production") == "production",
    )
    print("Operator binding:", json.dumps(info["operator_binding"]), flush=True)
    return (
        torch,
        native,
        candidate,
        adapter,
        helpers,
        flaggems_vllm.runtime.torch_device_fn,
        info,
    )


def operator_binding(op, candidate, exported, *, require_production):
    """Identify the measured callable; reject private clones in production mode."""
    binding = dict(
        canonical_source=str(Path(op.__file__).resolve()),
        canonical_sha256=hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest(),
        candidate_module=candidate.__name__,
        candidate_source=str(Path(candidate.__file__).resolve()),
        candidate_sha256=hashlib.sha256(
            Path(candidate.__file__).read_bytes()
        ).hexdigest(),
        candidate_is_canonical_module=candidate is op,
        exported_entry_is_candidate=exported is candidate.qsa_select_paged_prefill,
        public_function_module=candidate.qsa_select_paged_prefill.__module__,
    )
    if require_production and not (
        binding["candidate_is_canonical_module"]
        and binding["exported_entry_is_candidate"]
        and binding["public_function_module"] == op.__name__
        and binding["canonical_sha256"] == ab.INTEGRATED_PRODUCTION_SHA256
    ):
        raise RuntimeError("Production benchmark must use the actual exported entry")
    return binding


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
    for dtype in ("bfloat16", "float8_e4m3fn"):
        for rows in (8, 128):
            case = dict(workloads.cases()[0], query_length=rows, dtype=dtype)
            args = workloads.make_inputs(case, helpers, torch)
            native_args = (*args[:8], native_output(args[8], torch), args[9])
            functions = dict(
                baseline=lambda: native.qsa_select_paged_prefill(*native_args),
                candidate=lambda: candidate.qsa_select_paged_prefill(*args),
            )
            checked = validate_pipeline(
                functions, native_args, args, [], torch, helpers
            )
            torch.cuda.synchronize()
            checks.append(
                dict(
                    rows=rows,
                    dtype=dtype,
                    status="passed",
                    sampled_rows=checked,
                    native_topk="cooperative_topk" if rows <= 64 else "persistent_topk",
                )
            )
    return checks


def verify_geometry(case, args, chunks, *, select_all=False):
    expected = workloads.geometry(case)
    offset = 0
    signature = []
    for rows in expected["expected_chunk_rows"]:
        signature.append((offset, rows, expected["logits_width"]))
        offset += rows
    if ab.signatures(chunks) != ([] if select_all else signature):
        raise RuntimeError("Public pipeline differs from canonical scoring chunks")
    for index, key in ((0, "q_shape"), (1, "cache_shape"), (2, "page_table_shape")):
        if list(args[index].shape) != expected[key]:
            raise RuntimeError(f"Unexpected input shape: {index}")
    if case["kind"] in ("inference_perf", "observed_perf", "fp8_perf"):
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
                logits,
                lengths,
                actual,
                expected,
                dtype=a["score_args"][0].dtype,
                head_dim=a["score_args"][0].shape[2],
            )
            checked += 1
    return checked


def validate_reference_selection(reference, lengths, selected, q, torch, helpers):
    """Reuse existing FP8 normalization/tolerances; preserve BF16 checks."""
    if q.dtype == torch.float8_e4m3fn:
        helpers.assert_upstream_selection(
            reference, lengths, selected, dtype=q.dtype, head_dim=q.shape[2]
        )
    else:
        helpers.assert_selection(reference, lengths, selected)


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
        for side, selected in (("baseline", left), ("candidate", right)):
            try:
                validate_reference_selection(
                    reference, lengths, selected, q, torch, helpers
                )
            except AssertionError as error:
                raise AssertionError(
                    f"{side} pipeline selection failed at query row {row}"
                ) from error
        helpers.assert_native_selection(
            reference, lengths, right, left, dtype=q.dtype, head_dim=q.shape[2]
        )
    return len(rows)


def common_fingerprints(chunks, torch):
    # Each tensor gets its own argument zero: fingerprint() otherwise treats
    # argument eight as an output and intentionally omits its content hash.
    return [support.fingerprint((c["common_logits"],), torch)[0] for c in chunks]


def worker(options, case, path):
    result = dict(
        case=case,
        status="running",
        stages={},
        candidate_kind=getattr(options, "candidate", "production"),
    )
    if shared.enabled(options):
        result.update(
            measurement_context="shared_gpu_exploratory", acceptance_eligible=False
        )
    support.write_json(path, result)
    torch, native, candidate, adapter, helpers, device, info = runtime(
        Path(options.reference_file), path, options
    )
    result["runtime"] = info
    os.environ["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = str(case["budget_mb"])
    native.envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB = case["budget_mb"]
    args = workloads.make_inputs(case, helpers, torch)
    if args[0].dtype != getattr(torch, case["dtype"]) or args[1].dtype != args[0].dtype:
        raise RuntimeError("Input Q/cache dtype differs from workload definition")
    native_args = (*args[:8], native_output(args[8], torch), args[9])
    result["inputs"] = support.fingerprint(args, torch)
    result["native_output_layout"] = dict(
        shape=list(native_args[8].shape),
        stride=list(native_args[8].stride()),
        storage_offset=native_args[8].storage_offset(),
    )
    native_chunks = ab.observe(native, "_topk", native_args)
    production = getattr(candidate, "production_baseline", None)
    if production is not None:
        fallback_entry = candidate._qsa_prefill_score_kernel
        baseline_entry = production._qsa_prefill_score_kernel
        if fallback_entry is baseline_entry or any(
            a is b
            for a, b in zip(fallback_entry.kernel_cache, baseline_entry.kernel_cache)
        ):
            raise RuntimeError(
                "Trial fallback and production baseline share LibEntry cache"
            )
        result["score_cache_isolation"] = dict(
            independent_outer_entries=True,
            independent_device_caches=True,
            baseline_public_warmed_first=True,
            baseline_configs_before_public=support.score_configs(production),
        )
        # Replay actual production chunk order outside timing, before any
        # candidate fallback can run. Preserve its real public tuning behavior.
        production.qsa_select_paged_prefill(*args)
        result["score_cache_isolation"]["baseline_configs_after_public"] = (
            support.score_configs(production)
        )
    retained = getattr(candidate, "retained_baseline", None)
    retained_stage = getattr(candidate, "retained_stage_name", "score_vs_key_major")
    retained_kind = getattr(candidate, "retained_baseline_kind", "retained_key_major")
    if retained is not None:
        entry = retained._qsa_prefill_score_kernel
        for other in (
            production._qsa_prefill_score_kernel,
            candidate._qsa_prefill_score_kernel,
        ):
            if entry is other or any(
                a is b for a, b in zip(entry.kernel_cache, other.kernel_cache)
            ):
                raise RuntimeError(
                    "Retained key-major baseline shares outer score cache"
                )
        retained.qsa_select_paged_prefill(*args)
        result["retained_key_major_score_configs"] = support.score_configs(retained)
        result["retained_baseline_public_warmed_first"] = True
        result["retained_baseline_kind"] = retained_kind
        result["retained_baseline_score_configs"] = result[
            "retained_key_major_score_configs"
        ]
        if hasattr(candidate, "freeze_baseline"):
            result["eviction_config_match"] = candidate.freeze_baseline(candidate)
    fastpath = workloads.geometry(case).get("select_all", False)
    packed_trial = getattr(options, "candidate", "production") == "packed-grid"
    current_trial = (
        getattr(options, "candidate", "production")
        in accuracy_profiles.CURRENT_CANDIDATES
    )
    trace_cache = current_trial or (
        getattr(options, "compare_previous_tuning", False)
        and getattr(options, "previous_tuning", "eight") == "round1"
    )
    if trace_cache:
        candidate_chunks, candidate_trace = cache_trial.observe(candidate, args)
        result["cache_granularity"] = dict(
            production=candidate_trace,
            policy="Operator-local entry key appends exact existing LibTuner key",
        )
        if (
            getattr(options, "candidate", "production")
            in accuracy_profiles.PAGE_TABLE_CANDIDATES
        ):
            result["page_table_trial"] = dict(
                baseline_source_sha256=page_table.PRODUCTION_SHA256,
                bf16_only=options.candidate == "page-table-bf16",
                route=(
                    "two scalar page IDs when not IS_FP8 and BLOCK_N<=PAGE_SIZE; "
                    "vector fetch otherwise"
                    if options.candidate == "page-table-bf16"
                    else "two scalar page IDs when BLOCK_N<=PAGE_SIZE; vector fetch otherwise"
                ),
                configs=support.score_configs(candidate),
            )
        if packed_trial:
            result["packed_grid"] = dict(
                baseline_source_sha256=packed.PRODUCTION_SHA256,
                chunks=packed.diagnostics(
                    args, candidate_trace, workloads.geometry(case)["query_lengths"]
                ),
                packed_configs=support.score_configs(
                    candidate, "_qsa_prefill_packed_score_kernel"
                ),
            )
    else:
        candidate_chunks = (
            [] if fastpath else ab.observe(candidate, "persistent_topk", args)
        )
    result["workload"] = verify_geometry(
        case, args, candidate_chunks, select_all=fastpath
    )
    verify_geometry(case, native_args, native_chunks)
    if not fastpath and ab.signatures(native_chunks) != ab.signatures(candidate_chunks):
        raise RuntimeError("Baseline/candidate scoring geometry differs")
    for c in candidate_chunks:
        c.pop("common_logits")  # Both Top-K stages use only native frozen scores.
    if adapter is not None:
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
    if fastpath:
        # Native vLLM still executes its real score/TopK sequence. Production
        # selects all directly. Compare complete device pipelines only.
        def native_pipeline():
            for c in native_chunks:
                logits = native._prefill_logits(*c["score_args"], **c["score_kwargs"])
                native._topk(logits, *c["topk_args"][1:], **c["topk_kwargs"])

        functions = dict(
            pipeline=dict(
                baseline=native_pipeline,
                candidate=ab.select_all_stage(candidate, args),
            )
        )
        result["stages"].update(
            {
                name: dict(
                    status="not_applicable",
                    accepted=False,
                    reason="production select-all skips score and TopK",
                )
                for name in ("score", "topk")
            }
        )
    else:
        functions = ab.bind_stages(native, candidate, native_chunks, candidate_chunks)
    previous = None
    previous_args = None
    previous_chunks = []
    if getattr(options, "compare_previous_tuning", False):
        if fastpath:
            raise RuntimeError("Previous tuning comparison requires scored cases")
        previous_kind = getattr(options, "previous_tuning", "eight")
        previous_identity = cache_trial.previous_identity(previous_kind)
        previous = cache_trial.previous_module(candidate, previous_kind)
        previous_args = (*args[:8], native_output(args[8], torch), args[9])
        if trace_cache:
            previous_chunks, previous_trace = cache_trial.observe(
                previous, previous_args
            )
            result["cache_granularity"]["previous"] = previous_trace
        else:
            previous_chunks = ab.observe(previous, "persistent_topk", previous_args)
        verify_geometry(case, previous_args, previous_chunks)
        for chunk in previous_chunks:
            chunk.pop("common_logits")
        if ab.signatures(previous_chunks) != ab.signatures(candidate_chunks):
            raise RuntimeError("Previous tuning changed public chunk geometry")
        previous_functions = ab.bind_stages(
            native, previous, native_chunks, previous_chunks
        )
        for name in ("score", "pipeline"):
            functions[name + "_vs_previous_tuning"] = dict(
                baseline=previous_functions[name]["candidate"],
                candidate=functions[name]["candidate"],
            )
        result["previous_tuning"] = dict(
            **previous_identity,
            same_topk=previous.persistent_topk is candidate.persistent_topk,
            chunks=ab.signatures(previous_chunks),
            independent_rectangular_cache=True,
            independent_compact_cache=True,
        )
    production_chunks = []
    production_args = None
    if production is not None and not fastpath:
        # Older score-only trials reuse captured candidate buffers. Current
        # trials capture separate public baseline buffers for pipeline A/B.
        production_chunks = candidate_chunks

        def production_scores():
            for c in production_chunks:
                production._prefill_logits(*c["score_args"], **c["score_kwargs"])

        functions["score_vs_production"] = dict(
            baseline=production_scores, candidate=functions["score"]["candidate"]
        )
        if current_trial:
            production_args = (*args[:8], native_output(args[8], torch), args[9])
            production_chunks, production_trace = cache_trial.observe(
                production, production_args
            )
            verify_geometry(case, production_args, production_chunks)
            if production_args[8].data_ptr() == args[8].data_ptr() or any(
                a["score_args"][5].data_ptr() == b["score_args"][5].data_ptr()
                for a, b in zip(production_chunks, candidate_chunks)
            ):
                raise RuntimeError(
                    "Current-production pipeline baseline shares candidate buffers"
                )
            if ab.signatures(production_chunks) != ab.signatures(candidate_chunks):
                raise RuntimeError("Trial changed public chunk geometry")
            for chunk in production_chunks:
                chunk.pop("common_logits")
            result["cache_granularity"]["production_baseline"] = production_trace
            production_functions = ab.bind_stages(
                native, production, native_chunks, production_chunks
            )
            functions["score_vs_production"] = dict(
                baseline=production_functions["score"]["candidate"],
                candidate=functions["score"]["candidate"],
            )
            functions["pipeline_vs_production"] = dict(
                baseline=production_functions["pipeline"]["candidate"],
                candidate=functions["pipeline"]["candidate"],
            )
    matched_pair = None
    split_trial = getattr(options, "candidate", "production") == "query-split"
    if split_trial:
        result["query_split"] = dict(
            baseline_source_sha256=query_split.PRODUCTION_SHA256,
            gate="IS_FP8 and TILE_R>=32",
            query_parts=2,
            configs=support.score_configs(candidate),
        )
        eligible = case["dtype"] == "float8_e4m3fn" and any(
            t["route"] == "rectangular" and t["constants"]["TILE_R"] >= 32
            for t in result["cache_granularity"]["production_baseline"]
        )
        if eligible:
            from tools import qsa_prefill_query_split_matched as matched

            matched_pair = matched.modules(
                production,
                candidate,
                result["cache_granularity"]["production_baseline"],
            )
            functions = dict(
                score_matched=dict(
                    baseline=scoring.score_function(
                        matched_pair[0]._prefill_logits, production_chunks
                    ),
                    candidate=scoring.score_function(
                        matched_pair[1]._prefill_logits, candidate_chunks
                    ),
                ),
                **functions,
            )
        else:
            result["stages"]["score_matched"] = dict(
                status="not_applicable",
                accepted=False,
                reason="No FP8 rectangular baseline tile eligible to split",
            )
    stage_trial = getattr(options, "candidate", "production") == "query-loop-stage"
    stage_joint_trial = (
        getattr(options, "candidate", "production") == "query-stage-joint"
    )
    unroll_trial = getattr(options, "candidate", "production") == "query-loop-unroll"
    combined_trial = (
        stage_trial
        or stage_joint_trial
        or unroll_trial
        or getattr(options, "candidate", "production") == "query-joint-page"
    )
    joint_trial = (
        combined_trial or getattr(options, "candidate", "production") == "query-joint"
    )
    matched_parts = (1, 2) if joint_trial else None
    if joint_trial:
        traces = result["cache_granularity"]["production"]
        from tools.qsa_prefill_query_joint_policy import same_tiles, validate_choices

        validate_choices(traces, case["dtype"] == "float8_e4m3fn")
        result["query_joint"] = dict(
            baseline_source_sha256=query_joint.PRODUCTION_SHA256,
            gate="IS_FP8 and TILE_R>=32 and QUERY_PARTS==2",
            configs=support.score_configs(candidate),
        )
        eligible = case["dtype"] == "float8_e4m3fn" and any(
            t["route"] == "rectangular" and t["constants"]["TILE_R"] >= 32
            for t in traces
        )
        if eligible:
            from tools import qsa_prefill_query_split_matched as matched

            matched_pair = matched.modules(
                candidate, candidate, traces, query_parts=matched_parts
            )
            functions = dict(
                score_matched=dict(
                    baseline=scoring.score_function(
                        matched_pair[0]._prefill_logits, production_chunks
                    ),
                    candidate=scoring.score_function(
                        matched_pair[1]._prefill_logits, candidate_chunks
                    ),
                ),
                **functions,
            )
        else:
            result["stages"]["score_matched"] = dict(
                status="not_applicable",
                accepted=False,
                reason="No selected FP8 rectangular tile eligible to split",
            )
        if case["name"] == "observed_short_batch3_bfloat16":
            functions["pipeline_buffer_swap"] = query_joint.swapped_pipeline_functions(
                native,
                production,
                candidate,
                native_chunks,
                production_chunks,
                candidate_chunks,
            )

            def buffer_layout(chunks):
                return [
                    dict(
                        score_shape=list(c["score_args"][5].shape),
                        score_stride=list(c["score_args"][5].stride()),
                        score_alignment=c["score_args"][5].data_ptr() % 256,
                        topk_workspace_shape=list(c["topk_args"][3].shape),
                        topk_workspace_stride=list(c["topk_args"][3].stride()),
                        topk_workspace_alignment=c["topk_args"][3].data_ptr() % 256,
                        output_shape=list(c["topk_args"][2].shape),
                        output_stride=list(c["topk_args"][2].stride()),
                    )
                    for c in chunks
                ]

            result["bf16_control"] = dict(
                policy="Canonical uses trial buffers; trial uses canonical buffers",
                original_baseline=buffer_layout(production_chunks),
                original_candidate=buffer_layout(candidate_chunks),
                baseline_configs=production_trace,
                candidate_configs=traces,
                same_tile_configs=same_tiles(production_trace, traces),
                same_topk=candidate.persistent_topk is production.persistent_topk,
            )
    if combined_trial:
        result["query_page"] = dict(
            baseline_source_sha256=query_page.PRODUCTION_SHA256,
            page_gate="not IS_FP8 and BLOCK_N<=PAGE_SIZE",
            configs=support.score_configs(candidate),
            same_topk=candidate.persistent_topk is production.persistent_topk,
        )
    if stage_trial:
        result["loop_stage"] = loop_stage.metadata(
            candidate, support.score_configs(candidate)
        )
    if unroll_trial:
        result["loop_unroll"] = loop_unroll.metadata(
            candidate, support.score_configs(candidate)
        )
    if stage_joint_trial:
        result["stage_joint"] = stage_joint.metadata(
            candidate, support.score_configs(candidate)
        )
    if retained is not None and not fastpath:
        functions[retained_stage] = dict(
            baseline=scoring.score_function(retained._prefill_logits, candidate_chunks),
            candidate=functions["score"]["candidate"],
        )
    for stage in functions.values():
        for fn in stage.values():
            fn()  # Compile/autotune and allocate operator-internal scratch untimed.
    device.synchronize()
    if hasattr(candidate, "verify_matched"):
        candidate.verify_matched(candidate, result["eviction_config_match"])
        result["eviction_config_match"]["verified_before_timing"] = True
    if is_flagtree(options):
        from tools import qsa_prefill_kernel_resources as resources

        result["kernel_resources"] = resources.collect(
            native, production or candidate, candidate
        )
        if packed_trial:
            result["kernel_resources"]["candidate_packed"] = resources.libentry_records(
                candidate._qsa_prefill_packed_score_kernel
            )
        if previous is not None:
            result["kernel_resources"]["previous_tuning"] = dict(
                rectangular=resources.libentry_records(
                    previous._qsa_prefill_score_kernel
                ),
                compact=resources.libentry_records(
                    previous._qsa_prefill_compact_score_kernel
                ),
            )
        if retained is not None:
            result["kernel_resources"]["retained_key_major"] = (
                resources.libentry_records(retained._qsa_prefill_score_kernel)
            )
            result["kernel_resources"]["retained_baseline_kind"] = retained_kind
    if matched_pair is not None:
        result["query_joint" if joint_trial else "query_split"]["matched"] = (
            matched.freeze_and_records(matched_pair, query_parts=matched_parts)
        )
    timer, timer_info = scoring.make_timer(
        options.kernel_warmup_ms, options.kernel_rep_ms
    )
    result["timer"] = {k: v for k, v in timer_info.items() if k != "includes_topk"}
    result["production_score_configs"] = support.score_configs(candidate)
    if previous is not None:
        result["tuning_round1"] = dict(
            production=round1.tuning_state(candidate),
            previous=round1.tuning_state(previous),
        )
        result["previous_tuning"]["rectangular_configs"] = support.score_configs(
            previous
        )
        result["previous_tuning"]["compact_configs"] = support.score_configs(
            previous, "_qsa_prefill_compact_score_kernel"
        )
    if production is not None:
        result["baseline_production_score_configs"] = support.score_configs(production)
        result["candidate_fallback_score_configs"] = support.score_configs(candidate)
    if hasattr(candidate, "_compact_eligible"):
        result["compact_score_configs"] = support.score_configs(
            candidate, "_qsa_prefill_compact_score_kernel"
        )
        result["score_schedule"] = dict(
            rectangular_dot=(
                "keys_x_query"
                if getattr(options, "candidate", "production") in KEY_MAJOR_CANDIDATES
                or getattr(options, "candidate", "production")
                in ("production", *accuracy_profiles.CURRENT_CANDIDATES)
                else "query_x_keys"
            ),
            routing_policy="rectangular_row_capacity_ge_4x_chunk_rows",
            ragged_envelope=args[0].shape[0] < args[2].shape[0] * args[7],
            compact_chunks=[
                c["offset"]
                for c in candidate_chunks
                if candidate._compact_eligible(
                    args[0].shape[0], args[2].shape[0], args[7], c["rows"]
                )
            ],
            historical_baseline_sha256=(
                packed.PRODUCTION_SHA256
                if current_trial
                else ab.PRODUCTION_SHA256 if production is not None else None
            ),
            production_source_sha256=info["operator_binding"]["canonical_sha256"],
            candidate_source_sha256=info["operator_binding"]["candidate_sha256"],
            grids=[
                dict(
                    tile_r=tile,
                    chunks=[
                        dict(
                            rectangular=args[2].shape[0]
                            * math.ceil(min(c["rows"], args[7]) / tile),
                            compact=c["rows"] // tile + args[2].shape[0],
                        )
                        for c in candidate_chunks
                    ],
                )
                for tile in (4, 8, 16, 32, 64)
            ],
        )
    result["topk_backends"] = dict(
        baseline="vLLM native CUDA _C dispatcher",
        candidate=(
            "repo production persistent_topk, HAS_TLE=True"
            if is_flagtree(options)
            else "repo top_k_per_row_prefill, HAS_TLE=False"
        ),
    )
    guard = support.gpu_guard_module()

    def validate(name):
        if name in SCORE_STAGES:
            if name == "score_matched":
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
                        (
                            "baseline",
                            matched_pair[0],
                            production_chunks,
                            production_args,
                        ),
                        ("candidate", matched_pair[1], candidate_chunks, args),
                    )
                }
            score_baselines = {
                "score": (native, native_chunks, native_args),
                "score_vs_production": (
                    production,
                    production_chunks,
                    production_args or args,
                ),
                "score_vs_key_major": (retained, candidate_chunks, args),
                "score_vs_reuse": (retained, candidate_chunks, args),
                "score_vs_previous_tuning": (previous, previous_chunks, previous_args),
            }
            baseline_module, baseline_chunks, baseline_args = score_baselines[name]
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
                    ("baseline", baseline_module, baseline_chunks, baseline_args),
                    ("candidate", candidate, candidate_chunks, args),
                )
            }
        if name == "topk":
            return dict(
                sample_rows=validate_topk(
                    native, candidate, native_chunks, candidate_chunks, torch, helpers
                )
            )
        if name == "pipeline_buffer_swap":
            return dict(
                sample_rows=validate_pipeline(
                    functions[name],
                    args,
                    production_args,
                    native_chunks,
                    torch,
                    helpers,
                )
            )
        return dict(
            sample_rows=validate_pipeline(
                functions[name],
                (
                    previous_args
                    if name == "pipeline_vs_previous_tuning"
                    else (
                        production_args
                        if name == "pipeline_vs_production"
                        else native_args
                    )
                ),
                args,
                native_chunks,
                torch,
                helpers,
            )
        )

    profile_capture = getattr(options, "profile_capture", False)
    if profile_capture:
        from tools import qsa_prefill_profiling as profiling

        profiling.capture(
            options,
            result,
            path,
            functions,
            validate,
            torch,
            device,
            lambda label: support.guarded_telemetry(
                info["uuid"], label, result, path, guard
            ),
        )
    for name, stage in () if profile_capture else functions.items():
        entry = result["stages"][name] = dict(
            status="validating",
            accepted=False,
            includes_topk=name not in SCORE_STAGES,
            baseline_kind={
                "score_vs_production": "production",
                "score_matched": (
                    "joint_tile_full_query"
                    if joint_trial
                    else "production_fixed_config"
                ),
                "pipeline_buffer_swap": "production_swapped_buffers",
                "pipeline_vs_production": "production",
                "score_vs_key_major": "retained_key_major",
                "score_vs_reuse": "retained_reuse",
                "score_vs_previous_tuning": (
                    previous_identity["baseline_kind"] if previous else None
                ),
                "pipeline_vs_previous_tuning": (
                    previous_identity["baseline_kind"] if previous else None
                ),
            }.get(name, "native_vllm"),
        )
        entry["validation_before"] = validate(name)
        support.write_json(path, result)
        measured, observations, order = scoring.measure_scores(
            timer,
            stage["baseline"],
            stage["candidate"],
            options.rounds,
            lambda label: (
                shared.telemetry(info["uuid"], name + ":" + label, result, path)
                if shared.enabled(options)
                else support.guarded_telemetry(
                    info["uuid"], name + ":" + label, result, path, guard
                )
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
        entry.update(
            status="passed",
            measurement_validated=True,
            accepted=not shared.enabled(options),
        )
        support.write_json(path, result)
    if getattr(options, "diagnostic_self_control", False) and not profile_capture:
        from tools import qsa_prefill_profiling as profiling

        profiling.self_controls(
            options,
            result,
            path,
            functions,
            timer,
            lambda label: support.guarded_telemetry(
                info["uuid"], label, result, path, guard
            ),
        )
        result["diagnostics"]["validation_after_self_controls"] = {
            name: validate(name)
            for name in (
                "score",
                "score_vs_production",
                "pipeline",
                "pipeline_vs_production",
            )
        }
    if getattr(options, "query_control_cycles", 0):
        from tools import qsa_prefill_query_controls as controls

        controls.run(
            options,
            result,
            path,
            bench=sys.modules[__name__],
            torch=torch,
            helpers=helpers,
            device=device,
            native=native,
            candidate=candidate,
            production=production,
            native_chunks=native_chunks,
            production_chunks=production_chunks,
            candidate_chunks=candidate_chunks,
            production_args=production_args,
            args=args,
            monitor=lambda label: support.guarded_telemetry(
                info["uuid"], label, result, path, guard
            ),
        )
    if getattr(options, "combined_self_control_cycles", 0):
        from tools import qsa_prefill_combined_controls as controls

        controls.run(
            options,
            result,
            path,
            bench=sys.modules[__name__],
            torch=torch,
            helpers=helpers,
            device=device,
            native=native,
            candidate=candidate,
            production=production,
            native_chunks=native_chunks,
            production_chunks=production_chunks,
            candidate_chunks=candidate_chunks,
            production_args=production_args,
            args=args,
            monitor=lambda label: support.guarded_telemetry(
                info["uuid"], label, result, path, guard
            ),
        )
    result["common_logits_after"] = common_fingerprints(native_chunks, torch)
    if (
        joint_trial
        and support.score_configs(candidate) != result["query_joint"]["configs"]
    ):
        raise RuntimeError("Joint query tuning changed during timing")
    if (
        split_trial
        and support.score_configs(candidate) != result["query_split"]["configs"]
    ):
        raise RuntimeError("Query-split tuning changed during timing")
    if (
        matched_pair is not None
        and matched.freeze_and_records(matched_pair, query_parts=matched_parts)
        != result["query_joint" if joint_trial else "query_split"]["matched"]
    ):
        raise RuntimeError("Fixed-config query-split kernels changed during timing")
    if (
        "page_table_trial" in result
        and support.score_configs(candidate) != result["page_table_trial"]["configs"]
    ):
        raise RuntimeError("Page-table trial tuning changed during timing")
    if (
        packed_trial
        and support.score_configs(candidate, "_qsa_prefill_packed_score_kernel")
        != result["packed_grid"]["packed_configs"]
    ):
        raise RuntimeError("Packed-grid tuning changed during timing")
    if (
        stage_trial
        and loop_stage.metadata(candidate, support.score_configs(candidate))
        != result["loop_stage"]
    ):
        raise RuntimeError("Loop-stage metadata changed during timing")
    if (
        stage_joint_trial
        and stage_joint.metadata(candidate, support.score_configs(candidate))
        != result["stage_joint"]
    ):
        raise RuntimeError("Joint-stage metadata changed during timing")
    if (
        unroll_trial
        and loop_unroll.metadata(candidate, support.score_configs(candidate))
        != result["loop_unroll"]
    ):
        raise RuntimeError("Loop-unroll metadata changed during timing")
    if result["common_logits_before"] != result["common_logits_after"]:
        raise RuntimeError("Common Top-K logits were modified during this run")
    if previous is not None and (
        support.score_configs(previous)
        != result["previous_tuning"]["rectangular_configs"]
        or support.score_configs(previous, "_qsa_prefill_compact_score_kernel")
        != result["previous_tuning"]["compact_configs"]
        or previous.persistent_topk is not candidate.persistent_topk
    ):
        raise RuntimeError("Previous tuning or TopK binding changed during timing")
    if (
        production is not None
        and support.score_configs(production)
        != result["score_cache_isolation"]["baseline_configs_after_public"]
    ):
        raise RuntimeError(
            "Production baseline dispatch configuration changed after public warmup"
        )
    if (
        retained is not None
        and support.score_configs(retained)
        != result["retained_key_major_score_configs"]
    ):
        raise RuntimeError(
            "Retained key-major configuration changed after public warmup"
        )
    if hasattr(candidate, "verify_matched"):
        candidate.verify_matched(candidate, result["eviction_config_match"])
        result["eviction_config_match"]["verified_after_timing"] = True
    if is_flagtree(options):
        from tools import qsa_prefill_flagtree_ab as profile

        if (
            profile.compiler_runtime(
                torch, sys.modules["triton"], options.flagtree, options.report_dir
            )
            != info["flagtree_runtime"]
        ):
            raise RuntimeError("Compiler/driver identity changed during benchmark")
    result.update(status="passed", peak_allocated_bytes=device.max_memory_allocated())
    support.write_json(path, result)


def write_table(summary, directory):
    exploratory = (
        summary.get("manifest", {})
        .get("options", {})
        .get("shared_gpu_exploratory", False)
    )
    kind = summary.get("manifest", {}).get("options", {}).get("candidate")
    trial = kind in (
        "compact-ragged",
        "query-loop-stage",
        "query-stage-joint",
        "query-loop-unroll",
        *KEY_MAJOR_CANDIDATES,
    )
    candidate_column = (
        {
            "key-major": "gems_key_major_trial_us",
            "key-major-reuse": "gems_key_major_reuse_trial_us",
            "key-major-evict-first": "gems_key_major_evict_first_trial_us",
            "compact-ragged": "gems_compact_trial_us",
            "query-loop-stage": "gems_loop_stage_trial_us",
            "query-stage-joint": "gems_stage_joint_trial_us",
            "query-loop-unroll": "gems_loop_unroll_trial_us",
        }[kind]
        if trial
        else (
            "gems_production_us"
            if summary.get("manifest", {}).get("options", {}).get("compiler")
            == "flagtree"
            else "gems_prefill_topk_us"
        )
    )
    lines = [
        f"case\tstage\tstatus\tbaseline_us\t{candidate_column}\t"
        f"{'exploratory_speedup' if exploratory else 'speedup'}\tpaired_ratio_median\tbaseline_kind"
    ]
    for case in summary["cases"]:
        for name in STAGE_ORDER:
            if name not in case.get("stages", {}) and (
                "_vs_" in name or name in ("score_matched", "pipeline_buffer_swap")
            ):
                continue
            entry = case.get("stages", {}).get(name, {})
            accepted = case["status"] == "passed" and (
                entry.get("measurement_validated")
                if exploratory
                else entry.get("accepted")
            )
            values = [
                case["case"]["name"],
                name,
                (
                    ("exploratory" if exploratory else "passed")
                    if accepted
                    else (
                        entry.get("status", "not_collected")
                        if case["status"] == "passed"
                        else case["status"]
                    )
                ),
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
            values.append(entry.get("baseline_kind", "native_vllm"))
            lines.append("\t".join(map(str, values)))
    (directory / "triton-stage-ab.tsv").write_text("\n".join(lines) + "\n")
    grouped = {}
    for group in workloads.GROUPS:
        members = [c for c in summary["cases"] if c["case"].get("kind") == group]
        planned = [
            c
            for c in summary.get("manifest", {}).get("cases", [])
            if c.get("kind") == group
        ]
        if planned and len(members) != len(planned):
            continue
        for name in STAGE_ORDER:
            relevant = [
                c
                for c in members
                if name in c.get("stages", {})
                and c["stages"][name].get("status") != "not_applicable"
            ]
            if not relevant or any(c["status"] != "passed" for c in members):
                continue
            valid = [
                c["stages"][name]["speedup"]
                for c in relevant
                if c["status"] == "passed"
                and c.get("stages", {})
                .get(name, {})
                .get("measurement_validated" if exploratory else "accepted")
            ]
            # Do not publish a geometric mean over a silently incomplete set.
            if len(valid) == len(relevant):
                grouped[f"{group}:{name}"] = dict(
                    count=len(valid),
                    geometric_mean=math.exp(sum(map(math.log, valid)) / len(valid)),
                    acceptance_eligible=not exploratory,
                )
    summary["grouped_speedups"] = grouped
    support.write_json(directory / "grouped-speedups.json", grouped)
    summary["overall_acceptance"] = (
        packed.acceptance(summary)
        if summary.get("manifest", {}).get("options", {}).get("candidate")
        == "packed-grid"
        else (
            page_table.acceptance(
                summary,
                bf16_only=summary.get("manifest", {})
                .get("options", {})
                .get("candidate")
                == "page-table-bf16",
            )
            if summary.get("manifest", {}).get("options", {}).get("candidate")
            in accuracy_profiles.PAGE_TABLE_CANDIDATES
            else round1.acceptance(summary)
        )
    )
    if summary.get("manifest", {}).get("options", {}).get("candidate") == "query-split":
        summary["overall_acceptance"] = query_split.acceptance(summary)
    if summary.get("manifest", {}).get("options", {}).get("candidate") == "query-joint":
        summary["overall_acceptance"] = query_joint.acceptance(summary)
    if (
        summary.get("manifest", {}).get("options", {}).get("candidate")
        == "query-joint-page"
    ):
        summary["overall_acceptance"] = query_page.acceptance(summary)
    if (
        summary.get("manifest", {}).get("options", {}).get("candidate")
        == "query-loop-stage"
    ):
        summary["overall_acceptance"] = loop_stage.acceptance(summary)
    if (
        summary.get("manifest", {}).get("options", {}).get("candidate")
        == "query-stage-joint"
    ):
        summary["overall_acceptance"] = stage_joint.acceptance(summary)
    if (
        summary.get("manifest", {}).get("options", {}).get("candidate")
        == "query-loop-unroll"
    ):
        summary["overall_acceptance"] = loop_unroll.acceptance(summary)
    accuracy_profiles.annotate_acceptance(
        summary["overall_acceptance"],
        summary.get("manifest", {}).get("options", {}).get("accuracy_profile", "full"),
        summary.get("manifest", {}).get("options", {}).get("candidate", "production"),
        summary.get("accuracy", {}),
    )
    support.write_json(
        directory / "overall-acceptance.json", summary["overall_acceptance"]
    )


def child_command(options, directory, flag):
    return [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--compiler",
        getattr(options, "compiler", "stock"),
        "--flagtree",
        getattr(options, "flagtree", "/opt/flagtree"),
        "--suite",
        options.suite,
        "--candidate",
        getattr(options, "candidate", "production"),
        "--accuracy-profile",
        accuracy_profiles.name(options),
        "--report-dir",
        str(directory),
        "--reference-file",
        options.reference_file,
        "--rounds",
        str(options.rounds),
        "--combined-self-control-cycles",
        str(getattr(options, "combined_self_control_cycles", 0)),
        "--query-control-cycles",
        str(getattr(options, "query_control_cycles", 0)),
        "--kernel-warmup-ms",
        str(options.kernel_warmup_ms),
        "--kernel-rep-ms",
        str(options.kernel_rep_ms),
        "--max-initial-temp-c",
        str(getattr(options, "max_initial_temp_c", 75)),
        *(
            ["--compare-previous-tuning"]
            if getattr(options, "compare_previous_tuning", False)
            else []
        ),
        *(
            ["--previous-tuning", getattr(options, "previous_tuning", "eight")]
            if getattr(options, "compare_previous_tuning", False)
            else []
        ),
        *(
            [
                "--shared-gpu-exploratory",
                "--shared-min-free-mib",
                str(options.shared_min_free_mib),
            ]
            if shared.enabled(options)
            else []
        ),
        *(
            ["--diagnostic-self-control"]
            if getattr(options, "diagnostic_self_control", False)
            else []
        ),
        flag,
    ]


def runtime_identity(info):
    from tools.qsa_prefill_score_resume import fingerprint

    return dict(
        fingerprint(info),
        stock_wheel=info.get("stock_wheel"),
        compiler_profile=info.get("compiler_profile"),
        flagtree_runtime=info.get("flagtree_runtime"),
        native_topk_libraries=info.get("native_topk_libraries"),
        vllm_version=info.get("versions", {}).get("vllm"),
        operator_binding=info.get("operator_binding"),
    )


def child_environment(directory, label):
    env = os.environ.copy()
    env.pop("FLAGGEMS_DB_URL", None)
    env.update(
        TRITON_CACHE_DIR=str(directory / "cache" / label / "triton"),
        FLAGGEMS_CACHE_DIR=str(directory / "cache" / label / "gems"),
    )
    return env


def accuracy_worker(options, path):
    if not is_flagtree(options):
        raise RuntimeError("QSA correctness requires the FlagTree profile")
    *_, info = runtime(Path(options.reference_file), path, options)
    support.write_json(
        path,
        dict(
            status="running",
            runtime=info,
            accuracy_profile=accuracy_profiles.name(options),
        ),
    )
    os.environ.pop("PYTEST_ADDOPTS", None)
    import pytest

    code = pytest.main(
        [
            "-q",
            "-ra",
            "-x",
            "-c",
            str(ROOT / "pytest.ini"),
            *accuracy_profiles.pytest_targets(options, ROOT),
            "--record",
            "json",
            "--output",
            str(Path(options.report_dir) / "accuracy-full.json"),
        ]
    )
    *_, after = runtime(Path(options.reference_file), options=options)
    if runtime_identity(info) != runtime_identity(after):
        raise RuntimeError(
            "Compiler/GPU/native-extension identity changed during accuracy"
        )
    support.write_json(
        path,
        dict(
            status="passed" if code == 0 else "failed",
            runtime=after,
            accuracy_profile=accuracy_profiles.name(options),
        ),
    )
    return int(code)


def idle_gate(options, summary, directory, label, guard):
    if shared.enabled(options):
        return shared.admission(
            summary,
            directory,
            os.environ,
            label,
            options.shared_min_free_mib,
            options.cooldown_timeout,
            options.cooldown_interval,
            max_initial_temp_c=getattr(options, "max_initial_temp_c", 75),
        )
    if is_flagtree(options):
        from tools import qsa_prefill_flagtree_ab as profile

        return profile.cooldown_gate(
            summary,
            directory,
            os.environ,
            label,
            guard,
            options.cooldown_timeout,
            options.cooldown_interval,
            max_initial_temp_c=getattr(options, "max_initial_temp_c", 75),
        )
    return support.idle_gpu_gate(summary, directory, os.environ, label, guard)


def run_accuracy_gate(options, directory, summary, preflight, guard):
    if not idle_gate(options, summary, directory, "before_accuracy", guard):
        raise RuntimeError("GPU environment rejected before correctness suite")
    print(
        "Running "
        + accuracy_profiles.name(options)
        + " correctness, including native vLLM comparisons",
        flush=True,
    )
    with (directory / "accuracy-full.txt").open("w") as log:
        try:
            proc = subprocess.run(
                child_command(options, directory, "--accuracy"),
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=options.accuracy_timeout,
                env=child_environment(directory, "accuracy"),
            )
            error = None if proc.returncode == 0 else f"accuracy exit={proc.returncode}"
        except subprocess.TimeoutExpired:
            error = "Correctness suite timed out: " + accuracy_profiles.name(options)
    counts = {}
    if not error:
        counts, error = support.accuracy_results(directory / "accuracy-full.json")
    identity_path = directory / "accuracy-runtime.json"
    identity = json.loads(identity_path.read_text()) if identity_path.is_file() else {}
    if not error and (
        identity.get("status") != "passed"
        or identity.get("accuracy_profile", "full") != accuracy_profiles.name(options)
        or runtime_identity(identity["runtime"])
        != runtime_identity(preflight["runtime"])
    ):
        error = "Accuracy and benchmark profile/compiler/GPU/native-extension identities differ"
    if not error and accuracy_profiles.name(options) == "iteration":
        records = json.loads((directory / "accuracy-full.json").read_text())
        error = accuracy_profiles.validate(
            records, getattr(options, "candidate", "production")
        )
    if not error and accuracy_profiles.name(options) == "full":
        records = json.loads((directory / "accuracy-full.json").read_text())
        for name in (
            "test_parameterized_against_vllm",
            "test_native_vllm_topk_dispatch",
            "test_native_vllm_empty_and_select_all",
            "test_against_pinned_vllm",
            "test_upstream_prefill_against_vllm",
            "test_upstream_split_prefill",
            "test_upstream_no_complete_blocks_dim8",
            *(
                (
                    "test_integrated_public_entry_identity",
                    "test_integrated_observed_against_native",
                    "test_integrated_rectangular_every_config",
                    "test_integrated_compact_every_config",
                    "test_integrated_full_entry_and_retained_candidate",
                    "test_score_cache_separates_runtime_rows",
                )
                if getattr(options, "candidate", "production")
                in ("production", *accuracy_profiles.CURRENT_CANDIDATES)
                else (
                    "test_observed_envelopes_against_native",
                    "test_compact_every_config_and_chunk_boundary",
                    "test_compact_full_entry",
                )
            ),
        ):
            if not any("::" + name in node for node in records):
                error = "Missing required direct comparison coverage: " + name
                break
        if not error and getattr(options, "candidate", "production") in (
            "production",
            *accuracy_profiles.CURRENT_CANDIDATES,
        ):
            for case in workloads.fp8.cases():
                node = "::test_fp8_benchmark_against_native[" + case["name"] + "]"
                if not any(node in name for name in records):
                    error = "Missing required FP8 benchmark coverage: " + case["name"]
                    break
        if not error and getattr(options, "candidate", "production") == "packed-grid":
            for name in ("test_packed_score_boundaries", "test_packed_full_entry"):
                if not any("::" + name in node for node in records):
                    error = "Missing packed-grid correctness coverage: " + name
                    break
        if not error and getattr(options, "candidate", "production") in (
            "query-split",
            "query-joint",
            "query-joint-page",
            "query-loop-stage",
            "query-stage-joint",
            "query-loop-unroll",
        ):
            for name in ("test_query_split_score", "test_query_split_full_entry"):
                if not any("::" + name in node for node in records):
                    error = "Missing query-split correctness coverage: " + name
                    break
        if (
            not error
            and getattr(options, "candidate", "production") == "query-stage-joint"
        ):
            error = stage_joint.validate_full_coverage(records)
        if (
            not error
            and getattr(options, "candidate", "production") == "query-loop-unroll"
        ):
            error = loop_unroll.validate_full_coverage(records)
        if (
            not error
            and getattr(options, "candidate", "production") == "query-loop-stage"
        ):
            error = loop_stage.validate_full_coverage(records)
        if (
            not error
            and getattr(options, "candidate", "production") == "query-joint-page"
        ):
            error = query_page.validate_full_coverage(records)
        if (
            not error
            and getattr(options, "candidate", "production")
            in accuracy_profiles.PAGE_TABLE_CANDIDATES
        ):
            for name in ("test_page_table_score", "test_page_table_full_entry"):
                if not any("::" + name in node for node in records):
                    error = "Missing page-table correctness coverage: " + name
                    break
        if (
            not error
            and getattr(options, "candidate", "production") in KEY_MAJOR_CANDIDATES
        ):
            for name in (
                "test_key_major_every_config_and_chunk_boundary",
                "test_key_major_observed_against_native",
                "test_key_major_full_entry",
            ):
                if not any("::" + name in node for node in records):
                    error = "Missing required key-major comparison coverage: " + name
                    break
        if not error and getattr(options, "candidate", "production") in (
            "key-major-reuse",
            "key-major-evict-first",
        ):
            for name in (
                "test_reuse_new_configs_and_chunk_boundary",
                "test_reuse_observed_against_native",
                "test_reuse_full_entry",
            ):
                if not any("::" + name in node for node in records):
                    error = "Missing required reuse comparison coverage: " + name
                    break
        if (
            not error
            and getattr(options, "candidate", "production") == "key-major-evict-first"
        ):
            for name in (
                "test_eviction_every_config_and_chunk_boundary",
                "test_eviction_observed_against_native",
                "test_eviction_full_entry",
                "test_eviction_matched_baseline_config",
            ):
                if not any("::" + name in node for node in records):
                    error = "Missing required eviction comparison coverage: " + name
                    break
    summary["accuracy"] = dict(
        status="failed" if error else "passed",
        counts=counts,
        error=error,
        profile=accuracy_profiles.name(options),
        full_suite_validated=not error and accuracy_profiles.name(options) == "full",
    )
    support.write_json(directory / "stage-summary.json", summary)
    if error:
        raise RuntimeError("Accuracy gate failed; no timings collected: " + error)


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
                (
                    "Same verified FlagTree/ptxas13.4/CUDA13.4compat; "
                    "unmodified production persistent_topk; accuracy profile="
                    + accuracy_profiles.name(options)
                    + "; "
                    if is_flagtree(options)
                    else "Same verified stock Triton wheel; repo non-TLE Top-K variant; "
                )
                + "native CUDA vLLM Top-K; preallocated outer workspaces; "
                "actual sequential pipeline; no production source edits; "
                + "candidate="
                + getattr(options, "candidate", "production")
            ),
        ),
    )
    path = directory / "stage-summary.json"
    if shared.enabled(options):
        summary.update(
            measurement_context="shared_gpu_exploratory", acceptance_eligible=False
        )
        summary["manifest"][
            "policy"
        ] += "; SHARED GPU EXPLORATORY ONLY: never performance acceptance"
    support.write_json(path, summary)
    try:
        configure_environment(options)
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
        if not idle_gate(options, summary, directory, "before_preflight", guard):
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
                "Compiler/native-extension preflight failed; see preflight.txt/json"
            )
        if is_flagtree(options):
            if getattr(options, "skip_accuracy", False):
                summary["accuracy"] = dict(
                    status="skipped_by_request",
                    counts={},
                    error=None,
                    profile=accuracy_profiles.name(options),
                    full_suite_validated=False,
                    reason="--skip-accuracy: standalone pytest not run; timed numeric checks retained",
                )
                support.write_json(
                    directory / "accuracy-skipped.json", summary["accuracy"]
                )
                support.write_json(path, summary)
                print(
                    "Skipping standalone pytest; preflight and timed numeric/GPU checks remain enabled",
                    flush=True,
                )
            else:
                run_accuracy_gate(options, directory, summary, preflight, guard)
        for case in selected:
            if not idle_gate(
                options, summary, directory, "before_" + case["name"], guard
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
            from tools.qsa_prefill_report import show_timing

            show_timing(record, source=case_path)
        summary["status"] = (
            "passed"
            if all(c["status"] == "passed" for c in summary["cases"])
            else "failed"
        )
        return 0 if summary["status"] == "passed" else 1
    except BaseException:
        summary.update(
            status=(
                "environment_rejected"
                if summary.get("environment_rejected")
                else "failed"
            ),
            error=traceback.format_exc(),
        )
        print(summary["error"], file=sys.stderr)
        return 2 if summary.get("environment_rejected") else 1
    finally:
        write_table(summary, directory)
        support.write_json(path, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("kernel",), default="kernel")
    parser.add_argument("--compiler", choices=("stock", "flagtree"), default="stock")
    parser.add_argument("--flagtree", default="/opt/flagtree")
    parser.add_argument("--shared-gpu-exploratory", action="store_true")
    parser.add_argument("--shared-min-free-mib", type=int, default=5120)
    parser.add_argument(
        "--suite",
        choices=workloads.SUITES,
        default="fp8-prefill",
    )
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--compare-previous-tuning", action="store_true")
    parser.add_argument("--accuracy-profile", choices=("iteration", "full"))
    parser.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="Skip standalone pytest only; retain preflight, timed numeric validation and GPU guards",
    )
    parser.add_argument(
        "--previous-tuning",
        choices=("eight", "round1", "combined", "stage-joint"),
        default="eight",
    )
    parser.add_argument(
        "--candidate",
        choices=(
            "production",
            "compact-ragged",
            *accuracy_profiles.CURRENT_CANDIDATES,
            *KEY_MAJOR_CANDIDATES,
        ),
        default="production",
    )
    parser.add_argument("--diagnostic-self-control", action="store_true")
    parser.add_argument(
        "--profile-capture", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-accuracy-tests", action="store_true")
    parser.add_argument("--report-dir")
    parser.add_argument("--reference-file")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--query-control-cycles", type=int, default=0)
    parser.add_argument("--combined-self-control-cycles", type=int, default=0)
    parser.add_argument("--kernel-warmup-ms", type=float, default=200)
    parser.add_argument("--kernel-rep-ms", type=float, default=100)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--accuracy-timeout", type=int, default=7200)
    parser.add_argument("--cooldown-timeout", type=float, default=900)
    parser.add_argument("--cooldown-interval", type=float, default=10)
    parser.add_argument("--max-initial-temp-c", type=int)
    parser.add_argument("--accuracy", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--preflight", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.accuracy_profile is None:
        options.accuracy_profile = (
            "iteration"
            if is_flagtree(options)
            and options.candidate
            in ("production", *accuracy_profiles.CURRENT_CANDIDATES)
            else "full"
        )
    if options.accuracy_profile == "iteration" and (
        not is_flagtree(options)
        or options.candidate
        not in ("production", *accuracy_profiles.CURRENT_CANDIDATES)
    ):
        parser.error(
            "Iteration correctness requires FlagTree production/current score trials"
        )
    if options.skip_accuracy and not is_flagtree(options):
        parser.error("--skip-accuracy requires the verified FlagTree benchmark profile")
    if options.skip_accuracy and (
        options.accuracy or options.preflight or options.worker
    ):
        parser.error("--skip-accuracy is a parent scheduling option only")
    if options.max_initial_temp_c is None:
        options.max_initial_temp_c = 80 if is_flagtree(options) else 75
    if not 1 <= options.max_initial_temp_c <= 150:
        parser.error("Initial temperature limit must be in 1..150 C")
    if not is_flagtree(options) and options.max_initial_temp_c != 75:
        parser.error("Temperature overrides require the verified FlagTree profile")
    if options.candidate != "production" and options.compiler != "flagtree":
        parser.error("Private score trials require the verified FlagTree profile")
    if (
        options.candidate in accuracy_profiles.CURRENT_CANDIDATES
        and options.suite != "scored-prefill"
    ):
        parser.error("Current score trial requires the frozen scored-prefill30 suite")
    if options.compare_previous_tuning and not (
        is_flagtree(options)
        and options.suite == "scored-prefill"
        and (
            (
                options.candidate == "production"
                and options.previous_tuning in ("eight", "round1")
            )
            or (
                options.candidate in ("query-loop-stage", "query-stage-joint")
                and options.previous_tuning == "combined"
            )
            or (
                options.candidate == "query-loop-unroll"
                and options.previous_tuning == "stage-joint"
            )
        )
    ):
        parser.error(
            "Previous baseline requires production eight/round1 or stage trial combined"
        )
    if options.candidate in ("query-loop-stage", "query-stage-joint") and not (
        options.compare_previous_tuning and options.previous_tuning == "combined"
    ):
        parser.error(
            "Stage trials require --compare-previous-tuning --previous-tuning combined"
        )
    if options.candidate == "query-loop-unroll" and not (
        options.compare_previous_tuning and options.previous_tuning == "stage-joint"
    ):
        parser.error(
            "Loop-unroll requires --compare-previous-tuning --previous-tuning stage-joint"
        )
    if options.previous_tuning != "eight" and not options.compare_previous_tuning:
        parser.error("Select --compare-previous-tuning for the previous baseline")
    if options.shared_gpu_exploratory and options.compiler != "flagtree":
        parser.error("Shared diagnostics require the FlagTree CUDA13.4 profile")
    if options.list_accuracy_tests:
        targets = accuracy_profiles.pytest_targets(options, ROOT)
        print(
            json.dumps(
                dict(
                    profile=options.accuracy_profile,
                    selection=(
                        "exact_nodes"
                        if options.accuracy_profile == "iteration"
                        else "full_files"
                    ),
                    targets=targets,
                ),
                indent=2,
            )
        )
        return 0
    all_cases = workloads.cases(options.suite)
    if options.list_cases:
        print(json.dumps(all_cases, indent=2))
        return 0
    if (
        min(
            options.rounds,
            options.kernel_warmup_ms,
            options.kernel_rep_ms,
            options.timeout,
            options.accuracy_timeout,
            options.cooldown_timeout,
            options.cooldown_interval,
            options.shared_min_free_mib,
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
    if options.query_control_cycles:
        from tools.qsa_prefill_query_controls import CASES

        if (
            options.compiler != "flagtree"
            or options.candidate != "query-joint"
            or options.suite != "scored-prefill"
            or options.shared_gpu_exploratory
            or options.rounds != 2
            or options.query_control_cycles not in (4, 8, 12, 16)
            or (
                options.worker is None
                and not options.preflight
                and not options.accuracy
                and set(options.cases or ()) != set(CASES)
            )
            or (options.worker is not None and options.worker not in CASES)
        ):
            parser.error(
                "Query controls require FlagTree query-joint, the two focused cases, "
                "exclusive GPU, two rounds and4/8/12/16 cycles"
            )
    if options.combined_self_control_cycles:
        from tools.qsa_prefill_combined_controls import CASE

        if (
            options.compiler != "flagtree"
            or options.candidate != "query-joint-page"
            or options.suite != "scored-prefill"
            or options.shared_gpu_exploratory
            or options.query_control_cycles
            or options.rounds != 2
            or options.combined_self_control_cycles not in (4, 8, 12, 16)
            or (
                options.worker is None
                and not options.preflight
                and not options.accuracy
                and options.cases != [CASE]
            )
            or (options.worker is not None and options.worker != CASE)
        ):
            parser.error(
                "Combined self-controls require FlagTree query-joint-page, "
                "short_batch3 only, exclusive GPU, two rounds and4/8/12/16 cycles"
            )
    if options.profile_capture and not options.worker:
        parser.error("Hardware capture is a dedicated worker, never a benchmark parent")
    if (options.profile_capture or options.diagnostic_self_control) and (
        options.compiler != "flagtree"
        or options.candidate != "query-stage-joint"
        or options.shared_gpu_exploratory
        or options.query_control_cycles
        or options.combined_self_control_cycles
    ):
        parser.error("Profiling diagnostics require exclusive FlagTree stage-joint")
    directory = (
        Path(options.report_dir).resolve()
        if options.report_dir
        else Path(tempfile.mkdtemp(prefix="qsa-triton-ab-h800."))
    )
    if options.worker or options.preflight or options.accuracy:
        if not options.report_dir or not options.reference_file:
            parser.error("Child requires report directory and pinned reference file")
        if options.worker and options.worker not in definitions:
            parser.error("Unknown worker case")
        path = directory / (
            ("profile-" if options.profile_capture else "") + options.worker + ".json"
            if options.worker
            else "accuracy-runtime.json" if options.accuracy else "preflight.json"
        )
        try:
            if options.accuracy:
                return accuracy_worker(options, path)
            elif options.preflight:
                torch, native, candidate, _, helpers, _, info = runtime(
                    Path(options.reference_file), path, options
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
    allowed = {"runner.txt"}
    if is_flagtree(options):
        allowed |= {
            "compat134.json",
            "compat134-package.json",
            "flagtree134-tools.json",
            "toolchain-downloads.json",
        }
    if any(p.name not in allowed for p in directory.iterdir()):
        parser.error(
            "Report directory must be fresh; existing reports are not overwritten"
        )
    print("Reports:", directory, flush=True)
    selected = [c for c in all_cases if not options.cases or c["name"] in options.cases]
    return parent(options, selected, directory)


if __name__ == "__main__":
    sys.exit(main())
