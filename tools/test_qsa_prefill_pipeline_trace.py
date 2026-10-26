# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Check pinned-cache replay, pass extraction and the no-launch contract."""

import ast
import copy
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_pipeline_trace as trace


class PipelineTraceTests(unittest.TestCase):
    def test_cache_lookup_pins_actual_PTX_and_requires_original_IR(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "match"
            good.mkdir()
            p = good / "kernel.ptx"
            p.write_text("original PTX")
            expected = trace.sha(p)
            (good / "kernel.ttir").write_text("original TTIR")
            with self.assertRaisesRegex(RuntimeError, "missing"):
                trace.cache_input(root, expected)
            (good / "kernel.json").write_text("{}")
            wrong = root / "wrong.ptx"
            wrong.write_text("different PTX")
            ir, metadata = trace.cache_input(root, expected)
            self.assertEqual(ir.parent, good)
            self.assertEqual(metadata, good / "kernel.json")
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), expected)

    def test_dump_sections_cover_before_only_and_pipeline_internal_states(self):
        text = """// -----// IR Dump Before TritonGPUScheduleLoops (tritongpu-schedule-loops) //----- //
module {
  %k = tt.load %ptr {loop.stage = 1 : i32, loop.cluster = 2 : i32}
  %v = ttng.warp_group_dot %k, %q
}
// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
module { %bad = tt.store %p, %x }
// -----// SoftwarePipeliner internal IR Dump After: ExpandLoops
module {
  %x = ttg.async_copy_global_to_local %ptr, %buf
  %v = ttng.warp_group_dot_wait %acc
  tt.store %p, %v
}
"""
        result = trace.sections(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]["events"]), 2)
        self.assertEqual(len(result[1]["events"]), 3)
        self.assertNotIn("%bad", str(result))
        self.assertEqual(trace.sections("no dump"), [])

    def test_AST_contract_explicit_compile_target_without_runtime_launch(self):
        tree = ast.parse(Path(trace.__file__).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertEqual(sum(ast.unparse(n.func) == "triton.compile" for n in calls), 1)
        compile_call = next(n for n in calls if ast.unparse(n.func) == "triton.compile")
        self.assertEqual({k.arg for k in compile_call.keywords}, {"target", "options"})
        forbidden = {
            "run",
            "warmup",
            "_init_handles",
            "cuda",
            "synchronize",
            "empty",
            "empty_strided",
        }
        self.assertFalse(
            any(
                isinstance(n.func, ast.Attribute) and n.func.attr in forbidden
                for n in calls
            )
        )
        self.assertFalse(
            any(
                isinstance(n, ast.Import) and any(v.name == "torch" for v in n.names)
                for n in ast.walk(tree)
            )
        )
        self.assertIn('CUDA_VISIBLE_DEVICES=""', Path(trace.__file__).read_text())

    def test_prepare_refuses_wrong_report_and_changed_production(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "output"
            report.mkdir()
            source = root / "source"
            source.mkdir()
            p = source / "async-pipeline-summary.json"
            p.write_text(
                json.dumps(dict(status="not_promising", git_head=dict(stdout="wrong")))
            )
            with self.assertRaisesRegex(RuntimeError, "captured"):
                trace.prepare(source, report)
            p.write_text(
                json.dumps(
                    dict(
                        status="not_promising",
                        git_head=dict(stdout=trace.CODE),
                        source=dict(production_sha256=trace.PRODUCTION),
                    )
                )
            )
            with patch.object(trace, "sha", return_value="changed"):
                with self.assertRaisesRegex(RuntimeError, "Production"):
                    trace.prepare(source, report)

    def test_prepare_copies_four_exact_cache_inputs_without_mutating_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, report, cache = (root / n for n in ("source", "report", "cache"))
            for p in (source, report, cache):
                p.mkdir()
            production = root / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
            production.parent.mkdir(parents=True)
            production.write_text("synthetic production")
            identity = dict(production_sha256=trace.sha(production))
            trace.save(
                source / "async-pipeline-summary.json",
                dict(
                    status="not_promising",
                    git_head=dict(stdout=trace.CODE),
                    source=identity,
                ),
            )
            for side in ("baseline", "candidate"):
                for ext, data in (
                    ("ptx", side + " PTX"),
                    ("ttir", side + " TTIR"),
                    ("json", "{}"),
                ):
                    (cache / (side + "." + ext)).write_text(data)
            before = {p.name: p.read_bytes() for p in cache.iterdir()}
            for case in trace.CASES:
                trace.save(
                    source / (case + ".json"),
                    dict(
                        source=identity,
                        status="not_promising",
                        runtime=dict(environment=dict(TRITON_CACHE_DIR=str(cache))),
                    ),
                )
                records = {
                    side: [dict(ptx_sha256=trace.sha(cache / (side + ".ptx")))]
                    for side in ("baseline", "candidate")
                }
                trace.save(
                    source / ("compile-" + case + "-test.json"),
                    dict(boundary=False, records=records),
                )
            with patch.object(trace, "ROOT", root), patch.object(
                trace, "PRODUCTION", trace.sha(production)
            ):
                jobs = trace.prepare(source, report)
            self.assertEqual(len(jobs), 4)
            self.assertEqual(before, {p.name: p.read_bytes() for p in cache.iterdir()})
            for job in jobs:
                self.assertEqual(trace.sha(job["ir"]), job["ir_sha256"])
                self.assertTrue(Path(job["directory"], "spec.json").is_file())

    def test_child_compiles_explicit_target_and_detects_replay_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("init.py", "ptxas", "input.ttir"):
                (root / name).write_text(name)
            trace.save(
                root / "input.json",
                dict(target=dict(backend="cuda", arch=90, warp_size=32), num_warps=8),
            )
            job = dict(
                directory=str(root),
                ir=str(root / "input.ttir"),
                metadata=str(root / "input.json"),
                ir_sha256=trace.sha(root / "input.ttir"),
                metadata_sha256=trace.sha(root / "input.json"),
                expected_ptx_sha256=hashlib.sha256(b"original PTX").hexdigest(),
                runtime=dict(
                    environment=dict(TRITON_PTXAS_PATH=str(root / "ptxas")),
                    compiler_files={},
                    triton_init_sha256=trace.sha(root / "init.py"),
                    ptxas=dict(sha256=trace.sha(root / "ptxas")),
                ),
            )
            trace.save(root / "spec.json", job)
            triton = types.ModuleType("triton")
            triton.__file__ = str(root / "init.py")
            result = types.SimpleNamespace(
                asm=dict(ptx="original PTX", ttgir="original IR")
            )
            compiler = types.ModuleType("triton.backends.compiler")
            compiler.GPUTarget = lambda **kwargs: kwargs
            with patch.dict(
                sys.modules, {"triton": triton, "triton.backends.compiler": compiler}
            ), patch.object(sys, "path", list(sys.path)), patch.dict(os.environ):
                with patch.object(
                    triton, "compile", return_value=result, create=True
                ) as call:
                    self.assertEqual(trace.child(root / "spec.json", root), 0)
                    self.assertEqual(
                        call.call_args.kwargs["target"],
                        dict(backend="cuda", arch=90, warp_size=32),
                    )
                    result.asm["ptx"] = "different PTX"
                    self.assertEqual(trace.child(root / "spec.json", root), 2)
                    self.assertEqual(
                        json.loads((root / "result.json").read_text())["status"],
                        "replay_mismatch",
                    )

    def test_cached_JSON_tuple_options_are_hashable_without_value_changes(self):
        metadata = dict(
            cluster_dims=[1, 1, 1],
            supported_fp8_dtypes=["fp8e4b15", "fp8e4nv", "fp8e5"],
            deprecated_fp8_dot_operand_dtypes=["fp8e4b15"],
            allowed_dot_input_precisions=["tf32", "tf32x3", "ieee", "bf16x3", "bf16x6"],
            extern_libs=[
                [
                    "libdevice",
                    "/opt/flagtree/triton/backends/nvidia/lib/libdevice.10.bc",
                ]
            ],
            num_warps=8,
            num_stages=2,
            ptx_version=None,
        )
        before = copy.deepcopy(metadata)
        options_type = dataclasses.make_dataclass(
            "FrozenOptions", [(name, object) for name in metadata], frozen=True
        )
        with self.assertRaises(TypeError):
            hash(options_type(**metadata))
        restored = trace.restore_options(metadata)
        hash(options_type(**restored))
        self.assertEqual(metadata, before)
        self.assertEqual(json.loads(json.dumps(restored)), metadata)
        self.assertEqual(
            restored["extern_libs"],
            (
                (
                    "libdevice",
                    "/opt/flagtree/triton/backends/nvidia/lib/libdevice.10.bc",
                ),
            ),
        )
        self.assertEqual(restored["num_stages"], 2)

    def test_trace_coverage_requires_latencies_schedule_and_both_internal_states(self):
        events = [
            dict(pass_header=name)
            for name in (
                "AssignLatencies",
                "ScheduleLoops",
                "LowerLoops",
                "ExpandLoops",
            )
        ]
        self.assertTrue(all(trace.coverage(events).values()))
        self.assertFalse(all(trace.coverage(events[:-1]).values()))

    def test_runner_uploads_compressed_pass_trace_and_preserves_child_failure(self):
        text = (trace.ROOT / "tools/run_qsa_prefill_pipeline_trace.sh").read_text()
        self.assertIn('"$qsa_report"/*.tar.gz', text)
        self.assertIn('exit "$qsa_status"', text)
        self.assertIn("qsa_status=$?", text)
        self.assertNotIn("with_cuda_compat134", text)


if __name__ == "__main__":
    unittest.main()
