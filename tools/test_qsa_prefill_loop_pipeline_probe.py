# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Check single-hint AST scope, real IR motion gates and guarded probe plumbing."""

import ast
import copy
import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_loop_pipeline_probe as probe


def ir(early):
    copy_line = "      %next = ttg.async_copy_global_to_local %ptr, %buf"
    dot_line = "      %score = ttng.warp_group_dot %a, %b"
    body = [copy_line, dot_line] if early else [dot_line, copy_line]
    return "\n".join(
        [
            "module {",
            "  %initial = ttg.async_copy_global_to_local %ptr, %buf",
            "  %result = scf.for %i = %a to %b step %one {",
            *body,
            "    scf.yield %v",
            "  } loc(#loc1)",
            "}",
        ]
    )


PTX = "wgmma.mma_async.sync.aligned x; wgmma.wait_group.sync.aligned 0;"


class LoopPipelineTests(unittest.TestCase):
    def test_only_loop_iterator_changes(self):
        probe.verify_source()

        def fn(path, name):
            return next(
                n
                for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )

        before = fn(
            probe.ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py",
            "_qsa_prefill_score_kernel",
        )
        after = fn(probe.KERNEL, "_loop_pipeline_score_kernel")
        a = next(n for n in after.body if isinstance(n, ast.For))
        b = next(n for n in before.body if isinstance(n, ast.For))
        self.assertEqual(
            ast.unparse(a.iter), "tl.range(first_tile, last_tile, num_stages=2)"
        )
        a.iter = b.iter
        after.name = before.name
        after.decorator_list = before.decorator_list
        self.assertEqual(ast.dump(after), ast.dump(before))

    def test_ir_motion_excludes_prologue_and_post_dot(self):
        a, b = probe.prefetch(ir(False), PTX), probe.prefetch(ir(True), PTX)
        self.assertEqual(a["copies_before_first_dot"], 0)
        self.assertEqual(b["copies_before_first_dot"], 1)
        self.assertEqual(a["loop_copy_sites"], 1)
        self.assertEqual(a["mma_sites"], 1)
        self.assertIsNone(probe.prefetch("", PTX)["copies_before_first_dot"])
        self.assertEqual(
            probe.prefetch(ir(True) + "\n" + ir(True), PTX)["main_loops"], 2
        )

    def test_gate_rejects_no_motion_unknown_resources_and_spills(self):
        a = dict(
            spill_count=0,
            shared_bytes=106496,
            registers_per_thread=207,
            prefetch=probe.prefetch(ir(False), PTX),
        )
        b = dict(a, prefetch=probe.prefetch(ir(True), PTX))
        self.assertIsNone(probe.gate_reason(a, b))
        self.assertIn("no additional", probe.gate_reason(a, a))
        self.assertIn("spills", probe.gate_reason(a, dict(b, spill_count=1)))
        self.assertIn("spills", probe.gate_reason(a, dict(b, spill_count=None)))
        self.assertIn(
            "unknown", probe.gate_reason(a, dict(b, registers_per_thread=None))
        )
        self.assertIn(
            "larger shared", probe.gate_reason(a, dict(b, shared_bytes=114688))
        )
        self.assertIn(
            "unidentified",
            probe.gate_reason(a, dict(b, prefetch=probe.prefetch("", PTX))),
        )
        self.assertIn(
            "matrix work",
            probe.gate_reason(a, dict(b, prefetch=dict(b["prefetch"], mma_sites=2))),
        )

    def test_scope_routes_new_child_driver_and_restores_harness(self):
        old_file = probe.harness.__file__
        old_eligible = probe.harness.eligible
        with probe.scoped():
            self.assertIs(probe.base.make_pair, probe.make_pair)
            self.assertIs(probe.base.freeze_records, probe.freeze_records)
            self.assertEqual(probe.base.__file__, probe.__file__)
            self.assertIs(probe.harness.eligible, probe.eligible)
            self.assertTrue(
                probe.eligible(dict(IS_FP8=False, NUM_HEADS=4, HEAD_DIM=128, TILE_R=32))
            )
            self.assertTrue(
                probe.eligible(dict(IS_FP8=False, NUM_HEADS=4, HEAD_DIM=128, TILE_R=64))
            )
            self.assertFalse(
                probe.eligible(dict(IS_FP8=True, NUM_HEADS=4, HEAD_DIM=128, TILE_R=64))
            )
        self.assertEqual(probe.harness.__file__, old_file)
        self.assertIs(probe.harness.eligible, old_eligible)

    def test_negative_compile_saves_artifacts_and_does_not_time(self):
        key = (17, 0, 1088)
        kernel = types.SimpleNamespace(asm=dict(ptx=PTX, ttgir=ir(False)))
        entry = types.SimpleNamespace(
            jit=types.SimpleNamespace(decisions={key: True}),
            compiled={key: (kernel, {})},
        )
        empty = types.SimpleNamespace(compiled={})
        module = types.SimpleNamespace(
            _qsa_prefill_score_kernel=entry,
            _qsa_prefill_compact_score_kernel=empty,
            _qsa_prefill_fp8_store_score_kernel=empty,
        )
        record = dict(
            chunk=list(key),
            route="rectangular",
            constants={},
            spill_count=0,
            registers_per_thread=63,
            shared_bytes=73728,
        )
        with tempfile.TemporaryDirectory() as folder, patch.object(
            probe.harness, "RESOURCE_DIR", Path(folder)
        ), patch.object(probe.harness, "RESOURCE_PREFIX", "unit"), patch.object(
            probe.stage,
            "FREEZE",
            return_value=dict(
                baseline=[copy.deepcopy(record)], candidate=[copy.deepcopy(record)]
            ),
        ), patch.object(
            probe.base, "artifact_records", return_value=[]
        ) as export, patch.object(
            probe.harness, "IN_BOUNDARY", False
        ), patch.object(
            probe.harness, "LAST_REJECTION", None
        ):
            with self.assertRaisesRegex(RuntimeError, "NOT_PROMISING"):
                probe.freeze_records((module, module))
            export.assert_called_once()
            self.assertEqual(len(list(Path(folder).glob("compile-unit-*.json"))), 1)
            self.assertEqual(len(probe.harness.LAST_REJECTION), 1)

    def test_worker_classifies_compile_rejection_but_preserves_numeric_error(self):
        with tempfile.TemporaryDirectory() as folder:
            options = types.SimpleNamespace(
                report_dir=folder, worker_case="synthetic", rounds=2
            )
            path = Path(folder) / "synthetic.json"

            def reject(options):
                probe.harness.LAST_REJECTION = [{"reason": "no early copy"}]
                path.write_text(
                    json.dumps(dict(status="failed", stages={}, error="compile gate"))
                )
                return 1

            with patch.object(probe.base, "worker", side_effect=reject), patch.object(
                probe, "show"
            ):
                self.assertEqual(probe.worker(options), 0)
                self.assertEqual(
                    json.loads(path.read_text())["status"], "not_promising"
                )

            def numeric_error(options):
                path.write_text(
                    json.dumps(dict(status="failed", stages={}, error="wrong score"))
                )
                return 1

            with patch.object(
                probe.base, "worker", side_effect=numeric_error
            ), patch.object(probe, "show"):
                self.assertEqual(probe.worker(options), 1)
                self.assertEqual(json.loads(path.read_text())["error"], "wrong score")

    def test_coordinator_continues_after_one_configuration_is_rejected(self):
        from tools import benchmark_qsa_prefill_triton_stages as bench
        from tools import qsa_prefill_flagtree_ab as profile
        from tools import qsa_prefill_profiling as profiling

        data = probe.manifest()
        cases = [
            next(c for c in data["cases"] if c["case"]["name"] == n)
            for n in probe.DEFAULT_CASES
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prior = root / "prior"
            report = root / "new"
            prior.mkdir()
            report.mkdir()
            asm = root / "ptxas"
            asm.write_text("synthetic assembler")
            (prior / "stage-summary.json").write_text(
                json.dumps(
                    dict(
                        status="passed",
                        manifest=dict(
                            git_head=dict(stdout=data["baseline_code_commit"])
                        ),
                    )
                )
            )
            (prior / "flagtree134-tools.json").write_text(
                json.dumps(dict(toolchain=dict(ptxas=dict(path=str(asm)))))
            )
            reference = "synthetic native reference"
            (prior / "sources.json").write_text(
                json.dumps({"pinned_vllm/qsa_indexer.py": dict(text=reference)})
            )
            for c in cases:
                (prior / (c["case"]["name"] + ".json")).write_text(
                    json.dumps(
                        dict(
                            status="passed",
                            case=c["case"],
                            inputs=c["inputs"],
                            runtime={},
                        )
                    )
                )
            options = types.SimpleNamespace(
                report_dir=str(report),
                baseline_report=str(prior),
                cases=list(probe.DEFAULT_CASES),
                timeout=1,
            )
            calls = []

            def run(command, **kwargs):
                name = command[-1]
                calls.append(name)
                if len(calls) == 1:
                    result = dict(
                        status="not_promising",
                        stages={},
                        rejections=[dict(reason="no motion")],
                    )
                else:
                    result = dict(
                        status="passed",
                        stages={
                            k: dict(measurement_validated=True, speedup=1.0)
                            for k in ("score", "topk", "pipeline")
                        },
                        self_controls={str(i): {} for i in range(4)},
                    )
                (report / (name + ".json")).write_text(json.dumps(result))
                return 0

            with probe.scoped(), patch.object(
                profile, "PTXAS_SHA256", hashlib.sha256(asm.read_bytes()).hexdigest()
            ), patch.object(
                bench.ab,
                "REFERENCE_SHA256",
                hashlib.sha256(reference.encode()).hexdigest(),
            ), patch.object(
                probe.base, "compiler_identity", return_value=data["compiler_identity"]
            ), patch.object(
                profile, "select_gpu", return_value=True
            ), patch.object(
                bench, "idle_gate", return_value=True
            ), patch.object(
                bench, "child_environment", return_value={}
            ), patch.object(
                profiling, "run_process", side_effect=run
            ):
                self.assertEqual(probe.coordinate(options), 0)
            summary = json.loads((report / "loop-pipeline-summary.json").read_text())
            self.assertEqual(calls, list(probe.DEFAULT_CASES))
            self.assertEqual(summary["status"], "completed_with_rejections")
            self.assertEqual(summary["measured_cases"], 1)
            self.assertEqual(summary["aggregate"]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
