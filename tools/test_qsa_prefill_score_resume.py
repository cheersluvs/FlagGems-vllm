# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU-only checkpoint integrity checks; no GPU result is fabricated as current."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "resume", Path(__file__).with_name("qsa_prefill_score_resume.py")
)
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)
UUID = "GPU-2d17872f-4327-2beb-8049-604ecf6a3d66"


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "checkpoint.json"
        self.options = types.SimpleNamespace(
            rounds=7, kernel_warmup_ms=1000, kernel_rep_ms=100
        )
        self.cases = [
            dict(name="score", kind="single"),
            dict(name="zero_visible", kind="zero"),
            dict(name="pending", kind="single"),
        ]
        hashes = {}
        for name in [
            "tools/qsa_prefill_score_benchmark.py",
            "benchmark/base.py",
            "src/flaggems_vllm/ops/qsa_select_paged_prefill.py",
            "tools/qsa_prefill_gpu_guard.py",
        ]:
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("frozen")
            hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        runtime = {key: "same" for key in resume.RUNTIME_KEYS}
        runtime["uuid"] = UUID.removeprefix("GPU-")
        self.timer = dict(
            mode="kernel",
            unit="us",
            warmup_ms=1000,
            rep_ms=100,
            return_mode="median",
            cache_policy="default",
            includes_topk=False,
            sources=dict(do_bench=dict(path="runtime/testing.py", sha256="a" * 64)),
        )
        self.good = dict(
            case=self.cases[0],
            status="passed",
            runtime=runtime,
            score_timer=self.timer,
            performance_guard=dict(status="observed_clean"),
            telemetry=[dict(gpu_guard=dict(status="passed")) for _ in range(8)],
            score_kernel=dict(
                status="passed",
                baseline_us=dict(samples=[10] * 7, median=10),
                candidate_us=dict(samples=[8] * 7, median=8),
                speedup=1.25,
                paired_ratios=dict(samples=[1.25] * 7, median=1.25),
            ),
        )
        self.saved = dict(
            manifest=dict(
                options=dict(
                    mode="kernel",
                    single_row_ab=True,
                    require_idle_gpu=True,
                    **vars(self.options),
                ),
                source_hashes=hashes,
                git_head=dict(stdout="code\n"),
            ),
            accuracy=dict(status="passed", counts=dict(passed=161, skipped=1)),
            cases=[
                copy.deepcopy(self.good),
                dict(
                    case=self.cases[1],
                    status="passed",
                    runtime=runtime,
                    score_kernel=dict(status="not_applicable_no_score"),
                ),
                dict(
                    case=self.cases[2],
                    status="failed",
                    score_kernel=dict(status="passed"),
                ),
            ],
        )

    def load(self):
        self.path.write_text(json.dumps(self.saved))
        return resume.load(self.path, self.root, self.cases, self.options, UUID)

    def test_retains_only_completed_cases_and_tags_source(self):
        metadata, retained = self.load()
        self.assertEqual(metadata["completed_names"], ["score", "zero_visible"])
        self.assertEqual(len(retained), 2)
        self.assertEqual(retained[0]["score_kernel"], self.good["score_kernel"])
        self.assertEqual(retained[0]["measurement_origin"]["code_commit"], "code")
        self.assertEqual(metadata["prior_accuracy"]["counts"]["passed"], 161)

    def test_gpu_change_and_measurement_parameter_changes_reject(self):
        self.load()
        with self.assertRaisesRegex(ValueError, "same GPU"):
            resume.load(self.path, self.root, self.cases, self.options, "other")
        self.options.rounds = 3
        with self.assertRaisesRegex(ValueError, "parameter"):
            self.load()

    def test_frozen_source_change_rejects_but_diagnostic_source_can_change(self):
        (self.root / "tools/qsa_prefill_gpu_guard.py").write_text("new diagnostics")
        self.load()
        (self.root / "benchmark/base.py").write_text("different timing")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.load()

    def test_incomplete_or_rejected_telemetry_cannot_be_retained(self):
        self.saved["cases"][0]["telemetry"] = []
        with self.assertRaisesRegex(ValueError, "telemetry"):
            self.load()
        self.saved["cases"][0] = copy.deepcopy(self.good)
        self.saved["cases"][0]["performance_guard"]["status"] = "rejected"
        with self.assertRaisesRegex(ValueError, "guard"):
            self.load()

    def test_corrupt_latency_or_scope_cannot_be_retained(self):
        for change in ["samples", "median", "speedup", "topk"]:
            self.saved["cases"][0] = copy.deepcopy(self.good)
            case = self.saved["cases"][0]
            if change == "samples":
                case["score_kernel"]["baseline_us"]["samples"] = [10] * 6
            elif change == "median":
                case["score_kernel"]["candidate_us"]["median"] = 5
            elif change == "speedup":
                case["score_kernel"]["speedup"] = 2
            else:
                case["score_timer"]["includes_topk"] = True
            with self.assertRaises(ValueError):
                self.load()

    def test_failed_correctness_or_changed_case_cannot_be_retained(self):
        self.saved["accuracy"]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "correctness"):
            self.load()
        self.saved["accuracy"]["status"] = "passed"
        self.cases[0]["kind"] = "different"
        # Break the synthetic fixture's shared dictionary before checking.
        self.saved["cases"][0]["case"] = dict(name="score", kind="single")
        with self.assertRaisesRegex(ValueError, "changed checkpoint case"):
            self.load()

    def test_chained_resume_preserves_prior_results_and_original_provenance(self):
        metadata, retained = self.load()
        new = copy.deepcopy(self.good)
        new["case"] = self.cases[2]
        self.saved["retained_cases"] = retained
        self.saved["retained_accuracy"] = self.saved["accuracy"]
        self.saved["accuracy"] = dict(status="not_run_explicit_request")
        self.saved["cases"] = [new]
        metadata2, retained2 = self.load()
        self.assertEqual(len(retained2), 3)
        self.assertEqual(
            retained2[0]["measurement_origin"], retained[0]["measurement_origin"]
        )
        self.assertEqual(metadata2["prior_accuracy"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
