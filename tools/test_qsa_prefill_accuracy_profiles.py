# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Coverage gates for reduced iteration suites and unchanged full suites."""

import ast
import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_accuracy_profiles as profiles
from tools import qsa_prefill_score_resume as resume

ROOT = Path(__file__).resolve().parents[1]


class ProfileTests(unittest.TestCase):
    def test_exact_count_and_selected_functions_exist(self):
        for candidate, count in (
            ("production", 49),
            ("packed-grid", 48),
            ("page-table", 48),
            ("page-table-bf16", 48),
            ("query-split", 46),
            ("query-joint", 46),
            ("query-joint-page", 54),
        ):
            nodes = profiles.expected_nodes(candidate)
            self.assertEqual(len(nodes), count)
            self.assertEqual(len(set(nodes)), count)
            for node in nodes:
                file, name = node.split("::")
                tree = ast.parse((ROOT / file).read_text())
                self.assertIn(
                    name.split("[")[0],
                    {n.name for n in tree.body if isinstance(n, ast.FunctionDef)},
                )
            if candidate == "packed-grid":
                self.assertTrue(all(profiles.PACKED_FILE in n for n in nodes[:8]))
        with self.assertRaises(ValueError):
            profiles.expected_nodes("key-major")

    def test_exact_coverage_rejects_missing_extra_duplicate_or_skipped_cases(self):
        for candidate in ("production", *profiles.CURRENT_CANDIDATES):
            nodes = profiles.expected_nodes(candidate)
            records = {node: dict(result="passed") for node in nodes}
            self.assertIsNone(profiles.validate(records, candidate))
            absolute = {str(ROOT / node): v for node, v in records.items()}
            self.assertIsNone(profiles.validate(absolute, candidate))
            mutations = (
                lambda r: r.pop(nodes[-1]),
                lambda r: r.update({"unexpected::test": dict(result="passed")}),
                lambda r: r.update({str(ROOT / nodes[0]): dict(result="passed")}),
                lambda r: r[nodes[0]].update(result="skipped"),
                lambda r: r[nodes[0]].update(result="failed"),
            )
            for mutate in mutations:
                changed = copy.deepcopy(records)
                mutate(changed)
                self.assertIsNotNone(profiles.validate(changed, candidate))

    def test_full_targets_preserve_current_and_historical_suites(self):
        definitions = {
            "production": [profiles.PRODUCTION_FILE],
            "query-split": [profiles.QUERY_SPLIT_FILE, profiles.PRODUCTION_FILE],
            "query-joint": [profiles.QUERY_JOINT_FILE, profiles.PRODUCTION_FILE],
            "query-joint-page": [profiles.QUERY_PAGE_FILE, profiles.PRODUCTION_FILE],
            "packed-grid": [profiles.PACKED_FILE, profiles.PRODUCTION_FILE],
            "page-table": [profiles.PAGE_TABLE_FILE, profiles.PRODUCTION_FILE],
            "page-table-bf16": [
                profiles.PAGE_TABLE_BF16_FILE,
                profiles.PRODUCTION_FILE,
            ],
            "compact-ragged": [
                profiles.PRODUCTION_FILE,
                "tests/test_qsa_prefill_compact_trial.py",
            ],
            "key-major": [
                profiles.PRODUCTION_FILE,
                "tests/test_qsa_prefill_compact_trial.py",
                "tests/test_qsa_prefill_key_major_trial.py",
            ],
            "key-major-reuse": [
                profiles.PRODUCTION_FILE,
                "tests/test_qsa_prefill_compact_trial.py",
                "tests/test_qsa_prefill_key_major_trial.py",
                "tests/test_qsa_prefill_reuse_trial.py",
            ],
            "key-major-evict-first": [
                profiles.PRODUCTION_FILE,
                "tests/test_qsa_prefill_compact_trial.py",
                "tests/test_qsa_prefill_key_major_trial.py",
                "tests/test_qsa_prefill_reuse_trial.py",
                "tests/test_qsa_prefill_eviction_trial.py",
            ],
        }
        for candidate, files in definitions.items():
            options = types.SimpleNamespace(
                candidate=candidate, accuracy_profile="full"
            )
            self.assertEqual(
                profiles.pytest_targets(options, ROOT), [str(ROOT / p) for p in files]
            )
        self.assertEqual(profiles.name(types.SimpleNamespace()), "full")

    def test_cli_defaults_and_full_override_require_no_gpu_import(self):
        base = [
            sys.executable,
            "-I",
            str(Path(bench.__file__)),
            "--compiler",
            "flagtree",
            "--suite",
            "scored-prefill",
            "--list-accuracy-tests",
        ]
        for candidate, count in (
            ("production", 49),
            ("packed-grid", 48),
            ("page-table", 48),
            ("page-table-bf16", 48),
            ("query-split", 46),
            ("query-joint", 46),
            ("query-joint-page", 54),
        ):
            proc = subprocess.run(
                base + ["--candidate", candidate],
                capture_output=True,
                text=True,
                check=True,
            )
            result = json.loads(proc.stdout)
            self.assertEqual(result["profile"], "iteration")
            self.assertEqual(len(result["targets"]), count)
            proc = subprocess.run(
                base + ["--candidate", candidate, "--accuracy-profile", "full"],
                capture_output=True,
                text=True,
                check=True,
            )
            result = json.loads(proc.stdout)
            self.assertEqual(result["selection"], "full_files")
            self.assertEqual(result["profile"], "full")
        proc = subprocess.run(
            base + ["--candidate", "key-major", "--accuracy-profile", "iteration"],
            capture_output=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        options = types.SimpleNamespace(
            compiler="flagtree",
            candidate="production",
            suite="scored-prefill",
            reference_file="native.py",
            rounds=2,
            kernel_warmup_ms=200,
            kernel_rep_ms=100,
            accuracy_profile="iteration",
        )
        child = bench.child_command(options, Path("/tmp/report"), "--accuracy")
        self.assertEqual(child[child.index("--accuracy-profile") + 1], "iteration")

    def test_iteration_cannot_be_labeled_formal_and_full_retains_scope(self):
        for profile, candidate, eligible in (
            ("iteration", "production", False),
            ("full", "production", True),
            ("full", "packed-grid", False),
        ):
            result = profiles.annotate_acceptance(
                dict(measurement_valid=True, target_met=True),
                profile,
                candidate,
                dict(status="passed"),
            )
            self.assertEqual(result["formal_acceptance_eligible"], eligible)
            self.assertEqual(result["full_suite_validated"], profile == "full")
        result = profiles.annotate_acceptance(
            dict(measurement_valid=False), "full", "production", dict(status="passed")
        )
        self.assertTrue(result["full_suite_validated"])
        self.assertFalse(result["formal_acceptance_eligible"])

    def test_actual_iteration_gate_rejects_incomplete_or_wrong_profile_runtime(self):
        info = {k: "fixed" for k in resume.RUNTIME_KEYS}
        for candidate in ("production", *profiles.CURRENT_CANDIDATES):
            for failure in (None, "missing", "profile"):
                with self.subTest(
                    candidate=candidate, failure=failure
                ), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    records = {
                        n: dict(result="passed")
                        for n in profiles.expected_nodes(candidate)
                    }
                    if failure == "missing":
                        records.pop(next(iter(records)))
                    (root / "accuracy-full.json").write_text(json.dumps(records))
                    (root / "accuracy-runtime.json").write_text(
                        json.dumps(
                            dict(
                                status="passed",
                                runtime=info,
                                accuracy_profile=(
                                    "full" if failure == "profile" else "iteration"
                                ),
                            )
                        )
                    )
                    options = types.SimpleNamespace(
                        compiler="flagtree",
                        candidate=candidate,
                        suite="scored-prefill",
                        reference_file="native.py",
                        rounds=2,
                        kernel_warmup_ms=200,
                        kernel_rep_ms=100,
                        accuracy_timeout=10,
                        accuracy_profile="iteration",
                    )
                    summary = {}
                    with patch.object(
                        bench, "idle_gate", return_value=True
                    ), patch.object(
                        bench.subprocess,
                        "run",
                        return_value=types.SimpleNamespace(returncode=0),
                    ), contextlib.redirect_stdout(
                        io.StringIO()
                    ):
                        if failure:
                            with self.assertRaisesRegex(
                                RuntimeError, "Accuracy gate failed"
                            ):
                                bench.run_accuracy_gate(
                                    options, root, summary, dict(runtime=info), None
                                )
                        else:
                            bench.run_accuracy_gate(
                                options, root, summary, dict(runtime=info), None
                            )
                            self.assertEqual(
                                summary["accuracy"]["counts"],
                                dict(passed=len(profiles.expected_nodes(candidate))),
                            )
                            self.assertFalse(
                                summary["accuracy"]["full_suite_validated"]
                            )


if __name__ == "__main__":
    unittest.main()
