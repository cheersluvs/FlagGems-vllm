# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU gates for the sole loop transform and the frozen joint-stage baseline."""

import ast
import copy
import hashlib
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
from tools import qsa_prefill_cache_granularity as cache
from tools import qsa_prefill_loop_unroll_trial as trial
from tools import qsa_prefill_stage_joint_trial as joint
from tools import test_qsa_prefill_stage_joint as joint_tests

ROOT = Path(__file__).resolve().parents[1]


class UnrollTests(unittest.TestCase):
    def test_exact_kernel_transform_and_source_pins(self):
        old = (ROOT / "tools/qsa_prefill_query_page_kernel.py").read_text()
        new = (ROOT / "tools/qsa_prefill_loop_unroll_kernel.py").read_text()
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(), joint.KERNEL_SHA256)
        self.assertEqual(hashlib.sha256(new.encode()).hexdigest(), trial.KERNEL_SHA256)
        a, b = ast.parse(old), ast.parse(new)
        a.body[0] = copy.deepcopy(b.body[0])  # Module docstring only.
        fn_a, fn_b = [
            next(n for n in t.body if isinstance(n, ast.FunctionDef)) for t in (a, b)
        ]
        fn_a.decorator_list = copy.deepcopy(fn_b.decorator_list)
        loop_a = next(n for n in fn_a.body if isinstance(n, ast.For))
        loop_b = next(n for n in fn_b.body if isinstance(n, ast.For))
        self.assertEqual(ast.unparse(loop_a.iter), "range(first_tile, last_tile)")
        self.assertEqual(
            ast.unparse(loop_b.iter),
            "tl.range(first_tile, last_tile, loop_unroll_factor=2)",
        )
        loop_b.iter = copy.deepcopy(loop_a.iter)
        self.assertEqual(ast.dump(a), ast.dump(b))
        # Unroll factor is fixed; no tunable/contract parameter was added.
        self.assertEqual(ast.dump(fn_a.args), ast.dump(fn_b.args))

    def test_factory_new_jit_same_host_choices_topk_and_isolated_entry(self):
        old, choices = joint_tests.StageJointTests().configs()
        old_jit, new_jit = object(), object()
        template = types.SimpleNamespace(configs=old, keys=["rows"], fn=old_jit)
        production = types.SimpleNamespace(
            _qsa_prefill_score_kernel=joint_tests.Entry(template),
            persistent_topk=object(),
        )
        module = types.ModuleType("combined")
        module.__file__ = str(ROOT / "tools/qsa_prefill_query_page_kernel.py")
        module.production_baseline = production
        module.persistent_topk = production.persistent_topk
        module._qsa_prefill_score_kernel = joint_tests.Entry(template)
        module._prefill_logits = object()
        module.qsa_select_paged_prefill = object()
        module._qsa_prefill_compact_score_kernel = object()
        host, compact = module._prefill_logits, module._qsa_prefill_compact_score_kernel
        kernel = types.ModuleType("tools.qsa_prefill_loop_unroll_kernel")
        kernel.__file__ = str(ROOT / "tools/qsa_prefill_loop_unroll_kernel.py")
        kernel._qsa_prefill_score_kernel = joint_tests.Entry(
            types.SimpleNamespace(fn=new_jit)
        )
        pkg = types.ModuleType("flaggems_vllm")
        pkg.runtime = types.SimpleNamespace(get_tuned_config=lambda name: choices)
        utils = types.ModuleType("flaggems_vllm.utils")
        utils.libtuner = lambda **kw: lambda fn: types.SimpleNamespace(
            fn=fn, configs=kw["configs"], keys=kw["key"]
        )
        with patch.dict(
            sys.modules,
            {
                "flaggems_vllm": pkg,
                "flaggems_vllm.utils": utils,
                "tools.qsa_prefill_loop_unroll_kernel": kernel,
            },
        ), patch.object(trial.combined, "candidate_module", return_value=module):
            candidate = trial.candidate_module(production)
        self.assertIs(candidate._qsa_prefill_score_kernel.fn.fn, new_jit)
        self.assertIs(candidate._prefill_logits, host)
        self.assertIs(candidate._qsa_prefill_compact_score_kernel, compact)
        self.assertIs(candidate.persistent_topk, production.persistent_topk)
        self.assertEqual(candidate._qsa_prefill_score_kernel.fn.configs, choices)
        self.assertTrue(
            all(
                a is not b
                for a, b in zip(
                    candidate._qsa_prefill_score_kernel.kernel_cache,
                    production._qsa_prefill_score_kernel.kernel_cache,
                )
            )
        )
        info = trial.metadata(candidate, [dict(constexprs=choices[17].all_kwargs())])
        self.assertEqual(info["loop_unroll_factor"], 2)
        self.assertTrue(info["same_topk"])
        with self.assertRaises(RuntimeError):
            trial.metadata(candidate, [dict(constexprs=dict(num_stages=3))])

    def test_previous_factory_is_exact_joint_stage(self):
        canonical = object()
        module = types.SimpleNamespace(production_baseline=canonical)
        with patch.object(joint, "candidate_module", return_value=canonical) as factory:
            self.assertIs(cache.previous_module(module, "stage-joint"), canonical)
            factory.assert_called_once_with(canonical)
        self.assertEqual(
            cache.previous_identity("stage-joint"),
            dict(
                source_equivalent_sha256=joint.KERNEL_SHA256,
                config_group=joint.CONFIG_GROUP,
                baseline_kind=trial.BASELINE_KIND,
            ),
        )

    def summary(self):
        s = joint_tests.StageJointTests().summary()
        s["manifest"]["options"].update(
            candidate="query-loop-unroll", previous_tuning="stage-joint"
        )
        for c in s["cases"]:
            m = c.pop("stage_joint")
            m.update(kernel_sha256=trial.KERNEL_SHA256, loop_unroll_factor=2)
            c["loop_unroll"] = m
            c["previous_tuning"].update(cache.previous_identity("stage-joint"))
            for name in ("score_vs_previous_tuning", "pipeline_vs_previous_tuning"):
                c["stages"][name]["baseline_kind"] = trial.BASELINE_KIND
        return s

    def test_acceptance_rejects_wrong_baseline_and_changed_transform(self):
        s = self.summary()
        result = trial.acceptance(s)
        self.assertTrue(result["measurement_valid"], result)
        self.assertEqual(result["candidate"], "query-loop-unroll")
        self.assertFalse(result["formal_production_acceptance"])
        for field, value in [
            ("loop_unroll_factor", 1),
            ("kernel_sha256", joint.KERNEL_SHA256),
            ("same_topk", False),
        ]:
            bad = copy.deepcopy(s)
            bad["cases"][0]["loop_unroll"][field] = value
            self.assertFalse(trial.acceptance(bad)["measurement_valid"])
        for field, value in [
            ("baseline_kind", joint.BASELINE_KIND),
            ("config_group", "wrong"),
            ("source_equivalent_sha256", trial.KERNEL_SHA256),
        ]:
            bad = copy.deepcopy(s)
            bad["cases"][0]["previous_tuning"][field] = value
            self.assertFalse(trial.acceptance(bad)["measurement_valid"])
        bad = copy.deepcopy(s)
        bad["manifest"]["options"]["previous_tuning"] = "combined"
        self.assertFalse(trial.acceptance(bad)["measurement_valid"])

    def test_report_table_preserves_stage_data_and_private_acceptance(self):
        summary = self.summary()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            bench.write_table(summary, target)
            table = (target / "triton-stage-ab.tsv").read_text()
            self.assertIn("gems_loop_unroll_trial_us", table.splitlines()[0])
            self.assertIn("\ttopk\t", table)
            self.assertIn("\tpipeline\t", table)
            result = json.loads((target / "overall-acceptance.json").read_text())
            self.assertEqual(result["candidate"], "query-loop-unroll")
            self.assertTrue(result["measurement_valid"])
            self.assertFalse(result["formal_production_acceptance"])

    def test_iteration_full_coverage_and_gpu_forced_paths(self):
        nodes = profiles.expected_nodes("query-loop-unroll")
        self.assertEqual(len(nodes), 62)
        records = {n: dict(result="passed") for n in nodes}
        self.assertIsNone(profiles.validate(records, "query-loop-unroll"))
        selected = {k: v for k, v in records.items() if profiles.LOOP_UNROLL_FILE in k}
        for family, count in [
            ("test_combined_compact_every_config", 6),
            ("test_combined_full_entry", 12),
        ]:
            selected.update(
                {
                    profiles.LOOP_UNROLL_FILE
                    + "::"
                    + family
                    + f"[{i}]": dict(result="passed")
                    for i in range(count)
                }
            )
        self.assertIsNone(trial.validate_full_coverage(selected))
        selected.pop(next(iter(selected)))
        self.assertIsNotNone(trial.validate_full_coverage(selected))
        old = (ROOT / profiles.STAGE_JOINT_FILE).read_text()
        new = (ROOT / profiles.LOOP_UNROLL_FILE).read_text()
        a = ast.parse(
            old.replace(
                "qsa_prefill_stage_joint_trial", "qsa_prefill_loop_unroll_trial"
            )
        )
        b = ast.parse(new)
        a.body[0] = copy.deepcopy(b.body[0])
        self.assertEqual(ast.dump(a), ast.dump(b))
        self.assertEqual(new.count("itertools.product((2, 1), modes)"), 3)
        self.assertIn("configs=configs", new)

    def test_cli_runner_archive_and_baseline_restrictions(self):
        command = [
            sys.executable,
            str(Path(bench.__file__)),
            "--compiler",
            "flagtree",
            "--candidate",
            "query-loop-unroll",
            "--suite",
            "scored-prefill",
            "--compare-previous-tuning",
            "--previous-tuning",
            "stage-joint",
        ]
        for flag, count in [("--list-cases", 30), ("--list-accuracy-tests", 62)]:
            p = subprocess.run(command + [flag], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            data = json.loads(p.stdout)
            self.assertEqual(
                len(data if flag == "--list-cases" else data["targets"]), count
            )
        bad = list(command)
        bad[-1] = "combined"
        self.assertEqual(
            subprocess.run(bad + ["--list-cases"], capture_output=True).returncode, 2
        )
        runner = (ROOT / "tools/run_qsa_prefill_loop_unroll_h800.sh").read_text()
        self.assertIn("--rounds 2", runner)
        self.assertIn("--previous-tuning stage-joint", runner)
        self.assertNotIn("--cases", runner)
        for path in [
            "tools/qsa_prefill_loop_unroll_trial.py",
            "tools/qsa_prefill_loop_unroll_kernel.py",
            profiles.LOOP_UNROLL_FILE,
            "tools/test_qsa_prefill_loop_unroll.py",
        ]:
            self.assertIn(path, bench.SOURCES)


if __name__ == "__main__":
    unittest.main()
