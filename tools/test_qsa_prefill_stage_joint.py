# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU gates for paired stage2/stage1 choices and unchanged execution semantics."""

import ast
import copy
import hashlib
import json
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_accuracy_profiles as profiles
from tools import qsa_prefill_cache_granularity as cache
from tools import qsa_prefill_stage_joint_trial as trial
from tools import test_qsa_prefill_query_page as combined_tests

ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = ROOT / "src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml"


class Config:
    def __init__(self, entry):
        self.values = dict(
            entry["META"],
            num_warps=entry["num_warps"],
            num_stages=entry["num_stages"],
            num_ctas=entry.get("num_ctas", 1),
        )

    def all_kwargs(self):
        return dict(self.values)


class Entry:
    def __init__(self, fn):
        self.fn = fn
        self.kernel_cache = [{}, {}]


class StageJointTests(unittest.TestCase):
    def configs(self):
        y = yaml.safe_load(YAML_PATH.read_text())
        return tuple(
            [Config(c) for c in y[key]]
            for key in (
                "qsa_select_paged_prefill_query_joint_score",
                trial.CONFIG_GROUP,
            )
        )

    def test_only_new_yaml_group_changes_and_exact_paired_depths(self):
        current = yaml.safe_load(YAML_PATH.read_text())
        current.pop(trial.CONFIG_GROUP)
        current.pop("qsa_select_paged_prefill_fp8_store_score")
        # Pin all pre-existing groups without requiring history in shallow GPU clones.
        self.assertEqual(
            hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest(),
            "84e13917c76da70bffc68f30de9aaa3d803e2a4ddbbe7845152fdad2250b9adb",
        )
        a, b = self.configs()
        trial.validate_configs(a, b)
        for index in range(34):
            wrong = copy.deepcopy(b)
            wrong[index].values["K_TILES"] += 1
            with self.assertRaises(RuntimeError):
                trial.validate_configs(a, wrong)
        with self.assertRaises(RuntimeError):
            trial.validate_configs(a, b[:-1])
        with self.assertRaises(RuntimeError):
            trial.validate_configs(a, a)

    def test_dtype_pruning_preserves_both_depths_and_query_mode_rules(self):
        from tools.qsa_prefill_query_joint_policy import prune_configs

        _, choices = self.configs()
        configs = [
            types.SimpleNamespace(
                kwargs={
                    k: v for k, v in c.all_kwargs().items() if not k.startswith("num_")
                },
                num_stages=c.all_kwargs()["num_stages"],
            )
            for c in choices
        ]
        for fp8, count in ((False, 20), (True, 34)):
            active = prune_configs(configs, dict(IS_FP8=fp8))
            self.assertEqual(len(active), count)
            self.assertEqual(
                [sum(c.num_stages == stage for c in active) for stage in (2, 1)],
                [count // 2] * 2,
            )
            self.assertTrue(
                all(
                    c.kwargs["QUERY_PARTS"] == 1 or (fp8 and c.kwargs["TILE_R"] >= 32)
                    for c in active
                )
            )

    def test_factory_reuses_exact_combined_jit_and_host_with_independent_entry(self):
        old, new = self.configs()
        jit = object()
        template = types.SimpleNamespace(configs=old, keys=["rows"], fn=jit)
        production = types.SimpleNamespace(
            persistent_topk=object(), _qsa_prefill_score_kernel=Entry(template)
        )
        module = types.ModuleType("combined")
        module.__file__ = str(ROOT / "tools/qsa_prefill_query_page_kernel.py")
        module.production_baseline = production
        module.persistent_topk = production.persistent_topk
        module._qsa_prefill_score_kernel = Entry(template)
        module._qsa_prefill_compact_score_kernel = object()
        module._prefill_logits = object()
        module.qsa_select_paged_prefill = object()
        compact, host = module._qsa_prefill_compact_score_kernel, module._prefill_logits
        original_entry = module._qsa_prefill_score_kernel

        def libtuner(**kwargs):
            def decorate(fn):
                return types.SimpleNamespace(
                    fn=fn,
                    configs=kwargs["configs"],
                    keys=kwargs["key"],
                    prune=kwargs["prune_configs_by"],
                )

            return decorate

        package = types.ModuleType("flaggems_vllm")
        package.runtime = types.SimpleNamespace(get_tuned_config=lambda name: new)
        utils = types.ModuleType("flaggems_vllm.utils")
        utils.libtuner = libtuner
        with patch.dict(
            sys.modules, {"flaggems_vllm": package, "flaggems_vllm.utils": utils}
        ):
            with patch.object(trial.combined, "candidate_module", return_value=module):
                candidate = trial.candidate_module(production)
        e = candidate._qsa_prefill_score_kernel
        self.assertIs(e.fn.fn, jit)
        self.assertEqual(e.fn.keys, template.keys)
        self.assertIs(candidate._prefill_logits, host)
        self.assertIs(candidate._qsa_prefill_compact_score_kernel, compact)
        self.assertIs(candidate.persistent_topk, production.persistent_topk)
        for other in (production._qsa_prefill_score_kernel, original_entry):
            self.assertIsNot(e, other)
            self.assertTrue(
                all(a is not b for a, b in zip(e.kernel_cache, other.kernel_cache))
            )
        m = trial.metadata(candidate, [])
        self.assertEqual(m["kernel_sha256"], trial.KERNEL_SHA256)
        self.assertEqual(m["num_stages_choices"], [2, 1])
        valid = [dict(constexprs=new[i].all_kwargs()) for i in (0, 17)]
        self.assertEqual(trial.metadata(candidate, valid)["configs"], valid)
        self.assertTrue(m["same_topk"])
        with self.assertRaises(RuntimeError):
            trial.metadata(candidate, [dict(constexprs=dict(num_stages=3))])
        candidate.stage_joint_choices.pop()
        with self.assertRaises(RuntimeError):
            trial.metadata(candidate, [])

    def test_unchanged_math_source_and_previous_baseline_factory(self):
        kernel = ROOT / "tools/qsa_prefill_query_page_kernel.py"
        production = ROOT / "tools/qsa_prefill_pre_store_production.py"
        self.assertEqual(
            hashlib.sha256(kernel.read_bytes()).hexdigest(), trial.KERNEL_SHA256
        )
        self.assertEqual(
            hashlib.sha256(production.read_bytes()).hexdigest(), trial.PRODUCTION_SHA256
        )
        canonical = object()
        module = types.SimpleNamespace(production_baseline=canonical)
        with patch.object(
            trial.combined,
            "candidate_module",
            return_value=types.SimpleNamespace(__file__=str(kernel)),
        ) as factory:
            self.assertEqual(
                cache.previous_module(module, "combined").__file__, str(kernel)
            )
            factory.assert_called_once_with(canonical)
        self.assertEqual(
            cache.previous_identity("combined")["baseline_kind"], trial.BASELINE_KIND
        )

    def summary(self):
        s = combined_tests.CombinedTests.summary()
        s["manifest"]["options"].update(
            candidate="query-stage-joint",
            compare_previous_tuning=True,
            previous_tuning="combined",
        )
        _, choices = self.configs()
        for index, c in enumerate(s["cases"]):
            selected = choices[7 + (17 if index % 2 else 0)].all_kwargs()
            c["production_score_configs"] = [dict(constexprs=copy.deepcopy(selected))]
            c["query_page"]["configs"] = copy.deepcopy(c["production_score_configs"])
            for t in c.get("cache_granularity", {}).get("production", []):
                t["constants"] = copy.deepcopy(selected)
            c["stage_joint"] = dict(
                config_group=trial.CONFIG_GROUP,
                kernel_sha256=trial.KERNEL_SHA256,
                num_stages_choices=[2, 1],
                same_topk=True,
                configs=copy.deepcopy(c["production_score_configs"]),
                choices=[x.all_kwargs() for x in choices],
            )
            c["previous_tuning"] = dict(
                cache.previous_identity("combined"), same_topk=True
            )
            for stage in ("score", "pipeline"):
                entry = copy.deepcopy(c["stages"][stage])
                entry.update(
                    baseline_kind=trial.BASELINE_KIND, includes_topk=stage == "pipeline"
                )
                c["stages"][stage + "_vs_previous_tuning"] = entry
        return s

    def test_acceptance_requires_measured_baseline_and_valid_paired_stage_space(self):
        s = self.summary()
        snapshot = copy.deepcopy(s)
        result = trial.acceptance(s)
        self.assertEqual(s, snapshot)
        self.assertTrue(result["measurement_valid"] and result["target_met"])
        self.assertEqual(result["candidate"], "query-stage-joint")
        self.assertFalse(result["formal_production_acceptance"])
        for key in (
            "num_stages_choices",
            "choices",
            "same_topk",
            "configs",
            "kernel_sha256",
            "config_group",
        ):
            bad = copy.deepcopy(s)
            bad["cases"][0]["stage_joint"].pop(key)
            self.assertFalse(trial.acceptance(bad)["measurement_valid"])
        for name in ("score_vs_previous_tuning", "pipeline_vs_previous_tuning"):
            bad = copy.deepcopy(s)
            bad["cases"][0]["stages"].pop(name)
            self.assertFalse(trial.acceptance(bad)["measurement_valid"])
        bad = copy.deepcopy(s)
        bad["cases"][0]["stage_joint"]["choices"][0]["num_stages"] = 3
        self.assertFalse(trial.acceptance(bad)["measurement_valid"])
        bad = copy.deepcopy(s)
        bad["manifest"]["options"]["compare_previous_tuning"] = False
        self.assertFalse(trial.acceptance(bad)["measurement_valid"])

    def test_iteration_and_full_coverage_match_rebound_gpu_nodes(self):
        nodes = profiles.expected_nodes("query-stage-joint")
        self.assertEqual(len(nodes), 62)
        self.assertEqual(sum(profiles.STAGE_JOINT_FILE in n for n in nodes), 22)
        records = {n: dict(result="passed") for n in nodes}
        self.assertIsNone(profiles.validate(records, "query-stage-joint"))
        every = next(n for n in nodes if "every_config" in n)
        for status in ("skipped", "failed"):
            bad = copy.deepcopy(records)
            bad[every]["result"] = status
            self.assertIsNotNone(profiles.validate(bad, "query-stage-joint"))
        prefix = profiles.STAGE_JOINT_FILE + "::"
        full = {k: v for k, v in records.items() if k.startswith(prefix)}
        for family, count in (
            ("test_combined_compact_every_config", 6),
            ("test_combined_full_entry", 12),
        ):
            full.update(
                {
                    prefix + family + f"[{i}]": dict(result="passed")
                    for i in range(count)
                }
            )
        self.assertIsNone(trial.validate_full_coverage(full))
        full.pop(every)
        self.assertIsNotNone(trial.validate_full_coverage(full))
        tree = ast.parse((ROOT / profiles.STAGE_JOINT_FILE).read_text())
        forced = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and ast.unparse(n).startswith("kw['num_stages']")
        ]
        self.assertEqual(len(forced), 2)
        self.assertTrue(
            all(
                isinstance(n.value, ast.Name) and n.value.id == "stages" for n in forced
            )
        )
        products = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "product"
        ]
        self.assertEqual(len(products), 3)
        self.assertTrue(all(ast.literal_eval(n.args[0]) == (2, 1) for n in products))

    def test_cli_sources_and_runner_preserve_frozen_set_and_two_rounds(self):
        command = [
            sys.executable,
            str(Path(bench.__file__)),
            "--compiler",
            "flagtree",
            "--candidate",
            "query-stage-joint",
            "--suite",
            "scored-prefill",
            "--compare-previous-tuning",
            "--previous-tuning",
            "combined",
        ]
        p = subprocess.run(command + ["--list-cases"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(len(json.loads(p.stdout)), 30)
        p = subprocess.run(
            command + ["--list-accuracy-tests"], capture_output=True, text=True
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(len(json.loads(p.stdout)["targets"]), 62)
        runner = (ROOT / "tools/run_qsa_prefill_stage_joint_h800.sh").read_text()
        self.assertIn("--rounds 2", runner)
        self.assertIn("--accuracy-profile iteration", runner)
        self.assertNotIn("--cases", runner)
        self.assertNotIn("control-cycles", runner)
        self.assertIn("tools/qsa_prefill_stage_joint_trial.py", bench.SOURCES)
        self.assertIn(profiles.STAGE_JOINT_FILE, bench.SOURCES)
        # No timer/guard/validation helper was replaced to add the candidate.
        tree = ast.parse(Path(bench.__file__).read_text())
        report = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "write_table"
        )
        text = ast.unparse(report)
        self.assertLess(
            text.index("stage_joint.acceptance"),
            text.index("accuracy_profiles.annotate_acceptance"),
        )


if __name__ == "__main__":
    unittest.main()
