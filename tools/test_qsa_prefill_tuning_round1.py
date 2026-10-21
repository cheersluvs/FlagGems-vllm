# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Contracts for config-only changes, previous-eight isolation and30-case goals."""

import argparse
import copy
import hashlib
import json
import math
import statistics
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_cache_granularity as cache_trial
from tools import qsa_prefill_observed_workloads as workloads
from tools import qsa_prefill_triton_ab as ab
from tools import qsa_prefill_tuning_round1 as trial

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"


class Config:
    def __init__(self, **kwargs):
        self.values = kwargs

    def all_kwargs(self):
        return self.values


class TuningTests(unittest.TestCase):
    def test_only_decorator_changes_and_original_groups_are_frozen(self):
        current_source = PRODUCTION.read_text()
        source = cache_trial.round1_source(current_source)
        self.assertEqual(
            hashlib.sha256(
                source.replace(trial.CONFIG_GROUP, trial.PREVIOUS_CONFIG_GROUP).encode()
            ).hexdigest(),
            trial.PREVIOUS_SOURCE_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(current_source.encode()).hexdigest(),
            ab.INTEGRATED_PRODUCTION_SHA256,
        )
        groups = yaml.safe_load(
            (
                ROOT / "src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml"
            ).read_text()
        )
        old = groups[trial.PREVIOUS_CONFIG_GROUP]
        new = groups[trial.CONFIG_GROUP]
        self.assertEqual(
            hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest(),
            "0dd53937f05a2a598c0618fa6795148661ea921e60d0605e52fc9182e68c2a98",
        )
        self.assertEqual(
            hashlib.sha256(
                json.dumps(groups[trial.COMPACT_CONFIG_GROUP], sort_keys=True).encode()
            ).hexdigest(),
            "1b43ec0803c6af8eb15e9561cef94b445f0986a5dbc8625310432252b11036aa",
        )
        self.assertEqual(new[:8], old)
        self.assertEqual(
            new[8:],
            [
                dict(
                    META=dict(TILE_R=32, BLOCK_N=64, K_TILES=k),
                    num_warps=8,
                    num_stages=2,
                )
                for k in (16, 4)
            ],
        )
        self.assertEqual(len({json.dumps(c, sort_keys=True) for c in new}), 10)

    def test_previous_factory_rebinds_actual_functions_and_isolates_tuners(self):
        calls = []
        groups = {
            trial.PREVIOUS_CONFIG_GROUP: list(range(8)),
            trial.CONFIG_GROUP: list(range(10)),
            trial.COMPACT_CONFIG_GROUP: list(range(6)),
        }
        production = types.ModuleType("production")
        production.__file__ = str(PRODUCTION)
        production.persistent_topk = object()
        for name in ("_qsa_prefill_score_kernel", "_qsa_prefill_compact_score_kernel"):
            setattr(
                production,
                name,
                types.SimpleNamespace(
                    fn=types.SimpleNamespace(fn=object(), keys=["num_rows", "IS_FP8"]),
                    kernel_cache=({},),
                ),
            )
        exec(
            "def _prefill_logits(*, marker=1):\n"
            " return _qsa_prefill_score_kernel, _qsa_prefill_compact_score_kernel, marker\n"
            "def qsa_select_paged_prefill():\n"
            " return _prefill_logits(), persistent_topk\n",
            vars(production),
        )

        def tuner(**kwargs):
            calls.append(kwargs)
            return lambda jit: types.SimpleNamespace(
                fn=jit, keys=kwargs["key"], configs=kwargs["configs"]
            )

        fake = types.ModuleType("flaggems_vllm")
        fake.runtime = types.SimpleNamespace(get_tuned_config=lambda name: groups[name])
        utils = types.ModuleType("flaggems_vllm.utils")
        utils.libtuner = tuner
        utils.libentry = lambda: lambda fn: types.SimpleNamespace(
            fn=fn, kernel_cache=({},)
        )
        with patch.dict(
            sys.modules, {"flaggems_vllm": fake, "flaggems_vllm.utils": utils}
        ):
            previous = trial.previous_module(production)
        self.assertEqual(
            [c["configs"] for c in calls], [list(range(8)), list(range(6))]
        )
        self.assertEqual(previous._prefill_logits.__kwdefaults__, {"marker": 1})
        self.assertIs(
            previous.qsa_select_paged_prefill.__code__,
            production.qsa_select_paged_prefill.__code__,
        )
        self.assertIs(
            previous._prefill_logits.__code__, production._prefill_logits.__code__
        )
        self.assertIs(
            previous.qsa_select_paged_prefill()[1], production.persistent_topk
        )
        for name in ("_qsa_prefill_score_kernel", "_qsa_prefill_compact_score_kernel"):
            old, new = getattr(previous, name), getattr(production, name)
            self.assertIs(old.fn.fn, new.fn.fn)
            self.assertIsNot(old.kernel_cache[0], new.kernel_cache[0])
            self.assertIsNot(old.fn.keys, new.fn.keys)
        self.assertIs(previous._prefill_logits()[0], previous._qsa_prefill_score_kernel)
        self.assertIs(
            production._prefill_logits()[0], production._qsa_prefill_score_kernel
        )
        with patch.dict(
            sys.modules, {"flaggems_vllm": fake, "flaggems_vllm.utils": utils}
        ):
            previous_ten = cache_trial.previous_module(production, "round1")
        self.assertEqual(
            previous_ten._qsa_prefill_score_kernel.fn.configs, list(range(10))
        )
        self.assertIsNot(
            previous_ten._qsa_prefill_score_kernel.kernel_cache[0],
            previous._qsa_prefill_score_kernel.kernel_cache[0],
        )
        self.assertIs(previous_ten.persistent_topk, production.persistent_topk)
        with tempfile.TemporaryDirectory() as folder:
            changed = Path(folder) / "changed.py"
            changed.write_text(PRODUCTION.read_text() + "# changed\n")
            production.__file__ = str(changed)
            with patch.dict(
                sys.modules, {"flaggems_vllm": fake, "flaggems_vllm.utils": utils}
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "only the production config-group"
                ):
                    trial.previous_module(production)

    def test_scan_evidence_distinguishes_missing_resource_errors_and_cached_keys(self):
        a, b = Config(TILE_R=32), Config(TILE_R=64)
        tuner = types.SimpleNamespace(
            configs=[a, b],
            cache=object(),
            keys=["num_rows"],
            best_config=a,
            configs_timings={a: [0.1, 0.09, 0.11], b: [math.inf] * 3},
        )
        module = types.SimpleNamespace(
            _qsa_prefill_score_kernel=types.SimpleNamespace(fn=tuner),
            _qsa_prefill_compact_score_kernel=types.SimpleNamespace(
                fn=types.SimpleNamespace(configs=[a], cache=object(), keys=[])
            ),
        )
        state = trial.tuning_state(module)
        row = state["_qsa_prefill_score_kernel"]
        self.assertEqual(
            row["last_tuning_scan"][0]["latency_quantiles_ms"], [0.1, 0.09, 0.11]
        )
        self.assertIsNone(row["last_tuning_scan"][1]["latency_quantiles_ms"])
        self.assertEqual(
            state["_qsa_prefill_compact_score_kernel"]["last_tuning_scan"], []
        )
        self.assertEqual(row["last_tuner_selection"], a.all_kwargs())
        self.assertEqual(row["key_names"], ["num_rows"])
        json.dumps(state, allow_nan=False)


class AcceptanceTests(unittest.TestCase):
    @staticmethod
    def summary():
        selected = workloads.cases("scored-prefill")
        stages = {
            name: dict(
                status="passed",
                accepted=True,
                measurement_validated=True,
                baseline_us=dict(median=1.2),
                candidate_us=dict(median=1),
                speedup=1.2,
                paired_ratios=dict(median=1.2),
            )
            for name in (
                "score",
                "topk",
                "pipeline",
                "score_vs_previous_tuning",
                "pipeline_vs_previous_tuning",
            )
        }
        return dict(
            status="passed",
            accuracy=dict(status="passed"),
            manifest=dict(
                options=dict(
                    suite="scored-prefill",
                    compiler="flagtree",
                    candidate="production",
                    mode="kernel",
                    compare_previous_tuning=True,
                ),
                cases=selected,
            ),
            cases=[
                dict(case=c, status="passed", stages=copy.deepcopy(stages))
                for c in selected
            ],
        )

    def test_scored_suite_is_exactly_30_and_preserves_all_fixture_parameters(self):
        all32 = workloads.cases("combined-prefill")
        scored = workloads.cases("scored-prefill")
        self.assertEqual(len(scored), 30)
        self.assertEqual(
            [c for c in all32 if c not in scored],
            [c for c in workloads.observed_cases() if "select_all" in c["name"]],
        )
        self.assertTrue(
            all(not workloads.geometry(c).get("select_all") for c in scored)
        )
        self.assertEqual(len({c["name"] for c in scored}), 30)

    def test_arithmetic_mean_floor_and_primary_pipeline_not_score(self):
        s = self.summary()
        result = trial.acceptance(s)
        self.assertTrue(result["measurement_valid"] and result["target_met"])
        e = s["cases"][0]["stages"]["pipeline"]
        e.update(speedup=0.9, baseline_us=dict(median=0.9))
        result = trial.acceptance(s)
        self.assertTrue(result["measurement_valid"])
        self.assertFalse(result["target_met"])
        expected = statistics.mean([0.9] + [1.2] * 29)
        self.assertEqual(result["stages"]["pipeline"]["arithmetic_mean"], expected)
        self.assertNotAlmostEqual(
            expected, math.exp((math.log(0.9) + 29 * math.log(1.2)) / 30), places=5
        )
        self.assertEqual(result["stages"]["score"]["arithmetic_mean"], 1.2)
        for c in s["cases"]:
            c["stages"]["pipeline"].update(speedup=1.1, baseline_us=dict(median=1.1))
        self.assertFalse(trial.acceptance(s)["target_met"])
        s["cases"][0]["stages"]["pipeline"].update(
            speedup=0.98, baseline_us=dict(median=0.98)
        )
        for c in s["cases"][1:]:
            c["stages"]["pipeline"].update(speedup=1.2, baseline_us=dict(median=1.2))
        self.assertTrue(trial.acceptance(s)["target_met"])

    def test_incomplete_duplicate_changed_failed_shared_or_nonfinite_never_passes(self):
        mutations = [
            lambda s: s["cases"].pop(),
            lambda s: s["cases"].__setitem__(0, s["cases"][1]),
            lambda s: s["cases"][0]["case"].update(dim=96),
            lambda s: s["manifest"]["cases"].pop(),
            lambda s: s["cases"][0]["stages"]["pipeline"].update(status="failed"),
            lambda s: s["cases"][0]["stages"]["score"].update(accepted=False),
            lambda s: s["cases"][0]["stages"].pop("pipeline_vs_previous_tuning"),
            lambda s: s["cases"][0]["stages"]["pipeline"].update(speedup=math.nan),
            lambda s: s["cases"][0]["stages"]["pipeline"].update(speedup=5),
            lambda s: s["manifest"]["options"].update(shared_gpu_exploratory=True),
            lambda s: s["accuracy"].update(status="failed"),
            lambda s: s.update(status="running"),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(mutation=i):
                s = copy.deepcopy(self.summary())
                mutate(s)
                r = trial.acceptance(s)
                self.assertFalse(r["measurement_valid"])
                self.assertIsNone(r["target_met"])
                self.assertNotIn("stages", r)

    def test_report_writes_150_stage_rows_and_retains_topk_and_separate_groups(self):
        s = self.summary()
        with tempfile.TemporaryDirectory() as folder:
            bench.write_table(s, Path(folder))
            self.assertEqual(
                len((Path(folder) / "triton-stage-ab.tsv").read_text().splitlines()),
                151,
            )
            result = json.loads((Path(folder) / "overall-acceptance.json").read_text())
            self.assertTrue(result["target_met"])
            self.assertEqual(result["stages"]["topk"]["count"], 30)
            self.assertEqual(
                s["grouped_speedups"]["observed_perf:pipeline"]["count"], 8
            )
            self.assertEqual(s["grouped_speedups"]["fp8_perf:pipeline"]["count"], 10)

    def test_cli_lists_exact_set_propagates_comparison_and_rejects_wrong_profiles(self):
        base = [
            sys.executable,
            "-I",
            str(Path(bench.__file__)),
            "--suite",
            "scored-prefill",
            "--compiler",
            "flagtree",
            "--candidate",
            "production",
            "--compare-previous-tuning",
            "--list-cases",
        ]
        p = subprocess.run(base, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(p.stdout), workloads.cases("scored-prefill"))
        options = argparse.Namespace(
            compiler="flagtree",
            flagtree="/opt/flagtree",
            suite="scored-prefill",
            candidate="production",
            compare_previous_tuning=True,
            reference_file="native.py",
            rounds=3,
            kernel_warmup_ms=200,
            kernel_rep_ms=100,
            max_initial_temp_c=80,
        )
        child = bench.child_command(options, Path("/tmp/report"), "--worker")
        self.assertIn("--compare-previous-tuning", child)
        self.assertEqual(child[child.index("--previous-tuning") + 1], "eight")
        options.previous_tuning = "round1"
        child = bench.child_command(options, Path("/tmp/report"), "--worker")
        self.assertEqual(child[child.index("--previous-tuning") + 1], "round1")
        selected = [*base, "--previous-tuning", "round1"]
        result = subprocess.run(selected, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), workloads.cases("scored-prefill"))
        selected.remove("--compare-previous-tuning")
        self.assertNotEqual(subprocess.run(selected, capture_output=True).returncode, 0)
        for flag, value in [
            ("--compiler", "stock"),
            ("--suite", "combined-prefill"),
            ("--candidate", "key-major-reuse"),
        ]:
            args = list(base)
            args[args.index(flag) + 1] = value
            self.assertNotEqual(subprocess.run(args, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
