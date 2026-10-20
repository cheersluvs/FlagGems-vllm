# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU contracts for exact-JIT reuse, private tuners and incremental reports."""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from functools import cached_property
from pathlib import Path

import yaml

from tools import benchmark_qsa_prefill_triton_stages as bench

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "tools/qsa_prefill_reuse_trial.py"


def load(**namespace):
    nodes = [
        n for n in ast.parse(TRIAL.read_text()).body if isinstance(n, ast.FunctionDef)
    ]
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(TRIAL), "exec"), namespace
    )
    return namespace


class ReuseTests(unittest.TestCase):
    def test_expanded_configs_preserve_original_six_and_add_only_two(self):
        configs = yaml.safe_load(
            (
                ROOT / "src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml"
            ).read_text()
        )
        old = configs["qsa_select_paged_prefill_score"]
        trial = configs["qsa_select_paged_prefill_reuse_score"]
        self.assertEqual(len(old), 6)
        self.assertEqual(trial[:6], old)
        self.assertEqual(
            trial[6:],
            [
                dict(
                    META=dict(TILE_R=64, BLOCK_N=64, K_TILES=k),
                    num_warps=8,
                    num_stages=2,
                )
                for k in (4, 16)
            ],
        )
        # Semantic digest of the six choices validated in 83a19d9. Avoid a
        # git-history dependency so this check works in shallow card clones.
        self.assertEqual(
            hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest(),
            "1b43ec0803c6af8eb15e9561cef94b445f0986a5dbc8625310432252b11036aa",
        )

    def test_factory_keeps_jit_keys_and_default_tuner_untouched(self):
        jit = object()
        keys = ["num_rows", "HEAD_DIM"]
        template = types.SimpleNamespace(fn=jit, keys=keys, configs=["old"])
        modules = []
        tuner_calls = []

        def make(production):
            namespace = {}
            exec("def launch():\n return _qsa_prefill_score_kernel\n", namespace)
            namespace["_production_prefill_logits"] = namespace.pop("launch")
            namespace["_qsa_prefill_score_kernel"] = types.SimpleNamespace(
                fn=template, kernel_cache=({},)
            )
            module = types.SimpleNamespace(**namespace)
            # A real cloned module's function globals are its attribute dict.
            module._production_prefill_logits = types.FunctionType(
                module._production_prefill_logits.__code__, vars(module)
            )
            module.production_baseline = production
            modules.append(module)
            return module

        def tuner(**kwargs):
            tuner_calls.append(kwargs)
            return lambda fn: types.SimpleNamespace(
                fn=fn, keys=kwargs["key"], configs=kwargs["configs"]
            )

        ns = load(
            key_major=types.SimpleNamespace(
                candidate_module=make,
                _qsa_prefill_key_major_score_kernel=types.SimpleNamespace(fn=template),
            ),
            runtime=types.SimpleNamespace(
                get_tuned_config=lambda name: (
                    ["old", "new"]
                    if name == "qsa_select_paged_prefill_reuse_score"
                    else self.fail(name)
                )
            ),
            libtuner=tuner,
            libentry=lambda: lambda fn: types.SimpleNamespace(
                fn=fn, kernel_cache=({},)
            ),
        )
        production = types.SimpleNamespace(__name__="production")
        candidate = ns["candidate_module"](production)
        self.assertIs(candidate._qsa_prefill_score_kernel.fn.fn, jit)
        self.assertEqual(candidate._qsa_prefill_score_kernel.fn.keys, keys)
        self.assertIsNot(candidate._qsa_prefill_score_kernel.fn.keys, keys)
        self.assertEqual(template.configs, ["old"])
        self.assertIs(
            candidate._production_prefill_logits(), candidate._qsa_prefill_score_kernel
        )
        self.assertIs(candidate.retained_baseline, modules[1])
        self.assertIsNot(
            candidate._qsa_prefill_score_kernel.kernel_cache[0],
            candidate.retained_baseline._qsa_prefill_score_kernel.kernel_cache[0],
        )
        self.assertEqual(len(tuner_calls), 1)
        # Execute actual LibTuner hash properties to check tuning-DB separation.
        tree = ast.parse((ROOT / "src/flaggems_vllm/utils/libentry.py").read_text())
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LibTuner"
        )
        props = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("kernel_hash", "configs_hash")
        ]
        cls = ast.ClassDef(
            name="Hashes", bases=[], keywords=[], body=props, decorator_list=[]
        )
        env = dict(cached_property=cached_property, hashlib=hashlib)
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                "actual_libentry_hashes",
                "exec",
            ),
            env,
        )
        a, b = env["Hashes"](), env["Hashes"]()
        a.cache_key = b.cache_key = "same JIT"
        a.configs = ["old"]
        b.configs = ["old", "new"]
        self.assertNotEqual(a.kernel_hash, b.kernel_hash)

    def test_new_score_stage_is_score_only_and_keeps_topk_and_pipeline_reports(self):
        self.assertIn("score_vs_key_major", bench.SCORE_STAGES)
        self.assertEqual(bench.STAGE_ORDER[:3], ("score", "topk", "pipeline"))
        stages = {
            name: dict(
                status="passed",
                accepted=True,
                measurement_validated=True,
                baseline_us=dict(median=2),
                candidate_us=dict(median=1),
                speedup=2,
                paired_ratios=dict(median=2),
                baseline_kind=(
                    "retained_key_major"
                    if name == "score_vs_key_major"
                    else "native_vllm"
                ),
            )
            for name in bench.STAGE_ORDER
        }
        summary = dict(
            manifest=dict(
                options=dict(candidate="key-major-reuse", compiler="flagtree")
            ),
            cases=[
                dict(
                    case=dict(name="case", kind="upstream_perf"),
                    status="passed",
                    stages=stages,
                )
            ],
        )
        with tempfile.TemporaryDirectory() as folder:
            bench.write_table(summary, Path(folder))
            lines = (Path(folder) / "triton-stage-ab.tsv").read_text().splitlines()
            self.assertEqual(len(lines), 6)
            self.assertIn("gems_key_major_reuse_trial_us", lines[0])
            self.assertTrue(lines[-1].endswith("retained_key_major"))
            self.assertEqual(
                summary["grouped_speedups"]["upstream_perf:score_vs_key_major"][
                    "geometric_mean"
                ],
                2,
            )

    def test_cli_propagates_reuse_choice_and_rejects_stock(self):
        args = [
            sys.executable,
            "-I",
            str(ROOT / "tools/benchmark_qsa_prefill_triton_stages.py"),
            "--candidate",
            "key-major-reuse",
            "--suite",
            "combined-prefill",
            "--list-cases",
        ]
        proc = subprocess.run(
            args + ["--compiler", "flagtree"], capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(json.loads(proc.stdout)), 22)
        proc = subprocess.run(args, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("FlagTree", proc.stderr)
        options = argparse.Namespace(
            compiler="flagtree",
            flagtree="/opt/flagtree",
            suite="combined-prefill",
            candidate="key-major-reuse",
            reference_file="native.py",
            rounds=3,
            kernel_warmup_ms=200,
            kernel_rep_ms=100,
            max_initial_temp_c=80,
        )
        cmd = bench.child_command(options, Path("/tmp/report"), "--worker")
        self.assertEqual(cmd[cmd.index("--candidate") + 1], "key-major-reuse")
        self.assertEqual(cmd[cmd.index("--rounds") + 1], "3")


if __name__ == "__main__":
    unittest.main()
