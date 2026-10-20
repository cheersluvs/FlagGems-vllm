# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU contracts for a single-load hint and baseline-matched tuning."""

import ast
import copy
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_kernel_resources as resources

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "tools/qsa_prefill_eviction_trial.py"


def load():
    nodes = [
        n
        for n in ast.parse(TRIAL.read_text()).body
        if isinstance(n, ast.FunctionDef) and not n.decorator_list
    ]
    cfg = types.SimpleNamespace(
        kwargs=dict(TILE_R=64, BLOCK_N=64, K_TILES=16),
        num_warps=8,
        num_stages=2,
        num_ctas=1,
    )
    jit = object()
    template = types.SimpleNamespace(fn=jit, keys=["num_rows", "HEAD_DIM"])
    ns = dict(
        CONFIG_NAME="qsa_select_paged_prefill_reuse_score",
        runtime=types.SimpleNamespace(get_tuned_config=lambda name: [cfg]),
        _qsa_prefill_evict_first_score_kernel=types.SimpleNamespace(fn=template),
        libtuner=lambda **kwargs: lambda fn: types.SimpleNamespace(fn=fn, **kwargs),
        libentry=lambda: lambda fn: types.SimpleNamespace(fn=fn, kernel_cache=({},)),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRIAL), "exec"), ns)
    constants = ns["config_values"](cfg)
    baseline = types.SimpleNamespace(
        kernel_cache=({"real_public": (object(), constants)},)
    )
    candidate = types.SimpleNamespace(
        _qsa_prefill_score_kernel=types.SimpleNamespace(kernel_cache=({},)),
        retained_baseline=types.SimpleNamespace(_qsa_prefill_score_kernel=baseline),
    )
    return ns, candidate, cfg, constants, jit


class EvictionTests(unittest.TestCase):
    def test_factory_uses_eight_choice_reuse_baseline_and_private_hint_entry(self):
        ns, baseline, _, _, _ = load()
        baseline_module = baseline.retained_baseline
        module = types.SimpleNamespace()
        ns["key_major"] = types.SimpleNamespace(
            candidate_module=lambda production: module
        )
        ns["reuse"] = types.SimpleNamespace(
            candidate_module=lambda production: baseline_module
        )
        ns["compact"] = types.SimpleNamespace(
            isolated_score_entry=lambda entry: types.SimpleNamespace(
                fn=entry.fn, kernel_cache=({},)
            )
        )
        actual = ns["candidate_module"](types.SimpleNamespace(__name__="production"))
        self.assertIs(actual.retained_baseline, baseline_module)
        self.assertEqual(actual.retained_stage_name, "score_vs_reuse")
        self.assertEqual(actual.retained_baseline_kind, "retained_reuse")
        self.assertIs(actual.freeze_baseline, ns["freeze_baseline"])
        self.assertIs(actual.verify_matched, ns["verify_matched"])
        self.assertIsNot(
            actual._qsa_prefill_score_kernel,
            ns["_qsa_prefill_evict_first_score_kernel"],
        )

    def test_only_key_load_hint_differs_from_retained_kernel(self):
        def kernel(path, name):
            return next(
                n
                for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )

        original = kernel(
            ROOT / "tools/qsa_prefill_key_major_trial.py",
            "_qsa_prefill_key_major_score_kernel",
        )
        hinted = kernel(TRIAL, "_qsa_prefill_evict_first_score_kernel")
        self.assertEqual(ast.dump(hinted.args), ast.dump(original.args))
        hints = []
        for node in ast.walk(hinted):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "keys" for t in node.targets
            ):
                hints = [k for k in node.value.keywords if k.arg == "eviction_policy"]
                node.value.keywords = [
                    k for k in node.value.keywords if k.arg != "eviction_policy"
                ]
        self.assertEqual(len(hints), 1)
        self.assertEqual(ast.literal_eval(hints[0].value), "evict_first")
        self.assertEqual(
            ast.dump(ast.Module(body=hinted.body, type_ignores=[])),
            ast.dump(ast.Module(body=original.body, type_ignores=[])),
        )
        self.assertEqual(
            ast.dump(hinted.decorator_list[-1]), ast.dump(original.decorator_list[-1])
        )
        self.assertNotIn("torch.", TRIAL.read_text())

    def test_matching_uses_actual_baseline_choice_without_touching_it(self):
        ns, candidate, cfg, constants, jit = load()
        baseline = candidate.retained_baseline._qsa_prefill_score_kernel
        before = copy.deepcopy(constants)
        old = candidate._qsa_prefill_score_kernel
        record = ns["freeze_baseline"](candidate)
        self.assertEqual(record["status"], "matched")
        entry = candidate._qsa_prefill_score_kernel
        self.assertIsNot(entry, old)
        self.assertIs(entry.fn.fn, jit)
        self.assertEqual(entry.fn.configs, [cfg])
        self.assertEqual(entry.fn.key, ["num_rows", "HEAD_DIM"])
        self.assertEqual(ns["cached_configs"](baseline), [before])
        entry.kernel_cache[0]["hinted"] = (object(), dict(constants))
        ns["verify_matched"](candidate, record)
        entry.kernel_cache[0]["hinted"][1]["num_warps"] = 4
        with self.assertRaisesRegex(RuntimeError, "no longer match"):
            ns["verify_matched"](candidate, record)

    def test_ambiguous_unknown_or_prewarmed_candidate_is_rejected(self):
        for reason in ("ambiguous", "unknown", "prewarmed"):
            ns, candidate, _, constants, _ = load()
            if reason == "ambiguous":
                candidate.retained_baseline._qsa_prefill_score_kernel.kernel_cache[0][
                    "other"
                ] = (object(), dict(constants, K_TILES=4))
            elif reason == "unknown":
                constants["K_TILES"] = 128
            else:
                candidate._qsa_prefill_score_kernel.kernel_cache[0]["early"] = (
                    object(),
                    constants,
                )
            with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                ns["freeze_baseline"](candidate)

    def test_compact_and_select_all_controls_do_not_create_matched_tuner(self):
        ns, candidate, _, constants, _ = load()
        candidate.retained_baseline._qsa_prefill_score_kernel.kernel_cache[0].clear()
        original = candidate._qsa_prefill_score_kernel
        record = ns["freeze_baseline"](candidate)
        self.assertEqual(record["status"], "not_applicable")
        self.assertIs(candidate._qsa_prefill_score_kernel, original)
        ns["verify_matched"](candidate, record)
        original.kernel_cache[0]["unexpected"] = (object(), constants)
        with self.assertRaisesRegex(RuntimeError, "Unexpected rectangular"):
            ns["verify_matched"](candidate, record)

    def test_actual_worker_warms_retained_before_freezing_hint(self):
        ns, candidate, _, constants, _ = load()
        entry = candidate.retained_baseline._qsa_prefill_score_kernel
        entry.kernel_cache[0].clear()
        order = []

        def public(*args):
            order.append("public")
            entry.kernel_cache[0]["real_public"] = (object(), constants)

        candidate.retained_baseline.qsa_select_paged_prefill = public
        candidate.retained_stage_name = "score_vs_reuse"
        candidate.retained_baseline_kind = "retained_reuse"
        candidate.freeze_baseline = lambda module: (
            order.append("freeze"),
            ns["freeze_baseline"](module),
        )[1]
        worker = next(
            n
            for n in ast.parse(Path(bench.__file__).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "worker"
        )
        start = next(
            i
            for i, n in enumerate(worker.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "retained" for t in n.targets)
        )
        end = next(
            i
            for i, n in enumerate(worker.body[start:], start)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "fastpath" for t in n.targets)
        )
        result = {}
        env = dict(
            candidate=candidate,
            production=types.SimpleNamespace(
                _qsa_prefill_score_kernel=types.SimpleNamespace(kernel_cache=({},))
            ),
            args=(),
            result=result,
            support=types.SimpleNamespace(
                score_configs=lambda module: ns["cached_configs"](
                    module._qsa_prefill_score_kernel
                )
            ),
        )
        exec(
            compile(
                ast.Module(body=worker.body[start:end], type_ignores=[]),
                "actual_worker_preparation",
                "exec",
            ),
            env,
        )
        self.assertEqual(order, ["public", "freeze"])
        self.assertEqual(result["eviction_config_match"]["status"], "matched")
        self.assertEqual(result["retained_baseline_kind"], "retained_reuse")

    def test_report_labels_new_increment_and_retains_topk_pipeline(self):
        self.assertIn("score_vs_reuse", bench.SCORE_STAGES)
        stages = {
            n: dict(
                status="passed",
                accepted=True,
                measurement_validated=True,
                baseline_us=dict(median=2),
                candidate_us=dict(median=1),
                speedup=2,
                paired_ratios=dict(median=2),
                baseline_kind=(
                    "retained_reuse" if n == "score_vs_reuse" else "native_vllm"
                ),
            )
            for n in bench.STAGE_ORDER
            if n != "score_vs_key_major"
        }
        summary = dict(
            manifest=dict(
                options=dict(candidate="key-major-evict-first", compiler="flagtree")
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
            text = (Path(folder) / "triton-stage-ab.tsv").read_text()
            self.assertIn("gems_key_major_evict_first_trial_us", text)
            self.assertIn("\ttopk\tpassed\t", text)
            self.assertIn("\tpipeline\tpassed\t", text)
            self.assertIn("\tscore_vs_reuse\tpassed\t", text)
            self.assertNotIn("\tscore_vs_key_major\t", text)
            self.assertEqual(
                summary["grouped_speedups"]["upstream_perf:score_vs_reuse"][
                    "geometric_mean"
                ],
                2,
            )

    def test_cli_lists_22_cases_and_rejects_stock_compiler(self):
        command = [
            sys.executable,
            "-I",
            str(ROOT / "tools/benchmark_qsa_prefill_triton_stages.py"),
            "--candidate",
            "key-major-evict-first",
            "--suite",
            "combined-prefill",
            "--list-cases",
        ]
        proc = subprocess.run(
            command + ["--compiler", "flagtree"], capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(json.loads(proc.stdout)), 22)
        proc = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("FlagTree", proc.stderr)

    def test_policy_sites_are_static_text_counts_outside_timing(self):
        kernel = types.SimpleNamespace(
            metadata=types.SimpleNamespace(),
            asm=dict(
                cubin=b"bin",
                ptx="""
        // createpolicy.fractional.L2::evict_first.b64 ignored;
        createpolicy.fractional.L2::evict_first.b64 %rd3, 1.0;
        cp.async.cg.shared.global.L2::cache_hint [%rd1], [%rd2], 16, %rd3;
        ld.global.L1::evict_first.b32 %r1, [%rd2];
        """,
            ),
        )
        self.assertEqual(
            resources.compiled_record(kernel)["ptx_cache_policy_sites"],
            dict(fractional_createpolicy=1, evict_first=2, cp_async_l2_hint=1),
        )


if __name__ == "__main__":
    unittest.main()
