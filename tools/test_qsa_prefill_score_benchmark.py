# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""GPU-free checks for exact shared kernel-mode timing and score-only scope."""

import ast
import importlib.util
import statistics
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "score_benchmark", Path(__file__).with_name("qsa_prefill_score_benchmark.py")
)
scoring = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scoring)


class ScoreBenchmarkTests(unittest.TestCase):
    def test_shared_benchmark_kernel_branch_and_ms_to_us_conversion(self):
        source = Path(__file__).resolve().parents[1] / "benchmark/base.py"
        tree = ast.parse(source.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "Benchmark"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "get_latency"
        )
        calls, ran = [], []

        def do_bench(fn, **kwargs):
            calls.append(kwargs)
            fn()
            return 1.25

        testing = types.ModuleType("triton.testing")
        testing.do_bench = do_bench
        triton = types.ModuleType("triton")
        triton.__path__ = []
        triton.testing = testing
        consts = types.ModuleType("benchmark.consts")
        consts.BenchMode = types.SimpleNamespace(
            OPERATOR="operator",
            KERNEL="kernel",
            WRAPPER="wrapper",
            CUDAGRAPH="cudagraph",
        )
        namespace = dict(
            Config=object(), consts=consts, vendor_name="nvidia", triton=triton
        )
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        base = types.ModuleType("benchmark.base")
        original = object()
        base.Config = original

        # The actual shared function reads its module Config global.
        def method_call(instance, op):
            namespace["Config"] = base.Config
            return namespace["get_latency"](instance, op)

        base.Benchmark = type("Benchmark", (), {"get_latency": method_call})
        package = types.ModuleType("benchmark")
        package.__path__ = []
        package.base, package.consts = base, consts
        with patch.dict(
            sys.modules,
            {
                "benchmark": package,
                "benchmark.base": base,
                "benchmark.consts": consts,
                "triton": triton,
                "triton.testing": testing,
            },
        ):
            timer, info = scoring.make_timer(1000, 100)
            self.assertEqual(timer(lambda: ran.append("score")), 1250)
        self.assertIs(base.Config, original)
        self.assertEqual(ran, ["score"])
        self.assertEqual(
            calls, [dict(warmup=1000, rep=100, return_mode="median", grad_to_none=None)]
        )
        self.assertFalse(info["includes_topk"])
        self.assertEqual(info["mode"], "kernel")
        self.assertEqual(info["unit"], "us")

    def test_score_callable_uses_only_bound_launcher_and_chunk_arguments(self):
        calls = []
        chunks = [
            dict(
                score_args=(i, i + 1),
                score_kwargs=dict(launch_metadata="prepared"),
                topk_args=("MUST_NOT_USE",),
            )
            for i in range(3)
        ]
        fn = scoring.score_function(lambda *a, **kw: calls.append((a, kw)), chunks)
        fn()
        self.assertEqual(
            calls, [((i, i + 1), dict(launch_metadata="prepared")) for i in range(3)]
        )
        self.assertNotIn("topk", fn.__code__.co_names)

    def test_alternating_paired_kernel_measurement_and_speedup(self):
        baseline, candidate = object(), object()
        values = {baseline: 10, candidate: 8}
        order, points = [], []

        def timer(fn):
            order.append(fn)
            return values[fn]

        def stats(x):
            return dict(samples=x, median=statistics.median(x))

        result, observations, rounds = scoring.measure_scores(
            timer,
            baseline,
            candidate,
            3,
            lambda label: points.append(label) or label,
            stats,
        )
        self.assertEqual(
            order, [baseline, candidate, candidate, baseline, baseline, candidate]
        )
        self.assertEqual(result["speedup"], 1.25)
        self.assertEqual(result["paired_ratios"]["samples"], [1.25] * 3)
        self.assertEqual(len(observations), 4)
        self.assertEqual(len(rounds), 3)
        self.assertEqual(points[-1], "after_round_3")

    def test_request_sampling_covers_short_requests_and_chunk_boundaries(self):
        starts = [0, 7168]
        for _ in range(16):
            starts.append(starts[-1] + 64)
        rows = scoring.request_sample_rows(0, 8192, starts, lambda n: [0, n - 1])
        for begin, end in zip(starts, starts[1:]):
            self.assertIn(begin, rows)
            self.assertIn(end - 1, rows)
        # A score chunk can start/end inside a request; return local row indices.
        self.assertEqual(
            scoring.request_sample_rows(3, 8, [0, 5, 5, 9, 20], lambda n: [0, n - 1]),
            [0, 1, 2, 5, 6, 7],
        )

    def test_guard_failure_prevents_timing(self):
        def reject(_):
            raise RuntimeError("occupied")

        with self.assertRaisesRegex(RuntimeError, "occupied"), patch.object(
            scoring, "score_function"
        ) as timer:
            scoring.measure_scores(timer, object(), object(), 3, reject, lambda x: x)
        timer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
