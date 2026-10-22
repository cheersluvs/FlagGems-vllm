# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Exact-callable self controls, source pins, output owners and original AB/BA timing."""

import collections
import contextlib
import csv
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_combined_controls as controls
from tools import qsa_prefill_query_controls as common
from tools import qsa_prefill_query_split_matched as matched

ROOT = Path(__file__).resolve().parents[1]


class CombinedControlTests(unittest.TestCase):
    @staticmethod
    def modules():
        topk = object()
        production = types.SimpleNamespace(
            name="production",
            persistent_topk=topk,
            __file__=str(ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"),
        )
        candidate = types.SimpleNamespace(
            name="combined",
            persistent_topk=topk,
            production_baseline=production,
            __file__=str(ROOT / "tools/qsa_prefill_query_page_kernel.py"),
        )
        return production, candidate

    def test_schedule_has_every_comparison_stage_condition_and_balanced_positions(self):
        jobs = controls.schedule(4)
        self.assertEqual(len(jobs), 144)
        counts = collections.Counter(
            (j["comparison"], j["stage"], j["condition"]) for j in jobs
        )
        self.assertEqual(len(counts), 36)
        self.assertEqual(set(counts.values()), {4})
        for comparison in controls.COMPARISONS:
            positions = collections.Counter(
                i % 3 for i, j in enumerate(jobs) if j["comparison"] == comparison
            )
            self.assertEqual(positions, {0: 16, 1: 16, 2: 16})
        for condition in common.CONDITIONS:
            positions = []
            for cycle in range(1, 5):
                row = [j for j in jobs if j["cycle"] == cycle]
                positions.append(
                    next(i for i, j in enumerate(row) if j["condition"] == condition)
                    // 9
                )
            self.assertEqual(sorted(positions), [0, 1, 2, 3])
        for bad in (-1, 0, 1, 2, 5):
            with self.assertRaises(ValueError):
                controls.schedule(bad)

    def test_binding_checks_use_frozen_sources_and_exact_baseline_topk(self):
        p, c = self.modules()
        result = dict(case=dict(name=controls.CASE), candidate_kind="query-joint-page")
        self.assertEqual(
            controls.verify_bindings(result, p, c)["combined"]["sha256"],
            controls.COMBINED_SHA256,
        )
        c.production_baseline = object()
        with self.assertRaises(RuntimeError):
            controls.verify_bindings(result, p, c)
        c.production_baseline = p
        c.persistent_topk = object()
        with self.assertRaises(RuntimeError):
            controls.verify_bindings(result, p, c)
        c.persistent_topk = p.persistent_topk
        with patch.object(controls, "COMBINED_SHA256", "0" * 64), self.assertRaises(
            RuntimeError
        ):
            controls.verify_bindings(result, p, c)
        result["candidate_kind"] = "query-joint"
        with self.assertRaises(RuntimeError):
            controls.verify_bindings(result, p, c)

    def test_pipeline_checker_reads_exactly_one_launch_before_overwrite(self):
        launches = []
        args, native_chunks = object(), object()

        def launch():
            launches.append("launch")

        def checker(functions, left, right, chunks, torch, helpers):
            self.assertIs(left, args)
            self.assertIs(right, args)
            self.assertIs(chunks, native_chunks)
            functions["baseline"]()
            functions["candidate"]()
            self.assertEqual(launches, ["launch"])
            return 3

        context = (
            types.SimpleNamespace(validate_pipeline=checker),
            None,
            None,
            None,
            None,
            native_chunks,
        )
        self.assertEqual(
            controls.validate_side("pipeline", launch, None, None, args, context), 3
        )

    def test_cli_is_restricted_and_child_preserves_cycles(self):
        base = [
            "bench",
            "--compiler",
            "flagtree",
            "--candidate",
            "query-joint-page",
            "--suite",
            "scored-prefill",
            "--combined-self-control-cycles",
            "4",
            "--cases",
            controls.CASE,
        ]
        with tempfile.TemporaryDirectory() as folder:
            for extra, valid in (
                ([], True),
                (["--rounds", "1"], False),
                (["--shared-gpu-exploratory"], False),
                (["--candidate", "query-joint"], False),
                (["--cases", controls.CASE, "observed_short96_bfloat16"], False),
                (["--combined-self-control-cycles", "2"], False),
                (["--query-control-cycles", "4"], False),
            ):
                with patch.object(
                    sys, "argv", [*base, "--report-dir", folder, *extra]
                ), patch.object(
                    bench, "parent", return_value=7
                ) as parent, contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    if valid:
                        self.assertEqual(bench.main(), 7)
                        options, selected, _ = parent.call_args.args
                        self.assertEqual([c["name"] for c in selected], [controls.CASE])
                        command = bench.child_command(options, folder, "--worker")
                        self.assertEqual(
                            command[
                                command.index("--combined-self-control-cycles") + 1
                            ],
                            "4",
                        )
                        self.assertEqual(
                            command[command.index("--accuracy-profile") + 1],
                            "iteration",
                        )
                    else:
                        with self.assertRaises(SystemExit):
                            bench.main()
                        parent.assert_not_called()

    def harness(self, *, bad_shared=False, changed_resources=False):
        production, candidate = self.modules()
        writes = []
        validations = []
        timed = []

        class Tensor:
            shape = (4, 8)

            def __init__(self, pointer):
                self.pointer = pointer

            def stride(self):
                return (8, 1)

            def data_ptr(self):
                return self.pointer

        def chunks(bank, pointer):
            return [
                dict(
                    bank=bank,
                    score_args=[None] * 5 + [Tensor(pointer)],
                    topk_args=[None, None, Tensor(pointer + 1), Tensor(pointer + 2)],
                )
            ]

        banks = dict(a=chunks("a", 10), b=chunks("b", 20))
        owners = dict(a=object(), b=object())

        def bind(native, module, native_chunks, captured):
            result = {}
            for stage in ("score", "topk", "pipeline"):

                def launch(stage=stage):
                    writes.append((module.name, captured[0]["bank"], stage))

                result[stage] = dict(candidate=launch)
            return result

        result = dict(
            case=dict(name=controls.CASE),
            candidate_kind="query-joint-page",
            cache_granularity=dict(
                production=[
                    dict(
                        rows=288,
                        offset=0,
                        width=1088,
                        route="rectangular",
                        constants={},
                    )
                ]
            ),
        )

        def validate(stage, fn, module, captured, args, context):
            fn()
            bank = captured[0]["bank"]
            self.assertIs(args, owners[bank])
            self.assertEqual(writes[-1], (module.name, bank, stage))
            validations.append(writes[-1])
            entry = result["combined_self_controls"]["measurements"][-1]
            if (
                bad_shared
                and entry["comparison"] == "production_self"
                and entry["same_callable"]
            ):
                self.assertEqual(entry["baseline_bank"], entry["candidate_bank"])
                raise AssertionError("invalid first shared owner")
            return 3

        def timer(fn):
            fn()
            timed.append(fn)
            # Deliberately abnormal self ratios must be retained,not clamped.
            return (1.0, 2.0, 2.0, 1.0)[(len(timed) - 1) % 4]

        api = types.SimpleNamespace(
            ab=types.SimpleNamespace(bind_stages=bind),
            scoring=types.SimpleNamespace(
                make_timer=lambda *a: (timer, dict(mode="kernel", includes_topk=False)),
                measure_scores=bench.scoring.measure_scores,
            ),
            support=bench.support,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "case.json"
            with patch.object(common, "assert_choice"), patch.object(
                matched, "modules", side_effect=lambda a, b, t: (a, b)
            ), patch.object(
                common,
                "freeze",
                side_effect=[
                    dict(same_tile=True),
                    dict(same_tile=not changed_resources),
                ],
            ), patch.object(
                controls, "validate_side", side_effect=validate
            ):

                def call():
                    controls.run(
                        types.SimpleNamespace(
                            combined_self_control_cycles=4,
                            rounds=2,
                            kernel_warmup_ms=200,
                            kernel_rep_ms=100,
                        ),
                        result,
                        path,
                        bench=api,
                        torch=None,
                        helpers=None,
                        device=types.SimpleNamespace(synchronize=lambda: None),
                        native=None,
                        candidate=candidate,
                        production=production,
                        native_chunks=[],
                        production_chunks=banks["a"],
                        candidate_chunks=banks["b"],
                        production_args=owners["a"],
                        args=owners["b"],
                        monitor=lambda label: dict(label=label),
                    )

                if bad_shared or changed_resources:
                    with self.assertRaises((AssertionError, RuntimeError)):
                        call()
                    self.assertNotEqual(
                        result["combined_self_controls"]["status"], "passed"
                    )
                    if bad_shared:
                        entry = result["combined_self_controls"]["measurements"][-1]
                        self.assertFalse(entry["measurement_validated"])
                        self.assertTrue(entry["same_callable"])
                        self.assertEqual(
                            validations[-1],
                            ("production", entry["baseline_bank"], entry["stage"]),
                        )
                    return
                call()
            d = result["combined_self_controls"]
            self.assertEqual(d["status"], "passed")
            self.assertFalse(d["formal_acceptance_eligible"])
            self.assertEqual(len(d["measurements"]), 144)
            self.assertEqual(len(validations), 576)
            self.assertEqual(len(timed), 576)
            exact = 0
            for i, e in enumerate(d["measurements"]):
                self.assertEqual(e["speedup"], 0.5)
                self.assertEqual(
                    e["measurement_order"],
                    [["baseline", "candidate"], ["candidate", "baseline"]],
                )
                self.assertEqual(
                    e["same_module"], e["comparison"] != "production_vs_combined"
                )
                self.assertEqual(
                    e["same_callable"],
                    e["same_module"] and e["baseline_bank"] == e["candidate_bank"],
                )
                self.assertEqual(timed[4 * i] is timed[4 * i + 1], e["same_callable"])
                self.assertIs(timed[4 * i], timed[4 * i + 3])
                self.assertIs(timed[4 * i + 1], timed[4 * i + 2])
                exact += e["same_callable"]
                self.assertTrue(e["measurement_validated"])
            self.assertEqual(exact, 48)
            self.assertEqual(json.loads(path.read_text()), result)
            rows = list(
                csv.DictReader(
                    io.StringIO(path.with_suffix(".combined-controls.tsv").read_text()),
                    delimiter="\t",
                )
            )
            self.assertEqual(len(rows), 144)
            for row, e in zip(rows, d["measurements"]):
                self.assertEqual(row["comparison"], e["comparison"])
                self.assertEqual(float(row["speedup"]), 0.5)
                self.assertEqual(row["same_callable"], str(e["same_callable"]))

    def test_all_self_and_cross_conditions_use_the_intended_callable(self):
        self.harness()

    def test_bad_first_shared_result_is_checked_before_overwrite(self):
        self.harness(bad_shared=True)

    def test_fixed_kernel_change_rejects_run(self):
        self.harness(changed_resources=True)


if __name__ == "__main__":
    unittest.main()
