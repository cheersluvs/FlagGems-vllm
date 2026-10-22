# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU checks for balanced scheduling, fixed choices and shared-buffer validation."""

import contextlib
import copy
import csv
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_query_controls as controls
from tools import qsa_prefill_query_split_matched as matched

ROOT = Path(__file__).resolve().parents[1]


def trace():
    return [dict(rows=100, offset=0, width=4096, route="rectangular", constants={})]


class ControlTests(unittest.TestCase):
    def test_balanced_conditions_and_stage_coverage(self):
        for case, stages in ((controls.CASES[0], 1), (controls.CASES[1], 3)):
            jobs = controls.schedule(case, 4)
            self.assertEqual(len(jobs), 4 * 4 * stages)
            for condition in controls.CONDITIONS:
                positions = []
                for cycle in range(1, 5):
                    row = [j for j in jobs if j["cycle"] == cycle]
                    group = [j for j in row if j["condition"] == condition]
                    self.assertEqual(len(group), stages)
                    positions.append(row.index(group[0]) // stages)
                self.assertEqual(sorted(positions), [0, 1, 2, 3])
        transitions = [
            (a, b) for row in controls.CONDITION_ORDERS for a, b in zip(row, row[1:])
        ]
        self.assertEqual(len(set(transitions)), 12)
        for case, cycles in (("unknown", 4), (controls.CASES[0], 1)):
            with self.assertRaises(ValueError):
                controls.schedule(case, cycles)

    def test_choices_are_existing_yaml_and_traces_are_not_modified(self):
        groups = yaml.safe_load(
            (
                ROOT / "src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml"
            ).read_text()
        )
        original = trace()
        snapshot = copy.deepcopy(original)
        for case, joint, parts in (
            (controls.CASES[0], True, 1),
            (controls.CASES[0], True, 2),
            (controls.CASES[1], False, 1),
            (controls.CASES[1], True, 1),
        ):
            selected = controls.fixed_traces(original, case, joint=joint, parts=parts)
            group = (
                "qsa_select_paged_prefill_query_joint_score"
                if joint
                else "qsa_select_paged_prefill_tuning_round1_score"
            )
            configs = [
                types.SimpleNamespace(
                    kwargs=c["META"],
                    num_warps=c["num_warps"],
                    num_stages=c["num_stages"],
                    num_ctas=c.get("num_ctas", 1),
                )
                for c in groups[group]
            ]
            module = types.SimpleNamespace(
                _qsa_prefill_score_kernel=types.SimpleNamespace(
                    fn=types.SimpleNamespace(configs=configs)
                )
            )
            controls.assert_choice(module, selected)
            selected[0]["constants"]["K_TILES"] = 999
            with self.assertRaises(RuntimeError):
                controls.assert_choice(module, selected)
        self.assertEqual(original, snapshot)
        with self.assertRaises(ValueError):
            controls.fixed_traces(original, controls.CASES[1], joint=True, parts=2)
        with self.assertRaises(RuntimeError):
            controls.fixed_traces([], controls.CASES[0], joint=True)

    def test_freeze_normalizes_only_mode(self):
        a, b = object(), object()
        records = {
            a: [dict(chunk=[10, 0, 64], constants=dict(TILE_R=32, QUERY_PARTS=1))],
            b: [dict(chunk=[10, 0, 64], constants=dict(TILE_R=32, QUERY_PARTS=2))],
        }
        with patch.object(
            matched,
            "freeze_and_records",
            side_effect=lambda pair: dict(baseline=records[pair[0]]),
        ):
            self.assertTrue(controls.freeze((a, b))["same_tile"])
            records[b][0]["constants"]["TILE_R"] = 64
            with self.assertRaises(RuntimeError):
                controls.freeze((a, b))

    def test_pipeline_validation_checks_one_owner_at_a_time(self):
        args, chunks, native_chunks = object(), object(), object()
        fn = Mock()
        api = types.SimpleNamespace(validate_pipeline=Mock(return_value=7))
        context = (api, "torch", "helpers", "device", "native", native_chunks)
        self.assertEqual(
            controls.validate_side("pipeline", fn, None, chunks, args, context), 7
        )
        call = api.validate_pipeline.call_args.args
        self.assertEqual(call[0], dict(baseline=fn, candidate=fn))
        self.assertIs(call[1], args)
        self.assertIs(call[2], args)
        self.assertIs(call[3], native_chunks)

    def test_child_forwarding_and_restricted_cli(self):
        common = [
            "--compiler",
            "flagtree",
            "--candidate",
            "query-joint",
            "--suite",
            "scored-prefill",
            "--query-control-cycles",
            "4",
            "--cases",
            *controls.CASES,
        ]
        with tempfile.TemporaryDirectory() as folder:
            for extra, valid in (
                ([], True),
                (["--rounds", "1"], False),
                (["--shared-gpu-exploratory"], False),
                (["--query-control-cycles", "2"], False),
            ):
                with patch.object(
                    sys, "argv", ["bench", *common, "--report-dir", folder, *extra]
                ), patch.object(
                    bench, "parent", return_value=7
                ) as parent, contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    if valid:
                        self.assertEqual(bench.main(), 7)
                        options, cases, _ = parent.call_args.args
                        self.assertEqual(
                            {c["name"] for c in cases}, set(controls.CASES)
                        )
                        command = bench.child_command(options, folder, "--worker")
                        self.assertEqual(
                            command[command.index("--query-control-cycles") + 1], "4"
                        )
                    else:
                        with self.assertRaises(SystemExit):
                            bench.main()
                        parent.assert_not_called()

    def run_harness(self, case, *, fail_validation=False, change_resources=False):
        writes, validations = [], []
        topk = object()
        production = types.SimpleNamespace(name="production", persistent_topk=topk)
        candidate = types.SimpleNamespace(name="joint", persistent_topk=topk)

        class Tensor:
            shape = (4, 8)

            def __init__(self, pointer):
                self.pointer = pointer

            def stride(self):
                return (8, 1)

            def data_ptr(self):
                return self.pointer

        def chunks(bank, base):
            return [
                dict(
                    bank=bank,
                    score_args=[None] * 5 + [Tensor(base)],
                    topk_args=[None, None, Tensor(base + 1), Tensor(base + 2)],
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

        def validate(stage, fn, module, captured, args, context):
            fn()
            bank = captured[0]["bank"]
            self.assertIs(args, owners[bank])
            self.assertEqual(writes[-1], (module.name, bank, stage))
            validations.append(writes[-1])
            # BF16 cycle1: separate and swapped occupy the first24 checks.
            # The next pair shares bank B; reject its baseline before overwrite.
            if fail_validation and len(validations) == 25:
                raise AssertionError("invalid first shared owner")
            return 3

        def timer(fn):
            fn()
            return 1.0

        api = types.SimpleNamespace(
            ab=types.SimpleNamespace(bind_stages=bind),
            scoring=types.SimpleNamespace(
                make_timer=lambda *a: (timer, dict(mode="kernel", includes_topk=False)),
                measure_scores=bench.scoring.measure_scores,
            ),
            support=bench.support,
        )
        result = dict(case=dict(name=case), cache_granularity=dict(production=trace()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "case.json"
            with patch.object(controls, "assert_choice"), patch.object(
                matched,
                "modules",
                side_effect=lambda left, right, selected: (left, right),
            ), patch.object(
                controls,
                "freeze",
                side_effect=[
                    dict(same_tile=True),
                    dict(same_tile=not change_resources),
                ],
            ), patch.object(
                controls, "validate_side", side_effect=validate
            ):
                call = lambda: controls.run(
                    types.SimpleNamespace(
                        query_control_cycles=4,
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
                if fail_validation or change_resources:
                    with self.assertRaises((AssertionError, RuntimeError)):
                        call()
                    self.assertNotEqual(result["query_controls"]["status"], "passed")
                    if fail_validation:
                        self.assertEqual(len(validations), 25)
                        self.assertEqual(validations[-1], ("production", "b", "score"))
                        last = result["query_controls"]["measurements"][-1]
                        self.assertEqual(last["condition"], "shared_b")
                        self.assertFalse(last["measurement_validated"])
                    return
                call()
            diagnostic = result["query_controls"]
            self.assertEqual(diagnostic["status"], "passed")
            self.assertFalse(diagnostic["formal_acceptance_eligible"])
            jobs = controls.schedule(case, 4)
            self.assertEqual(len(diagnostic["measurements"]), len(jobs))
            self.assertEqual(len(validations), len(jobs) * 4)
            for entry, job in zip(diagnostic["measurements"], jobs):
                self.assertEqual(
                    entry["measurement_order"],
                    [["baseline", "candidate"], ["candidate", "baseline"]],
                )
                self.assertEqual(
                    (entry["baseline_bank"], entry["candidate_bank"]),
                    controls.CONDITIONS[job["condition"]],
                )
                self.assertTrue(entry["measurement_validated"])
            saved = json.loads(path.read_text())
            self.assertEqual(saved, result)
            rows = list(
                csv.DictReader(
                    io.StringIO(path.with_suffix(".controls.tsv").read_text()),
                    delimiter="\t",
                )
            )
            self.assertEqual(len(rows), len(jobs))

    def test_both_case_orchestrations_and_all_buffer_owners(self):
        for case in controls.CASES:
            with self.subTest(case=case):
                self.run_harness(case)

    def test_bad_first_shared_output_is_not_hidden(self):
        self.run_harness(controls.CASES[1], fail_validation=True)

    def test_recompilation_during_timing_is_rejected(self):
        self.run_harness(controls.CASES[0], change_resources=True)


if __name__ == "__main__":
    unittest.main()
