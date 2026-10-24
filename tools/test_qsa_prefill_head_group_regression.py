# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Regression fixture mapping, expanded checks and scoped child handoff."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_head_group_regression as probe


class HeadGroupRegressionTests(unittest.TestCase):
    def test_all30_original_geometry_and_upstream_configs_preserved(self):
        data = probe.manifest()
        self.assertEqual(len(data["cases"]), 30)
        original = json.loads(probe.gated.MANIFEST.read_text())
        self.assertEqual(data["cases"][:12], original["cases"])
        self.assertEqual(sum(len(c["traces"]) for c in data["cases"]), 53)
        self.assertEqual(
            sum(c["case"]["dtype"] == "float8_e4m3fn" for c in data["cases"]), 10
        )
        self.assertEqual(probe.verify_source()["manifest_sha256"], probe.MANIFEST_SHA)
        routes = {t["route"] for c in data["cases"] for t in c["traces"]}
        self.assertEqual(routes, {"compact", "fp8", "rectangular"})

    def test_only_previously_promising_case_activates_frozen_gate(self):
        active = []
        for item in probe.manifest()["cases"]:
            case = item["case"]
            for trace in item["traces"]:
                named = dict(
                    trace["constants"],
                    IS_FP8=case["dtype"] == "float8_e4m3fn",
                    NUM_HEADS=case["heads"],
                    HEAD_DIM=case["dim"],
                )
                if trace["route"] == "rectangular" and probe.gated.eligible(named):
                    active.append(case["name"])
        self.assertEqual(active, [probe.gated.REPEAT_CASE] * 4)

    def test_expanded_fixture_boundaries_and_head_dim_fallbacks(self):
        specs = probe.fixtures()
        self.assertEqual(len(specs), 10)
        self.assertEqual(
            {s["rows"] for s in specs if s["group"] == 1}, {1, 63, 64, 65, 129}
        )
        self.assertEqual(
            {(s["heads"], s["dim"]) for s in specs if s["group"] == 0},
            {(2, 64), (8, 128), (4, 256)},
        )
        self.assertEqual({s["parts"] for s in specs if s["group"] == 2}, {1, 2})

    def test_all_expanded_checks_are_executed_and_grouped(self):
        old = [dict(status="passed") for _ in range(3)]
        with patch.object(probe, "BOUNDARIES", return_value=old), patch.object(
            probe,
            "check_fixture",
            side_effect=lambda *args: dict(args[-1], status="passed"),
        ) as run:
            checks = probe.boundary_checks(None, None, None, None)
        self.assertEqual(run.call_count, 10)
        self.assertEqual([len(c["expanded"]) for c in checks], [3, 5, 2])
        self.assertTrue(all(c["status"] == "passed" for c in checks))

    def test_group_aggregates_preserve_stage_results(self):
        selected = [probe.manifest()["cases"][i]["case"] for i in (0, 12, 20)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "head-group-summary.json"
            cases = [
                dict(
                    case=c["name"],
                    stages={
                        k: dict(speedup=1 + i / 10)
                        for k in ("score", "topk", "pipeline")
                    },
                )
                for i, c in enumerate(selected)
            ]
            path.write_text(json.dumps(dict(status="passed", cases=cases)))
            with patch.object(probe, "COORDINATE", return_value=0):
                import types

                self.assertEqual(
                    probe.coordinate(
                        types.SimpleNamespace(
                            report_dir=folder, cases=[c["name"] for c in selected]
                        )
                    ),
                    0,
                )
            output = json.loads(path.read_text())
            self.assertEqual(output["cases"], cases)
            for i, name in enumerate(("upstream_bf16", "observed_bf16", "fp8")):
                self.assertEqual(output["group_aggregates"][name]["count"], 1)
                self.assertAlmostEqual(
                    output["group_aggregates"][name]["score"], 1 + i / 10
                )

    def test_targeted_four_round_cli_handoff(self):
        from tools import qsa_prefill_flagtree_ab as profile

        case = "fp8_tail113_float8_e4m3fn"

        def coordinate(options):
            self.assertEqual(options.cases, [case])
            self.assertEqual(options.rounds, 4)
            self.assertIs(probe.gated.base.make_pair, probe.gated.make_pair)
            return 0

        with patch.object(
            sys,
            "argv",
            [
                probe.__file__,
                "--report-dir",
                "/tmp/r",
                "--baseline-report",
                "/tmp/b",
                "--cases",
                case,
                "--rounds",
                "4",
            ],
        ), patch.object(profile, "configure"), patch.object(
            probe, "COORDINATE", side_effect=coordinate
        ):
            self.assertEqual(probe.main(), 0)

    def test_targeted_reporting_keeps_outlier_and_rejects_changed_binary(self):
        import types

        case = "fp8_tail113_float8_e4m3fn"
        row = dict(
            speedup=0.95,
            baseline_us=dict(samples=[266.0, 324.0, 267.0, 266.0]),
            candidate_us=dict(samples=[267.0, 354.0, 267.0, 267.0]),
            paired_ratios=dict(samples=[266 / 267, 324 / 354, 1.0, 266 / 267]),
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "head-group-summary.json"
            detail = Path(folder) / (case + ".json")
            stages = {k: row for k in ("score", "topk", "pipeline")}
            original = dict(status="passed", cases=[dict(case=case, stages=stages)])
            item = dict(
                rounds=4,
                stages=stages,
                self_controls={"production:score": row},
                kernels_before=dict(
                    candidate=[
                        dict(
                            head_group_gate=dict(active=False),
                            original_binary_verified=True,
                        )
                    ]
                ),
            )
            path.write_text(json.dumps(original))
            detail.write_text(json.dumps(item))
            options = types.SimpleNamespace(report_dir=folder, cases=[case])
            with patch.object(probe, "COORDINATE", return_value=0):
                self.assertEqual(probe.coordinate(options), 0)
                result = json.loads(path.read_text())
                self.assertEqual(result["cases"], original["cases"])
                proof = result["timing_recheck"]
                self.assertEqual(
                    proof["samples"]["score"]["candidate_us"],
                    [267.0, 354.0, 267.0, 267.0],
                )
                self.assertEqual(proof["rounds"], 4)
                self.assertAlmostEqual(
                    proof["samples"]["score"]["candidate_max_min_ratio"], 354 / 267
                )
                item["kernels_before"]["candidate"][0][
                    "original_binary_verified"
                ] = False
                detail.write_text(json.dumps(item))
                with self.assertRaisesRegex(RuntimeError, "exact original"):
                    probe.coordinate(options)

    def test_regression_cli_child_handoff_and_full_restoration(self):
        from tools import benchmark_qsa_prefill_triton_stages as bench
        from tools import qsa_prefill_flagtree_ab as profile

        gated, base = probe.gated, probe.gated.base
        old = (
            gated.__file__,
            gated.manifest,
            gated.verify_source,
            base.__file__,
            base.boundary_checks,
        )

        def coordinate(options):
            self.assertEqual(len(options.cases), 30)
            self.assertEqual(options.rounds, 2)
            self.assertEqual(options.promising_rounds, 4)
            self.assertEqual(
                Path(base.__file__).resolve(), Path(probe.__file__).resolve()
            )
            self.assertIs(base.manifest, probe.manifest)
            self.assertIs(base.verify_source, probe.verify_source)
            self.assertIs(base.boundary_checks, probe.boundary_checks)
            self.assertIs(base.make_pair, gated.make_pair)
            self.assertTrue(probe.EXTRA_SOURCES <= set(bench.SOURCES))
            return 0

        with patch.object(
            sys,
            "argv",
            [probe.__file__, "--report-dir", "/tmp/r", "--baseline-report", "/tmp/b"],
        ), patch.object(profile, "configure"), patch.object(
            probe, "COORDINATE", side_effect=coordinate
        ):
            self.assertEqual(probe.main(), 0)
        self.assertEqual(
            (
                gated.__file__,
                gated.manifest,
                gated.verify_source,
                base.__file__,
                base.boundary_checks,
            ),
            old,
        )


if __name__ == "__main__":
    unittest.main()
