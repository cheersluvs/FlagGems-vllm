# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Frozen stage-only changes and the actual scoped worker/coordinator protocol."""

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_bf16_stage_depth_probe as probe


class StageDepthTests(unittest.TestCase):
    def test_current_manifest_and_mixed_grouped_tail(self):
        m = probe.manifest()
        probe.verify_source()
        self.assertEqual(
            m["baseline_code_commit"], "314a9de2cde6ec360ca7d6c45fa07a72b88e9c7d"
        )
        self.assertEqual(
            [v["case"]["dtype"] for v in m["cases"]],
            ["bfloat16"] * 5 + ["float8_e4m3fn"],
        )
        mixed = next(
            c
            for c in m["cases"]
            if c["case"]["name"] == "perf_n2_q8192_p102400_bfloat16"
        )
        self.assertEqual(
            [t["constants"]["HEAD_GROUP"] for t in mixed["traces"]],
            [False, False, False, True],
        )

    def test_trace_copy_changes_only_stage_and_preserves_fallback(self):
        for c in probe.manifest()["cases"]:
            original = copy.deepcopy(c["traces"])
            left, right = probe.fixed_traces(original), probe.fixed_traces(
                original, True
            )
            self.assertEqual(original, c["traces"])
            for a, b in zip(left, right):
                if a["route"] == "rectangular":
                    b["constants"]["num_stages"] = 2
                self.assertEqual(a, b)
            self.assertEqual(left, original)
        boundary = [
            dict(
                route="rectangular",
                constants=dict(TILE_R=64, num_warps=4, num_stages=2),
            )
        ]
        self.assertEqual(
            probe.fixed_traces(boundary, True)[0]["constants"],
            dict(TILE_R=64, num_warps=4, num_stages=1, HEAD_GROUP=True),
        )
        self.assertNotIn("HEAD_GROUP", boundary[0]["constants"])

    def test_actual_factory_reuses_same_jit_and_distinct_configs(self):
        production = types.ModuleType("fake_production")
        production.persistent_topk = object()
        for name, _ in probe.base.ENTRIES:
            jit = object()
            setattr(
                production,
                name,
                types.SimpleNamespace(arg_names=[], fn=types.SimpleNamespace(fn=jit)),
            )
        exec(
            "def _prefill_logits(): return _qsa_prefill_score_kernel\n"
            "def qsa_select_paged_prefill(): return _prefill_logits()",
            vars(production),
        )
        traces = probe.manifest()["cases"][2]["traces"]
        pair = probe.make_pair(production, traces)
        self.assertIs(pair[0].persistent_topk, pair[1].persistent_topk)
        for name, route in probe.base.ENTRIES:
            a, b = getattr(pair[0], name), getattr(pair[1], name)
            self.assertIs(a.jit, getattr(production, name).fn.fn)
            self.assertIs(a.jit, b.jit)
            self.assertIsNot(a.configs, b.configs)
            self.assertEqual(list(a.configs), list(b.configs))
            for key, cfg in a.configs.items():
                self.assertEqual(
                    b.configs[key],
                    dict(cfg, num_stages=1) if route == "rectangular" else cfg,
                )
        self.assertIs(
            pair[1].qsa_select_paged_prefill(), pair[1]._qsa_prefill_score_kernel
        )
        self.assertIsNot(
            pair[1]._prefill_logits.__globals__, pair[0]._prefill_logits.__globals__
        )

    def test_freeze_proof_rejects_unintended_changes(self):
        rect = dict(
            chunk=[7, 0, 1024],
            route="rectangular",
            constants=dict(num_stages=2, HEAD_GROUP=True, num_warps=4),
            ptx_sha256="base",
            cubin_sha256="base",
        )
        fallback = dict(
            chunk=[8, 7, 1024],
            route="fp8",
            constants=dict(num_stages=1),
            ptx_sha256="same",
            cubin_sha256="same",
        )
        changed = copy.deepcopy(rect)
        changed["constants"]["num_stages"] = 1
        records = dict(
            baseline=[rect, fallback], candidate=[changed, copy.deepcopy(fallback)]
        )
        with patch.object(probe, "FREEZE", return_value=records):
            output = probe.freeze_records(None)
            self.assertTrue(output["candidate"][0]["stage_depth_changed"])
            self.assertTrue(output["candidate"][1]["original_binary_verified"])
            output["candidate"][0]["constants"]["num_warps"] = 8
            with self.assertRaisesRegex(RuntimeError, "more than pipeline"):
                probe.freeze_records(None)
            output["candidate"][0]["constants"]["num_warps"] = 4
            output["candidate"][1]["cubin_sha256"] = "changed"
            with self.assertRaisesRegex(RuntimeError, "original binary"):
                probe.freeze_records(None)

    def test_boundary_evidence_distinguishes_grouped_and_control(self):
        checks = [
            dict(dtype="torch.bfloat16", tile=32, warps=8),
            dict(dtype="torch.bfloat16", tile=64, warps=4),
            dict(dtype="torch.float8_e4m3fn", tile=64, warps=4),
        ]
        with patch.object(probe, "BOUNDARIES", return_value=checks):
            output = probe.boundary_checks(None, None, None, None)
        self.assertEqual([v["candidate_num_stages"] for v in output], [1, 1, 2])
        self.assertEqual([v["head_group"] for v in output], [False, True, False])
        self.assertTrue(output[-1]["original_binary_fallback_verified"])

    def test_cli_child_handoff_and_restoration(self):
        from tools import benchmark_qsa_prefill_triton_stages as bench
        from tools import qsa_prefill_flagtree_ab as profile

        old = (probe.base.__file__, probe.base.make_pair, probe.base.manifest)

        def run(options):
            self.assertEqual(len(options.cases), 6)
            self.assertEqual(options.rounds, 2)
            self.assertEqual(
                Path(probe.base.__file__).resolve(), Path(probe.__file__).resolve()
            )
            self.assertIs(probe.base.make_pair, probe.make_pair)
            self.assertIs(probe.base.freeze_records, probe.freeze_records)
            self.assertIs(probe.base.boundary_checks, probe.boundary_checks)
            self.assertIs(probe.base.manifest, probe.manifest)
            self.assertTrue(probe.EXTRA_SOURCES <= set(bench.SOURCES))
            return 0

        with patch.object(
            sys,
            "argv",
            [probe.__file__, "--report-dir", "/tmp/r", "--baseline-report", "/tmp/b"],
        ), patch.object(profile, "configure"), patch.object(
            probe, "coordinate", side_effect=run
        ):
            self.assertEqual(probe.main(), 0)
        self.assertEqual(
            (probe.base.__file__, probe.base.make_pair, probe.base.manifest), old
        )

    def test_coordinator_groups_control_separately_and_renames_summary(self):
        m = probe.manifest()
        cases = [
            dict(
                case=v["case"]["name"],
                status="passed",
                stages={
                    k: dict(speedup=1.1 if i < 5 else 1.0)
                    for k in ("score", "topk", "pipeline")
                },
            )
            for i, v in enumerate(m["cases"])
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "head-group-summary.json"
            path.write_text(json.dumps(dict(status="passed", cases=cases)))
            with patch.object(probe.base, "coordinate", return_value=0):
                self.assertEqual(
                    probe.coordinate(types.SimpleNamespace(report_dir=folder)), 0
                )
            self.assertFalse(path.exists())
            s = json.loads((Path(folder) / "stage-depth-summary.json").read_text())
            self.assertEqual(s["cases"], cases)
            self.assertEqual(s["group_aggregates"]["changed_bf16"]["count"], 5)
            self.assertAlmostEqual(s["group_aggregates"]["changed_bf16"]["score"], 1.1)
            self.assertEqual(
                s["group_aggregates"]["unchanged_fp8_control"]["score"], 1.0
            )

    def test_base_coordinator_rejects_old_code_report_before_toolchain(self):
        with tempfile.TemporaryDirectory() as folder:
            previous = Path(folder) / "old"
            previous.mkdir()
            report = Path(folder) / "report"
            report.mkdir()
            (previous / "stage-summary.json").write_text(
                json.dumps(
                    dict(
                        status="passed",
                        manifest=dict(
                            git_head=dict(
                                stdout="55a13f673a5b54a0d430a124f0e04ec5e2c155cf"
                            )
                        ),
                    )
                )
            )
            with probe.scoped():
                code = probe.base.coordinate(
                    types.SimpleNamespace(
                        report_dir=str(report), baseline_report=str(previous)
                    )
                )
            self.assertEqual(code, 1)
            s = json.loads((report / "head-group-summary.json").read_text())
            self.assertIn("pinned code commit", s["error"])


if __name__ == "__main__":
    unittest.main()
