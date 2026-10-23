# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Restricted routing, original-binary fallback and child handoff contracts."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_head_group_gated as probe


class GatedHeadGroupTests(unittest.TestCase):
    def test_exact_upstream12_and_mixed_tail_config(self):
        data = probe.manifest()
        self.assertEqual(len(data["cases"]), 12)
        probe.verify_source()
        mixed = next(
            c
            for c in data["cases"]
            if c["case"]["name"] == "perf_n1_q16384_p51200_bfloat16"
        )
        self.assertEqual(
            [t["constants"]["K_TILES"] for t in mixed["traces"]], [16, 16, 4]
        )
        active = [
            c["case"]["name"]
            for c in data["cases"]
            if any(
                probe.eligible(
                    dict(t["constants"], IS_FP8=False, NUM_HEADS=4, HEAD_DIM=128)
                )
                for t in c["traces"]
            )
        ]
        self.assertEqual(active, [probe.REPEAT_CASE])

    def test_gate_is_metadata_only_and_rejects_other_regimes(self):
        named = dict(IS_FP8=False, NUM_HEADS=4, HEAD_DIM=128, TILE_R=64, num_warps=4)
        self.assertTrue(probe.eligible(named))
        for key, value in (
            ("IS_FP8", True),
            ("IS_FP8", None),
            ("NUM_HEADS", 3),
            ("HEAD_DIM", 96),
            ("TILE_R", 32),
            ("num_warps", 8),
        ):
            self.assertFalse(probe.eligible(dict(named, **{key: value})))
        self.assertFalse(probe.eligible({}))

    def test_actual_jit_delegate_keeps_every_argument_and_config(self):
        calls = []

        class JIT:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, grid):
                def invoke(*args, **kwargs):
                    calls.append((self.name, grid, args, kwargs))
                    return self.name

                return invoke

        names = [
            "num_rows",
            "query_offset",
            "logits_width",
            "IS_FP8",
            "NUM_HEADS",
            "HEAD_DIM",
        ]
        args = (7, 13, 1088, False, 4, 128)
        for route, config, expected in (
            ("rectangular", dict(TILE_R=64, num_warps=4), "head"),
            ("rectangular", dict(TILE_R=32, num_warps=4), "original"),
            ("rectangular", dict(TILE_R=64, num_warps=8), "original"),
            ("fp8", dict(TILE_R=64, num_warps=4), "original"),
            ("compact", dict(TILE_R=64, num_warps=4), "original"),
        ):
            routed = probe.RoutedJIT(JIT("original"), JIT("head"), names, route)
            self.assertEqual(routed[(1, 2, 3)](*args, **config), expected)
            self.assertEqual(calls[-1], (expected, (1, 2, 3), args, config))
            self.assertEqual(
                routed.decisions[(7, 13, 1088)]["active"], expected == "head"
            )

    def test_fallback_binary_proof_and_active_record(self):
        records = dict(
            baseline=[
                dict(chunk=[7, 0, 1088], ptx_sha256="same", cubin_sha256="same"),
                dict(chunk=[9, 7, 1088], ptx_sha256="base", cubin_sha256="base"),
            ],
            candidate=[
                dict(chunk=[7, 0, 1088], ptx_sha256="same", cubin_sha256="same"),
                dict(chunk=[9, 7, 1088], ptx_sha256="new", cubin_sha256="new"),
            ],
        )
        module = types.SimpleNamespace()
        for index, (name, _) in enumerate(probe.base.ENTRIES):
            decisions = (
                {(7, 0, 1088): dict(active=False), (9, 7, 1088): dict(active=True)}
                if index == 0
                else {}
            )
            setattr(
                module,
                name,
                types.SimpleNamespace(jit=types.SimpleNamespace(decisions=decisions)),
            )
        with patch.object(probe, "FREEZE_RECORDS", return_value=records):
            result = probe.freeze_records((None, module))
            self.assertTrue(result["candidate"][0]["original_binary_verified"])
            self.assertTrue(result["candidate"][1]["head_group_gate"]["active"])
            result["candidate"][0]["cubin_sha256"] = "changed"
            with self.assertRaisesRegex(RuntimeError, "original binary"):
                probe.freeze_records((None, module))

    def test_promising_rounds_do_not_change_dispatch_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, expected in (
                (probe.REPEAT_CASE, 4),
                (probe.manifest()["cases"][0]["case"]["name"], 2),
            ):
                options = types.SimpleNamespace(
                    worker_case=name, rounds=2, promising_rounds=4, report_dir=folder
                )
                observed = []

                def run(o):
                    observed.append(o.rounds)
                    (Path(folder) / (name + ".json")).write_text('{"status":"passed"}')
                    return 0

                with patch.object(probe.base, "worker", side_effect=run):
                    self.assertEqual(probe.worker(options), 0)
                self.assertEqual(observed, [expected])
                self.assertEqual(options.rounds, 2)
                self.assertEqual(
                    json.loads((Path(folder) / (name + ".json")).read_text())["rounds"],
                    expected,
                )

    def test_cli_coordinator_handoff_keeps_restricted_child_script(self):
        from tools import benchmark_qsa_prefill_triton_stages as bench
        from tools import qsa_prefill_flagtree_ab as profile

        original_file = probe.base.__file__

        def coordinate(options):
            self.assertEqual(
                Path(probe.base.__file__).resolve(), Path(probe.__file__).resolve()
            )
            self.assertIs(probe.base.make_pair, probe.make_pair)
            self.assertIs(probe.base.freeze_records, probe.freeze_records)
            self.assertIs(probe.base.manifest, probe.manifest)
            self.assertTrue(probe.EXTRA_SOURCES <= set(bench.SOURCES))
            self.assertEqual(len(options.cases), 12)
            self.assertEqual(options.promising_rounds, 4)
            return 0

        with patch.object(
            sys,
            "argv",
            [
                probe.__file__,
                "--report-dir",
                "/tmp/report",
                "--baseline-report",
                "/tmp/baseline",
            ],
        ), patch.object(profile, "configure"), patch.object(
            probe.base, "coordinate", side_effect=coordinate
        ):
            self.assertEqual(probe.main(), 0)
        self.assertEqual(probe.base.__file__, original_file)
        self.assertIs(probe.base.make_pair, probe.MAKE_PAIR)


if __name__ == "__main__":
    unittest.main()
