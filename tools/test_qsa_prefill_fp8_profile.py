# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""FP8-only fixtures and unchanged collector/child/provenance gates."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_fp8_profile as profile


class FP8ProfileTests(unittest.TestCase):
    def test_seven_real_fp8_chunks_keep_main_and_tail_separate(self):
        data = profile.manifest()
        self.assertEqual([len(v["traces"]) for v in data["cases"]], [1, 2, 4])
        self.assertTrue(
            all(v["case"]["dtype"] == "float8_e4m3fn" for v in data["cases"])
        )
        tail = data["cases"][1]["traces"]
        self.assertEqual(tail[1]["rows"], 113)
        self.assertEqual([v["constants"]["TILE_R"] for v in tail], [64, 8])
        self.assertEqual(data["cases"][0]["traces"][0]["constants"]["num_stages"], 1)
        self.assertEqual(
            data["production_sha256"], profile.collector.stage.PRODUCTION_SHA
        )
        self.assertEqual(
            profile.verify_source()["fp8_profile_manifest_sha256"], profile.MANIFEST_SHA
        )

    def test_manifest_tampering_and_bf16_substitution_are_rejected(self):
        data = profile.manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for change in ("hash", "dtype", "geometry"):
                value = copy.deepcopy(data)
                if change == "dtype":
                    value["cases"][0]["case"]["dtype"] = "bfloat16"
                elif change == "geometry":
                    value["cases"][1]["traces"][1]["rows"] = 112
                path.write_text(json.dumps(value))
                with patch.object(profile, "MANIFEST", path), patch.object(
                    profile,
                    "MANIFEST_SHA",
                    "wrong" if change == "hash" else profile.collector.base.sha(path),
                ):
                    with self.assertRaises(RuntimeError):
                        profile.manifest()

    def test_adapter_reuses_original_collector_and_dispatches_children_to_itself(self):
        original = profile.collector.CASES
        worker = profile.collector.worker
        coordinate = profile.collector.coordinate
        capture = profile.collector.capture_sequence

        def selected():
            self.assertEqual(profile.collector.CASES, profile.CASES)
            self.assertIs(profile.collector.manifest, profile.manifest)
            self.assertIs(profile.collector.verify_source, profile.verify_source)
            self.assertEqual(Path(profile.collector.__file__), Path(profile.__file__))
            self.assertIs(profile.collector.worker, worker)
            self.assertIs(profile.collector.coordinate, coordinate)
            self.assertIs(profile.collector.capture_sequence, capture)
            self.assertTrue(
                profile.EXTRA_SOURCES.issubset(profile.collector.EXTRA_SOURCES)
            )
            return 0

        with patch.object(profile.collector, "main", side_effect=selected):
            self.assertEqual(profile.main(), 0)
        self.assertEqual(profile.collector.CASES, original)

    def test_offline_display_does_not_initialize_cuda_or_a_compiler(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", ["profile", "--show-report", directory]
        ), patch(
            "tools.qsa_prefill_flagtree_ab.configure",
            side_effect=AssertionError("GPU configured"),
        ):
            self.assertEqual(profile.main(), 0)
        runner = (
            profile.ROOT / "tools/run_qsa_prefill_fp8_profile_h800.sh"
        ).read_text()
        self.assertIn('--script "$qsa_repo/tools/qsa_prefill_fp8_profile.py"', runner)
        self.assertIn('exit "$qsa_status"', runner)
        self.assertIn("qsa_name=", runner)
        self.assertIn("回传失败；无需重测", runner)


if __name__ == "__main__":
    unittest.main()
