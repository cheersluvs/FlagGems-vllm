# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Controlled width invariants and diagnostic receipt/dispatch isolation."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import qsa_prefill_fp8_dot_width as p


class WidthProbeTests(unittest.TestCase):
    def test_only_visible_shift_changes_across_four_widths(self):
        first = p.visible_plan(p.WIDTHS[0])
        for width in p.WIDTHS:
            values = p.visible_plan(width)
            self.assertEqual(len(values), p.ROWS)
            self.assertEqual(max(values), width)
            self.assertEqual(min(values), width - (p.ROWS - 1) // 4)
            self.assertTrue(all(b - a in (0, 1) for a, b in zip(values, values[1:])))
            self.assertEqual(
                {v - a for v, a in zip(values, first)}, {width - p.WIDTHS[0]}
            )
            self.assertGreater(min(values), 512)
            self.assertLessEqual(width, 152 * 196)
            self.assertEqual(width % 64, 0)
            self.assertGreaterEqual(512 * 1024 * 1024 // (width * 4), p.ROWS)
        with self.assertRaises(ValueError):
            p.visible_plan(10000)

    def test_adapter_preserves_child_entry_and_selected_seed_case(self):
        original_worker = p.full.WORKER
        original_coordinate = p.full.COORDINATE

        def inspect():
            self.assertIs(p.full.WORKER, p.worker)
            self.assertIs(p.full.COORDINATE, p.coordinate)
            self.assertEqual(Path(p.full.__file__), Path(p.__file__))
            self.assertIn("tools/qsa_prefill_fp8_dot_width.py", p.runtime.EXTRA)
            self.assertEqual(sys.argv[-2:], ["--cases", p.CASE])
            return 0

        with patch.object(sys, "argv", ["width.py"]), patch.object(
            p.full, "main", inspect
        ):
            self.assertEqual(p.main(), 0)
        self.assertIs(p.full.WORKER, original_worker)
        self.assertIs(p.full.COORDINATE, original_coordinate)

    def test_coordinator_does_not_publish_last_width_as_full_scan_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "fp8-dot-runtime-summary.json"
            receipt.write_text(
                json.dumps(dict(status="passed", aggregate=dict(score=9.0), cases=[]))
            )
            path = root / "fp8-dot-width-summary.json"
            path.write_text(
                json.dumps(
                    dict(status="passed", widths=[dict(width=w) for w in p.WIDTHS])
                )
            )
            with patch.object(p, "COORDINATE", lambda options: 0):
                self.assertEqual(p.coordinate(SimpleNamespace(report_dir=tmp)), 0)
            value = json.loads(path.read_text())
            self.assertEqual(len(value["widths"]), 4)
            self.assertNotIn("aggregate", value["coordination"])
            self.assertFalse(receipt.exists())


if __name__ == "__main__":
    unittest.main()
