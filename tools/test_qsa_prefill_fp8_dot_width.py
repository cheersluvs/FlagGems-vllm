# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Controlled width invariants and diagnostic receipt/dispatch isolation."""

import hashlib
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

    def test_point_report_preserves_all_runtime_provenance_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {
                name: (name + "\n/absolute/original/path\n").encode()
                for name in p.POINT_FILES
            }
            for name, data in payload.items():
                (root / name).write_bytes(data)
            for width in p.WIDTHS:
                point = root / ("width-" + str(width))
                p.prepare_point_report(root, point)
                for name, data in payload.items():
                    self.assertEqual((point / name).read_bytes(), data)
            (root / "compat134.json").unlink()
            missing = root / "missing"
            with self.assertRaises(FileNotFoundError):
                p.prepare_point_report(root, missing)
            self.assertFalse(missing.exists())

    def test_staged_point_is_accepted_by_existing_loaded_cuda_verifier(self):
        from tools import qsa_prefill_flagtree_ab as profile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "libcuda.so.615.test"
            library.write_bytes(b"fake library for CPU provenance test")
            digest = hashlib.sha256(library.read_bytes()).hexdigest()
            for name in p.POINT_FILES:
                (root / name).write_text("{}")
            (root / "compat134.json").write_text(
                json.dumps(dict(libcuda=str(library), libcuda_sha256=digest))
            )
            point = root / "width-8192"
            p.prepare_point_report(root, point)

            def api_version(pointer):
                pointer._obj.value = 13040
                return 0

            def path(value):
                if str(value) == "/proc/self/maps":
                    return SimpleNamespace(read_text=lambda: "mapping " + str(library))
                return Path(value)

            with patch.object(profile, "Path", path), patch.object(
                profile, "LIBCUDA_SHA256", digest
            ), patch.object(
                profile.ctypes,
                "CDLL",
                lambda name: SimpleNamespace(cuDriverGetVersion=api_version),
            ):
                receipt = profile.loaded_cuda(point)
                self.assertEqual(receipt["driver_api"], 13040)
                self.assertEqual(receipt["libcuda"], str(library.resolve()))
                library.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    profile.loaded_cuda(point)

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
