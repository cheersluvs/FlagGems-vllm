# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Pinned profiler extraction without installing packages or invoking CUDA."""

import hashlib
import io
import json
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_ncu_portable as portable


class PortableTests(unittest.TestCase):
    def archive(self, folder, *, unsafe=False):
        path = folder / "tool.tar.xz"
        with tarfile.open(path, "w:xz") as package:
            info = tarfile.TarInfo("tool/ncu")
            if unsafe:
                info.type = tarfile.SYMTYPE
                info.linkname = "/outside/ncu"
                package.addfile(info)
            else:
                data = b"#!/bin/sh\necho profiler-version\n"
                info.size, info.mode = len(data), 0o755
                package.addfile(info, io.BytesIO(data))
        return path

    def test_pinned_archive_verification_and_extract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = self.archive(root)
            with patch.object(
                portable, "SHA256", portable.digest(archive)
            ), patch.object(portable, "SIZE", archive.stat().st_size):
                self.assertTrue(portable.verified(archive))
                executable = portable.extract(archive, root / "extract")
                self.assertEqual(executable.name, "ncu")
                archive.write_bytes(b"corrupt")
                with self.assertRaisesRegex(RuntimeError, "unverified"):
                    portable.extract(archive, root / "bad")

    def test_unsafe_archive_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = self.archive(root, unsafe=True)
            with patch.object(
                portable, "SHA256", portable.digest(archive)
            ), patch.object(
                portable, "SIZE", archive.stat().st_size
            ), self.assertRaises(
                tarfile.FilterError
            ):
                portable.extract(archive, root / "extract")

    def test_local_archive_preparation_records_version_and_never_downloads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = self.archive(root)
            with patch.object(
                portable, "SHA256", portable.digest(archive)
            ), patch.object(portable, "SIZE", archive.stat().st_size), patch.object(
                portable.platform, "system", return_value="Linux"
            ), patch.object(
                portable.platform, "machine", return_value="x86_64"
            ), patch.object(
                portable, "download"
            ) as download, patch.object(
                portable.subprocess,
                "run",
                return_value=types.SimpleNamespace(
                    returncode=0, stdout="ncu version", stderr=""
                ),
            ):
                executable = portable.prepare(root, str(archive))
            download.assert_not_called()
            self.assertTrue(Path(executable).is_file())
            state = json.loads((root / "ncu-preparation.json").read_text())
            self.assertEqual(state["status"], "ready")
            self.assertEqual(state["archive_sha256"], portable.digest(archive))
            self.assertEqual(state["version_check"]["stdout"], "ncu version")

    def test_size_and_digest_both_required(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "archive"
            path.write_bytes(b"bad")
            with patch.object(portable, "SIZE", 3):
                self.assertFalse(portable.verified(path))
            with patch.object(portable, "SHA256", hashlib.sha256(b"bad").hexdigest()):
                self.assertFalse(portable.verified(path))


if __name__ == "__main__":
    unittest.main()
