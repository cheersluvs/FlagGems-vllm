# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU checks for isolated compat extraction, verification and fresh exec."""

import hashlib
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import with_cuda_compat134 as boot


class CompatTests(unittest.TestCase):
    def test_verified_package_is_extracted_without_installing_and_execs_fresh_python(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = root / "report"
            report.mkdir()
            prefix = root / "compat"
            prefix.mkdir()
            calls = []

            def extract(command, **kw):
                calls.append(command)
                library = Path(command[-1]) / "usr/local/cuda-13.4/compat/libcuda.so.1"
                library.parent.mkdir(parents=True)
                library.write_bytes(b"library")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch.object(
                boot.tempfile, "mkdtemp", return_value=str(prefix)
            ), patch.object(
                boot, "download", side_effect=lambda path: path.write_bytes(b"package")
            ), patch.object(
                boot, "PACKAGE_SHA256", hashlib.sha256(b"package").hexdigest()
            ), patch.object(
                boot.subprocess, "run", side_effect=extract
            ), patch.object(
                boot.os, "execve"
            ) as execute, patch.dict(
                os.environ, LD_LIBRARY_PATH="/old/compat"
            ):
                self.assertEqual(
                    boot.run(
                        Path("/repo/diagnose.py"),
                        report,
                        ["--flagtree", "/opt/flagtree"],
                    ),
                    0,
                )
                self.assertEqual(os.environ["LD_LIBRARY_PATH"], "/old/compat")
            self.assertEqual(calls[0][:2], ["dpkg-deb", "--extract"])
            metadata = json.loads((report / "compat134.json").read_text())
            args = execute.call_args.args
            self.assertEqual(args[1][1:3], ["-I", "/repo/diagnose.py"])
            self.assertEqual(args[1][-2:], ["--flagtree", "/opt/flagtree"])
            self.assertEqual(
                args[2]["LD_LIBRARY_PATH"],
                metadata["compat_directory"] + ":/old/compat",
            )
            self.assertEqual(
                metadata["libcuda_sha256"], hashlib.sha256(b"library").hexdigest()
            )

    def test_checksum_failure_is_reported_without_extract_or_exec(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prefix = root / "compat"
            prefix.mkdir()
            with patch.object(
                boot.tempfile, "mkdtemp", return_value=str(prefix)
            ), patch.object(
                boot, "download", side_effect=lambda path: path.write_bytes(b"wrong")
            ), patch.object(
                boot.subprocess, "run"
            ) as extract, patch.object(
                boot.os, "execve"
            ) as execute:
                self.assertEqual(boot.run(Path("/repo/diagnose.py"), root, []), 1)
                extract.assert_not_called()
                execute.assert_not_called()
            state = json.loads((root / "ptxas-summary.json").read_text())
            self.assertEqual(state["status"], "compat_preparation_failed")
            self.assertIn("SHA256 mismatch", state["error"])


if __name__ == "__main__":
    unittest.main()
