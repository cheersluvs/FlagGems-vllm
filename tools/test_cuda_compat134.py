# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU checks for isolated compat extraction, verification and fresh exec."""

import hashlib
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import with_cuda_compat134 as boot


class CompatTests(unittest.TestCase):
    def setUp(self):
        candidates = patch.object(boot, "cache_candidates", return_value=[])
        candidates.start()
        self.addCleanup(candidates.stop)

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
            details = json.loads((root / "compat134-package.json").read_text())
            self.assertEqual(len(details["download_attempts"]), 3)
            self.assertTrue(all(e["bytes"] == 5 for e in details["download_attempts"]))
            self.assertTrue(
                all(
                    e["actual_sha256"] == hashlib.sha256(b"wrong").hexdigest()
                    for e in details["download_attempts"]
                )
            )

    def test_only_exact_verified_cached_package_is_copied_without_download(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bad, good, target = root / "bad.deb", root / "good.deb", root / "target.deb"
            bad.write_bytes(b"bad")
            good.write_bytes(b"package")
            with patch.object(
                boot, "cache_candidates", return_value=[bad, good]
            ), patch.object(
                boot, "PACKAGE_SHA256", hashlib.sha256(b"package").hexdigest()
            ), patch.object(
                boot, "download"
            ) as download:
                result = boot.prepare_package(target, root)
            download.assert_not_called()
            self.assertEqual(target.read_bytes(), b"package")
            self.assertEqual(good.read_bytes(), b"package")
            self.assertEqual(bad.read_bytes(), b"bad")
            self.assertEqual(result["source_kind"], "verified_local_cache")
            self.assertEqual(
                [e["status"] for e in result["cached_candidates"]],
                ["rejected", "verified"],
            )
            self.assertEqual(
                json.loads((root / "compat134-package.json").read_text()), result
            )

    def test_bad_download_then_verified_retry_records_both_observed_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payloads = iter((b"error page", b"package"))

            def download(path):
                path.write_bytes(next(payloads))
                return dict(http_status=200, content_type="application/octet-stream")

            with patch.object(boot, "download", side_effect=download), patch.object(
                boot, "PACKAGE_SHA256", hashlib.sha256(b"package").hexdigest()
            ):
                result = boot.prepare_package(root / "target.deb", root)
            self.assertEqual(result["source_kind"], "download")
            self.assertEqual(
                [e["status"] for e in result["download_attempts"]],
                ["failed", "verified"],
            )
            self.assertEqual([e["bytes"] for e in result["download_attempts"]], [10, 7])
            self.assertEqual(result["download_attempts"][0]["http_status"], 200)

    def test_changed_cache_during_copy_cannot_bypass_second_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            good, target = root / "good.deb", root / "target.deb"
            good.write_bytes(b"package")
            with patch.object(
                boot, "cache_candidates", return_value=[good]
            ), patch.object(
                boot, "PACKAGE_SHA256", hashlib.sha256(b"package").hexdigest()
            ), patch.object(
                boot.shutil,
                "copyfile",
                side_effect=lambda src, dst: dst.write_bytes(b"changed"),
            ), patch.object(
                boot, "download", side_effect=lambda path: path.write_bytes(b"package")
            ) as download:
                result = boot.prepare_package(target, root)
            self.assertEqual(download.call_count, 1)
            self.assertEqual(result["source_kind"], "download")
            self.assertIn("SHA256 mismatch", result["cached_candidates"][0]["error"])
            self.assertEqual(target.read_bytes(), b"package")

    def test_network_failure_retains_partial_download_size_and_never_extracts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prefix = root / "compat"
            prefix.mkdir()

            def download(path):
                path.write_bytes(b"partial")
                raise OSError("connection reset")

            with patch.object(
                boot.tempfile, "mkdtemp", return_value=str(prefix)
            ), patch.object(boot, "download", side_effect=download), patch.object(
                boot.subprocess, "run"
            ) as extract, patch.object(
                boot.os, "execve"
            ) as execute:
                self.assertEqual(boot.run(Path("/repo/diagnose.py"), root, []), 1)
            extract.assert_not_called()
            execute.assert_not_called()
            state = json.loads((root / "compat134-package.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(len(state["download_attempts"]), 3)
            self.assertTrue(
                all(
                    e["bytes"] == 7 and e["error"] == "connection reset"
                    for e in state["download_attempts"]
                )
            )

    def test_download_records_response_metadata_and_bytes_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            response = io.BytesIO(b"package")
            response.headers = {
                "Content-Type": "application/vnd.debian.binary-package",
                "Content-Length": "7",
            }
            response.status = 200
            response.geturl = lambda: boot.PACKAGE_URL
            path = Path(folder) / "target.deb"
            with patch.object(boot.urllib.request, "urlopen", return_value=response):
                metadata = boot.download(path)
            self.assertEqual(path.read_bytes(), b"package")
            self.assertEqual(metadata["http_status"], 200)
            self.assertEqual(metadata["content_length"], "7")
            self.assertEqual(metadata["response_url"], boot.PACKAGE_URL)


if __name__ == "__main__":
    unittest.main()
