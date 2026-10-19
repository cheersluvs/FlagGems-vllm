# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU checks for fixed-PTX evidence, tool extraction and isolated orchestration."""

import copy
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import validate_qsa_ptxas_h800 as probe


class EvidenceTests(unittest.TestCase):
    def probes(self):
        values = {
            name: dict(returncode=0, state=dict(passed=True)) for name in probe.PROBES
        }
        values["flagtree-old"] = dict(
            returncode=1,
            state=dict(passed=False, error="CUDA illegal instruction"),
            artifact=dict(
                async_l2_hint_count=4, fractional_policy_count=2, ptx_sha256="original"
            ),
        )
        values["identical-ptx-new"]["state"].update(
            ptx_edits=0,
            ptx_intervention="original",
            ptx_sha256="original",
            original_ptx_sha256="original",
            cubin_sha256="new",
        )
        values["new-cubin-stock"]["state"]["cubin_sha256"] = "new"
        values["fresh-jit-new"]["artifact"] = dict(
            async_l2_hint_count=4, fractional_policy_count=2
        )
        return values

    def test_only_configuration_success_accepts_the_claim(self):
        values = self.probes()
        self.assertEqual(probe.assess(values), "config_only_validated")
        values["fresh-jit-new"]["returncode"] = 1
        self.assertEqual(
            probe.assess(values), "assembler_fix_validated_fresh_jit_failed"
        )
        values["identical-ptx-new"]["returncode"] = 1
        self.assertEqual(probe.assess(values), "new_assembler_not_validated")

    def test_no_false_acceptance_for_missing_reproducer_or_edited_artifacts(self):
        for kind in (
            "stock",
            "old_pass",
            "no_policy",
            "edited_ptx",
            "different_binary",
            "compile_error",
        ):
            with self.subTest(kind=kind):
                v = self.probes()
                if kind == "stock":
                    v["stock"]["returncode"] = 1
                if kind == "old_pass":
                    v["flagtree-old"].update(returncode=0, state=dict(passed=True))
                if kind == "no_policy":
                    v["flagtree-old"]["artifact"]["async_l2_hint_count"] = 0
                if kind == "edited_ptx":
                    v["identical-ptx-new"]["state"]["ptx_edits"] = 1
                if kind == "different_binary":
                    v["new-cubin-stock"]["state"]["cubin_sha256"] = "other"
                if kind == "compile_error":
                    v["flagtree-old"]["state"]["error"] = "compile error"
                self.assertNotEqual(probe.assess(v), "config_only_validated")

    def test_extract_only_verified_regular_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "tools.tar.xz"
            target = root / "ptxas"
            with tarfile.open(archive, "w:xz") as tar:
                member = tarfile.TarInfo("package/bin/ptxas")
                member.size = 3
                tar.addfile(member, io.BytesIO(b"ELF"))
                other = tarfile.TarInfo("../../outside")
                other.size = 3
                tar.addfile(other, io.BytesIO(b"bad"))
            probe.extract_tool(archive, target, "ptxas", probe.shared.sha(archive))
            self.assertEqual(target.read_bytes(), b"ELF")
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                probe.extract_tool(archive, target, "ptxas", "wrong")
            with tarfile.open(archive, "w:xz") as tar:
                member = tarfile.TarInfo("package/bin/ptxas")
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                tar.addfile(member)
            with self.assertRaisesRegex(RuntimeError, "regular"):
                probe.extract_tool(archive, target, "ptxas", probe.shared.sha(archive))

    def test_actual_reassembly_keeps_ptx_abi_and_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            origin = root / "old"
            target = root / "new"
            origin.mkdir()
            target.mkdir()
            (origin / "kernel.ptx").write_text(
                ".version 8.7\ncreatepolicy.fractional.L2::evict_first.b64 %rd1, 1.0;\n"
            )
            (origin / "kernel.json").write_text('{"name":"qsa"}')
            (origin / "kernel.cubin").write_bytes(b"old")
            files = {p.name: probe.shared.sha(p) for p in origin.iterdir()}
            info = dict(
                files=files,
                assembler_commands=[
                    [
                        "/old/ptxas",
                        "-arch=sm_90a",
                        "--warn-on-spills",
                        str(origin / "kernel.ptx"),
                        "-o",
                        str(origin / "kernel.cubin"),
                    ]
                ],
            )
            probe.shared.write_json(origin / "replay.json", info)

            def assemble(command, **kw):
                self.assertEqual(
                    command[:3], ["/new/ptxas", "-arch=sm_90a", "--warn-on-spills"]
                )
                Path(command[command.index("-o") + 1]).write_bytes(b"new")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch.object(probe.shared.subprocess, "run", side_effect=assemble):
                evidence = probe.shared.reassemble(origin, target, "/new/ptxas")
            self.assertEqual(
                (origin / "kernel.ptx").read_bytes(),
                (target / "kernel.ptx").read_bytes(),
            )
            self.assertEqual(
                (origin / "kernel.json").read_bytes(),
                (target / "kernel.json").read_bytes(),
            )
            self.assertEqual(evidence["ptx_edits"], 0)
            self.assertEqual(evidence["ptx_intervention"], "original")

    def test_child_environment_clears_overrides_and_uses_unique_cache(self):
        with patch.dict(
            os.environ,
            dict(
                QSA_TRITON_RUNTIME_PATH="/bad",
                TRITON_PTXAS_PATH="/bad",
                TRITON_MOCK_PTX_VERSION="13.4",
                PYTHONPATH="/bad",
            ),
        ):
            env = probe.environment(Path("/report"), None, "stock")
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("QSA_TRITON_RUNTIME_PATH", env)
            self.assertNotIn("TRITON_PTXAS_PATH", env)
            self.assertNotIn("TRITON_MOCK_PTX_VERSION", env)
            fresh = probe.environment(
                Path("/report"), "/opt/flagtree", "fresh", "/new/ptxas"
            )
            self.assertEqual(fresh["TRITON_PTXAS_PATH"], "/new/ptxas")
            self.assertNotEqual(fresh["TRITON_CACHE_DIR"], env["TRITON_CACHE_DIR"])


class OrchestrationTests(unittest.TestCase):
    def test_automatic_selection_skips_busy_devices_and_reports_no_idle_gpu(self):
        first = "GPU-00000000-0000-0000-0000-000000000001"
        second = "GPU-00000000-0000-0000-0000-000000000002"
        inventory = dict(
            returncode=0, stdout=f"{first}, NVIDIA H800\n{second}, NVIDIA H800\n"
        )
        for available in (True, False):
            with self.subTest(available=available), patch.dict(
                os.environ, CUDA_VISIBLE_DEVICES="auto"
            ):
                verdicts = [
                    dict(status="rejected"),
                    dict(status="passed" if available else "rejected"),
                ]
                with patch.object(
                    probe.support, "command", return_value=inventory
                ), patch.object(probe.guard, "idle", side_effect=verdicts):
                    summary = {}
                    self.assertEqual(probe.select_gpu(summary), available)
                self.assertEqual(len(summary["gpu_selection_checks"]), 2)
                if available:
                    self.assertEqual(summary["gpu_uuid"], second)
                    self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], second)
                else:
                    self.assertTrue(summary["environment_rejected"])

    def exercise(self, root, rejected=False):
        runtime = root / "flagtree"
        (runtime / "triton").mkdir(parents=True)
        (runtime / "triton/__init__.py").touch()
        report = root / "report"
        report.mkdir()
        options = types.SimpleNamespace(flagtree=str(runtime))
        calls = []

        def job(folder, name, command, env):
            self.assertEqual(command[1], "-I")
            calls.append((name, command, env))
            self.assertNotIn("PYTHONPATH", env)
            if name == "prepare":
                return dict(name=name, returncode=0)
            destination = folder / name
            destination.mkdir()
            (destination / "replay.json").write_text("{}")
            return dict(name=name, returncode=0)

        def collect(folder, name, outcome):
            label = next(
                label
                for label in sorted(probe.PROBES, key=len, reverse=True)
                if name.endswith("-" + label)
            )
            return dict(copy.deepcopy(EvidenceTests().probes()[label]), name=name)

        def download(url, path):
            path.write_bytes(b"pinned")

        def gate(summary, *args):
            if rejected:
                summary["environment_rejected"] = True
            return not rejected

        tools = {name: dict(path="/new/" + name) for name in probe.TOOLS}
        with (
            patch.dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="GPU-00000000-0000-0000-0000-000000000000",
            ),
            patch.object(probe, "toolchain", return_value=tools),
            patch.object(probe, "download", side_effect=download),
            patch.object(
                probe.reference,
                "REFERENCE_SHA256",
                hashlib.sha256(b"pinned").hexdigest(),
            ),
            patch.object(probe.support, "idle_gpu_gate", side_effect=gate),
            patch.object(probe.shared, "run_job", side_effect=job),
            patch.object(probe, "collect", side_effect=collect),
            patch.object(probe, "check_identity"),
        ):
            code = probe.orchestrate(options, report)
        summary = json.loads((report / "ptxas-summary.json").read_text())
        return code, summary, calls

    def test_ten_cases_five_controls_and_isolated_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            code, summary, calls = self.exercise(Path(folder))
        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(len(summary["cases"]), 10)
        self.assertEqual(len(calls), 52)
        for name, command, env in calls:
            if name.endswith("-identical-ptx-new"):
                self.assertEqual(
                    command[command.index("--assembler") + 1], "/new/ptxas"
                )
                self.assertNotIn("TRITON_PTXAS_PATH", env)
            if name.endswith("-fresh-jit-new"):
                self.assertEqual(env["TRITON_PTXAS_PATH"], "/new/ptxas")

    def test_environment_rejection_keeps_summary_and_runs_no_child(self):
        with tempfile.TemporaryDirectory() as folder:
            code, summary, calls = self.exercise(Path(folder), True)
        self.assertEqual(code, 2)
        self.assertEqual(summary["status"], "environment_rejected")
        self.assertEqual(calls, [])


class RunnerTests(unittest.TestCase):
    def test_actual_shell_uploads_nested_reports_and_preserves_exit_status(self):
        runner = Path(probe.__file__).with_name("run_qsa_ptxas_h800.sh")
        for status in (0, 1, 2):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                binary = root / "bin"
                binary.mkdir()
                python = binary / "fake-python"
                python.write_text(
                    """#!/usr/bin/env python3
import os, sys
from pathlib import Path
if sys.argv[2] == '-':
    sys.argv = sys.argv[2:]
    exec(compile(sys.stdin.read(), '<report-copy>', 'exec'))
    sys.exit(0)
a=sys.argv[1:]
assert a[0]=='-I'
assert 'PYTHONPATH' not in os.environ
p=Path(a[a.index('--report-dir')+1])
if os.environ['MOCK_STATUS']!='2':
    (p/'ptxas-summary.json').write_text('{"status":"mock"}')
(p/'prefill/worker').mkdir(parents=True)
(p/'prefill/worker/kernel.sass.txt').write_text('sass')
(p/'prefill/caches/temp').mkdir(parents=True)
(p/'prefill/caches/temp/ignored.json').write_text('{}')
(p/'prefill/input.pt').write_bytes(b'tensor')
sys.exit(int(os.environ['MOCK_STATUS']))
"""
                )
                git = binary / "git"
                git.write_text(
                    """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
a=sys.argv[1:]
if a[:1]==['-C']: a=a[2:]
with open(os.environ['MOCK_GIT_LOG'],'a') as f:
    f.write(json.dumps(a)+'\\n')
if a[:2]==['rev-parse','HEAD']: print('0'*40)
"""
                )
                python.chmod(0o755)
                git.chmod(0o755)
                env = dict(
                    os.environ,
                    PATH=str(binary) + os.pathsep + os.environ["PATH"],
                    TMPDIR=folder,
                    QSA_STOCK_PYTHON=str(python),
                    MOCK_STATUS=str(status),
                    MOCK_GIT_LOG=str(root / "git.jsonl"),
                    PYTHONPATH="/bad",
                )
                result = subprocess.run(
                    ["bash", str(runner)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(
                    result.returncode, status, result.stdout + result.stderr
                )
                self.assertIn("报告回传完成", result.stdout)
                commands = [
                    json.loads(line)
                    for line in (root / "git.jsonl").read_text().splitlines()
                ]
                self.assertTrue(any(c[0] == "push" for c in commands))
                tree = Path(
                    next(c[4] for c in commands if c[:2] == ["worktree", "add"])
                )
                files = [p for p in tree.rglob("*") if p.is_file()]
                self.assertTrue(any(p.name == "kernel.sass.txt" for p in files))
                self.assertTrue(any(p.name == "ptxas-summary.json" for p in files))
                self.assertFalse(
                    any(p.name in {"ignored.json", "input.pt"} for p in files)
                )


if __name__ == "__main__":
    unittest.main()
