# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Exercise the actual run/upload shell with fake Python/Git, no network/GPU."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

RUNNER = Path(__file__).with_name("run_qsa_prefill_validation_ab_h800.sh").resolve()


class RunnerTests(unittest.TestCase):
    def exercise(
        self, status, baseline=None, kind="validation", guarded=False, kernel=False
    ):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binaries = root / "bin"
            binaries.mkdir()
            python = binaries / "fake-python"
            python.write_text(
                """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
a=sys.argv[1:]
flag={'host':'--host-ab', 'single-row':'--single-row-ab', 'validation':'--validation-ab'}[os.environ['MOCK_AB_KIND']]
assert flag in a and '--host-profile' in a
assert sum(f in a for f in ('--validation-ab','--host-ab','--single-row-ab')) == 1
assert ('--require-idle-gpu' in a) == (os.environ['MOCK_GUARDED']=='1')
if os.environ['MOCK_KERNEL']=='1':
    assert a[a.index('--mode')+1]=='kernel'
if os.environ.get('MOCK_BASELINE'):
    assert a[a.index('--validation-baseline')+1] == os.environ['MOCK_BASELINE']
p=Path(a[a.index('--report-dir')+1])
(p/'stage-summary.json').write_text(json.dumps({'mock':True}))
(p/'host-summary.tsv').write_text('mock host report')
if os.environ['MOCK_KERNEL']=='1':
    (p/'score-kernel-ab.tsv').write_text('mock score kernel speedup')
(p/'core_0.host-profile.txt').write_text('mock profile')
(p/'cache').mkdir()
(p/'cache/not-uploaded.txt').write_text('cache')
sys.exit(int(os.environ['MOCK_STATUS']))
"""
            )
            git = binaries / "git"
            git.write_text(
                """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
a=sys.argv[1:]
if a[0]=='-C':
    repository=a[1]
    a=a[2:]
with open(os.environ['MOCK_GIT_LOG'],'a') as f:
    f.write(json.dumps({'repo':repository,'args':a})+'\\n')
if a[:2]==['rev-parse','HEAD']:
    print('0'*40)
"""
            )
            python.chmod(0o755)
            git.chmod(0o755)
            env = os.environ.copy()
            env.update(
                PATH=str(binaries) + os.pathsep + env["PATH"],
                PYTHON=str(python),
                TMPDIR=folder,
                MOCK_STATUS=str(status),
                MOCK_GIT_LOG=str(root / "git.jsonl"),
                MOCK_BASELINE=baseline or "",
                MOCK_AB_KIND=kind,
                QSA_PREFILL_AB_KIND=kind,
                MOCK_GUARDED="1" if guarded else "0",
                MOCK_KERNEL="1" if kernel else "0",
            )
            result = subprocess.run(
                ["bash", str(RUNNER)]
                + (["--validation-baseline", baseline] if baseline else [])
                + (["--require-idle-gpu"] if guarded else [])
                + (["--mode", "kernel"] if kernel else []),
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, status, result.stdout + result.stderr)
            self.assertIn("结果分支：", result.stdout)
            logs = [
                json.loads(line)
                for line in (root / "git.jsonl").read_text().splitlines()
            ]
            self.assertTrue(any(row["args"][0] == "push" for row in logs))
            tree = next(
                row["args"][4] for row in logs if row["args"][:2] == ["worktree", "add"]
            )
            files = [p for p in Path(tree).rglob("*") if p.is_file()]
            self.assertEqual(
                {p.name for p in files},
                {
                    "stage-summary.json",
                    "host-summary.tsv",
                    "core_0.host-profile.txt",
                    "run-status.txt",
                }
                | ({"score-kernel-ab.tsv"} if kernel else set()),
            )
            saved = next(p for p in files if p.name == "run-status.txt")
            self.assertIn(f"benchmark_exit_code={status}", saved.read_text())

    def test_success_runs_and_uploads_all_report_types(self):
        self.exercise(0)

    def test_failed_benchmark_still_uploads_and_retains_failure_status(self):
        self.exercise(1)

    def test_stride_baseline_selection_is_forwarded_and_uploaded(self):
        self.exercise(0, baseline="overlap-retained")

    def test_host_mode_runs_and_uploads(self):
        for status in (0, 1):
            self.exercise(status, kind="host")

    def test_single_row_mode_runs_and_uploads(self):
        for status in (0, 1):
            self.exercise(status, kind="single-row")

    def test_guarded_run_forwards_gate_and_uploads_rejection_with_exit_two(self):
        for status in (0, 2):
            self.exercise(status, kind="single-row", guarded=True)

    def test_score_kernel_mode_forwards_and_uploads_success_or_failure(self):
        for status in (0, 1, 2):
            self.exercise(status, kind="single-row", guarded=True, kernel=True)


if __name__ == "__main__":
    unittest.main()
