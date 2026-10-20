# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Exercise isolated stock-Triton execution and automatic upload without a GPU."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

RUNNER = Path(__file__).with_name("run_qsa_prefill_triton_ab_h800.sh").resolve()


class RunnerTests(unittest.TestCase):
    runner = RUNNER
    runner_args = ()

    def test_success_and_failures_upload_reports_and_preserve_status(self):
        for status in (0, 1, 2):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                binary = root / "bin"
                binary.mkdir()
                python = binary / "fake-python"
                python.write_text(
                    """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
a=sys.argv[1:]
assert a[0]=='-I'
assert a[a.index('--mode')+1]=='kernel'
assert a[a.index('--suite')+1]=='upstream-prefill'
if a[1].endswith('with_cuda_compat134.py'):
    assert a[a.index('--script')+1].endswith('qsa_prefill_flagtree_ab.py')
else:
    assert os.environ['FLAGGEMS_FORCE_TLE']=='0'
keys=('PYTHONPATH','PYTHONHOME','QSA_TRITON_RUNTIME_PATH',
      'QSA_NATIVE_REFERENCE_PYTHON','CUDA_LAUNCH_BLOCKING')
assert not any(os.environ.get(k) for k in keys)
p=Path(a[a.index('--report-dir')+1])
if os.environ['MOCK_STATUS']!='2':
    (p/'stage-summary.json').write_text(json.dumps({'mock':True}))
    (p/'triton-stage-ab.tsv').write_text('score/topk/pipeline')
print('stock runtime test output')
sys.exit(int(os.environ['MOCK_STATUS']))
"""
                )
                git = binary / "git"
                git.write_text(
                    """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
a=sys.argv[1:]
if a[:1]==['-C']:
    directory=a[1];a=a[2:]
with open(os.environ['MOCK_GIT_LOG'],'a') as f:
    f.write(json.dumps({'repo':directory,'args':a})+'\\n')
if a[:2]==['rev-parse','HEAD']: print('0'*40)
"""
                )
                python.chmod(0o755)
                git.chmod(0o755)
                env = os.environ.copy()
                env.update(
                    PATH=str(binary) + os.pathsep + env["PATH"],
                    TMPDIR=folder,
                    QSA_STOCK_PYTHON=str(python),
                    MOCK_STATUS=str(status),
                    MOCK_GIT_LOG=str(root / "git.jsonl"),
                    QSA_TRITON_RUNTIME_PATH="/opt/flagtree",
                    PYTHONPATH="/opt/flagtree",
                    QSA_NATIVE_REFERENCE_PYTHON="must clear",
                    CUDA_LAUNCH_BLOCKING="1",
                )
                result = subprocess.run(
                    ["bash", str(self.runner), *self.runner_args],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(
                    result.returncode, status, result.stdout + result.stderr
                )
                self.assertIn("报告回传完成。", result.stdout)
                logs = [
                    json.loads(line)
                    for line in (root / "git.jsonl").read_text().splitlines()
                ]
                self.assertTrue(any(row["args"][0] == "push" for row in logs))
                tree = Path(
                    next(
                        row["args"][4]
                        for row in logs
                        if row["args"][:2] == ["worktree", "add"]
                    )
                )
                files = {p.name: p for p in tree.rglob("*") if p.is_file()}
                self.assertIn("stage-summary.json", files)
                if self.runner_args:
                    self.assertIn(
                        "flagtree134-shared-", str(files["stage-summary.json"])
                    )
                self.assertIn(
                    "stock runtime test output", files["runner.txt"].read_text()
                )
                self.assertIn(
                    f"benchmark_exit_code={status}", files["run-status.txt"].read_text()
                )
                if status == 2:
                    self.assertEqual(
                        json.loads(files["stage-summary.json"].read_text())["status"],
                        "failed",
                    )


class FlagTreeRunnerTests(RunnerTests):
    runner = RUNNER.with_name("run_qsa_prefill_flagtree_ab_h800.sh")


class SharedFlagTreeRunnerTests(FlagTreeRunnerTests):
    runner_args = ("--shared-gpu-exploratory",)


if __name__ == "__main__":
    unittest.main()
