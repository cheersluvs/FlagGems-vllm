# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Resume preserves old sources/runtime and never repeats independent timings."""

import hashlib
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_profile_resume as resume
from tools import qsa_prefill_profiling as profiling


def save(path, value):
    path.write_text(json.dumps(value))


class ResumeTests(unittest.TestCase):
    def fixture(self, root):
        code, original, output = (root / name for name in ("code", "original", "new"))
        for path in (code, original, output):
            path.mkdir()
        canonical = code / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
        canonical.parent.mkdir(parents=True)
        canonical.write_text("# immutable operator\n")
        reference = original / "qsa_indexer_vllm.py"
        reference.write_text("# pinned reference\n")
        sources = {
            name: dict(
                text=path.read_text(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for name, path in (
                (str(canonical.relative_to(code)), canonical),
                ("pinned_vllm/qsa_indexer.py", reference),
            )
        }
        save(original / "sources.json", sources)
        (original / "runner.txt").write_text("original run log")
        (output / "runner.txt").write_text("current tee log")
        library, ptxas = (root / name for name in ("libcuda.so", "ptxas"))
        library.write_bytes(b"driver")
        ptxas.write_bytes(b"assembler")
        save(
            original / "compat134.json",
            dict(
                libcuda=str(library),
                libcuda_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                child_ld_library_path=str(root),
            ),
        )
        save(
            original / "flagtree134-tools.json",
            dict(
                toolchain=dict(
                    ptxas=dict(
                        path=str(ptxas),
                        sha256=hashlib.sha256(ptxas.read_bytes()).hexdigest(),
                    )
                )
            ),
        )
        summary = dict(
            status="passed",
            manifest=dict(
                options=dict(candidate="query-stage-joint", flagtree="/opt/flagtree"),
                git_head=dict(stdout="old-commit"),
            ),
            cases=[
                dict(
                    status="passed",
                    case=dict(name=profiling.CASES[0]),
                    runtime=dict(
                        uuid="GPU-old",
                        operator_binding=dict(canonical_source=str(canonical)),
                    ),
                )
            ],
        )
        save(original / "stage-summary.json", summary)
        options = types.SimpleNamespace(
            resume_report=str(original),
            report_dir=str(output),
            flagtree="/opt/flagtree",
            timeout=1800,
            ncu=None,
        )
        return options, code, original, output

    def test_reuses_original_worker_tools_and_benchmark_without_retiming(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ, clear=False
        ):
            options, code, original, output = self.fixture(Path(folder))
            before = (original / "stage-summary.json").read_bytes()
            old, command, facts = resume.prepare(options)
            self.assertEqual(old, original.resolve())
            self.assertEqual((output / "runner.txt").read_text(), "current tee log")
            self.assertEqual(
                (output / "baseline-runner.txt").read_text(), "original run log"
            )
            self.assertEqual(
                command[2],
                str(code.resolve() / "tools/benchmark_qsa_prefill_triton_stages.py"),
            )
            self.assertNotIn("--skip-accuracy", command)
            self.assertFalse(facts["independent_benchmark_rerun"])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-old")
            self.assertEqual(
                os.environ["TRITON_PTXAS_PATH"], str(Path(folder) / "ptxas")
            )
            self.assertEqual((output / "stage-summary.json").read_bytes(), before)
            self.assertEqual((original / "stage-summary.json").read_bytes(), before)
            self.assertEqual(
                (output / "qsa_indexer_vllm.py").read_bytes(),
                (original / "qsa_indexer_vllm.py").read_bytes(),
            )

    def test_source_or_tool_changes_block_resume(self):
        for target in ("source", "ptxas", "libcuda.so"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                options, code, original, output = self.fixture(root)
                path = (
                    code / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
                    if target == "source"
                    else root / target
                )
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    resume.prepare(options)
                self.assertFalse((output / "stage-summary.json").exists())

    def test_original_report_cannot_be_resume_output(self):
        with tempfile.TemporaryDirectory() as folder:
            options, _, original, _ = self.fixture(Path(folder))
            options.report_dir = str(original)
            with self.assertRaisesRegex(ValueError, "differ"):
                resume.prepare(options)

    def test_discovery_prefers_explicit_path_then_path(self):
        with patch.object(profiling.shutil, "which", return_value="/path/ncu") as which:
            self.assertEqual(profiling.find_ncu("/explicit/ncu"), "/explicit/ncu")
            which.assert_not_called()
            self.assertEqual(profiling.find_ncu(), "/path/ncu")


if __name__ == "__main__":
    unittest.main()
