# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Check diagnostic evidence, without requiring a GPU or fabricating counters."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools import qsa_prefill_memory_probe as probe
from tools import qsa_prefill_profiling as profiling


class MemoryProbeTests(unittest.TestCase):
    def record(self, cache=None):
        compiled = dict(cubin_sha256="missing", ptx_sha256="missing")
        return dict(
            case=dict(
                name=probe.CASES[0],
                heads=4,
                dim=128,
                ratio=4,
                page_size=196,
                query_length=16384,
                prefix_tokens=102400,
            ),
            chunks=[dict(offset=0, rows=2, width=29696)],
            runtime=dict(environment=dict(TRITON_CACHE_DIR=str(cache))),
            cache_granularity=dict(
                production=[dict(compiled=compiled)],
                production_baseline=[dict(compiled=compiled)],
            ),
            kernel_resources=dict(native=dict(kernels=[compiled])),
        )

    def test_explicit_metrics_preserve_capture_and_avoid_sections(self):
        cmd = profiling.ncu_command(
            "ncu", "out.csv", ["python", "worker"], metrics=list(probe.REQUIRED)
        )
        self.assertNotIn("--section", cmd)
        self.assertEqual(cmd[cmd.index("--metrics") + 1], ",".join(probe.REQUIRED))
        for flag, value in (
            ("--replay-mode", "kernel"),
            ("--cache-control", "none"),
            ("--clock-control", "none"),
            ("--profile-from-start", "off"),
        ):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertEqual(
            profiling.ncu_command("ncu", "x", ["worker"]).count("--section"),
            len(profiling.SECTIONS),
        )

    def test_metric_query_optional_missing_is_explicit(self):
        command = Mock(
            return_value=dict(returncode=0, stdout="\n".join(probe.REQUIRED))
        )
        selected = probe.metric_selection("ncu", command)
        self.assertEqual(selected["metrics"], list(probe.REQUIRED))
        self.assertEqual(selected["missing_optional"], list(probe.OPTIONAL))
        self.assertEqual(
            command.call_args.args[0][1:3], ["--query-metrics-mode", "suffix"]
        )

    def test_failed_query_saved_and_required_na_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "query.json"
            command = Mock(
                return_value=dict(returncode=1, stdout="", stderr="query error")
            )
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                probe.metric_selection("ncu", command, path)
            self.assertEqual(json.loads(path.read_text())["returncode"], 1)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            probe.metric_selection(
                "ncu", Mock(return_value=dict(returncode=0, stdout=""))
            )

    def write_traffic(self, root, bad=None, missing_label=False):
        name = probe.CASES[0]
        (root / ("profile-" + name + ".json")).write_text(json.dumps(self.record()))
        path = root / ("profile-" + name + ".csv")
        with path.open("w") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["ID", "NVTX Range"] + list(probe.REQUIRED)
            )
            writer.writeheader()
            for index, side in enumerate(("native", "production", "candidate")):
                row = dict(zip(probe.REQUIRED, (2, 8, 0, 0, 1024, 0)))
                row.update(
                    {
                        "ID": index,
                        "NVTX Range": "" if missing_label else "qsa_score/" + side,
                    }
                )
                if bad:
                    row[probe.REQUIRED[0]] = bad
                writer.writerow(row)

    def test_traffic_zero_is_zero_optional_missing_is_none(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(
            io.StringIO()
        ):
            root = Path(folder)
            self.write_traffic(root)
            rows = probe.traffic_report(root, probe.CASES[0])
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["dram_write_mib"], 0)
            self.assertIsNone(rows[0]["store_sectors_per_request"])
            self.assertEqual(rows[0]["load_sectors_per_request"], 4)
            self.assertTrue(all(v is None for v in rows[0]["optional"].values()))
            self.assertEqual(rows[0]["useful_logits_bytes"], 204800)

    def test_traffic_rejects_missing_metrics_and_unidentified_side(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_traffic(root, bad="N/A")
            with self.assertRaisesRegex(RuntimeError, "missing/N/A"):
                probe.traffic_report(root, probe.CASES[0])
            self.write_traffic(root, missing_label=True)
            with self.assertRaisesRegex(RuntimeError, "NVTX identity"):
                probe.traffic_report(root, probe.CASES[0])

    def test_exact_binary_and_ptx_anchor_cache_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / "cache"
            cache.mkdir()
            cubin, ptx = cache / "kernel.cubin", cache / "kernel.ptx"
            cubin.write_bytes(b"binary")
            ptx.write_text("actual ptx")
            (cache / "kernel.ttgir").write_text("actual ttgir")
            record = self.record(cache)
            compiled = record["kernel_resources"]["native"]["kernels"][0]
            compiled.update(cubin_sha256=probe.sha(cubin), ptx_sha256=probe.sha(ptx))
            report_path = root / ("profile-" + probe.CASES[0] + ".json")
            report_path.write_text(json.dumps(record))
            command = Mock(side_effect=AssertionError("No disassembler requested"))
            result = probe.export_artifacts(root, probe.CASES[0], None, command)
            self.assertTrue(result["all_ptx_exported"])
            self.assertTrue(result["all_ttgir_exported"])
            self.assertEqual(
                result["entries"][0]["sides"], ["candidate", "native", "production"]
            )
            self.assertEqual(cubin.read_bytes(), b"binary")
            ptx.write_text("wrong ptx for same binary")
            result = probe.export_artifacts(root, probe.CASES[0], None, command)
            self.assertFalse(result["all_ptx_exported"])
            self.assertFalse(result["all_ttgir_exported"])

    def test_missing_ir_is_partial_and_outside_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / "cache"
            cache.mkdir()
            cubin = root / "kernel.cubin"
            cubin.write_bytes(b"outside")
            (root / "kernel.ptx").write_text("outside ptx")
            (cache / "kernel.cubin").symlink_to(cubin)
            (cache / "kernel.ptx").symlink_to(root / "kernel.ptx")
            record = self.record(cache)
            record["kernel_resources"]["native"]["kernels"][0].update(
                cubin_sha256=probe.sha(cubin), ptx_sha256=probe.sha(root / "kernel.ptx")
            )
            (root / ("profile-" + probe.CASES[0] + ".json")).write_text(
                json.dumps(record)
            )
            result = probe.export_artifacts(root, probe.CASES[0], None, Mock())
            self.assertFalse(result["all_ptx_exported"])
            self.assertFalse(result["all_ttgir_exported"])

    def test_coordinator_only_captures_two_cases_and_reports_partial_artifacts(self):
        from tools import benchmark_qsa_prefill_stages as support
        from tools import qsa_prefill_profile_resume as resume
        from tools import qsa_prefill_report as report_module

        for complete in (True, False):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                original, output = root / "original", root / "output"
                original.mkdir()
                output.mkdir()
                (original / "flagtree134-tools.json").write_text('{"toolchain": {}}')
                for name in probe.CASES:
                    (original / (name + ".json")).write_text('{"historical": true}')
                before = {p.name: p.read_bytes() for p in original.iterdir()}

                def prepare(options):
                    options.cases = list(probe.CASES) + ["other_original_case"]
                    return (
                        original,
                        ["original-worker"],
                        dict(independent_benchmark_rerun=False),
                    )

                argv = [
                    "probe",
                    "--resume-report",
                    str(original),
                    "--report-dir",
                    str(output),
                    "--ncu",
                    "ncu",
                ]
                artifacts = dict(all_ptx_exported=True, all_ttgir_exported=complete)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(probe.sys, "argv", argv))
                    stack.enter_context(
                        patch.object(resume, "prepare", side_effect=prepare)
                    )
                    stack.enter_context(
                        patch.object(
                            probe,
                            "metric_selection",
                            return_value=dict(
                                metrics=list(probe.REQUIRED),
                                missing_optional=list(probe.OPTIONAL),
                            ),
                        )
                    )
                    capture = stack.enter_context(
                        patch.object(profiling, "collect_counters", return_value=0)
                    )
                    stack.enter_context(
                        patch.object(probe, "traffic_report", return_value=[])
                    )
                    stack.enter_context(
                        patch.object(probe, "export_artifacts", return_value=artifacts)
                    )
                    show = stack.enter_context(
                        patch.object(report_module, "show_timing")
                    )
                    command = stack.enter_context(patch.object(support, "command"))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    self.assertEqual(probe.main(), 0 if complete else 2)
                self.assertEqual(capture.call_count, 1)
                self.assertEqual(capture.call_args.args[0].cases, list(probe.CASES))
                self.assertEqual(capture.call_args.kwargs["cache_report"], original)
                self.assertEqual(
                    capture.call_args.kwargs["metric_names"], list(probe.REQUIRED)
                )
                self.assertEqual(show.call_count, 2)
                command.assert_not_called()
                summary = json.loads((output / "memory-summary.json").read_text())
                self.assertFalse(summary["independent_benchmark_rerun"])
                self.assertEqual(
                    summary["status"], "passed" if complete else "partial_artifacts"
                )
                self.assertEqual(
                    before, {p.name: p.read_bytes() for p in original.iterdir()}
                )


if __name__ == "__main__":
    unittest.main()
