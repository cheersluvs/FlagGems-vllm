# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Readable saved results without GPU work, retiming or data normalization."""

import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools import qsa_prefill_report as report


class ReportTests(unittest.TestCase):
    def timing(self):
        def stage(a, b, speed):
            return dict(
                status="passed",
                baseline_us=dict(median=a),
                candidate_us=dict(median=b),
                speedup=speed,
                measurement_validated=True,
            )

        return dict(
            case=dict(name="sample"),
            status="passed",
            stages=dict(
                score=stage(51.424, 8.112, 6.33925),
                topk=stage(8.336, 11.248, 0.74111),
                pipeline=stage(56.744, 25.776, 2.20143),
                pipeline_vs_production=stage(16.416, 16.304, 1.00687),
            ),
        )

    def test_independent_comparisons_and_topk_are_shown_without_mutation(self):
        value = self.timing()
        old = copy.deepcopy(value)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report.show_timing(value)
        self.assertEqual(value, old)
        text = output.getvalue()
        for expected in (
            "topk",
            "11.248",
            "0.7411",
            "25.776",
            "16.304",
            "1.0069",
            "另一个独立计时段",
        ):
            self.assertIn(expected, text)

    def test_missing_zero_units_and_out_of_range_ratios(self):
        values, anomalies = report.metric_values(
            dict(
                launch__shared_mem_per_block_dynamic="20480",
                gpu__dram_throughput__unused="0",
                lts__t_sector_hit_rate__unused="bad",
                **{
                    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "0",
                    "lts__t_sector_hit_rate.pct": "102.94",
                },
            ),
            dict(launch__shared_mem_per_block_dynamic="byte/block"),
        )
        self.assertEqual(values[0], "N/A")
        self.assertEqual(values[1], "20.00")
        self.assertEqual(values[4], "0.00")
        self.assertEqual(values[5], "102.94*")
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(report.fmt(float("nan")), "N/A")

    def test_profile_rows_follow_nvtx_side_and_real_chunk(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            value = dict(
                status="passed",
                chunks=[dict(offset=9, rows=112)],
                cache_granularity=dict(
                    production_baseline=[dict(constants=dict(TILE_R=8, num_stages=2))],
                    production=[
                        dict(constants=dict(TILE_R=64, num_stages=1, QUERY_PARTS=2))
                    ],
                ),
            )
            (root / "profile-case.json").write_text(json.dumps(value))
            (root / "profile-case.csv").write_text(
                '"ID","NVTX Range","launch__registers_per_thread",'
                '"launch__shared_mem_per_block_dynamic","lts__t_sector_hit_rate.pct"\n'
                '"","","register/thread","byte/block","%"\n'
                "0,qsa_score/production,66,43008,99\n1,qsa_score/candidate,154,49152,102.94\n"
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                report.show_profile(root, "case", reused=True)
            text = output.getvalue()
            for expected in (
                "复用采集",
                "production",
                "candidate",
                "112",
                "42.00",
                "48.00",
                "P=2",
                "102.94*",
                "保留原值",
            ):
                self.assertIn(expected, text)

    def test_offline_cli_reads_report_without_gpu_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "stage-summary.json").write_text(
                json.dumps(
                    dict(
                        status="passed",
                        accuracy=dict(status="skipped_by_request"),
                        cases=[self.timing()],
                    )
                )
            )
            before = (root / "stage-summary.json").read_bytes()
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(Path(report.__file__)),
                    "--report-dir",
                    str(root),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("skipped_by_request", result.stdout)
            self.assertIn("not_run", result.stdout)
            self.assertEqual((root / "stage-summary.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
