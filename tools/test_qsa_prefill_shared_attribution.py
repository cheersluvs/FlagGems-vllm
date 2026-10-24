# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Source/IR/PC attribution tests; synthetic counters are never performance data."""

import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools import qsa_prefill_shared_attribution as probe


class AttributionTests(unittest.TestCase):
    def test_source_regions_and_wait_instruction_distinction(self):
        self.assertEqual(probe.region(probe.SOURCE, 470), "output_layout_or_store")
        self.assertEqual(probe.region(probe.SOURCE, 442), "K_operand_feed")
        self.assertEqual(probe.region(probe.SOURCE, 451), "dot_low")
        self.assertEqual(probe.region(probe.SOURCE, 453), "head_reduction")
        self.assertEqual(probe.region("standard.py", 213), "unmapped_or_inlined")
        for instruction, kind in [
            ("WARPGROUP.DEPBAR.LE 0x0", "mma_wait"),
            ("DEPBAR.LE SB0,0", "dependency_wait"),
            ("cp.async.wait_group 0;", "async_copy_wait"),
            ("bar.sync 0;", "cta_barrier"),
            ("LDS.128 R0,[R1]", "shared_load"),
            ("STS.128 [R1],R0", "shared_store"),
        ]:
            self.assertEqual(probe.family(instruction), kind)

    def test_static_context_maps_conversion_and_preserves_unknown(self):
        ptx = """
.file 1 "/tmp/qsa_select_paged_prefill.py"
.file 2 "/opt/standard.py"
.loc 1 470 12
st.shared.v4.b32 [x], {a,b,c,d};
bar.sync 0;
ld.shared.v4.b32 {a,b,c,d}, [x];
.loc 2 213 40 // standard.py:213:40 @[ qsa_select_paged_prefill.py:393:36 ]
bar.sync 0;
.loc 1 0 0
cp.async.wait_group 0;
"""
        ir = """
%a = ttg.convert_layout %b : tensor<f32> loc(#loc1)
#loc1 = loc("scores"(#loc2))
#loc2 = loc(callsite(#loc3 at #loc4))
#loc3 = loc("/opt/standard.py":213:40)
#loc4 = loc("/tmp/qsa_select_paged_prefill.py":470:12)
"""
        out = probe.static_map(ptx, ir)
        self.assertEqual(len(out["ptx"]), 5)
        self.assertEqual(out["ptx"][0]["region"], "output_layout_or_store")
        self.assertEqual(out["ptx"][3]["region"], "visible_max_reduction")
        self.assertEqual(out["ptx"][4]["region"], "unmapped_or_inlined")
        self.assertEqual(out["ir"][0]["source_line"], 470)
        self.assertEqual(out["kind"], "static_sites_not_cycles")

    def metric(self, ids, values):
        return types.SimpleNamespace(
            has_correlation_ids=lambda: True,
            correlation_ids=lambda: types.SimpleNamespace(
                num_instances=lambda: len(ids), value=lambda i: ids[i]
            ),
            num_instances=lambda: len(values),
            value=lambda i: values[i],
        )

    def action(self, metrics):
        return types.SimpleNamespace(
            metric_names=lambda: tuple(metrics),
            metric_by_name=lambda n: metrics[n],
            source_info=lambda pc: types.SimpleNamespace(
                file_name=lambda: str(probe.SOURCE),
                line=lambda: 470 if pc == 16 else 431,
            ),
            sass_by_pc=lambda pc: (
                "BAR.SYNC 0x0" if pc == 16 else "WARPGROUP.DEPBAR.LE 0x0"
            ),
        )

    def test_pc_reason_fractions_are_separate_and_not_time(self):
        name = "smsp__pcsamp_warps_issue_stalled_barrier"
        other = "smsp__pcsamp_warps_issue_stalled_short_scoreboard"
        metrics = {
            name: self.metric([16, 32], [30, 10]),
            other: self.metric([16, 32], [5, 15]),
            "derived__memory_l1_conflicts_shared_nway": self.metric([16], [2]),
        }
        result = probe.pc_rows(self.action(metrics))
        self.assertTrue(result["positive_stall_samples"])
        reasons = {v["metric"]: v for v in result["reasons"]}
        self.assertEqual(reasons[name]["total_samples"], 40)
        self.assertEqual(reasons[name]["groups"][0]["fraction_of_this_reason"], 0.75)
        self.assertEqual(reasons[other]["total_samples"], 20)
        self.assertEqual(len(reasons), 2)  # never sum the derived ratio
        self.assertEqual(result["rows"][1]["family"], "mma_wait")
        output = io.StringIO()
        result.update(case="synthetic", static=dict(counts=[]))
        with contextlib.redirect_stdout(output):
            probe.show(result)
        self.assertIn("不是耗时占比", output.getvalue())

    def test_pc_mismatch_duplicate_and_no_stall_coverage(self):
        name = "smsp__pcsamp_warps_issue_stalled_barrier"
        with self.assertRaisesRegex(RuntimeError, "length mismatch"):
            probe.pc_rows(self.action({name: self.metric([16], [1, 2])}))
        with self.assertRaisesRegex(RuntimeError, "Duplicate PC"):
            probe.pc_rows(self.action({name: self.metric([16, 16], [1, 2])}))
        self.assertFalse(
            probe.pc_rows(self.action({name: self.metric([16], [0])}))[
                "positive_stall_samples"
            ]
        )
        selected = "smsp__pcsamp_warps_issue_stalled_selected"
        self.assertFalse(
            probe.pc_rows(self.action({selected: self.metric([16], [100])}))[
                "positive_stall_samples"
            ]
        )

    def test_capture_flags_only_first_kernel_without_killing_worker(self):
        cmd = probe.capture_command(
            "ncu", "raw.csv", "source.ncu-rep", ["metric"], ["python", "worker"]
        )
        for flag, value in [
            ("--launch-count", "1"),
            ("--profile-from-start", "off"),
            ("--cache-control", "none"),
            ("--clock-control", "none"),
            ("--replay-mode", "kernel"),
            ("--section", "SourceCounters"),
        ]:
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertNotIn("--kill", cmd)
        self.assertEqual(cmd[-2:], ["python", "worker"])
        self.assertIn("--export", cmd)

    def test_offline_export_uses_report_not_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            name = "synthetic"
            (root / ("source-" + name + ".ncu-rep")).write_bytes(b"synthetic report")
            action = self.action(
                {"smsp__pcsamp_warps_issue_stalled_barrier": self.metric([16], [1])}
            )
            mod = types.SimpleNamespace(
                load_report=lambda path: types.SimpleNamespace(
                    num_ranges=lambda: 1,
                    range_by_idx=lambda i: types.SimpleNamespace(
                        num_actions=lambda: 1, action_by_idx=lambda j: action
                    ),
                )
            )
            command = Mock(
                return_value=dict(returncode=0, stdout="synthetic export", stderr="")
            )
            with patch.object(probe, "report_module", return_value=mod), patch.object(
                probe, "map_record", return_value=dict(counts=[])
            ), contextlib.redirect_stdout(io.StringIO()):
                result = probe.export_source("ncu", root, name, command)
            self.assertTrue(result["positive_stall_samples"])
            self.assertEqual(command.call_count, 2)
            for call in command.call_args_list:
                self.assertIn("--import", call.args[0])
                self.assertNotIn("--print-metric-name", call.args[0])
            self.assertTrue((root / ("source-" + name + ".sass.csv")).is_file())
            self.assertTrue((root / ("attribution-" + name + ".json")).is_file())

    def test_resume_accepts_only_exact_export_revision(self):
        import copy

        with patch.object(
            probe.current, "EXTRA_SOURCES", probe.current.EXTRA_SOURCES | probe.FILES
        ):
            expected = probe.current.verify_source()
        legacy = copy.deepcopy(expected)
        legacy["profile_sources"].update(probe.LEGACY_EXPORT_SOURCES)
        for source in (expected, legacy):
            self.assertEqual(
                probe.capture_source({"source": source}, {"source": source}), source
            )
        bad = copy.deepcopy(legacy)
        bad["production_sha256"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "Unknown capture"):
            probe.capture_source({"source": bad}, {"source": bad})
        bad = copy.deepcopy(legacy)
        bad["profile_sources"]["tools/qsa_prefill_production_profile.py"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "Unknown capture"):
            probe.capture_source({"source": bad}, {"source": bad})
        bad = copy.deepcopy(legacy)
        bad["profile_sources"][
            "tools/qsa_prefill_shared_attribution.py"
        ] = "unrecognized collector"
        with self.assertRaisesRegex(RuntimeError, "Unknown capture"):
            probe.capture_source({"source": bad}, {"source": bad})
        with self.assertRaisesRegex(RuntimeError, "source mismatch"):
            probe.capture_source({"source": expected}, {"source": legacy})

    def test_source_counter_query_and_wavefront_ratios(self):
        with tempfile.TemporaryDirectory() as folder:
            command = Mock(
                return_value=dict(
                    returncode=0,
                    stdout="inst_executed memory_l1_wavefronts_shared",
                    stderr="",
                )
            )
            selected = probe.select_source_metrics(
                "ncu", command, Path(folder) / "query.json"
            )
            self.assertEqual(
                selected["selected"], ["inst_executed", "memory_l1_wavefronts_shared"]
            )
            self.assertEqual(selected["missing"], ["memory_l1_wavefronts_shared_ideal"])
            command.return_value["returncode"] = 1
            with self.assertRaisesRegex(RuntimeError, "before capture"):
                probe.select_source_metrics(
                    "ncu", command, Path(folder) / "failed.json"
                )
        result = probe.pc_rows(
            self.action(
                {
                    "inst_executed": self.metric([16], [4]),
                    "memory_l1_wavefronts_shared": self.metric([16], [16]),
                    "memory_l1_wavefronts_shared_ideal": self.metric([16], [8]),
                }
            )
        )
        row = result["rows"][0]
        self.assertEqual(row["shared_wavefronts_per_instruction"], 4)
        self.assertEqual(row["ideal_shared_wavefronts_per_instruction"], 2)
        self.assertEqual(row["excess_shared_wavefronts"], 8)
        self.assertFalse(result["positive_stall_samples"])

    def test_module_bridge_discovery_is_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "extras/python").mkdir(parents=True)
            (root / "extras/python/ncu_report.py").write_text("# synthetic")
            old = list(sys.path)
            try:
                with patch.object(
                    probe.importlib, "import_module", return_value=object()
                ) as load:
                    probe.report_module(root / "ncu")
                    load.assert_called_once_with("ncu_report")
                    self.assertEqual(
                        sys.path[0], str((root / "extras/python").resolve())
                    )
            finally:
                sys.path[:] = old


if __name__ == "__main__":
    unittest.main()
