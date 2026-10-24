# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU checks of capture boundaries, binary identity, counters and resumability."""

import contextlib
import copy
import csv
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools import qsa_prefill_production_profile as probe


class ProductionProfileTests(unittest.TestCase):
    def test_current_three_case_sources_and_original_configs(self):
        m = probe.manifest()
        probe.verify_source()
        self.assertEqual(tuple(v["case"]["name"] for v in m["cases"]), probe.CASES)
        self.assertEqual([len(v["traces"]) for v in m["cases"]], [2, 4, 4])
        self.assertEqual(
            [v["traces"][0]["constants"]["TILE_R"] for v in m["cases"]], [32, 64, 64]
        )
        self.assertTrue(
            all(t["constants"]["HEAD_GROUP"] for t in m["cases"][1]["traces"])
        )
        self.assertTrue(all((probe.ROOT / n).is_file() for n in probe.EXTRA_SOURCES))

    def test_metric_query_required_and_optional_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "query.json"
            command = Mock(
                return_value=dict(returncode=0, stdout="\n".join(probe.REQUIRED))
            )
            selection = probe.selected_metrics("ncu", command, p)
            self.assertEqual(selection["metrics"], list(probe.REQUIRED))
            self.assertEqual(selection["missing_optional"], list(probe.OPTIONAL))
            command.return_value = dict(returncode=0, stdout="")
            with self.assertRaisesRegex(RuntimeError, "Required traffic"):
                probe.selected_metrics("ncu", command, p)
            command.return_value = dict(returncode=1, stdout="")
            with self.assertRaisesRegex(RuntimeError, "metric query failed"):
                probe.selected_metrics("ncu", command, p)

    def test_capture_only_scores_and_stops_profiler_on_failure(self):
        events = []
        cudart = types.SimpleNamespace(
            cudaProfilerStart=lambda: events.append("start") or 0,
            cudaProfilerStop=lambda: events.append("stop") or 0,
        )
        nvtx = types.SimpleNamespace(
            range_push=lambda label: events.append(label),
            range_pop=lambda: events.append("pop"),
        )
        device = types.SimpleNamespace(synchronize=lambda: events.append("sync"))
        chunks = [dict(offset=i * 64, rows=64) for i in range(3)]
        captured = probe.capture_sequence(
            cudart,
            nvtx,
            device,
            chunks,
            lambda c: events.append(("score", c["offset"])),
        )
        self.assertEqual(events[0], "start")
        self.assertEqual(events[-1], "stop")
        self.assertEqual(len(captured), 3)
        for i in range(3):
            self.assertEqual(
                events[1 + i * 4 : 5 + i * 4],
                [f"qsa_score/production/chunk_{i}", ("score", i * 64), "sync", "pop"],
            )
        events.clear()
        with self.assertRaisesRegex(RuntimeError, "launch failure"):
            probe.capture_sequence(
                cudart,
                nvtx,
                device,
                chunks,
                Mock(side_effect=RuntimeError("launch failure")),
            )
        self.assertEqual(events[-2:], ["pop", "stop"])

    def record(self):
        item = probe.manifest()["cases"][0]
        ident = probe.manifest()["compiler_identity"]
        runtime = dict(
            torch_version=ident["torch"],
            triton_version=ident["triton"],
            versions=dict(vllm=ident["vllm"]),
            compiler_files={
                k: dict(sha256=v) for k, v in ident["compiler_files"].items()
            },
            flagtree_runtime=dict(
                files={k: dict(sha256=v) for k, v in ident["flagtree_files"].items()},
                ptxas=dict(sha256=ident["ptxas"]),
                cuda=dict(libcuda_sha256=ident["libcuda"]),
            ),
        )
        runtime["operator_binding"] = dict(
            canonical_sha256=probe.stage.PRODUCTION_SHA,
            candidate_is_canonical_module=True,
            exported_entry_is_candidate=True,
        )
        compiled = [
            dict(
                chunk=[t["rows"], t["offset"], t["width"]],
                route=t["route"],
                constants=t["constants"],
                ptx_sha256=t["ptx_sha256"],
                cubin_sha256=t["cubin_sha256"],
                registers_per_thread=63,
                shared_bytes=73728,
                spill_count=0,
            )
            for t in item["traces"]
        ]
        record = dict(
            status="passed",
            source=probe.verify_source(),
            case=item["case"],
            inputs=item["inputs"],
            runtime=runtime,
            validation_before=8,
            validation_after=8,
            compiled_before=compiled,
            compiled_after=copy.deepcopy(compiled),
            logits_before="same",
            logits_after="same",
            chunks=[
                dict(
                    offset=t["offset"],
                    rows=t["rows"],
                    width=t["width"],
                    useful_logits_bytes=4096,
                )
                for t in item["traces"]
            ],
            captures=[
                dict(
                    index=i,
                    label=f"qsa_score/production/chunk_{i}",
                    offset=t["offset"],
                    rows=t["rows"],
                )
                for i, t in enumerate(item["traces"])
            ],
            telemetry=[dict(gpu_guard=dict(status="passed", reasons=[]))] * 2,
            artifacts=[],
        )
        return item, record

    def test_freeze_requires_exact_current_production_configs_and_binaries(self):
        item, record = self.record()
        resources = record["compiled_before"]
        with patch.object(
            probe.base, "freeze_records", return_value=dict(baseline=resources)
        ):
            self.assertEqual(probe.freeze(object(), item["traces"]), resources)
            resources[0]["constants"] = dict(resources[0]["constants"], num_stages=1)
            with self.assertRaisesRegex(RuntimeError, "configuration"):
                probe.freeze(object(), item["traces"])
            resources[0]["constants"] = item["traces"][0]["constants"]
            resources[0]["cubin_sha256"] = "changed"
            with self.assertRaisesRegex(RuntimeError, "binary"):
                probe.freeze(object(), item["traces"])

    def test_saved_evidence_rejects_binary_config_and_missing_validation(self):
        item, record = self.record()
        probe.checked_capture(record, item)
        for key, value in [
            ("validation_after", 0),
            ("logits_after", "different"),
            ("captures", []),
            ("telemetry", []),
        ]:
            x = copy.deepcopy(record)
            x[key] = value
            with self.assertRaises(RuntimeError):
                probe.checked_capture(x, item)
        x = copy.deepcopy(record)
        x["compiled_before"][0]["cubin_sha256"] = "changed"
        x["compiled_after"] = copy.deepcopy(x["compiled_before"])
        with self.assertRaisesRegex(RuntimeError, "configuration/binary"):
            probe.checked_capture(x, item)
        x = copy.deepcopy(record)
        x["compiled_before"][0]["constants"]["num_stages"] = 1
        x["compiled_after"] = copy.deepcopy(x["compiled_before"])
        with self.assertRaisesRegex(RuntimeError, "configuration/binary"):
            probe.checked_capture(x, item)

    def write_csv(self, folder, bad=None):
        item, record = self.record()
        name = item["case"]["name"]
        # Synthetic text artifacts exercise parser hashes, separate from the
        # frozen GPU-binary provenance checks exercised by checked_capture.
        for i, r in enumerate(record["compiled_before"]):
            path = folder / f"artifact-{i}.txt"
            path.write_text("synthetic compiler artifact")
            digest = probe.base.sha(path)
            r["ptx_sha256"] = digest
            record["artifacts"].append(
                dict(
                    chunk=r["chunk"],
                    files=dict(ptx=dict(path=path.name, sha256=digest)),
                )
            )
        record["compiled_after"] = copy.deepcopy(record["compiled_before"])
        (folder / ("profile-" + name + ".json")).write_text(json.dumps(record))
        (folder / ("profile-" + name + ".txt")).write_text("synthetic worker log")
        rows = []
        for i in range(2):
            row = dict(zip(probe.REQUIRED, (2, 8, 4, 16, 1048576, 524288)))
            row.update(
                {
                    "ID": i,
                    "NVTX Range": f"qsa_score/production/chunk_{i}",
                    "lts__t_sector_hit_rate.pct": 101.2,
                    "launch__registers_per_thread": 63,
                    "launch__shared_mem_per_block_dynamic": 73728,
                }
            )
            rows.append(row)
        if bad == "duplicate":
            rows[1]["NVTX Range"] = rows[0]["NVTX Range"]
        if bad == "missing":
            rows[0].pop(probe.REQUIRED[1])
        if bad == "resource":
            rows[0]["launch__registers_per_thread"] = 64
        if bad == "extra":
            rows.append(dict(rows[0], ID=2))
        path = folder / ("profile-" + name + ".csv")
        keys = [
            "ID",
            "NVTX Range",
            *probe.REQUIRED,
            "lts__t_sector_hit_rate.pct",
            "launch__registers_per_thread",
            "launch__shared_mem_per_block_dynamic",
        ]
        with path.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        return name

    def test_real_counter_parser_labels_ratios_missing_anomalies(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            name = self.write_csv(folder)
            selection = dict(missing_optional=list(probe.OPTIONAL))
            with patch.object(probe, "checked_capture"):
                result = probe.counter_rows(folder, name, selection)
                self.assertEqual(len(result["rows"]), 2)
                self.assertEqual(result["rows"][0]["store_sectors_per_request"], 4)
                self.assertEqual(
                    result["rows"][0]["store_sector_bytes_per_payload"], 0.125
                )
                self.assertEqual(result["rows"][0]["dram_read_mib"], 1)
                self.assertFalse(result["diagnostics_complete"])
                self.assertEqual(
                    result["rows"][0]["metrics"]["lts__t_sector_hit_rate.pct"], 101.2
                )
                self.assertIn(
                    "lts__t_sector_hit_rate.pct",
                    result["rows"][0]["anomalous_percentages"],
                )
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    probe.display_counters(result)
                self.assertIn("N/A", output.getvalue())
                self.assertIn("不是 benchmark", output.getvalue())
                for bad in ("duplicate", "missing", "resource", "extra"):
                    self.write_csv(folder, bad)
                    with self.assertRaises(RuntimeError):
                        probe.counter_rows(folder, name, selection)
                self.write_csv(folder)
                (folder / "artifact-0.txt").write_text("tampered")
                with self.assertRaisesRegex(RuntimeError, "artifact hash"):
                    probe.counter_rows(folder, name, selection)

    def test_resume_copies_artifacts_and_revalidates_raw_csv(self):
        with tempfile.TemporaryDirectory() as original, tempfile.TemporaryDirectory() as output:
            previous, report = Path(original), Path(output)
            name = self.write_csv(previous)
            with patch.object(probe, "checked_capture"):
                result = probe.reuse_case(
                    previous, report, name, dict(missing_optional=[])
                )
                self.assertEqual(result["case"], name)
                self.assertEqual(
                    (report / ("profile-" + name + ".csv")).read_bytes(),
                    (previous / ("profile-" + name + ".csv")).read_bytes(),
                )
                self.assertEqual(
                    (report / "artifact-0.txt").read_bytes(),
                    (previous / "artifact-0.txt").read_bytes(),
                )
            with patch.object(
                probe, "checked_capture", side_effect=RuntimeError("stale source")
            ):
                with self.assertRaisesRegex(RuntimeError, "stale source"):
                    probe.reuse_case(previous, report, name, dict(missing_optional=[]))

    def test_coordinator_launches_three_production_only_workers_without_timing(self):
        from tools import benchmark_qsa_prefill_triton_stages as bench

        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder)
            o = types.SimpleNamespace(
                report_dir=folder,
                baseline_report="/tmp/b",
                resume_report=None,
                timeout=1800,
            )

            def prepare(options, facts):
                facts.update(
                    ncu="ncu",
                    selection=dict(
                        metrics=list(probe.REQUIRED),
                        missing_optional=list(probe.OPTIONAL),
                    ),
                )

            def counters(report, name, selection):
                return dict(case=name, rows=[], diagnostics_complete=False)

            def run(cmd, **kw):
                self.assertEqual(cmd[cmd.index("--replay-mode") + 1], "kernel")
                self.assertEqual(cmd[cmd.index("--profile-from-start") + 1], "off")
                self.assertEqual(cmd[cmd.index("--worker-case") - 1], folder)
                name = cmd[cmd.index("--worker-case") + 1]
                (report / ("profile-" + name + ".csv")).write_text(
                    "synthetic raw counter export"
                )
                return 0

            with patch.object(
                sys, "argv", [probe.__file__, "--report-dir", folder]
            ), patch.object(probe, "prepare", side_effect=prepare), patch.object(
                bench, "idle_gate", return_value=True
            ), patch.object(
                probe.profiling, "run_process", side_effect=run
            ) as run_process, patch.object(
                probe, "counter_rows", side_effect=counters
            ), patch.object(
                probe, "display_counters"
            ):
                self.assertEqual(probe.coordinate(o), 0)
                self.assertEqual(run_process.call_count, 3)
            facts = json.loads((report / "production-profile-summary.json").read_text())
            self.assertTrue(facts["hardware_counters_complete"])
            self.assertFalse(facts["diagnostics_complete"])
            self.assertFalse(facts["benchmark_rerun"])
            self.assertEqual([v["case"] for v in facts["cases"]], list(probe.CASES))

    def test_historical_topk_is_labelled_and_not_retimed(self):
        stages = {
            k: dict(
                status="passed",
                measurement_validated=True,
                baseline_us=dict(median=10),
                candidate_us=dict(median=8),
                speedup=1.25,
            )
            for k in ("score", "topk", "pipeline")
        }
        record = dict(status="passed", case=dict(name=probe.CASES[0]), stages=stages)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            probe.historical_timing(record, probe.CASES[0])
        self.assertIn("78c3ca4f", output.getvalue())
        self.assertIn("topk", output.getvalue())
        self.assertIn("本次不重测", output.getvalue())
        record["stages"]["topk"]["measurement_validated"] = False
        with self.assertRaisesRegex(RuntimeError, "Unvalidated"):
            probe.historical_timing(record, probe.CASES[0])

    def test_offline_cli_does_not_configure_cuda(self):
        from tools import qsa_prefill_flagtree_ab as profile

        with patch.object(
            sys, "argv", [probe.__file__, "--show-report", "/tmp/example"]
        ), patch.object(probe, "show_report") as show, patch.object(
            profile, "configure"
        ) as configure:
            self.assertEqual(probe.main(), 0)
            show.assert_called_once_with("/tmp/example")
            configure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
