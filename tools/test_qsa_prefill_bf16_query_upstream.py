# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Frozen configuration provenance, completeness and guarded suite coordination."""

import contextlib
import copy
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_bf16_query_probe as probe
from tools import qsa_prefill_bf16_query_upstream as suite
from tools import qsa_prefill_upstream_workloads as upstream


class UpstreamProbeTests(unittest.TestCase):
    def record(self, name, first):
        saved, traces = suite.saved_record(name)
        identity = saved["compiler_identity"]
        runtime = dict(
            torch_version=identity["torch"],
            triton_version=identity["triton"],
            versions=dict(vllm=identity["vllm"]),
            compiler_files={
                k: dict(sha256=v) for k, v in identity["compiler_files"].items()
            },
            flagtree_runtime=dict(
                files={
                    k: dict(sha256=v) for k, v in identity["flagtree_files"].items()
                },
                ptxas=dict(sha256=identity["ptxas"]),
                cuda=dict(libcuda_sha256=identity["libcuda"]),
            ),
        )
        resources = {
            side: [
                dict(
                    chunk=[t["rows"], t["offset"], t["width"]],
                    constants=dict(t["constants"], QUERY_PARTS=parts),
                )
                for t in traces
            ]
            for side, parts in (("baseline", 1), ("candidate", 2))
        }
        row = dict(
            status="passed",
            measurement_validated=True,
            baseline_us=dict(samples=[10, 10], median=10),
            candidate_us=dict(samples=[9, 9], median=9),
            speedup=10 / 9,
            paired_ratios=dict(min=10 / 9, max=10 / 9),
            measurement_order=[["baseline", "candidate"], ["candidate", "baseline"]],
            validation_before=[3, 3],
            validation_after=[3, 3],
        )
        return dict(
            status="passed",
            case=saved["case"],
            inputs=saved["inputs"],
            fixed_traces=traces,
            source=probe.verify_source(),
            compiler_matches_capture=True,
            same_buffers=True,
            same_topk=True,
            query_parts_pair=[1, 2],
            mode="kernel",
            production_modified=False,
            acceptance_eligible=False,
            kernels_before=resources,
            kernels_after=copy.deepcopy(resources),
            common_before=["unchanged"],
            common_after=["unchanged"],
            runtime=runtime,
            telemetry=[dict(gpu_guard=dict(status="passed"))],
            boundary_checks=(
                [dict(status="passed", all_rows_checked=289)] * 2
                if name == first
                else [dict(status="covered_by_first_case", case=first)]
            ),
            stages={k: copy.deepcopy(row) for k in suite.STAGES},
            self_controls={
                side + ":" + stage: copy.deepcopy(row)
                for side in ("P1", "P2")
                for stage in ("score", "pipeline")
            },
        )

    def test_manifest_is_exact_twelve_with_real_mixed_chunk_configs(self):
        manifest = suite.load_manifest()
        self.assertEqual([x["case"] for x in manifest["cases"]], upstream.cases())
        self.assertEqual(sum(len(x["traces"]) for x in manifest["cases"]), 26)
        self.assertTrue(
            any(
                len({t["constants"]["TILE_R"] for t in x["traces"]}) > 1
                for x in manifest["cases"]
            )
        )
        for item in manifest["cases"]:
            self.assertNotIn("stages", item)  # Never transplant old timing.
            self.assertEqual(len(item["report_blob_sha1"]), 40)
            self.assertEqual(len(item["report_sha256"]), 64)
        with self.assertRaises(ValueError):
            suite.saved_record("unknown")

    def test_manifest_content_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            path.write_bytes(suite.MANIFEST.read_bytes() + b" ")
            with patch.object(suite, "MANIFEST", path), self.assertRaisesRegex(
                RuntimeError, "SHA256"
            ):
                suite.load_manifest()

    def test_actual_configs_and_proof_scope_are_checked(self):
        name = upstream.cases()[2]["name"]
        record = self.record(name, name)
        suite.validate_record(record, name, 2)
        variants = []
        for key in ("kernels_before", "kernels_after"):
            bad = copy.deepcopy(record)
            bad[key]["candidate"][0]["constants"]["num_stages"] = 2
            variants.append(bad)
        bad = copy.deepcopy(record)
        bad["fixed_traces"][0]["constants"]["TILE_R"] = 16
        variants.append(bad)
        bad = copy.deepcopy(record)
        bad["boundary_checks"] = []
        variants.append(bad)
        bad = copy.deepcopy(record)
        bad["stages"]["score"]["measurement_order"].reverse()
        variants.append(bad)
        bad = copy.deepcopy(record)
        bad["telemetry"][0]["gpu_guard"]["status"] = "rejected"
        variants.append(bad)
        bad = copy.deepcopy(record)
        bad["self_controls"].pop("P2:pipeline")
        variants.append(bad)
        for bad in variants:
            with self.assertRaises(RuntimeError):
                suite.validate_record(bad, name, 2)

    def test_aggregate_is_unweighted_per_case_and_retains_regression(self):
        names = [x["name"] for x in upstream.cases()[:2]]
        records = [self.record(n, names[0]) for n in names]
        records[0]["stages"]["pipeline"]["speedup"] = 1.1
        records[1]["stages"]["pipeline"]["speedup"] = 0.9
        result = suite.aggregate(records)["pipeline"]
        self.assertEqual(result["arithmetic_mean"], 1.0)
        self.assertEqual(result["minimum"], 0.9)
        self.assertEqual(result["below_one"], [names[1]])
        self.assertIsNone(suite.aggregate([])["score"]["arithmetic_mean"])

    def coordinate(self, root, fail_second=False, full=False):
        from tools import benchmark_qsa_prefill_triton_stages as bench
        from tools import qsa_prefill_profiling as profiling

        names = [x["name"] for x in upstream.cases()][: 12 if full else 3]
        options = types.SimpleNamespace(
            report_dir=str(root), cases=names, rounds=2, timeout=1800
        )
        calls = []

        def run(command, *, timeout):
            name = command[-1]
            calls.append(name)
            self.assertEqual(command[-3:-1], ["--worker", "--worker-case"])
            if fail_second and len(calls) == 2:
                return 124
            (root / (name + ".json")).write_text(
                json.dumps(self.record(name, names[0]))
            )
            return 0

        with patch.object(probe, "prepare_environment"), patch.object(
            profiling, "run_process", side_effect=run
        ), patch.object(
            bench, "idle_gate", return_value=True
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            code = suite.coordinate(options)
        return code, json.loads((root / "probe-summary.json").read_text()), calls

    def test_suite_worker_uses_requested_case_and_checks_boundaries_once(self):
        from tools.test_qsa_prefill_bf16_query_probe import ProbeTests

        for skip in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                name = upstream.cases()[2]["name"]
                code, record = ProbeTests().run_worker(
                    Path(folder), case_name=name, skip_boundaries=skip
                )
                self.assertEqual(code, 0, record.get("error"))
                self.assertEqual(record["case"]["name"], name)
                self.assertIsNone(record["fixed_config"])
                self.assertEqual(
                    record["boundary_checks"][0]["status"],
                    "covered_by_first_case" if skip else "passed",
                )
                self.assertTrue((Path(folder) / (name + ".tsv")).is_file())

    def test_full_suite_is_complete_only_with_all_twelve_validated_records(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            code, summary, calls = self.coordinate(root, full=True)
            self.assertEqual(code, 0, summary.get("error"))
            self.assertEqual(len(calls), 12)
            self.assertTrue(summary["full_upstream12_complete"])
            self.assertFalse(summary["acceptance_eligible"])
            self.assertEqual(summary["aggregate"]["pipeline"]["count"], 12)
            self.assertEqual(
                len((root / "upstream-probe-timings.tsv").read_text().splitlines()), 37
            )

    def test_failure_stops_later_workers_and_retains_partial_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            code, summary, calls = self.coordinate(root, fail_second=True)
            self.assertEqual(code, 1)
            self.assertEqual(len(calls), 2)
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(len(summary["cases"]), 1)
            self.assertFalse(summary["full_upstream12_complete"])
            self.assertEqual(summary["worker_exit_codes"][calls[-1]], 124)
            self.assertIn("exit=124", summary["error"])
            self.assertEqual(
                len((root / "upstream-probe-timings.tsv").read_text().splitlines()), 4
            )


if __name__ == "__main__":
    unittest.main()
