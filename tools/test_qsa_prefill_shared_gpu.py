# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Shared diagnostics must retain correctness/thermal checks and never accept perf."""

import argparse
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_flagtree_ab as profile
from tools import qsa_prefill_shared_gpu as shared
from tools import validate_qsa_ptxas_h800 as validation

UUID = "GPU-00000000-0000-0000-0000-000000000001"


class SharedTests(unittest.TestCase):
    def test_only_explicit_residency_rejections_are_waived(self):
        original = dict(
            status="rejected",
            queries={},
            processes=[dict(pid=2, used_memory_mib=75000)],
            reasons=[
                "Initial memory MiB 75000 exceeds limit 256",
                "GPU already has compute processes; no jobs will be stopped",
            ],
        )
        for free, reason, succeeds in (
            (5600, None, True),
            (4800, None, False),
            (5600, "SW Thermal Slowdown: Active", False),
            (5600, "Initial utilization percent 50 exceeds limit 5", False),
            (5600, "Cannot read thermal slowdown state", False),
        ):
            with self.subTest(free=free, reason=reason):
                check = copy.deepcopy(original)
                if reason:
                    check["reasons"].append(reason)
                with patch.object(
                    shared.guard, "idle", return_value=check
                ), patch.object(
                    shared.support,
                    "command",
                    return_value=dict(returncode=0, stdout=f"{UUID}, {free}, 81559\n"),
                ):
                    result = shared.inspect_gpu(UUID, 5120)
                self.assertEqual(result["status"] == "passed", succeeds)
                self.assertFalse(result["acceptance_eligible"])
                self.assertEqual(result["processes"][0]["pid"], 2)
                self.assertEqual(original["status"], "rejected")

    def test_unreadable_free_memory_fails_closed(self):
        check = dict(status="rejected", queries={}, processes=[], reasons=[])
        with patch.object(shared.guard, "idle", return_value=check), patch.object(
            shared.support,
            "command",
            return_value=dict(returncode=0, stdout=f"{UUID}, N/A, 81559\n"),
        ):
            self.assertEqual(shared.inspect_gpu(UUID, 5120)["status"], "rejected")

    def test_shared_telemetry_records_residents_but_rejects_real_thermal_slowdown(self):
        query = dict(returncode=0, stdout=f"{UUID}, 1, 1024\n{UUID}, 2, 75000\n")
        for active in (False, True):
            with self.subTest(active=active), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "case.json"
                point = dict(
                    label="round",
                    clocks=dict(
                        returncode=0,
                        stdout=f"time,uuid,temp,sm,mem,power\ntime,{UUID},40,1590,1593,100\n",
                    ),
                    performance=dict(
                        returncode=0,
                        stdout=(
                            f"SW Thermal Slowdown : {'Active' if active else 'Not Active'}\n"
                            "HW Thermal Slowdown : Not Active\n"
                        ),
                    ),
                )
                strict = shared.guard.timing(point, query, UUID, [1])
                self.assertEqual(strict["status"], "rejected")
                result = {}
                with patch.object(
                    shared.support, "telemetry", return_value=point
                ), patch.object(
                    shared.guard, "process_query", return_value=query
                ), patch.object(
                    shared.guard, "self_pids", return_value=[1]
                ):
                    if active:
                        with self.assertRaisesRegex(
                            RuntimeError, "SW Thermal Slowdown"
                        ):
                            shared.telemetry(UUID, "round", result, path)
                    else:
                        shared.telemetry(UUID, "round", result, path)
                        self.assertEqual(
                            result["performance_guard"]["status"], "shared_exploratory"
                        )
                        self.assertEqual(
                            point["co_resident_processes"],
                            [dict(pid=2, used_memory_mib=75000)],
                        )
                    self.assertFalse(result["performance_guard"]["acceptance_eligible"])
                    self.assertEqual(json.loads(path.read_text()), result)

    def test_exploratory_table_never_reports_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            stages = {
                name: dict(
                    measurement_validated=True,
                    accepted=False,
                    baseline_us=dict(median=10),
                    candidate_us=dict(median=20),
                    speedup=0.5,
                    paired_ratios=dict(median=0.5),
                )
                for name in ("score", "topk", "pipeline")
            }
            summary = dict(
                manifest=dict(
                    options=dict(compiler="flagtree", shared_gpu_exploratory=True)
                ),
                cases=[dict(case=dict(name="case"), status="passed", stages=stages)],
            )
            bench.write_table(summary, Path(folder))
            text = (Path(folder) / "triton-stage-ab.tsv").read_text()
            self.assertIn("exploratory_speedup", text)
            self.assertEqual(text.count("\texploratory\t"), 3)
            self.assertNotIn("\tpassed\t", text)
            self.assertTrue(all(not value["accepted"] for value in stages.values()))

    def test_bootstrap_and_children_propagate_explicit_shared_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            options = argparse.Namespace(
                report_dir=folder,
                flagtree="/opt/flagtree",
                shared_gpu_exploratory=True,
                shared_min_free_mib=5120,
                compiler="flagtree",
                suite="upstream-prefill",
                reference_file="native.py",
                rounds=7,
                kernel_warmup_ms=1000,
                kernel_rep_ms=100,
            )
            tool = dict(
                ptxas=dict(path="/tmp/verified-ptxas", sha256=profile.PTXAS_SHA256)
            )
            with patch.object(profile, "select_gpu", return_value=True), patch.object(
                validation, "toolchain", return_value=tool
            ), patch.object(profile.os, "execve") as execute:
                self.assertEqual(profile.prepare(options, []), 0)
                args = execute.call_args.args[1]
                self.assertIn("--shared-gpu-exploratory", args)
                self.assertEqual(args[args.index("--shared-min-free-mib") + 1], "5120")
            child = bench.child_command(options, Path(folder), "--worker")
            self.assertIn("--shared-gpu-exploratory", child)
            self.assertEqual(child[child.index("--shared-min-free-mib") + 1], "5120")

    def test_shared_auto_selection_checks_memory_instead_of_waiving_all_guards(self):
        options = argparse.Namespace(
            shared_gpu_exploratory=True, shared_min_free_mib=5120
        )
        other = UUID[:-1] + "2"
        query = dict(
            returncode=0, stdout=f"{UUID}, NVIDIA H800, 29\n{other}, NVIDIA H800, 30\n"
        )
        with patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "auto"}, clear=True
        ), patch.object(bench.support, "command", return_value=query), patch.object(
            shared,
            "inspect_gpu",
            side_effect=[
                dict(status="rejected", reasons=["low free memory"]),
                dict(status="passed", processes=[dict(pid=2)]),
            ],
        ) as inspect, contextlib.redirect_stdout(
            io.StringIO()
        ):
            summary = {}
            self.assertTrue(profile.select_gpu(summary, options))
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], other)
            self.assertEqual(
                [call.args[1] for call in inspect.call_args_list], [5120, 5120]
            )

    def test_shared_cooldown_allows_residency_without_disabling_temperature_gate(self):
        checks = [
            dict(
                status="rejected",
                reasons=["Initial temperature C 80 exceeds limit 75"],
                temperature_c=80,
                utilization_percent=0,
                processes=[dict(pid=2)],
            ),
            dict(
                status="passed",
                reasons=[],
                temperature_c=75,
                utilization_percent=0,
                processes=[dict(pid=2)],
            ),
        ]
        with tempfile.TemporaryDirectory() as folder, patch.object(
            shared, "inspect_gpu", side_effect=checks
        ), patch.object(profile.time, "sleep") as sleep, contextlib.redirect_stdout(
            io.StringIO()
        ):
            summary = {}
            self.assertTrue(
                shared.admission(
                    summary, Path(folder), {}, "before_case", 5120, 900, 10
                )
            )
            self.assertEqual(sleep.call_args.args, (10,))
            self.assertEqual(len(summary["shared_gpu_checks"]), 2)
            self.assertNotIn("environment_rejected", summary)


if __name__ == "__main__":
    unittest.main()
