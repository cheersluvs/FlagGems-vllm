# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""GPU-free checks of contention, thermal and namespace rejection policies."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "qsa_gpu_guard", Path(__file__).with_name("qsa_prefill_gpu_guard.py")
)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)
UUID = "GPU-198a9496-3d7a-f01d-ee60-190fe0bc5b51"
OTHER = "GPU-11111111-1111-1111-1111-111111111111"
CLEAN = "SW Thermal Slowdown : Not Active\n    HW Thermal Slowdown : Not Active\n"


def query(text):
    return dict(returncode=0, stdout=text, stderr="")


def idle_command(gpu=None, performance=None, processes=""):
    def command(args):
        assert args[:3] == ["nvidia-smi", "-i", UUID]
        if args[3].startswith("--query-gpu"):
            return query(gpu if gpu is not None else f"{UUID}, 40, 0, 1\n")
        if args[3].startswith("--query-compute-apps"):
            return query(processes)
        return query(CLEAN if performance is None else performance)

    return command


def point():
    return dict(
        clocks=query(
            "timestamp, uuid, temperature.gpu, clocks.current.sm, clocks.current.memory, power.draw\n"
            f"2026/10/15 18:00:00, {UUID}, 60, 1590 MHz, 2619 MHz, 130.0 W\n"
        ),
        performance=query(CLEAN),
    )


class GpuGuardTests(unittest.TestCase):
    def test_rejection_diagnostics_read_pid_identity_without_whitelisting(self):
        with patch.object(
            guard.Path, "read_bytes", return_value=b"python\0job.py\0--token\0private\0"
        ), patch.object(
            guard.Path,
            "read_text",
            return_value="Name:\tpython\nPid:\t456\nPPid:\t123\nNSpid:\t456\nVmSize:\t100\n",
        ):
            data = guard.process_details(
                [456], lambda args: dict(returncode=0, stdout="456 123 python")
            )
        self.assertEqual(data[0]["program"], "python")
        self.assertEqual(data[0]["script"], "job.py")
        self.assertNotIn("private", str(data))
        self.assertIn("PPid:\t123", data[0]["status"])
        self.assertNotIn("VmSize", str(data))
        self.assertEqual(
            guard.timing(point(), query(f"{UUID}, 456, 522"), UUID, [123])["status"],
            "rejected",
        )

    def test_exited_or_hidden_pid_diagnostics_do_not_raise(self):
        with patch.object(guard.Path, "read_bytes", side_effect=OSError("not visible")):
            data = guard.process_details(
                [456], lambda _: dict(returncode=1, stderr="gone")
            )
        self.assertIn("not visible", data[0]["proc_error"])
        self.assertEqual(data[0]["ps"]["returncode"], 1)

    def test_nvml_names_keep_only_program_identity(self):
        data = guard.process_names(
            UUID, lambda _: query(f"{UUID}, 456, /opt/python --token private, 610")
        )
        self.assertEqual(
            data["processes"], [dict(pid=456, program="python", memory_mib=610)]
        )
        self.assertNotIn("private", str(data))

    def test_idle_card_passes_with_explicit_identity(self):
        result = guard.idle(UUID, idle_command())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["device_uuid"], UUID)
        self.assertEqual(result["processes"], [])
        self.assertEqual(len(result["queries"]), 3)

    def test_temperature_override_keeps_other_rejections_and_legacy_default(self):
        command = idle_command(gpu=f"{UUID}, 79, 0, 1")
        self.assertEqual(guard.idle(UUID, command)["status"], "rejected")
        check = guard.idle(UUID, command, max_initial_temp_c=80)
        self.assertEqual(check["status"], "passed")
        self.assertEqual(check["max_initial_temp_c"], 80)
        check = guard.idle(
            UUID, idle_command(gpu=f"{UUID}, 81, 0, 1"), max_initial_temp_c=80
        )
        self.assertEqual(
            check["reasons"], ["Initial temperature C 81 exceeds limit 80"]
        )
        for command in (
            idle_command(gpu=f"{UUID}, 79, 0, 257"),
            idle_command(gpu=f"{UUID}, 79, 0, 1", processes=f"{UUID}, 5, 0"),
            idle_command(
                gpu=f"{UUID}, 79, 0, 1",
                performance=CLEAN.replace("Not Active", "Active"),
            ),
        ):
            self.assertEqual(
                guard.idle(UUID, command, max_initial_temp_c=80)["status"], "rejected"
            )
        for limit in (0, 151, float("nan"), "80"):
            self.assertEqual(
                guard.idle(
                    UUID,
                    lambda _: self.fail("must reject before queries"),
                    max_initial_temp_c=limit,
                )["status"],
                "rejected",
            )

    def test_ambiguous_or_multiple_device_selections_reject_before_query(self):
        for selector in ("", "0", "1,2", "GPU-abcd", UUID + "," + OTHER):
            with self.subTest(selector=selector), patch.object(
                guard, "process_query"
            ) as command:
                self.assertEqual(guard.idle(selector, command)["status"], "rejected")
                command.assert_not_called()

    def test_reported_competing_job_rejects(self):
        result = guard.idle(
            UUID,
            idle_command(
                gpu=f"{UUID}, 87, 97, 54192", processes=f"{UUID}, 896937, 54182"
            ),
        )
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["processes"][0]["pid"], 896937)
        self.assertEqual(len(result["reasons"]), 4)

    def test_each_idle_limit_and_compute_context_rejects_independently(self):
        for gpu, apps in (
            (f"{UUID}, 76, 0, 1", ""),
            (f"{UUID}, 40, 6, 1", ""),
            (f"{UUID}, 40, 0, 257", ""),
            (f"{UUID}, 40, 0, 1", f"{UUID}, 5, 0"),
        ):
            with self.subTest(gpu=gpu, apps=apps):
                self.assertEqual(
                    guard.idle(UUID, idle_command(gpu=gpu, processes=apps))["status"],
                    "rejected",
                )

    def test_only_isolated_trailing_utilization_may_be_rechecked(self):
        result = guard.idle(UUID, idle_command(gpu=f"{UUID}, 40, 20, 1"))
        self.assertTrue(result["retryable_utilization"])
        for gpu, apps in (
            (f"{UUID}, 76, 20, 1", ""),
            (f"{UUID}, 40, 20, 257", ""),
            (f"{UUID}, 40, 20, 1", f"{UUID}, 5, 0"),
        ):
            result = guard.idle(UUID, idle_command(gpu=gpu, processes=apps))
            self.assertFalse(result["retryable_utilization"])

    def test_thermal_slowdown_or_unknown_state_rejects(self):
        for text in (
            CLEAN.replace(
                "SW Thermal Slowdown : Not Active", "SW Thermal Slowdown : Active"
            ),
            CLEAN.replace(
                "HW Thermal Slowdown : Not Active", "HW Thermal Slowdown : Active"
            ),
            "SW Thermal Slowdown : Not Supported\n",
            "",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    guard.idle(UUID, idle_command(performance=text))["status"],
                    "rejected",
                )

    def test_missing_telemetry_never_passes(self):
        for response in (
            {"error": "timeout"},
            query("N/A"),
            dict(returncode=1, stderr="driver"),
        ):
            result = guard.idle(UUID, lambda _: response)
            self.assertEqual(result["status"], "rejected")

    def test_malformed_metrics_or_wrong_gpu_reject(self):
        for gpu in (
            f"{OTHER}, 40, 0, 1",
            f"{UUID}, N/A, 0, 1",
            f"{UUID}, 40, -1, 1",
            f"{UUID}, 40, 101, 1",
            f"{UUID}, 40, 0, -1",
            f"{UUID}, 40, 0, 1\n{UUID}, 40, 0, 1",
        ):
            with self.subTest(gpu=gpu):
                self.assertEqual(
                    guard.idle(UUID, idle_command(gpu=gpu))["status"], "rejected"
                )

    def test_other_gpu_process_is_not_a_competitor(self):
        result = guard.idle(UUID, idle_command(processes=f"{OTHER}, 123, 54182"))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["processes"], [])

    def test_timing_allows_only_identified_self_pid(self):
        result = guard.timing(point(), query(f"{UUID}, 123, 600"), UUID, [123])
        self.assertEqual(result["status"], "passed")
        result = guard.timing(
            point(), query(f"{UUID}, 123, 600\n{UUID}, 456, 1"), UUID, [123]
        )
        self.assertEqual(result["status"], "rejected")
        self.assertIn("Unattributed", str(result["reasons"]))
        self.assertEqual(
            guard.timing(point(), query(""), UUID, [123])["status"], "passed"
        )

    def test_timing_rejects_foreign_pid_and_malformed_process_data(self):
        for apps in (
            f"{UUID}, 896937, 54182",
            f"{UUID}, N/A, 1",
            "corrupt",
            f"{UUID}, 5, N/A",
        ):
            with self.subTest(apps=apps):
                self.assertEqual(
                    guard.timing(point(), query(apps), UUID, [123])["status"],
                    "rejected",
                )

    def test_timing_rechecks_thermal_state_and_clock_identity(self):
        for edit in ("thermal", "uuid", "clock", "missing"):
            p = point()
            if edit == "thermal":
                p["performance"] = query(
                    CLEAN.replace(
                        "SW Thermal Slowdown : Not Active",
                        "SW Thermal Slowdown : Active",
                    )
                )
            elif edit == "uuid":
                p["clocks"]["stdout"] = p["clocks"]["stdout"].replace(UUID, OTHER)
            elif edit == "clock":
                p["clocks"]["stdout"] = p["clocks"]["stdout"].replace(
                    "1590 MHz", "nan MHz"
                )
            else:
                p["clocks"] = {"error": "missing"}
            self.assertEqual(
                guard.timing(p, query(""), UUID, [123])["status"], "rejected"
            )

    def test_namespace_chain_is_recorded_without_guessing_new_pids(self):
        with patch.object(
            guard.Path, "read_text", return_value="Name:\tpython\nNSpid:\t987654\t321\n"
        ):
            self.assertEqual(set(guard.self_pids()), {os.getpid(), 987654, 321})
        with patch.object(guard.Path, "read_text", side_effect=OSError("no proc")):
            self.assertEqual(guard.self_pids(), [os.getpid()])


if __name__ == "__main__":
    unittest.main()
