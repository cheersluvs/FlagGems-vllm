# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check guarded capture ordering, strict containment and selected candidates."""

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qsa_finish_routing_probe_910b import VARIANTS, selected_variants
from qsa_int8_prefix_plan_timer_910b import measure_prefix_plan
from qsa_int8_stage_trace_timer_910b import stage_mean
from qsa_int8_trace_timer_910b import IncompleteScoreTrace


class EmptyFinishPairTests(unittest.TestCase):
    def test_pair_selects_no_slow_candidate(self):
        self.assertEqual(
            selected_variants(SimpleNamespace(finish_empty_pair=True)), ("empty1024",)
        )
        self.assertEqual(selected_variants(SimpleNamespace()), VARIANTS)

    def capture(self, guard):
        calls = []

        @contextlib.contextmanager
        def marker(name):
            calls.append("open")
            yield
            calls.append("close")

        events = [
            dict(ph="M", name="process_name", pid=1, args=dict(name="Ascend Hardware"))
        ]
        for i in range(4):
            events.extend(
                [
                    dict(ph="X", name=f"qsa/timed/{i}", pid=2, ts=i * 100, dur=80),
                    dict(
                        ph="X",
                        name="kernel",
                        pid=2,
                        ts=i * 100 + 10,
                        dur=3,
                        cat="cpu_op",
                    ),
                    dict(ph="X", name="kernel", pid=1, ts=i * 100 + 20, dur=5),
                ]
            )

        class Prof:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def export_chrome_trace(self, path):
                Path(path).write_text(json.dumps(events))

        api = SimpleNamespace(
            profile=lambda **kw: Prof(), ProfilerActivity=SimpleNamespace(CPU=0, NPU=1)
        )
        torch = SimpleNamespace(
            npu=SimpleNamespace(synchronize=lambda: calls.append("sync")),
            profiler=SimpleNamespace(record_function=marker),
        )
        result = dict(kernel_timer={})
        with tempfile.TemporaryDirectory() as root, patch.dict(
            "sys.modules", torch_npu=SimpleNamespace(profiler=api)
        ), patch(
            "qsa_int8_prefix_plan_timer_910b.time.sleep",
            side_effect=lambda _: calls.append("sleep"),
        ) as sleep:
            args = SimpleNamespace(
                rounds=2,
                kernel_active=1,
                kernel_warmup=0,
                report=root,
                case="pair",
                prefix_guard_ms=guard,
            )
            samples, orders = measure_prefix_plan(
                args,
                ["baseline", "candidate"],
                lambda *a: calls.append("launch"),
                dict(baseline=None, candidate=None),
                torch,
                result,
                expected_hosts=dict(baseline={"kernel": 1}, candidate={"kernel": 1}),
            )
            sleep_args = [c.args[0] for c in sleep.call_args_list]
        return calls, sleep_args, samples, orders, result

    def test_guard_is_inside_each_window_and_excluded_from_device_sums(self):
        calls, sleeps, samples, orders, result = self.capture(5)
        self.assertEqual(
            calls, ["open", "sleep", "launch", "sync", "sleep", "close"] * 4
        )
        self.assertEqual(sleeps, [0.005] * 8)
        self.assertEqual(orders, [["baseline", "candidate"], ["candidate", "baseline"]])
        for arm in samples.values():
            self.assertEqual([s["kernel_us"] for s in arm], [5, 5])
        self.assertEqual(result["single_capture_counts"]["tasks_per_window"], [1] * 4)
        self.assertEqual(result["kernel_timer"]["capture_guard_ms"], 5)

    def test_default_zero_guard_keeps_existing_launch_order(self):
        calls, sleeps, _, _, _ = self.capture(0)
        self.assertEqual(calls, ["open", "launch", "sync", "close"] * 4)
        self.assertEqual(sleeps, [])

    def test_uncontained_extra_or_missing_tasks_still_fail(self):
        base = [
            dict(ph="M", name="process_name", pid=1, args=dict(name="Ascend Hardware")),
            dict(ph="X", name="qsa/timed/0", pid=2, ts=100, dur=50),
        ]
        for ts, dur in [(99, 5), (148, 5), (200, 5)]:
            with self.assertRaises(IncompleteScoreTrace):
                stage_mean(
                    base + [dict(ph="X", name="kernel", pid=1, ts=ts, dur=dur)], 1
                )
        with self.assertRaises(IncompleteScoreTrace):
            stage_mean(base, 1)


if __name__ == "__main__":
    unittest.main()
