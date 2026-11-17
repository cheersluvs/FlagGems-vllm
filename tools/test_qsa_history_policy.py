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

"""Check conservative selection,real dispatch and fail-closed full-input summaries."""

import ast
import copy
import inspect
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_qsa_raw_tle_sort_harness as fixed_tests
from qsa_history_policy_910b import (
    boundary_cases,
    choose_policy,
    dispatcher,
    preflight,
    use_prefix,
)
from qsa_raw_history_merge_910b import harness_source
from run_qsa_raw_serial_harness_910b import (
    FIXED_OP_NAME,
    HISTORY_OP_NAME,
    cases,
    summarize,
)


def boundary_data():
    result = dict(status="passed", mode="history_boundary_tuning", cases=[])
    for rows, width in boundary_cases():
        blocks = (width + 511) // 512
        ratio = 0.9 if blocks == 1 else 0.99 if blocks == 2 else 1.1
        check = dict(
            passed=True,
            trace_scores_exact=True,
            selected_scores_exact=True,
            output_guards_unchanged=True,
        )
        result["cases"].append(
            dict(
                rows=rows,
                width=width,
                status="passed",
                precheck={n: dict(check) for n in ("snapshot", "prefix")},
                postcheck={n: dict(check) for n in ("snapshot", "prefix")},
                single_capture_counts=dict(
                    expected_host_counts=dict(s=6, p=6),
                    observed_host_counts=dict(s=6, p=6),
                    tasks_per_window=[1] * 12,
                    expected_tasks_per_window=[1] * 12,
                ),
                history_ir=dict(
                    prefix={
                        t: dict(
                            gather_references=0,
                            calls=dict(
                                sort32=1,
                                mrgsort=2,
                                merge_exhaust_sort4=1,
                                unpack_sort=1,
                            ),
                        )
                        for t in ("False", "True")
                    }
                ),
                samples=dict(
                    snapshot=[dict(kernel_us=100 * ratio) for _ in range(2)],
                    prefix=[dict(kernel_us=100) for _ in range(2)],
                ),
            )
        )
    return result


class PolicyTests(unittest.TestCase):
    def test_threshold_checks_every_larger_case_and_both_rounds(self):
        data = boundary_data()
        policy = choose_policy(data)
        self.assertEqual(policy["threshold"], 4)
        self.assertEqual(policy["max_blocks"], 58)
        row = next(c for c in data["cases"] if c["rows"] == 128 and c["width"] == 4096)
        row["samples"]["snapshot"][1]["kernel_us"] = 101
        self.assertEqual(choose_policy(data)["threshold"], 17)
        data["cases"][-1]["samples"]["snapshot"][0]["kernel_us"] = 101
        with self.assertRaisesRegex(ValueError, "No qualified"):
            choose_policy(data)

    def test_missing_checks_counts_or_nonfinite_timing_rejected(self):
        for kind in ("missing", "counts", "nan", "ir"):
            data = boundary_data()
            if kind == "missing":
                data["cases"].pop()
            elif kind == "counts":
                data["cases"][0]["single_capture_counts"]["tasks_per_window"] = []
            elif kind == "nan":
                data["cases"][0]["samples"]["prefix"][0]["kernel_us"] = float("nan")
            else:
                data["cases"][0]["history_ir"]["prefix"] = {}
            with self.assertRaises(ValueError):
                choose_policy(data)

    def test_dispatch_uses_measured_range_for_both_kernel_types(self):
        policy = choose_policy(boundary_data())
        module = SimpleNamespace(
            **{
                n: n
                for n in (
                    "raw_history_prepared_snapshot",
                    "raw_history_prepared_prefix",
                    "raw_history_topk_snapshot",
                    "raw_history_topk_prefix",
                )
            }
        )
        select = dispatcher(module, policy)
        for blocks in (1, 2, 3, 4, 8, 17, 58, 59, 200):
            prefix = 4 <= blocks <= 58
            self.assertEqual(use_prefix(blocks, policy), prefix)
            kernel, online, evidence, mode = select(blocks)
            expected = "prefix" if prefix else "snapshot"
            self.assertEqual(mode, expected)
            self.assertEqual(kernel, "raw_history_prepared_" + expected)
            self.assertEqual(online, "raw_history_topk_" + expected)
            self.assertEqual(evidence.keywords["expect_no_gather"], prefix)

    def test_preflight_inherits_compiler_and_retains_evidence_before_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)

            def worker(command, root, env, log, timeout):
                self.assertEqual(
                    env["QSA_CORE_MODE_FIX_MANIFEST"], "/test/compiler.json"
                )
                self.assertEqual(env["PYTHONPATH"], "/test/compiler-copy")
                self.assertIn("--history-boundary", command)
                self.assertEqual(command[command.index("--rounds") + 1], "2")
                dest = Path(command[command.index("--report") + 1])
                (dest / "raw-cube-chain.json").write_text(json.dumps(boundary_data()))
                (dest / "compiled-ir.txt").write_text("IR proof")
                return 0

            with patch.dict(
                os.environ,
                QSA_CORE_MODE_FIX_MANIFEST="/test/compiler.json",
                PYTHONPATH="/test/compiler-copy",
            ), patch(
                "run_qsa_raw_serial_harness_910b.run_worker", side_effect=worker
            ), patch(
                "subprocess.check_output", return_value="a" * 40
            ), redirect_stdout(
                StringIO()
            ):
                policy = preflight(report, report, 0)
            self.assertEqual(policy["threshold"], 4)
            self.assertEqual(
                (report / "boundary-compiled-ir.txt").read_text(), "IR proof"
            )
            self.assertTrue((report / "history-policy.json").is_file())
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)
            with patch("run_qsa_raw_serial_harness_910b.run_worker", return_value=1):
                with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                    preflight(report, report, 0)
            self.assertFalse((report / "history-policy.json").exists())

    def test_runtime_signatures_and_real_history_helpers_preserved(self):
        fs = {
            n.name: n
            for n in ast.parse(harness_source()).body
            if isinstance(n, ast.FunctionDef)
        }
        for name in (
            "raw_history_prepared_snapshot",
            "raw_history_prepared_prefix",
            "raw_history_topk_snapshot",
            "raw_history_topk_prefix",
        ):
            fn = fs[name]
            text = ast.unparse(fn)
            helper = (
                "_history_prefix"
                if name.endswith("prefix")
                else "_update_online_topk_ub_snapshot"
            )
            self.assertIn(helper, text)
            self.assertIn("bind_buffer=update_state_slot", text)
            cpu = copy.deepcopy(fn)
            cpu.decorator_list = []
            cpu.body = [ast.Pass()]
            for a in cpu.args.args:
                a.annotation = None
            env = {}
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[cpu], type_ignores=[])),
                    "sig",
                    "exec",
                ),
                env,
            )
            kwargs = dict(ROWS=17, WIDTH=8256, BLOCKS=17)
            count = 4 if "_topk_" in name else 9
            if count == 9:
                kwargs.update(SLOTS=2, TRACE=False)
            inspect.signature(env[name]).bind(
                *[object() for _ in range(count)], **kwargs
            )

    def populate(self, root):
        outcomes = fixed_tests.FixedHarnessTests().populate(root)
        data = boundary_data()
        policy = choose_policy(data)
        policy["source_code_commit"] = "a" * 40
        (root / "history-policy.json").write_text(json.dumps(policy))
        (root / "boundary-raw-cube-chain.json").write_text(json.dumps(data))
        for plan in cases():
            name = plan["name"]
            path = root / (name + "-all-score.json")
            d = json.loads(path.read_text())
            c = d["serial_harness"]
            c["mixed_identity"]["history"] = "prefix_by_threshold"
            c["generic_identity"].update(
                history="snapshot", local_sort="sort32_mrgsort_fixed512"
            )
            c["history_policy"] = policy
            c["mixed_routes"] = c["generic_routes"] = [
                dict(first=0, count=17, blocks=2),
                dict(first=17, count=17, blocks=8),
            ]
            c["history_dispatch"] = [
                dict(first=0, count=17, blocks=2, mode="snapshot"),
                dict(first=17, count=17, blocks=8, mode="prefix"),
            ]
            for key in ("mixed_compiled_artifacts", "generic_compiled_artifacts"):
                c[key] = [
                    dict(
                        history_mode=mode,
                        primitive_ir=dict(
                            gather_references=0 if mode == "prefix" else 2,
                            calls=dict(
                                sort32=1,
                                mrgsort=2,
                                merge_exhaust_sort4=1,
                                unpack_sort=1,
                            ),
                        ),
                    )
                    for mode in ("snapshot", "prefix")
                ]
            path.write_text(json.dumps(d))
            path = root / ("harness-" + name + ".json")
            d = json.loads(path.read_text())
            d[HISTORY_OP_NAME] = d.pop(FIXED_OP_NAME)
            d[HISTORY_OP_NAME]["details"][0]["result"][0]["shape_detail"][
                1
            ] = "history_prefix_slots2"
            path.write_text(json.dumps(d))
        return outcomes

    def test_summary_requires_policy_evidence_and_exact_piece_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcomes = self.populate(root)
            result = summarize(root, outcomes, mixed=True, fixed=True, history=True)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["arithmetic_mean_speedup"], 1.5)
            self.assertAlmostEqual(result["arithmetic_mean_snapshot_over_prefix"], 1.04)
            path = root / (cases()[0]["name"] + "-all-score.json")
            d = json.loads(path.read_text())
            d["serial_harness"]["history_dispatch"][0]["mode"] = "prefix"
            path.write_text(json.dumps(d))
            failed = summarize(root, outcomes, mixed=True, fixed=True, history=True)
            self.assertEqual(failed["status"], "incomplete")
            self.assertNotIn("arithmetic_mean_speedup", failed)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcomes = self.populate(root)
            policy = json.loads((root / "history-policy.json").read_text())
            policy["threshold"] = 1
            (root / "history-policy.json").write_text(json.dumps(policy))
            self.assertEqual(
                summarize(root, outcomes, mixed=True, fixed=True, history=True)[
                    "status"
                ],
                "incomplete",
            )


if __name__ == "__main__":
    unittest.main()
