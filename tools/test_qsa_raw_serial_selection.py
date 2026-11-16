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

"""Check serial chunk ordering and fail-closed stock harness aggregation."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsa_raw_serial_selection_910b_trial import SerialRawSelection  # noqa: E402
from run_qsa_raw_serial_harness_910b import (  # noqa: E402
    OP_NAME,
    cases,
    main,
    run_worker,
    summarize,
)


class SerialSelectionTests(unittest.TestCase):
    def setup_raw(self):
        log = []

        class Scores:
            def __getitem__(self, index):
                return index.start, index.stop

        def select(first, target, scratch):
            log.append(("topk", first, target, scratch))

        adapter = SimpleNamespace(
            output="output", _new_topk_scratch=lambda: "scratch", select_chunk=select
        )

        def piece(name):
            return dict(execute=lambda mode: log.append((name, mode)))

        jobs = [
            (dict(first=0, count=5), [piece("r0"), piece("r1")]),
            (dict(first=5, count=3), [piece("r2")]),
        ]
        return (
            SerialRawSelection(dict(adapter=adapter, jobs=jobs, scores=Scores())),
            log,
        )

    def test_topk_runs_after_all_request_pieces_with_original_chunk_offsets(self):
        raw, log = self.setup_raw()
        self.assertEqual(raw(), "output")
        self.assertEqual(
            log,
            [
                ("r0", "empty1024"),
                ("r1", "empty1024"),
                ("topk", 0, (None, 5), "scratch"),
                ("r2", "empty1024"),
                ("topk", 5, (None, 3), "scratch"),
            ],
        )

    def test_score_only_never_calls_topk(self):
        raw, log = self.setup_raw()
        raw.score_all()
        self.assertEqual(
            log, [("r0", "empty1024"), ("r1", "empty1024"), ("r2", "empty1024")]
        )

    def populate(self, root):
        outcomes = {}
        for plan in cases():
            name = plan["name"]
            outcomes[name] = 0
            detail = dict(
                status="passed",
                precheck=dict(passed=True),
                postcheck=dict(passed=True),
                serial_harness=dict(
                    precheck=dict(passed=True),
                    postcheck=dict(passed=True, raw_current_changed_index_sets=7),
                    stage_ms=dict(
                        current_combined=2,
                        current_score=0.5,
                        raw_score=0.3,
                        raw_topk=0.4,
                    ),
                ),
            )
            (root / (name + "-all-score.json")).write_text(json.dumps(detail))
            bench = {
                "qsa_select_paged_prefill_int8_910b": dict(
                    result="passed",
                    test_case=f"benchmark/test_qsa_raw_serial_int8_perf.py::test_qsa_raw_serial_kernel[{name}]",
                ),
                OP_NAME: dict(
                    details=[
                        dict(
                            mode="kernel",
                            dtype="torch.int8",
                            result=[
                                dict(
                                    shape_detail=[name, "auto"],
                                    latency_base=1.5,
                                    latency=1,
                                    speedup=1.5,
                                )
                            ],
                        )
                    ],
                ),
            }
            (root / ("harness-" + name + ".json")).write_text(json.dumps(bench))
        return outcomes

    def test_summary_requires_all_ten_and_postchecks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcomes = self.populate(root)
            result = summarize(root, outcomes)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["arithmetic_mean_speedup"], 1.5)
            name = cases()[0]["name"]
            path = root / (name + "-all-score.json")
            data = json.loads(path.read_text())
            data["serial_harness"]["postcheck"]["passed"] = False
            path.write_text(json.dumps(data))
            result = summarize(root, outcomes)
            self.assertEqual(result["status"], "incomplete")
            self.assertNotIn("arithmetic_mean_speedup", result)

    def test_summary_rejects_bad_latency_ratio_or_missing_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcomes = self.populate(root)
            name = cases()[0]["name"]
            path = root / ("harness-" + name + ".json")
            data = json.loads(path.read_text())
            data[OP_NAME]["details"][0]["result"][0]["speedup"] = 3
            path.write_text(json.dumps(data))
            self.assertEqual(summarize(root, outcomes)["status"], "incomplete")
            del outcomes[name]
            self.assertEqual(summarize(root, outcomes)["status"], "incomplete")

    def test_recovery_never_launches_a_worker_and_preserves_original_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            dest = root / "dest"
            source.mkdir()
            outcomes = self.populate(source)
            old = dict(
                status="incomplete",
                cases=[dict(case=n, exit_code=rc) for n, rc in outcomes.items()],
            )
            (source / "serial-harness-summary.json").write_text(json.dumps(old))
            argv = [
                "summary",
                "--serial-harness",
                "--summarize-only",
                "--source-report",
                str(source),
                "--report",
                str(dest),
            ]
            with patch.object(sys, "argv", argv), patch(
                "run_qsa_raw_serial_harness_910b.run_worker",
                side_effect=AssertionError("must not execute"),
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            self.assertEqual(
                json.loads((dest / "original-serial-harness-summary.json").read_text()),
                old,
            )
            self.assertTrue(
                json.loads((dest / "serial-harness-summary.json").read_text())[
                    "recovered_without_npu_execution"
                ]
            )

    def test_rejects_wrong_outcome_test_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcomes = self.populate(root)
            name = cases()[0]["name"]
            path = root / ("harness-" + name + ".json")
            data = json.loads(path.read_text())
            data["qsa_select_paged_prefill_int8_910b"]["test_case"] = "other_test"
            path.write_text(json.dumps(data))
            self.assertEqual(summarize(root, outcomes)["status"], "incomplete")

    def test_worker_streams_and_retains_log(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(
            io.StringIO()
        ) as captured:
            root = Path(tmp)
            log = root / "worker.txt"
            rc = run_worker(
                [sys.executable, "-c", "print('stage=1.25 ms',flush=True)"],
                root,
                dict(),
                log,
                10,
            )
            self.assertEqual(rc, 0)
            self.assertIn("stage=1.25 ms", log.read_text())
            self.assertIn("stage=1.25 ms", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
