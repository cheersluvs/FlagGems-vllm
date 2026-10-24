# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Verify the single scheduling change and compile gate without a GPU."""

import ast
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools import qsa_prefill_bf16_dot_schedule_probe as probe


def function(path, name):
    return next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


class DotScheduleTests(unittest.TestCase):
    def test_only_high_dot_assignment_moves(self):
        probe.verify_source()
        before = function(
            probe.ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py",
            "_qsa_prefill_score_kernel",
        )
        after = function(probe.KERNEL, "_dot_schedule_score_kernel")
        loop = next(n for n in after.body if isinstance(n, ast.For))
        group = next(
            n
            for n in loop.body
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Name)
            and n.test.id == "GROUP_HEADS"
        )
        self.assertEqual(
            [n.targets[0].id for n in group.body],
            ["low", "high", "low", "high", "scores"],
        )
        group.body[1], group.body[2] = group.body[2], group.body[1]
        after.name, after.decorator_list = before.name, before.decorator_list
        self.assertEqual(ast.dump(after), ast.dump(before))

    def test_exact_activation_and_original_fallback(self):
        args = dict(IS_FP8=False, NUM_HEADS=4, HEAD_DIM=128, TILE_R=64, HEAD_GROUP=True)
        self.assertTrue(probe.eligible(args))
        for key, value in [
            ("IS_FP8", True),
            ("NUM_HEADS", 8),
            ("HEAD_DIM", 256),
            ("TILE_R", 32),
            ("HEAD_GROUP", False),
        ]:
            self.assertFalse(probe.eligible(dict(args, **{key: value})))
        original, candidate = MagicMock(), MagicMock()
        names = [*args, "num_rows", "query_offset", "logits_width"]
        router = probe.RoutedJIT(original, candidate, names)
        router["grid"](**args, num_rows=9, query_offset=3, logits_width=2048)
        candidate.__getitem__.assert_called_once_with("grid")
        original.__getitem__.assert_not_called()

    def test_schedule_gate_uses_actual_wait_order_not_source_order(self):
        serial = "\n".join(
            [
                "wgmma.mma_async.sync.aligned x;",
                "wgmma.commit_group.sync.aligned;",
                "wgmma.wait_group.sync.aligned 0;",
            ]
            * 2
        )
        overlap = (
            "wgmma.mma_async.sync.aligned x; wgmma.mma_async.sync.aligned x; "
            "wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;"
        )
        a, b = probe.schedule(serial), probe.schedule(overlap)
        self.assertEqual(a["mma_before_first_wait"], 1)
        self.assertEqual(b["mma_before_first_wait"], 2)
        record = dict(
            spill_count=0, registers_per_thread=207, shared_bytes=106496, schedule=a
        )
        after = dict(record, schedule=b)
        self.assertIsNone(probe.gate_reason(record, after))
        self.assertIn("no additional", probe.gate_reason(record, record))
        self.assertIn("spill", probe.gate_reason(record, dict(after, spill_count=1)))
        self.assertIn("spill", probe.gate_reason(record, dict(after, spill_count=None)))
        self.assertIn(
            "unknown", probe.gate_reason(record, dict(after, registers_per_thread=None))
        )
        self.assertIn(
            "missing",
            probe.gate_reason(record, dict(after, schedule=probe.schedule(""))),
        )
        self.assertIn(
            "work changed",
            probe.gate_reason(record, dict(after, schedule=dict(b, mma_sites=3))),
        )
        self.assertIn(
            "wait sites",
            probe.gate_reason(record, dict(after, schedule=dict(b, wait_sites=3))),
        )

    def test_worker_negative_probe_does_not_hide_numeric_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            options = types.SimpleNamespace(
                report_dir=folder, worker_case="synthetic", rounds=2
            )
            path = Path(folder) / "synthetic.json"

            def rejected(options):
                probe.LAST_REJECTION = [{"reason": "no overlap"}]
                path.write_text(
                    json.dumps(dict(status="failed", stages={}, error="compile gate"))
                )
                return 1

            with patch.object(probe.base, "worker", side_effect=rejected), patch.object(
                probe, "show"
            ):
                self.assertEqual(probe.worker(options), 0)
                self.assertEqual(
                    json.loads(path.read_text())["status"], "not_promising"
                )

            def numeric_fail(options):
                path.write_text(
                    json.dumps(dict(status="failed", stages={}, error="wrong scores"))
                )
                return 1

            with patch.object(
                probe.base, "worker", side_effect=numeric_fail
            ), patch.object(probe, "show"):
                self.assertEqual(probe.worker(options), 1)
                self.assertEqual(json.loads(path.read_text())["error"], "wrong scores")

    def test_frozen_trace_preserves_all_configs(self):
        for item in probe.manifest()["cases"]:
            traces = item["traces"]
            self.assertEqual(traces, probe.FIXED_TRACES(traces, False))
        self.assertEqual(len(probe.DEFAULT_CASES), 2)
        self.assertTrue(
            all(
                c in [v["case"]["name"] for v in probe.manifest()["cases"]]
                for c in probe.DEFAULT_CASES
            )
        )

    def test_negative_probe_saves_evidence_before_no_timing(self):
        args = dict(IS_FP8=False, NUM_HEADS=4, HEAD_DIM=128, TILE_R=64, HEAD_GROUP=True)
        key = (17, 0, 1088)
        kernel = types.SimpleNamespace(
            asm=dict(
                ptx="wgmma.mma_async.sync.aligned x; wgmma.wait_group.sync.aligned 0;"
            )
        )
        jit = types.SimpleNamespace(decisions={key: True})
        entry = types.SimpleNamespace(jit=jit, compiled={key: (kernel, {})})
        empty = types.SimpleNamespace(compiled={})
        module = types.SimpleNamespace(
            _qsa_prefill_score_kernel=entry,
            _qsa_prefill_compact_score_kernel=empty,
            _qsa_prefill_fp8_store_score_kernel=empty,
        )
        record = dict(
            chunk=list(key),
            route="rectangular",
            constants=args,
            spill_count=0,
            registers_per_thread=207,
            shared_bytes=106496,
        )
        with tempfile.TemporaryDirectory() as folder, patch.object(
            probe, "RESOURCE_DIR", Path(folder)
        ), patch.object(probe, "RESOURCE_PREFIX", "unit"), patch.object(
            probe.stage,
            "FREEZE",
            return_value=dict(
                baseline=[copy.deepcopy(record)], candidate=[copy.deepcopy(record)]
            ),
        ), patch.object(
            probe.base, "artifact_records", return_value=[]
        ) as artifacts, patch.object(
            probe, "LAST_REJECTION", None
        ):
            with self.assertRaisesRegex(RuntimeError, "NOT_PROMISING"):
                probe.freeze_records((module, module))
            artifacts.assert_called_once()
            saved = list(Path(folder).glob("compile-unit-*.json"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(len(probe.LAST_REJECTION), 1)
            self.assertFalse(json.loads(saved[0].read_text())["boundary"])


if __name__ == "__main__":
    unittest.main()
