# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Actual all-case applicability and independent source-specialization gates."""

import ast
import copy
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools import qsa_prefill_fp8_dot_all as p
from tools.test_qsa_prefill_fp8_dot_runtime import FakeJIT

IR = "\n".join(
    [
        "%a = ttng.warp_group_dot %k, %q : !ttg.memdesc<64x128xf8E4M3FN> -> tensor<64x64xf32, #mma>",
        "%b = ttng.warp_group_dot %k, %r : !ttg.memdesc<64x128xf8E4M3FN> -> tensor<64x64xf32, #mma>",
    ]
)


def kernel(paired):
    waits = (1, 0) if paired else (0, 0)
    return SimpleNamespace(
        asm=dict(
            ttgir=IR,
            ptx="\n".join(f"wgmma.wait_group.sync.aligned {n};" for n in waits),
            cubin=b"test",
        ),
        metadata=SimpleNamespace(shared=36864, num_warps=4, name="test"),
        n_regs=114 if paired else 94,
        n_spills=0,
    )


class AllFP8Tests(unittest.TestCase):
    def test_all10_cases_have_actual_targets_including_compact(self):
        applied = []
        for item in p.full.manifest()["cases"]:
            traces = item["traces"]
            before = copy.deepcopy(traces)
            target = [t for t in traces if p.target(t)]
            new = p.candidate_traces(traces)
            self.assertEqual(traces, before)
            if target:
                applied.append(item["case"]["name"])
            for old, candidate in zip(traces, new):
                expected = dict(old["constants"])
                if p.target(old) and old["route"] == "fp8":
                    expected.update(TILE_R=32, QUERY_PARTS=2)
                self.assertEqual(candidate["constants"], expected)
        self.assertEqual(len(applied), 10)
        self.assertIn("fp8_compact_float8_e4m3fn", applied)
        self.assertIn("fp8_first_chunk_float8_e4m3fn", applied)
        self.assertIn("fp8_short_tail_float8_e4m3fn", applied)
        self.assertIn("fp8_two_skewed_float8_e4m3fn", applied)

    def test_compact_source_keeps_grid_loads_and_moves_only_independent_dot(self):
        a, b = p.compact_source(False), p.compact_source(True)
        moved = (
            "        scores_right = tl.dot(query_right, keys, out_dtype=tl.float32)\n"
        )
        self.assertEqual(a.replace(moved, ""), b.replace(moved, ""))
        for text in (a, b):
            self.assertEqual(text.count("tl.dot("), 2)
            self.assertEqual(text.count("tl.store("), 2)
            ast.parse(text)
        original_text = (p.ROOT / p.runtime.probe.compiler.OP).read_text()
        old = next(
            n
            for n in ast.parse(original_text).body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_qsa_prefill_compact_score_kernel"
        )
        new = next(n for n in ast.parse(b).body if isinstance(n, ast.FunctionDef))
        stop = next(
            i
            for i, n in enumerate(old.body)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "query"
        )
        self.assertEqual(
            [ast.dump(n) for n in old.body[: stop + 1]],
            [ast.dump(n) for n in new.body[: stop + 1]],
        )
        self.assertEqual(
            b.count("tl.load("),
            ast.get_source_segment(original_text, old).count("tl.load("),
        )
        # Row-major Q[32*4,128] reshaped to[2,64,128],then permuted to
        # [64,128,2] and split,reconstructs exactly the original row/head order.
        for row in range(32):
            for head in range(4):
                half, local_row = divmod(row, 16)
                for dim in (0, 31, 127):
                    split_offset = (half * 64 + local_row * 4 + head) * 128 + dim
                    self.assertEqual(split_offset, (row * 4 + head) * 128 + dim)

    def test_compact_proxy_uses_separate_split_control_without_query_parts_argument(
        self,
    ):
        fallback, control, candidate = (
            FakeJIT(kernel(False)),
            FakeJIT(kernel(False)),
            FakeJIT(kernel(True)),
        )
        proxy = p.SourceJIT(
            fallback,
            candidate,
            ["num_rows", "query_offset", "logits_width"],
            {(8896, 0, 1088)},
            {},
            control=control,
            compact=True,
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"QSA_DOT_ALL_REPORT": tmp, "QSA_DOT_ALL_CASE": "compact"}
        ):
            proxy[(330, 1, 5)](
                8896,
                0,
                1088,
                IS_FP8=True,
                NUM_HEADS=4,
                HEAD_DIM=128,
                TILE_R=32,
                num_requests=54,
                K_TILES=4,
                num_stages=2,
            )
        self.assertEqual(len(control.warmups), 1)
        self.assertFalse(fallback.warmups)
        self.assertEqual(len(candidate.launches), 1)

    def test_compile_gate_records_overlap_and_rejects_work_spill_or_unknown_wait(self):
        control, candidate = kernel(False), kernel(True)
        self.assertTrue(p.compile_gate(control, candidate)["overlap_retained"])
        self.assertFalse(p.compile_gate(control, control)["overlap_retained"])
        for field, value in (("n_spills", 1), ("n_regs", None)):
            bad = copy.deepcopy(candidate)
            setattr(bad, field, value)
            with self.assertRaises(RuntimeError):
                p.compile_gate(control, bad)
        bad = copy.deepcopy(candidate)
        bad.asm["ttgir"] = IR.replace("64x64xf32", "64x128xf32")
        with self.assertRaises(RuntimeError):
            p.compile_gate(control, bad)
        bad = copy.deepcopy(candidate)
        bad.asm["ptx"] = "wgmma.wait_group.sync.aligned 2;"
        with self.assertRaises(RuntimeError):
            p.compile_gate(control, bad)

    def test_new_multi_request_k4_specialization_is_prepared_before_execution(self):
        control, candidate = FakeJIT(kernel(False)), FakeJIT(kernel(True))
        names = ["num_rows", "query_offset", "logits_width"]
        proxy = p.SourceJIT(control, candidate, names, {(4519, 0, 8192)}, {})
        config = dict(
            IS_FP8=True,
            NUM_HEADS=4,
            HEAD_DIM=128,
            TILE_R=32,
            QUERY_PARTS=2,
            num_requests=2,
            K_TILES=4,
            num_stages=1,
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"QSA_DOT_ALL_REPORT": tmp, "QSA_DOT_ALL_CASE": "new"}
        ):
            proxy[(2, 142, 8)](4519, 0, 8192, **config)
            self.assertEqual(len(control.warmups), 1)
            self.assertEqual(len(candidate.warmups), 1)
            self.assertEqual(len(candidate.launches), 1)
            self.assertTrue(proxy.prepared[(4519, 0, 8192)]["files"])
            proxy.frozen = True
            proxy[(2, 142, 8)](4519, 0, 8192, **config)
            self.assertEqual(len(candidate.warmups), 1)
        proxy.targets.add((4520, 0, 8192))
        with self.assertRaises(RuntimeError):
            proxy[(2, 142, 8)](4520, 0, 8192, **config)
        self.assertEqual(len(candidate.launches), 2)

    def test_failed_new_specialization_does_not_execute_candidate(self):
        control, candidate = FakeJIT(kernel(False)), FakeJIT(kernel(True))
        candidate.kernel.n_spills = 1
        proxy = p.SourceJIT(
            control,
            candidate,
            ["num_rows", "query_offset", "logits_width"],
            {(4519, 0, 8192)},
            {},
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"QSA_DOT_ALL_REPORT": tmp, "QSA_DOT_ALL_CASE": "failed"}
        ), self.assertRaises(RuntimeError):
            proxy[(1, 142, 8)](
                4519,
                0,
                8192,
                IS_FP8=True,
                NUM_HEADS=4,
                HEAD_DIM=128,
                TILE_R=32,
                QUERY_PARTS=2,
                num_requests=2,
                K_TILES=4,
                num_stages=1,
            )
        self.assertFalse(candidate.launches)
        self.assertFalse(proxy.prepared)


if __name__ == "__main__":
    unittest.main()
