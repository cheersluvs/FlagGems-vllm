# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""No-GPU prelaunch executable gates, routing and unchanged fallback coverage."""

import copy
import types
import unittest
from unittest.mock import patch

from tools import qsa_prefill_fp8_dot_runtime as p


class FakeJIT:
    def __init__(self, kernel):
        self.kernel, self.warmups, self.launches = kernel, [], []

    def warmup(self, *args, **kwargs):
        self.warmups.append((args, kwargs))
        return self.kernel

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.launches.append((grid, args, kwargs))
            return self.kernel

        return run


class DotRuntimeTests(unittest.TestCase):
    def kernel(self):
        return types.SimpleNamespace(
            asm={"ptx": "wgmma.wait_group.sync.aligned 0;"},
            metadata=types.SimpleNamespace(shared=36864),
            n_regs=114,
            n_spills=0,
        )

    def expected(self, kernel):
        return dict(
            code_sha256=p.fp8.runtime.code_sha(kernel.asm["ptx"]),
            shared_bytes=36864,
            registers_per_thread=114,
        )

    def named(self):
        return dict(IS_FP8=True, NUM_HEADS=4, HEAD_DIM=128, **p.fp8.probe.CONFIG)

    def test_only_main_tile_changes_and_input_traces_are_immutable(self):
        original = [
            dict(route="fp8", constants=dict(p.fp8.probe.CONFIG)),
            dict(
                route="fp8", constants=dict(p.fp8.probe.CONFIG, TILE_R=8, QUERY_PARTS=1)
            ),
        ]
        old = copy.deepcopy(original)
        candidate = p.traces_for(original, "paired32")
        self.assertEqual(original, old)
        self.assertEqual(candidate[1], old[1])
        self.assertEqual(
            candidate[0]["constants"], dict(old[0]["constants"], TILE_R=32)
        )
        self.assertEqual(p.traces_for(original, "production"), original)
        for k, v in (
            ("IS_FP8", False),
            ("HEAD_DIM", 256),
            ("NUM_HEADS", 8),
            ("QUERY_PARTS", 1),
        ):
            self.assertFalse(p.eligible(dict(self.named(), **{k: v})))

    def test_prelaunch_gate_rejects_every_executable_resource_mismatch(self):
        kernel = self.kernel()
        expected = self.expected(kernel)
        p.check_kernel(kernel, expected)
        for field, value in (("n_regs", 115), ("n_spills", 1)):
            bad = copy.deepcopy(kernel)
            setattr(bad, field, value)
            with self.assertRaises(RuntimeError):
                p.check_kernel(bad, expected)
        bad = copy.deepcopy(kernel)
        bad.metadata.shared += 1
        with self.assertRaises(RuntimeError):
            p.check_kernel(bad, expected)
        bad = copy.deepcopy(kernel)
        bad.asm["ptx"] += "\nbar.sync 0;"
        with self.assertRaises(RuntimeError):
            p.check_kernel(bad, expected)

    def test_mismatch_never_launches_candidate_and_frozen_new_chunk_rejected(self):
        kernel = self.kernel()
        target, original = FakeJIT(kernel), FakeJIT(kernel)
        names = ["num_rows", "query_offset", "logits_width"]
        proxy = p.SourceJIT(original, target, names, self.expected(kernel))
        config = dict(self.named(), TILE_R=32)
        with patch.object(p, "check_kernel", side_effect=RuntimeError("mismatch")):
            with self.assertRaises(RuntimeError):
                proxy[(1, 2, 3)](35, 5, 8256, **config)
        self.assertFalse(target.launches)
        self.assertFalse(proxy.prepared)
        proxy[(1, 2, 3)](35, 5, 8256, **config)
        self.assertEqual(len(target.launches), 1)
        proxy.frozen = True
        proxy[(1, 2, 3)](35, 5, 8256, **config)
        with self.assertRaises(RuntimeError):
            proxy[(1, 2, 3)](36, 5, 8256, **config)
        self.assertEqual(len(target.launches), 2)
        proxy[(1, 2, 3)](113, 5, 8256, **dict(config, TILE_R=8, QUERY_PARTS=1))
        self.assertEqual(len(original.launches), 1)

    def test_fixed_entry_recomputes_grid_with_smaller_tile(self):
        kernel = self.kernel()
        target, original = FakeJIT(kernel), FakeJIT(kernel)
        names = ["num_rows", "query_offset", "logits_width"]
        auto_entry = types.SimpleNamespace(
            arg_names=names, fn=types.SimpleNamespace(fn=original)
        )
        traces = [
            dict(
                route="fp8",
                rows=64,
                offset=0,
                width=8256,
                constants=dict(p.fp8.probe.CONFIG),
            )
        ]
        entry = p.base.fixed.FixedEntry(
            auto_entry, p.traces_for(traces, "paired32"), "fp8"
        )
        entry.jit = p.SourceJIT(original, target, names, self.expected(kernel))
        grid = lambda meta: (1, (64 + meta["TILE_R"] - 1) // meta["TILE_R"], 9)
        entry[grid](64, 0, 8256, IS_FP8=True, NUM_HEADS=4, HEAD_DIM=128)
        self.assertEqual(target.launches[0][0], (1, 2, 9))
        self.assertEqual(target.launches[0][2]["TILE_R"], 32)
        self.assertEqual(traces[0]["constants"]["TILE_R"], 64)

    def test_unreviewed_ir_rejected_before_protocol_model(self):
        with self.assertRaisesRegex(RuntimeError, "Unreviewed"):
            p.lifetime("changed", "changed", "paired32")


if __name__ == "__main__":
    unittest.main()
