# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Selective routing and exact original-binary fallback contracts."""

import copy
import types
import unittest
from unittest.mock import patch

from tools import qsa_prefill_bf16_query_policy as policy
from tools import qsa_prefill_query_split_matched as fixed


class PolicyTests(unittest.TestCase):
    def config(self, rows, tiles, parts=1):
        return dict(
            TILE_R=rows,
            BLOCK_N=64,
            K_TILES=tiles,
            num_warps=4,
            num_stages=1,
            num_ctas=1,
            QUERY_PARTS=parts,
        )

    def test_parameter_gate_is_joint_and_legacy_behavior_is_retained(self):
        for rows, tiles, expected in (
            (64, 16, 2),
            (32, 4, 1),
            (64, 4, 1),
            (32, 16, 1),
            (16, 16, 1),
        ):
            config = self.config(rows, tiles)
            self.assertEqual(policy.parts(config, "large-tile"), expected)
            self.assertEqual(policy.parts(config, "large-tile", "baseline"), 1)
            self.assertEqual(policy.parts(config, "all"), 2 if rows >= 32 else 1)
        with self.assertRaises(ValueError):
            policy.parts(self.config(64, 16), "unknown")

    def test_proxy_uses_the_exact_original_jit_and_preserves_arguments(self):
        calls = []

        class JIT:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    calls.append((self.name, grid, args, kwargs))
                    return self.name

                return run

        baseline, split = JIT("original"), JIT("split")
        proxy = policy.RoutedJIT(baseline, split)
        self.assertIs(proxy.baseline, baseline)
        token = object()
        self.assertEqual(proxy[(2, 3)](token, QUERY_PARTS=1, TILE_R=32), "original")
        self.assertEqual(proxy[(2, 3)](token, QUERY_PARTS=2, TILE_R=64), "split")
        self.assertIs(calls[0][2][0], token)
        self.assertEqual(calls[0][1], (2, 3))
        with self.assertRaises(RuntimeError):
            proxy[(1,)](QUERY_PARTS=3)

    def pair(self):
        entries = []
        for side in range(2):
            configs = {
                (10, 0, 128): self.config(64, 16, 1 if side == 0 else 2),
                (5, 10, 128): self.config(32, 4, 1 if side == 0 else 2),
            }
            entries.append(
                types.SimpleNamespace(
                    _qsa_prefill_score_kernel=types.SimpleNamespace(
                        configs=configs, jit=object()
                    )
                )
            )
        return tuple(entries)

    def test_apply_changes_only_candidate_mode_and_preserves_original_entry(self):
        pair = self.pair()
        original = copy.deepcopy(pair[0]._qsa_prefill_score_kernel.configs)
        jit = pair[0]._qsa_prefill_score_kernel.jit
        policy.apply(pair, "large-tile")
        self.assertEqual(pair[0]._qsa_prefill_score_kernel.configs, original)
        self.assertIs(pair[1]._qsa_prefill_score_kernel.jit.baseline, jit)
        self.assertEqual(
            pair[1]._qsa_prefill_score_kernel.configs[(5, 10, 128)]["QUERY_PARTS"], 1
        )
        self.assertEqual(
            pair[1]._qsa_prefill_score_kernel.configs[(10, 0, 128)]["QUERY_PARTS"], 2
        )
        bad = self.pair()
        bad[1]._qsa_prefill_score_kernel.configs[(5, 10, 128)]["num_stages"] = 2
        with self.assertRaises(RuntimeError):
            policy.apply(bad, "large-tile")

    def records(self):
        left = [
            dict(
                chunk=[10, 0, 128],
                constants=self.config(64, 16),
                cubin_sha256="a",
                ptx_sha256="b",
            ),
            dict(
                chunk=[5, 10, 128],
                constants=self.config(32, 4),
                cubin_sha256="c",
                ptx_sha256="d",
            ),
        ]
        right = copy.deepcopy(left)
        right[0]["constants"]["QUERY_PARTS"] = 2
        right[0]["cubin_sha256"] = "new"
        right[0]["ptx_sha256"] = "new"
        return left, right

    def test_freeze_accepts_mixed_modes_but_rejects_changed_fallback_binary(self):
        left, right = self.records()

        def run(bad=False):
            pair = self.pair()
            values = [left, copy.deepcopy(right)]
            if bad:
                values[1][1]["cubin_sha256"] = "wrong-original"

            def record(modules):
                i = 0 if modules[0] is pair[0] else 1
                return dict(baseline=values[i])

            with patch.object(fixed, "freeze_and_records", side_effect=record):
                return policy.freeze(pair, "large-tile")

        result = run()
        self.assertTrue(result["configs_match"])
        self.assertEqual(result["query_parts_pair"], [1, "gated"])
        with self.assertRaisesRegex(RuntimeError, "original P1 binary"):
            run(True)
        right[1]["cubin_sha256"] = ""
        with self.assertRaises(RuntimeError):
            policy.verify_fallback(left, right, "large-tile")


if __name__ == "__main__":
    unittest.main()
