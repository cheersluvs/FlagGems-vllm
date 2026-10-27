# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Full matrix identity and explicit single-request candidate/fallback scope."""

import os
import unittest
from unittest.mock import patch

from tools import qsa_prefill_fp8_dot_full as p


class FullFP8Tests(unittest.TestCase):
    def test_all10_fixtures_and_changed5_controls5_are_explicit(self):
        data = p.manifest()
        self.assertEqual(len(data["cases"]), 10)
        changed = [
            c["case"]["name"].split("_float8")[0] for c in data["cases"] if p.active(c)
        ]
        self.assertEqual(
            changed,
            [
                "fp8_budget_exact",
                "fp8_dominant_long",
                "fp8_tail112",
                "fp8_tail113",
                "fp8_long100k",
            ],
        )
        profiled = p.TWO_MANIFEST()
        for old in profiled["cases"]:
            actual = next(v for v in data["cases"] if v["case"] == old["case"])
            self.assertEqual(actual["inputs"], old["inputs"])
            self.assertEqual(
                actual["traces"],
                [
                    {
                        k: v
                        for k, v in t.items()
                        if k not in ("ptx_sha256", "cubin_sha256")
                    }
                    for t in old["traces"]
                ],
            )

    def test_multi_request_cannot_resize_or_route_new_source(self):
        config = dict(IS_FP8=True, NUM_HEADS=4, HEAD_DIM=128, **p.fp8.probe.CONFIG)
        with patch.dict(os.environ, {"QSA_DOT_REQUESTS": "1"}):
            self.assertTrue(p.eligible(config))
            self.assertFalse(p.eligible(dict(config, num_requests=2)))
        with patch.object(p.runtime, "eligible", p.eligible):
            for item in p.manifest()["cases"]:
                count = len(item["case"]["query_lengths"])
                with patch.dict(os.environ, {"QSA_DOT_REQUESTS": str(count)}):
                    actual = p.runtime.traces_for(item["traces"], "paired32")
                for old, new in zip(item["traces"], actual):
                    expected = dict(old["constants"])
                    if (
                        count == 1
                        and old["route"] == "fp8"
                        and p.ELIGIBLE(dict(config, **old["constants"]))
                    ):
                        expected["TILE_R"] = 32
                    self.assertEqual(new["constants"], expected)
                    if count != 1:
                        self.assertEqual(new, old)

    def test_three_variant_compile_proof_is_preserved_under_two_runtime_variants(self):
        with patch.object(
            p.runtime, "VARIANTS", ("production", "paired32")
        ), patch.object(p, "PROOF", lambda: p.runtime.VARIANTS):
            self.assertEqual(p.proof(), p.ORIGINAL_VARIANTS)
            self.assertEqual(p.runtime.VARIANTS, ("production", "paired32"))

    def test_full_adapter_selects_two_runtime_variants_without_state_leaks(self):
        original_worker = p.runtime.worker
        original_coordinate = p.runtime.coordinate

        def inspect():
            self.assertEqual(p.runtime.VARIANTS, ("production", "paired32"))
            self.assertEqual(p.runtime.PAIRS, ((0, 1),))
            self.assertEqual(len(p.fp8.CASES), 10)
            self.assertIs(p.runtime.worker, p.worker)
            self.assertIs(p.runtime.coordinate, p.coordinate)
            self.assertIs(p.COORDINATE, original_coordinate)
            return 0

        with patch.object(p.runtime, "main", inspect):
            self.assertEqual(p.main(), 0)
        self.assertEqual(p.runtime.VARIANTS, p.ORIGINAL_VARIANTS)
        self.assertIs(p.runtime.worker, original_worker)
        self.assertIs(p.runtime.coordinate, original_coordinate)

    def test_boundaries_restore_two_validated_single_request_layouts(self):
        def check(*args):
            self.assertEqual(p.fp8.CASES, p.TWO_CASES)
            self.assertEqual(len(p.fp8.manifest()["cases"]), 2)
            self.assertEqual(os.environ["QSA_DOT_REQUESTS"], "1")
            return [dict(status="passed")]

        with patch.object(p.fp8, "manifest", p.manifest), patch.object(
            p.fp8, "CASES", tuple(c["case"]["name"] for c in p.manifest()["cases"])
        ), patch.object(p.fp8, "boundary_checks", check), patch.dict(
            os.environ, {"QSA_DOT_REQUESTS": "2"}
        ):
            self.assertEqual(
                p.boundary_checks(None, None, None, None), [dict(status="passed")]
            )
            self.assertEqual(len(p.fp8.CASES), 10)
            self.assertEqual(os.environ["QSA_DOT_REQUESTS"], "2")


if __name__ == "__main__":
    unittest.main()
