# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Check ABI-preserving preparation, no hidden baseline execution and scope."""

import ast
import hashlib
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_producer_runtime as runtime


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class FakeJIT:
    def __init__(self, original, candidate):
        self.original, self.candidate = original, candidate
        self.warmups, self.launches = [], []

    def warmup(self, *values, grid, **kwargs):
        self.warmups.append(dict(kwargs))
        return types.SimpleNamespace(
            asm=dict(ptx=self.candidate if "ir_override" in kwargs else self.original)
        )

    def __getitem__(self, grid):
        def launch(*values, **kwargs):
            self.launches.append(dict(kwargs))
            if "ir_override" not in kwargs:
                raise AssertionError("Hidden baseline launch")
            return types.SimpleNamespace(asm=dict(ptx=self.candidate))

        return launch


class ProducerRuntimeTests(unittest.TestCase):
    def test_program_identity_removes_only_debug_information(self):
        a = (
            '.version 9.4\n.target sm_90a\n.file 1 "old.py"\n'
            ".visible .entry score(.param .u64 ptr) {\n.loc 1 7 0\nL0:\n"
            "ld.global.b32 %r1, [%rd1];\n}\n.section .debug_info\n"
            "{ .b8 17; }\n.section .debug_macinfo { }\n"
        )
        b = (
            a.replace("old.py", "new.py")
            .replace(".loc 1 7 0", ".loc 1 99 0")
            .replace(".b8 17", ".b8 44")
            + "//comment\n"
        )
        self.assertNotEqual(digest(a), digest(b))
        self.assertEqual(runtime.program(a), runtime.program(b))
        for changed in (
            a.replace("ld.global.b32", "ld.global.b64"),
            a.replace("sm_90a", "sm_80"),
            a.replace(".param .u64", ".param .u32"),
            a.replace("L0:", "L1:"),
        ):
            self.assertNotEqual(runtime.code_sha(a), runtime.code_sha(changed))
        with self.assertRaisesRegex(RuntimeError, "Malformed"):
            runtime.program(a + "\n.section .debug_line\n{\n")

    def proxy(self, candidate="candidate PTX", artifact="candidate PTX"):
        original = '.target sm_90a\n.file 1 "now.py"\nmov.u32 %r1, 7;'
        jit = FakeJIT(original, candidate)
        capture = dict(
            case="profile",
            control_code_sha256=runtime.code_sha(original.replace("now.py", "old.py")),
            candidate_ptx_sha256=digest(artifact),
            override_file="/tmp/verified.ttgir",
        )
        proxy = runtime.OverrideJIT(
            jit, ["num_rows", "query_offset", "logits_width"], [capture]
        )
        return jit, proxy

    def test_prepare_never_executes_baseline_and_timed_calls_only_launch_candidate(
        self,
    ):
        jit, proxy = self.proxy()
        proxy[(1, 1, 1)](17, 5, 27648, TILE_R=32)
        self.assertEqual(len(jit.warmups), 2)
        self.assertEqual(len(jit.launches), 1)
        self.assertNotIn("ir_override", jit.warmups[0])
        self.assertTrue(jit.warmups[1]["ir_override"].endswith(".ttgir"))
        proxy.frozen = True
        for _ in range(3):
            proxy[(1, 1, 1)](17, 5, 27648, TILE_R=32)
        self.assertEqual(len(jit.warmups), 2)
        self.assertEqual(len(jit.launches), 4)
        self.assertTrue(
            all(v["ir_override"] == "/tmp/verified.ttgir" for v in jit.launches)
        )
        with self.assertRaisesRegex(RuntimeError, "after freeze"):
            proxy[(1, 1, 1)](18, 5, 27648)

    def test_unknown_profile_or_different_candidate_fails_before_any_launch(self):
        jit, proxy = self.proxy()
        proxy.captures[0]["control_code_sha256"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "No exact executable profile"):
            proxy[(1, 1, 1)](17, 5, 27648)
        self.assertEqual(jit.launches, [])
        jit, proxy = self.proxy(candidate="wrong candidate")
        with self.assertRaisesRegex(RuntimeError, "raw PTX differs"):
            proxy[(1, 1, 1)](17, 5, 27648)
        self.assertEqual(jit.launches, [])

    def test_boundary_plan_crosses_requests_includes_empty_request_and_tail(self):
        for requests in (1, 4):
            plan = runtime.boundary_plan(dict(requests=requests))
            self.assertEqual(plan["starts"][-1], plan["total_rows"])
            lo, hi = plan["query_offset"], plan["query_offset"] + plan["num_rows"]
            counts = [
                max(0, min(hi, b) - max(lo, a))
                for a, b in zip(plan["starts"], plan["starts"][1:])
            ]
            self.assertEqual(sum(counts), 17)
            if requests == 4:
                self.assertEqual(counts, [1, 8, 0, 8])
            self.assertLess(hi, plan["total_rows"])

    def test_scope_preserves_child_routing_and_restores_all_callbacks(self):
        old = runtime.base.make_pair
        with runtime.scoped():
            self.assertIs(runtime.base.make_pair, runtime.make_pair)
            self.assertIs(runtime.base.boundary_checks, runtime.boundary_checks)
            self.assertIs(runtime.base.freeze_records, runtime.freeze_records)
            self.assertEqual(runtime.base.__file__, runtime.__file__)
        self.assertIs(runtime.base.make_pair, old)

    def test_make_pair_uses_saved_factory_without_recursive_stage_callback(self):
        entry = types.SimpleNamespace(jit="original", arg_names=["num_rows"])
        pair = (
            types.SimpleNamespace(),
            types.SimpleNamespace(_qsa_prefill_score_kernel=entry),
        )
        with patch.object(
            runtime, "MAKE_PAIR", return_value=pair
        ) as call, patch.object(runtime, "PROFILES", []), runtime.scoped():
            self.assertIs(runtime.make_pair("production", []), pair)
        call.assert_called_once_with("production", [])
        self.assertIsInstance(entry.jit, runtime.OverrideJIT)

    def test_freeze_rejects_changed_resources_before_timing(self):
        key = (17, 5, 27648)
        proxy = types.SimpleNamespace(
            prepared={key: dict(profile="profile")}, frozen=False
        )
        pair = (
            None,
            types.SimpleNamespace(
                _qsa_prefill_score_kernel=types.SimpleNamespace(jit=proxy)
            ),
        )
        a = dict(
            chunk=list(key),
            route="rectangular",
            constants={},
            ptx_sha256="old",
            registers_per_thread=63,
            shared_bytes=73728,
            spill_count=0,
        )
        b = dict(a, ptx_sha256="new", registers_per_thread=68, shared_bytes=90112)
        cap = dict(case="profile", candidate_ptx_sha256="new", control=a, candidate=b)
        with patch.object(runtime, "PROFILES", [cap]), patch.object(
            runtime.stage,
            "FREEZE",
            return_value=dict(baseline=[a], candidate=[dict(b)]),
        ):
            rows = runtime.freeze_records(pair)
            self.assertTrue(proxy.frozen)
            self.assertIn("producer_receipt", rows["candidate"][0])
        with patch.object(runtime, "PROFILES", [cap]), patch.object(
            runtime.stage,
            "FREEZE",
            return_value=dict(baseline=[a], candidate=[dict(b, spill_count=1)]),
        ):
            with self.assertRaisesRegex(RuntimeError, "resources differ"):
                runtime.freeze_records(pair)

    def test_runner_and_source_do_not_set_global_IR_override(self):
        text = Path(runtime.__file__).read_text()
        tree = ast.parse(text)
        self.assertFalse(
            any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "compile"
                for n in ast.walk(tree)
            )
        )
        self.assertNotIn("TRITON_KERNEL_OVERRIDE", text)
        runner = (
            runtime.ROOT / "tools/run_qsa_prefill_producer_runtime.sh"
        ).read_text()
        self.assertIn('"$qsa_report"/*.ttgir', runner)
        self.assertIn("with_cuda_compat134.py", runner)
        self.assertIn('exit "$qsa_status"', runner)


if __name__ == "__main__":
    unittest.main()
