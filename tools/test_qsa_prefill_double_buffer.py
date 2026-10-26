# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU lifetime, narrow rewrite, protocol and compile-before-execute gates."""

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_double_buffer as probe
from tools import qsa_prefill_double_buffer_runtime as runtime

IR = """module {
  %zero = arith.constant 0 : i32
  %minus = arith.constant -1 : i32
  %one = arith.constant 1 : i32
  %three = arith.constant 3 : i32
  %buf = ttg.local_alloc : () -> !ttg.memdesc<3x64x128xbf16, #shared>
  %initial = ttg.memdesc_index %buf[%zero] : !ttg.memdesc<3x64x128xbf16, #shared>
  %copy0 = ttg.async_copy_global_to_local %p, %initial : tensor<64x128x!tt.ptr<bf16>, #blocked>
  %commit0 = ttg.async_commit_group tokens %copy0
  %result:3 = scf.for %i = %zero to %last step %one iter_args(%w = %zero, %r = %minus, %token = %commit0) -> (i32) {
    %wadd = arith.addi %w, %one : i32
    %wcmp = arith.cmpi sge, %wadd, %three : i32
    %wi = arith.select %wcmp, %zero, %wadd : i32
    %radd = arith.addi %r, %one : i32
    %rcmp = arith.cmpi sge, %radd, %three : i32
    %ri = arith.select %rcmp, %zero, %radd : i32
    %write = ttg.memdesc_index %buf[%wi] : !ttg.memdesc<3x64x128xbf16, #shared>
    %read = ttg.memdesc_index %buf[%ri] : !ttg.memdesc<3x64x128xbf16, #shared>
    %copy = ttg.async_copy_global_to_local %p, %write : tensor<64x128x!tt.ptr<bf16>, #blocked>
    %commit = ttg.async_commit_group tokens %copy
    %wait = ttg.async_wait %token {num = 1 : i32}
    %dot = ttng.warp_group_dot %read, %q
    %done = ttng.warp_group_dot_wait %dot {pendings = 0 : i32}
    tt.store %o, %done
    scf.yield %wi, %ri, %commit : i32, i32, !ttg.async.token
  }
  ttg.local_dealloc %buf : !ttg.memdesc<3x64x128xbf16, #shared>
}
"""
PTX = """.visible .entry score() {
L0:
cp.async.cg.shared.global [%r1], [%rd1], 0x10;
cp.async.commit_group;
cp.async.wait_group 1;
bar.sync 0;
wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 x;
wgmma.wait_group.sync.aligned 0;
bar.sync 0;
@%p1 bra L0;
cp.async.wait_group 0;
bar.sync 0;
ret;
}
"""


class DoubleBufferTests(unittest.TestCase):
    def test_model_and_unsafe_negative_controls(self):
        result = probe.protocol_model()
        self.assertEqual(len(result["checks"]), 34)
        self.assertEqual(result["states"], 4322)
        self.assertTrue(all(v["safe"] for v in result["checks"]))
        self.assertFalse(result["missing_reuse_barrier_counterexample"]["safe"])
        self.assertTrue(result["missing_reuse_barrier_counterexample"]["witness"])
        self.assertFalse(probe.model(1, 1, slots=1)["safe"])
        self.assertTrue(probe.model(16, 2, slots=3, reuse_barrier=False)["safe"])

    def test_rewrite_changes_only_ring_dimension_and_two_bounds(self):
        changed, proof = probe.two_slots(IR)
        self.assertEqual(changed.count("memdesc<2x64x128xbf16"), 5)
        self.assertEqual(proof["local_type_replacements"], 5)
        restored = changed.replace("  %qsa_ring_two = arith.constant 2 : i32\n", "")
        restored = restored.replace("memdesc<2x64x128xbf16", "memdesc<3x64x128xbf16")
        restored = restored.replace(", %qsa_ring_two : i32", ", %three : i32")
        self.assertEqual(restored, IR)
        self.assertIn("%three = arith.constant 3 : i32", changed)
        for bad in (
            IR.replace("{num = 1", "{num = 0"),
            IR.replace("%r = %minus", "%r = %zero"),
            IR.replace("scf.yield %wi, %ri", "scf.yield %ri, %wi"),
            IR.replace(
                "%commit0 = ttg.async_commit_group tokens %copy0", "%commit0 = other"
            ),
            IR.replace("  ttg.local_dealloc", "  tt.use %buf\n  ttg.local_dealloc"),
            IR.replace("3x64x128xbf16", "3x32x128xbf16"),
            IR + IR,
        ):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                probe.two_slots(bad)

    def test_final_ptx_requires_complete_cta_lifetime_protocol(self):
        result = probe.ptx_protocol(PTX)
        self.assertEqual(result["post_WGMMA_CTA_barriers"], 1)
        self.assertTrue(result["epilogue_drain"])
        for bad in (
            PTX.replace("@%p1 bra", "bra"),
            PTX.replace("L0:\n", "L0:\nbra L1;\nL1:\n"),
            PTX.replace("aligned 0;\nbar.sync 0;", "aligned 0;"),
            PTX.replace("wait_group 1;", "wait_group 2;"),
            PTX.replace("wait_group 1;\nbar.sync 0;", "wait_group 1;"),
            PTX.replace("cp.async.wait_group 0;", "cp.async.wait_group 1;"),
            PTX.replace("0x10;", "0x8;"),
        ):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                probe.ptx_protocol(bad)

    def test_resource_gate_rejects_spills_work_or_wrong_shared_reduction(self):
        schedule = dict(
            K_slots=3,
            mma_sites=8,
            dot_sites=1,
            waits=[0],
            async_copy_sites=1,
            early_K_copies=1,
        )
        control = dict(
            schedule=schedule,
            shared_bytes=90112,
            spill_store_bytes=0,
            spill_load_bytes=0,
        )
        candidate = dict(
            control, schedule=dict(schedule, K_slots=2), shared_bytes=73728
        )
        self.assertIsNone(probe.gate(control, candidate))
        for bad in (
            dict(candidate, shared_bytes=90112),
            dict(candidate, spill_load_bytes=8),
            dict(candidate, schedule=dict(schedule, K_slots=3)),
            dict(candidate, schedule=dict(schedule, K_slots=2, mma_sites=9)),
            dict(candidate, schedule=dict(schedule, K_slots=2, early_K_copies=0)),
        ):
            self.assertIsNotNone(probe.gate(control, bad))

    def run_adapter(self, status, compile_only=False, existing=False, worker=False):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            compiled = report / "compile"
            compiled.mkdir()
            (compiled / "double-buffer-summary.json").write_text(
                json.dumps(dict(status=status))
            )
            argv = ["driver", "--report-dir", directory, "--producer-report", directory]
            if compile_only:
                argv += ["--compile-only"]
            if existing:
                argv += ["--double-buffer-report", str(compiled)]
            if worker:
                argv += ["--worker-case", "case"]
            with patch.object(sys, "argv", argv), patch.dict(
                os.environ, {}, clear=True
            ), patch("subprocess.call", return_value=0) as compile_call, patch.object(
                runtime.runtime, "main", return_value=0
            ) as execute:
                code = runtime.main()
                self.assertEqual(code, 0)
                if status != "compile_feasible_pending_accuracy" or compile_only:
                    execute.assert_not_called()
                    summary = json.loads(
                        (report / "double-buffer-runtime-summary.json").read_text()
                    )
                    self.assertFalse(summary["gpu_kernel_executed"])
                else:
                    execute.assert_called_once()
                self.assertEqual(compile_call.call_count, 0 if existing else 1)

    def test_compile_gate_precedes_runtime_and_reuses_completed_compilations(self):
        self.run_adapter("not_promising")
        self.run_adapter("compile_feasible_pending_accuracy", compile_only=True)
        self.run_adapter("compile_feasible_pending_accuracy")
        self.run_adapter(
            "compile_feasible_pending_accuracy", existing=True, worker=True
        )

    def test_compiler_path_has_no_gpu_execution_or_driver_resource_queries(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                self.assertNotIn(
                    node.attr,
                    {"warmup", "_init_handles", "n_regs", "n_spills", "synchronize"},
                )
        compile_calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "triton.compile"
        ]
        self.assertEqual(len(compile_calls), 1)
        self.assertEqual(
            {k.arg for k in compile_calls[0].keywords}, {"target", "options"}
        )
        runner = (probe.ROOT / "tools/run_qsa_prefill_double_buffer.sh").read_text()
        self.assertIn('"$qsa_report"/*.tar.gz', runner)
        self.assertIn('exit "$qsa_status"', runner)


if __name__ == "__main__":
    unittest.main()
