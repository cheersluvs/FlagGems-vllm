# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Check Python-source-only compilation, captured ABI and strict feasibility gates."""

import ast
import unittest
from pathlib import Path

from tools import qsa_prefill_source_pipeline as p
from tools.test_qsa_prefill_producer_first import PTX, final_ir


class SourcePipelineTests(unittest.TestCase):
    def test_stage_two_changes_exactly_the_loop_stage_keyword(self):
        a, b = p.source("stages1"), p.source("stages2")
        self.assertEqual(b.replace("num_stages=2", "num_stages=1"), a)
        fn = next(n for n in ast.parse(a).body if isinstance(n, ast.FunctionDef))
        original = next(
            n
            for n in ast.parse((p.ROOT / p.EXPLICIT).read_text()).body
            if isinstance(n, ast.FunctionDef)
        )
        fn.name = original.name
        self.assertEqual(ast.dump(fn), ast.dump(original))
        source = ast.parse((p.ROOT / p.OP).read_text())
        original = next(
            n
            for n in source.body
            if isinstance(n, ast.FunctionDef) and n.name == "_qsa_prefill_score_kernel"
        )
        fn = next(
            n
            for n in ast.parse(p.source("production")).body
            if isinstance(n, ast.FunctionDef)
        )
        original.decorator_list = fn.decorator_list
        self.assertEqual(ast.dump(fn), ast.dump(original))
        with self.assertRaises(RuntimeError):
            p.source("other")

    def header(self):
        names = next(
            n
            for n in ast.parse(p.source("production")).body
            if isinstance(n, ast.FunctionDef)
        ).args.args
        return (
            "tt.func public @_qsa_prefill_score_kernel("
            + ", ".join(
                "%"
                + n.arg
                + (": !tt.ptr<bf16>" if i < 6 else ": i32")
                + (" {tt.divisibility = 16 : i32}" if i < 10 else "")
                + " loc(unknown)"
                for i, n in enumerate(names[:19])
                if n.arg != "stride_table_page"
            )
            + ") attributes {noinline=false} {"
        )

    def inputs(self):
        return [
            dict(argument=0, dtype="torch.bfloat16", shape=[16384, 4, 128]),
            dict(argument=1, dtype="torch.bfloat16", shape=[544, 196, 1, 128]),
            dict(argument=2, stride=[136, 1]),
        ]

    def test_captured_abi_keeps_runtime_parameters_and_only_specializes_known_one(self):
        constants = dict(TILE_R=64, BLOCK_N=64, K_TILES=16, HEAD_GROUP=True)
        sig, fixed, attrs = p.abi(
            self.header(), p.source("stages2"), self.inputs(), constants
        )
        self.assertEqual(len(sig), 27)
        self.assertEqual(sig["q_ptr"], "*bf16")
        self.assertEqual(sig["num_rows"], "i32")
        self.assertEqual(fixed["stride_table_page"], 1)
        self.assertEqual(fixed["PAGE_SIZE"], 196)
        self.assertEqual(attrs[(0,)], [("tt.divisibility", 16)])
        self.assertNotIn((12,), attrs)
        with self.assertRaises(RuntimeError):
            p.abi(
                self.header().replace("%num_rows: i32", "%unknown: i32"),
                p.source("stages2"),
                self.inputs(),
                constants,
            )
        bad = self.inputs()
        bad[2]["stride"] = [136, 2]
        with self.assertRaises(RuntimeError):
            p.abi(self.header(), p.source("stages2"), bad, constants)

    def test_direct_register_loads_are_reported_and_never_pass_gate(self):
        text = final_ir(True).replace("ttg.async_copy_global_to_local", "tt.load")
        result = p.inspect(text, PTX)
        self.assertIn("schedule_error", result)
        self.assertEqual(result["early_K_copies"], 0)
        self.assertIsNotNone(p.gate({}, dict(schedule=result)))

    def test_gate_requires_async_work_two_slots_resources_and_protocol(self):
        schedule = dict(
            K_slots=2,
            early_K_copies=1,
            async_copy_sites=16,
            dot_sites=2,
            waits=[0, 0],
            mma_sites=16,
        )
        control = dict(
            schedule=schedule,
            shared_bytes=106496,
            registers_per_thread=207,
            spill_store_bytes=0,
            spill_load_bytes=0,
        )
        candidate = dict(
            control, registers_per_thread=217, PTX_protocol=dict(status="passed")
        )
        self.assertIsNone(p.gate(control, candidate))
        for bad in (
            dict(candidate, shared_bytes=122880),
            dict(candidate, registers_per_thread=218),
            dict(candidate, spill_store_bytes=8),
            dict(candidate, protocol_error="missing drain"),
            dict(candidate, PTX_protocol={}),
            dict(candidate, schedule=dict(schedule, K_slots=3)),
            dict(candidate, schedule=dict(schedule, early_K_copies=0)),
            dict(candidate, schedule=dict(schedule, mma_sites=8)),
        ):
            self.assertIsNotNone(p.gate(control, bad))

    def test_source_compiler_has_no_gpu_run_ir_override_or_backend_patch(self):
        tree = ast.parse(Path(p.__file__).read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                self.assertNotIn(
                    n.attr,
                    {
                        "warmup",
                        "_init_handles",
                        "n_regs",
                        "n_spills",
                        "synchronize",
                        "make_ttgir",
                    },
                )
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "triton.compile"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0].args[0]), "src")
        self.assertEqual({k.arg for k in calls[0].keywords}, {"target", "options"})
        self.assertIn('CUDA_VISIBLE_DEVICES=""', Path(p.__file__).read_text())
        runner = (p.ROOT / "tools/run_qsa_prefill_source_pipeline.sh").read_text()
        self.assertIn('"$qsa_report"/*.tar.gz', runner)
        self.assertIn('exit "$qsa_status"', runner)


if __name__ == "__main__":
    unittest.main()
