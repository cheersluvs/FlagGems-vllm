# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Typed FP8 motion, reversible scheduling and conservative resource gates."""

import ast
import tempfile
import unittest
from pathlib import Path

from tools import qsa_prefill_fp8_producer as p
from tools.test_qsa_prefill_producer_first import IR, SOURCE


def scheduled():
    text = IR.replace("bf16", "f8E4M3FN")
    dot = "    %s = ttng.warp_group_dot %k, %q {loop.cluster = 0 : i32, loop.stage = 1 : i32}"
    return text.replace(dot, dot + "\n" + dot.replace("%s =", "%other ="))


def final(early, slots=2):
    copy = "    %k = ttg.async_copy_global_to_local %p, %buf : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>"
    dots = [
        "    %left = ttng.warp_group_dot %buf, %q",
        "    %l = ttng.warp_group_dot_wait %left {pendings = 0 : i32}",
        "    tt.store %out, %l",
        "    %right = ttng.warp_group_dot %buf, %q2",
        "    %r = ttng.warp_group_dot_wait %right {pendings = 0 : i32}",
    ]
    return "\n".join(
        [
            "module {",
            f"  %buf = ttg.local_alloc : () -> !ttg.memdesc<{slots}x64x128xf8E4M3FN, #shared>",
            "  %result = scf.for %i = %a to %b step %one {",
            *([copy] + dots if early else dots + [copy]),
            "    scf.yield",
            "  }",
            "}",
        ]
    )


def ptx(early):
    copy = "cp.async.cg.shared.global [%r1], [%rd1], 0x10;"
    dots = ["wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 x;"] * 8
    pre = ["cp.async.wait_group 1;", "bar.sync 0;"]
    post = ["wgmma.wait_group.sync.aligned 0;", "bar.sync 0;"]
    body = [copy, *pre, *dots, *post] if early else [*pre, *dots, *post, copy]
    return "\n".join(
        [
            ".visible .entry score() {",
            "L0:",
            *body,
            "@%p1 bra L0;",
            "cp.async.wait_group 0;",
            "bar.sync 0;",
            "ret;",
            "}",
        ]
    )


class FP8ProducerTests(unittest.TestCase):
    def test_rewrite_changes_only_cluster_digits_and_preserves_stages(self):
        text = scheduled()
        changed, proof = p.rewrite(text)
        self.assertEqual(p.producer.producer_first(changed)[0], text)
        self.assertEqual(proof["K_tile_bytes"], 8192)
        self.assertFalse(proof["buffer_rewrite_performed"])
        self.assertEqual(changed.count("loop.stage = 1"), text.count("loop.stage = 1"))
        for bad in (
            text.replace("f8E4M3FN", "bf16"),
            text.replace("loop.stage = 0", "loop.stage = 2"),
            text.replace("%other = ttng.warp_group_dot", "%other = other"),
            text + text,
        ):
            with self.assertRaises(RuntimeError):
                p.rewrite(bad)

    def test_actual_copy_fragments_must_all_precede_mma(self):
        a, b = p.inspect(final(False), ptx(False)), p.inspect(final(True, 3), ptx(True))
        self.assertEqual(a["after_last_MMA"], 1)
        self.assertFalse(a["all_K_fragments_before_first_MMA"])
        self.assertEqual(b["before_first_MMA"], 1)
        self.assertTrue(b["all_K_fragments_before_first_MMA"])
        self.assertEqual(b["K_slots"], 3)
        self.assertEqual(b["PTX_protocol"]["status"], "passed")
        mixed = ptx(True).replace(
            "@%p1 bra", "cp.async.cg.shared.global [%r2], [%rd2], 16;\n@%p1 bra"
        )
        self.assertFalse(p.motion(mixed)["all_K_fragments_before_first_MMA"])
        # A metadata-only four-byte async copy is not actual K movement.
        with self.assertRaises(RuntimeError):
            p.motion(ptx(True).replace("0x10", "0x4"))
        with self.assertRaises(RuntimeError):
            p.inspect(final(True).replace("f8E4M3FN", "bf16"), ptx(True))

    def test_gate_preserves_work_sync_and_accounts_for_fp8_ring_bytes(self):
        control = dict(
            schedule=p.inspect(final(False), ptx(False)),
            shared_bytes=57344,
            registers_per_thread=158,
            spill_store_bytes=0,
            spill_load_bytes=0,
        )
        candidate = dict(
            control, schedule=p.inspect(final(True, 3), ptx(True)), shared_bytes=65536
        )
        self.assertIsNone(p.gate(control, candidate))
        self.assertIsNone(
            p.gate(
                control,
                dict(
                    candidate,
                    schedule=p.inspect(final(True), ptx(True)),
                    shared_bytes=57344,
                ),
            )
        )
        for bad in (
            dict(candidate, spill_load_bytes=8),
            dict(candidate, shared_bytes=73728),
            dict(candidate, schedule=dict(candidate["schedule"], K_slots=4)),
            dict(candidate, schedule=dict(candidate["schedule"], mma_sites=4)),
            dict(candidate, schedule=dict(candidate["schedule"], PTX_protocol={})),
            dict(
                candidate,
                schedule=dict(
                    candidate["schedule"], all_K_fragments_before_first_MMA=False
                ),
            ),
        ):
            self.assertIsNotNone(p.gate(control, bad))

    def test_replay_to_resume_keeps_each_case_options_and_unchanged_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "replay"
            old.mkdir()
            out = root / "new"
            out.mkdir()
            (old / "passes.txt").write_text(
                "// IR Dump Before TritonGPUPipeline (tritongpu-pipeline)\n"
                + scheduled()
            )
            meta = old / "options.json"
            meta.write_text('{"num_stages":2,"ir_override":null}')
            job = dict(
                directory=str(old), case=p.CASES[0], metadata=str(meta), runtime={}
            )
            jobs = p.scheduled_jobs(job, out)
            self.assertEqual(len(jobs), 2)
            self.assertEqual(Path(jobs[0]["ir"]).read_text(), scheduled())
            self.assertNotEqual(Path(jobs[1]["ir"]).read_text(), scheduled())
            self.assertTrue(
                all(Path(j["metadata"]).read_bytes() == meta.read_bytes() for j in jobs)
            )
            self.assertTrue(all(j["ir"].endswith(".ttir") for j in jobs))

    def test_compile_adapter_cannot_execute_gpu_or_shrink_buffers(self):
        tree = ast.parse(Path(p.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                self.assertNotIn(
                    node.attr,
                    {
                        "warmup",
                        "_init_handles",
                        "n_regs",
                        "n_spills",
                        "synchronize",
                        "two_slots",
                    },
                )
        self.assertIn('CUDA_VISIBLE_DEVICES=""', Path(p.__file__).read_text())
        self.assertIn("ir_sha256", Path(p.__file__).read_text())
        self.assertEqual(len(p.CASES), 2)
        self.assertNotIn("bfloat16", "".join(p.CASES))
        runner = (p.ROOT / "tools/run_qsa_prefill_fp8_producer_h800.sh").read_text()
        self.assertIn('"$qsa_report"/*.tar.gz', runner)
        self.assertIn('exit "$qsa_status"', runner)
        # Original exact pipeline-tail extraction remains the tested helper.
        fn, code = p.producer.resume_function(SOURCE, {})
        self.assertTrue(callable(fn))
        self.assertIn("add_pipeline", code)


if __name__ == "__main__":
    unittest.main()
