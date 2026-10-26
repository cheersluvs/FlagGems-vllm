# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Check cluster-only mutation and exact-tail resume before remote compilation."""

import ast
import json
import tempfile
import types
import unittest
from pathlib import Path

from tools import qsa_prefill_producer_first as probe

IR = """#loc = loc("score.py":1:0)
module {
  %buf = ttg.local_alloc : () -> !ttg.memdesc<2x64x128xbf16, #shared>
  %v = scf.for %i = %first to %last step %one {
    %k = tt.load %p {loop.cluster = 1 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked>
    %s = ttng.warp_group_dot %k, %q {loop.cluster = 0 : i32, loop.stage = 1 : i32}
    tt.store %o, %s {loop.cluster = 0 : i32, loop.stage = 1 : i32}
    scf.yield
  }
}
"""
SOURCE = """class CUDABackend:
    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        forbidden_prefix()
        if capability // 10 in [8, 9]:
            forbidden_schedule()
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
        elif capability // 10 >= 10:
            forbidden_other_arch()
        passes.common.tail(pm)
        pm.run(mod, 'make_ttgir')
        metadata['cluster_dims'] = tuple(opt.cluster_dims)
        return mod
"""


def final_ir(early, slots=2):
    copy = "    %k = ttg.async_copy_global_to_local %p, %buf : tensor<64x128x!tt.ptr<bf16>, #blocked>"
    dot = "    %v = ttng.warp_group_dot %buf, %q"
    body = [copy, dot] if early else [dot, copy]
    return "\n".join(
        [
            "module {",
            f"  %buf = ttg.local_alloc : () -> !ttg.memdesc<{slots}x64x128xbf16, #shared>",
            "  %result = scf.for %i = %a to %b step %one {",
            *body,
            "    %r = ttng.warp_group_dot_wait %v {pendings = 0 : i32}",
            "    tt.store %o, %r",
            "    scf.yield",
            "  }",
            "}",
        ]
    )


PTX = "wgmma.mma_async.sync.aligned x; cp.async.cg.shared.global [%r1], [%rd1], 0x10;"


class ProducerFirstTests(unittest.TestCase):
    def test_snapshot_is_complete_and_unique(self):
        text = (
            "// -----// IR Dump Before OtherPass\nwrong\n"
            "// -----// IR Dump Before TritonGPUPipeline (tritongpu-pipeline)\n"
            + IR
            + "// -----// IR Dump Before NextPass\nwrong"
        )
        self.assertEqual(probe.snapshot(text), IR)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            probe.snapshot(text + text)
        with self.assertRaises(RuntimeError):
            probe.snapshot(IR)

    def test_only_cluster_digits_change_and_rewrite_is_involution(self):
        changed, count = probe.producer_first(IR)
        self.assertEqual(count, 3)
        self.assertEqual(probe.producer_first(changed)[0], IR)
        strip = lambda text: text.replace(
            "loop.cluster = 0", "loop.cluster = X"
        ).replace("loop.cluster = 1", "loop.cluster = X")
        self.assertEqual(strip(IR), strip(changed))
        self.assertIn("loop.cluster = 0 : i32, loop.stage = 0 : i32", changed)
        for bad in (
            IR.replace("loop.stage = 1", "loop.stage = 2"),
            IR.replace("loop.cluster = 1", "loop.cluster = 2"),
            IR + IR,
        ):
            with self.assertRaises(RuntimeError):
                probe.producer_first(bad)

    def test_resume_executes_pipeline_and_exact_tail_but_no_prefix(self):
        calls = []

        class PM:
            def enable_debug(self):
                calls.append("debug")
                return True

            def run(self, mod, tag):
                calls.append(tag)

        pm = PM()
        scope = dict(
            ir=types.SimpleNamespace(pass_manager=lambda context: pm),
            passes=types.SimpleNamespace(
                ttgpuir=types.SimpleNamespace(
                    add_pipeline=lambda p, n, d: calls.append(("pipeline", n, d))
                ),
                common=types.SimpleNamespace(tail=lambda p: calls.append("tail")),
            ),
        )
        fn, generated = probe.resume_function(SOURCE, scope)
        obj = types.SimpleNamespace(context=None)
        metadata = {}
        self.assertIs(
            fn(
                obj,
                metadata,
                types.SimpleNamespace(num_stages=2, cluster_dims=(1, 1, 1)),
                90,
            ),
            obj,
        )
        self.assertEqual(calls, ["debug", ("pipeline", 2, True), "tail", "make_ttgir"])
        old = ast.parse(SOURCE).body[0].body[0]
        new = ast.parse(generated).body[0]
        self.assertEqual(
            [ast.dump(n) for n in old.body[2:]], [ast.dump(n) for n in new.body[3:]]
        )
        self.assertEqual(metadata, dict(cluster_dims=(1, 1, 1)))
        with self.assertRaisesRegex(RuntimeError, "Unsupported"):
            probe.resume_function(SOURCE.replace("[8, 9]", "[9]"), scope)

    def test_schedule_typed_copy_buffer_growth_and_work_gate(self):
        a = dict(
            schedule=probe.inspect(final_ir(False), PTX),
            spill_store_bytes=0,
            spill_load_bytes=0,
        )
        b = dict(a, schedule=probe.inspect(final_ir(True, 3), PTX))
        self.assertEqual(a["schedule"]["early_K_copies"], 0)
        self.assertEqual(b["schedule"]["early_K_copies"], 1)
        self.assertEqual(b["schedule"]["K_slots"], 3)
        self.assertIsNone(probe.gate(a, b))
        self.assertIn("no early", probe.gate(a, a))
        self.assertIn("spills", probe.gate(a, dict(b, spill_load_bytes=16)))
        self.assertIn(
            "matrix", probe.gate(a, dict(b, schedule=dict(b["schedule"], mma_sites=2)))
        )
        self.assertIn(
            "buffer", probe.gate(a, dict(b, schedule=dict(b["schedule"], K_slots=4)))
        )
        self.assertIn(
            "wait", probe.gate(a, dict(b, schedule=dict(b["schedule"], waits=[])))
        )
        with self.assertRaisesRegex(RuntimeError, "Unidentified"):
            probe.inspect(
                final_ir(True).replace("ttg.async_copy_global_to_local", "tt.load"), PTX
            )

    def test_resources_never_treat_unknown_spills_as_zero(self):
        text = (
            "ptxas info : Used 63 registers\n"
            "ptxas info : 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"
        )
        self.assertEqual(
            probe.resources(text),
            dict(registers_per_thread=63, spill_store_bytes=0, spill_load_bytes=0),
        )
        self.assertEqual(
            probe.resources(
                text.replace("0 bytes spill stores", "32 bytes spill stores")
            )["spill_store_bytes"],
            32,
        )
        for bad in ("", "Used 63 registers", text + "\nUsed 64 registers"):
            with self.assertRaisesRegex(RuntimeError, "Unknown/ambiguous"):
                probe.resources(bad)

    def test_prepare_preserves_captured_files_and_original_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old"
            report = root / "new"
            source.mkdir()
            report.mkdir()
            jobs = []
            for case in probe.trace.CASES:
                folder = source / (case + "-baseline")
                folder.mkdir()
                ptx = folder / "replayed-ptx.txt"
                ptx.write_text("original PTX")
                metadata = folder / "cached.json"
                metadata.write_text('{"num_stages":2}')
                spec = dict(
                    expected_ptx_sha256=probe.trace.sha(ptx),
                    metadata=str(metadata),
                    metadata_sha256=probe.trace.sha(metadata),
                    original_resources=dict(constants=dict(num_stages=2)),
                )
                probe.trace.save(folder / "spec.json", spec)
                (folder / "passes.txt").write_text(
                    "// IR Dump Before TritonGPUPipeline (tritongpu-pipeline)\n" + IR
                )
                jobs.append(
                    dict(
                        name=folder.name,
                        exit_code=0,
                        coverage=dict(all=True),
                        result=dict(
                            status="passed",
                            actual_ptx_sha256=probe.trace.sha(ptx),
                            expected_ptx_sha256=probe.trace.sha(ptx),
                        ),
                    )
                )
            probe.trace.save(
                source / "pipeline-trace-summary.json",
                dict(
                    status="passed",
                    gpu_kernel_executed=False,
                    jobs=jobs
                    + [
                        dict(j, name=j["name"].replace("-baseline", "-candidate"))
                        for j in jobs
                    ],
                ),
            )
            before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            created = probe.prepare(source, report)
            self.assertEqual(len(created), 4)
            self.assertEqual(
                before,
                {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()},
            )
            self.assertEqual([x["cluster_changes"] for x in created], [0, 3, 0, 3])
            self.assertTrue(
                all(
                    json.loads(Path(x["metadata"]).read_text())["num_stages"] == 2
                    for x in created
                )
            )

    def test_no_GPU_execution_or_installed_source_edits(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        call = next(n for n in calls if ast.unparse(n.func) == "triton.compile")
        self.assertEqual({k.arg for k in call.keywords}, {"target", "options"})
        self.assertFalse(
            any(
                isinstance(n.func, ast.Attribute)
                and n.func.attr in {"warmup", "_init_handles", "empty", "synchronize"}
                for n in calls
            )
        )
        self.assertFalse(
            any(
                isinstance(n, ast.Attribute) and n.attr in {"n_regs", "n_spills"}
                for n in ast.walk(tree)
            )
        )
        self.assertIn('CUDA_VISIBLE_DEVICES=""', Path(probe.__file__).read_text())
        runner = (probe.ROOT / "tools/run_qsa_prefill_producer_first.sh").read_text()
        self.assertIn('"$qsa_report"/*.tar.gz', runner)
        self.assertIn('exit "$qsa_status"', runner)


if __name__ == "__main__":
    unittest.main()
