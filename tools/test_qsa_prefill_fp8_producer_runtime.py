# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""FP8 override routing, native tail, lifetime and scoped harness contracts."""

import copy
import hashlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_fp8_producer_runtime as p
from tools.test_qsa_prefill_double_buffer import IR
from tools.test_qsa_prefill_fp8_producer import ptx, scheduled


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class JIT:
    def __init__(self):
        self.warmups, self.launches = [], []

    def warmup(self, *values, grid, **kwargs):
        self.warmups.append(dict(kwargs))
        return types.SimpleNamespace(
            asm=dict(ptx="candidate" if "ir_override" in kwargs else "control")
        )

    def __getitem__(self, grid):
        def launch(*values, **kwargs):
            self.launches.append(dict(kwargs))
            return types.SimpleNamespace(
                asm=dict(ptx="candidate" if "ir_override" in kwargs else "control")
            )

        return launch


class FP8RuntimeTests(unittest.TestCase):
    def named(self, **changes):
        return dict(
            IS_FP8=True, NUM_HEADS=4, HEAD_DIM=128, **{**p.probe.CONFIG, **changes}
        )

    def capture(self):
        return dict(
            case="fp8_p2_main",
            control_code_sha256=p.runtime.code_sha("control"),
            candidate_ptx_sha256=digest("candidate"),
            override_file="/tmp/verified-fp8.ttgir",
        )

    def test_only_original_fp8_p2_main_config_uses_override(self):
        jit = JIT()
        proxy = p.RoutedJIT(
            jit, ["num_rows", "query_offset", "logits_width"], [self.capture()]
        )
        with patch.object(p.runtime, "prepared_artifacts"):
            proxy[(1, 1, 1)](35, 5, 8256, **self.named())
            self.assertEqual(len(jit.warmups), 2)
            self.assertEqual(len(jit.launches), 1)
            proxy.modified.frozen = True
            for _ in range(3):
                proxy[(1, 1, 1)](35, 5, 8256, **self.named())
            self.assertEqual(len(jit.warmups), 2)
            for change in (
                dict(TILE_R=8, QUERY_PARTS=1),
                dict(QUERY_PARTS=1),
                dict(num_stages=1),
            ):
                proxy[(1, 1, 1)](113, 16256, 8256, **self.named(**change))
                self.assertNotIn("ir_override", jit.launches[-1])
            self.assertEqual(len(jit.warmups), 2)
            with self.assertRaisesRegex(RuntimeError, "after freeze"):
                proxy[(1, 1, 1)](36, 5, 8256, **self.named())
        self.assertFalse(p.eligible(dict(self.named(), IS_FP8=False)))
        self.assertFalse(p.eligible(dict(self.named(), HEAD_DIM=256)))

    def test_wrong_candidate_is_rejected_before_any_gpu_launch(self):
        jit = JIT()
        capture = self.capture()
        capture["candidate_ptx_sha256"] = digest("wrong")
        proxy = p.RoutedJIT(
            jit, ["num_rows", "query_offset", "logits_width"], [capture]
        )
        with patch.object(p.runtime, "prepared_artifacts"):
            with self.assertRaisesRegex(RuntimeError, "raw PTX differs"):
                proxy[(1, 1, 1)](35, 5, 8256, **self.named())
        self.assertEqual(jit.launches, [])

    def test_fp8_three_slot_readonly_lifetime_and_negative_controls(self):
        ir = IR.replace("bf16", "f8E4M3FN").replace(
            "%dot = ttng.warp_group_dot %read, %q",
            "%dot = ttng.warp_group_dot %read, %q\n    %dot2 = ttng.warp_group_dot %read, %q2",
        )
        proof = p.lifetime(ir, ptx(True))
        self.assertEqual(proof["slots"], 3)
        self.assertFalse(proof["buffer_rewrite"])
        self.assertEqual(len(proof["model"]), 17)
        self.assertTrue(all(v["safe"] for v in proof["model"]))
        for bad in (
            ir.replace("3x64x128x", "2x64x128x"),
            ir.replace("%r = %minus", "%r = %zero"),
            ir.replace("num = 1 : i32", "num = 0 : i32"),
            ir.replace("scf.yield %wi, %ri", "scf.yield %ri, %wi"),
        ):
            with self.assertRaises((RuntimeError, StopIteration)):
                p.lifetime(bad, ptx(True))
        self.assertEqual(proof["K_tile_bytes"], 8192)

    def test_freeze_proves_actual_main_binary_resources_and_tail_fallback(self):
        key = (35, 5, 8256)
        tailkey = (113, 16256, 8256)
        before = dict(
            chunk=list(key),
            route="fp8",
            constants=p.probe.CONFIG,
            ptx_sha256="old",
            cubin_sha256="oldcubin",
            registers_per_thread=158,
            shared_bytes=57344,
            spill_count=0,
        )
        after = dict(
            before,
            ptx_sha256="new",
            cubin_sha256="newcubin",
            registers_per_thread=162,
            shared_bytes=65536,
        )
        tail = dict(
            before,
            chunk=list(tailkey),
            constants=dict(p.probe.CONFIG, TILE_R=8, QUERY_PARTS=1),
        )
        capture = dict(
            case="fp8_p2_main",
            candidate_ptx_sha256="new",
            control=copy.deepcopy(before),
            candidate=copy.deepcopy(after),
        )
        proxy = types.SimpleNamespace(
            prepared={key: dict(profile="fp8_p2_main")}, frozen=False
        )
        pair = (
            object(),
            types.SimpleNamespace(
                _qsa_prefill_fp8_store_score_kernel=types.SimpleNamespace(
                    jit=types.SimpleNamespace(modified=proxy)
                )
            ),
        )
        values = dict(baseline=[before, tail], candidate=[after, copy.deepcopy(tail)])
        with patch.object(p.stage, "FREEZE", return_value=values), patch.object(
            p.runtime, "PROFILES", [capture]
        ):
            result = p.freeze_records(pair)
            self.assertTrue(result["candidate"][1]["original_binary_verified"])
            self.assertTrue(proxy.frozen)
            values["candidate"][1]["cubin_sha256"] = "changed"
            with self.assertRaisesRegex(RuntimeError, "fallback changed"):
                p.freeze_records(pair)
            values["candidate"][1]["cubin_sha256"] = tail["cubin_sha256"]
            values["candidate"][0]["shared_bytes"] = 73728
            with self.assertRaisesRegex(RuntimeError, "resources"):
                p.freeze_records(pair)

    def test_profile_dedup_requires_identical_artifacts_and_prepared_ir_is_immutable(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            report = root / "out"
            report.mkdir()
            sources = {n: dict(sha256=p.base.sha(p.ROOT / n)) for n in p.probe.SOURCES}
            p.probe.trace.save(source / "sources.json", sources)
            jobs = []
            original = scheduled()
            changed, proof = p.probe.rewrite(original)
            for case in p.CASES:
                for side in ("control", "producer"):
                    folder = source / (case + "-" + side)
                    folder.mkdir()
                    text = "control" if side == "control" else "candidate"
                    (folder / "ptx.txt").write_text(text)
                    (folder / "kernel.cubin").write_text(text)
                    (folder / "ttgir.txt").write_text("final IR")
                    (folder / "scheduled.ttir").write_text(
                        original if side == "control" else changed
                    )
                    (folder / "options.json").write_text("{}")
                    record = dict(
                        status="compiled",
                        ptx_sha256=digest(text),
                        cubin_sha256=digest(text),
                        schedule={},
                    )
                    p.probe.trace.save(folder / "result.json", record)
                    p.probe.trace.save(
                        folder / "spec.json",
                        dict(
                            original_resources=dict(constants=p.probe.CONFIG),
                            ir_sha256=p.base.sha(folder / "scheduled.ttir"),
                            metadata_sha256=p.base.sha(folder / "options.json"),
                            rewrite_proof=proof,
                        ),
                    )
                    jobs.append(dict(name=folder.name, exit_code=0, result=record))
            jobs.extend(
                [dict(name=c + "-replay", exit_code=0, result={}) for c in p.CASES]
            )
            p.probe.trace.save(
                source / "fp8-producer-summary.json",
                dict(
                    status="compile_feasible_pending_review",
                    gpu_kernel_executed=False,
                    jobs=jobs,
                    decisions=[dict(reason=None), dict(reason=None)],
                ),
            )
            with patch.dict(
                os.environ,
                QSA_PRODUCER_REPORT=str(source),
                QSA_PRODUCER_RUNTIME_REPORT=str(report),
            ), patch.object(p.probe, "inspect", return_value={}), patch.object(
                p.probe, "gate", return_value=None
            ), patch.object(
                p, "lifetime", return_value={}
            ):
                captures = p.profiles(allow_create=True)
                self.assertEqual(len(captures), 1)
                self.assertEqual(captures[0]["covered_cases"], list(p.CASES))
                self.assertEqual(p.profiles(), captures)
                path = report / "fp8-producer-main.ttgir"
                path.write_text("changed")
                with self.assertRaisesRegex(RuntimeError, "Prepared FP8 IR changed"):
                    p.profiles()
                path.write_text("final IR")
                (source / (p.CASES[1] + "-producer") / "ttgir.txt").write_text(
                    "different candidate"
                )
                with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
                    p.profiles()

    def test_boundary_pages_cover_full_visible_width_without_changing_real_padding(
        self,
    ):
        original = [list(range(42)) + [-1] * (88 - 42)]
        snapshot = copy.deepcopy(original)
        # Reproduce the exact failed reference interval:23 columns of page42.
        self.assertEqual(
            [i for i in range(8255) if original[0][i // 196] < 0],
            list(range(8232, 8255)),
        )
        planned, proof = p.boundary_page_plan(original, 2712, 196, 8256)
        self.assertEqual(original, snapshot)
        self.assertEqual(planned[0][:42], original[0][:42])
        self.assertEqual(planned[0][42], 42)
        self.assertEqual(planned[0][43:], original[0][43:])
        self.assertEqual(proof["required_pages"], 43)
        self.assertEqual(
            proof["filled"],
            [dict(request=0, logical_page=42, previous=-1, physical_page=42)],
        )
        self.assertTrue(all(0 <= planned[0][i // 196] < 2712 for i in range(8256)))
        self.assertEqual(
            p.boundary_page_plan(planned, 2712, 196, 8256)[1]["filled"], []
        )
        self.assertEqual(p.boundary_page_plan([[0, 999, -1]], 2, 1, 2)[0], [[0, 1, -1]])
        for args in ((original, 0, 196, 8256), (original, 2712, 196, 88 * 196 + 1)):
            with self.assertRaises(ValueError):
                p.boundary_page_plan(*args)

    def test_private_boundary_table_executes_stride_preserving_copy(self):
        class Tensor:
            def __init__(self, rows, stride):
                self.rows = copy.deepcopy(rows)
                self.shape = (len(rows), len(rows[0]))
                self.dtype, self.device = "i32", "cuda"
                self._stride = stride

            def stride(self):
                return self._stride

            def tolist(self):
                return copy.deepcopy(self.rows)

            def copy_(self, other):
                self.rows = copy.deepcopy(other.rows)

        torch = types.SimpleNamespace(
            empty_strided=lambda shape, stride, **kw: Tensor(
                [[0] * shape[1] for _ in range(shape[0])], stride
            ),
            tensor=lambda rows, **kw: Tensor(rows, (len(rows[0]), 1)),
        )
        original = Tensor([list(range(42)) + [-1] * 46], (176, 2))
        private, proof = p.boundary_page_table(original, 2712, 196, 8256, torch)
        self.assertIsNot(original, private)
        self.assertEqual(private.stride(), original.stride())
        self.assertEqual(private.shape, original.shape)
        self.assertEqual(original.rows[0][42], -1)
        self.assertEqual(private.rows[0][42], 42)
        self.assertEqual(proof["stride"], [176, 2])

    def test_real_cli_worker_dispatch_uses_fp8_callbacks_and_no_bf16_work(self):
        def selected(options):
            self.assertEqual(options.worker_case, p.CASES[0])
            self.assertEqual(options.cases, list(p.CASES))
            self.assertIs(p.base.make_pair, p.make_pair)
            self.assertIs(p.base.freeze_records, p.freeze_records)
            self.assertIs(p.base.boundary_checks, p.boundary_checks)
            self.assertIs(p.base.manifest, p.manifest)
            self.assertEqual(Path(p.base.__file__), Path(p.__file__))
            return 0

        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys,
            "argv",
            [
                "driver",
                "--producer-report",
                directory,
                "--report-dir",
                directory,
                "--baseline-report",
                directory,
                "--worker-case",
                p.CASES[0],
            ],
        ), patch("tools.qsa_prefill_flagtree_ab.configure"), patch.object(
            p.runtime, "configure_report"
        ), patch.object(
            p.base, "worker", side_effect=selected
        ):
            self.assertEqual(p.main(), 0)

    def test_scoped_harness_keeps_original_timer_and_coordinator(self):
        old_main = p.runtime.main
        oldcases = p.runtime.CASES

        def inspect():
            self.assertEqual(p.runtime.CASES, p.CASES)
            self.assertIs(p.runtime.profiles, p.profiles)
            self.assertIs(p.runtime.make_pair, p.make_pair)
            self.assertIs(p.stage.manifest, p.manifest)
            self.assertIs(p.runtime.orchestration.manifest, p.manifest)
            self.assertEqual(Path(p.runtime.__file__), Path(p.__file__))
            self.assertEqual(
                [v["case"]["name"] for v in p.manifest()["cases"]], list(p.CASES)
            )
            return 0

        with patch.object(p.runtime, "main", side_effect=inspect), patch.object(
            sys, "argv", ["driver", "--help"]
        ):
            self.assertEqual(p.main(), 0)
        self.assertIs(p.runtime.main, old_main)
        self.assertEqual(p.runtime.CASES, oldcases)
        runner = (
            p.ROOT / "tools/run_qsa_prefill_fp8_producer_runtime_h800.sh"
        ).read_text()
        self.assertIn('exit "$qsa_status"', runner)
        self.assertIn("回传失败；无需重测", runner)


if __name__ == "__main__":
    unittest.main()
