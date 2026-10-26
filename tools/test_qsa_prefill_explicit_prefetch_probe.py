# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Check single-hint AST scope, real IR motion gates and guarded probe plumbing."""

import ast
import copy
import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_explicit_prefetch_probe as probe


def ir(early, metadata=False, direct=False):
    copy_line = (
        "      %next = tt.load %ptr : tensor<64x128x!tt.ptr<bf16>, #blocked>"
        if direct
        else "      %next = ttg.async_copy_global_to_local %ptr, %buf : tensor<64x128x!tt.ptr<bf16>, #blocked>"
    )
    dot_line = "      %score = ttng.warp_group_dot %a, %b"
    body = [copy_line, dot_line] if early else [dot_line, copy_line]
    if metadata:
        body.insert(
            0,
            "      %ids = ttg.async_copy_global_to_local %table, %meta : tensor<64x!tt.ptr<i32>, #slice>",
        )
    return "\n".join(
        [
            "module {",
            "  %buffer = ttg.local_alloc : () -> !ttg.memdesc<2x64x128xbf16, #shared>",
            "  %initial = ttg.async_copy_global_to_local %ptr, %buf : tensor<64x128x!tt.ptr<bf16>, #blocked>",
            "  %result = scf.for %i = %a to %b step %one {",
            *body,
            "    scf.yield %v",
            "  } loc(#loc1)",
            "}",
        ]
    )


PTX = "wgmma.mma_async.sync.aligned x; wgmma.wait_group.sync.aligned 0;"


class LoopPipelineTests(unittest.TestCase):
    def test_only_load_schedule_changes_math_store_and_prefix_are_exact(self):
        probe.verify_source()

        def fn(path, name):
            return next(
                n
                for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )

        before = fn(
            probe.ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py",
            "_qsa_prefill_score_kernel",
        )
        after = fn(probe.KERNEL, "_explicit_prefetch_score_kernel")
        a, b = (
            next(n for n in f.body if isinstance(n, ast.For)) for f in (after, before)
        )
        self.assertEqual(ast.dump(after.args), ast.dump(before.args))
        self.assertEqual(
            ast.unparse(a.iter), "tl.range(first_tile, last_tile, num_stages=1)"
        )
        self.assertEqual(
            [ast.unparse(n) for n in a.body[-3:]],
            ["columns = next_columns", "live = next_live", "keys = next_keys"],
        )
        math_start = lambda loop: next(
            i for i, n in enumerate(loop.body) if isinstance(n, ast.If)
        )
        self.assertEqual(
            [ast.dump(n) for n in a.body[math_start(a) : -3]],
            [ast.dump(n) for n in b.body[math_start(b) :]],
        )
        column = lambda f: next(
            i
            for i, n in enumerate(f.body)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "column_lanes"
        )
        ia, ib = column(after), column(before)
        self.assertEqual(
            [ast.dump(n) for n in after.body[: ia + 1]],
            [ast.dump(n) for n in before.body[: ib + 1]],
        )

        class FirstTile(ast.NodeTransformer):
            def visit_Name(self, n):
                return (
                    ast.copy_location(ast.Name(id="first_tile", ctx=n.ctx), n)
                    if n.id == "tile"
                    else n
                )

        initial = [FirstTile().visit(copy.deepcopy(n)) for n in b.body[: math_start(b)]]
        self.assertEqual(
            [ast.dump(n) for n in after.body[ia + 1 : -1]],
            [ast.dump(n) for n in initial],
        )

    def test_actual_prefetch_addresses_masks_and_loop_carry(self):
        import numpy as np

        class Vec(np.ndarray):
            def __new__(cls, value):
                return np.asarray(value).view(cls)

            def to(self, dtype):
                return Vec(self.astype(dtype))

        class Pointer:
            def __init__(self, kind, offset):
                self.kind, self.offset = kind, Vec(offset)

            def __add__(self, value):
                return Pointer(self.kind, self.offset + value)

        def lowered(path, name):
            fn = next(
                n
                for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            start = next(
                i
                for i, n in enumerate(fn.body)
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "column_lanes"
            )
            nodes = copy.deepcopy(fn.body[start:])
            loop = next(n for n in nodes if isinstance(n, ast.For))
            changed = []
            for n in loop.body:
                if isinstance(n, ast.If):
                    continue  # unchanged dot/reduction AST is checked separately
                if (
                    isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Attribute)
                    and n.value.func.attr == "store"
                ):
                    changed.append(
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id="consume", ctx=ast.Load()),
                                args=[
                                    ast.Name(id=k, ctx=ast.Load())
                                    for k in ("columns", "live", "keys")
                                ],
                                keywords=[],
                            )
                        )
                    )
                else:
                    changed.append(n)
            loop.body = changed
            return compile(
                ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                str(path),
                "exec",
            )

        original = lowered(
            probe.ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py",
            "_qsa_prefill_score_kernel",
        )
        candidate = lowered(probe.KERNEL, "_explicit_prefetch_score_kernel")
        tested = 0
        for page in (16, 64, 196):
            for first in (0, 1, 3):
                for span in (1, 2, 16):
                    for visible in (1, 31, 63, 64, 65, 195, 196, 197, 511, 1025):
                        if first * 64 >= visible:
                            continue
                        last = min(first + span, (visible + 63) // 64)
                        width = (visible + page - 1) // page
                        for capacity in sorted({0, max(1, width // 2), width}):
                            stride_table = capacity * 2 + 9
                            table_base = 5 + stride_table
                            data = np.full(2 * stride_table + 10, -777, dtype=np.int32)
                            ids = (np.arange(capacity) * 7) % 13
                            if capacity:
                                ids[::5] = -1
                                ids[1::7] = 13
                            data[table_base + np.arange(capacity) * 2] = ids
                            cache_block = (page + 3) * 136

                            def run(code):
                                consumed = []
                                events = []
                                counts = []

                                def load(ptr, mask, other):
                                    offsets, masks = np.broadcast_arrays(
                                        ptr.offset, np.asarray(mask, dtype=bool)
                                    )
                                    active = offsets[masks]
                                    if ptr.kind == "table":
                                        self.assertTrue(
                                            np.all(
                                                (active >= table_base)
                                                & (active < table_base + capacity * 2)
                                            )
                                        )
                                        self.assertTrue(
                                            np.all((active - table_base) % 2 == 0)
                                        )
                                    else:
                                        self.assertTrue(
                                            np.all(
                                                (active >= 11)
                                                & (active < 11 + 13 * cache_block)
                                            )
                                        )
                                    out = np.full(offsets.shape, other, dtype=np.int64)
                                    out[masks] = (
                                        data[active] if ptr.kind == "table" else active
                                    )
                                    events.append((ptr.kind, active.copy()))
                                    return Vec(out)

                                def consume(columns, live, keys):
                                    consumed.append(
                                        tuple(
                                            np.asarray(v).copy()
                                            for v in (columns, live, keys)
                                        )
                                    )
                                    counts.append(
                                        sum(kind == "cache" for kind, _ in events)
                                    )

                                env = dict(
                                    tl=types.SimpleNamespace(
                                        arange=lambda a, b: Vec(np.arange(a, b)),
                                        load=load,
                                        int64=np.int64,
                                        range=lambda *args, **kw: range(*args),
                                    ),
                                    first_tile=first,
                                    last_tile=last,
                                    BLOCK_N=64,
                                    PAGE_SIZE=page,
                                    HEAD_DIM=128,
                                    dims=Vec([0, 3, 127, 128]),
                                    max_visible=visible,
                                    table_width=capacity,
                                    request=1,
                                    stride_table_req=stride_table,
                                    stride_table_page=2,
                                    table_ptr=Pointer("table", 5),
                                    cache_ptr=Pointer("cache", 11),
                                    stride_cache_block=cache_block,
                                    stride_cache_token=136,
                                    num_pages=13,
                                    consume=consume,
                                )
                                exec(code, env)
                                return consumed, events, counts

                            ref, re, rc = run(original)
                            actual, ae, ac = run(candidate)
                            self.assertEqual(len(ref), last - first)
                            self.assertEqual(len(actual), len(ref))
                            for left, right in zip(ref, actual):
                                for x, y in zip(left, right):
                                    np.testing.assert_array_equal(x, y)
                            for kind in ("table", "cache"):
                                x = [v for k, v in re if k == kind and v.size]
                                y = [v for k, v in ae if k == kind and v.size]
                                self.assertEqual(len(x), len(y))
                                for u, v in zip(x, y):
                                    np.testing.assert_array_equal(u, v)
                            self.assertEqual(rc, list(range(1, last - first + 1)))
                            self.assertEqual(ac, list(range(2, last - first + 2)))
                            self.assertEqual(
                                [k for k, _ in ae[-2:]], ["table", "cache"]
                            )
                            self.assertTrue(all(v.size == 0 for _, v in ae[-2:]))
                            tested += 1
        self.assertGreater(tested, 300)
        print("Actual-AST prefetch address/mask/carry scenarios:", tested)

    def test_ir_motion_excludes_prologue_and_post_dot(self):
        a, b = probe.prefetch(ir(False), PTX), probe.prefetch(ir(True), PTX)
        self.assertEqual(a["copies_before_first_dot"], 0)
        self.assertEqual(b["copies_before_first_dot"], 1)
        self.assertEqual(a["loop_copy_sites"], 1)
        self.assertEqual(a["mma_sites"], 1)
        self.assertIsNone(probe.prefetch("", PTX)["copies_before_first_dot"])
        self.assertEqual(
            probe.prefetch(ir(True) + "\n" + ir(True), PTX)["main_loops"], 2
        )

    def test_metadata_only_copy_is_not_key_motion(self):
        self.assertEqual(
            probe.prefetch(ir(False, metadata=True), PTX)["copies_before_first_dot"], 0
        )
        self.assertEqual(
            probe.prefetch(ir(True, direct=True), PTX)["copies_before_first_dot"], 1
        )
        base = dict(
            spill_count=0,
            shared_bytes=106496,
            registers_per_thread=207,
            prefetch=probe.prefetch(ir(False), PTX),
        )
        for slots in (0, 3):
            after = dict(
                base,
                prefetch=dict(probe.prefetch(ir(True), PTX), key_shared_slots=slots),
            )
            self.assertIn("buffering", probe.gate_reason(base, after))

    def test_register_key_operand_without_shared_key_buffer_is_identified(self):
        code = (
            ir(True, direct=True)
            .replace(
                "  %buffer = ttg.local_alloc : () -> !ttg.memdesc<2x64x128xbf16, #shared>\n",
                "",
            )
            .replace(
                "ttng.warp_group_dot %a, %b",
                "ttng.warp_group_dot %a, %b : tensor<64x128xbf16, #mma> * !ttg.memdesc<128x128xbf16, #shared>",
            )
        )
        a = dict(
            spill_count=0,
            shared_bytes=106496,
            registers_per_thread=207,
            prefetch=probe.prefetch(ir(False), PTX),
        )
        b = dict(a, prefetch=probe.prefetch(code, PTX))
        self.assertEqual(b["prefetch"]["key_shared_slots"], 0)
        self.assertTrue(b["prefetch"]["register_key_operand"])
        self.assertIsNone(probe.gate_reason(a, b))

    def test_gate_rejects_no_motion_unknown_resources_and_spills(self):
        a = dict(
            spill_count=0,
            shared_bytes=106496,
            registers_per_thread=207,
            prefetch=probe.prefetch(ir(False), PTX),
        )
        b = dict(a, prefetch=probe.prefetch(ir(True), PTX))
        self.assertIsNone(probe.gate_reason(a, b))
        self.assertIn("no additional", probe.gate_reason(a, a))
        self.assertIn("spills", probe.gate_reason(a, dict(b, spill_count=1)))
        self.assertIn("spills", probe.gate_reason(a, dict(b, spill_count=None)))
        self.assertIn(
            "unknown", probe.gate_reason(a, dict(b, registers_per_thread=None))
        )
        self.assertIn(
            "larger shared", probe.gate_reason(a, dict(b, shared_bytes=114688))
        )
        self.assertIn(
            "unidentified",
            probe.gate_reason(a, dict(b, prefetch=probe.prefetch("", PTX))),
        )
        self.assertIn(
            "matrix work",
            probe.gate_reason(a, dict(b, prefetch=dict(b["prefetch"], mma_sites=2))),
        )

    def test_scope_routes_new_child_driver_and_restores_harness(self):
        old_file = probe.harness.__file__
        old_eligible = probe.harness.eligible
        with probe.scoped():
            self.assertIs(probe.base.make_pair, probe.make_pair)
            self.assertIs(probe.base.freeze_records, probe.freeze_records)
            self.assertEqual(probe.base.__file__, probe.__file__)
            self.assertIs(probe.harness.eligible, probe.eligible)
            self.assertTrue(
                probe.eligible(
                    dict(IS_FP8=False, NUM_HEADS=4, HEAD_DIM=128, TILE_R=32, BLOCK_N=64)
                )
            )
            self.assertTrue(
                probe.eligible(
                    dict(IS_FP8=False, NUM_HEADS=4, HEAD_DIM=128, TILE_R=64, BLOCK_N=64)
                )
            )
            self.assertFalse(
                probe.eligible(
                    dict(IS_FP8=True, NUM_HEADS=4, HEAD_DIM=128, TILE_R=64, BLOCK_N=64)
                )
            )
        self.assertEqual(probe.harness.__file__, old_file)
        self.assertIs(probe.harness.eligible, old_eligible)

    def test_negative_compile_saves_artifacts_and_does_not_time(self):
        key = (17, 0, 1088)
        kernel = types.SimpleNamespace(asm=dict(ptx=PTX, ttgir=ir(False)))
        entry = types.SimpleNamespace(
            jit=types.SimpleNamespace(decisions={key: True}),
            compiled={key: (kernel, {})},
        )
        empty = types.SimpleNamespace(compiled={})
        module = types.SimpleNamespace(
            _qsa_prefill_score_kernel=entry,
            _qsa_prefill_compact_score_kernel=empty,
            _qsa_prefill_fp8_store_score_kernel=empty,
        )
        record = dict(
            chunk=list(key),
            route="rectangular",
            constants={},
            spill_count=0,
            registers_per_thread=63,
            shared_bytes=73728,
        )
        with tempfile.TemporaryDirectory() as folder, patch.object(
            probe.harness, "RESOURCE_DIR", Path(folder)
        ), patch.object(probe.harness, "RESOURCE_PREFIX", "unit"), patch.object(
            probe.stage,
            "FREEZE",
            return_value=dict(
                baseline=[copy.deepcopy(record)], candidate=[copy.deepcopy(record)]
            ),
        ), patch.object(
            probe.base, "artifact_records", return_value=[]
        ) as export, patch.object(
            probe.harness, "IN_BOUNDARY", False
        ), patch.object(
            probe.harness, "LAST_REJECTION", None
        ):
            with self.assertRaisesRegex(RuntimeError, "NOT_PROMISING"):
                probe.freeze_records((module, module))
            export.assert_called_once()
            self.assertEqual(len(list(Path(folder).glob("compile-unit-*.json"))), 1)
            self.assertEqual(len(probe.harness.LAST_REJECTION), 1)

    def test_worker_classifies_compile_rejection_but_preserves_numeric_error(self):
        with tempfile.TemporaryDirectory() as folder:
            options = types.SimpleNamespace(
                report_dir=folder, worker_case="synthetic", rounds=2
            )
            path = Path(folder) / "synthetic.json"

            def reject(options):
                probe.harness.LAST_REJECTION = [{"reason": "no early copy"}]
                path.write_text(
                    json.dumps(dict(status="failed", stages={}, error="compile gate"))
                )
                return 1

            with patch.object(probe.base, "worker", side_effect=reject), patch.object(
                probe, "show"
            ):
                self.assertEqual(probe.worker(options), 0)
                self.assertEqual(
                    json.loads(path.read_text())["status"], "not_promising"
                )

            def numeric_error(options):
                path.write_text(
                    json.dumps(dict(status="failed", stages={}, error="wrong score"))
                )
                return 1

            with patch.object(
                probe.base, "worker", side_effect=numeric_error
            ), patch.object(probe, "show"):
                self.assertEqual(probe.worker(options), 1)
                self.assertEqual(json.loads(path.read_text())["error"], "wrong score")

    def test_coordinator_continues_after_one_configuration_is_rejected(self):
        from tools import benchmark_qsa_prefill_triton_stages as bench
        from tools import qsa_prefill_flagtree_ab as profile
        from tools import qsa_prefill_profiling as profiling

        data = probe.manifest()
        cases = [
            next(c for c in data["cases"] if c["case"]["name"] == n)
            for n in probe.DEFAULT_CASES
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prior = root / "prior"
            report = root / "new"
            prior.mkdir()
            report.mkdir()
            asm = root / "ptxas"
            asm.write_text("synthetic assembler")
            (prior / "stage-summary.json").write_text(
                json.dumps(
                    dict(
                        status="passed",
                        manifest=dict(
                            git_head=dict(stdout=data["baseline_code_commit"])
                        ),
                    )
                )
            )
            (prior / "flagtree134-tools.json").write_text(
                json.dumps(dict(toolchain=dict(ptxas=dict(path=str(asm)))))
            )
            reference = "synthetic native reference"
            (prior / "sources.json").write_text(
                json.dumps({"pinned_vllm/qsa_indexer.py": dict(text=reference)})
            )
            for c in cases:
                (prior / (c["case"]["name"] + ".json")).write_text(
                    json.dumps(
                        dict(
                            status="passed",
                            case=c["case"],
                            inputs=c["inputs"],
                            runtime={},
                        )
                    )
                )
            options = types.SimpleNamespace(
                report_dir=str(report),
                baseline_report=str(prior),
                cases=list(probe.DEFAULT_CASES),
                timeout=1,
            )
            calls = []

            def run(command, **kwargs):
                name = command[-1]
                calls.append(name)
                if len(calls) == 1:
                    result = dict(
                        status="not_promising",
                        stages={},
                        rejections=[dict(reason="no motion")],
                    )
                else:
                    result = dict(
                        status="passed",
                        stages={
                            k: dict(measurement_validated=True, speedup=1.0)
                            for k in ("score", "topk", "pipeline")
                        },
                        self_controls={str(i): {} for i in range(4)},
                    )
                (report / (name + ".json")).write_text(json.dumps(result))
                return 0

            with probe.scoped(), patch.object(
                profile, "PTXAS_SHA256", hashlib.sha256(asm.read_bytes()).hexdigest()
            ), patch.object(
                bench.ab,
                "REFERENCE_SHA256",
                hashlib.sha256(reference.encode()).hexdigest(),
            ), patch.object(
                probe.base, "compiler_identity", return_value=data["compiler_identity"]
            ), patch.object(
                profile, "select_gpu", return_value=True
            ), patch.object(
                bench, "idle_gate", return_value=True
            ), patch.object(
                bench, "child_environment", return_value={}
            ), patch.object(
                profiling, "run_process", side_effect=run
            ):
                self.assertEqual(probe.coordinate(options), 0)
            summary = json.loads(
                (report / "explicit-prefetch-summary.json").read_text()
            )
            self.assertEqual(calls, list(probe.DEFAULT_CASES))
            self.assertEqual(summary["status"], "completed_with_rejections")
            self.assertEqual(summary["measured_cases"], 1)
            self.assertEqual(summary["aggregate"]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
