# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Page-ID masked pointer semantics, unchanged math/configs and scoped routing."""

import ast
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_bf16_page_dedup_probe as probe


def function(path, name):
    return next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


def dedup_if(fn):
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    return next(
        n
        for n in loop.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "not IS_FP8 and BLOCK_N <= PAGE_SIZE"
    )


class PageDedupTests(unittest.TestCase):
    def test_kernel_math_and_vector_fallback_are_exact_production(self):
        probe.verify_source()
        production = function(
            probe.ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py",
            "_qsa_prefill_score_kernel",
        )
        candidate = function(probe.KERNEL, "_page_dedup_score_kernel")
        node = dedup_if(candidate)

        class Original(ast.NodeTransformer):
            def visit_If(self, n):
                if ast.dump(n.test) == ast.dump(node.test):
                    return n.orelse
                return self.generic_visit(n)

        self.assertEqual(
            ast.dump(
                Original().visit(
                    ast.Module(body=copy.deepcopy(candidate.body), type_ignores=[])
                )
            ),
            ast.dump(ast.Module(body=production.body, type_ignores=[])),
        )
        self.assertEqual(ast.dump(candidate.args), ast.dump(production.args))

    def test_actual_scalar_and_vector_reads_with_masked_pointer_bounds(self):
        import numpy as np

        node = dedup_if(function(probe.KERNEL, "_page_dedup_score_kernel"))
        vector = copy.deepcopy(node.orelse)

        class Pointer:
            def __init__(self, data, offset=0):
                self.data, self.offset = data, np.asarray(offset)

            def __add__(self, value):
                return Pointer(self.data, self.offset + value)

        code = compile(ast.Module(body=[node], type_ignores=[]), "actual_dedup", "exec")
        reference = compile(
            ast.Module(body=vector, type_ignores=[]), "actual_vector", "exec"
        )
        for page in (16, 64, 196):
            for block in (32, 64, 128):
                for tile in (0, 1, 2, 3, 7, 8, 16):
                    for visible in (
                        0,
                        1,
                        63,
                        64,
                        65,
                        195,
                        196,
                        197,
                        511,
                        512,
                        513,
                        1025,
                    ):
                        for fp8 in (False, True):
                            width = max(1, (visible + page - 1) // page)
                            stride = width * 2 + 5
                            data = np.full(stride * 2, -999, dtype=np.int32)
                            # Include negative/large IDs to check they propagate identically;
                            # the unchanged key-load mask handles their validity afterward.
                            ids = np.arange(width, dtype=np.int32) * 7 - 1
                            data[stride : stride + width * 2 : 2] = ids
                            columns = tile * block + np.arange(block)
                            logical = columns // page
                            live = (columns < visible) & (logical < width)
                            counts = []

                            def load(ptr, mask, other):
                                offsets, masks = np.broadcast_arrays(ptr.offset, mask)
                                self.assertTrue(
                                    np.all(
                                        (offsets[masks] >= stride)
                                        & (offsets[masks] < stride + width * 2)
                                    )
                                )
                                self.assertTrue(
                                    np.all((offsets[masks] - stride) % 2 == 0)
                                )
                                counts.append(int(np.sum(masks)))
                                out = np.full(offsets.shape, other, dtype=np.int32)
                                out[masks] = ptr.data[offsets[masks]]
                                return out

                            env = dict(
                                tl=types.SimpleNamespace(
                                    load=load, minimum=np.minimum, where=np.where
                                ),
                                IS_FP8=fp8,
                                BLOCK_N=block,
                                PAGE_SIZE=page,
                                tile=tile,
                                request=1,
                                stride_table_req=stride,
                                stride_table_page=2,
                                table_width=width,
                                max_visible=visible,
                                logical_page=logical,
                                live=live,
                                table_ptr=Pointer(data),
                            )
                            expected = dict(env)
                            exec(reference, expected)
                            counts.clear()
                            actual = dict(env)
                            exec(code, actual)
                            np.testing.assert_array_equal(
                                actual["physical_page"][live],
                                expected["physical_page"][live],
                            )
                            if not fp8 and block <= page:
                                self.assertLessEqual(sum(counts), 2)
                            else:
                                self.assertEqual(sum(counts), int(np.sum(live)))

    def test_actual_gate_and_delegation_preserve_all_arguments(self):
        calls = []

        class JIT:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    calls.append((self.name, grid, args, kwargs))
                    return self.name

                return run

        names = ["num_rows", "query_offset", "logits_width", "IS_FP8", "PAGE_SIZE"]
        for route, fp8, page, block, wanted in (
            ("rectangular", False, 196, 64, "dedup"),
            ("rectangular", False, 64, 64, "dedup"),
            ("rectangular", False, 16, 64, "original"),
            ("rectangular", True, 196, 64, "original"),
            ("fp8", True, 196, 64, "original"),
            ("compact", False, 196, 64, "original"),
        ):
            args = (7, 13, 1088, fp8, page)
            kwargs = dict(BLOCK_N=block, num_stages=2, HEAD_GROUP=True)
            jit = probe.RoutedJIT(JIT("original"), JIT("dedup"), names, route)
            self.assertEqual(jit[(1, 2, 3)](*args, **kwargs), wanted)
            self.assertEqual(calls[-1], (wanted, (1, 2, 3), args, kwargs))
            self.assertEqual(jit.decisions[(7, 13, 1088)]["active"], wanted == "dedup")
        self.assertFalse(probe.eligible({}))

    def test_factory_preserves_every_captured_config_and_topk(self):
        production = types.ModuleType("fake")
        production.persistent_topk = object()
        for name, _ in probe.base.ENTRIES:
            setattr(
                production,
                name,
                types.SimpleNamespace(
                    arg_names=[], fn=types.SimpleNamespace(fn=object())
                ),
            )
        exec(
            "def _prefill_logits(): return _qsa_prefill_score_kernel\n"
            "def qsa_select_paged_prefill(): return _prefill_logits()",
            vars(production),
        )
        for item in probe.manifest()["cases"]:
            pair = probe.make_pair(production, item["traces"], kernel=object())
            self.assertIs(pair[0].persistent_topk, pair[1].persistent_topk)
            for name, _ in probe.base.ENTRIES:
                a, b = getattr(pair[0], name), getattr(pair[1], name)
                self.assertEqual(a.configs, b.configs)
                self.assertIs(a.jit, b.jit.original)
            self.assertIs(
                pair[1].qsa_select_paged_prefill(), pair[1]._qsa_prefill_score_kernel
            )

    def test_freeze_rejects_config_changes_and_changed_fallback_binary(self):
        row = dict(
            chunk=[7, 0, 1088],
            route="rectangular",
            constants=dict(num_stages=2, HEAD_GROUP=True),
            ptx_sha256="a",
            cubin_sha256="a",
        )
        records = dict(baseline=[row], candidate=[copy.deepcopy(row)])
        module = types.SimpleNamespace()
        for index, (name, _) in enumerate(probe.base.ENTRIES):
            setattr(
                module,
                name,
                types.SimpleNamespace(
                    jit=types.SimpleNamespace(
                        decisions=(
                            {(7, 0, 1088): dict(active=False)} if index == 0 else {}
                        )
                    )
                ),
            )
        with patch.object(probe, "FREEZE", return_value=records):
            self.assertTrue(
                probe.freeze_records((None, module))["candidate"][0][
                    "original_binary_verified"
                ]
            )
            records["candidate"][0]["constants"]["num_stages"] = 1
            with self.assertRaisesRegex(RuntimeError, "configurations"):
                probe.freeze_records((None, module))
            records["candidate"][0]["constants"]["num_stages"] = 2
            records["candidate"][0]["ptx_sha256"] = "changed"
            with self.assertRaisesRegex(RuntimeError, "original binary"):
                probe.freeze_records((None, module))

    def test_boundary_pages_and_handoff_restore_context(self):
        pages = []
        helper = types.SimpleNamespace(
            make_inputs=lambda *a, **kw: pages.append(kw["page_size"])
        )

        def bounds(production, h, torch, device):
            h.make_inputs([], page_size=196)
            return [
                dict(dtype=d, tile=t, warps=w, status="passed")
                for d, t, w in [
                    ("torch.bfloat16", 32, 8),
                    ("torch.bfloat16", 64, 4),
                    ("torch.float8_e4m3fn", 64, 4),
                ]
            ]

        with patch.object(probe, "BOUNDARIES", side_effect=bounds):
            checks = probe.boundary_checks(None, helper, None, None)
        self.assertEqual(pages, [196, 64, 16])
        self.assertEqual([v["page_dedup_active"] for v in checks], [True, True, False])
        self.assertEqual(
            [v["page_dedup_active"] for v in checks[0]["extra_page_checks"]],
            [True, False],
        )
        from tools import benchmark_qsa_prefill_triton_stages as bench
        from tools import qsa_prefill_flagtree_ab as profile

        original = (probe.base.__file__, probe.base.make_pair, probe.stage.scoped)

        def run(options):
            self.assertEqual(len(options.cases), 6)
            self.assertEqual(options.rounds, 2)
            self.assertEqual(
                Path(probe.base.__file__).resolve(), Path(probe.__file__).resolve()
            )
            self.assertIs(probe.base.make_pair, probe.make_pair)
            self.assertIs(probe.base.freeze_records, probe.freeze_records)
            self.assertTrue(probe.EXTRA_SOURCES <= set(bench.SOURCES))
            return 0

        with patch.object(
            sys,
            "argv",
            [probe.__file__, "--report-dir", "/tmp/r", "--baseline-report", "/tmp/b"],
        ), patch.object(profile, "configure"), patch.object(
            probe, "coordinate", side_effect=run
        ):
            self.assertEqual(probe.main(), 0)
        self.assertEqual(
            (probe.base.__file__, probe.base.make_pair, probe.stage.scoped), original
        )

    def test_summary_separates_unchanged_control(self):
        cases = [
            dict(
                case=v["case"]["name"],
                status="passed",
                stages={
                    k: dict(speedup=1.02 if i < 5 else 1.0)
                    for k in ("score", "topk", "pipeline")
                },
            )
            for i, v in enumerate(probe.manifest()["cases"])
        ]
        with tempfile.TemporaryDirectory() as folder:
            old = Path(folder) / "head-group-summary.json"
            old.write_text(json.dumps(dict(status="passed", cases=cases)))
            with patch.object(probe.base, "coordinate", return_value=0):
                self.assertEqual(
                    probe.coordinate(types.SimpleNamespace(report_dir=folder)), 0
                )
            s = json.loads((Path(folder) / "page-dedup-summary.json").read_text())
            self.assertEqual(s["cases"], cases)
            self.assertEqual(s["group_aggregates"]["changed_bf16"]["count"], 5)
            self.assertEqual(
                s["group_aggregates"]["unchanged_fp8_control"]["score"], 1.0
            )
            self.assertFalse(old.exists())


if __name__ == "__main__":
    unittest.main()
