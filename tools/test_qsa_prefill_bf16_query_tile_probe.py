# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Exercise larger-tile mapping, frozen configs, launch grid and binary proof."""

import ast
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_bf16_query_tile_probe as probe


def function(path, name):
    return next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


class QueryTileTests(unittest.TestCase):
    def test_kernel_is_production_except_group_tile_guard(self):
        probe.verify_source()
        candidate = function(probe.KERNEL, "_query_tile_score_kernel")
        original = function(
            probe.ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py",
            "_qsa_prefill_score_kernel",
        )
        self.assertEqual(ast.dump(candidate.args), ast.dump(original.args))
        candidate.name = original.name
        candidate.decorator_list = original.decorator_list = []
        candidate.body = candidate.body[2:]  # private input-domain assertions
        group = next(n for n in candidate.body if isinstance(n, ast.AnnAssign))
        self.assertEqual(group.target.id, "GROUP_HEADS")
        group.value.values[-1] = ast.parse("TILE_R == 64", mode="eval").body
        self.assertEqual(ast.dump(candidate), ast.dump(original))

    def test_actual_kernel_rows_heads_page_mapping_masks_and_canaries(self):
        import numpy as np

        class Pointer:
            def __init__(self, data, offset=0):
                self.data, self.offset = data, np.asarray(offset)

            def __add__(self, value):
                return Pointer(self.data, self.offset + value)

        def load(ptr, mask=True, other=0):
            offsets, masks = np.broadcast_arrays(ptr.offset, mask)
            self.assertTrue(
                np.all((offsets[masks] >= 0) & (offsets[masks] < ptr.data.size))
            )
            out = np.full(offsets.shape, other, dtype=ptr.data.dtype)
            out[masks] = ptr.data[offsets[masks]]
            return out

        def store(ptr, values, mask):
            offsets, values, masks = np.broadcast_arrays(ptr.offset, values, mask)
            self.assertTrue(
                np.all((offsets[masks] >= 64) & (offsets[masks] < ptr.data.size - 64))
            )
            ptr.data[offsets[masks]] = values[masks]

        class ArrayCast(ast.NodeTransformer):
            def visit_Call(self, node):
                node = self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
                    return ast.Call(
                        func=ast.Name(id="cast", ctx=ast.Load()),
                        args=[node.func.value, *node.args],
                        keywords=[],
                    )
                return node

        fn = copy.deepcopy(function(probe.KERNEL, "_query_tile_score_kernel"))
        fn.decorator_list = []
        fn = ArrayCast().visit(fn)
        program = compile(
            ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
            "actual_larger_tile",
            "exec",
        )
        rng = np.random.default_rng(59)
        rows, width, page = 259, 128, 196
        q = rng.standard_normal((rows, 4, 132)).astype(np.float32)
        cache = rng.standard_normal((4, page, 132)).astype(np.float32)
        starts = np.asarray([97, 98, 227, 227, 356], dtype=np.int32)
        table = np.asarray([[2], [1], [-1], [3]], dtype=np.int32)
        visible = np.asarray(
            ([0, 1, 31, 32, 33, 63, 64, 65, 127, 128] * 26)[:rows], dtype=np.int32
        )
        for tile, grouped in ((64, False), (128, True)):
            for offset, count in ((0, 259), (3, 128), (64, 129), (127, 132)):
                out = np.full(128 + count * width, -12345.0, dtype=np.float32)
                expected = out.copy()
                pid = [0, 0, 0]
                tl = types.SimpleNamespace(
                    constexpr=object(),
                    float32=np.float32,
                    int64=np.int64,
                    static_assert=lambda yes: self.assertTrue(yes),
                    program_id=lambda axis: pid[axis],
                    load=load,
                    store=store,
                    maximum=np.maximum,
                    minimum=np.minimum,
                    max=np.max,
                    cdiv=lambda a, b: (a + b - 1) // b,
                    arange=np.arange,
                    dot=lambda a, b, **kw: a @ b,
                    reshape=np.reshape,
                    sum=np.sum,
                    where=np.where,
                )
                env = dict(
                    tl=tl,
                    triton=types.SimpleNamespace(
                        next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
                        cdiv=tl.cdiv,
                    ),
                    cast=lambda a, dtype: a.astype(dtype),
                )
                exec(program, env)
                run = env["_query_tile_score_kernel"]
                kwargs = dict(
                    q_ptr=Pointer(q.ravel()),
                    cache_ptr=Pointer(cache.ravel()),
                    table_ptr=Pointer(table.ravel()),
                    starts_ptr=Pointer(starts),
                    visible_ptr=Pointer(visible),
                    logits_ptr=Pointer(out, 64),
                    stride_q_row=4 * 132,
                    stride_q_head=132,
                    stride_cache_block=page * 132,
                    stride_cache_token=132,
                    stride_table_req=1,
                    stride_table_page=1,
                    num_rows=count,
                    query_offset=offset,
                    max_query_len=132,
                    logits_width=width,
                    num_requests=4,
                    num_pages=4,
                    table_width=1,
                    PAGE_SIZE=page,
                    NUM_HEADS=4,
                    HEAD_DIM=128,
                    IS_FP8=False,
                    TILE_R=tile,
                    BLOCK_N=64,
                    K_TILES=16,
                    HEAD_GROUP=grouped,
                )
                for request in range(4):
                    pid[0] = request
                    for query_tile in range(tl.cdiv(132, tile)):
                        pid[1] = query_tile
                        run(**kwargs)
                for request in range(4):
                    lo, hi = starts[request : request + 2] - 97
                    for row in range(max(lo, offset), min(hi, offset + count)):
                        scores = np.maximum(
                            cache[table[request, 0], :width, :128] @ q[row, :, :128].T,
                            0,
                        ).sum(axis=1)
                        n = visible[row]
                        expected[
                            64
                            + (row - offset) * width : 64
                            + (row - offset) * width
                            + n
                        ] = scores[:n]
                np.testing.assert_allclose(out, expected, rtol=2e-5, atol=1e-4)

    def test_config_change_factory_and_actual_candidate_grid(self):
        production = types.ModuleType("fake_production")
        production.persistent_topk = object()
        names = [
            "num_rows",
            "query_offset",
            "logits_width",
            "IS_FP8",
            "NUM_HEADS",
            "HEAD_DIM",
        ]
        calls = []

        class JIT:
            def __getitem__(self, grid):
                def run(*args, **kw):
                    calls.append((grid, args, kw))
                    return object()

                return run

        for name, _ in probe.base.ENTRIES:
            setattr(
                production,
                name,
                types.SimpleNamespace(
                    arg_names=names, fn=types.SimpleNamespace(fn=JIT())
                ),
            )
        exec(
            "def _prefill_logits(): return _qsa_prefill_score_kernel\n"
            "def qsa_select_paged_prefill(): return _prefill_logits()",
            vars(production),
        )
        counts = []
        for item in probe.manifest()["cases"]:
            before = copy.deepcopy(item["traces"])
            pair = probe.make_pair(production, item["traces"], JIT())
            active = 0
            for t in before:
                name = next(n for n, route in probe.base.ENTRIES if route == t["route"])
                args = (
                    t["rows"],
                    t["offset"],
                    t["width"],
                    item["case"]["dtype"] == "float8_e4m3fn",
                    4,
                    128,
                )
                left, right = [getattr(m, name) for m in pair]
                key = (t["rows"], t["offset"], t["width"])
                changed = t["route"] == "rectangular"
                self.assertEqual(
                    right.configs[key],
                    (
                        dict(left.configs[key], TILE_R=left.configs[key]["TILE_R"] * 2)
                        if changed
                        else left.configs[key]
                    ),
                )
                grid = lambda c: (4, (8192 + c["TILE_R"] - 1) // c["TILE_R"], 10)
                right[grid](*args)
                self.assertEqual(calls[-1][0], grid(right.configs[key]))
                self.assertEqual(right.jit.decisions[key]["active"], changed)
                active += changed
            self.assertEqual(before, item["traces"])
            self.assertIs(pair[0].persistent_topk, pair[1].persistent_topk)
            counts.append(active)
        self.assertEqual(counts, [2, 4, 4, 4, 1, 0])

    def test_freeze_proof_rejects_head_or_stage_change_and_binary_change(self):
        a = dict(
            chunk=[7, 0, 1088],
            route="rectangular",
            constants=dict(TILE_R=64, HEAD_GROUP=True, num_stages=2),
            ptx_sha256="old",
            cubin_sha256="old",
        )
        b = copy.deepcopy(a)
        b["constants"]["TILE_R"] = 128
        records = dict(baseline=[a], candidate=[b])
        module = types.SimpleNamespace()
        for i, (name, _) in enumerate(probe.base.ENTRIES):
            setattr(
                module,
                name,
                types.SimpleNamespace(
                    jit=types.SimpleNamespace(
                        decisions={(7, 0, 1088): dict(active=True)} if i == 0 else {}
                    )
                ),
            )
        with patch.object(probe, "FREEZE", return_value=records):
            self.assertTrue(
                probe.freeze_records((None, module))["candidate"][0][
                    "head_group_preserved"
                ]
            )
            b["constants"]["HEAD_GROUP"] = False
            with self.assertRaisesRegex(RuntimeError, "non-tile"):
                probe.freeze_records((None, module))
            b["constants"] = dict(a["constants"])
            b["ptx_sha256"] = "different"
            getattr(module, probe.base.ENTRIES[0][0]).jit.decisions[(7, 0, 1088)][
                "active"
            ] = False
            with self.assertRaisesRegex(RuntimeError, "original binary"):
                probe.freeze_records((None, module))

    def test_boundaries_default_cli_handoff_and_restoration(self):
        fp8, bf16 = object(), object()
        torch = types.SimpleNamespace(bfloat16=bf16, float8_e4m3fn=fp8)
        seen = []

        def fixture(p, h, t, d, dtype, tile, warps, counts):
            seen.append((tile, dtype, counts))
            return dict(status="passed")

        with patch.object(probe, "boundary_fixture", side_effect=fixture):
            self.assertEqual(len(probe.boundary_checks(None, None, torch, None)), 3)
        self.assertEqual([x[2] for x in seen], [(259,), (127, 128, 4)] * 3)
        from tools import benchmark_qsa_prefill_triton_stages as bench
        from tools import qsa_prefill_flagtree_ab as profile

        old = (probe.base.__file__, probe.base.make_pair, probe.stage.scoped)

        def coordinate(o):
            self.assertEqual(tuple(o.cases), probe.DEFAULT_CASES)
            self.assertEqual(o.rounds, 2)
            self.assertIs(probe.base.make_pair, probe.make_pair)
            self.assertEqual(
                Path(probe.base.__file__).resolve(), Path(probe.__file__).resolve()
            )
            self.assertTrue(probe.EXTRA_SOURCES <= set(bench.SOURCES))
            self.assertTrue(
                all((probe.ROOT / n).is_file() for n in probe.EXTRA_SOURCES)
            )
            return 0

        with patch.object(
            sys,
            "argv",
            [probe.__file__, "--report-dir", "/tmp/r", "--baseline-report", "/tmp/b"],
        ), patch.object(profile, "configure"), patch.object(
            probe, "coordinate", side_effect=coordinate
        ):
            self.assertEqual(probe.main(), 0)
        self.assertEqual(
            (probe.base.__file__, probe.base.make_pair, probe.stage.scoped), old
        )

    def test_aggregation_separates_active_from_unchanged_controls(self):
        cases = [
            dict(
                case=name,
                status="passed",
                stages={
                    k: dict(speedup=(1.1 if i < 2 else 1.0))
                    for k in ("score", "topk", "pipeline")
                },
            )
            for i, name in enumerate(probe.DEFAULT_CASES)
        ]
        with tempfile.TemporaryDirectory() as folder:
            old = Path(folder) / "head-group-summary.json"
            old.write_text(json.dumps(dict(status="passed", cases=cases)))
            for i, c in enumerate(cases):
                (Path(folder) / (c["case"] + ".json")).write_text(
                    json.dumps(
                        dict(gate_summary=dict(active_chunks=(2 if i < 2 else 0)))
                    )
                )
            with patch.object(probe.base, "coordinate", return_value=0):
                self.assertEqual(
                    probe.coordinate(types.SimpleNamespace(report_dir=folder)), 0
                )
            summary = json.loads((Path(folder) / "query-tile-summary.json").read_text())
            self.assertEqual(
                summary["group_aggregates"]["query_tile_active"]["count"], 2
            )
            self.assertEqual(
                summary["group_aggregates"]["unchanged_controls"]["score"], 1.0
            )


if __name__ == "__main__":
    unittest.main()
