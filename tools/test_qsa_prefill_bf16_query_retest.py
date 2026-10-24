# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Execute actual query mapping/reduction/store code and frozen retest policy."""

import ast
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_bf16_query_retest as probe


def function(path, name):
    return next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


class QueryRetestTests(unittest.TestCase):
    def test_request_key_mapping_signature_and_frozen_source(self):
        probe.verify_source()
        candidate = function(probe.KERNEL, "_query_split_score_kernel")
        original = function(
            probe.ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py",
            "_qsa_prefill_score_kernel",
        )
        self.assertEqual(ast.dump(candidate.args), ast.dump(original.args))
        self.assertEqual(
            [ast.dump(n) for n in candidate.body[:6]],
            [ast.dump(n) for n in original.body[:6]],
        )
        a = next(n for n in candidate.body if isinstance(n, ast.For))
        b = next(n for n in original.body if isinstance(n, ast.For))
        index = next(
            i
            for i, n in enumerate(a.body)
            if isinstance(n, ast.Assign) and n.targets[0].id == "scores_left"
        )
        self.assertEqual(
            [ast.dump(n) for n in a.body[:index]], [ast.dump(n) for n in b.body[:index]]
        )

    def test_actual_query_halves_relu_head_sum_masks_and_canaries(self):
        import numpy as np

        fn = function(probe.KERNEL, "_query_split_score_kernel")
        first = next(
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.AnnAssign) and n.target.id == "HALF_R"
        )
        end = next(
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "column_lanes"
        )
        loop = next(n for n in fn.body if isinstance(n, ast.For))
        begin = next(
            i
            for i, n in enumerate(loop.body)
            if isinstance(n, ast.Assign) and n.targets[0].id == "scores_left"
        )
        body = copy.deepcopy(fn.body[first:end] + loop.body[begin:])

        class Pointer:
            def __init__(self, data, offset=0):
                self.data, self.offset = data, np.asarray(offset)

            def __add__(self, value):
                return Pointer(self.data, self.offset + value)

        def load(ptr, mask, other):
            offsets, masks = np.broadcast_arrays(ptr.offset, mask)
            self.assertTrue(
                np.all((offsets[masks] >= 0) & (offsets[masks] < ptr.data.size))
            )
            out = np.full(offsets.shape, other, dtype=np.float32)
            out[masks] = ptr.data[offsets[masks]]
            return out

        def store(ptr, values, mask):
            offsets, values, masks = np.broadcast_arrays(ptr.offset, values, mask)
            self.assertTrue(
                np.all((offsets[masks] >= 64) & (offsets[masks] < ptr.data.size - 64))
            )
            ptr.data[offsets[masks]] = values[masks]

        tl = types.SimpleNamespace(
            arange=np.arange,
            load=load,
            store=store,
            dot=lambda a, b, **kw: a @ b,
            reshape=np.reshape,
            trans=np.transpose,
            split=lambda x: (x[:, 0], x[:, 1]),
            maximum=np.maximum,
            sum=np.sum,
            where=np.where,
            float32=np.float32,
            constexpr=object(),
        )
        rng = np.random.default_rng(41)
        q = rng.standard_normal((129, 4, 132)).astype(np.float32)
        keys = rng.standard_normal((64, 128)).astype(np.float32)
        visible = rng.integers(0, 65, 129)
        for start in (0, 5, 64, 97):
            for count in (1, 31, 32, 33, 63, 64):
                stop = min(start + count, 129)
                offset = max(0, start - 3)
                rows = stop - offset
                out = np.full(128 + rows * 64, -12345.0, dtype=np.float32)
                expected = out.copy()
                vis = np.zeros(64, dtype=np.int32)
                vis[: stop - start] = visible[start:stop]
                live = (np.arange(64) < max(vis)) & (np.arange(64) % 11 != 0)
                env = dict(
                    tl=tl,
                    triton=types.SimpleNamespace(
                        next_power_of_2=lambda n: 1 << (n - 1).bit_length()
                    ),
                    TILE_R=64,
                    HEAD_DIM=128,
                    HEADS_PADDED=4,
                    NUM_HEADS=4,
                    BLOCK_N=64,
                    q_ptr=Pointer(q.ravel()),
                    stride_q_row=4 * 132,
                    stride_q_head=132,
                    row_start=start,
                    request_end=stop,
                    visible=vis,
                    keys=keys,
                    columns=np.arange(64),
                    live=live,
                    query_offset=offset,
                    logits_width=64,
                    logits_ptr=Pointer(out, 64),
                )
                exec(
                    compile(
                        ast.Module(body=body, type_ignores=[]),
                        "actual_query_split",
                        "exec",
                    ),
                    env,
                )
                for row in range(start, stop):
                    values = np.maximum(keys @ q[row, :, :128].T, 0).sum(axis=1)
                    values = np.where(live, values, -np.inf)
                    indices = 64 + (row - offset) * 64 + np.arange(visible[row])
                    expected[indices] = values[: visible[row]]
                np.testing.assert_allclose(out, expected, rtol=2e-5, atol=1e-4)

    def test_fixed_configs_replace_head_group_only_for_active_chunks(self):
        counts = []
        for item in probe.manifest()["cases"]:
            left = probe.split_traces(item["traces"])
            right = probe.split_traces(item["traces"], True)
            active = 0
            for a, b in zip(left, right):
                c = item["case"]
                named = dict(
                    a["constants"],
                    IS_FP8=c["dtype"] == "float8_e4m3fn",
                    NUM_HEADS=c["heads"],
                    HEAD_DIM=c["dim"],
                )
                yes = a["route"] == "rectangular" and probe.eligible(named)
                self.assertEqual(
                    b["constants"],
                    dict(a["constants"], HEAD_GROUP=False) if yes else a["constants"],
                )
                active += yes
            self.assertEqual(left, item["traces"])
            counts.append(active)
        self.assertEqual(counts, [0, 0, 1, 4, 1, 0])

    def test_actual_factory_applies_only_the_head_mode_switch(self):
        production = types.ModuleType("fake_production")
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
        kernel = object()
        for item in probe.manifest()["cases"]:
            pair = probe.make_pair(production, item["traces"], kernel)
            self.assertIs(pair[0].persistent_topk, pair[1].persistent_topk)
            for name, route in probe.base.ENTRIES:
                a, b = getattr(pair[0], name), getattr(pair[1], name)
                expected = {
                    (t["rows"], t["offset"], t["width"]): t["constants"]
                    for t in probe.split_traces(item["traces"], True)
                    if t["route"] == route
                }
                self.assertEqual(b.configs, expected)
                self.assertIs(a.jit, b.jit.original)
                self.assertIs(b.jit.candidate, kernel)
            self.assertIs(
                pair[1].qsa_select_paged_prefill(), pair[1]._qsa_prefill_score_kernel
            )

    def test_actual_delegate_and_fallback_gate(self):
        calls = []

        class JIT:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, grid):
                def run(*args, **kw):
                    calls.append((self.name, grid, args, kw))
                    return self.name

                return run

        names = [
            "num_rows",
            "query_offset",
            "logits_width",
            "IS_FP8",
            "NUM_HEADS",
            "HEAD_DIM",
        ]
        for route, fp8, h, d, tile, kt, wanted in [
            ("rectangular", False, 4, 128, 64, 16, "split"),
            ("rectangular", False, 4, 128, 32, 16, "original"),
            ("rectangular", False, 4, 128, 64, 4, "original"),
            ("rectangular", False, 8, 128, 64, 16, "original"),
            ("rectangular", True, 4, 128, 64, 16, "original"),
            ("compact", False, 4, 128, 64, 16, "original"),
        ]:
            jit = probe.RoutedJIT(JIT("original"), JIT("split"), names, route)
            args = (17, 31, 1088, fp8, h, d)
            kw = dict(TILE_R=tile, K_TILES=kt, HEAD_GROUP=False, num_stages=2)
            self.assertEqual(jit[(1, 2, 3)](*args, **kw), wanted)
            self.assertEqual(calls[-1], (wanted, (1, 2, 3), args, kw))
        self.assertFalse(probe.eligible({}))

    def test_freeze_proves_head_mode_change_and_unchanged_controls(self):
        original = dict(
            chunk=[7, 0, 1088],
            route="rectangular",
            constants=dict(HEAD_GROUP=True, num_stages=2),
            ptx_sha256="old",
            cubin_sha256="old",
        )
        candidate = copy.deepcopy(original)
        candidate["constants"]["HEAD_GROUP"] = False
        records = dict(baseline=[original], candidate=[candidate])
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
                    "head_group_replaced"
                ]
            )
            candidate["constants"]["num_stages"] = 1
            with self.assertRaisesRegex(RuntimeError, "non-algorithm"):
                probe.freeze_records((None, module))
            candidate["constants"] = dict(original["constants"])
            candidate["ptx_sha256"] = "different"
            getattr(module, probe.base.ENTRIES[0][0]).jit.decisions[(7, 0, 1088)][
                "active"
            ] = False
            with self.assertRaisesRegex(RuntimeError, "original binary"):
                probe.freeze_records((None, module))

    def test_boundary_schedules_and_cli_handoff(self):
        fp8, bf16 = object(), object()
        torch = types.SimpleNamespace(bfloat16=bf16, float8_e4m3fn=fp8)
        seen = []

        def fixture(p, h, t, d, dtype, tile, warps, counts):
            seen.append((tile, dtype, counts))
            return dict(status="passed")

        with patch.object(probe, "boundary_fixture", side_effect=fixture):
            checks = probe.boundary_checks(None, None, torch, None)
        self.assertEqual(len(seen), 6)
        self.assertEqual([x[2] for x in seen], [(129,), (33, 64, 32)] * 3)
        self.assertEqual(len(checks), 3)
        from tools import benchmark_qsa_prefill_triton_stages as bench
        from tools import qsa_prefill_flagtree_ab as profile

        old = (probe.base.__file__, probe.base.make_pair, probe.stage.scoped)

        def coordinate(o):
            self.assertEqual(len(o.cases), 6)
            self.assertEqual(o.rounds, 2)
            self.assertIs(probe.base.make_pair, probe.make_pair)
            self.assertIs(probe.base.freeze_records, probe.freeze_records)
            self.assertEqual(
                Path(probe.base.__file__).resolve(), Path(probe.__file__).resolve()
            )
            self.assertTrue(probe.EXTRA_SOURCES <= set(bench.SOURCES))
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

    def test_summary_keeps_binary_controls_separate(self):
        m = probe.manifest()
        active = [0, 0, 1, 4, 1, 0]
        cases = [
            dict(
                case=v["case"]["name"],
                status="passed",
                stages={
                    k: dict(speedup=1.02 if active[i] else 1.0)
                    for k in ("score", "topk", "pipeline")
                },
            )
            for i, v in enumerate(m["cases"])
        ]
        with tempfile.TemporaryDirectory() as folder:
            old = Path(folder) / "head-group-summary.json"
            old.write_text(json.dumps(dict(status="passed", cases=cases)))
            for i, c in enumerate(cases):
                (Path(folder) / (c["case"] + ".json")).write_text(
                    json.dumps(dict(gate_summary=dict(active_chunks=active[i])))
                )
            with patch.object(probe.base, "coordinate", return_value=0):
                self.assertEqual(
                    probe.coordinate(types.SimpleNamespace(report_dir=folder)), 0
                )
            result = json.loads((Path(folder) / "query-split-summary.json").read_text())
            self.assertEqual(result["cases"], cases)
            self.assertEqual(
                result["group_aggregates"]["query_split_active"]["count"], 3
            )
            self.assertEqual(
                result["group_aggregates"]["unchanged_controls"]["count"], 3
            )
            self.assertEqual(
                result["group_aggregates"]["unchanged_controls"]["score"], 1.0
            )


if __name__ == "__main__":
    unittest.main()
