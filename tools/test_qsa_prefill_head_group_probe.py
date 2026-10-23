# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Execute real query/head mapping and fixed-choice probe orchestration on CPU."""

import ast
import contextlib
import copy
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_head_group_probe as probe


class HeadGroupTests(unittest.TestCase):
    def test_manifest_and_read_only_production_identity(self):
        data = probe.manifest()
        self.assertEqual(len(data["cases"]), 6)
        self.assertEqual(
            sum(c["case"]["dtype"] == "bfloat16" for c in data["cases"]), 5
        )
        self.assertEqual(
            data["report_commit"], "e1c188ecf888d876674023447d99ee8249df8dea"
        )
        self.assertEqual(
            probe.verify_source()["production_sha256"], data["production_sha256"]
        )
        for c in data["cases"]:
            for t in c["traces"]:
                self.assertEqual(t["constants"]["num_stages"], 2)
                self.assertEqual(t["constants"]["K_TILES"], 16)

    def test_actual_query_mapping_and_head_relu_reduction(self):
        import numpy as np

        path = probe.ROOT / "tools/qsa_prefill_head_group_kernel.py"
        fn = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef)
        )
        first = next(
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "query_lanes"
        )
        end = next(
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "column_lanes"
        )
        loop = next(n for n in fn.body if isinstance(n, ast.For))
        a = next(
            i
            for i, n in enumerate(loop.body)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "low"
        )
        body = copy.deepcopy(fn.body[first:end] + loop.body[a:-1])

        class Pointer:
            def __init__(self, data, offset=0):
                self.data, self.offset = data, np.asarray(offset)

            def __add__(self, value):
                return Pointer(self.data, self.offset + value)

        def load(ptr, mask, other):
            offsets, mask = np.broadcast_arrays(ptr.offset, mask)
            self.assertTrue(
                np.all((offsets[mask] >= 0) & (offsets[mask] < ptr.data.size))
            )
            out = np.full(offsets.shape, other, dtype=np.float32)
            out[mask] = ptr.data[offsets[mask]]
            return out

        tl = types.SimpleNamespace(
            arange=lambda a, b: np.arange(a, b),
            load=load,
            dot=lambda a, b, **kw: a @ b,
            reshape=np.reshape,
            maximum=np.maximum,
            sum=np.sum,
            float32=np.float32,
            constexpr=object(),
        )
        rng = np.random.default_rng(17)
        for tile in (4, 32, 64):
            for dim in (8, 24, 128):
                rows = 17
                q = rng.standard_normal((rows, 4, dim + 4)).astype(np.float32)
                d = max(32, 1 << (dim - 1).bit_length())
                keys = np.zeros((64, d), dtype=np.float32)
                keys[:, :dim] = rng.standard_normal((64, dim))
                env = dict(
                    tl=tl,
                    triton=types.SimpleNamespace(
                        next_power_of_2=lambda n: 1 << (n - 1).bit_length()
                    ),
                    IS_FP8=True,
                    q_ptr=Pointer(q.ravel()),
                    BLOCK_R=max(tile, 8),
                    HEADS_PADDED=2,
                    row_start=3,
                    NUM_HEADS=4,
                    HEAD_DIM=dim,
                    TILE_R=tile,
                    BLOCK_N=64,
                    request_end=17,
                    stride_q_row=4 * (dim + 4),
                    stride_q_head=dim + 4,
                    dims=np.arange(d),
                    keys=keys,
                )
                exec(
                    compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"),
                    env,
                )
                expected = np.zeros((64, max(tile, 8)), dtype=np.float32)
                for row in range(min(tile, rows - 3)):
                    expected[:, row] = np.maximum(
                        keys[:, :dim] @ q[row + 3, :, :dim].T, 0
                    ).sum(axis=1)
                np.testing.assert_allclose(
                    env["scores"], expected, rtol=2e-5, atol=1e-4
                )

    def test_key_loads_request_mapping_and_bf16_mask_are_exact(self):
        def fn(path, name):
            return next(
                n
                for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )

        old = fn(
            probe.ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py",
            "_qsa_prefill_score_kernel",
        )
        new = fn(
            probe.ROOT / "tools/qsa_prefill_head_group_kernel.py",
            "_head_group_score_kernel",
        )
        self.assertEqual(
            [ast.dump(n) for n in old.body[:6]], [ast.dump(n) for n in new.body[:6]]
        )
        oldloop = next(n for n in old.body if isinstance(n, ast.For))
        newloop = next(n for n in new.body if isinstance(n, ast.For))
        self.assertEqual(
            [ast.dump(n) for n in oldloop.body[:7]],
            [ast.dump(n) for n in newloop.body[:7]],
        )
        self.assertEqual(
            ast.dump(oldloop.body[-1]), ast.dump(newloop.body[-1].orelse[0])
        )
        self.assertNotIn(
            "torch", (probe.ROOT / "tools/qsa_prefill_head_group_kernel.py").read_text()
        )

    def test_clone_routes_all_entries_without_mutating_source(self):
        candidate = object()
        fake_kernel = types.ModuleType("tools.qsa_prefill_head_group_kernel")
        fake_kernel._head_group_score_kernel = candidate
        source = types.ModuleType("production")
        source.persistent_topk = object()

        class Entry:
            def __init__(self):
                self.arg_names = []
                self.fn = types.SimpleNamespace(fn=object())

        originals = {}
        for name, _ in probe.ENTRIES:
            entry = Entry()
            setattr(source, name, entry)
            originals[name] = entry
        exec(
            "def _prefill_logits():\n return _qsa_prefill_fp8_store_score_kernel\n"
            "def qsa_select_paged_prefill():\n return _prefill_logits()",
            vars(source),
        )
        traces = [dict(rows=3, offset=0, width=8, route="fp8", constants={})]
        with patch.dict(sys.modules, {fake_kernel.__name__: fake_kernel}):
            pair = probe.make_pair(source, traces)
        for name, route in probe.ENTRIES:
            self.assertIs(getattr(source, name), originals[name])
            self.assertIs(getattr(pair[0], name).jit, originals[name].fn.fn)
            self.assertIs(
                getattr(pair[1], name).jit,
                candidate if route != "compact" else originals[name].fn.fn,
            )
        self.assertIs(
            pair[1].qsa_select_paged_prefill(),
            pair[1]._qsa_prefill_fp8_store_score_kernel,
        )
        self.assertIs(pair[0].persistent_topk, pair[1].persistent_topk)

    def run_worker(self, root, fail=False):
        from tools import benchmark_qsa_prefill_triton_stages as bench

        class View:
            def __getitem__(self, key):
                return self

        tensor = View()
        args = (tensor,) * 10
        topk = lambda *a, **kw: None
        pair = tuple(
            types.SimpleNamespace(
                _prefill_logits=lambda *a, **kw: tensor, persistent_topk=topk
            )
            for _ in range(2)
        )
        chunk = dict(
            offset=0,
            rows=3,
            width=8,
            score_args=(tensor,),
            score_kwargs={},
            topk_args=(tensor, tensor, tensor, tensor),
            topk_kwargs={},
            common_logits=tensor,
        )
        case = probe.manifest()["cases"][0]["case"]
        item = dict(
            case=case, inputs=["stable"], traces=[dict(offset=0, rows=3, width=8)]
        )
        options = types.SimpleNamespace(
            report_dir=str(root),
            worker_case=case["name"],
            cases=[case["name"]],
            rounds=2,
            kernel_warmup_ms=200,
            kernel_rep_ms=100,
        )
        timer_calls = []

        def timer(fn):
            timer_calls.append(fn)
            fn()
            return 10.0

        def pipeline(*a, **kw):
            if fail:
                raise AssertionError("injected pipeline failure")
            return 3

        patches = (
            patch.object(probe, "verify_source", return_value={"stable": True}),
            patch.object(probe, "manifest", return_value=dict(compiler_identity={})),
            patch.object(probe, "saved_record", return_value=item),
            patch.object(probe, "compiler_identity", return_value={}),
            patch.object(
                probe, "boundary_checks", return_value=[dict(status="passed")] * 3
            ),
            patch.object(probe, "make_pair", return_value=pair),
            patch.object(
                probe, "freeze_records", return_value=dict(baseline=[], candidate=[])
            ),
            patch.object(probe, "artifact_records", return_value=[]),
            patch.object(
                bench,
                "runtime",
                return_value=(
                    object(),
                    object(),
                    object(),
                    None,
                    object(),
                    types.SimpleNamespace(synchronize=lambda: None),
                    dict(uuid="GPU-test"),
                ),
            ),
            patch.object(bench.workloads, "make_inputs", return_value=args),
            patch.object(bench.support, "fingerprint", return_value=["stable"]),
            patch.object(bench.ab, "observe", return_value=[chunk]),
            patch.object(bench, "verify_geometry"),
            patch.object(bench.support, "sample_rows", return_value=[0]),
            patch.object(bench.scoring, "validate_scores", return_value=3),
            patch.object(bench, "validate_reference_selection"),
            patch.object(bench, "validate_pipeline", side_effect=pipeline),
            patch.object(
                bench.scoring, "make_timer", return_value=(timer, dict(mode="kernel"))
            ),
            patch.object(bench.support, "gpu_guard_module", return_value=object()),
            patch.object(
                bench.support, "guarded_telemetry", return_value=dict(status="passed")
            ),
        )
        with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()):
            for p in patches:
                stack.enter_context(p)
            code = probe.worker(options)
        return (
            code,
            json.loads((root / (case["name"] + ".json")).read_text()),
            timer_calls,
        )

    def test_worker_times_all_stages_and_controls_with_pre_post_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            code, result, timers = self.run_worker(Path(folder))
            self.assertEqual(code, 0, result.get("error"))
            self.assertEqual(set(result["stages"]), {"score", "topk", "pipeline"})
            self.assertEqual(len(result["self_controls"]), 4)
            self.assertEqual(len(timers), 28)
            self.assertTrue(result["same_topk"] and result["same_buffers"])
            self.assertFalse(
                result["production_modified"] or result["acceptance_eligible"]
            )
            for v in {**result["stages"], **result["self_controls"]}.values():
                self.assertTrue(v["measurement_validated"])
                self.assertEqual(
                    v["measurement_order"],
                    [["baseline", "candidate"], ["candidate", "baseline"]],
                )
                self.assertTrue(v["validation_before"] and v["validation_after"])

    def test_worker_stops_on_numeric_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            code, result, _ = self.run_worker(Path(folder), fail=True)
            self.assertEqual(code, 1)
            self.assertEqual(result["status"], "failed")
            self.assertIn("injected pipeline failure", result["error"])


if __name__ == "__main__":
    unittest.main()
