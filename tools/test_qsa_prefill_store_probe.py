# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Store-mask partition, exact fallback, frozen source and paired-timer contracts."""

import ast
import contextlib
import copy
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_query_split_matched as fixed
from tools import qsa_prefill_store_probe as probe


class StoreProbeTests(unittest.TestCase):
    def test_only_output_stores_change_in_the_private_scorer(self):
        probe.verify_source()
        old = ast.parse(
            (probe.ROOT / "tools/qsa_prefill_query_page_kernel.py").read_text()
        )
        new = ast.parse(
            (probe.ROOT / "tools/qsa_prefill_store_probe_kernel.py").read_text()
        )
        old.body.pop(0)
        new.body.pop(0)
        helper = next(
            n
            for n in new.body
            if isinstance(n, ast.FunctionDef) and n.name == "_bulk_tail_store"
        )
        new.body.remove(helper)
        expected = ast.parse(
            """
@triton.jit
def _bulk_tail_store(ptrs, values, valid_rows, columns, visible, logits_width):
    limit = tl.minimum(visible, logits_width)
    full = tl.multiple_of(limit // 4 * 4, 4)
    tl.store(ptrs, values, mask=valid_rows & (columns < full))
    tl.store(ptrs, values, mask=valid_rows & (columns >= full) & (columns < limit))
"""
        ).body[0]
        self.assertEqual(ast.dump(helper), ast.dump(expected))
        original = [
            n
            for n in ast.walk(old)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.store"
        ]
        replaced = [
            n
            for n in ast.walk(new)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "_bulk_tail_store"
        ]
        self.assertEqual(len(original), 3)
        self.assertEqual(len(replaced), 3)
        for a, b in zip(original, replaced):
            self.assertEqual(
                [ast.dump(n) for n in a.args], [ast.dump(n) for n in b.args]
            )
            params = {k.arg: ast.unparse(k.value) for k in b.keywords}
            mask = ast.parse(
                f"({params['valid_rows']}) & ({params['columns']} < {params['visible']}) "
                f"& ({params['columns']} < {params['logits_width']})",
                mode="eval",
            ).body
            self.assertEqual(ast.dump(a.keywords[0].value), ast.dump(mask))
            b.func, b.keywords = copy.deepcopy(a.func), copy.deepcopy(a.keywords)
        self.assertEqual(ast.dump(old), ast.dump(new))

    def test_bulk_tail_masks_are_disjoint_and_cover_only_original_output(self):
        for width in (4, 8, 80, 29696):
            for visible in (0, 1, 2, 3, 4, 5, 7, 8, 9, width - 1, width, width + 1):
                limit = min(width, visible)
                full = limit // 4 * 4
                self.assertEqual(full % 4, 0)
                for valid in (False, True):
                    for column in range(-0, min(width + 4, 100)):
                        bulk = valid and column < full
                        tail = valid and column >= full and column < limit
                        expected = valid and column < visible and column < width
                        self.assertFalse(bulk and tail)
                        self.assertEqual(int(bulk) + int(tail), int(expected))

    def tensor(self, **changes):
        values = dict(
            element_size=lambda: 4,
            dtype="torch.float32",
            is_contiguous=lambda: True,
            ndim=2,
            shape=(4, 80),
            data_ptr=lambda: 256,
        )
        values.update(changes)
        return types.SimpleNamespace(**values)

    def test_alignment_gate_and_original_jit_fallback(self):
        calls = []

        class JIT:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, grid):
                return lambda *a, **kw: calls.append((self.name, grid, a, kw))

        jit = probe.StoreJIT(
            JIT("original"), JIT("bulk"), ["logits_ptr", "logits_width"]
        )
        for value, width, side in (
            (self.tensor(), 80, "bulk"),
            (self.tensor(data_ptr=lambda: 260), 80, "original"),
            (self.tensor(shape=(4, 81)), 81, "original"),
            (self.tensor(is_contiguous=lambda: False), 80, "original"),
            (self.tensor(dtype="torch.int32"), 80, "original"),
        ):
            jit[(1, 2, 3)](value, width, num_warps=4, QUERY_PARTS=2)
            name, grid, args, kwargs = calls[-1]
            self.assertEqual(name, side)
            self.assertEqual(grid, (1, 2, 3))
            self.assertIs(args[0], value)
            self.assertEqual(kwargs, dict(num_warps=4, QUERY_PARTS=2))

    def test_saved_trace_requires_original_geometry_and_query_mode(self):
        for name in probe.CASES:
            case = dict(
                name=name,
                requests=1,
                query_length=16384,
                prefix_tokens=102400,
                heads=4,
                dim=128,
                ratio=4,
                page_size=196,
            )
            if name == probe.CASES[1]:
                case.update(
                    query_lengths=[16384],
                    prefix_tokens=[102400],
                    query_base=0,
                    batch_max_seq_len=118784,
                )
            traces = [
                dict(
                    offset=o,
                    rows=n,
                    width=29696,
                    route="rectangular",
                    constants=dict(
                        probe.shared.CONFIG,
                        QUERY_PARTS=2 if name == probe.CASES[1] else 1,
                    ),
                )
                for o, n in ((0, 4519), (4519, 4519), (9038, 4519), (13557, 2827))
            ]
            record = dict(
                case=case,
                status="passed",
                candidate_kind="query-stage-joint",
                cache_granularity=dict(production=traces),
                runtime=dict(
                    operator_binding=dict(candidate_sha256=probe.shared.OLD_KERNEL_SHA)
                ),
            )
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / (name + ".json")
                path.write_text(json.dumps(record))
                self.assertEqual(probe.load_record(folder, name)[1], traces)
                record["cache_granularity"]["production"][0]["constants"][
                    "num_stages"
                ] = 2
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(RuntimeError, "configuration changed"):
                    probe.load_record(folder, name)

    def test_actual_artifacts_report_store_widths_without_profiler_or_gpu(self):
        asm = dict(
            ptx="// st.global.v2.b32 ignored\n st.global.v4.b32 [x], {a,b,c,d};\n st.global.b32 [x], a;",
            ttgir="actual IR",
        )
        entry = types.SimpleNamespace(
            compiled={(4, 0, 80): (types.SimpleNamespace(asm=asm), dict(QUERY_PARTS=2))}
        )
        module = types.SimpleNamespace(_qsa_prefill_score_kernel=entry)
        with tempfile.TemporaryDirectory() as folder:
            records = probe.artifact_records((module, module), Path(folder), "case")
            self.assertEqual(len(records), 2)
            self.assertEqual(
                records[1]["store_ops"], {"st.global.b32": 1, "st.global.v4.b32": 1}
            )
            self.assertEqual(
                (Path(folder) / records[0]["files"]["ptx"]["path"]).read_text(),
                asm["ptx"],
            )

    def run_worker(
        self,
        root,
        *,
        fail_pipeline=False,
        case_name=None,
        skip_boundaries=False,
        query_policy="all",
    ):
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
        options = types.SimpleNamespace(
            report_dir=str(root),
            baseline_report="saved",
            rounds=2,
            kernel_warmup_ms=200,
            kernel_rep_ms=100,
        )

        options.worker_case = probe.CASES[0]

        def timer(fn):
            fn()
            return 10.0

        def pipeline(*a, **kw):
            if fail_pipeline:
                raise AssertionError("injected pipeline numerical failure")
            return 3

        with contextlib.ExitStack() as stack:
            patches = (
                patch.object(
                    probe,
                    "load_record",
                    return_value=(dict(inputs=["stable"], runtime={}), []),
                ),
                patch.object(probe, "artifact_records", return_value=[]),
                patch.object(probe, "compiler_identity", return_value={}),
                patch.object(
                    probe, "boundary_checks", return_value=[dict(status="passed")]
                ),
                patch.object(probe, "make_pair", return_value=pair),
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
                    fixed,
                    "freeze_and_records",
                    return_value=dict(baseline=[], candidate=[], configs_match=True),
                ),
                patch.object(
                    bench.scoring,
                    "make_timer",
                    return_value=(timer, dict(mode="kernel")),
                ),
                patch.object(bench.support, "gpu_guard_module", return_value=object()),
                patch.object(
                    bench.support,
                    "guarded_telemetry",
                    return_value=dict(status="passed"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            )
            for item in patches:
                stack.enter_context(item)
            code = probe.worker(options)
        return code, json.loads((root / (probe.CASES[0] + ".json")).read_text())

    def test_worker_records_all_stages_and_four_controls_with_original_timer(self):
        with tempfile.TemporaryDirectory() as folder:
            code, result = self.run_worker(Path(folder))
            self.assertEqual(code, 0, result.get("error"))
            self.assertEqual(result["status"], "passed")
            self.assertEqual(set(result["stages"]), {"score", "topk", "pipeline"})
            self.assertEqual(len(result["self_controls"]), 4)
            self.assertTrue(result["same_buffers"] and result["same_topk"])
            self.assertFalse(result["acceptance_eligible"])
            self.assertEqual(result["kernels_before"], result["kernels_after"])
            for value in result["stages"].values():
                self.assertTrue(value["measurement_validated"])
                self.assertEqual(
                    value["measurement_order"],
                    [["baseline", "candidate"], ["candidate", "baseline"]],
                )
            self.assertTrue((Path(folder) / (probe.CASES[0] + ".tsv")).is_file())

    def test_failed_pipeline_validation_stops_without_a_success_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            code, result = self.run_worker(Path(folder), fail_pipeline=True)
            self.assertEqual(code, 1)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(
                result["stages"]["pipeline"].get("measurement_validated", False)
            )
            self.assertIn("injected pipeline numerical failure", result["error"])


if __name__ == "__main__":
    unittest.main()
