# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Probe isolation, provenance rejection and same-buffer launch contracts."""

import contextlib
import copy
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_bf16_query_probe as probe
from tools import qsa_prefill_query_split_matched as fixed
from tools import qsa_prefill_stage_joint_trial as joint


class ProbeTests(unittest.TestCase):
    def record(self):
        traces = [
            dict(
                offset=o,
                rows=n,
                width=29696,
                route="rectangular",
                constants=dict(probe.CONFIG),
            )
            for o, n in ((0, 4519), (4519, 4519), (9038, 4519), (13557, 2827))
        ]
        return dict(
            status="passed",
            candidate_kind="query-stage-joint",
            cache_granularity=dict(production=traces),
            runtime=dict(operator_binding=dict(candidate_sha256=probe.OLD_KERNEL_SHA)),
        )

    def test_actual_private_source_changes_only_split_gate(self):
        facts = probe.verify_source()
        self.assertEqual(facts["baseline_sha256"], probe.OLD_KERNEL_SHA)
        self.assertNotEqual(facts["baseline_sha256"], facts["candidate_sha256"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tools").mkdir()
            old = probe.ROOT / "tools/qsa_prefill_query_page_kernel.py"
            (root / "tools/qsa_prefill_query_page_kernel.py").write_bytes(
                old.read_bytes()
            )
            candidate = root / "tools/qsa_prefill_bf16_query_probe_kernel.py"
            candidate.write_text(
                (probe.ROOT / candidate.relative_to(root))
                .read_text()
                .replace("out_dtype=tl.float32", "out_dtype=tl.float16", 1)
            )
            with patch.object(probe, "ROOT", root), self.assertRaisesRegex(
                RuntimeError, "beyond"
            ):
                probe.verify_source()

    def test_rejects_relabelled_or_reconfigured_saved_baselines(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / (probe.CASE + ".json")
            value = self.record()
            path.write_text(json.dumps(value))
            self.assertEqual(probe.baseline_record(folder)[0], value)
            variants = []
            for key, change in (("num_stages", 2), ("TILE_R", 32), ("QUERY_PARTS", 2)):
                bad = copy.deepcopy(value)
                bad["cache_granularity"]["production"][0]["constants"][key] = change
                variants.append(bad)
            bad = copy.deepcopy(value)
            bad["cache_granularity"]["production"][3]["offset"] += 1
            variants.append(bad)
            bad = copy.deepcopy(value)
            bad["runtime"]["operator_binding"]["candidate_sha256"] = "changed"
            variants.append(bad)
            bad = copy.deepcopy(value)
            bad["status"] = "failed"
            variants.append(bad)
            for bad in variants:
                path.write_text(json.dumps(bad))
                with self.assertRaises(RuntimeError):
                    probe.baseline_record(folder)

    def test_only_candidate_jit_changes_without_touching_shared_source(self):
        original, replacement = object(), object()
        topk = object()
        pair = tuple(
            types.SimpleNamespace(
                _qsa_prefill_score_kernel=types.SimpleNamespace(jit=original),
                persistent_topk=topk,
            )
            for _ in range(2)
        )
        kernel = types.ModuleType("tools.qsa_prefill_bf16_query_probe_kernel")
        kernel._qsa_prefill_score_kernel = types.SimpleNamespace(
            fn=types.SimpleNamespace(fn=replacement)
        )
        source, production, traces = (
            object(),
            object(),
            self.record()["cache_granularity"]["production"],
        )
        with patch.dict("sys.modules", {kernel.__name__: kernel}), patch.object(
            joint, "candidate_module", return_value=source
        ), patch.object(fixed, "modules", return_value=pair) as build:
            actual = probe.make_pair(production, traces)
        build.assert_called_once_with(source, source, traces, query_parts=(1, 2))
        self.assertIs(actual[0]._qsa_prefill_score_kernel.jit, original)
        self.assertIs(actual[1]._qsa_prefill_score_kernel.jit, replacement)
        self.assertIs(actual[0].persistent_topk, actual[1].persistent_topk)

    def test_pipeline_stores_feed_same_chunk_topk_and_frozen_scores_are_separate(self):
        calls = []
        score_outputs = [object(), object()]
        chunks = [
            dict(
                score_args=(i,),
                score_kwargs={},
                topk_args=(None, object(), object(), object()),
                topk_kwargs=dict(k=512),
            )
            for i in range(2)
        ]
        common = [object(), object()]

        def score(index):
            calls.append(("score", index))
            return score_outputs[index]

        def topk(*args, **kwargs):
            calls.append(("topk", args, kwargs))

        module = types.SimpleNamespace(_prefill_logits=score, persistent_topk=topk)
        bundle = probe.bind(module, chunks, common)
        bundle["pipeline"]()
        self.assertEqual([x[0] for x in calls], ["score", "topk", "score", "topk"])
        for call, scores, chunk in zip(calls[1::2], score_outputs, chunks):
            self.assertIs(call[1][0], scores)
            for a, b in zip(call[1][1:], chunk["topk_args"][1:]):
                self.assertIs(a, b)
        calls.clear()
        bundle["topk"]()
        for call, scores in zip(calls, common):
            self.assertIs(call[1][0], scores)
        self.assertNotEqual(common, score_outputs)

    def test_formatter_keeps_topk_and_does_not_mask_partial_failure(self):
        samples = dict(median=10)
        row = dict(
            baseline_us=samples,
            candidate_us=samples,
            speedup=1,
            paired_ratios=dict(min=0.99, max=1.01),
            measurement_validated=True,
        )
        result = dict(
            stages=dict(topk=row),
            self_controls={"P1:score": dict(status="validating")},
            error="original validation failure",
        )
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            probe.print_results(result)
        for word in ("topk", "1.0000", "original validation failure", "全30组"):
            self.assertIn(word, stream.getvalue())

    def run_worker(self, root, *, fail_pipeline=False):
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
                    "baseline_record",
                    return_value=(dict(inputs=["stable"], runtime={}), []),
                ),
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
        return code, json.loads((root / "probe.json").read_text())

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
            self.assertTrue((Path(folder) / "probe-timings.tsv").is_file())

    def test_failed_validation_stops_probe_and_preserves_error(self):
        with tempfile.TemporaryDirectory() as folder:
            code, result = self.run_worker(Path(folder), fail_pipeline=True)
            self.assertEqual(code, 1)
            self.assertEqual(result["status"], "failed")
            self.assertIn("injected pipeline numerical failure", result["error"])
            self.assertFalse(
                result["stages"]["pipeline"].get("measurement_validated", False)
            )
            self.assertEqual(result["self_controls"], {})
            self.assertFalse((Path(folder) / "probe-timings.tsv").exists())

    def test_compiler_identity_ignores_paths_but_rejects_binary_change(self):
        info = dict(
            torch_version="2.10",
            triton_version="3.6",
            versions=dict(vllm="0.31"),
            compiler_files=dict(c=dict(path="old", sha256="compiler")),
            flagtree_runtime=dict(
                files=dict(t=dict(path="old", sha256="triton")),
                ptxas=dict(path="old", sha256="assembler"),
                cuda=dict(libcuda="old", libcuda_sha256="driver"),
            ),
        )
        other = copy.deepcopy(info)
        other["flagtree_runtime"]["cuda"]["libcuda"] = "new-extracted-path"
        self.assertEqual(probe.compiler_identity(info), probe.compiler_identity(other))
        other["flagtree_runtime"]["files"]["t"]["sha256"] = "different"
        self.assertNotEqual(
            probe.compiler_identity(info), probe.compiler_identity(other)
        )


if __name__ == "__main__":
    unittest.main()
