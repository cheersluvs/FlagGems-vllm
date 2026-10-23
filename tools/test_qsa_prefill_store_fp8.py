# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""FP8 suite provenance, BF16/compact fallback and boundary-reuse contracts."""

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_store_fp8 as suite
from tools import qsa_prefill_store_probe as probe


class FP8StoreTests(unittest.TestCase):
    def test_manifest_pins_all_ten_current_cases_and_real_chunk_choices(self):
        data = suite.load_manifest()
        self.assertEqual(len(data["cases"]), 10)
        self.assertEqual(
            sum(t["route"] == "compact" for c in data["cases"] for t in c["traces"]), 1
        )
        self.assertEqual(
            suite.saved_record(suite.names()[0])[0]["compiler_identity"],
            data["compiler_identity"],
        )
        with self.assertRaises(ValueError):
            suite.saved_record("unknown")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            changed = copy.deepcopy(data)
            changed["cases"][0]["traces"][0]["rows"] -= 1
            path.write_text(json.dumps(changed))
            with patch.object(suite, "MANIFEST", path):
                with self.assertRaisesRegex(RuntimeError, "hash changed"):
                    suite.load_manifest()
                with patch.object(
                    suite, "MANIFEST_SHA", hashlib.sha256(path.read_bytes()).hexdigest()
                ):
                    with self.assertRaisesRegex(RuntimeError, "geometry changed"):
                        suite.load_manifest()

    def test_fp8_only_jit_gate_leaves_bf16_and_unknown_dtype_original(self):
        calls = []

        class JIT:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, grid):
                return lambda *a, **kw: calls.append((self.name, a, kw))

        routed = probe.StoreJIT(
            JIT("original"), JIT("bulk"), ["logits_ptr", "logits_width"], fp8_only=True
        )
        logits = types.SimpleNamespace(
            element_size=lambda: 4,
            dtype="torch.float32",
            is_contiguous=lambda: True,
            ndim=2,
            shape=(4, 80),
            data_ptr=lambda: 256,
        )
        for flag in (False, True, None):
            kwargs = {} if flag is None else dict(IS_FP8=flag)
            routed[(1,)](logits, 80, **kwargs)
            self.assertEqual(calls[-1][0], "bulk" if flag is True else "original")
            self.assertIs(calls[-1][1][0], logits)
            self.assertEqual(calls[-1][2], kwargs)

    def six_checks(self):
        return [
            dict(
                status="passed",
                dtype=dtype,
                width=width,
                shift=shift,
                all_rows_checked=15,
                bulk_active=dtype == "torch.float8_e4m3fn"
                and width == 80
                and shift == 64,
                fallback_binary_verified=not (
                    dtype == "torch.float8_e4m3fn" and width == 80 and shift == 64
                ),
            )
            for dtype in ("torch.bfloat16", "torch.float8_e4m3fn")
            for width, shift in ((80, 64), (81, 64), (80, 65))
        ]

    def test_boundaries_run_once_and_reuse_requires_exact_identity_and_fallback(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            probe, "compiler_identity", return_value={"compiler": "same"}
        ):
            options = types.SimpleNamespace(
                report_dir=folder,
                cases=["first", "later"],
                worker_case="first",
                store_policy="fp8-only",
            )
            count = []
            result = suite.boundary_evidence(
                options, {}, lambda: count.append(True) or self.six_checks()
            )
            self.assertEqual(len(result), 6)
            options.worker_case = "later"
            result = suite.boundary_evidence(
                options, {}, lambda: self.fail("Re-ran boundary suite")
            )
            self.assertEqual(len(count), 1)
            self.assertEqual(result[0]["case"], "first")
            path = Path(folder) / "store-boundary.json"
            self.assertEqual(result[0]["evidence_sha256"], probe.sha(path))
            evidence = json.loads(path.read_text())
            evidence["checks"][0]["fallback_binary_verified"] = False
            path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(RuntimeError, "fallback boundary"):
                suite.boundary_evidence(options, {}, lambda: None)

    def test_compact_and_bf16_fallback_require_original_ptx_and_cubin(self):
        tensor = types.SimpleNamespace(
            element_size=lambda: 4,
            dtype="torch.float32",
            is_contiguous=lambda: True,
            ndim=2,
            shape=(4, 80),
            data_ptr=lambda: 256,
        )
        trace = dict(rows=4, offset=0, width=80, route="rectangular")
        chunks = [dict(score_args=(None,) * 5 + (tensor,))]
        original = dict(
            chunk=[4, 0, 80], ptx_sha256="old-ptx", cubin_sha256="old-cubin"
        )
        resources = dict(baseline=[original], candidate=[copy.deepcopy(original)])
        for route, dtype, active in (
            ("rectangular", "float8_e4m3fn", True),
            ("compact", "float8_e4m3fn", False),
            ("rectangular", "bfloat16", False),
        ):
            trace["route"] = route
            rows = suite.gate_records(
                dict(dtype=dtype), [trace], chunks, resources, "fp8-only"
            )
            self.assertEqual(rows[0]["bulk_active"], active)
        trace["route"] = "compact"
        resources["candidate"][0]["cubin_sha256"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "Fallback differs"):
            suite.gate_records(
                dict(dtype="float8_e4m3fn"), [trace], chunks, resources, "fp8-only"
            )

    def test_compact_actual_artifact_export_is_included(self):
        compiled = types.SimpleNamespace(
            asm=dict(ptx="st.global.b32 [x], a;", ttgir="compact IR")
        )
        entry = types.SimpleNamespace(compiled={(4, 0, 80): (compiled, {})})
        module = types.SimpleNamespace(_qsa_prefill_compact_score_kernel=entry)
        with tempfile.TemporaryDirectory() as folder:
            records = probe.artifact_records((module, module), Path(folder), "compact")
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r["route"] == "compact" for r in records))

    def test_cli_selects_all_fp8_cases_and_rejects_cross_suite_or_ungated_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            base = [
                "probe",
                "--report-dir",
                folder,
                "--baseline-report",
                "original",
                "--suite",
                "fp8",
            ]
            with patch.object(probe.sys, "argv", base), patch.object(
                probe, "coordinate", return_value=0
            ) as run:
                self.assertEqual(probe.main(), 0)
                options = run.call_args.args[0]
                self.assertEqual(options.cases, suite.names())
                self.assertEqual(options.store_policy, "fp8-only")
            for extra in (["--store-policy", "aligned"], ["--cases", probe.CASES[0]]):
                with patch.object(
                    probe.sys, "argv", base + extra
                ), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                    SystemExit
                ):
                    probe.main()

    def test_aggregation_includes_topk_and_labels_only_completed_subset(self):
        stage = dict(
            measurement_validated=True,
            baseline_us=dict(median=10),
            candidate_us=dict(median=8),
            speedup=1.25,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            aggregate = suite.show_summary(
                [
                    dict(
                        case="a",
                        status="passed",
                        stages={s: stage for s in ("score", "topk", "pipeline")},
                    )
                ]
            )
        self.assertEqual(set(aggregate), {"score", "topk", "pipeline"})
        self.assertEqual(
            aggregate["pipeline"], dict(count=1, arithmetic_mean=1.25, minimum=1.25)
        )


if __name__ == "__main__":
    unittest.main()
