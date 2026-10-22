# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Composition proofs, real page-mask emulation, isolated host and report gates."""

import ast
import copy
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_accuracy_profiles as profiles
from tools import qsa_prefill_query_page_trial as trial
from tools import test_qsa_prefill_page_table as page
from tools import test_qsa_prefill_tuning_round1 as goals

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "tools/qsa_prefill_query_page_kernel.py"


def score(path):
    return page.score(ast.parse(path.read_text()))


def page_branch(fn):
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    return loop, next(n for n in loop.body if isinstance(n, ast.If))


class CombinedTests(unittest.TestCase):
    def test_only_joint_change_is_exact_retained_bf16_page_branch(self):
        old = score(ROOT / "tools/qsa_prefill_query_joint_kernel.py")
        new = score(KERNEL)
        loop, branch = page_branch(new)
        _, retained = page_branch(
            score(ROOT / "tools/qsa_prefill_page_table_bf16_kernel.py")
        )
        self.assertEqual(ast.dump(branch), ast.dump(retained))
        original_loop = next(n for n in old.body if isinstance(n, ast.For))
        vector = next(
            n
            for n in original_loop.body
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "physical_page"
        )
        self.assertEqual(ast.dump(branch.orelse[0]), ast.dump(vector))
        loop.body[loop.body.index(branch)] = copy.deepcopy(vector)
        self.assertEqual(ast.dump(old), ast.dump(new))

    def test_bf16_specialization_recovers_exact_page_candidate(self):
        original = score(ROOT / "tools/qsa_prefill_page_table_bf16_kernel.py")
        new = score(KERNEL)
        self.assertEqual(new.args.args.pop().arg, "QUERY_PARTS")
        new.decorator_list = copy.deepcopy(original.decorator_list)

        class FullQuery(ast.NodeTransformer):
            def visit_AnnAssign(self, node):
                return None if node.target.id == "SPLIT_QUERY" else node

            def visit_If(self, node):
                if isinstance(node.test, ast.Name) and node.test.id == "SPLIT_QUERY":
                    return [self.visit(n) for n in node.orelse]
                return self.generic_visit(node)

        new = FullQuery().visit(new)
        self.assertEqual(ast.dump(original), ast.dump(new))

    def test_actual_bf16_and_fp8_page_address_masks(self):
        for fp8 in (False, True):
            case = page.PageTableTests()
            case.kernel = KERNEL
            case.fp8 = fp8
            case.test_actual_load_branch_matches_vector_lookup_at_all_live_columns()

    def test_factory_reuses_host_bytecode_topk_and_has_independent_caches(self):
        case = page.PageTableTests()
        case.kernel = KERNEL
        adapter = types.SimpleNamespace(
            candidate_module=lambda p, **kw: trial.candidate_module(p)
        )
        with patch.object(page, "trial", adapter):
            case.test_factory_preserves_host_topk_and_isolates_aligned_entry_classes()
        kernel = ast.parse(KERNEL.read_text())
        self.assertTrue(
            any(
                isinstance(n, ast.keyword) and n.arg == "prune_configs_by"
                for n in ast.walk(kernel)
            )
        )

    @staticmethod
    def summary():
        summary = goals.AcceptanceTests.summary()
        summary["manifest"]["options"].update(
            candidate="query-joint-page", compare_previous_tuning=False
        )
        for c in summary["cases"]:
            c["runtime"] = dict(
                operator_binding=dict(canonical_sha256=trial.PRODUCTION_SHA256)
            )
            for key in ("score_vs_production", "pipeline_vs_production"):
                c["stages"][key] = copy.deepcopy(c["stages"]["pipeline"])
            c["production_score_configs"] = [dict(TILE_R=8, QUERY_PARTS=1)]
            c["query_page"] = dict(
                baseline_source_sha256=trial.PRODUCTION_SHA256,
                page_gate="not IS_FP8 and BLOCK_N<=PAGE_SIZE",
                configs=copy.deepcopy(c["production_score_configs"]),
                same_topk=True,
            )
            if c["case"]["name"] == "observed_short_batch3_bfloat16":
                c["stages"]["pipeline_buffer_swap"] = copy.deepcopy(
                    c["stages"]["pipeline"]
                )
                c["stages"]["pipeline_buffer_swap"].update(
                    baseline_kind="production_swapped_buffers", includes_topk=True
                )
                c["bf16_control"] = dict(same_topk=True)
        fp = next(c for c in summary["cases"] if c["case"]["dtype"] == "float8_e4m3fn")
        fp["cache_granularity"] = dict(
            production=[dict(route="rectangular", constants=dict(TILE_R=64))]
        )
        fp["query_joint"] = dict(
            matched=dict(configs_match=True, query_parts_pair=[1, 2])
        )
        fp["stages"]["score_matched"] = copy.deepcopy(fp["stages"]["score"])
        fp["stages"]["score_matched"].update(
            baseline_kind="joint_tile_full_query", includes_topk=False
        )
        return summary

    def test_acceptance_preserves_joint_and_production_contract(self):
        s = self.summary()
        snapshot = copy.deepcopy(s)
        result = trial.acceptance(s)
        self.assertEqual(s, snapshot)
        self.assertTrue(result["measurement_valid"] and result["target_met"])
        self.assertTrue(result["experimental"])
        self.assertEqual(result["candidate"], "query-joint-page")
        self.assertFalse(result["formal_production_acceptance"])
        for key in ("page_gate", "same_topk", "configs", "baseline_source_sha256"):
            bad = copy.deepcopy(s)
            bad["cases"][0]["query_page"].pop(key)
            self.assertFalse(trial.acceptance(bad)["measurement_valid"])
        for stage in (
            "score_vs_production",
            "pipeline_vs_production",
            "pipeline_buffer_swap",
        ):
            bad = copy.deepcopy(s)
            target = next(c for c in bad["cases"] if stage in c["stages"])
            target["stages"].pop(stage)
            self.assertFalse(trial.acceptance(bad)["measurement_valid"])
        bad = copy.deepcopy(s)
        bad["manifest"]["options"]["candidate"] = "query-joint"
        self.assertFalse(trial.acceptance(bad)["measurement_valid"])
        fp = next(c for c in s["cases"] if c["case"]["dtype"] == "float8_e4m3fn")
        fp["stages"].pop("score_matched")
        self.assertFalse(trial.acceptance(s)["measurement_valid"])

    def test_full_candidate_coverage_rejects_missing_skipped_or_partial_families(self):
        prefix = profiles.QUERY_PAGE_FILE + "::"
        records = {
            prefix + name: dict(result="passed")
            for name in (*profiles.QUERY_SPLIT_CASES, *profiles.PAGE_TABLE_CASES)
        }
        for name, count in (
            ("test_combined_rectangular_every_config", 8),
            ("test_combined_compact_every_config", 6),
            ("test_combined_full_entry", 12),
        ):
            records.update(
                {prefix + name + f"[{i}]": dict(result="passed") for i in range(count)}
            )
        self.assertEqual(len(records), 40)
        self.assertIsNone(trial.validate_full_coverage(records))
        absolute = {str(ROOT / key): value for key, value in records.items()}
        self.assertIsNone(trial.validate_full_coverage(absolute))
        for key in records:
            bad = copy.deepcopy(records)
            bad.pop(key)
            self.assertIsNotNone(trial.validate_full_coverage(bad))
            bad = copy.deepcopy(records)
            bad[key]["result"] = "skipped"
            self.assertIsNotNone(trial.validate_full_coverage(bad))
        bad = dict(records, **{str(ROOT / next(iter(records))): dict(result="passed")})
        self.assertIsNotNone(trial.validate_full_coverage(bad))

    def test_combined_cli_routes_full_file_and_frozen30(self):
        options = types.SimpleNamespace(
            candidate="query-joint-page", accuracy_profile="full"
        )
        self.assertEqual(
            profiles.pytest_targets(options, ROOT),
            [
                str(ROOT / profiles.QUERY_PAGE_FILE),
                str(ROOT / profiles.PRODUCTION_FILE),
            ],
        )
        self.assertEqual(len(profiles.expected_nodes("query-joint-page")), 54)
        script = (ROOT / "tools/run_qsa_prefill_query_page_h800.sh").read_text()
        self.assertIn("--accuracy-profile full", script)
        self.assertIn("--rounds 2", script)
        self.assertNotIn("--cases", script)
        self.assertNotIn("--query-control-cycles", script)
        runtime = next(
            n
            for n in ast.parse(Path(bench.__file__).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "runtime"
        )
        self.assertIn("query_page.candidate_module(op)", ast.unparse(runtime))
        worker = next(
            n
            for n in ast.parse(Path(bench.__file__).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "worker"
        )
        self.assertIn("combined_trial or", ast.unparse(worker))


if __name__ == "__main__":
    unittest.main()
