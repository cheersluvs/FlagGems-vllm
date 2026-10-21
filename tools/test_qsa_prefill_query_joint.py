# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Joint mode search, identical math, fixed-mode diagnostics and buffer controls."""

import ast
import copy
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_query_joint_policy as policy
from tools import qsa_prefill_query_joint_trial as trial
from tools import qsa_prefill_query_split_matched as matched
from tools import test_qsa_prefill_page_table as page
from tools import test_qsa_prefill_query_split as split
from tools import test_qsa_prefill_tuning_round1 as goals

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "tools/qsa_prefill_query_joint_kernel.py"


class JointTests(unittest.TestCase):
    def test_only_signature_gate_and_tuner_plumbing_change(self):
        old = split.function(split.KERNEL)
        new = split.function(KERNEL)
        self.assertEqual(new.args.args[-1].arg, "QUERY_PARTS")
        new.args.args.pop()
        old_gate = next(
            n
            for n in old.body
            if isinstance(n, ast.AnnAssign) and n.target.id == "SPLIT_QUERY"
        )
        new_gate = next(
            n
            for n in new.body
            if isinstance(n, ast.AnnAssign) and n.target.id == "SPLIT_QUERY"
        )
        self.assertEqual(
            ast.dump(new_gate.value),
            ast.dump(
                ast.parse(
                    "IS_FP8 and TILE_R >= 32 and QUERY_PARTS == 2", mode="eval"
                ).body
            ),
        )
        new_gate.value = copy.deepcopy(old_gate.value)
        new.decorator_list = copy.deepcopy(old.decorator_list)
        self.assertEqual(ast.dump(old), ast.dump(new))
        for parts in (1, 2):
            for fp8 in (True, False):
                for tile in (4, 8, 16, 32, 64):
                    condition = compile(
                        ast.parse(
                            "IS_FP8 and TILE_R >= 32 and QUERY_PARTS == 2", mode="eval"
                        ),
                        "gate",
                        "eval",
                    )
                    self.assertEqual(
                        eval(
                            condition, dict(IS_FP8=fp8, TILE_R=tile, QUERY_PARTS=parts)
                        ),
                        fp8 and tile >= 32 and parts == 2,
                    )

    def test_yaml_keeps_all_original_choices_and_prunes_bf16_duplicates(self):
        groups = yaml.safe_load(
            (
                ROOT / "src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml"
            ).read_text()
        )
        base = groups["qsa_select_paged_prefill_tuning_round1_score"]
        joint = groups[policy.CONFIG_GROUP]
        configs = [types.SimpleNamespace(kwargs=dict(c["META"])) for c in joint]
        self.assertEqual(len(joint), 17)
        full = [
            {**c, "META": {k: v for k, v in c["META"].items() if k != "QUERY_PARTS"}}
            for c in joint
            if c["META"]["QUERY_PARTS"] == 1
        ]
        halves = [
            {**c, "META": {k: v for k, v in c["META"].items() if k != "QUERY_PARTS"}}
            for c in joint
            if c["META"]["QUERY_PARTS"] == 2
        ]
        self.assertEqual(full, base)
        self.assertEqual(halves, [c for c in base if c["META"]["TILE_R"] >= 32])
        self.assertEqual(len(policy.prune_configs(configs, {"IS_FP8": True})), 17)
        pruned = policy.prune_configs(configs, {}, IS_FP8=False)
        self.assertEqual(len(pruned), 10)
        self.assertTrue(all(c.kwargs["QUERY_PARTS"] == 1 for c in pruned))
        for fp8 in (False, True):
            traces = [
                dict(route="rectangular", constants=c.kwargs)
                for c in policy.prune_configs(configs, {"IS_FP8": fp8})
            ]
            policy.validate_choices(traces, fp8)
        with self.assertRaisesRegex(RuntimeError, "ineligible"):
            policy.validate_choices(
                [dict(route="rectangular", constants=dict(QUERY_PARTS=2, TILE_R=32))],
                False,
            )

    def test_control_config_label_ignores_only_query_mode(self):
        left = [
            dict(
                rows=64,
                offset=0,
                width=1024,
                route="rectangular",
                constants=dict(TILE_R=8, num_warps=4),
            )
        ]
        right = copy.deepcopy(left)
        right[0]["constants"]["QUERY_PARTS"] = 1
        self.assertTrue(policy.same_tiles(left, right))
        right[0]["constants"]["num_warps"] = 8
        self.assertFalse(policy.same_tiles(left, right))

    def test_factory_keeps_host_topk_and_independent_caches(self):
        test = page.PageTableTests()
        test.kernel = KERNEL
        adapter = types.SimpleNamespace(
            candidate_module=lambda p, **kw: trial.candidate_module(p)
        )
        with patch.object(page, "trial", adapter):
            test.test_factory_preserves_host_topk_and_isolates_aligned_entry_classes()

    def test_fixed_modes_match_every_other_config_and_freeze(self):
        kernel = types.SimpleNamespace(
            asm={"ptx": "", "cubin": b""}, metadata=types.SimpleNamespace()
        )
        calls = []

        class Jit:
            def __getitem__(self, grid):
                def launch(*a, **kw):
                    calls.append(kw)
                    return kernel

                return launch

        entry = types.SimpleNamespace(
            arg_names=["num_rows", "query_offset", "logits_width"],
            fn=types.SimpleNamespace(fn=Jit()),
        )
        traces = [
            dict(
                rows=64,
                offset=0,
                width=1024,
                route="rectangular",
                constants=dict(TILE_R=64, QUERY_PARTS=1),
            ),
            dict(
                rows=3,
                offset=64,
                width=1024,
                route="rectangular",
                constants=dict(TILE_R=8, QUERY_PARTS=1),
            ),
        ]
        pair = []
        for parts in (1, 2):
            e = matched.FixedEntry(entry, traces, "rectangular", query_parts=parts)
            for t in traces:
                e[lambda c: (c["TILE_R"],)](t["rows"], t["offset"], t["width"])
            pair.append(
                types.SimpleNamespace(
                    _qsa_prefill_score_kernel=e,
                    _qsa_prefill_compact_score_kernel=matched.FixedEntry(
                        entry, traces, "compact"
                    ),
                )
            )
        self.assertEqual([c["QUERY_PARTS"] for c in calls], [1, 1, 2, 1])
        records = matched.freeze_and_records(pair, query_parts=(1, 2))
        self.assertTrue(records["configs_match"])
        self.assertEqual(records["query_parts_pair"], [1, 2])
        with self.assertRaisesRegex(RuntimeError, "requested"):
            matched.freeze_and_records(pair, query_parts=(1, 1))

    def test_joint_acceptance_requires_fixed_comparison_and_control(self):
        s = goals.AcceptanceTests.summary()
        s["manifest"]["options"].update(
            candidate="query-joint", compare_previous_tuning=False
        )
        for c in s["cases"]:
            c["runtime"] = dict(
                operator_binding=dict(canonical_sha256=trial.PRODUCTION_SHA256)
            )
            for k in ("score_vs_production", "pipeline_vs_production"):
                c["stages"][k] = copy.deepcopy(c["stages"]["pipeline"])
            if c["case"]["name"] == "observed_short_batch3_bfloat16":
                c["stages"]["pipeline_buffer_swap"] = copy.deepcopy(
                    c["stages"]["pipeline"]
                )
                c["stages"]["pipeline_buffer_swap"].update(
                    baseline_kind="production_swapped_buffers", includes_topk=True
                )
                c["bf16_control"] = dict(same_topk=True)
        fp = next(c for c in s["cases"] if c["case"]["dtype"] == "float8_e4m3fn")
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
        self.assertTrue(trial.acceptance(s)["measurement_valid"])
        fp["query_joint"]["matched"]["query_parts_pair"] = [1, 1]
        self.assertFalse(trial.acceptance(s)["measurement_valid"])
        fp["query_joint"]["matched"]["query_parts_pair"] = [1, 2]
        control = next(
            c
            for c in s["cases"]
            if c["case"]["name"] == "observed_short_batch3_bfloat16"
        )
        control["stages"].pop("pipeline_buffer_swap")
        self.assertFalse(trial.acceptance(s)["measurement_valid"])

    def test_buffer_swap_replays_correct_owner_and_validation_outputs(self):
        calls = []

        def module(label, topk_name):
            obj = types.SimpleNamespace(_prefill_logits=lambda *a, **kw: (label, a[5]))
            setattr(
                obj,
                topk_name,
                lambda logits, *a, **kw: calls.append((label, logits, a[1])),
            )
            return obj

        native = module("native", "_topk")
        production = module("production", "persistent_topk")
        candidate = module("joint", "persistent_topk")

        def chunk(label):
            return dict(
                offset=0,
                rows=3,
                width=4,
                score_args=(None,) * 5 + (label,),
                score_kwargs={},
                topk_args=(None, None, label + "_out", None),
                topk_kwargs={},
                common_logits=None,
            )

        n, p, c = [chunk(x) for x in ("native", "production", "candidate")]
        functions = trial.swapped_pipeline_functions(
            native, production, candidate, [n], [p], [c]
        )
        functions["baseline"]()
        functions["candidate"]()
        self.assertEqual(
            calls,
            [
                ("production", ("production", "candidate"), "candidate_out"),
                ("joint", ("joint", "production"), "production_out"),
            ],
        )
        worker = next(
            n
            for n in ast.parse(Path(bench.__file__).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "worker"
        )
        validate = next(
            n
            for n in worker.body
            if isinstance(n, ast.FunctionDef) and n.name == "validate"
        )
        seen = []
        ns = dict(
            bench.__dict__,
            functions={"pipeline_buffer_swap": functions},
            args=("candidate_output",),
            production_args=("production_output",),
            native_chunks=[],
            torch=None,
            helpers=None,
            validate_pipeline=lambda *a: seen.append(a) or 1,
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[validate], type_ignores=[])),
                "validate",
                "exec",
            ),
            ns,
        )
        ns["validate"]("pipeline_buffer_swap")
        self.assertEqual(seen[0][1:3], (("candidate_output",), ("production_output",)))


if __name__ == "__main__":
    unittest.main()
