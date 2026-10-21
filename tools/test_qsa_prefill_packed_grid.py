# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU contracts and execution of the actual Triton scheduler's scalar prefix."""

import argparse
import ast
import copy
import hashlib
import json
import random
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_packed_grid as trial
from tools import test_qsa_prefill_tuning_round1 as tuning_tests

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
KERNEL = ROOT / "tools/qsa_prefill_packed_grid_kernel.py"


def function(tree, name):
    return next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


class Pointer:
    def __init__(self, values, index=0):
        self.values, self.index = values, index

    def __add__(self, offset):
        return Pointer(self.values, self.index + offset)


class PackedTests(unittest.TestCase):
    def test_diagnostics_count_query_slots_without_claiming_flops(self):
        args = [
            types.SimpleNamespace(shape=(160, 4, 128)),
            None,
            types.SimpleNamespace(shape=(4, 6)),
            None,
            None,
            None,
            None,
            80,
        ]
        trace = dict(
            offset=0,
            rows=160,
            width=1088,
            route="packed",
            constants=dict(TILE_R=32, BLOCK_N=64, K_TILES=4),
        )
        result = trial.diagnostics(args, [trace], [80, 80, 0, 0])[0]
        self.assertEqual(
            (
                result["rectangular_query_slots"],
                result["packed_query_slots"],
                result["useful_query_slots"],
                result["empty_query_slots"],
                result["issued_programs"],
            ),
            (12, 9, 6, 3, 45),
        )
        with self.assertRaisesRegex(RuntimeError, "fixture geometry"):
            trial.diagnostics(args, [trace], [80, 80])

    def test_only_scheduler_and_launcher_routing_differ(self):
        self.assertEqual(
            hashlib.sha256(SOURCE.read_bytes()).hexdigest(), trial.PRODUCTION_SHA256
        )
        production, candidate = ast.parse(SOURCE.read_text()), ast.parse(
            KERNEL.read_text()
        )
        old, new = function(production, "_qsa_prefill_score_kernel"), function(
            candidate, "_qsa_prefill_packed_score_kernel"
        )
        self.assertEqual(ast.dump(old.args), ast.dump(new.args))
        self.assertEqual(
            [ast.dump(n) for n in old.decorator_list],
            [ast.dump(n) for n in new.decorator_list],
        )

        def arithmetic(node):
            index = next(
                i
                for i, n in enumerate(node.body)
                if isinstance(n, ast.AnnAssign) and n.target.id == "HEADS_PADDED"
            )
            return [ast.dump(n) for n in node.body[index:]]

        self.assertEqual(arithmetic(old), arithmetic(new))
        oldh, newh = function(production, "_prefill_logits"), function(
            candidate, "_prefill_logits"
        )
        # Remove just the extra elif; every other host statement must match.
        chain = next(
            n
            for n in newh.body
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Name)
            and n.test.id == "compact"
        )
        self.assertEqual(chain.orelse[0].test.func.id, "_packed_eligible")
        chain.orelse = chain.orelse[0].orelse
        self.assertEqual(ast.dump(oldh), ast.dump(newh))
        groups = yaml.safe_load(
            (
                ROOT / "src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml"
            ).read_text()
        )
        configs = groups["qsa_select_paged_prefill_tuning_round1_score"]
        self.assertEqual(len(configs), 10)
        self.assertEqual(max(c["META"]["TILE_R"] for c in configs), trial.MAX_TILE_R)

    def test_actual_scalar_mapping_covers_every_request_tile_once(self):
        node = function(
            ast.parse(KERNEL.read_text()), "_qsa_prefill_packed_score_kernel"
        )
        end = next(
            i
            for i, n in enumerate(node.body)
            if isinstance(n, ast.AnnAssign) and n.target.id == "HEADS_PADDED"
        )
        prefix = copy.deepcopy(node.body[:end]) + [
            ast.parse("return request,row_start,request_end").body[0]
        ]
        args = ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg=n)
                for n in (
                    "starts_ptr",
                    "query_offset",
                    "num_rows",
                    "num_requests",
                    "TILE_R",
                    "tl",
                )
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )
        fun = ast.FunctionDef(name="mapping", args=args, body=prefix, decorator_list=[])
        env = {}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[fun], type_ignores=[])),
                str(KERNEL),
                "exec",
            ),
            env,
        )
        definitions = []
        for case in bench.workloads.cases("combined-prefill"):
            g = bench.workloads.geometry(case)
            offset = 0
            for rows in g["expected_chunk_rows"]:
                definitions.append((g["query_lengths"], offset, rows))
                offset += rows
        rng = random.Random(20261021)
        for _ in range(100):
            lengths = [rng.randrange(0, 80) for _ in range(rng.randrange(1, 9))]
            if not sum(lengths):
                continue
            offset = rng.randrange(sum(lengths))
            rows = rng.randrange(1, sum(lengths) - offset + 1)
            definitions.append((lengths, offset, rows))
        for lengths, offset, rows in definitions:
            starts = [97]
            for n in lengths:
                starts.append(starts[-1] + n)
            for tile in (4, 8, 16, 32, 64):
                expected = []
                for req, (begin, end) in enumerate(zip(starts, starts[1:])):
                    begin, end = max(begin - 97, offset), min(end - 97, offset + rows)
                    expected += [(req, start, end) for start in range(begin, end, tile)]
                actual = []
                for slot in range(rows // tile + len(lengths)):
                    tl = types.SimpleNamespace(
                        program_id=lambda axis: slot,
                        load=lambda ptr: ptr.values[ptr.index],
                        minimum=min,
                        maximum=max,
                    )
                    result = env["mapping"](
                        Pointer(starts), offset, rows, len(lengths), tile, tl
                    )
                    if result is not None:
                        actual.append(result)
                self.assertEqual(
                    sorted(actual), expected, (lengths, offset, rows, tile)
                )

    def test_policy_selects_only_profitable_metadata_cases(self):
        self.assertFalse(trial.eligible(1, 16384, 4519))
        self.assertFalse(trial.eligible(2, 16000, 64))
        self.assertTrue(trial.eligible(4, 4096, 5041))
        self.assertTrue(trial.eligible(2, 10375, 73))
        self.assertFalse(trial.eligible(4, 4096, 16384))
        self.assertFalse(trial.eligible(0, 0, 0))
        selected = []
        gate = function(ast.parse(SOURCE.read_text()), "_compact_eligible")
        env = {}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[gate], type_ignores=[])),
                str(SOURCE),
                "exec",
            ),
            env,
        )
        for case in bench.workloads.cases("scored-prefill"):
            g = bench.workloads.geometry(case)
            n = len(g["query_lengths"])
            maximum = max(g["query_lengths"])
            if any(
                not env["_compact_eligible"](sum(g["query_lengths"]), n, maximum, r)
                and trial.eligible(n, maximum, r)
                for r in g["expected_chunk_rows"]
            ):
                selected.append(case["name"])
        self.assertEqual(len(selected), 6)

    def test_trial_factory_preserves_public_host_topk_and_cache_alignment(self):
        class Entry:
            def __init__(self, fn):
                self.fn, self.kernel_cache = fn, ({},)

        production = types.ModuleType("production")
        production.__file__ = str(SOURCE)
        production.persistent_topk = object()
        for name in ("_qsa_prefill_score_kernel", "_qsa_prefill_compact_score_kernel"):
            setattr(
                production,
                name,
                Entry(
                    types.SimpleNamespace(configs=[1], keys=["num_rows"], fn=object())
                ),
            )
        exec(
            "def qsa_select_paged_prefill():\n return _prefill_logits(),persistent_topk\n",
            vars(production),
        )
        kernel = types.ModuleType("tools.qsa_prefill_packed_grid_kernel")
        kernel.__file__ = str(KERNEL)
        kernel._qsa_prefill_packed_score_kernel = Entry(
            types.SimpleNamespace(configs=[1], keys=["num_rows"], fn=object())
        )
        exec(
            "def _prefill_logits():\n return _qsa_prefill_packed_score_kernel,_packed_eligible\n",
            vars(kernel),
        )
        utils = types.ModuleType("flaggems_vllm.utils")
        utils.libtuner = lambda **kwargs: lambda jit: types.SimpleNamespace(
            fn=jit, configs=kwargs["configs"], keys=kwargs["key"]
        )
        with patch.dict(
            sys.modules,
            {
                "flaggems_vllm.utils": utils,
                "tools.qsa_prefill_packed_grid_kernel": kernel,
            },
        ):
            candidate = trial.candidate_module(production)
        self.assertIs(candidate.production_baseline, production)
        self.assertIs(
            candidate.qsa_select_paged_prefill.__code__,
            production.qsa_select_paged_prefill.__code__,
        )
        self.assertIs(
            candidate.qsa_select_paged_prefill()[1], production.persistent_topk
        )
        self.assertIs(candidate._prefill_logits()[1], trial.eligible)
        for name in ("_qsa_prefill_score_kernel", "_qsa_prefill_compact_score_kernel"):
            self.assertIs(
                type(getattr(candidate, name)), type(getattr(production, name))
            )
            self.assertIsNot(
                getattr(candidate, name).kernel_cache[0],
                getattr(production, name).kernel_cache[0],
            )
            self.assertIs(
                getattr(candidate, name).fn.fn, getattr(production, name).fn.fn
            )

    def test_experimental_acceptance_requires_baseline_pin_and_both_comparisons(self):
        s = tuning_tests.AcceptanceTests.summary()
        s["manifest"]["options"].update(
            candidate="packed-grid", compare_previous_tuning=False
        )
        for c in s["cases"]:
            c["runtime"] = {
                "operator_binding": {"canonical_sha256": trial.PRODUCTION_SHA256}
            }
            for name in ("score_vs_production", "pipeline_vs_production"):
                c["stages"][name] = copy.deepcopy(c["stages"]["pipeline"])
        r = trial.acceptance(s)
        self.assertTrue(
            r["measurement_valid"] and r["target_met"] and r["experimental"]
        )
        self.assertFalse(r["formal_production_acceptance"])
        for mutate in (
            lambda x: x["cases"][0]["stages"].pop("pipeline_vs_production"),
            lambda x: x["cases"][0]["runtime"]["operator_binding"].update(
                canonical_sha256="wrong"
            ),
            lambda x: x["manifest"]["options"].update(shared_gpu_exploratory=True),
        ):
            copy_s = copy.deepcopy(s)
            mutate(copy_s)
            r = trial.acceptance(copy_s)
            self.assertFalse(r["measurement_valid"])
            self.assertIsNone(r["target_met"])

    def test_cli_and_worker_preserve_trial_and_kernel_profile(self):
        command = [
            sys.executable,
            "-I",
            str(Path(bench.__file__)),
            "--compiler",
            "flagtree",
            "--candidate",
            "packed-grid",
            "--suite",
            "scored-prefill",
            "--list-cases",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(
            json.loads(result.stdout), bench.workloads.cases("scored-prefill")
        )
        command[command.index("flagtree")] = "stock"
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        options = argparse.Namespace(
            compiler="flagtree",
            flagtree="/opt/flagtree",
            candidate="packed-grid",
            suite="scored-prefill",
            reference_file="native.py",
            rounds=2,
            kernel_warmup_ms=200,
            kernel_rep_ms=100,
        )
        child = bench.child_command(options, Path("/tmp/report"), "--worker")
        self.assertEqual(child[child.index("--candidate") + 1], "packed-grid")
        self.assertEqual(child[child.index("--rounds") + 1], "2")


if __name__ == "__main__":
    unittest.main()
