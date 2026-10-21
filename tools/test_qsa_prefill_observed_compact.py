# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU workload contracts and execution of the actual compact mapping AST."""

import ast
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_qsa_prefill_chunk_setup as chunk

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_observed_workloads as workloads
from tools import qsa_prefill_triton_ab as ab

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "tools/qsa_prefill_compact_trial.py"
PRODUCTION = ROOT / "tools/qsa_prefill_frozen_production.py"


def mapping():
    tree = ast.parse(TRIAL.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_qsa_prefill_compact_score_kernel"
    )
    fn.decorator_list = []
    fn.args = (
        ast.parse(
            "def f(starts_ptr, num_requests, query_offset, num_rows, TILE_R): pass"
        )
        .body[0]
        .args
    )
    stop = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.AnnAssign))
    fn.body = fn.body[:stop] + ast.parse("return request, row_start, request_end").body
    ns = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
            str(TRIAL),
            "exec",
        ),
        ns,
    )
    return ns, ns[fn.name]


class Pointer:
    def __init__(self, values, index=0):
        self.values, self.index = values, index

    def __add__(self, index):
        return Pointer(self.values, self.index + index)


class WorkloadTests(unittest.TestCase):
    def test_matrix_and_recorded_envelopes(self):
        self.assertEqual(
            [len(workloads.cases(s)) for s in workloads.SUITES], [12, 10, 10, 32, 30]
        )
        expected = [
            [96, 1, 4, 128, 96, 4096],
            [384, 1, 4, 128, 384, 16384],
            [368, 1, 4, 128, 368, 32781],
            [288, 3, 4, 128, 96, 4101],
            [8896, 54, 4, 128, 3200, 4100],
            [16000, 2, 4, 128, 12400, 32769],
            [16368, 2, 4, 128, 16000, 32768],
            [16368, 2, 4, 128, 16000, 32770],
            [800, 1, 4, 128, 800, 800],
            [14400, 18, 4, 128, 800, 1025],
        ]
        cases = workloads.observed_cases()
        self.assertEqual([c["recorded_shape"] for c in cases], expected)
        for c in cases:
            info = workloads.geometry(c)
            q = info["query_lengths"]
            self.assertEqual(
                [sum(q), len(q), 4, 128, max(q), info["batch_max_seq_len"]],
                c["recorded_shape"],
            )
            self.assertEqual(info["query_start_loc"][-1], sum(q))
            self.assertLessEqual(info["score_workspace_bytes"], 512 * 1024**2)
            self.assertEqual(
                info["sequence_lengths"], [c["batch_max_seq_len"]] * len(q)
            )
        self.assertEqual(sum(workloads.geometry(c)["select_all"] for c in cases), 2)
        self.assertEqual(workloads.geometry(cases[6])["expected_chunk_rows"], [16368])
        self.assertEqual(
            workloads.geometry(cases[7])["expected_chunk_rows"], [16304, 64]
        )

    def test_exact_causal_metadata_does_not_round_recorded_bound(self):
        case = workloads.observed_cases()[7]
        recorded = {}

        def build(q, c, **kw):
            recorded.update(q=q, compressed=c, **kw)
            return tuple(range(10))

        args = workloads.make_inputs(
            case,
            types.SimpleNamespace(make_inputs=build),
            types.SimpleNamespace(bfloat16="bf16"),
        )
        self.assertEqual(args[9], 32770)
        self.assertEqual(recorded["compressed"], [8193, 8193])
        self.assertEqual(recorded["visible"][0], (32770 - 16000 + 1) // 4)
        self.assertEqual(recorded["visible"][16000], (32770 - 368 + 1) // 4)
        self.assertEqual(recorded["visible"][-1], 32770 // 4)

    def test_isolated_cli_lists_all_suites_without_gpu(self):
        for suite, count in zip(workloads.SUITES, (12, 10, 10, 32, 30)):
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(Path(bench.__file__)),
                    "--suite",
                    suite,
                    "--list-cases",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(len(json.loads(result.stdout)), count)


class CompactMappingTests(unittest.TestCase):
    def test_sparse_gate_selects_only_low_utilization_envelopes(self):
        fn = next(
            n
            for n in ast.parse(TRIAL.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "_compact_eligible"
        )
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(TRIAL), "exec"), ns)
        eligible = ns["_compact_eligible"]
        selected = []
        for case in workloads.upstream.cases() + workloads.observed_cases():
            info = workloads.geometry(case)
            for count in info["expected_chunk_rows"]:
                if eligible(
                    info["q_shape"][0],
                    info["page_table_shape"][0],
                    max(info["query_lengths"]),
                    count,
                ):
                    selected.append(case["name"])
        self.assertEqual(selected, ["observed_ragged54_bfloat16"])
        self.assertFalse(eligible(200, 8, 99, 200))
        self.assertTrue(eligible(200, 8, 100, 200))  # Exactly fourfold capacity.
        self.assertFalse(eligible(800, 8, 100, 800))  # Uniform requests.
        self.assertFalse(eligible(200, 8, 100, 64))  # Small chunk.
        self.assertFalse(eligible(0, 0, 0, 0))

    def test_private_dispatch_preserves_fallbacks_workspaces_and_topk(self):
        for rows, requests, max_query in (
            (1, 2, 39),
            (65, 2, 39),
            (73, 2, 39),
            (78, 2, 39),
            (8896, 54, 3200),
            (200, 5, 160),
            (200, 8, 99),
            (200, 8, 100),
        ):
            for budget in ("512", "1", "0"):
                if rows > 256 and budget != "512":
                    continue
                original, dot, topks, allocations = chunk.load(PRODUCTION)
                op = types.ModuleType("production")
                op.__dict__.update(original)
                op.__file__ = str(PRODUCTION)
                op._qsa_prefill_score_kernel.kernel_cache = ({},)
                op._qsa_prefill_score_kernel._cpu_cache = {}
                compact = []

                class Kernel:
                    def __getitem__(self, grid):
                        def invoke(*a, **kw):
                            compact.append(
                                (
                                    grid(dict(TILE_R=16, BLOCK_N=64, K_TILES=4)),
                                    tuple(map(chunk.normalize, a)),
                                    kw,
                                )
                            )

                        return invoke

                ns = dict(
                    copy=copy,
                    hashlib=hashlib,
                    types=types,
                    Path=Path,
                    PRODUCTION_SHA256=ab.PRODUCTION_SHA256,
                    _qsa_prefill_compact_score_kernel=Kernel(),
                )
                functions = [
                    n
                    for n in ast.parse(TRIAL.read_text()).body
                    if isinstance(n, ast.FunctionDef)
                    and n.name
                    in (
                        "_prefill_logits",
                        "candidate_module",
                        "isolated_score_entry",
                        "_compact_eligible",
                    )
                ]
                exec(
                    compile(
                        ast.Module(body=functions, type_ignores=[]), str(TRIAL), "exec"
                    ),
                    ns,
                )
                candidate = ns["candidate_module"](op)
                self.assertIs(
                    candidate.qsa_select_paged_prefill.__code__,
                    op.qsa_select_paged_prefill.__code__,
                )
                self.assertIsNot(
                    candidate._qsa_prefill_score_kernel, op._qsa_prefill_score_kernel
                )
                self.assertIsNot(
                    candidate._qsa_prefill_score_kernel.kernel_cache[0],
                    op._qsa_prefill_score_kernel.kernel_cache[0],
                )
                self.assertIs(
                    candidate._production_prefill_logits.__code__,
                    op._prefill_logits.__code__,
                )
                for name in (
                    "_validate_inputs",
                    "_prefill_score_metadata",
                    "persistent_topk",
                    "_select_all_kernel",
                ):
                    self.assertIs(getattr(candidate, name), getattr(op, name))
                args = chunk.inputs(rows=rows, padded=True)
                args[2] = chunk.Array(
                    (requests, 512), "i32", 30000000, (2048, 2), label="input_2"
                )
                args[3] = chunk.Array((requests + 1,), "i32", 40000000, label="input_3")
                args[7] = max_query
                with patch.dict(os.environ, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=budget):
                    op.qsa_select_paged_prefill(*args)
                    old_calls = list(dot)
                    old_topks = [
                        (tuple(map(chunk.normalize, a)), kw) for a, kw in topks
                    ]
                    old_bytes = [v.numel() * v.element_size() for v in allocations]
                    dot.clear()
                    topks.clear()
                    allocations.clear()
                    candidate.qsa_select_paged_prefill(*args)
                eligible = [
                    c
                    for c in old_calls
                    if rows < requests * max_query
                    and c[1][12] > max_query
                    and requests * max_query >= 4 * c[1][12]
                ]
                self.assertEqual(dot, [c for c in old_calls if c not in eligible])
                self.assertEqual([c[1:] for c in compact], [c[1:] for c in eligible])
                self.assertEqual(
                    [(tuple(map(chunk.normalize, a)), kw) for a, kw in topks], old_topks
                )
                self.assertEqual(
                    [v.numel() * v.element_size() for v in allocations], old_bytes
                )
                self.assertIs(op._prefill_logits, original["_prefill_logits"])

    def test_real_libentry_key_reproduces_tail_priming_and_private_cache_prevents_it(
        self,
    ):
        source = ast.parse((ROOT / "src/flaggems_vllm/utils/libentry.py").read_text())
        entry = next(
            n
            for n in source.body
            if isinstance(n, ast.ClassDef) and n.name == "LibEntry"
        )
        key = next(
            n for n in entry.body if isinstance(n, ast.FunctionDef) and n.name == "key"
        )
        functions = [
            n
            for n in ast.parse(TRIAL.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "isolated_score_entry"
        ]
        ns = dict(copy=copy)
        exec(
            compile(
                ast.Module(body=[key, *functions], type_ignores=[]), str(TRIAL), "exec"
            ),
            ns,
        )

        class Entry:
            def __init__(self):
                self.kernel_cache, self._cpu_cache = ({},), {}

            def run(self, rows, offset):
                # Other production arguments are identical in these chunks.
                k = ns["key"](self, [], [rows, offset, 16000], [8232, 2])
                return self.kernel_cache[0].setdefault(k, 8 if rows == 64 else 64)

        polluted = Entry()
        self.assertEqual(polluted.run(64, 16304), 8)
        self.assertEqual(polluted.run(16304, 0), 8)  # Actual outer key aliases rows.
        production = Entry()
        fallback = ns["isolated_score_entry"](production)
        self.assertEqual(fallback.run(64, 16304), 8)
        self.assertFalse(production.kernel_cache[0])
        self.assertEqual(production.run(16304, 0), 64)
        self.assertEqual(fallback.run(64, 16304), 8)

    def test_every_row_has_exactly_one_writer_for_empty_ragged_and_cut_requests(self):
        ns, fn = mapping()
        rng = random.Random(20261020)
        samples = [(0, 7, 0, 33, 1, 65, 0), (16000, 368), (0, 1, 0)]
        samples += [
            tuple(rng.randrange(0, 100) for _ in range(rng.randrange(1, 30)))
            for _ in range(60)
        ]
        for lengths in samples:
            total = sum(lengths)
            if not total:
                continue
            starts = [97]
            for q in lengths:
                starts.append(starts[-1] + q)
            cuts = {(0, total), (total - 1, 1)}
            for _ in range(4):
                offset = rng.randrange(total)
                cuts.add((offset, rng.randrange(1, total - offset + 1)))
            for offset, count in cuts:
                for tile in (4, 8, 16, 32, 64):
                    written = []
                    for slot in range(count // tile + len(lengths)):
                        ns["tl"] = types.SimpleNamespace(
                            program_id=lambda axis: slot,
                            load=lambda ptr: ptr.values[ptr.index],
                            minimum=min,
                            maximum=max,
                        )
                        item = fn(Pointer(starts), len(lengths), offset, count, tile)
                        if item is not None:
                            request, first, end = item
                            for row in range(first, min(first + tile, end)):
                                self.assertTrue(
                                    starts[request] - 97
                                    <= row
                                    < starts[request + 1] - 97
                                )
                                written.append(row)
                    self.assertEqual(
                        sorted(written),
                        list(range(offset, offset + count)),
                        (lengths, offset, count, tile),
                    )

    def test_scoring_arithmetic_is_identical_and_production_is_frozen(self):
        def arithmetic(path, name):
            fn = next(
                n
                for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            index = next(
                i for i, n in enumerate(fn.body) if isinstance(n, ast.AnnAssign)
            )
            return ast.dump(ast.Module(body=fn.body[index:], type_ignores=[]))

        self.assertEqual(
            hashlib.sha256(PRODUCTION.read_bytes()).hexdigest(), ab.PRODUCTION_SHA256
        )
        self.assertEqual(
            arithmetic(PRODUCTION, "_qsa_prefill_score_kernel"),
            arithmetic(TRIAL, "_qsa_prefill_compact_score_kernel"),
        )

    def test_fastpath_capture_times_only_original_fill_and_restores_globals(self):
        calls = []

        class Kernel:
            def __getitem__(self, grid):
                return lambda *a, **kw: calls.append((grid, a, kw))

        module = types.ModuleType("fake")
        kernel = module._select_all_kernel = Kernel()
        module._prefill_logits = module.persistent_topk = object()
        exec(
            "def qsa_select_paged_prefill(*args):\n    _select_all_kernel[(3,2)](*args, K=512)\n",
            vars(module),
        )
        bound = ab.select_all_stage(module, (1, 2))
        self.assertIs(module._select_all_kernel, kernel)
        calls.clear()
        bound()
        self.assertEqual(calls, [((3, 2), (1, 2), dict(K=512))])

    def test_na_stages_and_groups_do_not_publish_fake_score_ratios(self):
        case = workloads.observed_cases()[-1]
        stages = {
            name: dict(status="not_applicable", accepted=False)
            for name in ("score", "topk")
        }
        stages["pipeline"] = dict(
            status="passed",
            accepted=True,
            measurement_validated=True,
            speedup=2,
            baseline_us=dict(median=4),
            candidate_us=dict(median=2),
            paired_ratios=dict(median=2),
        )
        summary = dict(
            manifest=dict(options={}),
            cases=[dict(case=case, status="passed", stages=stages)],
        )
        with tempfile.TemporaryDirectory() as folder:
            bench.write_table(summary, Path(folder))
            table = (Path(folder) / "triton-stage-ab.tsv").read_text()
            self.assertIn("\tscore\tnot_applicable\t", table)
            self.assertNotIn("observed_perf:score", summary["grouped_speedups"])
            self.assertEqual(
                summary["grouped_speedups"]["observed_perf:pipeline"]["geometric_mean"],
                2,
            )


if __name__ == "__main__":
    unittest.main()
