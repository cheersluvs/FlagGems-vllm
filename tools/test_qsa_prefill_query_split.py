# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU contracts for query subtiles, original controls and fixed-config replay."""

import ast
import copy
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import benchmark_qsa_prefill_triton_stages as bench
from tools import qsa_prefill_query_split_matched as matched
from tools import qsa_prefill_query_split_trial as trial
from tools import test_qsa_prefill_page_table as page
from tools import test_qsa_prefill_tuning_round1 as goals

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "tools/qsa_prefill_query_split_kernel.py"


def function(path):
    return page.score(ast.parse(path.read_text()))


class ScalarLanes(ast.NodeTransformer):
    """Evaluate actual per-element query pointer/mask AST without array deps."""

    def visit_Subscript(self, node):
        return self.visit(node.value)


class SplitTests(unittest.TestCase):
    def test_original_path_and_all_outer_logic_are_exact(self):
        old, new = function(page.PRODUCTION), function(KERNEL)
        new.body = [
            n
            for n in new.body
            if not (isinstance(n, ast.AnnAssign) and n.target.id == "SPLIT_QUERY")
        ]
        branches = [
            n
            for n in ast.walk(new)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Name)
            and n.test.id == "SPLIT_QUERY"
        ]
        self.assertEqual(len(branches), 2)
        self.assertEqual(
            ast.dump(branches[0].orelse[0]),
            ast.dump(
                next(
                    n
                    for n in old.body
                    if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == "query"
                )
            ),
        )

        class Original(ast.NodeTransformer):
            def visit_If(self, node):
                if isinstance(node.test, ast.Name) and node.test.id == "SPLIT_QUERY":
                    return node.orelse
                return self.generic_visit(node)

        new = Original().visit(new)
        self.assertEqual(ast.dump(old), ast.dump(new))

    def test_each_split_dot_and_epilogue_preserves_original_full_d_math(self):
        old = function(page.PRODUCTION)
        new = function(KERNEL)
        loop = next(n for n in old.body if isinstance(n, ast.For))
        begin = next(
            i
            for i, n in enumerate(loop.body)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "scores"
        )
        reference = copy.deepcopy(loop.body[begin:])

        # Original TILE_R padding mask is unnecessary for exact two half tiles.
        class Names(ast.NodeTransformer):
            def visit_Subscript(self, n):
                if isinstance(n.value, ast.Name) and n.value.id == "valid_rows":
                    return ast.parse("rows[None, :] < request_end", mode="eval").body
                return self.generic_visit(n)

        reference = [Names().visit(n) for n in reference]
        new_loop = next(n for n in new.body if isinstance(n, ast.For))
        branch = next(n for n in new_loop.body if isinstance(n, ast.If))
        for label in ("left", "right"):
            selected = [
                copy.deepcopy(n)
                for n in branch.body
                if (
                    isinstance(n, ast.Assign)
                    and n.targets[0].id.endswith("_" + label)
                    and n.targets[0].id != "rows_" + label
                )
                or (isinstance(n, ast.Expr) and label in ast.unparse(n))
            ]

            class Normalize(ast.NodeTransformer):
                def visit_Name(self, n):
                    if n.id.endswith("_" + label):
                        n.id = n.id[: -len("_" + label)]
                    if n.id == "HALF_R":
                        n.id = "BLOCK_R"
                    return n

            selected = [Normalize().visit(n) for n in selected]
            self.assertEqual(
                [ast.dump(n) for n in reference], [ast.dump(n) for n in selected]
            )

    def test_actual_split_query_load_addresses_masks_and_half_visibility(self):
        fn = function(KERNEL)
        branch = next(
            n
            for n in fn.body
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Name)
            and n.test.id == "SPLIT_QUERY"
        )
        assignments = [
            n
            for n in branch.body
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ("query_left", "query_right")
        ]
        tree = ast.Module(body=copy.deepcopy(assignments), type_ignores=[])
        code = compile(
            ast.fix_missing_locations(ScalarLanes().visit(tree)), str(KERNEL), "exec"
        )
        for tile in (32, 64):
            half = tile // 2
            for heads, dim in ((1, 32), (3, 96), (4, 128), (5, 64)):
                hp = 1 << (heads - 1).bit_length()
                dp = 1 << (dim - 1).bit_length()
                for row_start, end in ((17, 18), (17, 36), (17, 81)):
                    visited = set()
                    for lane in range(half * hp):
                        row, h = divmod(lane, hp)
                        for d in (0, dim - 1, dp - 1):
                            reads = []

                            def load(ptr, mask, other):
                                if mask:
                                    reads.append(ptr)
                                    self.assertIn(ptr, memory)
                                    return memory[ptr]
                                return other

                            sr = (heads + 1) * dim
                            sh = dim
                            memory = {
                                r * sr + h * sh + d: (r, h, d)
                                for r in range(row_start, end)
                                for h in range(heads)
                                for d in (0, dim - 1, dp - 1)
                                if d < dim
                            }
                            env = dict(
                                q_ptr=0,
                                row_start=row_start,
                                HALF_R=half,
                                half_rows=row,
                                half_heads=h,
                                dims=d,
                                stride_q_row=sr,
                                stride_q_head=sh,
                                request_end=end,
                                NUM_HEADS=heads,
                                HEAD_DIM=dim,
                                tl=types.SimpleNamespace(load=load),
                            )
                            exec(code, env)
                            for label, offset in (("left", 0), ("right", half)):
                                r = row_start + offset + row
                                live = r < end and h < heads and d < dim
                                self.assertEqual(
                                    env["query_" + label], (r, h, d) if live else 0.0
                                )
                                if live:
                                    visited.add((r, h, d))
                    self.assertTrue(visited)
            # Execute the actual visibility reshape/transpose/split AST.
            vis = next(
                n
                for n in branch.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
            )
            tl = types.SimpleNamespace(
                reshape=lambda x, shape: [
                    x[i * shape[1] : (i + 1) * shape[1]] for i in range(shape[0])
                ],
                trans=lambda x: list(zip(*x)),
                split=lambda x: tuple([v[i] for v in x] for i in range(2)),
            )
            env = dict(tl=tl, visible=list(range(tile)), HALF_R=half)
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[vis], type_ignores=[])),
                    str(KERNEL),
                    "exec",
                ),
                env,
            )
            self.assertEqual(env["visible_left"], list(range(half)))
            self.assertEqual(env["visible_right"], list(range(half, tile)))

    def test_factory_retains_original_host_topk_and_cache_isolation(self):
        test = page.PageTableTests()
        test.kernel = KERNEL
        adapter = types.SimpleNamespace(
            candidate_module=lambda p, **kw: trial.candidate_module(p)
        )
        with patch.object(page, "trial", adapter):
            test.test_factory_preserves_host_topk_and_isolates_aligned_entry_classes()

    def test_acceptance_and_table_retain_pipeline_goal_and_fixed_score_stage(self):
        s = goals.AcceptanceTests.summary()
        s["manifest"]["options"].update(
            candidate="query-split", compare_previous_tuning=False
        )
        for c in s["cases"]:
            c["runtime"] = dict(
                operator_binding=dict(canonical_sha256=trial.PRODUCTION_SHA256)
            )
            for k in ("score_vs_production", "pipeline_vs_production", "score_matched"):
                c["stages"][k] = copy.deepcopy(c["stages"]["score"])
        self.assertTrue(trial.acceptance(s)["measurement_valid"])
        self.assertTrue(trial.acceptance(s)["target_met"])
        self.assertIn("score_matched", bench.SCORE_STAGES)
        self.assertIn("score_matched", bench.STAGE_ORDER)
        with tempfile.TemporaryDirectory() as d:
            bench.write_table(s, Path(d))
            self.assertIn(
                "\tscore_matched\t", (Path(d) / "triton-stage-ab.tsv").read_text()
            )
        first = next(c for c in s["cases"] if c["case"]["dtype"] == "float8_e4m3fn")
        first["cache_granularity"] = dict(
            production_baseline=[dict(route="rectangular", constants=dict(TILE_R=64))]
        )
        first["query_split"] = dict(matched=dict(configs_match=True))
        first["stages"]["score_matched"].update(
            baseline_kind="production_fixed_config",
            includes_topk=False,
        )
        self.assertTrue(trial.acceptance(s)["measurement_valid"])
        fixed = first["stages"].pop("score_matched")
        self.assertFalse(trial.acceptance(s)["measurement_valid"])
        first["stages"]["score_matched"] = fixed
        first["query_split"]["matched"]["configs_match"] = False
        self.assertFalse(trial.acceptance(s)["measurement_valid"])
        first["query_split"]["matched"]["configs_match"] = True
        s["cases"][0]["stages"].pop("pipeline_vs_production")
        self.assertFalse(trial.acceptance(s)["measurement_valid"])


class FixedTests(unittest.TestCase):
    def test_exact_chunk_configs_and_unknown_chunk_rejection(self):
        calls = []
        kernel = types.SimpleNamespace(
            asm={"ptx": "", "cubin": b""}, metadata=types.SimpleNamespace()
        )

        class Jit:
            def __getitem__(self, grid):
                def launch(*values, **kwargs):
                    calls.append((grid, values, kwargs))
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
                constants={"TILE_R": 64},
            ),
            dict(
                rows=3,
                offset=64,
                width=1024,
                route="rectangular",
                constants={"TILE_R": 8},
            ),
        ]
        fixed = matched.FixedEntry(entry, traces, "rectangular")
        for t in traces:
            _, cfg = fixed[lambda c: (c["TILE_R"],)](t["rows"], t["offset"], t["width"])
            self.assertEqual(cfg, t["constants"])
            self.assertEqual(calls[-1][0], (cfg["TILE_R"],))
            self.assertEqual(calls[-1][2], cfg)
        fixed.frozen = True
        with self.assertRaisesRegex(RuntimeError, "not captured"):
            fixed[(1,)](2, 999, 1024)
        pair = []
        for _ in range(2):
            pair.append(
                types.SimpleNamespace(
                    _qsa_prefill_score_kernel=fixed,
                    _qsa_prefill_compact_score_kernel=matched.FixedEntry(
                        entry, traces, "compact"
                    ),
                )
            )
        self.assertTrue(matched.freeze_and_records(pair)["configs_match"])
        fixed.compiled.pop((3, 64, 1024))
        with self.assertRaisesRegex(RuntimeError, "during timing"):
            fixed[(1,)](3, 64, 1024)
        with self.assertRaisesRegex(RuntimeError, "omitted"):
            matched.freeze_and_records(pair)
        with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
            matched.modules(None, None, traces + traces)

    def test_fixed_modules_rebind_host_and_keep_independent_launch_records(self):
        calls = []
        kernel = types.SimpleNamespace(
            asm={"ptx": "", "cubin": b""}, metadata=types.SimpleNamespace()
        )

        class Jit:
            def __getitem__(self, grid):
                def launch(*a, **kw):
                    calls.append(grid)
                    return kernel

                return launch

        sources = []
        for name in ("baseline", "candidate"):
            module = types.ModuleType(name)
            module.persistent_topk = object()
            entry = types.SimpleNamespace(
                arg_names=["num_rows", "query_offset", "logits_width"],
                fn=types.SimpleNamespace(fn=Jit()),
            )
            module._qsa_prefill_score_kernel = entry
            module._qsa_prefill_compact_score_kernel = entry
            exec(
                "def _prefill_logits():\n return _qsa_prefill_score_kernel[lambda c:(c['TILE_R'],)](64,0,1024)\n"
                "def qsa_select_paged_prefill():\n return _prefill_logits()\n",
                vars(module),
            )
            sources.append(module)
        traces = [
            dict(
                rows=64,
                offset=0,
                width=1024,
                route="rectangular",
                constants={"TILE_R": 64},
            )
        ]
        pair = matched.modules(*sources, traces)
        for module, original in zip(pair, sources):
            self.assertIs(
                module._prefill_logits.__code__, original._prefill_logits.__code__
            )
            self.assertIs(
                module.qsa_select_paged_prefill.__code__,
                original.qsa_select_paged_prefill.__code__,
            )
            self.assertIs(module.persistent_topk, original.persistent_topk)
            module.qsa_select_paged_prefill()
        self.assertEqual(calls, [(64,), (64,)])
        self.assertIsNot(
            pair[0]._qsa_prefill_score_kernel.compiled,
            pair[1]._qsa_prefill_score_kernel.compiled,
        )
        self.assertTrue(matched.freeze_and_records(pair)["configs_match"])


if __name__ == "__main__":
    unittest.main()
