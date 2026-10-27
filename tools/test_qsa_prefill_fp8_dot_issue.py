# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Source-only FP8 movement, actual ABI type names and conservative gates."""

import ast
import copy
import unittest

from tools import qsa_prefill_fp8_dot_issue as p


class DotIssueTests(unittest.TestCase):
    def test_move_preserves_every_ast_node_except_statement_order(self):
        a, b = p.source("production"), p.source("paired")
        self.assertEqual(
            b.replace(p.ANCHOR + p.MOVE, p.ANCHOR).replace(
                "            scores_right = tl.reshape",
                p.MOVE + "            scores_right = tl.reshape",
                1,
            ),
            a,
        )
        trees = [ast.parse(s) for s in (a, b)]
        for tree in trees:
            fn = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == p.NAME
            )
            loop = next(n for n in fn.body if isinstance(n, ast.For))
            branch = next(
                n
                for n in loop.body
                if isinstance(n, ast.If)
                and isinstance(n.test, ast.Name)
                and n.test.id == "SPLIT_QUERY"
            )
            right = next(
                n
                for n in branch.body
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "scores_right"
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == "dot"
            )
            branch.body.remove(right)
            branch.body.insert(1, right)
        self.assertEqual(ast.dump(trees[0]), ast.dump(trees[1]))
        self.assertEqual(a.count("tl.dot("), b.count("tl.dot("))
        with self.assertRaises(ValueError):
            p.source("unknown")

    def header(self):
        fn = next(
            n
            for n in ast.parse(p.source("production")).body
            if isinstance(n, ast.FunctionDef) and n.name == p.NAME
        )
        return (
            "tt.func public @"
            + p.NAME
            + "("
            + ", ".join(
                "%"
                + n.arg
                + ": "
                + (
                    "!tt.ptr<f8E4M3FN>"
                    if i < 2
                    else (
                        "!tt.ptr<i32>" if i < 5 else "!tt.ptr<f32>" if i == 5 else "i32"
                    )
                )
                + " {tt.divisibility = 16 : i32} loc(unknown)"
                for i, n in enumerate(fn.args.args[:19])
                if n.arg not in ("stride_table_page", "num_requests")
            )
            + ") attributes {noinline=false} {"
        )

    def test_astsource_uses_triton_type_names_not_mlir_type_names(self):
        sig, constants, attrs = p.abi(self.header())
        self.assertEqual(sig["q_ptr"], "*fp8e4nv")
        self.assertEqual(sig["cache_ptr"], "*fp8e4nv")
        self.assertEqual(sig["logits_ptr"], "*fp32")
        self.assertEqual(sig["num_rows"], "i32")
        self.assertEqual(len(sig), 27)
        self.assertEqual(constants["QUERY_PARTS"], 2)
        self.assertTrue(constants["IS_FP8"])
        self.assertEqual(attrs[(0,)], [("tt.divisibility", 16)])
        for header in (
            self.header().replace("f8E4M3FN", "bf16"),
            self.header().replace("%num_rows:", "%unexpected:"),
        ):
            with self.assertRaises(RuntimeError):
                p.abi(header)

    def test_gate_rejects_unchanged_code_spills_resources_or_work(self):
        a = dict(
            code_sha256="a",
            registers_per_thread=158,
            shared_bytes=57344,
            spill_store_bytes=0,
            spill_load_bytes=0,
            instructions=dict(dot_sites=2, mma_sites=8),
        )
        b = dict(a, code_sha256="b")
        self.assertIsNone(p.gate(a, b))
        self.assertIsNotNone(p.gate(a, a))
        for name, value in (
            ("spill_store_bytes", 8),
            ("spill_load_bytes", 8),
            ("registers_per_thread", 159),
            ("shared_bytes", 65536),
        ):
            self.assertIsNotNone(p.gate(a, dict(b, **{name: value})))
        c = copy.deepcopy(b)
        c["instructions"]["mma_sites"] = 4
        self.assertIsNotNone(p.gate(a, c))

    def test_tile32_accepts_overlap_cost_only_under_original_ceiling(self):
        original = dict(registers_per_thread=158, shared_bytes=57344)
        sequential = dict(
            code_sha256="sequential",
            registers_per_thread=90,
            shared_bytes=32768,
            spill_store_bytes=0,
            spill_load_bytes=0,
            instructions=dict(dot_sites=2, mma_sites=8),
        )
        order = [
            "%a = ttng.warp_group_dot",
            "%b = ttng.warp_group_dot",
            "%l = ttng.warp_group_dot_wait {pendings = 1 : i32}",
            "tt.reduce",
            "tt.store",
            "%r = ttng.warp_group_dot_wait {pendings = 0 : i32}",
        ]
        paired = dict(
            sequential,
            code_sha256="paired",
            registers_per_thread=120,
            instructions=dict(dot_sites=2, mma_sites=8, dot_epilogue_order=order),
        )
        self.assertIsNone(p.tile32_gate(original, sequential, paired))
        for key, value in (
            ("registers_per_thread", 159),
            ("shared_bytes", 65536),
            ("spill_load_bytes", 4),
            ("code_sha256", "sequential"),
        ):
            self.assertIsNotNone(
                p.tile32_gate(original, sequential, dict(paired, **{key: value}))
            )
        for changed in (
            [line.replace("pendings = 1", "pendings = 0") for line in order],
            [order[0], order[2], order[1], *order[3:]],
            [line.replace("pendings = 1", "unknown = 1") for line in order],
        ):
            bad = dict(
                paired,
                instructions=dict(paired["instructions"], dot_epilogue_order=changed),
            )
            self.assertIsNotNone(p.tile32_gate(original, sequential, bad))
        bad = dict(paired, instructions=dict(paired["instructions"], mma_sites=4))
        self.assertIsNotNone(p.tile32_gate(original, sequential, bad))


if __name__ == "__main__":
    unittest.main()
