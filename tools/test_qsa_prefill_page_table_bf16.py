# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Prove dtype scope is the only difference from the earlier page-table trial."""

import ast
import copy
import unittest

from tools import test_qsa_prefill_page_table as common

KERNEL = common.ROOT / "tools/qsa_prefill_page_table_bf16_kernel.py"


class BF16PageTableTests(common.PageTableTests):
    kernel = KERNEL
    bf16_only = True

    def test_only_new_change_is_the_constexpr_dtype_gate(self):
        old = common.score(ast.parse(common.KERNEL.read_text()))
        new = common.score(ast.parse(KERNEL.read_text()))
        old_loop = next(n for n in old.body if isinstance(n, ast.For))
        new_loop = next(n for n in new.body if isinstance(n, ast.For))
        old_branch = next(n for n in old_loop.body if isinstance(n, ast.If))
        new_branch = next(n for n in new_loop.body if isinstance(n, ast.If))
        self.assertEqual(
            ast.dump(new_branch.test),
            ast.dump(
                ast.parse("not IS_FP8 and BLOCK_N <= PAGE_SIZE", mode="eval").body
            ),
        )
        # Specialize the constexpr gate: FP8 recovers the exact old vector
        # lookup; BF16 recovers the measured scalar/general branch unchanged.
        for fp8 in (False, True):
            condition = compile(ast.Expression(new_branch.test), "condition", "eval")
            for block, page in ((64, 196), (128, 196), (128, 16)):
                selected = eval(
                    condition, dict(IS_FP8=fp8, BLOCK_N=block, PAGE_SIZE=page)
                )
                self.assertEqual(selected, not fp8 and block <= page)
        new_branch.test = copy.deepcopy(old_branch.test)
        self.assertEqual(ast.dump(new), ast.dump(old))

    def test_actual_fp8_vector_lookup_at_all_live_columns(self):
        self.fp8 = True
        self.test_actual_load_branch_matches_vector_lookup_at_all_live_columns()


if __name__ == "__main__":
    unittest.main()
