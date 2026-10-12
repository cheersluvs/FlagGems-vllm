# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage of scheduling and cache addressing (not GPU validation)."""

import ast
import hashlib
import unittest
from pathlib import Path

from plan import addresses, cache_stride, geometry, positive_list


class PlanTests(unittest.TestCase):
    def test_candidate_branch_locals_are_disjoint(self):
        # Naming-policy regression, not a substitute for Triton/device compilation.
        path = Path(__file__).resolve().parent / "candidates.py"
        module = ast.parse(path.read_text())
        kernel = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "head_kernel"
        )

        def defined_names(statements):
            return {
                node.id
                for statement in statements
                for node in ast.walk(statement)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            }

        branches = [node for node in ast.walk(kernel) if isinstance(node, ast.If)]
        self.assertTrue(branches)
        for branch in branches:
            self.assertFalse(
                defined_names(branch.body) & defined_names(branch.orelse),
                f"shared branch-local names at line {branch.lineno}",
            )

    def test_frozen_baseline(self):
        path = Path(__file__).resolve().parent / "frozen_a.py"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "04a4eeddc6dda1f66cfed8cef51bcfe8b004750eca9d753db247d66655e77a8d",
        )

    def test_exactly_once(self):
        for n, h, hpp in [(1, 1, 4), (17, 7, 4), (512, 64, 4), (9, 65, 8)]:
            groups = (h + hpp - 1) // hpp
            for arm in ("A", "B", "B1", "C"):
                q_work, kv_work = [], []
                grid, kv_tpp, merge = geometry(arm, n, h, hpp)
                for pid in range(grid):
                    if arm == "A":
                        tile, slot = divmod(pid, h + 1)
                        for tok in range(tile * 4, min(n, tile * 4 + 4)):
                            if slot == h:
                                kv_work.append(tok)
                            else:
                                q_work.append((tok, slot))
                    elif pid < n * groups:
                        tok, group = divmod(pid, groups)
                        q_work.extend(
                            (tok, head)
                            for head in range(group * hpp, min(h, (group + 1) * hpp))
                        )
                        if merge and group == groups - 1:
                            kv_work.append(tok)
                    else:
                        start = (pid - n * groups) * kv_tpp
                        kv_work.extend(range(start, min(n, start + kv_tpp)))
                self.assertEqual(
                    sorted(q_work), [(t, hd) for t in range(n) for hd in range(h)]
                )
                self.assertEqual(sorted(kv_work), list(range(n)))

    def test_cache_disjoint(self):
        for bs in (1, 16, 64):
            covered = set()
            for slot in range(2 * bs):
                data, scale = addresses(slot, bs)
                offsets = set(range(data, data + 576)) | set(range(scale, scale + 8))
                self.assertEqual(len(offsets), 584)
                self.assertFalse(covered & offsets)
                self.assertLess(max(offsets), 2 * cache_stride(bs))
                covered |= offsets

    def test_bad_parameters(self):
        for value in ("0", "-1", "2,2", ""):
            with self.assertRaises(ValueError):
                positive_list(value)
        with self.assertRaises(ValueError):
            positive_list("3", powers_of_two=True)
        with self.assertRaises(ValueError):
            geometry("unknown", 1, 64, 4)
        with self.assertRaises(ValueError):
            geometry("A", 0, 64, 4)


if __name__ == "__main__":
    unittest.main()
