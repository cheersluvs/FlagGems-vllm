# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU checks of actual generated query packing and native operand contracts."""

import ast
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import probe_qsa_raw_cube_chain_910b as driver
from qsa_cube_query_pack_probe_910b import geometry, source
from qsa_raw_score_chunks_910b import nz_offset


class Index(int):
    def __getitem__(self, index):
        return int(self)

    def to(self, dtype):
        return int(self)


def expression(node):
    return compile(ast.Expression(node), "query_pack_probe", "eval")


def function(queries, suffix):
    return next(
        n
        for n in ast.parse(source(queries)).body
        if isinstance(n, ast.FunctionDef) and n.name == f"raw_query{queries}_{suffix}"
    )


def calls(fn, kind):
    return [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == kind
    ]


def mask(load):
    return next(k.value for k in load.keywords if k.arg == "mask")


class QueryPackTests(unittest.TestCase):
    def test_native_variant_symbols_expand_to_distinct_abi_entries(self):
        cxx = shutil.which("c++") or shutil.which("clang++")
        if not cxx:
            self.skipTest(
                "No host C++ preprocessor; native ccec symbol gate still required"
            )
        cpp = Path(__file__).with_name("qsa_raw_cube_query_pack_910b.cpp").read_text()
        cpp = cpp.replace('#include "kernel_operator.h"', "")
        for queries in (4, 8, 16):
            output = subprocess.run(
                [cxx, "-E", "-P", "-x", "c++", f"-DQSA_QUERY_PACK={queries}", "-"],
                input=cpp,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
            self.assertIn(f"_mlir_ciface_qsa_raw_chain_address_query{queries}(", output)
            self.assertIn(f"_mlir_ciface_qsa_raw_qk_address_query{queries}(", output)
            self.assertNotIn("_queryQSA_QUERY_PACK", output)

    def test_variant_build_flags_and_artifacts_do_not_overwrite_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "custom-plain.bc"
            original.write_bytes(b"baseline")
            seen = []

            def command(args, *unused):
                seen.append(args)
                if "-o" in args:
                    path = Path(args[args.index("-o") + 1])
                    path.write_bytes(
                        b"native" if "-emit-llvm" in args else b"LLVM text"
                    )
                return dict(returncode=0)

            with patch.object(
                driver, "tool", side_effect=lambda name, *_: "/mock/" + name
            ), patch.object(driver, "include_dirs", return_value=[]), patch.object(
                driver, "command", side_effect=command
            ), patch.object(
                driver, "inspect_bitcode", return_value=dict(exported_symbols=True)
            ):
                result = {}
                path = driver.build_bitcode(
                    root,
                    result,
                    Path(__file__).with_name("qsa_raw_cube_query_pack_910b.cpp"),
                    stem="query8",
                    defines=("QSA_QUERY_PACK=8",),
                )
            self.assertEqual(original.read_bytes(), b"baseline")
            self.assertEqual(path, root / "query8-plain.bc")
            build = next(args for args in seen if "-emit-llvm" in args)
            self.assertIn("-DQSA_QUERY_PACK=8", build)
            self.assertIn("--cce-aicore-only", build)
            self.assertTrue((root / "query8-plain.bc.gz").is_file())
            self.assertTrue((root / "query8-source.cpp.txt").is_file())

    def test_actual_query_and_weight_nz_addresses_cover_padded_groups(self):
        for queries in (4, 8, 16):
            fn = function(queries, "group")
            stores = calls(fn, "tl.store")
            q_addr = expression(
                next(
                    n.args[0]
                    for n in stores
                    if ast.unparse(n.args[0]).startswith("PQ +")
                )
            )
            w_addr = expression(
                next(
                    n.args[0]
                    for n in stores
                    if ast.unparse(n.args[0]).startswith("PW +")
                )
            )
            load = next(
                n
                for n in calls(fn, "tl.load")
                if ast.unparse(n.args[0]).startswith("Q +")
            )
            q_src, q_mask = expression(load.args[0]), expression(mask(load))
            assignments = {
                n.targets[0].id: expression(n.value)
                for n in ast.walk(fn)
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ("flat", "live")
            }
            for rows in (1, 5, 17):
                g = geometry(rows, 2, queries)
                heads = queries * 4
                seen_q = set()
                seen_w = set()
                seen_inputs = set()
                for group in range(g["groups"]):
                    for hb in range(queries // 4):
                        for h in range(16):
                            env = dict(
                                QUERIES=queries,
                                HEADS=heads,
                                ROWS=rows,
                                group=group,
                                head_block=hb,
                                heads=Index(h),
                                PQ=0,
                                PW=0,
                                Q=0,
                            )
                            env["flat"] = Index(eval(assignments["flat"], {}, env))
                            for b in range(4):
                                for dim in range(32):
                                    env.update(block=b, dims=Index(dim))
                                    dst = eval(q_addr, {}, env)
                                    self.assertEqual(
                                        dst,
                                        group * heads * 128
                                        + nz_offset(
                                            hb * 16 + h, b * 32 + dim, heads, 32
                                        ),
                                    )
                                    self.assertNotIn(dst, seen_q)
                                    seen_q.add(dst)
                                    if eval(q_mask, {}, env):
                                        src = eval(q_src, {}, env)
                                        self.assertLess(src, rows * 4 * 128)
                                        seen_inputs.add(src)
                            for out in range(16):
                                env["out_rows"] = Index(out)
                                dst = eval(w_addr, {}, env)
                                self.assertEqual(
                                    dst,
                                    group * heads * 16
                                    + nz_offset(hb * 16 + h, out, heads, 16),
                                )
                                self.assertNotIn(dst, seen_w)
                                seen_w.add(dst)
                                live = eval(assignments["live"], {}, env)
                                self.assertEqual(
                                    live, (hb * 16 + h) // 4 == out and out < queries
                                )
                self.assertEqual(seen_q, set(range(g["groups"] * heads * 128)))
                self.assertEqual(seen_w, set(range(g["groups"] * heads * 16)))
                self.assertEqual(seen_inputs, set(range(rows * 4 * 128)))

    def test_actual_finish_mapping_keeps_each_query_and_dense_mask(self):
        for queries in (4, 8, 16):
            fn = function(queries, "finish_part")
            assignments = {
                n.targets[0].id: expression(n.value)
                for n in fn.body
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ("raw_base", "live")
            }
            rows, width, tiles = 17, 257, 2
            for row in range(rows):
                visible = [0, 1, 63, 64, 65][row % 5]
                for tile in range((width + 63) // 64):
                    for c in range(64):
                        env = dict(
                            row=row,
                            tile=tile,
                            c=c,
                            QUERIES=queries,
                            RAW_TILES=tiles,
                            col=tile * 64 + c,
                            WIDTH=width,
                            visible=visible,
                        )
                        live = eval(assignments["live"], {}, env)
                        self.assertEqual(live, tile * 64 + c < min(width, visible))
                        if live:
                            base = eval(assignments["raw_base"], {}, env)
                            self.assertEqual(
                                base,
                                ((row // queries) * tiles + tile) * 1024
                                + (row % queries) * 64,
                            )
                            self.assertLess(
                                base + c,
                                geometry(rows, tiles, queries)["raw_payload_bytes"]
                                // 4,
                            )

    def test_actual_cube_abi_byte_offsets_match_storage_and_are_disjoint(self):
        for queries in (4, 8, 16):
            fn = function(queries, "cube")
            assignments = {
                n.targets[0].id: expression(n.value)
                for n in ast.walk(fn)
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ("q", "key", "weight", "output")
            }
            g = geometry(17, 3, queries)
            outputs = set()
            for group in range(g["groups"]):
                for tile in range(3):
                    unit = group * 3 + tile
                    env = dict(
                        group=Index(group),
                        tile=Index(tile),
                        unit=Index(unit),
                        QUERIES=queries,
                        q_base=0,
                        key_base=0,
                        weight_base=0,
                        output_base=0,
                        tl=SimpleNamespace(uint64=None),
                    )
                    observed = {k: eval(v, {}, env) for k, v in assignments.items()}
                    self.assertEqual(
                        observed,
                        dict(
                            q=group * g["q_bytes"],
                            key=tile * 8192,
                            weight=group * g["w_bytes"],
                            output=unit * 4096,
                        ),
                    )
                    outputs.add(observed["output"])
            self.assertEqual(len(outputs), g["native_calls"])
            self.assertEqual(max(outputs) + 4096, g["raw_payload_bytes"])

    def test_native_second_cube_shapes_and_reused_buffers(self):
        cpp = Path(__file__).with_name("qsa_raw_cube_query_pack_910b.cpp").read_text()
        self.assertIn("mm.m = O;\n    mm.k = M;", cpp)
        self.assertIn("store_result<float, O>(out, acc_float)", cpp)
        self.assertIn("ld.repeatTimes = M / 16", cpp)
        self.assertIn("M * O * sizeof(half)", cpp)
        for queries in (4, 8, 16):
            g = geometry(17, 3, queries)
            heads = queries * 4
            self.assertGreaterEqual(g["l0a_bytes"], 16 * heads * 2)
            self.assertGreaterEqual(g["l0b_bytes"], heads * 64 * 2)
            self.assertGreaterEqual(g["l0c_bytes"], 16 * 64 * 4)
            self.assertLessEqual(g["l0c_bytes"], 32768)

    def test_payload_and_call_reduction_do_not_claim_less_math(self):
        g4 = geometry(128, 401, 4)
        for queries in (8, 16):
            g = geometry(128, 401, queries)
            self.assertEqual(g["native_calls"] * queries, g4["native_calls"] * 4)
            self.assertEqual(
                g["raw_payload_bytes"] * queries, g4["raw_payload_bytes"] * 4
            )
            # Total first-Cube M dimension is constant; second K grows per call.
            self.assertEqual(
                g["native_calls"] * g["heads"], g4["native_calls"] * g4["heads"]
            )

    def test_invalid_geometry_rejected_and_generated_source_parses(self):
        for args in ((0, 1, 4), (1, 0, 4), (1, 1, 2), (1, 1, 32)):
            with self.assertRaises(ValueError):
                geometry(*args)
        for queries in (4, 8, 16):
            tree = ast.parse(source(queries))
            names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
            self.assertEqual(len(names), len(set(names)))
            self.assertIn(f"raw_query{queries}_cube", names)


if __name__ == "__main__":
    unittest.main()
