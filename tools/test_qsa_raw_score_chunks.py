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

"""CPU contract checks for NZ packing, half weighting and binary audit evidence."""

import ast
import struct
import unittest
from types import SimpleNamespace

from qsa_k_pack_2d_probe_910b import source as k_pack_source
from qsa_raw_score_chunks_910b import (
    check_address_ir,
    decoded_instructions,
    elf_identity,
    kernel_source,
    nz_offset,
)
from qsa_raw_vector_profile_910b import copy_geometry, copy_source, summarize_vector_ir


def half(value):
    return struct.unpack("e", struct.pack("e", value))[0]


class RawScoreChunkTests(unittest.TestCase):
    def test_two_dimensional_part_matches_original_flat_nz_addresses(self):
        tree = ast.parse(k_pack_source())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "raw_k_2d_part"
        )
        expressions = {
            n.targets[0].id: n.value for n in fn.body if isinstance(n, ast.Assign)
        }
        for part in list(range(32)) + [3200, 3207]:
            fields = {
                key: eval(
                    compile(ast.Expression(expressions[key]), "k_part", "eval"),
                    {"part": part},
                )
                for key in ("tile", "row_start", "dim_start")
            }
            for r in range(32):
                for c in range(32):
                    old_i = part % 8 * 1024 + r * 32 + c
                    new_row = fields["tile"] * 64 + fields["row_start"] + r
                    new_dim = fields["dim_start"] + c
                    self.assertEqual(new_row, part // 8 * 64 + (old_i // 32) % 64)
                    self.assertEqual(new_dim, old_i // 2048 * 32 + old_i % 32)
                    self.assertEqual(
                        part * 1024 + r * 32 + c, fields["tile"] * 8192 + old_i
                    )
                    self.assertEqual(new_row // 192, fields["tile"] // 3)

    def test_two_dimensional_mask_at_row_tile_and_page_boundaries(self):
        for visible in (0, 1, 31, 32, 33, 63, 64, 65, 191, 192, 193, 257):
            seen = set()
            for part in range(((visible + 63) // 64) * 8):
                tile = part // 8
                for r in range(32):
                    row = tile * 64 + part % 2 * 32 + r
                    for c in range(32):
                        dimension = (part % 8 // 2) * 32 + c
                        if row < visible:
                            self.assertNotIn((row, dimension), seen)
                            seen.add((row, dimension))
            self.assertEqual(seen, {(r, c) for r in range(visible) for c in range(128)})

    def test_two_dimensional_pack_variants_keep_identical_part_body(self):
        tree = ast.parse(k_pack_source())
        fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in ("raw_k_2d_grid", "raw_k_2d_capped"):
            calls = [
                n
                for n in ast.walk(fns[name])
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "raw_k_2d_part"
            ]
            self.assertEqual(len(calls), 1)
        self.assertFalse(
            any(isinstance(n, ast.For) for n in ast.walk(fns["raw_k_2d_grid"]))
        )
        self.assertEqual(
            sum(isinstance(n, ast.For) for n in ast.walk(fns["raw_k_2d_capped"])), 1
        )

    def test_profile_controls_preserve_payload_and_matched_grids(self):
        for rows, width, tiles in ((128, 1088, 5), (128, 8192, 65), (112, 29696, 401)):
            g = copy_geometry(rows, width, tiles)
            self.assertEqual(g["key_grid"][0] * g["key_block"], tiles * 64 * 128)
            self.assertEqual(g["key_payload_bytes"], tiles * 8192)
            self.assertEqual(g["score_payload_bytes"], rows * width * 4)
            self.assertEqual(g["score_grid"], (rows, (width + 255) // 256))
            for kind in ("key", "score"):
                elements = g[kind + "_elements"]
                cores = g[kind + "_capped_grid"][0]
                blocks = [
                    i
                    for core in range(cores)
                    for i in range(core, (elements + 1023) // 1024, cores)
                ]
                self.assertEqual(sorted(blocks), list(range((elements + 1023) // 1024)))
                self.assertLessEqual(cores, 20)
        with self.assertRaises(ValueError):
            copy_geometry(0, 1088, 5)

    def test_profile_ir_counts_are_static_and_ignore_missing_assembly(self):
        text = (
            "%x = scf.for\n%v = memref.load %p[%i]\ntensor.insert %v into %t\n"
            "scf.yield {DiscreteMemAccess}\n} {ExtractedLoadOrStore}"
        )
        summary = summarize_vector_ir({"bcmlir": text, "npubin": b"ELF"})
        self.assertEqual(set(summary), {"bcmlir"})
        for key in (
            "scf_for",
            "scalar_memref_load",
            "tensor_insert",
            "discrete_access_markers",
            "extracted_access_markers",
        ):
            self.assertEqual(summary["bcmlir"][key], 1)
        self.assertEqual(summarize_vector_ir({}), {})

    def test_profile_copy_kernels_have_distinct_names_and_no_reduction(self):
        tree = ast.parse(copy_source())
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        self.assertEqual(
            {n.name for n in fns},
            {
                "raw_copy_k_grid",
                "raw_copy_out_grid",
                "raw_copy_k_capped",
                "raw_copy_out_capped",
            },
        )
        for fn in fns:
            calls = [
                ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)
            ]
            self.assertNotIn("tl.sum", calls)
            self.assertNotIn("tl.dot", calls)

    def test_native_weight_load_contract_requires_transposed_storage(self):
        tree = ast.parse(kernel_source("dialect-cann-bitcode"))
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "raw_pack_qw"
        )
        assignments = {
            n.targets[0].id: n.value for n in function.body if isinstance(n, ast.Assign)
        }
        packed = []
        for j in range(256):
            out_row = eval(
                compile(ast.Expression(assignments["out_row"]), "packer", "eval"),
                {"j": j},
            )
            in_head = eval(
                compile(ast.Expression(assignments["in_head"]), "packer", "eval"),
                {"j": j},
            )
            packed.append(100 * out_row + in_head)
        # Native LoadData2D transpose consumes W.T from L1, not W.
        for row in range(16):
            for head in range(16):
                self.assertEqual(packed[head * 16 + row], 100 * row + head)

    def test_raw_base_casts_and_byte_offsets_for_every_operand(self):
        class Index(int):
            def to(self, _dtype):
                return int(self)

        tree = ast.parse(kernel_source("dialect-cann-bitcode"))
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "raw_score_tiles"
        )
        assignments = {
            n.targets[0].id: n.value for n in ast.walk(fn) if isinstance(n, ast.Assign)
        }
        for name, operand in (
            ("q_base", "PQ"),
            ("key_base", "PK"),
            ("weight_base", "PW"),
            ("output_base", "Raw"),
        ):
            self.assertEqual(
                ast.unparse(assignments[name]), f"tl.cast({operand}, tl.uint64)"
            )
        for unit in (0, 1, 5, 159):
            group, tile = divmod(unit, 5)
            scope = dict(
                group=Index(group),
                tile=Index(tile),
                unit=Index(unit),
                q_base=0x100000,
                key_base=0x200000,
                weight_base=0x300000,
                output_base=0x400000,
                tl=SimpleNamespace(uint64="u64"),
            )
            wanted = dict(
                q=0x100000 + group * 16 * 128,
                key=0x200000 + tile * 64 * 128,
                weight=0x300000 + group * 16 * 16 * struct.calcsize("e"),
                output=0x400000 + unit * 16 * 64 * struct.calcsize("f"),
            )
            for name, expected in wanted.items():
                actual = eval(
                    compile(ast.Expression(assignments[name]), "addresses", "eval"),
                    scope,
                )
                self.assertEqual(actual, expected)

    def test_ir_gate_rejects_the_returned_offset_view_lowering(self):
        bad = (
            "%view = memref.reinterpret_cast %arg0 to offset: [%delta]\n"
            "%p = memref.extract_aligned_pointer_as_index %view : memref<1xf16> -> index"
        )
        with self.assertRaisesRegex(RuntimeError, "offset_views"):
            check_address_ir(bad)
        good = "\n".join(
            f"%p{i} = memref.extract_aligned_pointer_as_index %arg{i} : memref<?xi8> -> index"
            for i in range(4)
        )
        self.assertEqual(check_address_ir(good)["base_pointer_extractions"], 4)
        with self.assertRaises(RuntimeError):
            check_address_ir("")

    def test_nz_addresses_against_independent_block_enumeration(self):
        for rows, columns, c0 in ((16, 128, 32), (64, 128, 32), (16, 16, 16)):
            order = [
                (row, column)
                for cb in range(0, columns, c0)
                for rb in range(0, rows, 16)
                for row in range(rb, rb + 16)
                for column in range(cb, cb + c0)
            ]
            for offset, (row, column) in enumerate(order):
                self.assertEqual(nz_offset(row, column, rows, c0), offset)

    def test_packer_inverse_covers_queries_and_key_page_boundaries(self):
        for rows in (16, 64):
            for i in range(rows * 128):
                column = i // (rows * 32) * 32 + i % 32
                row = i // 32 % rows
                self.assertEqual(nz_offset(row, column, rows, 32), i)
        for tile in (0, 2, 3, 4, 64, 400):
            for offset in (0, 31, 32, 63):
                column = tile * 64 + offset
                self.assertEqual((tile // 3) * 192 + column % 192, column)

    def test_four_query_weights_do_not_mix_heads_or_tail_rows(self):
        dots = [4099, -4099, 0, 2047] * 4
        qs = [half(0.017 + i / 8192) for i in range(16)]
        weights = [half(0.5 + i / 32) for i in range(16)]
        for live in (1, 3, 4):
            matrix = [
                [
                    half(qs[h] * weights[h] * 1024) if r < live and h // 4 == r else 0
                    for h in range(16)
                ]
                for r in range(16)
            ]
            positive = [half(max(x / 1024, 0)) for x in dots]
            out = [sum(a * b for a, b in zip(row, positive)) for row in matrix]
            for r in range(live):
                wanted = sum(
                    half(max(dots[h] / 1024, 0)) * half(qs[h] * weights[h] * 1024)
                    for h in range(r * 4, r * 4 + 4)
                )
                self.assertEqual(out[r], wanted)
            self.assertTrue(all(x == 0 for x in out[live:]))

    def test_scratch_offsets_and_dense_output_tail(self):
        groups, tiles = 3, 5
        seen = set()
        for row in range(groups * 4):
            for column in range(tiles * 64):
                offset = (
                    ((row // 4) * tiles + column // 64) * 1024
                    + row % 4 * 64
                    + column % 64
                )
                self.assertLess(offset, groups * tiles * 1024)
                self.assertNotIn(offset, seen)
                seen.add(offset)

    def test_binary_evidence_requires_decoded_instruction_text(self):
        self.assertEqual(
            decoded_instructions(
                "file format elf64\n 0:\tdeadbeef\t.word 0x1\n 4:\t00 00\t<unknown>"
            ),
            [],
        )
        text = " 0:\t00 00\tMAD.s8 a,b,c\n 4:\t01 02\tFIX.L0C.TO.L1 a,b\n"
        self.assertEqual(
            [x[0] for x in decoded_instructions(text)], ["mad.s8", "fix.l0c.to.l1"]
        )
        binary = bytearray(64)
        binary[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<H", binary, 18, 0x1029)
        struct.pack_into("<I", binary, 48, 0x940000)
        self.assertEqual(elf_identity(binary)["machine"], "0x1029")
        self.assertEqual(elf_identity(b"not ELF")["status"], "not_elf")

    def test_both_generated_entries_have_no_array_buffers_in_cube_wrapper(self):
        for entry in ("dialect-cann-bitcode", "process-local-registry-bitcode"):
            tree = ast.parse(kernel_source(entry))
            function = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "raw_score_tiles"
            )
            calls = [
                ast.unparse(n.func)
                for n in ast.walk(function)
                if isinstance(n, ast.Call)
            ]
            self.assertNotIn("tl.load", calls)
            self.assertNotIn("tl.store", calls)
            self.assertNotIn("tl.arange", calls)


if __name__ == "__main__":
    unittest.main()
