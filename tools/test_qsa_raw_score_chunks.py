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

from qsa_raw_score_chunks_910b import (
    decoded_instructions,
    elf_identity,
    kernel_source,
    nz_offset,
)


def half(value):
    return struct.unpack("e", struct.pack("e", value))[0]


class RawScoreChunkTests(unittest.TestCase):
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
