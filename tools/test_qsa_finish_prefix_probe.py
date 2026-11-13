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

"""Exercise generated diagnostic addresses, live sinks, and signed comparisons."""

import ast
import copy
import unittest

from qsa_finish_prefix_probe_910b import (
    MODES,
    READS,
    ir_liveness,
    signed_differences,
    source,
)


class Integer(int):
    def to(self, dtype):
        return self


class Floating(float):
    def to(self, dtype):
        return self


class Pointer:
    def __init__(self, name, offset=0):
        self.name, self.offset = name, offset

    def __add__(self, offset):
        return Pointer(self.name, self.offset + int(offset))


class ScalarTL:
    float32 = int64 = None

    def __init__(self, buffers):
        self.buffers = buffers
        self.reads = set()
        self.writes = {}
        self.column = 0
        self.pid = 0

    def arange(self, begin, end):
        return self.column

    def full(self, shape, value, dtype):
        return value

    def where(self, condition, yes, no):
        return yes if condition else no

    def load(self, pointer, mask=True, other=0):
        if not mask:
            return Integer(other) if isinstance(other, int) else Floating(other)
        data = self.buffers[pointer.name]
        if not 0 <= pointer.offset < len(data):
            raise AssertionError(
                f"Out-of-bounds read: {pointer.name}/{pointer.offset}/{len(data)}"
            )
        self.reads.add(pointer.name)
        value = data[pointer.offset]
        return Integer(value) if isinstance(value, int) else Floating(value)

    def store(self, pointer, value, mask=True):
        if not mask:
            return
        if not 0 <= pointer.offset < len(self.buffers[pointer.name]):
            raise AssertionError(
                f"Out-of-bounds write: {pointer.name}/{pointer.offset}"
            )
        if pointer.offset in self.writes:
            raise AssertionError("Duplicate output element")
        self.writes[pointer.offset] = value

    def program_id(self, axis):
        return self.pid


def device_functions(tl):
    nodes = []
    for original in ast.parse(source()).body:
        if not isinstance(original, ast.FunctionDef):
            continue
        node = copy.deepcopy(original)
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
        nodes.append(node)
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = dict(
        tl=tl,
        triton=type(
            "Triton", (), dict(cdiv=staticmethod(lambda a, b: (a + b - 1) // b))
        ),
    )
    exec(compile(tree, "generated_finish_diagnostic", "exec"), namespace)
    return namespace


def fixture():
    rows, width, tiles, maximum = 17, 257, 2, 65
    visible = ([0, 1, 63, 64, 65] * 4)[:rows]
    raw = [float(i % 257 - 128) / 16 for i in range(2 * tiles * 16 * 64)]
    ks = [float(i + 1) / 1024 for i in range(3 * 192)]
    return (
        rows,
        width,
        tiles,
        maximum,
        dict(
            Raw=raw, KS=ks, Table=[2, 0], Visible=visible, Out=[None] * (rows * width)
        ),
    )


class FinishPrefixTests(unittest.TestCase):
    def test_every_prefix_has_exact_sink_values_and_safe_partial_bounds(self):
        rows, width, tiles, maximum, buffers = fixture()
        for label, mode in MODES.items():
            tl = ScalarTL(buffers)
            fn = device_functions(tl)["raw_finish_prefix_part"]
            for row in range(rows):
                for tile in range((width + 63) // 64):
                    for c in range(64):
                        tl.column = c
                        fn(
                            *(
                                Pointer(k)
                                for k in ("Raw", "KS", "Table", "Visible", "Out")
                            ),
                            row,
                            tile,
                            width,
                            tiles,
                            maximum,
                            mode,
                        )
            self.assertEqual(set(tl.writes), set(range(rows * width)))
            self.assertEqual(tl.reads, READS[label])
            for row in range(rows):
                for col in range(width):
                    limit = (
                        maximum if label == "no_visible" else buffers["Visible"][row]
                    )
                    want = -float("inf")
                    if col < limit:
                        raw = buffers["Raw"][
                            ((row // 16) * tiles + col // 64) * 1024
                            + (row % 16) * 64
                            + col % 64
                        ]
                        ks = buffers["KS"][
                            buffers["Table"][col // 192] * 192 + col % 192
                        ]
                        want = dict(
                            fill=0.0, raw=raw, ks=ks, sum=raw + ks, no_visible=raw * ks
                        )[label]
                    self.assertEqual(
                        tl.writes[row * width + col], want, (label, row, col)
                    )

    def test_zero_visible_never_reads_raw_key_or_table(self):
        rows, width, tiles, maximum, buffers = fixture()
        buffers["Visible"] = [0] * rows
        for label in ("fill", "raw", "ks", "sum"):
            tl = ScalarTL(buffers)
            fn = device_functions(tl)["raw_finish_prefix_part"]
            tl.column = 0
            fn(
                *(Pointer(k) for k in ("Raw", "KS", "Table", "Visible", "Out")),
                0,
                4,
                width,
                tiles,
                maximum,
                MODES[label],
            )
            self.assertEqual(tl.reads, {"Visible"})
            self.assertEqual(tl.writes[256], -float("inf"))

    def test_dense_copy_loops_cover_output_once_in_both_granularities(self):
        rows, width, _, _, _ = fixture()
        n = rows * width
        data = [float(i % 17 - 8) for i in range(n)]
        for block, label in ((64, "copy64"), (1024, "copy1024")):
            tl = ScalarTL(dict(Dense=data, Out=[None] * n))
            fn = device_functions(tl)["raw_finish_" + label]
            units = rows * ((width + 63) // 64) if block == 64 else (n + 1023) // 1024
            cores = min(20, units)
            for pid in range(cores):
                for c in range(block):
                    tl.pid = pid
                    tl.column = c
                    if block == 64:
                        fn(Pointer("Dense"), Pointer("Out"), rows, width, cores)
                    else:
                        fn(Pointer("Dense"), Pointer("Out"), n, cores)
            self.assertEqual(tl.reads, {"Dense"})
            self.assertEqual(tl.writes, dict(enumerate(data)))

    def test_ir_gate_rejects_eliminated_inputs_and_output(self):
        header = 'func.func @test(%arg0: memref<?xf32> loc("Raw"), %arg1: memref<?xf16> loc("KS"), '
        header += (
            '%arg2: memref<?xi32> loc("Table"), %arg3: memref<?xi32> loc("Visible"), '
        )
        header += '%arg4: memref<?xf32> loc("Out")) {'
        lines = [header]
        for i in range(5):
            lines += [
                f"%v{i} = memref.reinterpret_cast %arg{i} to offset: [0]",
                f"%s{i} = memref.subview %v{i}[0] [64] [1]",
            ]
        lines += [f"%x{i} = memref.load %s{i}[0]" for i in range(4)]
        lines += ["bufferization.materialize_in_destination %tensor in writable %s4"]
        text = "\n".join(lines)
        self.assertTrue(ir_liveness(text, "full")["passed"])
        with self.assertRaisesRegex(RuntimeError, "live-input/sink"):
            ir_liveness(text.replace("%x1 = memref.load %s1[0]", ""), "full")
        with self.assertRaisesRegex(RuntimeError, "live-input/sink"):
            ir_liveness(text.replace(lines[-1], ""), "full")
        with self.assertRaisesRegex(RuntimeError, "live-input/sink"):
            ir_liveness(text, "no_visible")

    def test_ir_gate_follows_copy_and_fill_roots(self):
        text = """func.func @test(%arg0: memref<?xf32> loc("Dense"), %arg1: memref<?xf32> loc("Out")) {
%src = memref.reinterpret_cast %arg0 to offset: [0]
%dst = memref.reinterpret_cast %arg1 to offset: [0]
memref.copy %src, %dst : memref<64xf32> to memref<64xf32>
}"""
        self.assertTrue(ir_liveness(text, "copy1024")["passed"])
        text = text.replace(
            "memref.copy %src, %dst : memref<64xf32> to memref<64xf32>",
            "linalg.copy ins(%src : memref<64xf32>) outs(%dst : memref<64xf32>)",
        )
        self.assertTrue(ir_liveness(text, "copy64")["passed"])
        text = """func.func @test(%arg0: memref<?xi32> loc("Visible"), %arg1: memref<?xf32> loc("Out")) {
%v = memref.load %arg0[0]
linalg.fill ins(%zero : f32) outs(%arg1 : memref<?xf32>)
}"""
        self.assertTrue(ir_liveness(text, "fill")["passed"])

    def test_conditional_differences_preserve_negative_values(self):
        med = dict(
            fill=12, raw=10, ks=15, sum=9, full=8, no_visible=7, copy64=6, copy1024=5
        )
        delta = signed_differences(med)
        self.assertEqual(delta["raw_minus_fill"], -2)
        self.assertEqual(delta["full_minus_sum"], -1)
        self.assertEqual(delta["copy64_minus_copy1024"], 1)


if __name__ == "__main__":
    unittest.main()
