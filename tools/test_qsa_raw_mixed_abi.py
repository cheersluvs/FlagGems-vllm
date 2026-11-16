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

"""Check mixed symbol signatures and C++ asm-label syntax without a device."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from qsa_raw_mixed_abi_910b import SYMBOL, check_alias_llvm


class MixedAbiTests(unittest.TestCase):
    def definitions(
        self, suffix=".cube", args="i64 %0, i64 %1, i64 %2, i64 %3, i64 %4"
    ):
        return f'define void @{SYMBOL}({args}) {{}}\ndefine void @"{SYMBOL}{suffix}"({args}) {{}}\n'

    def test_normal_and_cube_exports_are_both_required(self):
        result = check_alias_llvm(self.definitions())
        self.assertEqual(result["cube_symbol"], SYMBOL + ".cube")
        for source in (
            self.definitions(suffix=".vector"),
            self.definitions().splitlines()[0],
            self.definitions().splitlines()[1],
        ):
            with self.assertRaises(RuntimeError):
                check_alias_llvm(source)

    def test_abi_gate_rejects_wrong_arity_or_pointer_operands(self):
        for args in (
            "i64 %0, i64 %1",
            "i64 %0, i64 %1, i64 %2, i64 %3, i64* %4",
            "i64 %0, i64 %1, i64 %2, i64 %3, i64 %4, i32 %5",
        ):
            with self.assertRaises(RuntimeError):
                check_alias_llvm(self.definitions(args=args))

    def test_operator_asm_label_exports_cube_name_with_same_five_arguments(self):
        clang = shutil.which("clang++")
        if not clang:
            self.skipTest("Host Clang unavailable; CANN build remains a card gate")
        source = (
            Path(__file__).with_name("qsa_raw_cube_mixed_alias_910b.cpp").read_text()
        )
        stub = """
#define __aicore__
#define QSA_QUERY_PACK 16
#define QSA_SYMBOL(name) name##_query16
typedef unsigned long long uint64_t;
typedef long long int64_t;
extern "C" void _mlir_ciface_qsa_raw_chain_address_query16(uint64_t,uint64_t,uint64_t,int64_t,uint64_t) {}
"""
        source = source.replace('#include "qsa_raw_cube_query_pack_910b.cpp"', stub)
        with tempfile.TemporaryDirectory() as tmp:
            cpp = Path(tmp) / "alias.cpp"
            out = Path(tmp) / "alias.ll"
            cpp.write_text(source)
            subprocess.run(
                [
                    clang,
                    "--target=aarch64-linux-gnu",
                    "-std=c++17",
                    "-O0",
                    "-S",
                    "-emit-llvm",
                    str(cpp),
                    "-o",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            check_alias_llvm(out.read_text())
        # This proves host C++ symbol spelling only; no CANN/device claim.
        self.assertIn("(q, key, weight, scale_bits, out);", source)


if __name__ == "__main__":
    unittest.main()
