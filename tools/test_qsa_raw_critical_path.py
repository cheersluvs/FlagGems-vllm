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

"""CPU checks for matched events, unchanged work and observable ablation outputs."""

import ast
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from qsa_raw_critical_path_910b import source
from qsa_raw_cube_block512_910b import source as block_source


def functions(text):
    return {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}


def calls(node, name):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == name
    ]


class CriticalTests(unittest.TestCase):
    def test_full_is_identical_and_light_keeps_complete_cube(self):
        fs = functions(source())
        original = functions(block_source())["raw_mixed_block512"]
        full = fs["raw_critical_full"]
        self.assertEqual(
            ast.dump(ast.Module(body=full.body, type_ignores=[])),
            ast.dump(ast.Module(body=original.body, type_ignores=[])),
        )
        self.assertEqual(
            ast.dump(full.body[0]), ast.dump(fs["raw_critical_light"].body[0])
        )
        for fn in fs.values():
            self.assertEqual([a.arg for a in fn.args.args][-2:], ["Replay", "Audit"])

    def test_all_arms_retain_ordered_event_protocol(self):
        fs = functions(source())
        expected = None
        for fn in fs.values():
            events = [
                ast.unparse(n)
                for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and ast.unparse(n.func).startswith("al.sync_block_")
            ]
            if expected is None:
                expected = events
            self.assertEqual(events, expected)
            self.assertEqual(len(events), 6)
        replay = fs["raw_critical_replay"]
        native = calls(replay.body[0], "raw.call")
        self.assertEqual(len(native), 1)
        self.assertEqual(native[0].args[0].id, "replay_tick_op")
        self.assertEqual(len(native[0].args[1].elts), 5)

    def test_replay_changes_only_vector_payload_address(self):
        fs = functions(source())
        replay = fs["raw_critical_replay"].body[1]
        full = fs["raw_critical_full"].body[1]

        class Normalize(ast.NodeTransformer):
            def visit_Assign(self, n):
                if isinstance(n.targets[0], ast.Name) and n.targets[0].id == "received":
                    n.value = ast.Constant(0)
                return n

        self.assertEqual(
            ast.dump(Normalize().visit(replay)), ast.dump(Normalize().visit(full))
        )
        text = source()
        self.assertIn("Replay +", text)
        self.assertIn("replay_tick_op", text)

    def test_native_tick_body_writes_progress_without_score_compute(self):
        clang = shutil.which("clang++")
        if not clang:
            self.skipTest("Host Clang unavailable")
        cpp = Path(__file__).with_name("qsa_raw_critical_path_910b.cpp").read_text()
        shim = r"""
#include <cstdint>
#include <cassert>
#define __aicore__
#define __gm__
namespace AscendC {
enum {PIPE_ALL};
template<int P> void PipeBarrier(){}
template<class T> struct GlobalTensor {
  T* p;
  void SetGlobalBuffer(T* ptr,int){p=ptr;}
  void SetValue(int i,T value){p[i]=value;}
};
}
"""
        body = cpp.replace('#include "qsa_raw_cube_query_pack_910b.cpp"', shim)
        body += """
int main(){int32_t marker=-1;
  _mlir_ciface_qsa_raw_replay_tick(0,0,0,58,(uint64_t)&marker);
  assert(marker==58);
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tick.cpp"
            binary = Path(tmp) / "tick"
            path.write_text(body)
            subprocess.run(
                [clang, "-std=c++17", str(path), "-o", str(binary)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)
        for forbidden in ("Mmad(", "run_block512", "DataCopy("):
            self.assertNotIn(forbidden, cpp)

    def test_light_has_observable_checksum_without_sort_or_state(self):
        light = functions(source())["raw_critical_light"].body[1]
        text = ast.unparse(light)
        for removed in (
            "sort_1d_pack",
            "unpack_sort",
            "_update_online_topk",
            "state_storage",
        ):
            self.assertNotIn(removed, text)
        stores = calls(light, "tl.store")
        audit = [n for n in stores if "Audit" in ast.unparse(n.args[0])]
        self.assertEqual(len(audit), 1)
        self.assertIn("checksum", ast.unparse(audit[0]))
        for branch in [n for n in ast.walk(light) if isinstance(n, ast.If)]:
            self.assertNotIn("Audit", ast.unparse(branch))
        self.assertTrue(calls(light, "tl.load"))
        self.assertTrue(calls(light, "tl.sum"))


if __name__ == "__main__":
    unittest.main()
