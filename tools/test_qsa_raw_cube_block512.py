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

"""Verify batching preserves addresses, Vector code and native preparation scope."""

import ast
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from qsa_raw_cube_block512_910b import BLOCK_SYMBOL, check_block_llvm, source
from qsa_raw_mixed_online_910b import boundary_geometry, boundary_source
from test_qsa_raw_mixed_online import Index


def function(text, name):
    return next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


class Block512Tests(unittest.TestCase):
    def test_vector_state_events_and_topk_are_unchanged(self):
        old = function(boundary_source(), "raw_mixed_boundaries")
        new = function(source(), "raw_mixed_block512")
        self.assertEqual(ast.dump(old.body[1]), ast.dump(new.body[1]))
        for fn in (old, new):
            cube = fn.body[0]
            self.assertNotIn("tle.dsa.alloc", ast.unparse(cube))
        outer = next(
            n
            for n in ast.walk(new.body[0])
            if isinstance(n, ast.For) and n.target.id == "cube_block"
        )
        self.assertFalse(
            any(isinstance(n, ast.For) for n in ast.walk(outer) if n is not outer)
        )
        calls = [
            n
            for n in ast.walk(outer)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "raw.call"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[0].id, "block512_chain_op")
        self.assertEqual(len(calls[0].args[1].elts), 5)

    def test_actual_block_addresses_cover_old_eight_tile_destinations(self):
        fn = function(source(), "raw_mixed_block512")
        assigns = {
            n.targets[0].id: n.value
            for n in ast.walk(fn.body[0])
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        }
        for rows, width in ((16, 512), (17, 8256), (112, 29696)):
            g = boundary_geometry(rows, width, 2)
            for group in range(g["groups"]):
                for block in range(g["steps"]):
                    scope = dict(
                        qbase=0,
                        kbase=0,
                        wbase=0,
                        rbase=0,
                        SLOTS=2,
                        cube_block=Index(block),
                        cube_slot=Index(block % 2),
                        tl=SimpleNamespace(
                            program_id=lambda _: Index(group), uint64=None
                        ),
                    )
                    key = eval(
                        compile(ast.Expression(assigns["key"]), "key", "eval"),
                        {},
                        scope,
                    )
                    dest = eval(
                        compile(ast.Expression(assigns["dest"]), "dest", "eval"),
                        {},
                        scope,
                    )
                    for tile in range(8):
                        self.assertEqual(key + tile * 8192, (block * 8 + tile) * 8192)
                        self.assertEqual(
                            dest + tile * 4096,
                            ((group * 2 + block % 2) * 8 + tile) * 4096,
                        )
                        self.assertLessEqual(
                            dest + (tile + 1) * 4096, g["ring_elements"] * 4
                        )

    def test_llvm_gate_requires_both_new_exports_and_five_scalar_operands(self):
        args = "i64 %0, i64 %1, i64 %2, i64 %3, i64 %4"
        plain = f"define void @{BLOCK_SYMBOL}({args}) {{}}\n"
        mixed = f'define void @"{BLOCK_SYMBOL}.cube"({args}) {{}}\n'
        self.assertEqual(
            check_block_llvm(plain + mixed)["cube_symbol"], BLOCK_SYMBOL + ".cube"
        )
        for bad in (plain, mixed, plain + mixed.replace("i64 %4", "ptr %4")):
            with self.assertRaises(RuntimeError):
                check_block_llvm(bad)

    def test_compiled_native_body_loads_q_weights_once_and_writes_all_eight_tiles(self):
        clang = shutil.which("clang++")
        if not clang:
            self.skipTest("Host Clang unavailable")
        cpp = Path(__file__).with_name("qsa_raw_cube_block512_910b.cpp").read_text()
        # Execute the actual C++ scheduling body with counting stubs. This checks
        # call/address placement, not CANN instruction semantics or numerical math.
        shim = r"""
#include <cstdint>
#include <vector>
#include <cassert>
#define __aicore__
#define __gm__
#define QSA_QUERY_PACK 16
using half = int16_t;
constexpr int M=64,O=16,K=128,N=64;
static int allocations=0, allocators=0, cubes=0, fixes=0, q_l0=0;
static std::vector<uintptr_t> loads, stores;
struct TPipe { TPipe(){++allocators;} template<class T> void InitBuffer(T&,int){++allocations;} };
enum class TPosition {A1,B1,A2,B2,CO1};
template<class T> struct LocalTensor {template<class U> LocalTensor<U> ReinterpretCast(){return {};}};
template<TPosition P> struct TBuf {template<class T> LocalTensor<T> Get(){return {};}};
template<class T> struct GlobalTensor {T* p; void SetGlobalBuffer(T* ptr,int){p=ptr;}};
template<class T> void DataCopy(LocalTensor<T>,GlobalTensor<T> g,int){loads.push_back((uintptr_t)g.p);}
struct DataCopyCO12DstParams {int mSize,nSize,dstStride,srcStride,quantPre,reluPre,channelSplit,nz2ndEn;};
enum QuantMode_t {DEQF16};
template<class A,class B> void DataCopy(A,B,DataCopyCO12DstParams){++fixes;}
enum class HardEvent {MTE2_MTE1,MTE1_M,M_FIX,FIX_MTE1};
template<HardEvent E> void fence(){}
struct LoadData2DParams {int startIndex,repeatTimes,srcStride,dstGap;bool ifTranspose;};
template<class A,class B> void LoadData(A,B,LoadData2DParams){}
template<class T> void load_matrix(LocalTensor<T>,LocalTensor<T>,int,int,bool){if(sizeof(T)==1)++q_l0;}
struct MmadParams {int m,n,k;bool cmatrixInitVal,cmatrixSource;};
template<class A,class B,class C> void Mmad(A,B,C,MmadParams){++cubes;}
enum {PIPE_M};
template<int P> void PipeBarrier(){}
void SetFixpipePreQuantFlag(uint64_t){}
template<class T,int Rows=M> void store_result(T* p,LocalTensor<T>){stores.push_back((uintptr_t)p);}
"""
        body = cpp.replace('#include "qsa_raw_cube_query_pack_910b.cpp"', shim)
        main = r"""
int main(){
  int8_t q[M*K], keys[8*N*K]; half w[M*O]; float out[8*O*N];
  _mlir_ciface_qsa_raw_block512_address_query16((uint64_t)q,(uint64_t)keys,(uint64_t)w,0,(uint64_t)out);
  assert(allocators==1 && allocations==7 && cubes==16 && fixes==8 && q_l0==8);
  assert(loads.size()==10 && stores.size()==8);
  assert(loads[0]==(uintptr_t)q && loads[1]==(uintptr_t)w);
  for(int t=0;t<8;++t){
    assert(loads[t+2]==(uintptr_t)(keys+t*N*K));
    assert(stores[t]==(uintptr_t)(out+t*O*N));
  }
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.cpp"
            binary = Path(tmp) / "model"
            path.write_text(body + main)
            subprocess.run(
                [clang, "-std=c++17", str(path), "-o", str(binary)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            subprocess.run(
                [str(binary)], check=True, capture_output=True, text=True, timeout=10
            )


if __name__ == "__main__":
    unittest.main()
