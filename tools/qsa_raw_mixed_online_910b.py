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

"""Generate a pointer-only Cube producer and Vector online TopK consumer."""

import ast
from pathlib import Path


def geometry(width, slots):
    if width < 512 or width % 512 or slots not in (1, 2):
        raise ValueError("Positive whole512 updates and one/two slots required")
    return dict(
        rows=16,
        width=width,
        steps=width // 512,
        slots=slots,
        ring_elements=slots * 8 * 16 * 64,
        query_bytes=8192,
        weight_bytes=2048,
    )


def source():
    original = (
        Path(__file__).with_name("probe_qsa_int8_mixed_pipeline_910b.py").read_text()
    )
    tree = ast.parse(original)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_update_online_topk_ub_snapshot"
    )
    helper = "@triton.jit\n" + ast.get_source_segment(original, fn) + "\n"
    return (
        """
import triton.experimental.tle as tle
from triton.experimental.tle.language.dsa.ascend.custom_ops import SORT_IMPL_BASE
pipe = al.PIPE
"""
        + helper
        + """
@triton.jit
def raw_mixed_online(PQ,PK,PW,LogicalKS,Visible,Ring,Indices,Values,Trace,
                     WIDTH:tl.constexpr,SLOTS:tl.constexpr,TRACE:tl.constexpr):
    with al.scope(core_mode="cube"):
        qbase = tl.cast(PQ,tl.uint64)
        kbase = tl.cast(PK,tl.uint64)
        wbase = tl.cast(PW,tl.uint64)
        rbase = tl.cast(Ring,tl.uint64)
        for block in range(WIDTH//512):
            slot = block%SLOTS
            al.sync_block_wait("vector","cube",slot,pipe.PIPE_MTE3,pipe.PIPE_FIX)
            for inner in range(8):
                key = kbase+(block*8+inner).to(tl.uint64)*8192
                dest = rbase+(slot*8+inner).to(tl.uint64)*4096
                bits = tl.full((),0x3a800000,tl.int64)
                raw.call(chain_op,[qbase,key,wbase,bits,dest],output_indices=[])
            al.sync_block_set("cube","vector",slot,pipe.PIPE_FIX,pipe.PIPE_MTE2)
        for slot in tl.static_range(SLOTS):
            al.sync_block_wait("vector","cube",slot,pipe.PIPE_MTE3,pipe.PIPE_FIX)

    with al.scope(core_mode="vector"):
        state_storage = tle.dsa.alloc([8*1024],dtype=tl.float32,mem_addr_space=tle.dsa.ascend.UB)
        word = tl.arange(0,1024)
        initial = tl.where(word%2==0,-float("inf"),tl.full((),-1,tl.int32).to(tl.float32,bitcast=True))
        for query in range(8):
            state_slot = tle.dsa.subview(state_storage,[query*1024],[1024],[1])
            tle.dsa.to_buffer(initial,space=tle.dsa.ascend.UB,bind_buffer=state_slot)
        tl.debug_barrier()
        for slot in tl.static_range(SLOTS):
            al.sync_block_set("vector","cube",slot,pipe.PIPE_MTE3,pipe.PIPE_FIX)
        for block in range(WIDTH//512):
            slot = block%SLOTS
            al.sync_block_wait("cube","vector",slot,pipe.PIPE_FIX,pipe.PIPE_MTE2)
            for query in range(8):
                row = al.sub_vec_id()*8+query
                tile = tl.arange(0,8)
                col = tl.arange(0,64)
                lane = tl.arange(0,512)
                received = tl.load(Ring+(slot*8+tile[:,None])*1024+row*64+col[None,:])
                ks = tl.load(LogicalKS+block*512+lane).to(tl.float32)
                visible = tl.load(Visible+row)
                score = tl.where(block*512+lane<visible,tl.reshape(received,(512,))*ks,-float("inf"))
                if TRACE:
                    tl.store(Trace+row*WIDTH+block*512+lane,score)
                pairs = tle.dsa.ascend.raw("sort_1d_pack",score,tl.zeros((2048,),tl.float32),
                    True,512,block*512,SORT_IMPL_BASE,out=tl.zeros((1024,),tl.float32))
                state_slot = tle.dsa.subview(state_storage,[query*1024],[1024],[1])
                state = tle.dsa.to_tensor(state_slot)
                state = _update_online_topk_ub_snapshot(state,pairs,512,512)
                tle.dsa.to_buffer(state,space=tle.dsa.ascend.UB,bind_buffer=state_slot)
                tl.debug_barrier()
            tl.debug_barrier()
            al.sync_block_set("vector","cube",slot,pipe.PIPE_MTE3,pipe.PIPE_FIX)
        for query in range(8):
            row = al.sub_vec_id()*8+query
            state_slot = tle.dsa.subview(state_storage,[query*1024],[1024],[1])
            state = tle.dsa.to_tensor(state_slot)
            values,indices = tle.dsa.ascend.raw("unpack_sort",state,512,
                out=[tl.zeros((512,),tl.float32),tl.zeros((512,),tl.int32)])
            rank = tl.arange(0,512)
            valid = rank<tl.load(Visible+row)
            tl.store(Indices+row*512+rank,tl.where(valid,indices,-1))
            tl.store(Values+row*512+rank,tl.where(valid,values,-float("inf")))
"""
    )
