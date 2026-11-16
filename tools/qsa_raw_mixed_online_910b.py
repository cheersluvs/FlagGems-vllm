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
        for cube_block in range(WIDTH//512):
            cube_slot = cube_block%SLOTS
            al.sync_block_wait("vector","cube",cube_slot,pipe.PIPE_MTE3,pipe.PIPE_FIX)
            for cube_inner in range(8):
                key = kbase+(cube_block*8+cube_inner).to(tl.uint64)*8192
                dest = rbase+(cube_slot*8+cube_inner).to(tl.uint64)*4096
                bits = tl.full((),0x3a800000,tl.int64)
                raw.call(mixed_chain_op,[qbase,key,wbase,bits,dest],output_indices=[])
            al.sync_block_set("cube","vector",cube_slot,pipe.PIPE_FIX,pipe.PIPE_MTE2)
        for drain_slot in tl.static_range(SLOTS):
            al.sync_block_wait("vector","cube",drain_slot,pipe.PIPE_MTE3,pipe.PIPE_FIX)

    with al.scope(core_mode="vector"):
        state_storage = tle.dsa.alloc([8*1024],dtype=tl.float32,mem_addr_space=tle.dsa.ascend.UB)
        word = tl.arange(0,1024)
        initial = tl.where(word%2==0,-float("inf"),tl.full((),-1,tl.int32).to(tl.float32,bitcast=True))
        for init_query in range(8):
            init_state_slot = tle.dsa.subview(state_storage,[init_query*1024],[1024],[1])
            tle.dsa.to_buffer(initial,space=tle.dsa.ascend.UB,bind_buffer=init_state_slot)
        tl.debug_barrier()
        for initial_credit in tl.static_range(SLOTS):
            al.sync_block_set("vector","cube",initial_credit,pipe.PIPE_MTE3,pipe.PIPE_FIX)
        for vector_block in range(WIDTH//512):
            vector_slot = vector_block%SLOTS
            al.sync_block_wait("cube","vector",vector_slot,pipe.PIPE_FIX,pipe.PIPE_MTE2)
            for update_query in range(8):
                update_row = al.sub_vec_id()*8+update_query
                tile = tl.arange(0,8)
                col = tl.arange(0,64)
                lane = tl.arange(0,512)
                received = tl.load(Ring+(vector_slot*8+tile[:,None])*1024+update_row*64+col[None,:])
                ks = tl.load(LogicalKS+vector_block*512+lane).to(tl.float32)
                visible = tl.load(Visible+update_row)
                score = tl.where(vector_block*512+lane<visible,tl.reshape(received,(512,))*ks,-float("inf"))
                if TRACE:
                    tl.store(Trace+update_row*WIDTH+vector_block*512+lane,score)
                pairs = tle.dsa.ascend.raw("sort_1d_pack",score,tl.zeros((2048,),tl.float32),
                    True,512,vector_block*512,SORT_IMPL_BASE,out=tl.zeros((1024,),tl.float32))
                update_state_slot = tle.dsa.subview(state_storage,[update_query*1024],[1024],[1])
                update_state = tle.dsa.to_tensor(update_state_slot)
                update_state = _update_online_topk_ub_snapshot(update_state,pairs,512,512)
                tle.dsa.to_buffer(update_state,space=tle.dsa.ascend.UB,bind_buffer=update_state_slot)
                tl.debug_barrier()
            tl.debug_barrier()
            al.sync_block_set("vector","cube",vector_slot,pipe.PIPE_MTE3,pipe.PIPE_FIX)
        for final_query in range(8):
            final_row = al.sub_vec_id()*8+final_query
            final_state_slot = tle.dsa.subview(state_storage,[final_query*1024],[1024],[1])
            final_state = tle.dsa.to_tensor(final_state_slot)
            values,indices = tle.dsa.ascend.raw("unpack_sort",final_state,512,
                out=[tl.zeros((512,),tl.float32),tl.zeros((512,),tl.int32)])
            rank = tl.arange(0,512)
            valid = rank<tl.load(Visible+final_row)
            tl.store(Indices+final_row*512+rank,tl.where(valid,indices,-1))
            tl.store(Values+final_row*512+rank,tl.where(valid,values,-float("inf")))
"""
    )


def boundary_cases():
    return ((16, 4096), (1, 1088), (15, 1088), (17, 8256), (112, 8512), (128, 29696))


def boundary_geometry(rows, width, slots):
    if rows < 1 or width < 512 or slots not in (1, 2) or (rows + 15) // 16 > 20:
        raise ValueError(
            "Positive rows, width>=512, <=20 groups and one/two slots required"
        )
    groups = (rows + 15) // 16
    steps = (width + 511) // 512
    return dict(
        rows=rows,
        width=width,
        slots=slots,
        groups=groups,
        steps=steps,
        padded_width=steps * 512,
        padded_rows=groups * 16,
        tiles=steps * 8,
        table_columns=(width + 191) // 192,
        ring_elements=groups * slots * 8 * 16 * 64,
        query_bytes=8192,
        weight_bytes=2048,
    )


def padded_pack_source():
    from qsa_fused_ks_pack_probe_910b import source as logical_source
    from qsa_k_pack_2d_probe_910b import source as k_source

    text = k_source() + logical_source()
    tree = ast.parse(text)
    names = ("raw_k_2d_part", "raw_pack_k_ks")
    funcs = {
        n.name: ast.get_source_segment(text, n)
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in names
    }
    result = "\n".join("@triton.jit\n" + funcs[n] for n in names)
    result = result.replace("raw_k_2d_part", "raw_k_2d_part_padded").replace(
        "raw_pack_k_ks", "raw_pack_k_ks_padded"
    )
    if result.count("physical = tl.load(Table+tile//3)") != 2:
        raise RuntimeError("Frozen packing table access changed")
    return result.replace(
        "physical = tl.load(Table+tile//3)",
        "physical = tl.load(Table+tl.where(tile*64<MAX_VISIBLE,tile//3,0),mask=tile*64<MAX_VISIBLE,other=0)",
    )


def boundary_source():
    text = source()
    tree = ast.parse(text)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "raw_mixed_online"
    )
    text = "@triton.jit\n" + ast.get_source_segment(text, fn)
    changes = {
        "def raw_mixed_online(": ("def raw_mixed_boundaries("),
        "WIDTH:tl.constexpr,SLOTS:tl.constexpr,TRACE:tl.constexpr)": (
            "WIDTH:tl.constexpr,SLOTS:tl.constexpr,TRACE:tl.constexpr,ROWS:tl.const"
            "expr)"
        ),
        "range(WIDTH//512)": ("range(triton.cdiv(WIDTH,512))"),
        "rbase = tl.cast(Ring,tl.uint64)": (
            "rbase = tl.cast(Ring,tl.uint64)\n        group_q = qbase+tl.program_id("
            "0).to(tl.uint64)*8192\n        group_w = wbase+tl.program_id(0).to(tl.u"
            "int64)*2048"
        ),
        "(cube_slot*8+cube_inner)": (
            "((tl.program_id(0)*SLOTS+cube_slot)*8+cube_inner)"
        ),
        "[qbase,key,wbase,bits,dest]": ("[group_q,key,group_w,bits,dest]"),
        "update_row = al.sub_vec_id()*8+update_query": (
            "update_local_row = al.sub_vec_id()*8+update_query\n                upda"
            "te_row = tl.program_id(0)*16+update_local_row"
        ),
        "Ring+(vector_slot*8+tile[:,None])*1024+update_row*64": (
            "Ring+((tl.program_id(0)*SLOTS+vector_slot)*8+tile[:,None])*1024+update"
            "_local_row*64"
        ),
        "tl.load(Visible+update_row)": (
            "tl.load(Visible+tl.where(update_row<ROWS,update_row,0),mask=update_row<ROWS,other=0)"
        ),
        "tl.where(vector_block*512+lane<visible,": (
            "tl.where((update_row<ROWS)&(vector_block*512+lane<WIDTH)&(vector_block"
            "*512+lane<visible),"
        ),
        "tl.store(Trace+update_row*WIDTH+vector_block*512+lane,score)": (
            "tl.store(Trace+update_row*WIDTH+vector_block*512+lane,score,mask=(upda"
            "te_row<ROWS)&(vector_block*512+lane<WIDTH))"
        ),
        "final_row = al.sub_vec_id()*8+final_query": (
            "final_row = tl.program_id(0)*16+al.sub_vec_id()*8+final_query"
        ),
        "tl.load(Visible+final_row)": (
            "tl.load(Visible+tl.where(final_row<ROWS,final_row,0),mask=final_row<ROWS,other=0)"
        ),
        "tl.store(Indices+final_row*512+rank,tl.where(valid,indices,-1))": (
            "tl.store(Indices+final_row*512+rank,tl.where(valid,indices,-1),mask=fi"
            "nal_row<ROWS)"
        ),
        'tl.store(Values+final_row*512+rank,tl.where(valid,values,-float("inf")))': (
            'tl.store(Values+final_row*512+rank,tl.where(valid,values,-float("inf")'
            "),mask=final_row<ROWS)"
        ),
    }
    for old, new in changes.items():
        if old not in text:
            raise RuntimeError("Frozen mixed expression changed: " + old)
        text = text.replace(old, new)
    return "\n" + padded_pack_source() + "\n" + text + "\n"


def harness_source():
    text = boundary_source()
    tree = ast.parse(text)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "raw_mixed_boundaries"
    )
    text = "@triton.jit\n" + ast.get_source_segment(text, fn)
    text = text.replace("raw_mixed_boundaries(", "raw_mixed_prepared(")
    text = text.replace("ROWS:tl.constexpr):", "ROWS:tl.constexpr,BLOCKS):")
    text = text.replace("range(triton.cdiv(WIDTH,512))", "range(BLOCKS)")
    # The key count varies each causal chunk: retain one runtime-bound packer.
    pack = (
        padded_pack_source()
        .replace("MAX_VISIBLE:tl.constexpr", "MAX_VISIBLE")
        .replace("TILES:tl.constexpr", "TILES")
    )
    return "\n" + pack + "\n" + text + "\n" + online_topk_source()


def online_topk_source():
    tree = ast.parse(boundary_source())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "raw_mixed_boundaries"
    )
    vector = fn.body[1].body
    body = []
    for stmt in vector:
        # Standalone Vector stage has no ring or cross-core handshakes.
        if isinstance(stmt, ast.For) and stmt.target.id == "initial_credit":
            continue
        if isinstance(stmt, ast.For) and stmt.target.id == "vector_block":
            stmt.iter = ast.parse("range(BLOCKS)", mode="eval").body
            stmt.body = [
                n
                for n in stmt.body
                if not (
                    isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call)
                    and ast.unparse(n.value.func).startswith("al.sync_block_")
                )
            ]
            stmt.body = [
                n
                for n in stmt.body
                if not (isinstance(n, ast.Assign) and n.targets[0].id == "vector_slot")
            ]
            update = next(n for n in stmt.body if isinstance(n, ast.For))
            keep = []
            for n in update.body:
                if isinstance(n, ast.Assign):
                    name = n.targets[0].id
                    if name in ("tile", "col", "received", "ks", "visible"):
                        continue
                    if name == "update_local_row":
                        n.value = ast.parse("update_query", mode="eval").body
                    if name == "update_row":
                        n.value = ast.parse(
                            "tl.program_id(0)*8+update_query", mode="eval"
                        ).body
                    if name == "score":
                        n.value = ast.parse(
                            "tl.load(Staged+update_row*WIDTH+vector_block*512+lane,"
                            "mask=(update_row<ROWS)&(vector_block*512+lane<WIDTH),"
                            'other=-float("inf"))',
                            mode="eval",
                        ).body
                if isinstance(n, ast.If):
                    continue
                keep.append(n)
            update.body = keep
        if isinstance(stmt, ast.For) and stmt.target.id == "final_query":
            row = next(
                n
                for n in stmt.body
                if isinstance(n, ast.Assign) and n.targets[0].id == "final_row"
            )
            row.value = ast.parse("tl.program_id(0)*8+final_query", mode="eval").body
        body.append(stmt)
    shell = ast.parse(
        "def raw_online_topk_from_scores(Staged,Visible,Indices,Values,"
        "ROWS:tl.constexpr,WIDTH:tl.constexpr,BLOCKS): pass"
    )
    shell.body[0].body = body
    return "\n@triton.jit\n" + ast.unparse(ast.fix_missing_locations(shell)) + "\n"
