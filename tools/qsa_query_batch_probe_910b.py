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

"""Batch local sort primitives while preserving per-query online history."""

import ast
import copy

from qsa_raw_history_merge_910b import source as history_source


def source():
    tree = ast.parse(history_source())
    baseline = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "raw_history_prefix"
    )
    functions = []
    for batch in (2, 4):
        fn = copy.deepcopy(baseline)
        fn.name = "raw_history_batch" + str(batch)
        outer = next(
            n
            for n in ast.walk(fn.body[1])
            if isinstance(n, ast.For) and n.target.id == "vector_block"
        )
        update = next(
            n
            for n in outer.body
            if isinstance(n, ast.For) and n.target.id == "update_query"
        )
        replacement = ast.parse(
            f"""
for query_group in range({8 // batch}):
    lane = tl.arange(0,512)
    local_rows = al.sub_vec_id()*8+query_group*{batch}+tl.arange(0,{batch})
    rows = tl.program_id(0)*16+local_rows
    received = tl.load(
        Ring+((tl.program_id(0)*SLOTS+vector_slot)*8+lane[None,:]//64)*1024
        +local_rows[:,None]*64+lane[None,:]%64)
    ks = tl.load(LogicalKS+vector_block*512+lane).to(tl.float32)
    visible = tl.load(Visible+tl.where(rows<ROWS,rows,0),mask=rows<ROWS,other=0)
    score = tl.where(
        (rows[:,None]<ROWS)&(vector_block*512+lane[None,:]<WIDTH)
        &(vector_block*512+lane[None,:]<visible[:,None]),received*ks[None,:],-float("inf"))
    if TRACE:
        tl.store(Trace+rows[:,None]*WIDTH+vector_block*512+lane[None,:],score,
            mask=(rows[:,None]<ROWS)&(vector_block*512+lane[None,:]<WIDTH))
    flat_lane = tl.arange(0,{batch*512})
    indices = (vector_block*512+flat_lane%512).to(tl.uint32)
    local32 = tle.dsa.ascend.raw("sort32",tl.reshape(score,({batch*512},)),indices,{batch*16},
        out=tl.zeros(({batch*1024},),tl.float32))
    tl.debug_barrier()
    local128 = tle.dsa.ascend.raw("mrgsort",local32,0,32,64,96,32,32,32,32,False,15,{batch*4},
        out=tl.zeros(({batch*1024},),tl.float32))
    tl.debug_barrier()
    sorted_rows = tle.dsa.ascend.raw("mrgsort",local128,0,128,256,384,128,128,128,128,False,15,{batch},
        out=tl.zeros(({batch*1024},),tl.float32))
    tl.debug_barrier()
    for batch_lane in tl.static_range({batch}):
        update_query = query_group*{batch}+batch_lane
        pairs = al.extract_slice(sorted_rows,(batch_lane*1024,),(1024,),(1,))
        update_state_slot = tle.dsa.subview(state_storage,[update_query*1024],[1024],[1])
        update_state = tle.dsa.to_tensor(update_state_slot)
        update_state = _history_prefix(update_state,pairs,512,512)
        tle.dsa.to_buffer(update_state,space=tle.dsa.ascend.UB,bind_buffer=update_state_slot)
        tl.debug_barrier()
"""
        ).body[0]
        outer.body[outer.body.index(update)] = replacement
        functions.append(fn)
    return (
        "\n"
        + "\n".join(ast.unparse(ast.fix_missing_locations(f)) for f in functions)
        + "\n"
    )


def ir_evidence(kernel, batch):
    text = kernel.asm.get("bcmlir", "")
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    lines = [line for line in text.splitlines() if "hivm.hir.custom" in line]
    counts = {
        name: sum('"' + name + '"' in line for line in lines)
        for name in ("sort32", "mrgsort", "merge_exhaust_sort4", "unpack_sort")
    }
    expected = dict(sort32=1, mrgsort=2, merge_exhaust_sort4=batch, unpack_sort=1)
    if counts != expected or "triton_gather" in text:
        raise RuntimeError("Unexpected batched primitive structure: " + repr(counts))
    return dict(
        calls=counts,
        gather_references=0,
        lines=lines,
        scope="Static IR, not dynamic instruction counts or overlap proof",
    )
