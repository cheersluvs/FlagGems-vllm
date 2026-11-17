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

"""Compose the fixed32-to128-to512 network with existing registered primitives."""

import ast
import copy

from qsa_raw_cube_block512_910b import source as block_source


def source():
    generic = ast.parse(block_source()).body[0]
    generic.name = "raw_tle_sort_generic"
    fixed = copy.deepcopy(generic)
    fixed.name = "raw_tle_sort_fixed"
    outer = next(
        n
        for n in fixed.body[1].body
        if isinstance(n, ast.For) and n.target.id == "vector_block"
    )
    update = next(
        n
        for n in outer.body
        if isinstance(n, ast.For) and n.target.id == "update_query"
    )
    pos = next(
        i
        for i, n in enumerate(update.body)
        if isinstance(n, ast.Assign) and n.targets[0].id == "pairs"
    )
    update.body[pos : pos + 1] = ast.parse(
        """
local_indices = (vector_block*512+lane).to(tl.uint32)
local32 = tle.dsa.ascend.raw("sort32",score,local_indices,16,out=tl.zeros((1024,),tl.float32))
tl.debug_barrier()
local128 = tle.dsa.ascend.raw("mrgsort",local32,0,32,64,96,32,32,32,32,False,15,4,out=tl.zeros((1024,),tl.float32))
tl.debug_barrier()
pairs = tle.dsa.ascend.raw("mrgsort",local128,0,128,256,384,128,128,128,128,False,15,1,out=tl.zeros((1024,),tl.float32))
tl.debug_barrier()
"""
    ).body
    return (
        "\n".join(ast.unparse(ast.fix_missing_locations(fn)) for fn in (generic, fixed))
        + "\n"
    )


def ir_evidence(kernel, fixed):
    text = kernel.asm.get("bcmlir", "")
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    lines = [line for line in text.splitlines() if "hivm.hir.custom" in line]
    counts = {
        name: sum('"' + name + '"' in line for line in lines)
        for name in (
            "sort_1d_pack",
            "sort32",
            "mrgsort",
            "merge_exhaust_sort4",
            "unpack_sort",
        )
    }
    expected = (
        dict(sort_1d_pack=0, sort32=1, mrgsort=2)
        if fixed
        else dict(sort_1d_pack=1, sort32=0, mrgsort=0)
    )
    if any(counts[name] != count for name, count in expected.items()):
        raise RuntimeError("Unexpected fixed-sort primitive calls: " + repr(counts))
    if counts["merge_exhaust_sort4"] != 1 or counts["unpack_sort"] != 1:
        raise RuntimeError("History merge or final unpack missing")
    return dict(
        calls=counts,
        lines=lines,
        scope="Pre-device IR primitive structure;not hardware instruction counts",
    )


def harness_source():
    from qsa_raw_mixed_online_910b import harness_source as original_harness_source

    common = original_harness_source()
    funcs = {
        n.name: n for n in ast.parse(common).body if isinstance(n, ast.FunctionDef)
    }
    proven = ast.parse(source()).body
    generic = copy.deepcopy(proven[0])
    generic.name = "raw_tle_generic_prepared"
    generic.args = copy.deepcopy(funcs["raw_mixed_prepared"].args)
    for loop in ast.walk(generic):
        if isinstance(loop, ast.For) and loop.target.id in (
            "cube_block",
            "vector_block",
        ):
            loop.iter = ast.parse("range(BLOCKS)", mode="eval").body
    fixed = copy.deepcopy(generic)
    fixed.name = "raw_tle_fixed_prepared"
    online = copy.deepcopy(funcs["raw_online_topk_from_scores"])
    online.name = "raw_tle_generic_topk"
    online_fixed = copy.deepcopy(online)
    online_fixed.name = "raw_tle_fixed_topk"
    reference_update = next(
        n
        for n in ast.walk(proven[1])
        if isinstance(n, ast.For) and n.target.id == "update_query"
    )
    begin = next(
        i
        for i, n in enumerate(reference_update.body)
        if isinstance(n, ast.Assign) and n.targets[0].id == "local_indices"
    )
    end = (
        next(
            i
            for i, n in enumerate(reference_update.body)
            if isinstance(n, ast.Assign) and n.targets[0].id == "pairs"
        )
        + 2
    )
    network = reference_update.body[begin:end]
    for fn in (fixed, online_fixed):
        update = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.For) and n.target.id == "update_query"
        )
        pos = next(
            i
            for i, n in enumerate(update.body)
            if isinstance(n, ast.Assign) and n.targets[0].id == "pairs"
        )
        update.body[pos : pos + 1] = copy.deepcopy(network)
    return (
        common
        + "\n"
        + "\n".join(
            ast.unparse(ast.fix_missing_locations(fn))
            for fn in (generic, fixed, online, online_fixed)
        )
        + "\n"
    )
