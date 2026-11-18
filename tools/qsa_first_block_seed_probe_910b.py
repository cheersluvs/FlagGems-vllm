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

"""Seed short online TopK history with the first sorted key block."""

import ast
import copy

from qsa_raw_history_merge_910b import source as history_source


def source():
    kernel = next(
        node
        for node in ast.parse(history_source()).body
        if isinstance(node, ast.FunctionDef) and node.name == "raw_history_snapshot"
    )
    kernel.name = "raw_history_firstseed"
    vector = kernel.body[1].body
    loops = {node.target.id: node for node in vector if isinstance(node, ast.For)}
    tile = loops["vector_block"]
    # Peel the first Vector block so no query needs a branch around its
    # update. The peeled block still consumes and returns slot zero.
    first = copy.deepcopy(tile)
    update = next(
        node
        for node in first.body
        if isinstance(node, ast.For) and node.target.id == "update_query"
    )
    history_load = next(
        i
        for i, node in enumerate(update.body)
        if isinstance(node, ast.Assign) and node.targets[0].id == "update_state"
    )
    load, merge, commit = update.body[history_load : history_load + 3]
    if (
        ast.unparse(load.value.func) != "tle.dsa.to_tensor"
        or ast.unparse(merge.value.func) != "_update_online_topk_ub_snapshot"
        or ast.unparse(commit.value.func) != "tle.dsa.to_buffer"
    ):
        raise RuntimeError("Unexpected snapshot history update")
    commit.value.args[0] = ast.Name(id="pairs", ctx=ast.Load())
    del update.body[history_load : history_load + 2]

    tile.iter.args.insert(0, ast.Constant(1))
    start = vector.index(tile)
    vector[start:start] = (
        ast.parse("vector_block = tl.full((), 0, tl.int32)").body + first.body
    )
    # Keep the allocation. Every slot is written during the peeled block,
    # including rows masked out of the final stores.
    del vector[1 : vector.index(loops["init_query"]) + 1]
    kernel.body.insert(
        0, ast.parse("tl.static_assert(512 < WIDTH and WIDTH <= 1536)").body[0]
    )
    return "\n" + ast.unparse(ast.fix_missing_locations(kernel)) + "\n"


def ir_evidence(kernel):
    text = kernel.asm.get("bcmlir", "")
    if isinstance(text, bytes):
        text = text.decode()
    calls = [line for line in text.splitlines() if "hivm.hir.custom" in line]
    counts = {
        name: sum(f'"{name}"' in line for line in calls)
        for name in ("sort32", "mrgsort", "unpack_sort", "merge_exhaust_sort4")
    }
    expected = dict(sort32=2, mrgsort=4, unpack_sort=1, merge_exhaust_sort4=1)
    if counts != expected:
        raise RuntimeError(
            "Unexpected first-block seed sort structure: " + repr(counts)
        )
    return dict(calls=counts, peeled_first_block=True, retained_snapshot_merge=True)
