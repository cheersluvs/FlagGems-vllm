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

"""Test unpacking the local sort directly when there is one key block."""

import ast
import re

from qsa_raw_history_merge_910b import source as history_source


def source():
    kernel = next(
        node
        for node in ast.parse(history_source()).body
        if isinstance(node, ast.FunctionDef) and node.name == "raw_history_snapshot"
    )
    kernel.name = "raw_history_single"
    vector = kernel.body[1].body
    loops = {node.target.id: node for node in vector if isinstance(node, ast.For)}
    tile = loops["vector_block"]
    update = next(
        node
        for node in tile.body
        if isinstance(node, ast.For) and node.target.id == "update_query"
    )
    finish = loops["final_query"]
    # Keep the same scoring, sort, and ring handshake. Consume each sorted
    # tile in its own query iteration, before its temporary buffer is reused.
    first_state = next(
        i
        for i, node in enumerate(update.body)
        if isinstance(node, ast.Assign) and node.targets[0].id == "update_state_slot"
    )
    start_unpack = next(
        i
        for i, node in enumerate(finish.body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
    )

    class BindCurrentRow(ast.NodeTransformer):
        def visit_Name(self, node):
            names = {"final_state": "pairs", "final_row": "update_row"}
            if node.id in names:
                node.id = names[node.id]
            return node

    # Move the existing unpack, padding and stores together.
    output = [BindCurrentRow().visit(node) for node in finish.body[start_unpack:]]
    update.body[first_state:-1] = output
    del vector[vector.index(finish)]
    del vector[: vector.index(loops["init_query"]) + 1]
    kernel.body.insert(0, ast.parse("tl.static_assert(WIDTH == 512)").body[0])
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
    if counts != dict(sort32=1, mrgsort=2, unpack_sort=1, merge_exhaust_sort4=0):
        raise RuntimeError("Unexpected single-block sort structure: " + repr(counts))
    if "triton_gather" in text or re.search(
        r"memref<8192xf32, #hivm.address_space<ub>>", text
    ):
        raise RuntimeError("Single-block kernel retains online-history IR")
    return dict(calls=counts, history_merge=False, history_ub=False)
