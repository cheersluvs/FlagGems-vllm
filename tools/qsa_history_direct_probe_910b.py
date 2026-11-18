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

"""Test explicit UB merge output binding on the current mixed prefix kernel."""

import ast
import re

from qsa_raw_history_merge_910b import ir_evidence as primitive_evidence
from qsa_raw_history_merge_910b import source as history_source

# The merge may emit 1024 proposals even though the next iteration reads 512.
# Each proposal occupies two FP32 words (score and bitcast int32 index).
DIRECT_UPDATE = """
update_state_slot = tle.dsa.subview(state_storage,[update_query*2048],[2048],[1])
destination = tle.dsa.to_tensor(update_state_slot)
old_prefix = al.extract_slice(destination,(0,),(1024,),(1,))
packed = tl.zeros((4096,),tl.float32)
packed = tle.dsa.insert_slice(packed,old_prefix,(0,),(1024,),(1,))
packed = tle.dsa.insert_slice(packed,pairs,(1024,),(1024,),(1,))
tle.dsa.ascend.raw(
    "merge_exhaust_sort4",packed,2,0,512,1024,1536,512,512,0,0,
    out=[destination,tl.zeros((4,),tl.int32)])
"""


def source():
    original = next(
        node
        for node in ast.parse(history_source()).body
        if isinstance(node, ast.FunctionDef) and node.name == "raw_history_prefix"
    )
    original.name = "raw_history_direct"
    vector = original.body[1].body
    for stmt in vector:
        if isinstance(stmt, ast.Assign):
            name = stmt.targets[0].id
            if name == "state_storage":
                stmt.value.args[0] = ast.parse("[8*2048]", mode="eval").body
            elif name == "word":
                stmt.value.args[1] = ast.Constant(2048)
    loops = {stmt.target.id: stmt for stmt in vector if isinstance(stmt, ast.For)}
    # Initialize the whole destination deterministically. Final unpack sees
    # only the leading 512 proposals, as in the prefix control.
    init_slot = loops["init_query"].body[0].value
    init_slot.args[1:3] = [
        ast.parse("[init_query*2048]", mode="eval").body,
        ast.parse("[2048]", mode="eval").body,
    ]
    final_slot = next(
        stmt.value
        for stmt in loops["final_query"].body
        if isinstance(stmt, ast.Assign) and stmt.targets[0].id == "final_state_slot"
    )
    final_slot.args[1] = ast.parse("[final_query*2048]", mode="eval").body
    update = next(
        stmt
        for stmt in loops["vector_block"].body
        if isinstance(stmt, ast.For) and stmt.target.id == "update_query"
    )
    start = next(
        index
        for index, stmt in enumerate(update.body)
        if isinstance(stmt, ast.Assign) and stmt.targets[0].id == "update_state_slot"
    )
    commit = update.body[start + 3]
    if ast.unparse(commit).split("(")[0] != "tle.dsa.to_buffer":
        raise RuntimeError("Unexpected prefix state commit")
    update.body[start : start + 4] = ast.parse(DIRECT_UPDATE).body
    return "\n" + ast.unparse(ast.fix_missing_locations(original)) + "\n"


def ir_evidence(kernel, direct):
    evidence = primitive_evidence(kernel, complete=False, expect_no_gather=True)
    text = kernel.asm["bcmlir"]
    if isinstance(text, bytes):
        text = text.decode()
    merge = next(line for line in evidence["lines"] if '"merge_exhaust_sort4"' in line)
    unpack = next(line for line in evidence["lines"] if '"unpack_sort"' in line)
    lines = text.splitlines()
    definitions = dict(
        (match[1], match[2])
        for line in lines
        if (match := re.match(r"\s*(%[\w.]+)(?::\d+)? = (.*)", line))
    )
    ssa = r"%[\w.]+(?:#\d+)?"

    def operand(line, pattern):
        match = re.search(pattern + "(" + ssa + ")", line)
        if match is None:
            raise RuntimeError("Expected history IR operand: " + line)
        return match[1]

    def root(value):
        # Follow view/conversion operations only. Stop at copies, insert_slice
        # and custom results so a materialized tensor cannot masquerade as UB.
        while value in definitions:
            op = definitions[value]
            match = re.match(
                r"(?:bufferization.to_tensor|bufferization.to_memref|memref.subview|"
                r"memref.cast|memref.reinterpret_cast|tensor.extract_slice) ("
                + ssa
                + ")",
                op,
            )
            if match is None:
                break
            value = match[1]
        return value

    history = root(operand(unpack, r" ins\("))
    dest = operand(merge, r" outs\(")
    packed = operand(merge, r" ins\(")
    if "tensor<2048xf32>" not in merge or not re.search(
        r"memref<" + str(16384 if direct else 8192) + r"xf32, #hivm.address_space<ub>>",
        definitions.get(history, ""),
    ):
        raise RuntimeError("History allocation or merge output shape missing")
    history_copies = []
    for line in lines:
        if "hivm.hir.copy " in line:
            if root(operand(line, r" outs\(")) == history:
                history_copies.append(line.strip())
        elif "memref.copy " in line:
            if root(operand(line, r"memref\.copy " + ssa + r", ")) == history:
                history_copies.append(line.strip())
    if len(history_copies) != (1 if direct else 2):
        raise RuntimeError(
            "Unexpected number of copies into history: " + repr(history_copies)
        )
    if (root(dest) == history) != direct:
        raise RuntimeError("Merge destination did not match requested history binding")
    if direct:
        packed_op = definitions.get(packed, "")
        old_insert = operand(packed_op, r" into ")
        old_prefix = operand(definitions.get(old_insert, ""), r"tensor\.insert_slice ")
        if root(old_prefix) != history or root(packed) == history:
            raise RuntimeError(
                "Independent packed inputs must snapshot old history before merge"
            )
    return dict(
        **evidence,
        direct_history_destination=direct,
        copies_into_history=history_copies,
        per_query_destination_words=2048 if direct else 1024,
    )
