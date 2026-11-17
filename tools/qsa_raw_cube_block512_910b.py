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

"""One native call per512-key block with unchanged mixed Vector consumer."""

import ast
import re
from pathlib import Path

from qsa_raw_mixed_abi_910b import SYMBOL, check_alias_llvm
from qsa_raw_mixed_online_910b import boundary_source

BLOCK_SYMBOL = "_mlir_ciface_qsa_raw_block512_address_query16"


def source():
    tree = ast.parse(boundary_source())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "raw_mixed_boundaries"
    )
    fn.name = "raw_mixed_block512"
    cube = fn.body[0].body
    outer = next(
        n for n in cube if isinstance(n, ast.For) and n.target.id == "cube_block"
    )
    inner = next(
        n for n in outer.body if isinstance(n, ast.For) and n.target.id == "cube_inner"
    )

    class SingleBlock(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "cube_inner":
                return ast.copy_location(ast.Constant(0), node)
            if node.id == "mixed_chain_op":
                return ast.copy_location(
                    ast.Name(id="block512_chain_op", ctx=ast.Load()), node
                )
            return node

    pos = outer.body.index(inner)
    outer.body[pos : pos + 1] = [SingleBlock().visit(n) for n in inner.body]
    return "\n" + ast.unparse(ast.fix_missing_locations(fn)) + "\n"


def check_block_llvm(text):
    if not re.search(
        r'^define[^\n]*@"?' + re.escape(BLOCK_SYMBOL) + r'"?\(', text, re.M
    ):
        raise RuntimeError("Block512 ordinary export missing")
    result = check_alias_llvm(text.replace(BLOCK_SYMBOL, SYMBOL))
    result.update(cube_symbol=BLOCK_SYMBOL + ".cube", original_symbol=BLOCK_SYMBOL)
    return result


def bind_block512(report, result, save):
    from probe_qsa_raw_cube_chain_910b import build_bitcode
    from triton.experimental.tle.raw import dialect

    entry = dict(custom_symbols=dict(chain=BLOCK_SYMBOL.removeprefix("_mlir_ciface_")))
    result["block512_native"] = entry
    save()
    path = build_bitcode(
        report,
        entry,
        Path(__file__).with_suffix(".cpp"),
        stem="query16-block512",
        defines=("QSA_QUERY_PACK=16",),
    )
    entry["alias_signature"] = check_block_llvm(
        (report / "query16-block512-plain.ll.txt").read_text()
    )
    structure = entry["bitcode_structure"]
    if not (structure["candidate_structure_present"] and structure["exported_symbols"]):
        raise RuntimeError("Block512 two-Cube structure missing")

    def stub(*args, **kwargs):
        pass

    stub.__name__ = "qsa_raw_block512_mixed_query16"
    bound = dialect(
        name="cann",
        file=str(path),
        format="bitcode",
        extern_func_name=BLOCK_SYMBOL.removeprefix("_mlir_ciface_"),
        pipeline={"core": "cube", "pipe": "PIPE_M"},
    )(stub)
    entry["source_contract"] = dict(
        key_tiles=8,
        q_loads=1,
        weight_loads=1,
        local_allocator_instances=1,
        per_tile_fences_unchanged=True,
    )
    save()
    return bound
