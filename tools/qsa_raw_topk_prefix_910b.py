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

"""Observable sort/merge prefixes with explicit fixed-history semantics."""

import ast
import copy

from qsa_raw_critical_path_910b import source as critical_source


def source():
    full = ast.parse(critical_source()).body[0]
    full.name = "raw_topk_full"
    first_config = next(
        i for i, a in enumerate(full.args.args) if a.annotation is not None
    )
    full.args.args.insert(first_config, ast.arg(arg="Seed"))
    result = [full]
    for mode in ("sort", "merge"):
        fn = copy.deepcopy(full)
        fn.name = "raw_topk_" + mode
        vec = fn.body[1].body
        vec[:] = [
            n
            for n in vec
            if not (isinstance(n, ast.For) and n.target.id == "final_query")
        ]
        if mode == "sort":
            vec[:] = [
                n
                for n in vec
                if not (
                    isinstance(n, ast.Assign)
                    and n.targets[0].id in ("state_storage", "word", "initial")
                    or isinstance(n, ast.For)
                    and n.target.id == "init_query"
                )
            ]
        else:
            init = next(
                n for n in vec if isinstance(n, ast.For) and n.target.id == "init_query"
            )
            init.body.insert(
                0,
                ast.parse(
                    "initial = tl.load(Seed+(tl.program_id(0)*16+al.sub_vec_id()*8+init_query)"
                    "*1024+word,mask=tl.program_id(0)*16+al.sub_vec_id()*8+init_query<ROWS,other=0)"
                ).body[0],
            )
            vec[:] = [
                n
                for n in vec
                if not (isinstance(n, ast.Assign) and n.targets[0].id == "initial")
            ]
        outer = next(
            n for n in vec if isinstance(n, ast.For) and n.target.id == "vector_block"
        )
        update = next(
            n
            for n in outer.body
            if isinstance(n, ast.For) and n.target.id == "update_query"
        )
        stop = "pairs" if mode == "sort" else "update_state"
        positions = [
            i
            for i, n in enumerate(update.body)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == stop
        ]
        end = positions[-1] + 1
        value = "pairs" if mode == "sort" else "update_state"
        update.body[end:] = ast.parse(
            f"""
prefix_word = tl.arange(0,1024)
prefix_value = tl.where(prefix_word%2==0,{value},-float("inf"))
prefix_sum = tl.sum(tl.where(prefix_value>-float("inf"),prefix_value,0)*(prefix_word//2+1),0)
tl.store(Audit+update_row*triton.cdiv(WIDTH,512)+vector_block,prefix_sum,mask=update_row<ROWS)
tl.debug_barrier()
"""
        ).body
        result.append(fn)
    return "\n".join(ast.unparse(ast.fix_missing_locations(fn)) for fn in result) + "\n"
