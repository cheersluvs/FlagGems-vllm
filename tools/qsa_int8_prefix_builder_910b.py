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

"""Create guarded, observable score prefixes for diagnostic timing only."""

import ast
import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

FROZEN = "13718b109b2ffd93164b87c077c9d991b59caaa7f073302f3b7aced74a787334"
PREFIXES = (
    "mask_store",
    "index",
    "k_load",
    "q_load",
    "dot",
    "dot_fp32",
    "query_scaled",
    "dequant",
    "full",
)
FULL_SCALE_VARIANTS = ("key_first", "scale_product")


def assignment_index(body, name):
    found = [
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(found) != 1:
        raise RuntimeError(f"Expected one assignment to {name}: {found}")
    return found[0]


def build_source(source, prefix):
    """Keep a true source prefix and materialize a documented checksum sink."""
    if hashlib.sha256(source.encode()).hexdigest() != FROZEN:
        raise RuntimeError("Frozen source drift; refuse prefix generation")
    tree = ast.parse(source)
    kernel = copy.deepcopy(
        next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_quant_score_trial"
        )
    )
    loops = [n for n in kernel.body if isinstance(n, ast.For)]
    if len(loops) != 1:
        raise RuntimeError("Expected one work loop")
    loop = loops[0]
    if prefix not in (*PREFIXES, *FULL_SCALE_VARIANTS):
        raise ValueError(prefix)
    if prefix in ("key_first", "scale_product"):
        groups = [n for n in loop.body if isinstance(n, ast.For)]
        if len(groups) != 1:
            raise RuntimeError("Expected one original head-group loop")
        group = groups[0]
        target = group.body[assignment_index(group.body, "dequant")]
        expression = {
            "key_first": "dot.to(tl.float32) * key_scale[:, None] * query_scale[None, :]",
            "scale_product": "dot.to(tl.float32) * (query_scale[None, :] * key_scale[:, None])",
        }[prefix]
        target.value = ast.parse(expression, mode="eval").body
    elif prefix != "full":
        stores = [
            n
            for n in loop.body
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "store"
        ]
        if len(stores) != 1:
            raise RuntimeError("Expected one final score store")
        store = copy.deepcopy(stores[0])
        if prefix == "mask_store":
            loop.body = loop.body[: assignment_index(loop.body, "physical")]
            loop.body += ast.parse("scores = tl.full((BLOCK_N,), 0, tl.float32)").body
        elif prefix == "index":
            loop.body = loop.body[: assignment_index(loop.body, "keys")]
            loop.body += ast.parse("scores = token.to(tl.float32)").body
        elif prefix == "k_load":
            loop.body = loop.body[: assignment_index(loop.body, "keys") + 1]
            loop.body += ast.parse(
                "scores = tl.sum(keys.to(tl.int32), axis=1).to(tl.float32)"
            ).body
        else:
            groups = [(i, n) for i, n in enumerate(loop.body) if isinstance(n, ast.For)]
            if len(groups) != 1:
                raise RuntimeError("Expected one head-group loop")
            index, group = groups[0]
            stop = {
                "q_load": "query",
                "dot": "dot",
                "dot_fp32": "dot",
                "query_scaled": "query_scale",
                "dequant": "dequant",
            }[prefix]
            group.body = group.body[: assignment_index(group.body, stop) + 1]
            sink = {
                "q_load": (
                    "scores += (tl.sum(keys.to(tl.int32), axis=1) + "
                    "tl.sum(tl.sum(query.to(tl.int32), axis=0), axis=0)).to(tl.float32)"
                ),
                "dot": "scores += tl.sum(dot, axis=1).to(tl.float32)",
                "dot_fp32": "scores += tl.sum(dot.to(tl.float32), axis=1)",
                "query_scaled": "scores += tl.sum(dot.to(tl.float32) * query_scale[None, :], axis=1)",
                "dequant": "scores += tl.sum(dequant, axis=1)",
            }[prefix]
            group.body += ast.parse(sink).body
            loop.body = loop.body[: index + 1]
        loop.body.append(store)
    module = ast.Module(
        body=ast.parse("import triton\nimport triton.language as tl\n").body + [kernel],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"


def load_prefix(report, case, prefix):
    source = Path(__file__).with_name("qsa_int8_score_910b_trial.py").read_text()
    text = build_source(source, prefix)
    directory = Path(report) / "generated-prefixes"
    directory.mkdir(exist_ok=True)
    path = directory / f"{case}_{prefix}.py"
    path.write_text(text)
    (Path(report) / f"{case}-{prefix}-generated-source.txt").write_text(text)
    name = f"_qsa_prefix_{case}_{prefix}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module._quant_score_trial
