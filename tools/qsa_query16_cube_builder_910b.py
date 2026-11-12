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

"""Clone the selected query16 kernel and replace only its reduction block."""

import ast
import copy
import hashlib
import importlib.util
import inspect
import sys
import textwrap

EXPECTED = """
dequant = dot.to(tl.float32) * query_scale[:, None] * key_scale[None, :]
weighted = tl.maximum(dequant, 0.0) * weights[:, None]
scores = tl.sum(tl.reshape(weighted, (16, 4, BLOCK_N)), axis=1)
"""
REPLACEMENT = """
reduction_weights = tl.where(tl.arange(0, 16)[:, None] == tl.arange(0, 64)[None, :] // 4,
    (query_scale * weights)[None, :], 0.0).to(tl.float16)
reduction_input = (tl.maximum(dot.to(tl.float32), 0.0) * (1.0 / 1024.0)).to(tl.float16)
scores = tl.dot(reduction_weights, reduction_input, out_dtype=tl.float32) * 1024.0 * key_scale[None, :]
"""


HALF_VECTOR = """
reduction_input = (tl.maximum(dot.to(tl.float32), 0.0) * (1.0 / 1024.0)).to(tl.float16).to(tl.float32)
reduction_weights = (query_scale * weights).to(tl.float16).to(tl.float32)
weighted = reduction_input * reduction_weights[:, None]
scores = tl.sum(tl.reshape(weighted, (16, 4, BLOCK_N)), axis=1) * 1024.0 * key_scale[None, :]
"""
QK_SINK = """
scores = tl.sum(tl.reshape(dot.to(tl.float32), (16, 4, BLOCK_N)), axis=1)
"""
REPLACEMENTS = dict(cube=REPLACEMENT, half_vector=HALF_VECTOR, qk_sink=QK_SINK)


def rewrite_reduction(fn, mode="cube"):
    original = copy.deepcopy(fn)
    expected = ast.parse(textwrap.dedent(EXPECTED)).body
    replacement = ast.parse(textwrap.dedent(REPLACEMENTS[mode])).body
    hits = 0
    for parent in ast.walk(fn):
        for field, block in ast.iter_fields(parent):
            if not isinstance(block, list):
                continue
            for index in range(len(block) - 2):
                if all(
                    ast.dump(a) == ast.dump(b)
                    for a, b in zip(block[index : index + 3], expected)
                ):
                    block[index : index + 3] = copy.deepcopy(replacement)
                    hits += 1
    if hits != 1:
        raise ValueError(f"Expected exactly one frozen reduction block, found {hits}")
    return ast.fix_missing_locations(fn), original


def build_reduction_kernel(kernel, report, label, mode="cube"):
    source = textwrap.dedent(inspect.getsource(kernel.fn))
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    changed, original = rewrite_reduction(fn, mode)
    changed.name = f"_qsa_query16_{mode}_reduction"
    module_text = (
        "import triton\nimport triton.language as tl\n\n" + ast.unparse(changed) + "\n"
    )
    directory = report / "generated-reductions"
    directory.mkdir(exist_ok=True)
    path = directory / f"{label}_q16_cube.py"
    path.write_text(module_text)
    (report / f"{label}-q16-cube-source.txt").write_text(module_text)
    (report / f"{label}-q16-original-source.txt").write_text(source)
    module_name = f"_qsa_q16_cube_{label}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, changed.name), dict(
        original_kernel=original.name,
        original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        candidate_source_sha256=hashlib.sha256(module_text.encode()).hexdigest(),
        mode=mode,
        rewrite="only dequant/weighted/scores assignments; query16/page loops/QK/store preserved",
    )
