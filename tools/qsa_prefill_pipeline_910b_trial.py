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

"""Standalone score + existing Ascend TopK integration trial, not production."""

import ast
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

_TOPK = None


def load_topk():
    """Load unchanged TopK kernels without importing unrelated package ops.

    The single package import of has_triton_tle is bound directly to the same
    repository utility. Kernel ASTs, decorators and host algorithms are intact.
    This import isolation is only for the standalone probe.
    """
    global _TOPK
    if _TOPK is not None:
        return _TOPK
    root = Path(__file__).resolve().parents[1]
    helper = root / "src/flaggems_vllm/utils/triton_version_utils.py"
    spec = importlib.util.spec_from_file_location("_qsa_910b_version_utils", helper)
    utils = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = utils
    spec.loader.exec_module(utils)
    source = root / "src/flaggems_vllm/runtime/backend/_ascend/ops/persistent_topk.py"
    text = source.read_text()
    tree = ast.parse(text)
    imports = [
        n
        for n in tree.body
        if isinstance(n, ast.ImportFrom)
        and n.module == "flaggems_vllm.utils.triton_version_utils"
    ]
    if len(imports) != 1 or [(n.name, n.asname) for n in imports[0].names] != [
        ("has_triton_tle", None)
    ]:
        raise RuntimeError("Unexpected TopK utility dependency; review loader")
    tree.body.remove(imports[0])
    module = types.ModuleType("_qsa_910b_existing_topk")
    module.__file__ = str(source)
    module.has_triton_tle = utils.has_triton_tle
    sys.modules[module.__name__] = module
    exec(compile(tree, str(source), "exec"), module.__dict__)
    module.probe_identity = dict(
        source=str(source),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        utility=str(helper),
        utility_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        has_tle=module.HAS_TLE_PTK,
        loader="same source AST; only has_triton_tle import resolved directly from repository utility",
    )
    _TOPK = module
    return module


def qsa_select_paged_prefill(
    q,
    cache,
    table,
    starts,
    visible,
    token_topk,
    ratio,
    max_query_len,
    output,
    max_seq_len,
    *,
    chunk_rows=16,
    trace=None,
):
    import torch
    from qsa_prefill_score_910b_trial import score

    if ratio <= 0 or token_topk % ratio or chunk_rows <= 0:
        raise ValueError("Invalid TopK/ratio/chunk size")
    k = token_topk // ratio
    if k not in (512, 1024, 2048) or output.shape != (q.shape[0], k):
        raise ValueError("Invalid group output shape or K")
    if output.dtype != torch.int32 or not output.is_contiguous():
        raise ValueError("Output must be contiguous int32")
    if max_query_len < 0 or max_seq_len < 0:
        raise ValueError("Lengths must be nonnegative")
    if output.device != q.device:
        raise ValueError("Output and Q must share the same NPU")
    rows = q.shape[0]
    if not rows:
        return
    topk = load_topk()
    capacity = table.shape[1] * cache.shape[1]
    width = min(capacity, ((max_seq_len + ratio - 1) // ratio + 63) // 64 * 64)
    if width <= 0:
        raise ValueError("Score width must be positive")
    chunk_rows = min(chunk_rows, rows)
    logits = torch.empty((chunk_rows, width), device=q.device, dtype=torch.float32)
    workspace = torch.empty((1024 * 1024,), device=q.device, dtype=torch.uint8)
    for first in range(0, rows, chunk_rows):
        count = min(chunk_rows, rows - first)
        chunk = logits[:count]
        score(q, cache, table, starts, visible, chunk, offset=first)
        topk.persistent_topk(
            chunk,
            visible[first : first + count].contiguous(),
            output[first : first + count],
            workspace,
            k=k,
            max_seq_len=width,
        )
        if trace is not None:
            trace.append(
                dict(
                    offset=first,
                    rows=count,
                    width=width,
                    logits_pointer=logits.data_ptr(),
                    topk_workspace_pointer=workspace.data_ptr(),
                )
            )
