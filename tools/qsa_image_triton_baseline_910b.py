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

"""Audited image score source plus its Torch TopK path, stopping at group indices."""

import ast
import hashlib
import sys
import types
from pathlib import Path

SOURCE_SHA256 = "329ddd6d5f57f38cdbdebf0b7459cf6b69760016215827d6614fbb31d24990bb"
FUNCTIONS = ("_qsa_mqa_paged_kernel", "_validate_mqa", "qsa_mqa_paged")


def load_score():
    import torch
    import torch_npu  # noqa: F401
    import triton
    import triton.language as tl

    source = Path(__file__).with_name("qsa_image_triton_910b_source.txt")
    payload = source.read_bytes()
    assert (
        hashlib.sha256(payload).hexdigest() == SOURCE_SHA256
    ), "Image source snapshot changed"
    parsed = ast.parse(payload.decode())
    functions = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    assert tuple(node.name for node in functions) == FUNCTIONS
    # Keep function ASTs untouched; avoid importing the image's unrelated vLLM modules.
    nodes = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *functions,
    ]
    module = types.ModuleType("_qsa_audited_image_score_910b")
    module.__file__ = str(source)
    module.__dict__.update(
        torch=torch, triton=triton, tl=tl, HAS_TRITON=True, math=__import__("math")
    )
    sys.modules[module.__name__] = module
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            str(source),
            "exec",
        ),
        module.__dict__,
    )
    identity = dict(
        image="quay.io/ascend/vllm-ascend:qwen3.8-next-a3",
        image_index_sha256="2cbb274af050cb5be7db7318e1c461fe22ec0643c54f1ed32aa7266513a7d421",
        image_source="/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/qwen4_exp/qsa.py",
        snapshot=str(source),
        source_sha256=SOURCE_SHA256,
        unchanged_function_ast_sha256={
            node.name: hashlib.sha256(
                ast.dump(node, include_attributes=False).encode()
            ).hexdigest()
            for node in functions
        },
        triton_source=triton.__file__,
        score_function_bodies_modified=False,
        source_loading="Only imports replaced by current torch/FlagTree bindings; selected function ASTs unchanged",
    )
    return module.qsa_mqa_paged, identity


def build(args, context):
    import torch

    score, identity = load_score()
    q, cache, table = context["q"], context["cache"], context["table"]
    owners, positions, sequences = (
        context["owners"],
        context["positions"],
        context["sequence_lengths"],
    )
    rows, k = context["rows"], context["k"]
    capacity = table.shape[1] * cache.shape[1]
    chunk = min(rows, 128, max(1, (128 * 1024**2) // (capacity * 4)))
    chunks = [(first, min(chunk, rows - first)) for first in range(0, rows, chunk)]
    out = torch.empty((rows, k), device=q.device, dtype=torch.int32)
    frozen = torch.empty((rows, capacity), device=q.device, dtype=torch.float32)

    def score_chunk(first, count, owner=owners, pos=positions, seq=sequences):
        return score(
            q[first : first + count],
            cache,
            table,
            owner[first : first + count],
            pos[first : first + count],
            seq,
            4,
        )

    # Isolated TopK uses this baseline's own scaled scores, including capacity padding.
    for first, count in chunks:
        logits, visible = score_chunk(first, count)
        assert torch.equal(visible.cpu(), context["visible_cpu"][first : first + count])
        frozen[first : first + count].copy_(logits)
    for row, expected in context["references"].items():
        actual = frozen[row].cpu()
        length = len(expected)
        torch.testing.assert_close(
            actual[:length], expected / (q.shape[2] ** 0.5), rtol=1e-3, atol=1e-3
        )
        assert torch.isneginf(actual[length:]).all()

    def topk_into(logits, target):
        # Preserve the original torch.topk defaults and int32 conversion/copy.
        target.copy_(torch.topk(logits, k, dim=-1).indices.to(torch.int32))

    def score_only():
        returned = None
        for first, count in chunks:
            returned = score_chunk(first, count)
        return returned

    def topk_only():
        for first, count in chunks:
            topk_into(frozen[first : first + count], out[first : first + count])

    def groups(owner=owners, pos=positions, seq=sequences):
        for first, count in chunks:
            logits, _ = score_chunk(first, count, owner, pos, seq)
            topk_into(logits, out[first : first + count])

    def adapter():
        # Common-ABI diagnostic adapter; not part of the image selector source.
        # Upstream12 requests share a sequence upper bound, as recorded in the manifest.
        starts = context["starts"]
        row_ids = torch.arange(rows, device=q.device, dtype=starts.dtype)
        ends = starts[1:] - starts[0]
        owner = (row_ids[:, None] >= ends[None, :]).sum(1).long()
        begin = starts[owner] - starts[0]
        length = starts[owner + 1] - starts[owner]
        seq = torch.full(
            (table.shape[0],),
            context["max_seq_len"],
            device=q.device,
            dtype=torch.int32,
        )
        pos = (context["max_seq_len"] - length + row_ids - begin).to(torch.int64)
        groups(owner.to(torch.int32), pos, seq)

    identity.update(
        chunk_rows=chunk,
        score_columns=capacity,
        score_divisor=q.shape[2] ** 0.5,
        topk="torch.topk(logits,k,dim=-1), default sorted=True; .indices.to(int32) then caller-output copy",
        output="int32 compressed-group indices; token-index expansion omitted",
        host_adaptation="Group-output adapter writes TopK indices directly to output slices; no extra staging copy",
        metadata_boundary=(
            "groups_prepared uses engine metadata; groups_adapter includes synthetic "
            "common-ABI mapping/positions/sequence metadata"
        ),
        logits_budget_bytes=128 * 1024**2,
        full_score_staging_bytes=rows * capacity * 4,
        applicability="Upstream12 equal query lengths and equal per-request sequence upper bounds",
    )
    jobs = dict(
        image_score=score_only,
        image_topk=topk_only,
        image_groups_prepared=groups,
        image_groups_adapter=adapter,
    )
    return jobs, out, identity
