# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Small BF16 QSA scoring compile probe; not a production implementation."""

import torch
import torch_npu  # noqa: F401 -- initialize NPU before importing Triton
import triton
import triton.language as tl


@triton.jit
def _score_probe(q, k, out):
    # A single 64-key tile and four real query heads. Pad heads to 16 for dot.
    dims = tl.arange(0, 128)
    heads = tl.arange(0, 16)
    keys = tl.arange(0, 64)
    query = tl.load(
        q + heads[None, :] * 128 + dims[:, None],
        mask=heads[None, :] < 4,
        other=0.0,
    )
    cache = tl.load(k + keys[:, None] * 128 + dims[None, :])
    scores = tl.dot(cache, query, out_dtype=tl.float32)
    scores = tl.where(heads[None, :] < 4, tl.maximum(scores, 0.0), 0.0)
    tl.store(out + keys, tl.sum(scores, axis=1))


def run(device_index):
    torch.npu.set_device(device_index)
    torch.manual_seed(20261028)
    q_cpu = torch.randn((4, 128), dtype=torch.bfloat16)
    k_cpu = torch.randn((64, 128), dtype=torch.bfloat16)
    expected = (k_cpu.float() @ q_cpu.float().T).relu().sum(dim=1)
    q, k = q_cpu.to("npu"), k_cpu.to("npu")
    output = torch.empty((64,), dtype=torch.float32, device="npu")
    _score_probe[(1,)](q, k, output, num_warps=4)
    torch.npu.synchronize()
    actual = output.cpu()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-2)
    return {
        "status": "passed",
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "input_dtype": str(q.dtype),
        "output_dtype": str(output.dtype),
        "max_abs_error": (actual - expected).abs().max().item(),
        "scope": "one BF16 dot/ReLU/head-sum tile; no paging, TopK or performance claim",
    }
