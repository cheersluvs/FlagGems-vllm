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

"""BF16 native LightningIndexer adapter for the QSA prefill group-index ABI.

Metadata values must be valid and request boundaries monotone. All adaptation
is device-side; native output is masked, never repaired by a replacement TopK.
This is a test baseline, not the FlagGems production implementation.
"""

import math


def qsa_select_paged_prefill(
    q,
    k_cache,
    page_table,
    query_start_loc,
    visible_blocks,
    token_topk,
    compress_ratio,
    max_query_len,
    block_indices,
    max_seq_len,
    *,
    native=None,
):
    import torch

    if q.device.type != "npu" or q.dtype != torch.bfloat16:
        raise NotImplementedError("910B native baseline requires NPU BF16")
    if q.ndim != 3 or not 1 <= q.shape[1] <= 64 or q.shape[2] != 128:
        raise ValueError("Native baseline requires H<=64 and D=128")
    if k_cache.ndim != 4 or k_cache.shape[2:] != (1, 128):
        raise ValueError("Cache must be [pages,page_size,1,128]")
    if k_cache.dtype != q.dtype:
        raise ValueError("Q and K must have matching BF16 dtypes")
    if not 0 < k_cache.shape[1] <= 1024 or k_cache.shape[1] % 16:
        raise ValueError("Native cache page size must be a multiple of 16, <=1024")
    if compress_ratio <= 0 or token_topk % compress_ratio:
        raise ValueError("Invalid compression ratio or token TopK")
    k = token_topk // compress_ratio
    if k not in (512, 1024, 2048):
        raise ValueError("Baseline group TopK must be 512/1024/2048")
    rows = q.shape[0]
    if page_table.ndim != 2 or query_start_loc.shape != (page_table.shape[0] + 1,):
        raise ValueError("Invalid request metadata shapes")
    if visible_blocks.shape != (rows,) or visible_blocks.dtype != torch.int32:
        raise ValueError("Visibility must be int32 [rows]")
    if block_indices.shape != (rows, k) or block_indices.dtype != torch.int32:
        raise ValueError("Output must be int32 [rows,K]")
    if not block_indices.is_contiguous():
        raise ValueError("Output must be contiguous")
    if max_query_len < 0 or max_seq_len < 0:
        raise ValueError("Lengths must be nonnegative")
    if page_table.dtype not in (
        torch.int32,
        torch.int64,
    ) or query_start_loc.dtype not in (torch.int32, torch.int64):
        raise ValueError("Page table and request starts must be integer metadata")
    if any(
        t.device != q.device
        for t in (k_cache, page_table, query_start_loc, visible_blocks, block_indices)
    ):
        raise ValueError("All tensors must be on the same NPU")
    block_indices.fill_(-1)
    if not rows:
        return
    # Counting request ends handles repeated boundaries and a nonzero base.
    # The temporary [rows,requests] mapping is part of adapter overhead.
    ends = query_start_loc[1:] - query_start_loc[0]
    row_ids = torch.arange(rows, device=q.device, dtype=ends.dtype)
    owner = (row_ids[:, None] >= ends[None, :]).sum(1).long()
    row_table = page_table[owner].to(torch.int32).contiguous()
    cu_query = torch.arange(1, rows + 1, device=q.device, dtype=torch.int32)
    weights = torch.full(
        (rows, q.shape[1]),
        1 / math.sqrt(q.shape[2]),
        device=q.device,
        dtype=torch.float32,
    )
    if native is None:
        import torch_npu

        native = torch_npu.npu_lightning_indexer
    groups, _ = native(
        q.contiguous(),
        k_cache.contiguous(),
        weights,
        actual_seq_lengths_query=cu_query,
        actual_seq_lengths_key=visible_blocks.contiguous(),
        block_table=row_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=k,
        sparse_mode=0,
        pre_tokens=2**63 - 1,
        next_tokens=2**63 - 1,
        return_value=False,
    )
    if groups.shape != (rows, 1, k) or groups.dtype != torch.int32:
        raise RuntimeError("Unexpected native group output")
    groups = groups.squeeze(1)
    block_indices.copy_(
        torch.where((groups >= 0) & (groups < visible_blocks[:, None]), groups, -1)
    )
