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

"""Benchmark for top_k_per_row_decode (DeepSeek V4 decode-phase top-K).

Shapes match DeepSeek V4 production config (vocab=129280, top_k=1024).
The baseline uses vLLM's CUDA kernel when available,
falling back to a pure-PyTorch reference (torch.topk).
"""

import pytest
import torch

import flaggems_vllm

from . import base


pytestmark = pytest.mark.skipif(
    not flaggems_vllm.runtime.torch_device_fn.is_available(),
    reason="accelerator device required",
)

# --- vLLM CUDA baseline (preferred) with PyTorch fallback ---
try:
    import vllm._custom_ops  # noqa: F401 — loads torch.ops._C

    # Importing vLLM is NOT proof the op exists: a non-CUDA build (e.g. MUSA)
    # imports fine but exposes no top_k_per_row_decode, and HAS_VLLM would
    # then be a lie -- the benchmark would report a SpeedUp against a baseline
    # that does not exist. Check for the symbol itself, after the import.
    if not hasattr(torch.ops._C, "top_k_per_row_decode"):
        raise AttributeError("vLLM build exposes no top_k_per_row_decode")

    def _vllm_top_k_per_row_decode(
        logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
    ):
        torch.ops._C.top_k_per_row_decode(
            logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
        )

    HAS_VLLM = True
except (ImportError, AttributeError, RuntimeError):
    # RuntimeError because a misconfigured vLLM should mean "no baseline", not a
    # collection error: with two platform plugins registered it raises
    # "Only one platform plugin can be activated" at import, which aborted
    # collection of this whole file on an MTT box.
    HAS_VLLM = False
    _vllm_top_k_per_row_decode = None


# --- PyTorch baseline, for backends where vLLM exposes no kernel ---
#
# This file previously skipped the whole benchmark when vLLM was absent, so a
# backend without the kernel produced no output and exit code 0 -- indisting-
# uishable from a clean run. torch.topk is exactly equivalent for these shapes
# (next_n == 1 and seq_lens == vocab_size, so every row is the full row).
_DECODE_REF_CHECKED = None


def _torch_topk_decode(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    global _DECODE_REF_CHECKED
    key = (tuple(logits.shape), logits.data_ptr())
    if _DECODE_REF_CHECKED != key:
        assert next_n == 1 and int(seq_lens.min()) == logits.shape[1], (
            "torch.topk baseline assumes next_n=1 and full-length rows; this "
            "shape needs a per-row reference"
        )
        _DECODE_REF_CHECKED = key
    k = min(top_k, logits.shape[1])
    _, idx = torch.topk(logits, k, dim=1, largest=True, sorted=False)
    indices[:, :k] = idx.to(torch.int32)


class TopKPerRowDecodeBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_rows, vocab_size, next_n, top_k, stride0, stride1"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            # DeepSeek-V4-Flash
            (1, 262144, 1, 512, 262144, 1),
            (496, 262144, 1, 512, 262144, 1),
            (512, 262144, 1, 512, 262144, 1),
            (16, 262144, 1, 512, 262144, 1),
            (32, 262144, 1, 512, 262144, 1),
            (48, 262144, 1, 512, 262144, 1),
            (40, 262144, 1, 512, 262144, 1),
            (56, 262144, 1, 512, 262144, 1),
            (4, 262144, 1, 512, 262144, 1),
            (8, 262144, 1, 512, 262144, 1),
            (24, 262144, 1, 512, 262144, 1),
        ]

    def get_input_iter(self, dtype):
        for num_rows, vocab_size, next_n, top_k, stride0, stride1 in self.shapes:
            torch.manual_seed(42)
            buf = torch.randn(
                (num_rows - 1) * stride0 + (vocab_size - 1) * stride1 + 1,
                device=self.device,
                dtype=torch.float32,
            )
            logits = torch.as_strided(buf, (num_rows, vocab_size), (stride0, stride1))

            batch_size = num_rows // next_n
            seq_lens = torch.full(
                (batch_size,), vocab_size, dtype=torch.int32, device=self.device
            )
            indices = torch.zeros(
                (num_rows, top_k), dtype=torch.int32, device=self.device
            )

            yield (
                logits,
                next_n,
                seq_lens,
                indices,
                num_rows,
                stride0,
                stride1,
                top_k,
            )


@pytest.mark.top_k_per_row_decode
def test_top_k_per_row_decode():
    which = ("vLLM top_k_per_row_decode" if HAS_VLLM
             else "torch.topk  (no vLLM kernel on this backend)")
    print(f"\n  baseline = {which}")
    bench = TopKPerRowDecodeBenchmark(
        op_name="top_k_per_row_decode",
        torch_op=_vllm_top_k_per_row_decode if HAS_VLLM else _torch_topk_decode,
        gems_op=flaggems_vllm.top_k_per_row_decode,
        dtypes=[torch.float32],
    )
    bench.run()
