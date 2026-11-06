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

"""Prepared INT8 selection adapter for 910B integration validation."""

import torch
import triton
from qsa_int8_score_910b_trial import _quant_score_trial


class PreparedInt8Selection:
    """Reuse prepared inputs, bounded scores and unchanged persistent TopK.

    This trial is not registered as a production operator. Inputs are already
    quantized; row tables and visibility use compressed-group coordinates.
    """

    def __init__(
        self, values, output, scratch, workspace, topk, *, reuse_topk_scratch=False
    ):
        self.values, self.output, self.scratch = values, output, scratch
        self.workspace, self.topk = workspace, topk
        self.reuse_topk_scratch = reuse_topk_scratch
        q, cache = values["q"], values["cache"]
        rows = q.shape[0]
        if q.ndim != 3 or q.shape[1:] != (4, 128) or q.dtype != torch.int8:
            raise NotImplementedError("Validated INT8 adapter requires Q[rows,4,128]")
        if (
            cache.ndim != 4
            or cache.shape[1:] != (192, 1, 128)
            or cache.dtype != torch.int8
        ):
            raise NotImplementedError("Validated INT8 adapter requires page192 K")
        if q.device.type != "npu" or "910B" not in torch.npu.get_device_name(
            q.device.index
        ):
            raise NotImplementedError("910B NPU required")
        expected = {
            "qs": (rows, 4),
            "weights": (rows, 4),
            "ks": cache.shape[:3],
            "visible": (rows,),
        }
        for name, shape in expected.items():
            if values[name].shape != shape:
                raise ValueError(f"Bad shape: {name}")
        for name in ("qs", "ks", "weights"):
            if values[name].dtype != torch.float16:
                raise ValueError(f"FP16 required: {name}")
        table = values["row_table"]
        if (
            table.ndim != 2
            or table.shape[0] != rows
            or table.dtype != torch.int32
            or values["visible"].dtype != torch.int32
        ):
            raise ValueError("INT32 per-query page table and visibility required")
        if scratch.ndim != 2 or scratch.dtype != torch.float32 or scratch.shape[0] < 1:
            raise ValueError("Nonempty FP32 score scratch required")
        if not 0 < scratch.shape[1] <= table.shape[1] * 192:
            raise ValueError("Scratch width outside page table capacity")
        if output.shape != (rows, 512) or output.dtype != torch.int32:
            raise ValueError("INT32 output[rows,512] required")
        if (
            workspace.ndim != 1
            or workspace.dtype != torch.uint8
            or workspace.numel() < 1024 * 1024
        ):
            raise ValueError("At least 1 MiB TopK workspace required")
        tensors = list(values.values()) + [output, scratch, workspace]
        if any(t.device != q.device or not t.is_contiguous() for t in tensors):
            raise NotImplementedError("Contiguous operands on the same NPU required")
        self.rows, self.width, self.chunk = rows, scratch.shape[1], scratch.shape[0]
        self.chunks = [
            (first, min(self.chunk, rows - first))
            for first in range(0, rows, self.chunk)
        ]

    def score_into(self, first, count, target):
        """Launch the unchanged INT8 score kernel with grid20/N128."""
        if (
            first < 0
            or count < 1
            or first + count > self.rows
            or target.shape != (count, self.width)
        ):
            raise ValueError("Invalid score slice")
        if (
            target.dtype != torch.float32
            or target.device != self.scratch.device
            or not target.is_contiguous()
        ):
            raise ValueError("Contiguous FP32 target on the same NPU required")
        v = self.values
        q = v["q"][first : first + count]
        return _quant_score_trial[(min(20, count * triton.cdiv(self.width, 128)),)](
            q,
            v["cache"],
            v["qs"][first : first + count],
            v["ks"],
            v["weights"][first : first + count],
            v["row_table"][first : first + count],
            v["visible"][first : first + count],
            target,
            target,
            HEADS=4,
            PAGE=192,
            COLUMNS=v["row_table"].shape[1],
            ROWS=count,
            WIDTH=self.width,
            BLOCK_N=128,
            MASK_RAW_DOT=True,
            WRITE_RAW_DOT=False,
            num_warps=4,
        )

    def score_all(self):
        for first, count in self.chunks:
            self.score_into(first, count, self.scratch[:count])

    def _new_topk_scratch(self):
        if self.reuse_topk_scratch and self.topk._psegmerge_applicable(
            1, self.width, 512
        ):
            return self.topk._PersistentTopKScratch()
        return None

    def select_chunk(self, first, scores, persistent_scratch=None):
        scratch_kwargs = (
            {}
            if persistent_scratch is None
            else {"persistent_scratch": persistent_scratch}
        )
        for local in range(0, scores.shape[0], 32):
            end = min(scores.shape[0], local + 32)
            self.topk.persistent_topk(
                scores[local:end],
                self.values["visible"][first + local : first + end],
                self.output[first + local : first + end],
                self.workspace,
                k=512,
                max_seq_len=self.width,
                block=256,
                **scratch_kwargs,
            )

    def topk_all(self, staged_scores):
        if (
            staged_scores.shape != (self.rows, self.width)
            or staged_scores.dtype != torch.float32
            or staged_scores.device != self.scratch.device
            or not staged_scores.is_contiguous()
        ):
            raise ValueError(
                "Prepared contiguous FP32 scores required for isolated TopK"
            )
        persistent_scratch = self._new_topk_scratch()
        for first, count in self.chunks:
            self.select_chunk(
                first, staged_scores[first : first + count], persistent_scratch
            )

    def __call__(self, *, persistent_scratch=None):
        if persistent_scratch is None:
            persistent_scratch = self._new_topk_scratch()
        for first, count in self.chunks:
            self.score_into(first, count, self.scratch[:count])
            self.select_chunk(first, self.scratch[:count], persistent_scratch)
        return self.output
