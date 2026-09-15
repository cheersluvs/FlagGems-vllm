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

"""top_k_per_row_decode on Hygon BW1000: split low-row decode across programs,
and launch every kernel directly.

WHY. The generic non-TLE path launches one program per row. At vocab 262144
that is ~322 us of device work per row (the kernel is ~19 us + 1.16 ns per
element, and only its STEP 0 ever runs), against vLLM's ~72 us. Below one row
per SM the card is mostly idle -- ratio 0.20-0.28 at 1-16 rows.

Splitting a row into chunks is the obvious answer and was measured as useless
here, but that measurement was WALL time on a host-bound path: every Triton
launch costs ~110 us of host dispatch on this box (device ~2 us when idle), and
a split adds three launches. Launching a cached CompiledKernel directly costs
~12 us instead. With that, the MetaX two-pass split pays in full
(tools/hygon_decode_split_direct.py, vocab 262144, every point correct):

    rows   shipped   best split   ratio vs vLLM
       1     0.224        16          0.968
       4     0.199        16          0.762
       8     0.219         8          0.640
      16     0.276         8          0.595
      32     0.435         4          0.692
      56     0.614         4          0.837

WHAT. Five launches, all direct after the first call of a given plan:

    _chunk_lens   per-chunk valid lengths from seq_lens (a kernel, not torch ops)
    stage 1       the generic kernel over every chunk of every row; the logits
                  are passed as-is with stride0 = CHUNK, so no view is needed
    gather        chunk-local indices -> candidate values, padding -> -FLT_MAX
    merge         the generic kernel again, over split * top_k candidates
    remap         merged positions -> row indices, -1 where the row ran out

The global top-k of a row is contained in the union of its chunks' top-k (x in
the global top-k has at most k-1 larger elements in its own chunk), so both
passes are the existing kernel and nothing algorithmic is new.

DIRECT LAUNCH SAFETY. Triton specialises a compiled kernel on pointer
alignment (data_ptr % 16) and on integer values (== 1, % 16). A cached kernel
must never be launched with arguments it was not specialised for, so the plan
key holds every integer argument and the alignment of each caller tensor.
Internal buffers are fresh allocations. If a Triton version returns no
CompiledKernel from `run`, the plan falls back to ordinary JIT launches.

Rows at or beyond the SM count, next_n != 1, non-unit stride1, a strided row
layout, or a vocabulary the split cannot divide go to the generic operator.
FLAGGEMS_HYGON_TOPK_DECODE_SPLIT=0 disables the split; n > 1 forces a factor.
"""

import functools
import os
import threading
from importlib import import_module

import torch
import triton
import triton.language as tl

_generic = import_module("flaggems_vllm.ops.top_k_per_row_decode")

# Smallest chunk worth a 2048-bin histogram pass.
MIN_CHUNK = 8192

# Measured best split by row count (above). Between measured points the smaller
# neighbouring factor is used.
_SPLIT_BY_ROWS = ((4, 16), (16, 8))
_SPLIT_DEFAULT = 4


@triton.jit
def _chunk_lens(
    seq_lens_ptr,
    out_ptr,
    SPLIT: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < SPLIT
    seq_len = tl.load(seq_lens_ptr + row)
    n = tl.minimum(tl.maximum(seq_len - c * CHUNK, 0), CHUNK)
    tl.store(out_ptr + row * SPLIT + c, n.to(tl.int32), mask=m)


@triton.jit
def _gather_candidates(
    logits_ptr,
    cand_ptr,
    out_ptr,
    stride0,
    stride1,
    floor,
    SPLIT: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    NCAND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Position p in a row's candidate array is chunk p // TOPK, slot p % TOPK.
    A padding index (-1, from a chunk past seq_len) becomes `floor`."""
    row = tl.program_id(0)
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = p < NCAND
    chunk_id = p // TOPK
    local = tl.load(
        cand_ptr + (row * SPLIT + chunk_id) * TOPK + (p % TOPK), mask=m, other=-1
    )
    ok = m & (local >= 0)
    val = tl.load(
        logits_ptr + row * stride0 + (chunk_id * CHUNK + local) * stride1,
        mask=ok,
        other=floor,
    )
    tl.store(out_ptr + row * NCAND + p, tl.where(ok, val, floor), mask=m)


@triton.jit
def _remap_indices(
    cand_ptr,
    merged_ptr,
    out_ptr,
    SPLIT: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    m = j < TOPK
    pos = tl.load(merged_ptr + row * TOPK + j, mask=m, other=0)
    live = m & (pos >= 0)
    chunk_id = pos // TOPK
    local = tl.load(
        cand_ptr + (row * SPLIT + chunk_id) * TOPK + (pos % TOPK),
        mask=live,
        other=-1,
    )
    tl.store(
        out_ptr + row * TOPK + j,
        tl.where(live & (local >= 0), chunk_id * CHUNK + local, -1),
        mask=m,
    )


@functools.lru_cache(maxsize=1)
def _sm_count():
    try:
        props = torch.cuda.get_device_properties(0)
        return int(getattr(props, "multi_processor_count", 0)) or 80
    except Exception:  # noqa: BLE001 - detection must never break dispatch
        return 80


def _forced_split():
    raw = os.environ.get("FLAGGEMS_HYGON_TOPK_DECODE_SPLIT")
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _split_factor(num_rows, vocab_size, top_k):
    forced = _forced_split()
    if forced == 0:
        return 1
    if forced:
        split = forced
    elif num_rows >= _sm_count():
        return 1
    else:
        split = _SPLIT_DEFAULT
        for max_rows, factor in _SPLIT_BY_ROWS:
            if num_rows <= max_rows:
                split = factor
                break
    while split > 1 and (
        vocab_size % split or vocab_size // split < max(MIN_CHUNK, top_k)
    ):
        split //= 2
    return split


class _Launch:
    """One kernel of the pipeline: JIT on first use, direct afterwards."""

    __slots__ = ("jit", "grid", "grid3", "constexprs", "num_warps", "runner")

    def __init__(self, jit, grid, constexprs, num_warps):
        self.jit = jit
        self.grid = grid
        self.grid3 = tuple(grid) + (1,) * (3 - len(grid))
        self.constexprs = constexprs
        self.num_warps = num_warps
        self.runner = None

    def __call__(self, *args):
        if self.runner is not None:
            self.runner(*args, *self.constexprs.values())
            return
        ck = self.jit.run(
            *args,
            **self.constexprs,
            num_warps=self.num_warps,
            grid=self.grid,
            warmup=False,
        )
        if ck is not None:
            self.runner = ck[self.grid3]


class _Plan:
    """Buffers and launchers for one (shape, specialisation) of the split."""

    def __init__(self, dev, dtype, num_rows, vocab, top_k, split):
        chunk = vocab // split
        nv = num_rows * split
        ncand = split * top_k
        gen = _generic
        block = gen.NUM_THREADS_PER_BLOCK
        warps = gen._num_warps(block)
        self.chunk, self.ncand = chunk, ncand
        self.floor = torch.finfo(dtype).min
        self.sub_lens = torch.empty((nv,), dtype=torch.int32, device=dev)
        self.cand = torch.empty((nv, top_k), dtype=torch.int32, device=dev)
        self.vals = torch.empty((num_rows, ncand), dtype=dtype, device=dev)
        self.merged = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
        self.merge_lens = torch.full((num_rows,), ncand, dtype=torch.int32, device=dev)
        self.scratch1 = self._scratch(gen, nv, dev)
        self.scratch2 = self._scratch(gen, num_rows, dev)
        gblock = min(1024, triton.next_power_of_2(ncand))
        self.lens = _Launch(
            _chunk_lens,
            (num_rows,),
            {"SPLIT": split, "CHUNK": chunk, "BLOCK": triton.next_power_of_2(split)},
            1,
        )
        self.stage1 = _Launch(
            gen.non_tle_top_k_per_row_decode,
            (nv,),
            {"TOPK": top_k, "BLOCK_SIZE": block},
            warps,
        )
        self.gather = _Launch(
            _gather_candidates,
            (num_rows, triton.cdiv(ncand, gblock)),
            {
                "SPLIT": split,
                "TOPK": top_k,
                "CHUNK": chunk,
                "NCAND": ncand,
                "BLOCK": gblock,
            },
            4,
        )
        self.merge = _Launch(
            gen.non_tle_top_k_per_row_decode,
            (num_rows,),
            {"TOPK": top_k, "BLOCK_SIZE": block},
            warps,
        )
        self.remap = _Launch(
            _remap_indices,
            (num_rows,),
            {
                "SPLIT": split,
                "TOPK": top_k,
                "CHUNK": chunk,
                "BLOCK": triton.next_power_of_2(top_k),
            },
            4,
        )

    @staticmethod
    def _scratch(gen, n, dev):
        return (
            torch.empty((n, gen.NUM_BINS), dtype=torch.int32, device=dev),
            torch.empty((n, gen.NUM_FILNAL_ITEMS), dtype=torch.float32, device=dev),
            torch.empty((n,), dtype=torch.int32, device=dev),
            torch.empty((n,), dtype=torch.int32, device=dev),
            torch.empty((n,), dtype=torch.int32, device=dev),
            torch.empty((n,), dtype=torch.int32, device=dev),
        )

    def run(self, logits, seq_lens, indices, stride0):
        chunk, ncand = self.chunk, self.ncand
        self.lens(seq_lens, self.sub_lens)
        self.stage1(
            logits, self.cand, self.sub_lens, 1, chunk, 1, chunk, *self.scratch1
        )
        self.gather(logits, self.cand, self.vals, stride0, 1, self.floor)
        self.merge(
            self.vals, self.merged, self.merge_lens, 1, ncand, 1, ncand, *self.scratch2
        )
        self.remap(self.cand, self.merged, indices)


_PLANS = {}
_PLANS_MAX = 32
_LOCK = threading.Lock()


def _aligned(t):
    return t.data_ptr() % 16 == 0


def top_k_per_row_decode(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    """Split low-row decode across programs, launching every kernel directly."""
    vocab_size = logits.shape[1]
    split = _split_factor(num_rows, vocab_size, top_k)
    if (
        split == 1
        or next_n != 1
        or stride1 != 1
        or stride0 != vocab_size
        or logits.dtype != torch.float32
        or seq_lens.dtype != torch.int32
    ):
        return _generic.top_k_per_row_decode(
            logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
        )

    key = (
        logits.device,
        num_rows,
        vocab_size,
        top_k,
        split,
        _aligned(logits),
        _aligned(seq_lens),
        _aligned(indices),
    )
    with _LOCK:
        plan = _PLANS.get(key)
        if plan is None:
            if len(_PLANS) >= _PLANS_MAX:
                _PLANS.pop(next(iter(_PLANS)))
            plan = _PLANS[key] = _Plan(
                logits.device, logits.dtype, num_rows, vocab_size, top_k, split
            )
        plan.run(logits, seq_lens, indices, stride0)
    return indices
