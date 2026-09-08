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

"""Split a decode row across programs on MetaX C550.

The generic non-TLE path launches `[(num_rows,)]` -- one program per row, with
no intra-row parallelism at any vocabulary size. On a 104-SM card a one-row
decode therefore runs on a single SM, and the whole 262144-element scan is
serialised inside it. Measured against vLLM's own kernel on this card:

    rows   generic   vLLM      ratio
       1   0.335 ms  0.088 ms  0.27x
       8   0.377 ms  0.098 ms  0.28x
      56   0.531 ms  0.379 ms  0.70x
     496   2.26  ms  2.87  ms  1.27x

The kernel is not the problem: once the grid fills the card we win. Only the
grid is.

Neither stage needs a new kernel. The global top-k of a row is contained in the
union of its chunks' top-k -- if x is in the global top-512 then at most 511
elements of x's own chunk exceed it, so x is in that chunk's top-512 -- so the
existing kernel run over a chunked view IS stage one, and run again over the
gathered candidates IS the merge.

Measured ceiling for the two stages alone (num_rows=1, vocab=262144, top_k=512):

    split  chunk    stage1   gather   merge    total   vs generic
        1  262144   0.3382   0.0174   0.0396   0.3952       0.86x
        8   32768   0.0655   0.0177   0.0416   0.1249       2.72x
       16   16384   0.0420   0.0177   0.0412   0.1009       3.36x
       32    8192   0.0413   0.0179   0.0419   0.1010       3.36x
       64    4096    0.0474  0.0189   0.0611   0.1275       2.66x
      128    2048   0.0552   0.0222   0.1099   0.1874       1.81x

Two things in that table set the policy below. The curve is U-shaped because
each program zeroes and scans NUM_BINS=2048 histogram bins whatever its chunk
size, so once a chunk approaches the bin count the bookkeeping costs more than
the data -- hence MIN_CHUNK. And both stages bottom out near 0.041 ms, which is
this kernel's per-launch floor rather than any amount of work, so more than two
passes cannot pay for itself here.

That table counts one gather and no index bookkeeping, so it is a ceiling, not
a prediction: the real figure for this file is whatever
`tools/vendor_probe.sh tools/run_topk_bench.py` reports.
"""

import functools
import os
from importlib import import_module

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime

_generic = import_module("flaggems_vllm.ops.top_k_per_row_decode")

# A chunk below this makes the 2048-bin histogram cost more than the data it
# summarises; measured as the point where the split curve turns back upward.
MIN_CHUNK = 8192

# What the sweep actually says to hold constant. Chunk size, not program count,
# is the variable: 32768 lands within 2% of the measured optimum at every row
# count from 1 to 56, while the best program count ranged from 32 to 448 and
# tracked nothing. 56 rows at 448 programs on a 104-SM card is optimal, and 24
# rows is better at 192 programs than at 96 -- so "fill one wave", which is
# what this file assumed before the sweep, is simply the wrong model.
TARGET_CHUNK = 32768


@functools.lru_cache(maxsize=32)
def _chunk_starts(split, chunk, device, dtype):
    """First index of each chunk. Rebuilding this per call showed up in the
    profile as a device-to-device Memcpy on every single decode."""
    return torch.arange(split, device=device, dtype=dtype) * chunk


@functools.lru_cache(maxsize=32)
def _merge_lens(num_rows, width, device):
    return torch.full((num_rows,), width, dtype=torch.int32, device=device)


@functools.lru_cache(maxsize=1)
def _sm_count():
    try:
        props = runtime.torch_device_fn.get_device_properties(0)
        return int(getattr(props, "multi_processor_count", 0)) or 1
    except Exception:  # noqa: BLE001 - detection must never break dispatch
        return 1


def _split_forced():
    """FLAGGEMS_METAX_TOPK_SPLIT: 0 disables splitting, n>1 forces that factor.

    0 exists so one benchmark run can measure both arms: between-run drift on
    this box reached 10-18% on shapes whose code did not change, so an "after"
    compared with a "before" from minutes earlier measures the box as much as
    the change.

    A forced factor exists because the automatic rule has never been tuned
    against the fused kernels -- it was fitted when the bookkeeping between the
    two passes still cost more than the passes. `tools/metax_split_sweep.py`
    uses this to measure the real surface before the rule is set from it.
    """
    raw = os.environ.get("FLAGGEMS_METAX_TOPK_SPLIT")
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _split_factor(num_rows, vocab_size):
    """Chunks per row: fill the card, but keep each chunk worth a histogram.

    Only powers of two that divide the vocabulary exactly are considered. An
    inexact split would make the last chunk shorter than the strided view
    claims, and that view would then read past the end of its row.
    """
    forced = _split_forced()
    if forced == 0:
        return 1
    if num_rows < 1 or vocab_size < 2 * MIN_CHUNK:
        return 1

    if forced:
        # Still refuse a factor the view cannot express or the histogram cannot
        # pay for -- a sweep should explore the real surface, not a broken one.
        if vocab_size % forced or vocab_size // forced < MIN_CHUNK:
            return 1
        return forced

    # Past one row per SM the card is already full and splitting only adds
    # passes: measured, 496 rows is fastest unsplit and every split is worse.
    if num_rows >= _sm_count():
        return 1

    # Below that, split down to TARGET_CHUNK.
    split = 1
    while vocab_size % (split * 2) == 0 and vocab_size // (split * 2) >= TARGET_CHUNK:
        split *= 2

    # TARGET_CHUNK alone leaves a one-row call on 8 programs of 104, and that
    # costs real ratio: holding chunk at 32768 there measured 0.921 against
    # vLLM where splitting on down to MIN_CHUNK measured 1.042. So once the
    # card is still not full at the target chunk, keep going to MIN_CHUNK.
    # Chunk governs everywhere else; program count governs only here.
    while (
        split * num_rows < _sm_count()
        and vocab_size % (split * 2) == 0
        and vocab_size // (split * 2) >= MIN_CHUNK
    ):
        split *= 2
    return split


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
    """Candidate values for one row, in one launch.

    Stage one leaves chunk-local indices; the merge wants values. Doing that in
    torch cost a compare, a masked_fill, a dtype cast, a gather and a second
    masked_fill -- five launches whose combined device time exceeded the two
    real kernels they sat between.

    Position p in the candidate array is chunk p // TOPK, slot p % TOPK. A
    padding index (-1, written by stage one for chunks past this row's
    seq_len) becomes `floor` so it cannot survive the merge.
    """
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
    """Candidate positions back to this row's own index space.

    Recomputed from cand_ptr rather than read from a materialised global-index
    array: the array would be another NCAND-wide write and another gather, and
    the arithmetic is two integer ops.
    """
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


def top_k_per_row_decode(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    """Two passes over the existing kernel when one program per row wastes the card."""
    vocab_size = logits.shape[1]
    split = _split_factor(num_rows, vocab_size)

    # next_n != 1 gives each row its own length offset, which the chunked view
    # cannot express. A non-contiguous row layout cannot be re-strided into
    # chunks at all. Either way the generic path is still correct, just slower.
    if (
        split == 1
        or next_n != 1
        or stride1 != 1
        or stride0 != vocab_size
        or top_k > MIN_CHUNK
    ):
        return _generic.top_k_per_row_decode(
            logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
        )

    dev = logits.device
    chunk = vocab_size // split
    n_virtual = num_rows * split

    # --- stage 1: every chunk of every row, in one launch -------------------
    # A chunk that starts past this row's seq_len gets length 0, which the
    # kernel answers with all -1 padding (its row_len <= TOPK branch).
    starts = _chunk_starts(split, chunk, dev, torch.int32)
    sub_lens = (
        (seq_lens.reshape(-1, 1).to(torch.int32) - starts.reshape(1, -1))
        .clamp_(0, chunk)
        .reshape(-1)
    )
    view = logits.as_strided((n_virtual, chunk), (chunk, 1))
    cand_idx = torch.empty((n_virtual, top_k), dtype=torch.int32, device=dev)
    _generic.top_k_per_row_decode(
        view, 1, sub_lens, cand_idx, n_virtual, chunk, 1, top_k
    )

    # --- candidates -> values, one launch ------------------------------------
    ncand = split * top_k
    vals = torch.empty((num_rows, ncand), dtype=logits.dtype, device=dev)
    block = min(1024, triton.next_power_of_2(ncand))
    _gather_candidates[(num_rows, triton.cdiv(ncand, block))](
        logits,
        cand_idx,
        vals,
        stride0,
        stride1,
        torch.finfo(logits.dtype).min,
        SPLIT=split,
        TOPK=top_k,
        CHUNK=chunk,
        NCAND=ncand,
        BLOCK=block,
    )

    # --- stage 2: the same kernel again, over the candidates -----------------
    merged = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
    _generic.top_k_per_row_decode(
        vals, 1, _merge_lens(num_rows, ncand, dev), merged, num_rows, ncand, 1, top_k
    )

    _remap_indices[(num_rows,)](
        cand_idx,
        merged,
        indices,
        SPLIT=split,
        TOPK=top_k,
        CHUNK=chunk,
        BLOCK=triton.next_power_of_2(top_k),
    )
    return indices
