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
from importlib import import_module

import torch

from flaggems_vllm import runtime

_generic = import_module("flaggems_vllm.ops.top_k_per_row_decode")

# A chunk below this makes the 2048-bin histogram cost more than the data it
# summarises; measured as the point where the split curve turns back upward.
MIN_CHUNK = 8192

# Programs to aim for, as a multiple of the SM count. One wave is enough: the
# measured optimum at num_rows=1 was 16-32 programs, and pushing to 128 was
# slower, so there is nothing to gain from oversubscribing.
WAVES = 1


@functools.lru_cache(maxsize=1)
def _sm_count():
    try:
        props = runtime.torch_device_fn.get_device_properties(0)
        return int(getattr(props, "multi_processor_count", 0)) or 1
    except Exception:  # noqa: BLE001 - detection must never break dispatch
        return 1


def _split_factor(num_rows, vocab_size):
    """Chunks per row: fill the card, but keep each chunk worth a histogram.

    Only powers of two that divide the vocabulary exactly are considered. An
    inexact split would make the last chunk shorter than the strided view
    claims, and that view would then read past the end of its row.
    """
    if num_rows < 1 or vocab_size < 2 * MIN_CHUNK:
        return 1
    want = -(-_sm_count() * WAVES // num_rows)  # ceil
    split = 1
    while (
        split * 2 <= want
        and vocab_size % (split * 2) == 0
        and vocab_size // (split * 2) >= MIN_CHUNK
    ):
        split *= 2
    return split


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
    starts = torch.arange(split, device=dev, dtype=torch.int32) * chunk

    # --- stage 1: every chunk of every row, in one launch -------------------
    # A chunk that starts past this row's seq_len gets length 0, which the
    # kernel answers with all -1 padding (its row_len <= TOPK branch).
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

    # --- candidates, in this row's own index space --------------------------
    # Keep one shape for the whole block. Mixing (rows, split, top_k) with
    # (rows, split * top_k) here does not broadcast, it raises.
    per_chunk = cand_idx.reshape(num_rows, split, top_k)
    keep = (per_chunk >= 0).reshape(num_rows, split * top_k)
    global_idx = torch.where(
        per_chunk >= 0,
        per_chunk + starts.reshape(1, split, 1),
        torch.full_like(per_chunk, -1),
    ).reshape(num_rows, split * top_k)

    # Padding slots must lose the merge outright. finfo.min rather than -inf:
    # the radix pass reads these as bits, and a finite floor needs no special
    # case anywhere downstream.
    vals = torch.gather(view, 1, cand_idx.clamp_min(0).long()).reshape(
        num_rows, split * top_k
    )
    vals = torch.where(keep, vals, torch.finfo(vals.dtype).min).contiguous()

    # --- stage 2: the same kernel again, over the candidates -----------------
    merge_lens = torch.full(
        (num_rows,), split * top_k, dtype=torch.int32, device=dev
    )
    merged = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
    _generic.top_k_per_row_decode(
        vals, 1, merge_lens, merged, num_rows, split * top_k, 1, top_k
    )

    indices.copy_(torch.gather(global_idx, 1, merged.long()).to(torch.int32))
    return indices
