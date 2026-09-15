"""Run the eager torch_npu baseline in token chunks only where it cannot fit whole.

eager_baseline.py (beside run_eager_isolated.py, in myowncode/) is left exactly
as it is. This wraps it: a shape whose rows (tokens x heads) are at most
UNCHUNKED_MAX_ROWS -- 65536 x 128, the largest the baseline ever completed -- is
passed through in one call, so every number measured so far is untouched. Only
larger shapes, 98304 x 128 and 131072 x 128, are cut into token chunks of
CHUNK_ROWS rows, each a contiguous view of q/kv/slot/positions, so the baseline's
in-place writes land in the caller's tensors.

Chunking a TIMED baseline is only acceptable if what it adds is negligible: each
chunk re-launches the composition's few dozen small kernels. probe_eager_chunk.py
measures that at shapes where both forms fit, before any chunked number is used.
"""

import os
import sys

EAGER_DIR = os.environ.get(
    "EAGER_DIR", os.path.join(os.environ.get("REPO", os.getcwd()), "myowncode"))
sys.path.insert(0, EAGER_DIR)

import eager_baseline  # noqa: E402

UNCHUNKED_MAX_ROWS = int(os.environ.get("EAGER_UNCHUNKED_MAX_ROWS", 65536 * 128))
CHUNK_ROWS = int(os.environ.get("EAGER_CHUNK_ROWS", 32768 * 128))
eager = eager_baseline.eager_fused_deepseek_v4


def eager_chunked(q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, bs,
                  max_rows=None, chunk_rows=None):
    max_rows = UNCHUNKED_MAX_ROWS if max_rows is None else max_rows
    chunk_rows = CHUNK_ROWS if chunk_rows is None else chunk_rows
    n = q.shape[0]
    rows_per_token = q.numel() // (n * q.shape[-1])
    if n * rows_per_token <= max_rows:
        return eager(q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, bs)
    step = max(1, chunk_rows // rows_per_token)
    ins = slot_mapping.shape[0]
    for s in range(0, n, step):
        e = min(n, s + step)
        eager(q[s:e], kv[s:e], k_cache, slot_mapping[min(s, ins):min(e, ins)],
              positions[s:e], cos_sin_cache, eps, bs)
