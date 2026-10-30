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

"""Test-only replacement of QSA TopK with the repository Ascend prefill op."""

import hashlib
import importlib
import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import triton
import triton.language as tl


@triton.jit
def _zero_starts(starts, rows: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(starts + offsets, 0, offsets < rows)


def inspect_runtime():
    result = dict(
        python=sys.executable,
        sys_path=list(sys.path),
        triton=dict(version=triton.__version__, source=triton.__file__),
        imports={},
    )
    tle_root = Path(triton.__file__).resolve().parent / "experimental" / "tle"
    result["tle_directory"] = str(tle_root)
    result["tle_python_files"] = [
        str(path.relative_to(tle_root)) for path in sorted(tle_root.rglob("*.py"))[:100]
    ]
    # Retry each original import separately to expose exceptions swallowed upstream.
    for name in (
        "triton.experimental.tle",
        "triton.experimental.tle.language.dsa.ascend.custom_ops",
    ):
        try:
            imported = importlib.import_module(name)
            result["imports"][name] = dict(
                status="passed", source=getattr(imported, "__file__", None)
            )
        except Exception:
            result["imports"][name] = dict(
                status="failed", error=traceback.format_exc()
            )
    try:
        tle = importlib.import_module("triton.experimental.tle")
        raw = tle.dsa.ascend.raw
        result["raw_ops"] = {
            name: hasattr(raw, name)
            for name in ("sort_1d_pack", "unpack_sort", "gather_mask_custom_pattern")
        }
    except Exception:
        result["raw_ops_error"] = traceback.format_exc()
    print("TOPK_IMPORT_DIAGNOSTIC", json.dumps(result), flush=True)
    return result


@contextmanager
def candidate(diagnostics=None):
    import torch
    from qsa_prefill_production_910b import load_production

    _, module, topk, _ = load_production()
    prefill = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.ops.top_k_per_row_prefill"
    )
    identity = dict(
        implementation="Ascend top_k_per_row_prefill",
        source=prefill.__file__,
        sha256=hashlib.sha256(Path(prefill.__file__).read_bytes()).hexdigest(),
        upstream_commit="c9d96bf4a7c3773cdc5a53447c66dcbff8d66698",
        tle_import_ok=prefill._TLE_IMPORT_OK,
        test_only=True,
        adapter="zero row starts via Triton; visible_blocks as row ends",
        timing="includes row-start allocation/fill and all internal prefill allocations/synchronization",
    )
    print("TOPK_CANDIDATE", identity, flush=True)
    runtime = inspect_runtime()
    if diagnostics is not None:
        diagnostics.update(identity=identity, runtime=runtime)
    if not prefill._TLE_IMPORT_OK:
        failures = {
            name: item["error"]
            for name, item in runtime["imports"].items()
            if item["status"] == "failed"
        }
        raise NotImplementedError(
            "Ascend prefill TopK original import gate is false; import failures: "
            + json.dumps(failures)
        )
    original = topk.persistent_topk

    def select(
        logits, lengths, output, workspace, k=512, max_seq_len=None, *, block=None
    ):
        # Keep the production ABI; prefill owns its scratch buffers.
        rows = logits.shape[0]
        if not rows:
            return output
        if max_seq_len is not None and max_seq_len != logits.shape[1]:
            raise ValueError("Candidate requires max_seq_len equal to logits width")
        starts = torch.empty((rows,), device=logits.device, dtype=torch.int32)
        _zero_starts[(triton.cdiv(rows, 256),)](starts, rows, BLOCK=256)
        prefill.top_k_per_row_prefill(
            logits, starts, lengths, output, rows, *logits.stride(), k
        )
        return output

    with patch.object(module, "persistent_topk", select), patch.object(
        topk, "persistent_topk", select
    ), patch.object(
        module, "_qsa_prefill_topk_identity", identity, create=True
    ), patch.object(
        module, "_qsa_persistent_topk_control", original, create=True
    ):
        yield identity


@contextmanager
def persistent_control(module, topk):
    original = module._qsa_persistent_topk_control
    with patch.object(module, "persistent_topk", original), patch.object(
        topk, "persistent_topk", original
    ):
        yield
