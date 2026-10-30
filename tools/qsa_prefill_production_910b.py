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

"""Verify normal Ascend registration and observe public calls without kernel edits."""

import hashlib
import importlib
import inspect
import sys
from pathlib import Path
from unittest.mock import patch


def load_production():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    import torch
    import triton

    import flaggems_vllm

    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.ops.qsa_select_paged_prefill"
    )
    topk = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.ops.persistent_topk"
    )
    backend = importlib.import_module("flaggems_vllm.runtime.backend._ascend.ops")
    assert flaggems_vllm.vendor_name == "ascend", flaggems_vllm.vendor_name
    assert flaggems_vllm.qsa_select_paged_prefill is module.qsa_select_paged_prefill
    assert backend.qsa_select_paged_prefill is module.qsa_select_paged_prefill
    assert module.persistent_topk is topk.persistent_topk
    assert len(inspect.signature(module.qsa_select_paged_prefill).parameters) == 10
    paths = {
        "package": flaggems_vllm.__file__,
        "operator": module.__file__,
        "topk": topk.__file__,
    }
    for source in paths.values():
        assert Path(source).resolve().is_relative_to(root / "src"), source
    identity = dict(
        sources=paths,
        hashes={
            name: hashlib.sha256(Path(source).read_bytes()).hexdigest()
            for name, source in paths.items()
        },
        import_mode="normal package imports and public backend dispatch",
        triton=dict(version=triton.__version__, source=triton.__file__),
    )
    identity["launch_policy"] = dict(
        programs=module._score_programs(torch.npu.current_device()),
        topk_block=module._QSA_TOPK_BLOCK,
        topk_default_block=topk._BLOCK,
    )
    topk.probe_identity = dict(
        source=topk.__file__,
        sha256=identity["hashes"]["topk"],
        has_tle=topk.HAS_TLE_PTK,
        loader="normal package import",
    )
    return flaggems_vllm.qsa_select_paged_prefill, module, topk, identity


def load_topk():
    return load_production()[2]


def qsa_select_paged_prefill(*operands, chunk_rows=16, trace=None):
    public, module, topk, identity = load_production()
    original_score = module._score
    original_topk = module.persistent_topk
    original_owners = module._build_owners
    calls = []
    owner_checks = []

    def owners(q, table, starts):
        result = original_owners(q, table, starts)
        bounds = starts.cpu().tolist()
        expected = [
            sum(row >= end - bounds[0] for end in bounds[1:])
            for row in range(q.shape[0])
        ]
        assert result.cpu().tolist() == expected, "Incorrect Triton request mapping"
        owner_checks.append(True)
        return result

    def score(*args, **kwargs):
        calls.append(
            dict(
                offset=kwargs["offset"],
                rows=args[5].shape[0],
                width=args[5].shape[1],
                logits_pointer=args[5].data_ptr(),
            )
        )
        return original_score(*args, **kwargs)

    def select(*args, **kwargs):
        calls[-1]["topk_workspace_pointer"] = args[3].data_ptr()
        return original_topk(*args, **kwargs)

    # Instrument only Python launch functions; the public API and kernels are intact.
    with patch.object(module, "_score", score), patch.object(
        module, "persistent_topk", select
    ), patch.object(module, "_build_owners", owners):
        returned = public(*operands)
    assert len(owner_checks) == int(operands[0].shape[0] > 0)
    if trace is not None:
        trace.extend(calls)
    return returned


def check_rejections(device):
    import torch

    public = load_production()[0]
    q = torch.empty((1, 4, 128), device=device, dtype=torch.bfloat16)
    cache = torch.empty((4, 192, 1, 128), device=device, dtype=torch.bfloat16)
    table = torch.empty((1, 4), device=device, dtype=torch.int32)
    starts = torch.empty((2,), device=device, dtype=torch.int32)
    visible = torch.empty((1,), device=device, dtype=torch.int32)
    out = torch.empty((1, 512), device=device, dtype=torch.int32)
    args = [q, cache, table, starts, visible, 2048, 4, 1, out, 2048]
    cases = (
        ("ratio_zero", 6, 0, ValueError),
        ("topk_bool", 5, True, TypeError),
        ("topk_not_divisible", 5, 2049, ValueError),
        ("unsupported_k", 5, 256, NotImplementedError),
        ("negative_max_len", 9, -1, ValueError),
        ("fp16", 0, torch.empty_like(q, dtype=torch.float16), NotImplementedError),
        ("dimension64", 0, q[:, :, :64], NotImplementedError),
        ("output_int64", 8, torch.empty_like(out, dtype=torch.int64), ValueError),
    )
    passed = []
    for name, index, value, exception in cases:
        operands = args.copy()
        operands[index] = value
        try:
            public(*operands)
        except exception:
            passed.append(name)
        else:
            raise AssertionError(f"Expected {exception.__name__} for {name}")
    return passed
