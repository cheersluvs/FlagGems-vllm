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

"""Restricted BF16 H4/page192 chunk32 cases and budget verification."""

import os
from unittest.mock import patch

CASES = {
    **{
        f"chunk_rows{n}": dict(lengths=[n], groups=8192, timing=n == 32)
        for n in (31, 32, 33, 63, 64, 65)
    },
    "chunk_width4096": dict(lengths=[32], groups=4096, timing=True),
    "chunk_width16384": dict(lengths=[32], groups=16384, timing=True),
    "chunk_ragged": dict(lengths=[17, 0, 16], groups=8192, timing=False),
    "chunk_budget_zero": dict(lengths=[3], groups=1024, timing=False, budget_mb=0),
    "chunk_budget_1mb": dict(
        lengths=[17, 0, 16], groups=25600, timing=False, budget_mb=1
    ),
}


def check_budget(args, context, candidate, trial, configure, verify, probe):
    config = CASES[args.worker]
    if "budget_mb" not in config:
        return
    budget = config["budget_mb"]
    first = len(probe["public_launches"])
    try:
        with configure("n128_c32_public_total"), patch.dict(
            os.environ, {"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": str(budget)}
        ):
            candidate.recording = True
            trial(*context["operands"])
    finally:
        candidate.recording = False
    launches = probe["public_launches"][first:]
    cap = min(
        context["rows"], 32, max(1, budget * 1024 * 1024 // (context["width"] * 4))
    )
    expected = [
        (offset, min(cap, context["rows"] - offset))
        for offset in range(0, context["rows"], cap)
    ]
    assert [(item["offset"], item["rows"]) for item in launches] == expected
    for item in launches:
        item["variant"] = "n128_c32_budget_check"
        item["budget_mb"] = budget
    probe["budget_check"] = dict(
        budget_mb=budget,
        chunk_rows=cap,
        launches=launches,
        logits_bytes=cap * context["width"] * 4,
        zero_budget_rule="At least one row, matching production semantics",
        correctness=verify(context["output"]),
    )
    print("CHUNK_BUDGET_CHECK", probe["budget_check"], flush=True)
