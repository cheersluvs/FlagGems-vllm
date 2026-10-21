# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Isolate page-table fetch against the exact current production operator."""

import hashlib
import types
from pathlib import Path

PRODUCTION_SHA256 = "f1c5e71a3f7d5975fa639f5081a8e16a334fc3e0bf90d508808f5ed65a716d1c"


def candidate_module(production, *, bf16_only=False):
    from flaggems_vllm.utils import libtuner

    if bf16_only:
        from tools import qsa_prefill_page_table_bf16_kernel as kernel
    else:
        from tools import qsa_prefill_page_table_kernel as kernel

    if (
        hashlib.sha256(Path(production.__file__).read_bytes()).hexdigest()
        != PRODUCTION_SHA256
    ):
        raise RuntimeError("Page-table trial baseline source changed")
    suffix = ".page_table_bf16_trial" if bf16_only else ".page_table_trial"
    module = types.ModuleType(production.__name__ + suffix)
    module.__dict__.update(vars(production))
    module.__name__ = production.__name__ + suffix
    module.__file__ = kernel.__file__
    module.production_baseline = production
    for name, entry in (
        ("_qsa_prefill_score_kernel", kernel._qsa_prefill_score_kernel),
        (
            "_qsa_prefill_compact_score_kernel",
            production._qsa_prefill_compact_score_kernel,
        ),
    ):
        tuner = libtuner(
            configs=list(entry.fn.configs), key=list(entry.fn.keys), strategy="default"
        )(entry.fn.fn)
        cloned = type(entry)(tuner)
        baseline = getattr(production, name)
        if cloned is baseline or any(
            a is b for a, b in zip(cloned.kernel_cache, baseline.kernel_cache)
        ):
            raise RuntimeError("Page-table trial shares a production cache")
        setattr(module, name, cloned)
    for name in ("_prefill_logits", "qsa_select_paged_prefill"):
        original = getattr(production, name)
        cloned = types.FunctionType(
            original.__code__,
            vars(module),
            name,
            original.__defaults__,
            original.__closure__,
        )
        cloned.__kwdefaults__ = original.__kwdefaults__
        setattr(module, name, cloned)
    if module.persistent_topk is not production.persistent_topk:
        raise RuntimeError("Page-table trial changed TopK")
    return module


def acceptance(summary, *, bf16_only=False):
    from tools import qsa_prefill_packed_grid as current

    return current.acceptance(
        summary, expected_candidate="page-table-bf16" if bf16_only else "page-table"
    )
