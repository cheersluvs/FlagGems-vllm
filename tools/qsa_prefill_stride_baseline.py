# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Benchmark-only stride baseline after retaining the overlap optimization."""

import hashlib
import inspect
import types

BASELINE_COMMIT = "f9d754887a55e068d1275738b9f25ec7822e68e5"
BASELINE_OP_SHA256 = "f11a0d61ce4287e6321ee86a5c1870234cd88e906cb7325a8ccfd9e3106ea14e"
SOURCE_HASHES = {
    "_check_nonoverlapping_strides": "c5093a98e16a3f6f58ead65a5a862ef13e6e3145bdb19735a881a314a648b873",
    "_validate_inputs": "c6d4d33f2de988026e2c51b3d13c3ee75c2c109f65b8fb5eba756486b621ba49",
    "_output_overlaps_inputs": "abfd0e120bc8839579d13974e97b8f596ec28cac6da0c15646d45d77f4926697",
    "qsa_select_paged_prefill": "277945267d171b819dc6c282e41368d259ed3831a26faed8d04b13e7f75e508a",
}


def _check_nonoverlapping_strides(tensor, name):
    extent = 1
    for stride, size in sorted(zip(tensor.stride(), tensor.shape)):
        if size > 1:
            if stride < extent:
                raise NotImplementedError(f"{name} must have nonoverlapping strides")
            extent += (size - 1) * stride


def baseline_entry(op):
    for name, expected in SOURCE_HASHES.items():
        function = (
            _check_nonoverlapping_strides
            if name == "_check_nonoverlapping_strides"
            else getattr(op, name)
        )
        source = inspect.getsource(function).strip()
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            raise RuntimeError(f"Stride-only baseline requires pinned source: {name}")
    original = op.qsa_select_paged_prefill
    namespace = dict(original.__globals__)
    namespace["_check_nonoverlapping_strides"] = _check_nonoverlapping_strides
    validator = op._validate_inputs
    namespace["_validate_inputs"] = types.FunctionType(
        validator.__code__, namespace, validator.__name__, validator.__defaults__
    )
    for name in (
        "_prefill_logits",
        "persistent_topk",
        "_select_all_kernel",
        "_output_overlaps_inputs",
    ):
        if namespace[name] is not getattr(op, name):
            raise RuntimeError(f"A/B must share the same callable: {name}")
    return types.FunctionType(
        original.__code__,
        namespace,
        "qsa_full_stride_baseline",
        original.__defaults__,
        original.__closure__,
    )
