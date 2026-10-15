# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU differential host execution with captured real launch arguments."""

import contextlib
import importlib.util
import math
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_qsa_prefill_host_validation as host

FROZEN = Path(__file__).with_name("qsa_prefill_chunk_baseline.py")
NAMES = host.NAMES | {
    "_prefill_score_metadata",
    "_prefill_logits",
    "qsa_select_paged_prefill",
}


class Array(host.Tensor):
    def __init__(
        self, shape, dtype="bf16", address=100000, strides=None, label="input"
    ):
        super().__init__(shape, dtype, address, strides)
        self.label = label

    def element_size(self):
        return {"bf16": 2, "fp8": 1, "i32": 4, "i64": 8, "f32": 4, "u8": 1}[self.dtype]

    def __getitem__(self, key):
        start, stop, step = key.indices(self.shape[0])
        assert step == 1
        return Array(
            (max(0, stop - start), *self.shape[1:]),
            self.dtype,
            self.address + start * self.strides[0] * self.element_size(),
            self.strides,
            self.label,
        )

    def view(self, *shape):
        assert math.prod(shape) == self.numel() and self.is_contiguous()
        return Array(shape, self.dtype, self.address, label=self.label)


def normalize(value):
    if isinstance(value, Array):
        return (value.label, value.address, value.shape, value.stride(), value.dtype)
    return value


def load():
    namespace = host.load_functions(host.PRODUCTION, NAMES)
    calls, topks, allocations = [], [], []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                # Grid algebra must agree for every tile variant, not just the
                # one selected by a fake cache. GPU numerics are tested on-card.
                grids = (
                    [
                        grid(dict(TILE_R=r, BLOCK_N=n, K_TILES=t))
                        for r, n, t in (
                            (4, 64, 1),
                            (8, 64, 4),
                            (16, 64, 4),
                            (32, 64, 4),
                            (32, 128, 4),
                            (64, 64, 16),
                        )
                    ]
                    if callable(grid)
                    else grid
                )
                calls.append((grids, tuple(map(normalize, args)), kwargs))

            return launch

    def empty(shape, *, dtype, device):
        result = Array(
            shape,
            dtype,
            1000000000 + len(allocations) * 100000000,
            label=f"allocation_{len(allocations)}",
        )
        allocations.append(result)
        return result

    def topk(*args, **kwargs):
        topks.append((args, kwargs))

    namespace.update(
        torch=types.SimpleNamespace(
            Tensor=Array,
            bfloat16="bf16",
            float8_e4m3fn="fp8",
            int32="i32",
            int64="i64",
            float32="f32",
            uint8="u8",
            empty=empty,
            cuda=types.SimpleNamespace(
                get_device_capability=lambda _: (9, 0),
                device=lambda _: contextlib.nullcontext(),
            ),
        ),
        triton=types.SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y),
        os=os,
        _qsa_prefill_score_kernel=Kernel(),
        _select_all_kernel=Kernel(),
        persistent_topk=topk,
    )
    return namespace, calls, topks, allocations


def inputs(rows=73, dtype="bf16", padded=False, dim=128, heads=4):
    shapes = [
        (rows, heads, dim),
        (1024, 16, 1, dim),
        (2, 512),
        (3,),
        (rows,),
        (rows, 512),
    ]
    tensors = [
        Array(s, dtype if i < 2 else "i32", (i + 1) * 10000000, label=f"input_{i}")
        for i, s in enumerate(shapes)
    ]
    if padded:
        tensors[0].strides = (heads * dim + 64, dim, 1)
        tensors[1].strides = (16 * dim + 64, dim, dim, 1)
        tensors[2].strides = (2048, 2)
    return [*tensors[:5], 2048, 4, 39, tensors[5], 32768]


def baseline_module():
    spec = importlib.util.spec_from_file_location("qsa_chunk_frozen", FROZEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChunkSetupTests(unittest.TestCase):
    def test_old_and_new_host_launches_match_for_chunks_layouts_and_dtypes(self):
        for dtype in ("bf16", "fp8"):
            for budget in ("512", "1", "0"):
                for padded in (False, True):
                    for heads, dim in ((4, 128), (1, 7), (5, 96)):
                        with self.subTest(
                            dtype=dtype,
                            budget=budget,
                            padded=padded,
                            heads=heads,
                            dim=dim,
                        ):
                            new, calls, topks, allocations = load()
                            args = inputs(
                                dtype=dtype, padded=padded, heads=heads, dim=dim
                            )
                            op = types.SimpleNamespace(**new)
                            baseline = baseline_module().baseline_entry(op)
                            original = dict(new)
                            with patch.dict(
                                os.environ, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=budget
                            ):
                                baseline(*args)
                                old_calls = list(calls)
                                old_topks = [
                                    (tuple(map(normalize, a)), kw) for a, kw in topks
                                ]
                                old_bytes = [
                                    x.numel() * x.element_size() for x in allocations
                                ]
                                calls.clear()
                                topks.clear()
                                allocations.clear()
                                new["qsa_select_paged_prefill"](*args)
                            self.assertEqual(calls, old_calls)
                            self.assertEqual(
                                [(tuple(map(normalize, a)), kw) for a, kw in topks],
                                old_topks,
                            )
                            self.assertEqual(
                                [x.numel() * x.element_size() for x in allocations],
                                old_bytes,
                            )
                            self.assertTrue(
                                all(new[k] is v for k, v in original.items())
                            )
                            workspace = allocations[0]
                            chunk_rows = (
                                73 if budget == "512" else 32 if budget == "1" else 1
                            )
                            self.assertEqual(workspace.shape, (chunk_rows, 8192))
                            self.assertEqual(
                                [a[0].shape[0] for a, _ in topks],
                                [
                                    min(chunk_rows, 73 - i)
                                    for i in range(0, 73, chunk_rows)
                                ],
                            )
                            for a, _ in topks:
                                self.assertEqual(a[0].address, workspace.address)
                                if a[0].shape[0] == chunk_rows:
                                    self.assertIs(a[0], workspace)
                                else:
                                    self.assertIsNot(a[0], workspace)

    def test_metadata_is_prepared_once_and_is_fresh_for_each_call(self):
        namespace, _, _, allocations = load()
        prepare = namespace["_prefill_score_metadata"]
        prepared, observed = [], []
        score = namespace["_prefill_logits"]

        def tracked_prepare(*args):
            value = prepare(*args)
            prepared.append(value)
            return value

        def tracked_score(*args, **kwargs):
            observed.append(kwargs["launch_metadata"])
            return score(*args, **kwargs)

        namespace.update(
            _prefill_score_metadata=tracked_prepare, _prefill_logits=tracked_score
        )
        with patch.dict(os.environ, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="0"):
            namespace["qsa_select_paged_prefill"](*inputs())
            self.assertEqual(len(prepared), 1)
            self.assertTrue(all(value is prepared[0] for value in observed))
            observed.clear()
            namespace["qsa_select_paged_prefill"](
                *inputs(padded=True, dtype="fp8", heads=5, dim=96)
            )
        self.assertEqual(len(prepared), 2)
        self.assertIsNot(prepared[0], prepared[1])
        self.assertNotEqual(prepared[0], prepared[1])
        self.assertTrue(all(value is prepared[1] for value in observed))
        self.assertIsNot(allocations[0], allocations[2])

    def test_legacy_flat_private_launcher_matches_frozen_launcher(self):
        namespace, calls, _, _ = load()
        frozen = baseline_module().baseline_entry(types.SimpleNamespace(**namespace))
        old_score = frozen.__globals__["_prefill_logits"]
        args = inputs()
        buffer = Array((32 * 8192,), "f32", 1000000000, label="workspace")
        for count in (1, 9, 32):
            launch = (*args[:5], buffer, args[7], 8192, 32, count)
            old_score(*launch)
            old = calls.pop()
            namespace["_prefill_logits"](*launch)
            self.assertEqual(calls.pop(), old)

    def test_empty_select_all_and_invalid_budget_do_not_prepare_metadata(self):
        namespace, _, _, allocations = load()

        def forbidden(*args):
            self.fail("Unexpected score metadata preparation")

        namespace["_prefill_score_metadata"] = forbidden
        namespace["qsa_select_paged_prefill"](*inputs(rows=0))
        args = inputs(rows=1)
        args[-1] = 17
        namespace["qsa_select_paged_prefill"](*args)
        self.assertEqual(allocations, [])
        for budget in ("-1", "invalid"):
            with patch.dict(os.environ, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=budget):
                with self.assertRaises(ValueError):
                    namespace["qsa_select_paged_prefill"](*inputs())
        self.assertEqual(allocations, [])

    def test_baseline_source_pins_and_shared_kernel_guard(self):
        namespace, _, _, _ = load()
        module = baseline_module()
        op = types.SimpleNamespace(**namespace)
        baseline = module.baseline_entry(op)
        self.assertIsNot(baseline.__code__, op.qsa_select_paged_prefill.__code__)
        self.assertIsNot(baseline.__globals__, namespace)
        self.assertIs(
            baseline.__globals__["_qsa_prefill_score_kernel"],
            op._qsa_prefill_score_kernel,
        )
        self.assertIs(baseline.__globals__["persistent_topk"], op.persistent_topk)
        changed = types.SimpleNamespace(**vars(op))
        changed._validate_inputs = lambda *a: None
        with self.assertRaisesRegex(RuntimeError, "pinned source"):
            module.baseline_entry(changed)
        changed = types.SimpleNamespace(**vars(op))
        changed.persistent_topk = object()
        with self.assertRaisesRegex(RuntimeError, "same callable"):
            module.baseline_entry(changed)
        changed = types.SimpleNamespace(**vars(op))
        changed._prefill_score_metadata = lambda *a: None
        with self.assertRaisesRegex(RuntimeError, "pinned candidate source"):
            module.baseline_entry(changed)


if __name__ == "__main__":
    unittest.main()
