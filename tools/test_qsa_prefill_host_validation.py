# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU metadata differential checks; real Torch alias cases live in tests/."""

import ast
import hashlib
import importlib.util
import inspect
import math
import random
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
FROZEN = ROOT / "tools/qsa_prefill_validation_baseline.py"


class Tensor:
    def __init__(self, shape, dtype="bf16", address=100000, strides=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.address = address
        self.strides = (
            tuple(strides)
            if strides is not None
            else tuple(math.prod(self.shape[i + 1 :]) for i in range(len(shape)))
        )
        self.device = types.SimpleNamespace(type="cuda", index=0)
        self.requires_grad = False

    @property
    def ndim(self):
        return len(self.shape)

    def stride(self, dim=None):
        return self.strides if dim is None else self.strides[dim]

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return {"bf16": 2, "fp8": 1, "i32": 4, "i64": 8}[self.dtype]

    def data_ptr(self):
        return self.address

    def is_contiguous(self):
        if not self.numel():
            return True
        extent = 1
        for size, stride in zip(reversed(self.shape), reversed(self.strides)):
            if size != 1:
                if stride != extent:
                    return False
                extent *= size
        return True


def load_functions(path, names):
    tree = ast.parse(path.read_text())
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    for node in body:
        node.decorator_list = []
    torch = types.SimpleNamespace(
        Tensor=Tensor,
        bfloat16="bf16",
        float8_e4m3fn="fp8",
        int32="i32",
        int64="i64",
        cuda=types.SimpleNamespace(get_device_capability=lambda _: (9, 0)),
    )
    namespace = dict(
        torch=torch,
        runtime=types.SimpleNamespace(
            device=types.SimpleNamespace(vendor_name="nvidia")
        ),
    )
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


NAMES = {
    "_check_nonoverlapping_strides",
    "_overlaps",
    "_output_overlaps_inputs",
    "_validate_inputs",
}
OLD = load_functions(FROZEN, NAMES)
NEW = load_functions(PRODUCTION, NAMES)


def historical_entry_namespace():
    # The earlier trials deliberately pin the pre-preparation public body.
    namespace = load_functions(
        FROZEN.with_name("qsa_prefill_chunk_baseline.py"), {"qsa_select_paged_prefill"}
    )
    namespace.update(load_functions(PRODUCTION, NAMES))
    return namespace


def inputs(rows=3, dtype="bf16", padded=False, k=512):
    tensors = [
        Tensor((rows, 4, 128), dtype, strides=(640, 128, 1) if padded else None),
        Tensor(
            (128, 24, 1, 128), dtype, strides=(3200, 128, 128, 1) if padded else None
        ),
        Tensor((2, 64), "i32", strides=(128, 2) if padded else None),
        Tensor((3,), "i32"),
        Tensor((rows,), "i32"),
        Tensor((rows, k), "i32"),
    ]
    for i, tensor in enumerate(tensors):
        tensor.address = (i + 1) * 20000000
    return [*tensors[:5], k * 4, 4, 33, tensors[-1], 5120]


def outcome(fn, args):
    try:
        return ("ok", fn(*args))
    except Exception as exc:
        return (type(exc).__name__, str(exc))


class ValidationTests(unittest.TestCase):
    def test_random_stride_layouts_preserve_acceptance_and_errors(self):
        rng = random.Random(20261016)
        for _ in range(20000):
            shape = tuple(rng.randrange(6) for _ in range(rng.randrange(5)))
            strides = (
                tuple(rng.randrange(-2, 32) for _ in shape)
                if rng.randrange(2)
                else None
            )
            tensor = Tensor(shape, strides=strides)
            self.assertEqual(
                outcome(NEW["_check_nonoverlapping_strides"], (tensor, "input")),
                outcome(OLD["_check_nonoverlapping_strides"], (tensor, "input")),
                (shape, strides),
            )
        # Empty tensors are contiguous according to Torch, but the original
        # conservative rule still rejects their nontrivial overlapping axis.
        empty = Tensor((0, 3), strides=(0, 0))
        self.assertTrue(empty.is_contiguous())
        self.assertEqual(
            outcome(NEW["_check_nonoverlapping_strides"], (empty, "input")),
            ("NotImplementedError", "input must have nonoverlapping strides"),
        )

    def test_stride_baseline_changes_only_its_private_validator_namespace(self):
        path = FROZEN.with_name("qsa_prefill_stride_baseline.py")
        spec = importlib.util.spec_from_file_location("stride_baseline", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        current = historical_entry_namespace()
        for name in ("_prefill_logits", "persistent_topk", "_select_all_kernel"):
            current[name] = object()
        original_stride = current["_check_nonoverlapping_strides"]
        original_validator = current["_validate_inputs"]
        op = types.SimpleNamespace(
            **{
                k: v
                for k, v in current.items()
                if k in NAMES
                or k
                in {
                    "qsa_select_paged_prefill",
                    "_prefill_logits",
                    "persistent_topk",
                    "_select_all_kernel",
                }
            }
        )
        baseline = module.baseline_entry(op)
        self.assertIs(baseline.__code__, op.qsa_select_paged_prefill.__code__)
        old_validator = baseline.__globals__["_validate_inputs"]
        self.assertIs(old_validator.__code__, original_validator.__code__)
        self.assertIs(old_validator.__globals__, baseline.__globals__)
        self.assertIs(
            old_validator.__globals__["_check_nonoverlapping_strides"],
            module._check_nonoverlapping_strides,
        )
        self.assertIs(current["_check_nonoverlapping_strides"], original_stride)
        self.assertIs(current["_validate_inputs"], original_validator)
        self.assertIsNot(baseline.__globals__, current)
        for name in (
            "_prefill_logits",
            "persistent_topk",
            "_select_all_kernel",
            "_output_overlaps_inputs",
        ):
            self.assertIs(baseline.__globals__[name], current[name])
        for rows in (0, 1, 36):
            for padded in (False, True):
                args = inputs(rows=rows, padded=padded)
                self.assertEqual(
                    outcome(old_validator, args), outcome(original_validator, args)
                )
        for name in (
            "_validate_inputs",
            "_output_overlaps_inputs",
            "qsa_select_paged_prefill",
        ):
            mutated = types.SimpleNamespace(**vars(op))
            setattr(mutated, name, outcome)
            with self.assertRaisesRegex(RuntimeError, "pinned source"):
                module.baseline_entry(mutated)
        mutated = types.SimpleNamespace(**vars(op))
        mutated.qsa_select_paged_prefill = load_functions(
            PRODUCTION, {"qsa_select_paged_prefill"}
        )["qsa_select_paged_prefill"]
        with self.assertRaisesRegex(RuntimeError, "pinned source"):
            module.baseline_entry(mutated)

    def test_twenty_thousand_byte_hull_cases_match_old_checker(self):
        rng = random.Random(20261015)
        for _ in range(20000):
            output = Tensor(
                (rng.randrange(8), rng.randrange(13)),
                "i32",
                100000 + rng.randrange(256),
            )
            others = []
            for _ in range(5):
                shape = tuple(rng.randrange(6) for _ in range(rng.randrange(1, 5)))
                strides = (
                    tuple(rng.randrange(32) for _ in shape)
                    if rng.randrange(2)
                    else None
                )
                others.append(
                    Tensor(
                        shape,
                        rng.choice(["bf16", "fp8", "i32", "i64"]),
                        99800 + rng.randrange(1000),
                        strides,
                    )
                )
            self.assertEqual(
                NEW["_output_overlaps_inputs"](output, others),
                any(OLD["_overlaps"](output, tensor) for tensor in others),
            )

    def test_validator_results_and_exceptions_match(self):
        for rows in (0, 1, 36):
            for dtype in ("bf16", "fp8"):
                for padded in (False, True):
                    for k in (512, 1024, 2048):
                        args = inputs(rows, dtype, padded, k)
                        self.assertEqual(
                            outcome(NEW["_validate_inputs"], args),
                            outcome(OLD["_validate_inputs"], args),
                        )
                        for index in range(5):
                            mutated = inputs(rows, dtype, padded, k)
                            mutated[8].address = mutated[index].address
                            self.assertEqual(
                                outcome(NEW["_validate_inputs"], mutated),
                                outcome(OLD["_validate_inputs"], mutated),
                            )
        mutations = [
            lambda a: setattr(a[0], "requires_grad", True),
            lambda a: setattr(a[1], "dtype", "fp8"),
            lambda a: setattr(a[2], "dtype", "bf16"),
            lambda a: setattr(a[4], "strides", (0,)),
            lambda a: setattr(a[8], "strides", (1024, 2)),
            lambda a: a.__setitem__(6, True),
            lambda a: a.__setitem__(9, -1),
            lambda a: setattr(a[0], "strides", (0, 128, 1)),
        ]
        for mutate in mutations:
            args = inputs()
            mutate(args)
            self.assertEqual(
                outcome(NEW["_validate_inputs"], args),
                outcome(OLD["_validate_inputs"], args),
            )

    def test_public_entry_source_hash_and_shared_kernel_builder(self):
        source = FROZEN.read_text()
        constants = {
            n.targets[0].id: ast.literal_eval(n.value)
            for n in ast.parse(source).body
            if isinstance(n, ast.Assign)
        }
        namespace = dict(
            OLD, inspect=inspect, hashlib=hashlib, types=types, **constants
        )
        node = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == "baseline_entry"
        )
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(FROZEN), "exec"),
            namespace,
        )
        current = historical_entry_namespace()
        functions = {
            name: object()
            for name in ("_prefill_logits", "persistent_topk", "_select_all_kernel")
        }
        current.update(functions)
        original_validator = current["_validate_inputs"]
        op = types.SimpleNamespace(
            qsa_select_paged_prefill=current["qsa_select_paged_prefill"], **functions
        )
        baseline = namespace["baseline_entry"](op)
        self.assertIs(baseline.__code__, op.qsa_select_paged_prefill.__code__)
        self.assertIs(baseline.__globals__["_validate_inputs"], OLD["_validate_inputs"])
        self.assertIs(current["_validate_inputs"], original_validator)
        self.assertIsNot(baseline.__globals__, current)
        for name in functions:
            self.assertIs(baseline.__globals__[name], functions[name])
        with self.assertRaisesRegex(RuntimeError, "public entry"):
            namespace["baseline_entry"](
                types.SimpleNamespace(qsa_select_paged_prefill=outcome)
            )
        with self.assertRaisesRegex(RuntimeError, "public entry"):
            namespace["baseline_entry"](
                types.SimpleNamespace(
                    qsa_select_paged_prefill=load_functions(
                        PRODUCTION, {"qsa_select_paged_prefill"}
                    )["qsa_select_paged_prefill"]
                )
            )


if __name__ == "__main__":
    unittest.main()
