# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Execute the real cache-key/run methods with a CPU-only fake device runtime."""

import ast
import contextlib
import hashlib
import inspect
import types
import unittest
from collections import OrderedDict
from itertools import starmap
from pathlib import Path
from unittest.mock import patch

from tools import qsa_prefill_cache_granularity as trial
from tools import qsa_prefill_triton_ab as ab

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "src/flaggems_vllm/ops/qsa_select_paged_prefill.py"
LIBENTRY = ROOT / "src/flaggems_vllm/utils/libentry.py"


class Tensor:
    def __init__(self, dtype):
        self.dtype = dtype

    def data_ptr(self):
        return 4096


def cache_runtime():
    """Extract actual methods, preserving their control flow and parameter lists."""
    production = ast.parse(PRODUCTION.read_text())
    library = ast.parse(LIBENTRY.read_text())
    score = next(
        n
        for n in production.body
        if isinstance(n, ast.FunctionDef) and n.name == "_qsa_prefill_score_kernel"
    )
    local = next(n for n in production.body if isinstance(n, ast.ClassDef))
    entry = next(
        n for n in library.body if isinstance(n, ast.ClassDef) and n.name == "LibEntry"
    )
    tuner = next(
        n for n in library.body if isinstance(n, ast.ClassDef) and n.name == "LibTuner"
    )
    methods = [
        n
        for n in entry.body
        if isinstance(n, ast.FunctionDef) and n.name in ("key", "run")
    ]
    get_key = next(
        n for n in tuner.body if isinstance(n, ast.FunctionDef) and n.name == "get_key"
    )
    signature_function = ast.FunctionDef(
        name="signature_function", args=score.args, body=[ast.Pass()], decorator_list=[]
    )
    base = ast.ClassDef(
        name="LibEntry", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    tree = ast.Module(body=[base, local, signature_function], type_ignores=[])
    env = dict(
        OrderedDict=OrderedDict,
        inspect=inspect,
        runtime=types.SimpleNamespace(
            device=types.SimpleNamespace(vendor_name="nvidia")
        ),
        torch_device_fn=types.SimpleNamespace(current_device=lambda: 0),
        major_version=3,
        minor_version=6,
        tl=types.SimpleNamespace(constexpr=object()),
        starmap=starmap,
    )
    exec(compile(ast.fix_missing_locations(tree), str(PRODUCTION), "exec"), env)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[get_key], type_ignores=[])),
            str(LIBENTRY),
            "exec",
        ),
        env,
    )
    signature = inspect.signature(env["signature_function"])
    dns = {"num_rows", "query_offset", "max_query_len"}
    params = [
        types.SimpleNamespace(
            num=i,
            name=p.name,
            is_constexpr=p.annotation is env["tl"].constexpr,
            do_not_specialize=p.name in dns,
            default=p.default,
        )
        for i, p in enumerate(signature.parameters.values())
    ]
    keys = ast.literal_eval(
        next(
            k.value
            for d in score.decorator_list
            if isinstance(d, ast.Call)
            and isinstance(d.func, ast.Name)
            and d.func.id == "libtuner"
            for k in d.keywords
            if k.arg == "key"
        )
    )

    class JIT:
        pass

    class Kernel:
        def __init__(self, tile):
            self.tile, self.launches = tile, []

        def __getitem__(self, grid):
            return lambda *args: self.launches.append((grid, args))

    class Tuner:
        get_key = env["get_key"]

        def __init__(self):
            self.fn = JIT()
            self.fn.params = params
            self.keys, self.strategy, self.calls = keys, None, []

        def run(self, *args, **kwargs):
            named = dict(zip(signature.parameters, args), **kwargs)
            self.calls.append(self.get_key(named))
            tile = 32 if named["num_rows"] >= 64 else 4
            self.best_config = types.SimpleNamespace(
                num_warps=8,
                num_stages=2,
                num_ctas=1,
                pre_hook=None,
                kwargs=dict(TILE_R=tile, BLOCK_N=64, K_TILES=4),
            )
            kernel = Kernel(tile)
            kernel[named["grid"](self.best_config.kwargs)](*args)
            return kernel

    env["triton"] = types.SimpleNamespace(
        runtime=types.SimpleNamespace(
            JITFunction=JIT,
            Autotuner=Tuner,
            Heuristics=type("Heuristics", (), {}),
        )
    )

    def make(local=True):
        cls = env["_QSAScoreEntry" if local else "LibEntry"]
        result = cls()
        result.fn = Tuner()
        result.arg_names = list(signature.parameters)
        result.divisibility, result.kernel_cache = 16, ({},)
        result._has_flagtune_tuner = False
        result.lock, result.signature = contextlib.nullcontext(), signature
        result.jit_function = result.fn.fn
        result.specialize_indices = [
            p.num for p in params if not p.is_constexpr and not p.do_not_specialize
        ]
        result.do_not_specialize_indices = [
            p.num for p in params if p.do_not_specialize
        ]
        return result

    def launch(entry, rows=64, offset=0, dtype="bfloat16", maximum=81):
        values = {p.name: 1 for p in params if not p.is_constexpr}
        for name in list(values)[:6]:
            values[name] = Tensor(dtype if name in ("q_ptr", "cache_ptr") else "int32")
        values.update(
            num_rows=rows, query_offset=offset, max_query_len=maximum, logits_width=1088
        )
        return entry.run(
            *values.values(),
            PAGE_SIZE=16,
            NUM_HEADS=4,
            HEAD_DIM=128,
            IS_FP8=dtype != "bfloat16",
            grid=lambda meta: (1, (rows + meta["TILE_R"] - 1) // meta["TILE_R"], 1),
        )

    return make, launch


class CacheTests(unittest.TestCase):
    def test_source_normalization_recovers_exact_previous_ten(self):
        source = PRODUCTION.read_text()
        self.assertEqual(
            hashlib.sha256(source.encode()).hexdigest(), ab.INTEGRATED_PRODUCTION_SHA256
        )
        restored = trial.round1_source(source)
        self.assertEqual(
            hashlib.sha256(restored.encode()).hexdigest(), trial.ROUND1_SOURCE_SHA256
        )
        self.assertEqual(trial.round1_source(restored), restored)
        with self.assertRaisesRegex(
            RuntimeError, "normalization mismatch|outside the operator-local"
        ):
            trial.round1_source(source + "# unexpected body edit\n")

    def test_real_run_separates_tail_and_reuses_repeated_rows(self):
        make, launch = cache_runtime()
        for local, expected_calls in ((False, 1), (True, 2)):
            entry = make(local)
            first = launch(entry)
            second = launch(entry, rows=17, offset=64)
            repeat = launch(entry, offset=128)
            self.assertEqual(len(entry.fn.calls), expected_calls)
            self.assertEqual(len(entry.kernel_cache[0]), expected_calls)
            self.assertIs(first[0], repeat[0])
            self.assertEqual(first[1]["TILE_R"], 32)
            self.assertEqual(second[1]["TILE_R"], 4 if local else 32)
            if local:
                self.assertIsNot(first[0], second[0])
            else:
                self.assertIs(first[0], second[0])

    def test_existing_tuner_key_and_dtype_order_are_preserved(self):
        make, launch = cache_runtime()
        entry = make()
        launch(entry)
        key = next(iter(entry.kernel_cache[0]))
        self.assertEqual(key[-1], entry.fn.calls[-1])
        self.assertEqual(
            key[-1][-6:], ("bfloat16", "bfloat16", "int32", "int32", "int32", "int32")
        )
        entry.fn.strategy = [lambda value: value for _ in entry.fn.keys]
        launch(entry, rows=17, offset=64)
        self.assertEqual(list(entry.kernel_cache[0])[-1][-1], entry.fn.calls[-1])
        launch(entry, maximum=97)
        launch(entry, dtype="float8_e4m3fn")
        self.assertEqual(len(entry.fn.calls), 4)

    def test_baseline_identity_and_selector_are_explicit(self):
        for kind, count in (("eight", "eight"), ("round1", "ten")):
            identity = trial.previous_identity(kind)
            self.assertIn(count, identity["baseline_kind"])
            production = object()
            with patch(
                "tools.qsa_prefill_tuning_round1.previous_module",
                return_value=production,
            ) as factory:
                self.assertIs(trial.previous_module(production, kind), production)
            self.assertEqual(
                factory.call_args.kwargs["rectangular_group"], identity["config_group"]
            )
        for function, args in (
            (trial.previous_identity, ("unknown",)),
            (trial.previous_module, (None, "unknown")),
        ):
            with self.assertRaises(ValueError):
                function(*args)

    def test_observation_records_actual_configs_without_extra_launches(self):
        from tools import qsa_prefill_kernel_resources as resources

        class Entry:
            arg_names = ("num_rows", "query_offset", "logits_width")

            def __init__(self):
                self.calls = 0
                self.kernel_cache = ({},)
                self.fn = types.SimpleNamespace(
                    get_key=lambda values: (values["num_rows"],)
                )

            def __getitem__(self, grid):
                def invoke(*values, **kwargs):
                    self.calls += 1
                    self.kernel_cache[0][values[0]] = object()
                    return object(), dict(TILE_R=values[0])

                return invoke

        rectangular, compact = Entry(), Entry()
        module = types.SimpleNamespace(
            _qsa_prefill_score_kernel=rectangular,
            _qsa_prefill_compact_score_kernel=compact,
        )

        def observe(module, topk, args):
            self.assertEqual(topk, "persistent_topk")
            chunks = []
            for offset, rows in ((0, 64), (64, 17)):
                module._qsa_prefill_score_kernel[(1,)](rows, offset, 1088)
                chunks.append(dict(offset=offset, rows=rows, width=1088))
            return chunks

        with patch.object(ab, "observe", observe), patch.object(
            resources, "compiled_record", return_value={"registers": 32}
        ):
            chunks, traces = trial.observe(module, ())
        self.assertEqual(rectangular.calls, 2)
        self.assertEqual(compact.calls, 0)
        self.assertEqual([t["constants"]["TILE_R"] for t in traces], [64, 17])
        self.assertEqual([t["entry_count"] for t in traces], [1, 2])
        self.assertEqual(
            ab.signatures(chunks),
            [(t["offset"], t["rows"], t["width"]) for t in traces],
        )
        self.assertIs(module._qsa_prefill_score_kernel, rectangular)
        self.assertIs(module._qsa_prefill_compact_score_kernel, compact)

        def mismatched(module, topk, args):
            observe(module, topk, args)
            return []

        with patch.object(ab, "observe", mismatched), patch.object(
            resources, "compiled_record", return_value={}
        ):
            with self.assertRaisesRegex(RuntimeError, "chunk geometry"):
                trial.observe(module, ())
        self.assertIs(module._qsa_prefill_score_kernel, rectangular)


if __name__ == "__main__":
    unittest.main()
