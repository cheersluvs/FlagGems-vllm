# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU contracts for operand transposition, cache isolation and diagnostics."""

import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_qsa_prefill_chunk_setup as chunk

from tools import qsa_prefill_kernel_resources as resources

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "tools/qsa_prefill_frozen_production.py"
COMPACT = ROOT / "tools/qsa_prefill_compact_trial.py"
TRIAL = ROOT / "tools/qsa_prefill_key_major_trial.py"


def functions(path):
    return {
        n.name: n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
    }


class LayoutTests(unittest.TestCase):
    def test_only_dot_and_corresponding_axes_change(self):
        old = functions(PRODUCTION)["_qsa_prefill_score_kernel"]
        new = functions(TRIAL)["_qsa_prefill_key_major_score_kernel"]
        self.assertEqual(ast.dump(old.args), ast.dump(new.args))
        self.assertEqual(
            [ast.dump(d) for d in old.decorator_list],
            [ast.dump(d) for d in new.decorator_list],
        )
        # Undo precisely the axis change; every mask, bound, pointer, loop and
        # dtype must then match production. This also freezes the six configs.
        text = ast.unparse(new)
        start = text.index("query = tl.load(")
        end = text.index("column_lanes =", start)
        text = text[:start] + transpose(text[start:end]) + text[end:]
        start = text.index("keys = tl.load(")
        end = text.index("scores = tl.dot(", start)
        text = text[:start] + transpose(text[start:end]) + text[end:]
        text = text.replace("tl.dot(keys, query,", "tl.dot(query, keys,")
        text = text.replace(
            "(BLOCK_N, BLOCK_R, HEADS_PADDED)", "(BLOCK_R, HEADS_PADDED, BLOCK_N)"
        )
        text = text.replace("real_heads[None, None, :]", "real_heads[None, :, None]")
        text = text.replace("tl.sum(scores, axis=2)", "tl.sum(scores, axis=1)")
        start = text.index("tl.store(")
        text = text[:start] + transpose(text[start:])
        restored = ast.parse(text).body[0]
        restored.name = old.name
        self.assertEqual(ast.dump(restored), ast.dump(old))

    def test_private_factory_preserves_real_host_calls_and_compact_route(self):
        # Execute actual public bytecode with metadata-only fake tensors; check
        # every captured score grid/argument and TopK call and allocation size.
        for rows, requests, max_query, budget in (
            (73, 2, 39, "512"),
            (73, 2, 39, "0"),
            (8896, 54, 3200, "512"),
        ):
            ns, score_calls, topks, allocations = chunk.load(PRODUCTION)
            op = types.ModuleType("production")
            op.__dict__.update(ns)
            op.__file__ = str(PRODUCTION)
            op._qsa_prefill_score_kernel.kernel_cache = ({},)
            op._qsa_prefill_score_kernel._cpu_cache = {}
            captured = []

            class Kernel:
                def __init__(self, label):
                    self.label = label
                    self.kernel_cache, self._cpu_cache = ({},), {}

                def __getitem__(self, grid):
                    def invoke(*args, **kw):
                        captured.append(
                            (
                                self.label,
                                grid(dict(TILE_R=16, BLOCK_N=64, K_TILES=4)),
                                tuple(map(chunk.normalize, args)),
                                kw,
                            )
                        )

                    return invoke

            compact_ns = dict(
                copy=copy,
                hashlib=hashlib,
                types=types,
                Path=Path,
                PRODUCTION_SHA256=hashlib.sha256(PRODUCTION.read_bytes()).hexdigest(),
                _qsa_prefill_compact_score_kernel=Kernel("compact"),
            )
            nodes = functions(COMPACT)
            exec(
                compile(
                    ast.Module(
                        body=[
                            nodes[n]
                            for n in (
                                "_compact_eligible",
                                "isolated_score_entry",
                                "_prefill_logits",
                                "candidate_module",
                            )
                        ],
                        type_ignores=[],
                    ),
                    str(COMPACT),
                    "exec",
                ),
                compact_ns,
            )
            trial_ns = dict(
                types=types,
                compact=types.SimpleNamespace(**compact_ns),
                _qsa_prefill_key_major_score_kernel=Kernel("rectangular"),
            )
            exec(
                compile(
                    ast.Module(
                        body=[functions(TRIAL)["candidate_module"]], type_ignores=[]
                    ),
                    str(TRIAL),
                    "exec",
                ),
                trial_ns,
            )
            candidate = trial_ns["candidate_module"](op)
            self.assertIsNot(
                candidate._qsa_prefill_score_kernel.kernel_cache[0],
                op._qsa_prefill_score_kernel.kernel_cache[0],
            )
            self.assertIs(candidate.persistent_topk, op.persistent_topk)
            args = chunk.inputs(rows=rows, padded=True)
            args[2] = chunk.Array(
                (requests, 512), "i32", 30000000, (2048, 2), label="input_2"
            )
            args[3] = chunk.Array((requests + 1,), "i32", 40000000, label="input_3")
            args[7] = max_query
            with patch.dict(os.environ, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=budget):
                op.qsa_select_paged_prefill(*args)
                expected = list(score_calls)
                expected_topks = [
                    (tuple(map(chunk.normalize, a)), kw) for a, kw in topks
                ]
                expected_alloc = [(a.shape, a.dtype) for a in allocations]
                score_calls.clear()
                topks.clear()
                allocations.clear()
                candidate.qsa_select_paged_prefill(*args)
            self.assertFalse(score_calls)
            self.assertEqual(
                [(tuple(map(chunk.normalize, a)), kw) for a, kw in topks],
                expected_topks,
            )
            self.assertEqual([(a.shape, a.dtype) for a in allocations], expected_alloc)
            self.assertEqual(len(captured), len(expected))
            for (label, grid, a, kw), (grids, b, old_kw) in zip(captured, expected):
                eligible = candidate._compact_eligible(rows, requests, max_query, a[12])
                self.assertEqual(label, "compact" if eligible else "rectangular")
                self.assertEqual(a, b)
                self.assertEqual(kw, old_kw)
                self.assertEqual(
                    grid,
                    (a[12] // 16 + requests, 1, grids[2][2]) if eligible else grids[2],
                )

    def test_cli_and_stock_profile_rejection(self):
        script = ROOT / "tools/benchmark_qsa_prefill_triton_stages.py"
        command = [
            sys.executable,
            "-I",
            str(script),
            "--candidate",
            "key-major",
            "--suite",
            "combined-prefill",
            "--list-cases",
        ]
        proc = subprocess.run(
            command + ["--compiler", "flagtree"], capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(json.loads(proc.stdout)), 32)
        proc = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("FlagTree", proc.stderr)


def transpose(text):
    return (
        text.replace("[:, None]", "[TRANSPOSE]")
        .replace("[None, :]", "[:, None]")
        .replace("[TRANSPOSE]", "[None, :]")
    )


class ResourceTests(unittest.TestCase):
    def test_static_counts_and_missing_values_are_explicit(self):
        kernel = types.SimpleNamespace(
            asm=dict(
                ptx="""// ld.shared.u32 ignored;
                @%p1 ld.global.v4.b32 x, [y];
                /* wgmma.mma_async.sync fake */
                wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 x, y;
                wgmma.fence.sync.aligned;
                wgmma.wait_group.sync.aligned 0;
                st.shared.u32 [x], y;
                cp.async.cg.shared.global [x], [y], 16;
                """,
                cubin=b"cubin",
            ),
            metadata=types.SimpleNamespace(name="kernel", shared=73728, num_warps=4),
        )
        result = resources.compiled_record(kernel)
        self.assertIsNone(result["registers_per_thread"])
        self.assertIsNone(result["spill_count"])
        self.assertEqual(result["shared_bytes"], 73728)
        self.assertEqual(result["ptx_static_sites"]["dot"], 1)
        self.assertEqual(result["ptx_static_sites"]["shared_load"], 0)
        self.assertEqual(result["ptx_static_sites"]["global_load"], 1)
        self.assertEqual(result["ptx_static_sites"]["async_copy"], 1)
        jit = types.SimpleNamespace(
            device_caches={0: ({"a": kernel, "b": kernel}, None, None, None)}
        )
        self.assertEqual(len(resources.native_records(jit)["kernels"]), 1)
        self.assertEqual(resources.native_records(object())["status"], "unavailable")
        self.assertEqual(
            resources.native_records(types.SimpleNamespace(device_caches={0: ({},)}))[
                "status"
            ],
            "not_launched",
        )
        self.assertEqual(
            resources.native_records(types.SimpleNamespace(device_caches={0: ([],)}))[
                "status"
            ],
            "unavailable",
        )


if __name__ == "__main__":
    unittest.main()
