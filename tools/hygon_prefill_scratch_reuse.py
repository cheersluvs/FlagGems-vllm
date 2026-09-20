"""Measure non-TLE scratch allocation versus a reusable per-shape plan.

The public Hygon override allocates six scratch tensors on every invocation.
This probe keeps the exact same non-TLE kernel and geometry, but allocates its
scratch once and launches it repeatedly. It reports wall-clock time, so the
comparison includes host allocation and launch setup rather than only device
kernel time. No production code is changed.
"""

from __future__ import annotations

import gc
import pathlib
import statistics
import subprocess
import sys
import time
from importlib import import_module

import torch

from hygon_prefill_audit import emit, occupancy

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHAPES = (
    ("sparse_long", 64, 129280, 1024, 129280),
    ("sparse_4_8193", 4, 8193, 512, 8456),
    ("dense_many", 16383, 4095, 512, 4352),
    ("sparse_4_16385", 4, 16385, 512, 16648),
    ("dense_4100", 12961, 4100, 512, 4352),
    ("dense_5115", 16380, 5115, 512, 5376),
    ("dense_short", 4100, 1025, 512, 1288),
)


def make_inputs(rows, vocab, stride0, seed):
    torch.manual_seed(seed)
    buf = torch.randn(
        (rows - 1) * stride0 + vocab,
        device="cuda",
        dtype=torch.float32,
    )
    logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), vocab, dtype=torch.int32, device="cuda")
    return logits, starts, ends


def configure(mod, ov, rows, vocab):
    geo = ov._geometry(rows, vocab)
    if geo is None:
        default_block, default_warps = ov._GENERIC_DEFAULTS[id(mod)]
        mod.NUM_THREADS_PER_BLOCK = default_block
        mod._num_warps = default_warps
    else:
        mod.NUM_THREADS_PER_BLOCK = geo[0]
        mod._num_warps = lambda block_size, w=geo[1]: w
    return geo


def active_module(ov, vocab, top_k):
    if vocab <= ov.DENSE_VOCAB_PER_TOPK * top_k:
        if (
            ov._dense_short_bins is not None
            and top_k == 512
            and vocab <= ov.SHORT_BINS_MAX_VOCAB
        ):
            return ov._dense_short_bins
        return ov._dense_vec2 or ov._dense_carry or ov._dense
    return ov._sparse


def check_values(out, data, top_k):
    logits, starts, ends = data
    rows, vocab = logits.shape
    cols = torch.arange(vocab, device=logits.device)[None, :]
    live = (cols >= starts[:, None]) & (cols < ends[:, None])
    want = torch.topk(torch.where(live, logits, float("-inf")), top_k, dim=1).values
    want = want.sort(dim=1).values
    valid = torch.arange(top_k, device=logits.device)[None, :] < (ends - starts)[:, None]
    absolute = torch.where(valid, out + starts[:, None], 0).long()
    got = torch.where(valid, logits.gather(1, absolute), float("-inf")).sort(dim=1).values
    if not torch.equal(got, want):
        raise AssertionError("reusable scratch output differs from torch.topk")


class ReusablePlan:
    def __init__(self, mod, rows, vocab, top_k):
        self.mod = mod
        self.rows = rows
        self.top_k = top_k
        self.hist = torch.empty(
            (rows, mod.NUM_BINS), device="cuda", dtype=torch.int32
        )
        self.final_logits = torch.empty(
            (rows, mod.NUM_FILNAL_ITEMS), device="cuda", dtype=torch.float32
        )
        self.final_cnt = torch.empty((rows,), device="cuda", dtype=torch.int32)
        self.threshold = torch.empty((rows,), device="cuda", dtype=torch.int32)
        self.final_bin_size = torch.empty((rows,), device="cuda", dtype=torch.int32)
        self.found = torch.empty((rows,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((rows, top_k), device="cuda", dtype=torch.int32)
        self._args = (
            self.hist,
            self.final_logits,
            self.final_cnt,
            self.threshold,
            self.final_bin_size,
            self.found,
        )

    def run(self, data, stride0):
        logits, starts, ends = data
        self.mod.non_tle_top_k_per_row_prefill[(self.rows,)](
            logits,
            self.out,
            starts,
            ends,
            stride0,
            1,
            logits.shape[1],
            *self._args,
            TOPK=self.top_k,
            BLOCK_SIZE=self.mod.NUM_THREADS_PER_BLOCK,
            ROW_OFFSET=0,
            num_warps=self.mod._num_warps(self.mod.NUM_THREADS_PER_BLOCK),
        )
        return self.out


def wall_us(fn, rounds=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(rounds):
        begin = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - begin) * 1e6)
    return statistics.median(values)


def main():
    import flaggems_vllm

    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    emit(
        "probe",
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    occupancy("before")
    for name, rows, vocab, top_k, stride0 in SHAPES:
        mod = active_module(ov, vocab, top_k)
        geo = configure(mod, ov, rows, vocab)
        data = make_inputs(rows, vocab, stride0, 71)
        output = torch.empty((rows, top_k), device="cuda", dtype=torch.int32)
        plan = ReusablePlan(mod, rows, vocab, top_k)
        plan.run(data, stride0)
        torch.cuda.synchronize()
        check_values(plan.out, data, top_k)

        def public():
            flaggems_vllm.top_k_per_row_prefill(
                data[0], data[1], data[2], output, rows, stride0, 1, top_k
            )

        def cached():
            plan.run(data, stride0)

        public_us = wall_us(public)
        cached_us = wall_us(cached)
        emit(
            "benchmark",
            shape=name,
            rows=rows,
            vocab=vocab,
            top_k=top_k,
            geometry=geo,
            module=getattr(mod, "__name__", None),
            public_wall_us=public_us,
            cached_wall_us=cached_us,
            allocation_reuse_ratio=public_us / cached_us if cached_us else None,
        )
        del plan, data, output
        gc.collect()
        torch.cuda.empty_cache()
    occupancy("after")
    emit("probe_complete", ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise