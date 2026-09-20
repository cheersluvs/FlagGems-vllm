"""Probe a non-TLE radix final-select replacement on Hygon.

The shipped non-TLE path ranks the final candidate list with an O(n**2)
scalar loop. This probe keeps the current histogram path and replaces only
that final loop with the existing four-pass radix selection. Its 256 counters
are placed in the per-row global scratch tail. The generated module is
isolated and is not enabled by this file.
"""

from __future__ import annotations

import importlib.util
import pathlib
import statistics
import subprocess
import sys
import tempfile
from importlib import import_module

from hygon_prefill_audit import emit, occupancy
from hygon_prefill_audit_source import function_text, replace_once

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


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_radix_source(source):
    """Add a global-scratch copy of the TLE final selector."""
    original = function_text(source, "_final_select_radix")
    helper = original.replace(
        "def _final_select_radix(",
        "def _final_select_radix_non_tle(",
        1,
    )
    alloc = """    s_radix_counts = tle.gpu.alloc(
        [RADIX_SIZE_FINAL],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_radix_count_ptr = tle.gpu.local_ptr(s_radix_counts, (0,))
    radix_count_vec_ptr = s_radix_count_ptr + bins
"""
    helper = replace_once(
        helper,
        alloc,
        "    radix_count_vec_ptr = s_radix_count_ptr + bins\n",
    )
    helper = replace_once(
        helper,
        "    s_final_logits_ptr,\n    s_final_cnt_ptr,\n",
        "    s_final_logits_ptr,\n    s_radix_count_ptr,\n    s_final_cnt_ptr,\n",
    )
    helper = replace_once(
        helper,
        "                prefix_sum, _ = tle.cumsum(counts, axis=0, reverse=False)\n",
        "                prefix_sum = tl.cumsum(counts, axis=0) - counts\n",
    )
    source += "\n\n" + helper + "\n"

    source = replace_once(source, "NUM_BINS = 2048\n", "NUM_BINS = 2304\n")
    non_tle = function_text(source, "non_tle_top_k_per_row_prefill")
    non_tle = replace_once(
        non_tle,
        "NUM_BINS: tl.constexpr = 2048",
        "NUM_BINS: tl.constexpr = 2304",
    )
    source = replace_once(
        source,
        function_text(source, "non_tle_top_k_per_row_prefill"),
        non_tle,
    )
    old_call = """        if USE_RADIX_FINAL and HAS_TLE:
            _final_select_radix(
                s_histogram_ptr,
                s_final_logits_ptr,
                s_final_cnt_ptr,
                s_found_topk_values_ptr,
                s_out_indices_ptr,
                s_out_logits_ptr,
                TOPK=TOPK,
                BLOCK_SIZE=BLOCK_SIZE,
                MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
            )
        else:
"""
    new_call = """        if USE_RADIX_FINAL:
            if HAS_TLE:
                _final_select_radix(
                    s_histogram_ptr,
                    s_final_logits_ptr,
                    s_final_cnt_ptr,
                    s_found_topk_values_ptr,
                    s_out_indices_ptr,
                    s_out_logits_ptr,
                    TOPK=TOPK,
                    BLOCK_SIZE=BLOCK_SIZE,
                    MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                )
            else:
                _final_select_radix_non_tle(
                    s_histogram_ptr,
                    s_final_logits_ptr,
                    s_histogram_ptr + NUM_FINAL_ITEMS,
                    s_final_cnt_ptr,
                    s_found_topk_values_ptr,
                    s_out_indices_ptr,
                    s_out_logits_ptr,
                    TOPK=TOPK,
                    BLOCK_SIZE=BLOCK_SIZE,
                    MULTIPLE_BLOCKS_PER_ROW=MULTIPLE_BLOCKS_PER_ROW,
                )
        else:
"""
    source = replace_once(source, old_call, new_call)
    source = replace_once(
        source,
        "        USE_RADIX_FINAL=False,\n        HAS_TLE=False,\n",
        "        USE_RADIX_FINAL=True,\n        HAS_TLE=False,\n",
    )
    return source


def make_inputs(rows, vocab, stride0, seed, tied=False):
    import torch

    torch.manual_seed(seed)
    buf = torch.randn(
        (rows - 1) * stride0 + vocab,
        device="cuda",
        dtype=torch.float32,
    )
    if tied:
        buf = (buf * 4).round() / 4
    logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), vocab, dtype=torch.int32, device="cuda")
    return logits, starts, ends


def configure(mod, ov, rows, vocab):
    geo = ov._geometry(rows, vocab)
    if geo is not None:
        mod.NUM_THREADS_PER_BLOCK = geo[0]
        mod._num_warps = lambda block_size, w=geo[1]: w
    return geo


def check_values(out, data, top_k):
    import torch

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
        raise AssertionError("non-TLE radix final values differ from torch.topk")


def device_time(fn, needle, iters=5):
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    gpu = [e for e in prof.events() if getattr(e.device_type, "name", "") == "CUDA"]
    matched = [e for e in gpu if needle in e.name]
    if len(matched) != iters:
        raise RuntimeError(f"expected {iters} {needle} events, got {len(matched)}")
    return statistics.median(e.time_range.elapsed_us() for e in matched)


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


def main():
    import torch
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
    with tempfile.TemporaryDirectory(prefix="hygon_radix_final_") as folder:
        path = pathlib.Path(folder) / "radix_final.py"
        base = pathlib.Path(active_module(ov, 129280, 1024).__file__).read_text()
        candidate_source = build_radix_source(base)
        compile(candidate_source, "<hygon-non-tle-radix>", "exec")
        path.write_text(candidate_source)
        candidate = load("flaggems_vllm.ops._hygon_radix_final", path)
        emit("setup", source=str(path), scratch_bins=2304, radix_counts=256)
        for name, rows, vocab, top_k, stride0 in SHAPES:
            geo = configure(candidate, ov, rows, vocab)
            normal = make_inputs(rows, vocab, stride0, 41)
            tied = make_inputs(rows, vocab, stride0, 42, tied=True)
            output = torch.empty((rows, top_k), device="cuda", dtype=torch.int32)
            for label, data in (("normal", normal), ("tied", tied)):
                candidate.top_k_per_row_prefill(
                    data[0], data[1], data[2], output, rows, stride0, 1, top_k
                )
                torch.cuda.synchronize()
                check_values(output, data, top_k)
                emit("validation", shape=name, case=label, geometry=geo, ok=True)

            def control(data=normal):
                flaggems_vllm.top_k_per_row_prefill(
                    data[0], data[1], data[2], output, rows, stride0, 1, top_k
                )

            def trial(data=normal):
                candidate.top_k_per_row_prefill(
                    data[0], data[1], data[2], output, rows, stride0, 1, top_k
                )

            base_us = device_time(control, "top_k_per_row", iters=5)
            cand_us = device_time(trial, "top_k_per_row", iters=5)
            emit(
                "benchmark",
                shape=name,
                rows=rows,
                vocab=vocab,
                top_k=top_k,
                baseline_us=base_us,
                candidate_us=cand_us,
                ratio=base_us / cand_us if cand_us else None,
            )
    occupancy("after")
    emit("probe_complete", ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise