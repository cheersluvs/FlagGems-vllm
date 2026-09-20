"""Probe short-row adaptive bins (#4) and a small-row top-k sort (#6).

This is an experiment only.  It builds isolated Triton module copies and does
not change the shipped operator.  The two arms are deliberately kept
separate:

* ``bins`` changes STEP 0's key width and the one-scan width.  The module still
  reserves the production 2048-element scratch row because STEP 1/2 and the
  final candidate-index storage can need it.  This answers whether the
  narrower histogram work itself helps before attempting a scratch-layout
  rewrite.
* ``bitonic`` loads only a short row into one CTA and uses ``tl.topk``.  It is
  compared with the public Hygon operator, including the same row bounds and
  exact tie-value validation.  It is not enabled by the probe.

Run on BW1000 through ``tools/hygon_prefill_next_run.sh short_special``.
"""

import importlib.util
import math
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
from importlib import import_module

from hygon_prefill_audit import emit, occupancy

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN_SHAPES = (
    # Full benchmark row: the adaptive heuristic selects 256.
    ("full_1025", 4100, 1025, 512, 1288, 1025),
    # Same short row in a larger padded vocabulary: dispatch must use row_len,
    # not logits.shape[1].
    ("partial_1025", 4100, 4095, 512, 4352, 1025),
    # The heuristic selects 512 here; this is a known possible crossover.
    ("full_4095", 16383, 4095, 512, 4352, 4095),
    ("full_5115", 16380, 5115, 512, 5376, 5115),
)
BIN_CANDIDATES = (2048, 1024, 512, 256)

SHORT_LENGTHS = (513, 640, 768, 896, 1024, 1025, 1152, 1536, 2048)
SHORT_ROWS = 4100
SHORT_VOCAB = 4095
SHORT_TOPK = 512
SHORT_STRIDE = 4352
SHORT_WARPS = (4, 8, 16)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


def heuristic_bins(row_len):
    return min(2048, max(256, next_power_of_2(max(1, (row_len + 7) // 8))))


def patch_bins(source, bins):
    """Narrow only STEP 0 in an already-built production route."""
    if bins not in BIN_CANDIDATES:
        raise ValueError(f"unsupported bin candidate: {bins}")
    shift = 5 + (11 - int(math.log2(bins)))
    old = "bin_idx = (mapped >> 5).to(tl.uint32)"
    if source.count(old) != 1:
        raise RuntimeError("STEP 0 key extraction marker drifted")
    source = source.replace(old, f"bin_idx = (mapped >> {shift}).to(tl.uint32)", 1)
    old = "RADIX_SIZE: tl.constexpr = RADIX10_SIZE if STEP == 3 else RADIX11_SIZE"
    new = (
        "RADIX_SIZE: tl.constexpr = ("
        "RADIX10_SIZE if STEP == 3 else "
        f"({bins} if STEP == 0 else RADIX11_SIZE)"
        ")"
    )
    if source.count(old) != 1:
        raise RuntimeError("one-scan radix-width marker drifted")
    return source.replace(old, new, 1)


def device_time(fn, needle, iters=8):
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    gpu = [e for e in prof.events() if getattr(e.device_type, "name", "") == "CUDA"]
    matched = [e for e in gpu if needle in e.name]
    if len(matched) != iters or len(gpu) != iters:
        raise RuntimeError(
            f"Expected {iters} single-kernel events for {needle}; "
            f"matched={len(matched)} total={len(gpu)} "
            f"names={sorted(set(e.name for e in gpu))}"
        )
    values = [e.time_range.elapsed_us() for e in matched]
    return statistics.median(values)


def make_inputs(rows, vocab, top_k, stride0, length, seed, tied=False):
    import torch

    torch.manual_seed(seed)
    buf = torch.randn((rows - 1) * stride0 + vocab, device="cuda", dtype=torch.float32)
    if tied:
        buf = (buf * 4).round() / 4
    x = torch.as_strided(buf, (rows, vocab), (stride0, 1))
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.full((rows,), length, device="cuda", dtype=torch.int32)
    return x, starts, ends


def check_values(out, tensors, top_k):
    import torch

    x, starts, ends = tensors
    rows, vocab = x.shape
    cols = torch.arange(vocab, device=x.device)[None, :]
    live = (cols >= starts[:, None]) & (cols < ends[:, None])
    want = torch.topk(torch.where(live, x, float("-inf")), top_k, dim=1).values
    want = want.sort(dim=1).values
    valid = torch.arange(top_k, device=x.device)[None, :] < (ends - starts)[:, None]
    if not bool(torch.all(torch.where(valid, (out >= 0) & (out < (ends - starts)[:, None]), out == -1))):
        raise AssertionError("short-special output bounds/padding invalid")
    absolute = torch.where(valid, out + starts[:, None], 0).long()
    got = torch.where(valid, x.gather(1, absolute), float("-inf")).sort(dim=1).values
    if not torch.equal(got, want):
        raise AssertionError("short-special selected values differ from torch.topk")


def configure(mod, ov, rows, vocab):
    geo = ov._geometry(rows, vocab) if ov._GEOMETRY else None
    if geo is None:
        mod.NUM_THREADS_PER_BLOCK = mod._GENERIC_DEFAULTS[id(mod)][0] if hasattr(mod, "_GENERIC_DEFAULTS") else mod.NUM_THREADS_PER_BLOCK
        return None
    mod.NUM_THREADS_PER_BLOCK = geo[0]
    mod._num_warps = lambda block_size, w=geo[1]: w
    return geo


def run_bins():
    import torch
    import flaggems_vllm

    ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
    if flaggems_vllm.top_k_per_row_prefill is not ov.top_k_per_row_prefill:
        raise RuntimeError("public prefill operator is not the Hygon override")
    if not ov._ONESCAN_PATH or not ov._ENABLED or not ov._GEOMETRY:
        raise RuntimeError("expected shipped Hygon one-scan and geometry paths")
    dense_source = pathlib.Path(ov._VEC2_PATH or ov._CARRY_PATH).read_text()
    sparse_source = pathlib.Path(ov._ONESCAN_PATH).read_text()
    emit(
        "bins_setup",
        source_dense=ov._VEC2_PATH or ov._CARRY_PATH,
        source_sparse=ov._ONESCAN_PATH,
        heuristic={str(length): heuristic_bins(length) for length in (1025, 4095, 5115)},
    )
    with tempfile.TemporaryDirectory(prefix="hygon_short_bins_") as folder:
        folder = pathlib.Path(folder)
        arms = {}
        for shape_name, rows, vocab, top_k, stride0, length in BIN_SHAPES:
            dense = vocab <= ov.DENSE_VOCAB_PER_TOPK * top_k
            base = dense_source if dense else sparse_source
            for bins in BIN_CANDIDATES:
                path = folder / f"bins_{shape_name}_{bins}.py"
                path.write_text(patch_bins(base, bins))
                arms[(shape_name, bins)] = load(
                    f"flaggems_vllm.ops._hygon_short_bins_{shape_name}_{bins}", path
                )
            geo = ov._geometry(rows, vocab)
            emit(
                "bins_shape",
                shape=shape_name,
                rows=rows,
                vocab=vocab,
                row_len=length,
                top_k=top_k,
                dense=dense,
                geometry=geo,
                adaptive_bins=heuristic_bins(length),
            )
            tensors = make_inputs(rows, vocab, top_k, stride0, length, 42)
            tied = make_inputs(rows, vocab, top_k, stride0, length, 43, tied=True)
            output = torch.empty((rows, top_k), device="cuda", dtype=torch.int32)
            for bins in BIN_CANDIDATES:
                mod = arms[(shape_name, bins)]
                configure(mod, ov, rows, vocab)
                for label, data in (("normal", tensors), ("tied", tied)):
                    mod.top_k_per_row_prefill(data[0], data[1], data[2], output, rows, stride0, 1, top_k)
                    torch.cuda.synchronize()
                    check_values(output, data, top_k)
                    emit("bins_validation", shape=shape_name, bins=bins, case=label, ok=True)

            def control(data=tensors):
                flaggems_vllm.top_k_per_row_prefill(
                    data[0], data[1], data[2], output, rows, stride0, 1, top_k
                )

            readings = {}
            for bins in BIN_CANDIDATES:
                mod = arms[(shape_name, bins)]

                def candidate(mod=mod, data=tensors):
                    mod.top_k_per_row_prefill(
                        data[0], data[1], data[2], output, rows, stride0, 1, top_k
                    )

                ratios = []
                values = []
                for round_id in range(3):
                    order = ("control", "candidate", "candidate", "control")
                    fns = {"control": control, "candidate": candidate}
                    order_values = [(arm, device_time(fns[arm], "non_tle_top_k_per_row_prefill")) for arm in order]
                    values.extend(order_values)
                    pair = dict(order_values)
                    ratios.extend([pair["control"] / pair["candidate"]] * 2)
                    emit(
                        "bins_round",
                        shape=shape_name,
                        bins=bins,
                        round=round_id,
                        readings=order_values,
                        ratio=pair["control"] / pair["candidate"],
                    )
                readings[str(bins)] = {
                    "median_candidate_us": statistics.median([v for arm, v in values if arm == "candidate"]),
                    "median_ratio": statistics.median(ratios),
                    "min_ratio": min(ratios),
                    "max_ratio": max(ratios),
                }
                emit("bins_summary", shape=shape_name, bins=bins, **readings[str(bins)])
            emit("bins_shape_summary", shape=shape_name, readings=readings)


def bitonic_plan(tensors, top_k, block, warps):
    import torch
    from hygon_prefill_bitonic_kernel import bitonic_topk_indices

    launcher = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )._Launch
    x, starts, ends = tensors
    rows = x.shape[0]
    guard = torch.full((rows * top_k + 32,), -123456, device=x.device, dtype=torch.int32)
    out = guard[16:-16].view(rows, top_k)
    launch = launcher(
        bitonic_topk_indices,
        (rows,),
        dict(TOPK=top_k, BLOCK=block),
        warps,
    )

    def run():
        launch(x, starts, ends, out, x.stride(0))

    run.out = out
    run.guard = guard
    return run


def run_bitonic():
    import torch
    import triton
    import flaggems_vllm

    ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
    output = torch.empty((SHORT_ROWS, SHORT_TOPK), device="cuda", dtype=torch.int32)
    emit(
        "bitonic_setup",
        rows=SHORT_ROWS,
        vocab=SHORT_VOCAB,
        top_k=SHORT_TOPK,
        lengths=SHORT_LENGTHS,
        warps=SHORT_WARPS,
    )
    for length in SHORT_LENGTHS:
        tensors = make_inputs(
            SHORT_ROWS, SHORT_VOCAB, SHORT_TOPK, SHORT_STRIDE, length, 123
        )
        tied = make_inputs(
            SHORT_ROWS, SHORT_VOCAB, SHORT_TOPK, SHORT_STRIDE, length, 124, tied=True
        )
        for label, data in (("normal", tensors), ("tied", tied)):
            candidate = bitonic_plan(data, SHORT_TOPK, next_power_of_2(length), 8)
            candidate()
            torch.cuda.synchronize()
            if not bool((candidate.guard[:16] == -123456).all() & (candidate.guard[-16:] == -123456).all()):
                raise AssertionError("bitonic output guard overwritten")
            check_values(candidate.out, data, SHORT_TOPK)
            emit("bitonic_validation", length=length, case=label, block=next_power_of_2(length), ok=True)

        def control():
            flaggems_vllm.top_k_per_row_prefill(
                tensors[0], tensors[1], tensors[2], output,
                SHORT_ROWS, SHORT_STRIDE, 1, SHORT_TOPK
            )

        baseline = device_time(control, "non_tle_top_k_per_row_prefill")
        candidates = {}
        for warps in SHORT_WARPS:
            candidate = bitonic_plan(
                tensors, SHORT_TOPK, next_power_of_2(length), warps
            )
            candidate()
            torch.cuda.synchronize()
            ratios = []
            samples = []
            for round_id in range(3):
                candidate_us = device_time(candidate, "bitonic_topk_indices")
                control_us = device_time(control, "non_tle_top_k_per_row_prefill")
                samples.append((control_us, candidate_us))
                ratios.append(control_us / candidate_us)
                emit(
                    "bitonic_round",
                    length=length,
                    block=next_power_of_2(length),
                    warps=warps,
                    round=round_id,
                    control_us=control_us,
                    candidate_us=candidate_us,
                    ratio=control_us / candidate_us,
                )
            candidates[str(warps)] = {
                "median_control_us": statistics.median(v[0] for v in samples),
                "median_candidate_us": statistics.median(v[1] for v in samples),
                "median_ratio": statistics.median(ratios),
                "min_ratio": min(ratios),
                "max_ratio": max(ratios),
            }
        best_warps = max(candidates, key=lambda w: candidates[w]["median_ratio"])
        emit(
            "bitonic_summary",
            length=length,
            block=next_power_of_2(length),
            baseline_us=baseline,
            best_warps=int(best_warps),
            best=candidates[best_warps],
            candidates=candidates,
        )


def main():
    import torch  # noqa: F401 - fail early on a non-Hygon host
    import flaggems_vllm  # noqa: F401
    import vllm._custom_ops  # noqa: F401 - register the baseline

    emit(
        "probe",
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        host=platform.node(),
        env={k: os.environ.get(k) for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "FLAGGEMS_FORCE_TLE")},
        bin_shapes=BIN_SHAPES,
        bin_candidates=BIN_CANDIDATES,
        short_lengths=SHORT_LENGTHS,
    )
    occupancy("before")
    run_bins()
    run_bitonic()
    occupancy("after")
    emit("probe_complete", ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise