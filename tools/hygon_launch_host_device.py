"""Is the ~80 us every top_k_per_row call pays on Hygon host-side or device-side?

tools/hygon_decode_path_cost.py: a one-row decode call costs ~110 us even when
seq_len=1 makes the kernel return almost at once, and the same holds for
prefill -- so the cost does not depend on the work executed. A trivial
kernel's launch floor is ~30 us. Yet the benchmark times a 4-row prefill call
at vocab 8193 at ~32 us in total, through the same kernel.

Two different things could be that ~80 us:

  HOST    Triton's per-call JIT dispatch: binding ~15 runtime arguments and
          several constexprs, building the specialization key, looking up the
          cache -- pure Python, proportional to the argument list, and slow on a
          slow host CPU
  DEVICE  per-launch setup of a larger kernel on the card

For each of four calls this measures three numbers:

    submit  perf_counter around the call WITHOUT synchronising: host time only
    wall    CUDA events around the call with synchronise
    device  the kernel's own time from torch.profiler

    trivial     3 args, 1 constexpr
    wide_noop   the operator's argument list and constexprs, doing nothing
    decode_s1   the real decode call with seq_len=1 (kernel returns at once)
    prefill_e1  the real prefill call with row_end=1

and one more line: the same decode call launched through a pre-bound kernel
(`kernel.run` args prepared once) if the Triton version exposes it.

    tools/vendor_probe.sh tools/hygon_launch_host_device.py hygon_launch_host_device
"""

import sys
import time
from importlib import import_module

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

import flaggems_vllm  # noqa: F401  (runtime init)

K = 512
VOCAB = 4096


@triton.jit
def trivial(x_ptr, y_ptr, N: tl.constexpr):
    lane = tl.arange(0, N)
    tl.store(y_ptr + lane, tl.load(x_ptr + lane))


@triton.jit
def wide_noop(
    logits_ptr,
    out_indices_ptr,
    seq_lens_ptr,
    next_n,
    stride0,
    stride1,
    vocab_size,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_final_cnt_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_found_topk_values_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    lane = tl.arange(0, BLOCK_SIZE)
    tl.store(out_indices_ptr + lane, lane, mask=lane < TOPK)


def submit_us(fn, iters=300, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / iters * 1e6


def wall_us(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000


def device_us(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    best = 0.0
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0)
        if t and ev.count and t / ev.count > best and ev.device_type is not None:
            best = t / ev.count
    return best / 1.0  # profiler reports microseconds


def main():
    dec = import_module("flaggems_vllm.ops.top_k_per_row_decode")
    pre = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
    dev = "cuda"
    logits = torch.randn(1, VOCAB, dtype=torch.float32, device=dev)
    idx = torch.empty(1, K, dtype=torch.int32, device=dev)
    lens1 = torch.full((1,), 1, dtype=torch.int32, device=dev)
    starts = torch.zeros(1, dtype=torch.int32, device=dev)
    ends1 = torch.full((1,), 1, dtype=torch.int32, device=dev)
    x = torch.randn(512, device=dev)
    y = torch.empty_like(x)
    sc = [
        torch.empty(s, dtype=d, device=dev)
        for s, d in (
            ((1, 2048), torch.int32),
            ((1, 2048), torch.float32),
            ((1,), torch.int32),
            ((1,), torch.int32),
            ((1,), torch.int32),
            ((1,), torch.int32),
        )
    ]
    block = dec.NUM_THREADS_PER_BLOCK
    warps = dec._num_warps(block)

    calls = {
        "trivial": lambda: trivial[(1,)](x, y, N=512, num_warps=warps),
        "wide_noop": lambda: wide_noop[(1,)](
            logits,
            idx,
            lens1,
            1,
            VOCAB,
            1,
            VOCAB,
            *sc,
            TOPK=K,
            BLOCK_SIZE=block,
            num_warps=warps,
        ),
        "decode_s1": lambda: dec.top_k_per_row_decode(
            logits, 1, lens1, idx, 1, VOCAB, 1, K
        ),
        "prefill_e1": lambda: pre.top_k_per_row_prefill(
            logits, starts, ends1, idx, 1, VOCAB, 1, K
        ),
    }
    print(f"host python: {sys.version.split()[0]}  triton {triton.__version__}\n")
    print(f"  {'call':<12} {'submit us':>10} {'wall us':>9} {'device us':>10}")
    for name, fn in calls.items():
        s = submit_us(fn)
        w = wall_us(fn)
        d = device_us(fn)
        print(f"  {name:<12} {s:>10.1f} {w:>9.1f} {d:>10.1f}")
    print("\n  submit ~= wall  -> the cost is host-side (JIT dispatch / host code)")
    print("  device ~= wall  -> the cost is on the card")
    print("  wide_noop vs trivial isolates what the argument list alone costs.")


if __name__ == "__main__":
    sys.exit(main())
