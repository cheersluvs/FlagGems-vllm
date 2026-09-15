"""One-row decode on Hygon: DEVICE time only, by vocab and by radix step.

tools/hygon_launch_host_device.py settled that the ~110 us floor of a one-row
call is HOST dispatch (device ~2 us). Host and device overlap, so at vocab
262144 -- every benchmark decode shape -- the wall time is the device's 325 us
and the host floor is invisible. The benchmark gap there is device work:
325 us against vLLM's 71 us for the same 262144 elements.

The earlier vocab sweep timed WALL, so its "flat 0.155 ms up to 65536" was the
host floor hiding the device curve. This redoes it with the profiler's device
time of the kernel alone, and prices the radix steps by capping them in a
patched copy of the module (`for step_idx in tl.static_range(0, 4)`):

    full      the shipped kernel
    steps<=1  only STEP 0     -- WRONG answers wherever step 0 is not enough
    steps<=2  STEP 0 and 1
    steps<=3

Correctness is checked against torch.topk for every point, so a capped variant
that stays CORRECT shows the dropped steps were not needed at that vocab, and
its time shows what they cost. vLLM's device time is listed for reference.

    tools/vendor_probe.sh tools/hygon_decode_device_steps.py hygon_decode_device_steps
"""

import importlib.util
import pathlib
import sys
import tempfile
from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, profile

import flaggems_vllm

VOCABS = (4096, 16384, 65536, 131072, 262144)
K = 512
SRC = pathlib.Path(flaggems_vllm.__file__).parent / "ops" / "top_k_per_row_decode.py"
LOOP = "    for step_idx in tl.static_range(0, 4):"


def build(steps):
    src = SRC.read_text()
    if steps is None:
        return import_module("flaggems_vllm.ops.top_k_per_row_decode")
    assert src.count(LOOP) == 1, "step loop moved"
    src = src.replace(LOOP, f"    for step_idx in tl.static_range(0, {steps}):")
    d = pathlib.Path(tempfile.mkdtemp(prefix=f"decsteps{steps}_"))
    f = d / f"decsteps{steps}.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location(f"decsteps{steps}", f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def device_us(fn, match, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = 0.0
    for ev in prof.key_averages():
        if not any(m in ev.key for m in match):
            continue
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0)
        total += t or 0.0
    return total / iters  # microseconds


def main():
    torch.manual_seed(0)
    inputs = {}
    for v in VOCABS:
        logits = torch.randn(1, v, dtype=torch.float32, device="cuda")
        lens = torch.full((1,), v, dtype=torch.int32, device="cuda")
        inputs[v] = (
            logits,
            lens,
            torch.topk(logits, K, dim=1).values.sort(dim=1).values,
        )

    print("device microseconds per call, 1 row, top_k 512 (! = WRONG answer)\n")
    print(f"  {'variant':<10} " + " ".join(f"{v:>9}" for v in VOCABS))
    rows = {}
    for label, steps in (
        ("full", None),
        ("steps<=3", 3),
        ("steps<=2", 2),
        ("steps<=1", 1),
    ):
        mod = build(steps)
        cells = []
        for v in VOCABS:
            logits, lens, want = inputs[v]
            idx = torch.empty(1, K, dtype=torch.int32, device="cuda")
            mod.top_k_per_row_decode(logits, 1, lens, idx, 1, v, 1, K)
            torch.cuda.synchronize()
            got = logits.gather(1, idx.long().clamp(0, v - 1)).sort(dim=1).values
            ok = torch.allclose(got, want)
            us = device_us(
                lambda m=mod, a=logits, s=lens, o=idx, n=v: m.top_k_per_row_decode(
                    a, 1, s, o, 1, n, 1, K
                ),
                ("top_k_per_row_decode",),
            )
            rows.setdefault(label, {})[v] = us
            cells.append(f"{us:>8.1f}{' ' if ok else '!'}")
        print(f"  {label:<10} " + " ".join(cells))

    try:
        import vllm._custom_ops  # noqa: F401

        cells = []
        for v in VOCABS:
            logits, lens, _ = inputs[v]
            idx = torch.empty(1, K, dtype=torch.int32, device="cuda")
            us = device_us(
                lambda a=logits, s=lens, o=idx, n=v: torch.ops._C.top_k_per_row_decode(
                    a, 1, s, o, 1, n, 1, K
                ),
                ("topKPerRowDecode",),
            )
            cells.append(f"{us:>8.1f} ")
        print(f"  {'vLLM':<10} " + " ".join(cells))
    except Exception as e:  # noqa: BLE001
        print(f"  vLLM: unavailable ({type(e).__name__})")

    full = rows.get("full", {})
    if full:
        xs = [v for v in VOCABS if v >= 16384]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(full[x] for x in xs) / n
        e = sum((x - mx) * (full[x] - my) for x in xs) / sum((x - mx) ** 2 for x in xs)
        print(
            f"\n  full kernel, fitted over 16384..262144: fixed {my - e * mx:.1f} us, "
            f"{e * 1000:.3f} ns/element"
        )
    print("  A capped variant that stays CORRECT at a vocab shows those steps were")
    print("  unnecessary there; the gap to `full` is what they cost.")


if __name__ == "__main__":
    sys.exit(main())
