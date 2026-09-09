"""Ablate the real prefill kernel, one component at a time.

Five attributions for prefill's ~17 ns-per-unit-of-top_k term have now been
refuted by measurement: the rank sort (13x its loop bound changed nothing), bin
walking (the slope grows with vocabulary instead of shrinking), a scatter store
(1.8 ns/k against the 17 needed), the 65536 radix-final gate (no step in the
curve) and, last, the 2048-bin `tl.histogram` cliff -- which is real, but this
operator does not call tl.histogram at all. It builds its histogram with one
GLOBAL atomic_add per element.

So stop reasoning about it. Each step of the kernel does four things:

    1  zero 2048 global counters
    2  one atomic_add per element into them
    3  read them back, prefix-sum, write back
    4  _process_bins, which compacts the survivors

This rewrites the operator's source with one of those neutered, imports the
result as a real module -- @triton.jit needs inspect.getsource, so it has to be
a file on disk, not an exec of a string -- and times it. Ablated runs are
WRONG BY CONSTRUCTION; only the timing means anything, and the `none` row is
there to prove the harness reproduces the original before any delta is read.

    python tools/metax_prefill_ablate.py
"""

import importlib.util
import pathlib
import shutil
import sys
import tempfile

import torch

import flaggems_vllm

DEV = flaggems_vllm.device
ROWS, VOCAB, TOPK = 4160, 4096, 512

SRC = pathlib.Path(flaggems_vllm.__file__).parent / "ops" / "top_k_per_row_prefill.py"

# Each ablation is (name, old, new, what it removes).
ABLATIONS = [
    ("none", None, None, "baseline -- must match the real operator"),
    (
        "no_atomic",
        "    tl.atomic_add(\n        s_histogram_ptr + bin_idx,\n        ones,\n"
        "        mask=is_partial_match,\n        sem=\"relaxed\",\n        scope=\"cta\",\n    )",
        "    tl.atomic_add(\n        s_histogram_ptr + bin_idx * 0,\n        ones,\n"
        "        mask=is_partial_match,\n        sem=\"relaxed\",\n        scope=\"cta\",\n    )",
        "same atomic count, all to ONE address: isolates address spread",
    ),
    (
        "no_clear",
        "        tl.store(s_histogram_ptr + clear_bins, 0)",
        "        tl.store(s_histogram_ptr + clear_bins, 0, mask=clear_bins < 0)",
        "the 2048-counter zeroing",
    ),
    (
        "no_scan",
        "            counts = tl.load(s_histogram_ptr + bins)",
        "            counts = tl.load(s_histogram_ptr + bins) * 0",
        "makes the prefix sum trivial, keeping its loads and stores",
    ),
]


def build(name, old, new):
    """Write a patched copy of the operator and import it as its own module."""
    src = SRC.read_text()
    if old is not None:
        if old not in src:
            return None, f"pattern not found -- source moved"
        src = src.replace(old, new, 1)
    d = pathlib.Path(tempfile.mkdtemp(prefix=f"ablate_{name}_"))
    f = d / f"ablated_{name}.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location(f"ablated_{name}", f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(d, ignore_errors=True)
        return None, f"import failed: {type(exc).__name__}: {exc}"
    return (mod, d), None


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000 / (ROWS / 104)


def main():
    print(f"device {DEV} | rows {ROWS} vocab {VOCAB} top_k {TOPK}")
    print("per-program microseconds. Ablated rows are WRONG by construction;")
    print("read the deltas, not the answers.\n")

    logits = torch.randn(ROWS, VOCAB, device=DEV, dtype=torch.float32)
    starts = torch.zeros(ROWS, dtype=torch.int32, device=DEV)
    ends = torch.full((ROWS,), VOCAB, dtype=torch.int32, device=DEV)
    out = torch.empty((ROWS, TOPK), dtype=torch.int32, device=DEV)

    real = timed(lambda: flaggems_vllm.top_k_per_row_prefill(
        logits, starts, ends, out, ROWS, logits.stride(0), logits.stride(1), TOPK))
    print(f"  {'the shipped operator':<26} {real:8.3f}\n")
    print(f"  {'ablation':<26} {'us/prog':>8} {'delta':>8}   removes")

    base = None
    for name, old, new, what in ABLATIONS:
        built, err = build(name, old, new)
        if err:
            print(f"  {name:<26} {'--':>8} {'':>8}   {err}")
            continue
        mod, d = built
        try:
            t = timed(lambda: mod.top_k_per_row_prefill(
                logits, starts, ends, out, ROWS,
                logits.stride(0), logits.stride(1), TOPK))
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<26} {'FAILED':>8} {'':>8}   {type(exc).__name__}: {exc}")
            shutil.rmtree(d, ignore_errors=True)
            continue
        if name == "none":
            base = t
            note = "harness OK" if abs(t - real) / real < 0.05 else "!! DOES NOT MATCH"
            print(f"  {name:<26} {t:8.3f} {'':>8}   {note}")
        else:
            print(f"  {name:<26} {t:8.3f} {base - t:>+8.3f}   {what}")
        shutil.rmtree(d, ignore_errors=True)

    print("\n  A delta near 8.7 us is the k term. If none of them is, the term")
    print("  lives in _process_bins and that is where the next cut goes.")


if __name__ == "__main__":
    sys.exit(main())
