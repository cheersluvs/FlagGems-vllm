"""What is one-row decode's ~125 us of fixed kernel work on Hygon?

tools/hygon_decode_floor_fit.py: at one row the operator costs 0.155 ms FLAT
from vocab 4096 to 65536 at every launch geometry, then jumps at 262144; a
trivial kernel's launch floor is only ~30 us. So ~125 us is the kernel's own
work, independent of both data size and parallelism.

The non-TLE decode kernel hardcodes USE_RADIX_FINAL=False, so its final select
is always the rank sort -- `for j in tl.range(0, final_cnt)` inside a loop over
tiles of the candidates, O(final_cnt^2) -- whose work depends only on how many
candidates share the threshold bin. That fits "flat in vocab". It is still a
guess, so:

  1. INSTRUMENT: launch the non-TLE kernel directly with our own scratch and read
     back final_cnt (the final select's workload), final_bin_size and the count
     written directly, per vocab.
  2. ABLATE the final select, which is the LAST stage -- removing it changes no
     upstream control flow, so these deltas are attributions (unlike the
     prefill clear ablation that turned out to be control flow):
       no_rank_loop    the O(final_cnt^2) inner loop only
       no_final_select the whole final stage (loads, ranks, stores)

    tools/vendor_probe.sh tools/hygon_decode_fixed_ablate.py hygon_decode_fixed_ablate
"""

import importlib.util
import pathlib
import shutil
import sys
import tempfile
from importlib import import_module

import torch

import flaggems_vllm

VOCABS = (4096, 16384, 65536, 262144)
K = 512
SRC = pathlib.Path(flaggems_vllm.__file__).parent / "ops" / "top_k_per_row_decode.py"

ABLATIONS = [
    ("none", None, None, "baseline, must match the shipped operator"),
    (
        "no_rank_loop",
        "                for j in tl.range(0, final_cnt):",
        "                for j in tl.range(0, final_cnt * 0):",
        "the O(final_cnt^2) inner rank loop",
    ),
    (
        "no_final_select",
        "            sort_chunks = tl.cdiv(final_cnt, BLOCK_SIZE)",
        "            sort_chunks = tl.cdiv(final_cnt * 0, BLOCK_SIZE)",
        "the whole final select: loads, ranks and stores",
    ),
]


def build(name, old, new):
    src = SRC.read_text()
    if old is not None:
        if src.count(old) != 1:
            return None, f"pattern found {src.count(old)} times -- source moved"
        src = src.replace(old, new)
    d = pathlib.Path(tempfile.mkdtemp(prefix=f"decabl_{name}_"))
    f = d / f"decabl_{name}.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location(f"decabl_{name}", f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(d, ignore_errors=True)
        return None, f"import failed: {type(exc).__name__}: {exc}"
    return mod, ""


def timed(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000  # us


def main():
    dec = import_module("flaggems_vllm.ops.top_k_per_row_decode")
    torch.manual_seed(0)
    inputs = {}
    for v in VOCABS:
        logits = torch.randn(1, v, dtype=torch.float32, device="cuda")
        lens = torch.full((1,), v, dtype=torch.int32, device="cuda")
        inputs[v] = (logits, lens)

    print("=== 1. what the final select has to do, per vocab (1 row, k=512)")
    print(
        f"  {'vocab':>7} {'final_cnt':>9} {'final_bin_size':>14} "
        f"{'found (direct)':>14} {'threshold bin':>13}"
    )
    for v in VOCABS:
        logits, lens = inputs[v]
        idx = torch.empty(1, K, dtype=torch.int32, device="cuda")
        sc = {
            name: torch.full(shape, -7, dtype=dt, device="cuda")
            for name, shape, dt in (
                ("hist", (1, dec.NUM_BINS), torch.int32),
                ("flog", (1, dec.NUM_FILNAL_ITEMS), torch.float32),
                ("fcnt", (1,), torch.int32),
                ("tbin", (1,), torch.int32),
                ("fbsz", (1,), torch.int32),
                ("found", (1,), torch.int32),
            )
        }
        dec.non_tle_top_k_per_row_decode[(1,)](
            logits,
            idx,
            lens,
            1,
            v,
            1,
            v,
            sc["hist"],
            sc["flog"],
            sc["fcnt"],
            sc["tbin"],
            sc["fbsz"],
            sc["found"],
            TOPK=K,
            BLOCK_SIZE=dec.NUM_THREADS_PER_BLOCK,
            num_warps=dec._num_warps(dec.NUM_THREADS_PER_BLOCK),
        )
        torch.cuda.synchronize()
        want = torch.topk(logits, K, dim=1).values.sort(dim=1).values
        got = logits.gather(1, idx.long().clamp(0, v - 1)).sort(dim=1).values
        ok = "" if torch.allclose(got, want) else "   WRONG"
        print(
            f"  {v:>7} {int(sc['fcnt'][0]):>9} {int(sc['fbsz'][0]):>14} "
            f"{int(sc['found'][0]):>14} {int(sc['tbin'][0]):>13}{ok}"
        )

    print("\n=== 2. ablating the final select (us per call, 1 row)")
    print(f"  {'variant':<16} " + " ".join(f"{v:>9}" for v in VOCABS) + "   removes")
    rows = {}
    for name, old, new, what in ABLATIONS:
        mod, err = build(name, old, new)
        if mod is None:
            print(f"  {name:<16} {err}")
            continue
        cells = []
        for v in VOCABS:
            logits, lens = inputs[v]
            idx = torch.empty(1, K, dtype=torch.int32, device="cuda")
            us = timed(
                lambda m=mod, a=logits, s=lens, o=idx, n=v: m.top_k_per_row_decode(
                    a, 1, s, o, 1, n, 1, K
                )
            )
            rows.setdefault(name, {})[v] = us
            cells.append(f"{us:>9.1f}")
        print(f"  {name:<16} " + " ".join(cells) + f"   {what}")
    if "none" in rows:
        print(f"\n  {'delta vs none':<16} ")
        for name in rows:
            if name == "none":
                continue
            print(
                f"  {name:<16} "
                + " ".join(f"{rows['none'][v] - rows[name][v]:>9.1f}" for v in VOCABS)
            )
    print("\n  Shipped operator for reference (should match 'none'):")
    cells = []
    for v in VOCABS:
        logits, lens = inputs[v]
        idx = torch.empty(1, K, dtype=torch.int32, device="cuda")
        cells.append(
            f"{timed(lambda a=logits, s=lens, o=idx, n=v: dec.top_k_per_row_decode(a, 1, s, o, 1, n, 1, K)):>9.1f}"
        )
    print(f"  {'shipped':<16} " + " ".join(cells))


if __name__ == "__main__":
    sys.exit(main())
