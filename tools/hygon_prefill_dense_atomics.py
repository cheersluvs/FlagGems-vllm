"""Why the dense histogram's atomics cost half the kernel.

tools/hygon_prefill_dense_budget.py put the STEP-0 histogram atomics at
475-756 us -- 53-56% of the dense kernel -- where vLLM's whole kernel on
12961x4100 is 683 us. Two explanations lead to opposite fixes:

  * CONFLICTS. Standard-normal logits crowd a few hundred of the 2048 bins, and
    a partially-masked atomic to one address serialises at ~12 ns per taken
    lane on this card. Fix: replicate the histogram, or merge equal bins in a
    wave before the atomic.
  * TRANSACTIONS. One scattered atomic per element is simply too many,
    whatever the addresses. Fix: issue fewer -- e.g. a compare-and-count
    pass (no atomics) to find the coarse bucket holding rank top_k, and
    atomics only for the elements inside it.

Every arm is the cut1 PREFIX of the shipped dense kernel (histogram only,
nothing downstream), so only the atomic's ADDRESS changes and nothing after it
can do different work:

    cut1      the real 11-bit key bins
    cut1hash  bits 11-21 of the element's own 32-bit ordered key: the same
              number of atomics, spread near-uniformly over 2048 bins, so
              almost no two lanes of a wave share an address
    cut1flat  every atomic to bin 0: all lanes share one address, which the
              hardware collapses into one operation when every lane is taken
    cut1r     no atomic at all, the keys summed in a register (the read and the
              key math alone)
    cut0      the clear and the per-program prologue
    cut0n     the same without the 2048-bin clear -- the floor looked like
              12961 rows x 8 KB of clearing at ~1.2 TB/s; this checks it

    cut1hash ~= cut1   -> transactions dominate
    cut1hash << cut1   -> conflicts dominate

The child also measures the data itself on row 0 of each shape: how many of
the 2048 bins are populated, the fullest bin, and within a 512-element tile how
many elements share their bin with another element of the same tile -- the
quantity conflicts are made of.

    tools/vendor_probe.sh tools/hygon_prefill_dense_atomics.py hygon_prefill_dense_atomics
"""

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_db", HERE / "hygon_prefill_dense_budget.py"
)
db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(db)

ARMS = ["base", "cut0", "cut0n", "cut1", "cut1hash", "cut1flat", "cut1r"]

EXTRA = r"""

def _fn_span(src, name):
    import ast as _ast

    for node in _ast.parse(src).body:
        if isinstance(node, _ast.FunctionDef) and node.name == name:
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            ls = src.splitlines(keepends=True)
            return sum(map(len, ls[: first - 1])), sum(map(len, ls[: node.end_lineno]))
    raise ValueError("no function " + name)


def _in_fn(src, name, old, new):
    a, b = _fn_span(src, name)
    body = src[a:b]
    if body.count(old) != 1:
        raise ValueError(f"{name}: {old!r} found {body.count(old)} times")
    return src[:a] + body.replace(old, new, 1) + src[b:]


ATOMIC_ADDR = "        s_histogram_ptr + bin_idx,\n"


def build_arm(src, arm):
    cut, readonly = int(arm[3]), arm == "cut1r"
    src = build_cut(src, cut, readonly)
    if arm == "cut1hash":
        src = _in_fn(
            src, "_distribute_to_bins", ATOMIC_ADDR,
            "        s_histogram_ptr\n"
            "        + ((_convert_to_uint32(logits) >> 11) & 0x7FF).to(tl.int32),\n",
        )
    elif arm == "cut1flat":
        src = _in_fn(
            src, "_distribute_to_bins", ATOMIC_ADDR, "        s_histogram_ptr + bin_idx * 0,\n"
        )
    elif arm == "cut0n":
        src = _in_fn(
            src, "_process_histogram_step",
            "    tl.store(s_histogram_ptr + radix_bins, tl.zeros([RADIX_SIZE], tl.int32))\n",
            "",
        )
    compile(src, f"<dense-{arm}>", "exec")
    return src
"""

INSTALL = r"""

def _install():
    import hashlib
    import os
    from importlib import import_module

    arm = os.environ.get("FLAGGEMS_PROBE_CUT_ARM", "base")
    ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
    if arm == "base":
        return
    with open(ov._VEC2_FINAL_PATH) as fh:
        source = build_arm(fh.read(), arm)
    base = ov._private_dir()
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    path = os.path.join(base, f"top_k_per_row_prefill_probe_{arm}_{digest}.py")
    if not os.path.exists(path):
        with open(path + ".tmp", "w") as fh:
            fh.write(source)
        os.replace(path + ".tmp", path)
    with open(path) as fh:
        assert fh.read() == source, "probe copy changed on disk"
    mod = ov._load_copy(f"flaggems_vllm.ops._top_k_per_row_prefill_probe_{arm}", path)
    ov._GENERIC_DEFAULTS[id(mod)] = (mod.NUM_THREADS_PER_BLOCK, mod._num_warps)
    ov._dense_vec2_final = mod


_install()
"""

PLUGIN = db.BUILDER + EXTRA + INSTALL

DATA = r"""
import torch

dev = "cuda"
for num_rows, vocab, stride0 in ((16383, 4095, 4352), (12961, 4100, 4360), (16380, 5115, 5376)):
    torch.manual_seed(42)
    buf = torch.randn((num_rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
    row = buf[:vocab]
    b = row.half().view(torch.int16).to(torch.int32) & 0xFFFF
    mapped = torch.where((b & 0x8000) != 0, b, (~b) & 0x7FFF)
    key = (mapped >> 5).long()
    h = torch.bincount(key, minlength=2048)
    shared = []
    for t0 in range(0, vocab - 511, 512):
        kt = key[t0 : t0 + 512]
        c = torch.bincount(kt, minlength=2048)
        shared.append(float((c[kt] > 1).float().mean()))
    print(
        f"DATA {num_rows}x{vocab}: populated bins {int((h > 0).sum())} of 2048, "
        f"fullest bin {int(h.max())}, mean per populated bin {float(h[h > 0].float().mean()):.1f}; "
        f"in a 512-tile {100 * sum(shared) / len(shared):.0f}% of elements share their bin"
    )
"""


def main():
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True
    ).stdout
    if dirty.strip():
        raise SystemExit("the source tree is modified:\n" + dirty)
    plugdir = tempfile.mkdtemp(prefix="hygon_atom_")
    (pathlib.Path(plugdir) / "hygon_cut_plugin.py").write_text(PLUGIN)

    print("### the data", flush=True)
    r = subprocess.run([sys.executable, "-c", DATA], capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if ln.startswith("DATA"):
            print("  " + ln[5:], flush=True)
    if r.returncode:
        print("\n".join(r.stderr.strip().splitlines()[-5:]), flush=True)

    res = {}
    for rep in range(2):
        order = ARMS if rep == 0 else ARMS[::-1]
        for arm in order:
            env = dict(os.environ)
            env["FLAGGEMS_PROBE_CUT_ARM"] = arm
            env["PYTHONPATH"] = plugdir + os.pathsep + env.get("PYTHONPATH", "")
            print(f"### rep {rep + 1}, arm {arm}", flush=True)
            r = subprocess.run(
                [sys.executable, "-c", db.CHILD],
                capture_output=True,
                text=True,
                env=env,
            )
            lines = [x[6:] for x in r.stdout.splitlines() if x.startswith("CHILD")]
            if len(lines) != 3:
                print("  ! failed:", flush=True)
                for ln in (r.stdout + r.stderr).strip().splitlines()[-10:]:
                    print(f"    | {ln[:200]}", flush=True)
                continue
            for ln in lines:
                print(f"      {ln}", flush=True)
                shape = ln.split()[0]
                us = float(ln.split("us=")[1].split()[0])
                regs = ln.split("regs=")[1].split()[0]
                res.setdefault((arm, shape), []).append((us, regs))

    shapes = ["16383x4095", "12961x4100", "16380x5115"]
    best = {k: min(u for u, _ in v) for k, v in res.items()}
    print("\n  us per call, min over reps\n")
    print("  " + f"{'arm':>9} " + " ".join(f"{s:>12}" for s in shapes) + "   regs")
    for arm in ARMS:
        cells = [
            f"{best[(arm, s)]:12.1f}" if (arm, s) in best else f"{'FAILED':>12}"
            for s in shapes
        ]
        regs = next((v[0][1] for (a, _), v in res.items() if a == arm), "?")
        print(f"  {arm:>9} " + " ".join(cells) + f"   {regs}")

    print("\n  what each difference prices, us\n")
    rows = [
        ("atomics, real bins (cut1 - cut1r)", "cut1r", "cut1"),
        ("atomics, hashed bins (cut1hash - cut1r)", "cut1r", "cut1hash"),
        ("atomics, one address (cut1flat - cut1r)", "cut1r", "cut1flat"),
        ("conflicts (cut1 - cut1hash)", "cut1hash", "cut1"),
        ("the 2048-bin clear (cut0 - cut0n)", "cut0n", "cut0"),
    ]
    for label, lo, hi in rows:
        cells = [
            (
                f"{best[(hi, s)] - best[(lo, s)]:12.1f}"
                if (hi, s) in best and (lo, s) in best
                else f"{'-':>12}"
            )
            for s in shapes
        ]
        print(f"  {label:<42}" + " ".join(cells))


if __name__ == "__main__":
    main()
