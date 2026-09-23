"""Where the time goes in the dense prefill kernel, phase by phase.

The dense shapes are one kernel, so the sampled path's method (time each
launch) does not apply; this truncates the kernel instead. STEP 0 is four
phases, and every benchmark dense row finishes inside STEP 0:

    P1  clear + histogram   first read of the row, one global atomic per element
    P2  threshold scan      one RADIX_SIZE-wide cumsum
    P3  collection          second read; prefix-sum slots for the sure ones, the
                            threshold bin to a final buffer
    P4  final select        the small network over the threshold bin (27-50)

cutK keeps phases up to K and drops everything after. Each arm is WRONG, but
every one is a PREFIX: nothing downstream of the cut runs, so nothing
downstream can do different work -- the confound that made the `flat` and
ablation arms on the sampled path misleading does not arise. Differences of
successive cuts give each phase's cost.

    base    the shipped module, untouched
    cut4    the same kernel rebuilt through this harness -- must equal base,
            and must answer correctly; it validates the harness
    cut3    no final select
    cut2    no collection
    cut1    histogram only
    cut1r   P1 with the atomics replaced by a register sum of the keys, so the
            first read and the key math remain and the atomics do not:
            cut1 - cut1r prices the histogram atomics
    cut0    the clear and the per-program prologue only: the floor

The one confound that remains is the compiler: a truncated kernel may use
fewer registers and so run more waves. The compiled kernel's register and
spill counts are printed per arm so that can be seen rather than assumed.

Context: an older rocprof run on this route (reports/hygon_rocprof_prefill2)
fetched 2.13x the row bytes, i.e. BOTH reads reach HBM, where vLLM plausibly
reads once. This measures what each read actually costs now.

    tools/vendor_probe.sh tools/hygon_prefill_dense_budget.py hygon_prefill_dense_budget
"""

import os
import pathlib
import subprocess
import sys
import tempfile

ARMS = ["base", "cut4", "cut3", "cut2", "cut1", "cut1r", "cut0"]

BUILDER = r'''
import re as _re


def _lines_of(src, name):
    import ast as _ast

    for node in _ast.parse(src).body:
        if isinstance(node, _ast.FunctionDef) and node.name == name:
            return node.lineno, node.end_lineno
    raise ValueError("no function " + name)


def _indent(block, n=4):
    return "\n".join((" " * n + l) if l.strip() else l for l in block.split("\n"))


TOUCH = """

@triton.jit
def _probe_touch(logits, in_range, ones, logit_pattern, s_histogram_ptr, h_ro,
                 STEP: tl.constexpr):
    bin_idx, _m = _extract_bin_idx(logits, in_range, logit_pattern, STEP=STEP)
    return h_ro + tl.sum(tl.ravel(bin_idx.to(tl.int32)), axis=0)
"""


def build_cut(src, cut, readonly):
    lines = src.split("\n")
    a, b = _lines_of(src, "_process_histogram_step")

    def find(text, lo, hi):
        hits = [i for i in range(lo - 1, hi) if lines[i] == text]
        if len(hits) != 1:
            raise ValueError(f"{text!r}: {len(hits)} hits")
        return hits[0]

    clr = find(
        "    tl.store(s_histogram_ptr + radix_bins, tl.zeros([RADIX_SIZE], tl.int32))",
        a, b,
    )
    if lines[clr + 1] != "    tl.debug_barrier()":
        raise ValueError("clear barrier moved")
    p1 = clr + 2
    p2 = find("    last_value = tl.load(s_found_topk_values_ptr)", a, b)
    p3 = find("    found_ptrs = s_found_topk_values_ptr + zeros", a, b)
    p3e = find("    tl.store(s_found_topk_values_ptr, slot_base)", a, b)

    P1 = "\n".join(lines[p1:p2])
    if readonly:
        if P1.count("_distribute_to_bins(") != 7:
            raise ValueError("histogram call sites moved")
        P1 = P1.replace("_distribute_to_bins(", "h_ro = _probe_touch(")
        P1, n = _re.subn(
            r"s_histogram_ptr,(\s*)STEP=STEP,", r"s_histogram_ptr,\1h_ro,\1STEP=STEP,", P1
        )
        if n != 7:
            raise ValueError(f"call tails {n}")
        P1 = (
            "    h_ro = tl.zeros((), tl.int32)\n" + P1
            + "\n    tl.store(s_histogram_ptr, h_ro)"
        )
    P2 = "\n".join(lines[p2:p3])
    P3 = "\n".join(lines[p3 : p3e + 1])
    step = (
        "    if _PROBE_CUT >= 1:\n" + _indent(P1) + "\n"
        + "    if _PROBE_CUT >= 2:\n" + _indent(P2) + "\n"
        + "        if _PROBE_CUT >= 3:\n" + _indent(P3, 8) + "\n"
        + "    else:\n"
        + "        final_bin_size = tl.full((), 0, tl.int32)"
    )
    lines = lines[:p1] + step.split("\n") + lines[p3e + 1 :]
    src = "\n".join(lines)

    lines = src.split("\n")
    a, b = _lines_of(src, "_top_k_per_row_job")
    f0 = [i for i in range(a - 1, b) if lines[i] == "    if not continue_to_next_step:"]
    f1 = [
        i for i in range(a - 1, b)
        if lines[i] == "    # out_indices_ptr is identical to s_out_indices_ptr for non-tle"
    ]
    if len(f0) != 1 or len(f1) != 1:
        raise ValueError("final-select block moved")
    blk = "\n".join(lines[f0[0] : f1[0]]).rstrip("\n")
    lines = (
        lines[: f0[0]]
        + ("    if _PROBE_CUT >= 4:\n" + _indent(blk) + "\n").split("\n")
        + lines[f1[0] :]
    )
    src = "\n".join(lines)
    anchor = "import triton.language as tl\n"
    if src.count(anchor) != 1:
        raise ValueError("import anchor")
    src = src.replace(anchor, anchor + f"\n_PROBE_CUT = tl.constexpr({cut})\n", 1)
    if readonly:
        src += TOUCH
    compile(src, f"<dense-cut{cut}{'r' if readonly else ''}>", "exec")
    return src
'''

PLUGIN = (
    BUILDER
    + r"""

def _install():
    import hashlib
    import os
    from importlib import import_module

    arm = os.environ.get("FLAGGEMS_PROBE_CUT_ARM", "base")
    ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
    if arm == "base":
        return
    cut, readonly = int(arm[3]), arm.endswith("r")
    with open(ov._VEC2_FINAL_PATH) as fh:
        source = build_cut(fh.read(), cut, readonly)
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
)

CHILD = r"""
import os, torch, triton
import hygon_cut_plugin  # installs the arm
from importlib import import_module

ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
arm = os.environ["FLAGGEMS_PROBE_CUT_ARM"]
dev = "cuda"
for num_rows, vocab, top_k, stride0 in (
    (16383, 4095, 512, 4352), (12961, 4100, 512, 4360), (16380, 5115, 512, 5376)
):
    torch.manual_seed(42)
    buf = torch.randn((num_rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
    x = torch.as_strided(buf, (num_rows, vocab), (stride0, 1))
    st = torch.zeros(num_rows, dtype=torch.int32, device=dev)
    en = torch.full((num_rows,), vocab, dtype=torch.int32, device=dev)
    out = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
    k = ov.top_k_per_row_prefill(x, st, en, out, num_rows, stride0, 1, top_k)
    torch.cuda.synchronize()
    ok = "-"
    if arm in ("base", "cut4"):
        ref = torch.topk(x, top_k, dim=1).values
        got = torch.gather(x, 1, out.long().clamp(min=0)).sort(dim=1, descending=True)[0]
        ok = "ok" if float((got - ref).abs().max()) == 0.0 and int((out < 0).sum()) == 0 else "WRONG"
    f = lambda: ov.top_k_per_row_prefill(x, st, en, out, num_rows, stride0, 1, top_k)
    t = min(triton.testing.do_bench(f, warmup=20, rep=300) for _ in range(3)) * 1e3
    regs = getattr(k, "n_regs", "?")
    spills = getattr(k, "n_spills", "?")
    print(f"CHILD {num_rows}x{vocab} us={t:.1f} regs={regs} spills={spills} answer={ok}"
          f" route={ov._select_module(x, num_rows, top_k).__name__.rsplit('.', 1)[-1]}")
    del buf, x, out
    torch.cuda.empty_cache()
"""


def main():
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True
    ).stdout
    if dirty.strip():
        raise SystemExit("the source tree is modified:\n" + dirty)
    plugdir = tempfile.mkdtemp(prefix="hygon_cut_")
    (pathlib.Path(plugdir) / "hygon_cut_plugin.py").write_text(PLUGIN)

    res = {}
    for rep in range(2):
        order = ARMS if rep == 0 else ARMS[::-1]
        for arm in order:
            env = dict(os.environ)
            env["FLAGGEMS_PROBE_CUT_ARM"] = arm
            env["PYTHONPATH"] = plugdir + os.pathsep + env.get("PYTHONPATH", "")
            print(f"### rep {rep + 1}, arm {arm}", flush=True)
            r = subprocess.run(
                [sys.executable, "-c", CHILD], capture_output=True, text=True, env=env
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
                res.setdefault((arm, shape), []).append((us, ln))

    shapes = ["16383x4095", "12961x4100", "16380x5115"]
    print("\n  us per call, min over reps (do_bench, our op alone)\n")
    print(
        "  " + f"{'arm':>6} " + " ".join(f"{s:>12}" for s in shapes) + "   regs/spills"
    )
    best = {}
    for arm in ARMS:
        cells, extra = [], ""
        for s in shapes:
            v = res.get((arm, s))
            if v:
                best[(arm, s)] = min(x[0] for x in v)
                cells.append(f"{best[(arm, s)]:12.1f}")
                ln = v[0][1]
                extra = (
                    ln.split("regs=")[1].split()[0]
                    + "/"
                    + ln.split("spills=")[1].split()[0]
                )
            else:
                cells.append(f"{'FAILED':>12}")
        print(f"  {arm:>6} " + " ".join(cells) + f"   {extra}")

    print("\n  phase costs, us (differences of successive cuts)\n")
    phases = [
        ("floor (cut0)", None, "cut0"),
        ("P1 histogram (cut1-cut0)", "cut0", "cut1"),
        ("   of which atomics (cut1-cut1r)", "cut1r", "cut1"),
        ("P2 scan (cut2-cut1)", "cut1", "cut2"),
        ("P3 collection (cut3-cut2)", "cut2", "cut3"),
        ("P4 final select (cut4-cut3)", "cut3", "cut4"),
        ("harness check (cut4-base)", "base", "cut4"),
    ]
    for label, lo, hi in phases:
        cells = []
        for s in shapes:
            if (hi, s) in best and (lo is None or (lo, s) in best):
                v = best[(hi, s)] - (best[(lo, s)] if lo else 0.0)
                tot = best.get(("cut4", s))
                share = f" ({100 * v / tot:4.1f}%)" if tot and lo != "base" else ""
                cells.append(f"{v:8.1f}{share:>8}")
            else:
                cells.append(f"{'-':>16}")
        print(f"  {label:<34}" + " ".join(cells))


if __name__ == "__main__":
    main()
