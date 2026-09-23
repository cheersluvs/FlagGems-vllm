"""Direction E: 256 STEP-0 bins, a register histogram, and a 512-wide final
network, on the three big dense prefill shapes.

WHY THE THREE HAVE TO GO TOGETHER. Narrowing STEP 0 has been measured twice
and lost on these shapes -- but both times with the global-atomic histogram,
and both times with the final network capped at 256:

  * the crossover probe behind SHORT_BINS_MAX_VOCAB: 512 bins win up to
    vocab 1536, are neutral at 1792 and regress on 4095/5115;
  * tools/hygon_prefill_cap_cost.py: at today's 27-50 threshold-bin
    candidates the network is nearly free (cap128 0.997), but at 256 bins the
    threshold bin holds ~217-322, and above 256 a row falls off the cascade
    into the O(m^2) counting ranker.

So E is: fewer bins to make STEP 0 cheaper, a register histogram so the
narrower histogram does not pay global atomics at all, and a 512 branch so
the bigger threshold bin still lands in a network.

WHAT I EXPECT, stated before the run: tl.histogram has not beaten a global
atomic ONCE on this operator (generic 2048 bins 8.5x slower; finish 256 bins
a wash; prepare 256+16 bins 8 us slower, from per-call cost). The one
argument for trying again is occupancy: prepare ran 64 programs, the dense
shapes run thousands, and a per-call latency can hide there where it could
not in prepare.

HOW. Nothing in the override file changes. A pytest plugin (-p) builds a new
module copy from the override's own validated VEC2 source -- the same way the
override builds its routes -- and installs it as `_dense_vec2_final`, which
is what these three shapes route to (rows >= 8192, top_k 512, vocab in
[2048, 5120]). Every other shape is untouched.

    base    the shipped route
    net512  VEC2 + final network with a 512 branch added, 2048 bins. The
            HARNESS CONTROL: same algorithm as base, so it must match base;
            the unused 512 branch is compiled in, so this also prices it
    n256    + STEP 0 narrowed to 256 bins, still global atomics
    n256r   + STEP 0's histogram in registers (tl.histogram, then one
            vector store so the scan reads it exactly as before)

The narrowing is the one the shipped 512-bin route already uses (the STEP-0
key shift and STEP 0's RADIX_SIZE), not the bins256 probe's, whose dense arms
answered wrong. The register histogram replaces all seven _distribute_to_bins
call sites in _process_histogram_step, for STEP 0 only; STEP 1-3 keep their
atomics. Out-of-row lanes already load -inf, key 252 of 256 -- the bottom,
which only an input with fewer than top_k finite values can reach, and those
rows then take STEP 1, which masks properly.

Correctness per arm on all three dense shapes: normal, narrow band (collapses
the STEP-0 key; forces STEP 1-3) and partial rows against torch.topk; the full
prefill suite; the benchmark twice with the arm order rotated.

    tools/vendor_probe.sh tools/hygon_prefill_dense_narrow.py hygon_prefill_dense_narrow
"""

import os
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile

PASSES = 2
BENCH = ["benchmark/test_top_k_per_row_prefill.py", "--mode", "kernel"]
TESTS = ["tests/test_top_k_per_row_prefill.py"]
DENSE = [(16383, 4095, 512), (12961, 4100, 512), (16380, 5115, 512)]
ARMS = ["base", "net512", "n256", "n256r"]

# ---------------------------------------------------------------------------
# The builder. Executed locally for the preflight and on the card inside the
# plugin, so both run the same text.

BUILDER = r'''
import re as _re


def _fn_span(src, name):
    import ast as _ast

    for node in _ast.parse(src).body:
        if isinstance(node, _ast.FunctionDef) and node.name == name:
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            lines = src.splitlines(keepends=True)
            return sum(map(len, lines[: first - 1])), sum(map(len, lines[: node.end_lineno]))
    raise ValueError("no function " + name)


def _once(src, old, new, what):
    n = src.count(old)
    if n != 1:
        raise ValueError(f"{what}: found {n} times")
    return src.replace(old, new, 1)


HELPER = """

@triton.jit
def _dist_or_hist(logits, in_range, ones, logit_pattern, s_histogram_ptr, h0,
                  STEP: tl.constexpr):
    # STEP 0 counts into a register histogram; STEP 1-3 keep their atomics.
    if STEP == 0:
        bin_idx, _match = _extract_bin_idx(logits, in_range, logit_pattern, STEP=STEP)
        h0 += tl.histogram(tl.ravel(bin_idx.to(tl.int32)), 256)
    else:
        _distribute_to_bins(
            logits, in_range, ones, logit_pattern, s_histogram_ptr, STEP=STEP
        )
    return h0
"""


def build_e_source(vec2_source, final_builder_source, arm):
    src = vec2_source
    if arm in ("n256", "n256r"):
        src = _once(
            src,
            "bin_idx = (mapped >> 5).to(tl.uint32)",
            "bin_idx = (mapped >> 8).to(tl.uint32)",
            "STEP-0 key",
        )
        src = _once(
            src,
            "RADIX_SIZE: tl.constexpr = RADIX10_SIZE if STEP == 3 else RADIX11_SIZE",
            "RADIX_SIZE: tl.constexpr = ("
            "RADIX10_SIZE if STEP == 3 else (256 if STEP == 0 else RADIX11_SIZE))",
            "STEP-0 radix",
        )
    if arm == "n256r":
        a, b = _fn_span(src, "_process_histogram_step")
        fn = src[a:b]
        clear = (
            "    tl.store(s_histogram_ptr + radix_bins, tl.zeros([RADIX_SIZE], tl.int32))\n"
            "    tl.debug_barrier()\n"
        )
        end = "    last_value = tl.load(s_found_topk_values_ptr)\n"
        i0 = fn.index(clear) + len(clear)
        i1 = fn.index(end)
        region = fn[i0:i1]
        if region.count("_distribute_to_bins(") != 7:
            raise ValueError("histogram call sites moved")
        region = region.replace("_distribute_to_bins(", "h0 = _dist_or_hist(")
        region, n = _re.subn(
            r"s_histogram_ptr,(\s*)STEP=STEP,", r"s_histogram_ptr,\1h0,\1STEP=STEP,", region
        )
        if n != 7:
            raise ValueError(f"call tails: {n}")
        fn = (
            fn[:i0]
            + "    h0 = tl.zeros([256], tl.int32)\n"
            + region
            + "    if STEP == 0:\n"
            + "        tl.store(s_histogram_ptr + tl.arange(0, 256), h0)\n"
            + fn[i1:]
        )
        src = src[:a] + fn + src[b:] + HELPER
    ns = {}
    exec(
        _once(
            final_builder_source,
            "for i, cap in enumerate((64, 128, 256)):",
            "for i, cap in enumerate((64, 128, 256, 512)):",
            "final network caps",
        ),
        ns,
    )
    src = ns["build_final_source"](src)
    compile(src, f"<hygon-prefill-{arm}>", "exec")
    return src
'''

PLUGIN = (
    BUILDER
    + r"""

def _install():
    import hashlib
    import os
    from importlib import import_module

    arm = os.environ.get("FLAGGEMS_PROBE_E_ARM", "base")
    ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
    if arm == "base":
        return
    fsrc_mod = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused._top_k_per_row_prefill_final_source"
    )
    with open(ov._VEC2_PATH) as fh:
        vec2 = fh.read()
    with open(fsrc_mod.__file__) as fh:
        fbuilder = fh.read()
    source = build_e_source(vec2, fbuilder, arm)
    base = ov._private_dir()
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    path = os.path.join(base, f"top_k_per_row_prefill_probe_e_{arm}_{digest}.py")
    if not os.path.exists(path):
        with open(path + ".tmp", "w") as fh:
            fh.write(source)
        os.replace(path + ".tmp", path)
    with open(path) as fh:
        assert fh.read() == source, "probe copy changed on disk"
    mod = ov._load_copy(f"flaggems_vllm.ops._top_k_per_row_prefill_probe_e_{arm}", path)
    ov._GENERIC_DEFAULTS[id(mod)] = (mod.NUM_THREADS_PER_BLOCK, mod._num_warps)
    ov._dense_vec2_final = mod


_install()
"""
)

CHILD = r"""
import torch, triton
import hygon_e_plugin  # installs the arm
from importlib import import_module

ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
dev = "cuda"
print("ROUTE", ov._dense_vec2_final.__name__)


def inputs(num_rows, vocab, stride0, kind):
    torch.manual_seed(42)
    buf = torch.randn((num_rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
    if kind == "band":
        buf = 10.0 + 0.2 * torch.rand_like(buf)
    x = torch.as_strided(buf, (num_rows, vocab), (stride0, 1))
    assert x.stride(0) == stride0 and x.stride(1) == 1
    if kind == "partial":
        g = torch.Generator(device="cpu").manual_seed(7)
        st = torch.randint(0, 200, (num_rows,), generator=g).to(torch.int32)
        en = (vocab - torch.randint(0, 200, (num_rows,), generator=g)).to(torch.int32)
    else:
        st = torch.zeros(num_rows, dtype=torch.int32)
        en = torch.full((num_rows,), vocab, dtype=torch.int32)
    return x, st.to(dev), en.to(dev)


def check(num_rows, vocab, top_k, stride0, kind):
    x, st, en = inputs(num_rows, vocab, stride0, kind)
    out = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
    ov.top_k_per_row_prefill(x, st, en, out, num_rows, stride0, 1, top_k)
    torch.cuda.synchronize()
    col = torch.arange(vocab, device=dev)[None, :]
    inside = (col >= st[:, None].long()) & (col < en[:, None].long())
    ref = torch.topk(x.masked_fill(~inside, float("-inf")), top_k, dim=1).values
    pads = int((out < 0).sum())
    got = torch.gather(x, 1, st[:, None].long() + out.long().clamp(min=0))
    err = float((got.sort(dim=1, descending=True)[0] - ref).abs().max())
    del x, st, en, out, col, inside, ref, got
    torch.cuda.empty_cache()
    return "ok" if err == 0.0 and pads == 0 else f"WRONG({err:.1e},{pads})"


for num_rows, vocab, top_k, stride0 in (
    (16383, 4095, 512, 4352), (12961, 4100, 512, 4360), (16380, 5115, 512, 5376)
):
    checks = [check(num_rows, vocab, top_k, stride0, k) for k in ("normal", "band", "partial")]
    x, st, en = inputs(num_rows, vocab, stride0, "normal")
    out = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
    f = lambda: ov.top_k_per_row_prefill(x, st, en, out, num_rows, stride0, 1, top_k)
    t = min(triton.testing.do_bench(f, warmup=20, rep=200) for _ in range(3)) * 1e3
    print(f"CHILD {num_rows}x{vocab} us={t:.1f} checks={'/'.join(checks)}")
    del x, st, en, out
    torch.cuda.empty_cache()
"""


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True)


def occupancy(tag):
    import shutil

    for cmd in (["hy-smi"], ["rocm-smi"]):
        exe = shutil.which(cmd[0]) or (
            f"/opt/dtk/bin/{cmd[0]}"
            if pathlib.Path(f"/opt/dtk/bin/{cmd[0]}").exists()
            else None
        )
        if not exe:
            continue
        print(f"--- card occupancy {tag}: {cmd[0]}")
        print("\n".join(sh(exe, *cmd[1:]).stdout.strip().splitlines()[:25]))
        return
    print(f"--- card occupancy {tag}: no smi tool found")


def parse_bench(out):
    rows = {}
    for mm in re.finditer(
        r"SUCCESS\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[torch\.Size\(\[(\d+), (\d+)\]\)"
        r".*?, (\d+), (\d+), 1, (\d+)\]",
        out,
    ):
        rows[(int(mm.group(4)), int(mm.group(5)), int(mm.group(8)))] = (
            float(mm.group(3)),
            float(mm.group(1)),
        )
    return rows


def parse_tests(out):
    p = re.search(r"(\d+) passed", out)
    k = re.search(r"(\d+) skipped", out)
    f = re.search(r"(\d+) failed", out)
    if not (p or f):
        return "no result"
    s = f"{p.group(1) if p else 0} passed"
    if k:
        s += f", {k.group(1)} skipped"
    if f:
        s += f", {f.group(1)} FAILED"
    return s


def why(r):
    text = (r.stdout + r.stderr).splitlines()
    hits = [
        ln.strip()
        for ln in text
        if ("Error" in ln or "error:" in ln) and "error_msg = str(e)" not in ln
    ]
    return (hits[-1] if hits else "no output")[:220], text[-12:]


def geo(vals):
    g = 1.0
    for v in vals:
        g *= v
    return g ** (1.0 / len(vals))


def main():
    dirty = sh("git", "status", "--porcelain", "--", "src").stdout
    if dirty.strip():
        raise SystemExit("the source tree is modified:\n" + dirty)
    plugdir = tempfile.mkdtemp(prefix="hygon_e_")  # 0700, this user's
    (pathlib.Path(plugdir) / "hygon_e_plugin.py").write_text(PLUGIN)

    def env_for(arm):
        e = dict(os.environ)
        e["FLAGGEMS_PROBE_E_ARM"] = arm
        e["PYTHONPATH"] = plugdir + os.pathsep + e.get("PYTHONPATH", "")
        return e

    occupancy("before")
    child, tests, bench, broken = {}, {}, {a: [] for a in ARMS}, {}
    for arm in ARMS:
        print(f"### child (route, checks, do_bench), arm {arm}", flush=True)
        r = subprocess.run(
            [sys.executable, "-c", CHILD],
            capture_output=True,
            text=True,
            env=env_for(arm),
        )
        lines = [x for x in r.stdout.splitlines() if x.startswith(("CHILD", "ROUTE"))]
        if len(lines) != 1 + len(DENSE):
            broken[arm], tail = why(r)
            print(f"      ! {arm}: {broken[arm]}", flush=True)
            for ln in tail:
                print(f"        | {ln[:200]}", flush=True)
            continue
        child[arm] = lines
        for ln in lines:
            print(f"      {ln}", flush=True)

        print(f"### tests, arm {arm}", flush=True)
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-rf", "-p", "hygon_e_plugin"]
            + TESTS,
            capture_output=True,
            text=True,
            env=env_for(arm),
        )
        tests[arm] = parse_tests(r.stdout)
        print(f"      {tests[arm]}", flush=True)
        for ln in [x for x in r.stdout.splitlines() if x.startswith("FAILED")][:3]:
            print(f"        {ln[:220]}", flush=True)

    live = [a for a in ARMS if a not in broken]
    for p in range(PASSES):
        order = live[p % len(live) :] + live[: p % len(live)] if live else []
        for arm in order:
            print(f"### benchmark pass {p + 1}, arm {arm}", flush=True)
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-s", "-p", "hygon_e_plugin"]
                + BENCH,
                capture_output=True,
                text=True,
                env=env_for(arm),
            )
            rows = parse_bench(r.stdout)
            if not rows:
                broken[arm], tail = why(r)
                print(f"      ! {arm}: {broken[arm]}", flush=True)
                continue
            bench[arm].append(rows)

    good = [a for a in ARMS if len(bench[a]) == PASSES]
    print("\n  do_bench on our op alone, us (min of 3), and correctness\n")
    for arm in ARMS:
        if arm in child:
            print(
                f"  {arm:>7}: " + " | ".join(ln.split(" ", 1)[1] for ln in child[arm])
            )
        else:
            print(f"  {arm:>7}: FAILED {broken.get(arm, '?')}")

    print("\n  benchmark SpeedUp, pass 1 / pass 2\n")
    head = "  " + f"{'arm':>7} " + " ".join(f"{r}x{v:<14}" for r, v, _ in DENSE)
    print(head + "  dense-3 geo   vs base   tests")
    b0 = None
    for arm in ARMS:
        if arm not in good:
            print(f"  {arm:>7}   FAILED: {broken.get(arm, 'incomplete')}")
            continue
        cells, g3 = [], []
        for shp in DENSE:
            v = [bench[arm][p][shp][0] for p in range(PASSES)]
            cells.append(f"{v[0]:.3f}/{v[1]:.3f}")
        for p in range(PASSES):
            g3.append(geo([bench[arm][p][s][0] for s in DENSE]))
        g = statistics.mean(g3)
        if arm == "base":
            b0 = g
        rel = f"{g / b0:8.3f}" if b0 else "       -"
        print(
            f"  {arm:>7} "
            + " ".join(f"{c:>20}" for c in cells)
            + f"  {g:10.3f}  {rel}   {tests.get(arm, '-')}"
        )

    print("\n  every other shape must not move (geomean of the other four):")
    for arm in good:
        gm = [
            geo([v[0] for k, v in bench[arm][p].items() if k not in DENSE])
            for p in range(PASSES)
        ]
        print(f"      {arm:>7}: " + " / ".join(f"{x:.3f}" for x in gm))
    if good:
        for shp in DENSE:
            lat = [bench[a][p][shp][1] for a in good for p in range(PASSES)]
            print(
                f"  vLLM latency on {shp[0]}x{shp[1]}, max/min {max(lat) / min(lat):.2f}"
            )
    occupancy("after")


if __name__ == "__main__":
    main()
