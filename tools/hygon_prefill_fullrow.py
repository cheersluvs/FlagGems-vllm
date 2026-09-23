"""Make the full-row geometry a constexpr instead of a runtime branch.

WHAT IS ACTUALLY THERE. `_top_k_per_row_job` computes

    assume_aligned = ((row_start == 0) & (row_end == vocab_size)
                      & (stride1 == 1) & ((vocab_size % BLOCK_SIZE) == 0))

at RUNTIME and then uses `tl.assume` hints. Those help the compiler inside a
branch; they do not remove the others. So every program carries all three
collection bodies -- the aligned one, the `stride1 == 1` one with its
rem_tiles and rem_elems loops, and the fully strided scalar one -- compiled
in, four times over (`for step_idx in tl.static_range(0, 4)`), plus the
`row_len <= TOPK` early-out. Register allocation answers to the widest of
them whether or not it runs.

Three separate constexprs can collapse that, and they are NOT equally cheap
to obtain, so the arms are a ladder and each adds exactly one:

    base    FULL_ROW=0 STRIDE1=0 VOCAB=0 -- the constexprs added but unused.
            A no-op control: it also proves the patch itself costs nothing.
    s1      + STRIDE1 = logits.stride(1). **FREE**: it is a Python int on the
            host already. Kills the strided branch outright.
    fr      + FULL_ROW: row_start = 0, row_end = vocab for every row. NOT free
            -- those live in device tensors and the host cannot read them
            without a sync, which `--mode kernel` would not show. This arm is
            a CEILING, not a proposal.
    frv     + VOCAB = logits.shape[1], also a host-side Python int. With it,
            `assume_aligned`, `row_len`, and every remainder trip count fold
            at compile time. This is the whole ceiling.

WHY MEASURE THE CEILING BEFORE SOLVING THE GATE. If the ladder tops out near
1%, the gating question never needs an answer. If it does not, the follow-up
is whether vLLM's prefill ever passes a partial row -- a contract question
about the caller, not a sync.

CORRECTNESS. `fr` and `frv` are wrong for partial rows by construction, and
the suite has partial-range cases, so every arm runs the FULL TEST SUITE as
well as the benchmark and the table prints both. `s1` should pass everything;
if it does not, it is not free after all.

SCOPE. (64,129280) is routed to the sampled path and never reaches this
kernel, so it must NOT move -- it is a control here. The other six shapes are
the measurement, which is the inverse of every probe for the last two days.

This patches the GENERIC source, which is what all five override copies are
built from, so the specialisation reaches every dense route. The probe checks
that the override's own text transforms still find their anchors afterwards.

RECOVERY, if killed between the swap and the restore:

    git checkout -- src/flaggems_vllm/ops/top_k_per_row_prefill.py \
        src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py

    tools/vendor_probe.sh tools/hygon_prefill_fullrow.py hygon_prefill_fullrow
"""

import os
import pathlib
import re
import subprocess
import sys

GENERIC = pathlib.Path("src/flaggems_vllm/ops/top_k_per_row_prefill.py")
OVERRIDE = pathlib.Path(
    "src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py"
)

# Round 1 patched only the generic launch and every arm, `base` included,
# failed identically: with scratch reuse on (the default) EVERY non-sampled
# production call goes through `_top_k_per_row_prefill_reuse`, which launches
# `mod.non_tle_top_k_per_row_prefill` itself and bypasses the generic host
# function. So the new constexprs were missing there -- and, worse, had they
# had defaults, production would have silently run the UNspecialised kernel
# and this probe would have reported "no effect". Changing a kernel's
# signature needs an audit of its CALLERS, not just of the patch anchors.
REUSE_OLD = """        TOPK=top_k,
        BLOCK_SIZE=mod.NUM_THREADS_PER_BLOCK,
        ROW_OFFSET=0,
        num_warps=mod._num_warps(mod.NUM_THREADS_PER_BLOCK),
    )"""
REUSE_NEW = """        TOPK=top_k,
        BLOCK_SIZE=mod.NUM_THREADS_PER_BLOCK,
        ROW_OFFSET=0,
        FULL_ROW=bool(mod._SPEC_FULL_ROW),
        STRIDE1=(stride1 if mod._SPEC_STRIDE1 else 0),
        VOCAB=(logits.shape[1] if mod._SPEC_VOCAB else 0),
        num_warps=mod._num_warps(mod.NUM_THREADS_PER_BLOCK),
    )"""


def patched_override(src):
    n = src.count(REUSE_OLD)
    assert n == 1, f"the override's reuse launch found {n} times"
    return src.replace(REUSE_OLD, REUSE_NEW, 1)


def preflight(generic_src, override_src):
    """Every launch of the kernel whose signature changed, and whether it
    forwards the knobs. The production one MUST; any other relies on the
    defaults and is listed so it is a decision, not an accident."""
    import ast

    sites = []
    for label, src in (("generic", generic_src), ("override", override_src)):
        for node in ast.walk(ast.parse(src)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Subscript)
                and ast.unparse(node.func.value).endswith(
                    "non_tle_top_k_per_row_prefill"
                )
            ):
                kw = {k.arg for k in node.keywords}
                sites.append(
                    (label, node.lineno, {"FULL_ROW", "STRIDE1", "VOCAB"} <= kw)
                )
    for label, line, ok in sites:
        print(f"      {label}:{line}  forwards knobs: {ok}")
    prod = [s for s in sites if s[0] == "override"]
    assert prod and all(s[2] for s in prod), "the production launch does not forward"
    return sites


PASSES = 2
BENCH = ["benchmark/test_top_k_per_row_prefill.py", "--mode", "kernel"]
TESTS = ["tests/test_top_k_per_row_prefill.py"]
SAMPLED = (64, 129280, 1024)

# (tag, FULL_ROW, STRIDE1, VOCAB) -- 1 means "take the host's value"
ARMS = [
    ("base", 0, 0, 0),
    ("s1", 0, 1, 0),
    ("fr", 1, 1, 0),
    ("frv", 1, 1, 1),
]

KNOBS = """
# --- probe: tools/hygon_prefill_fullrow.py, reverted after the run ---
_SPEC_FULL_ROW = int(os.environ.get("FLAGGEMS_SPEC_FULL_ROW", "0"))
_SPEC_STRIDE1 = int(os.environ.get("FLAGGEMS_SPEC_STRIDE1", "0"))
_SPEC_VOCAB = int(os.environ.get("FLAGGEMS_SPEC_VOCAB", "0"))
"""

SIG_OLD = """    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
):
    VEC: tl.constexpr = 4
    NUM_BINS: tl.constexpr = 2048
    NUM_FILNAL_ITEMS: tl.constexpr = 2048"""
SIG_NEW = """    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
    FULL_ROW: tl.constexpr = False,
    STRIDE1: tl.constexpr = 0,
    VOCAB: tl.constexpr = 0,
):
    VEC: tl.constexpr = 4
    NUM_BINS: tl.constexpr = 2048
    NUM_FILNAL_ITEMS: tl.constexpr = 2048"""

# FULL_ROW, STRIDE1 and VOCAB are constexpr, so these are Python ifs and only
# one side is ever traced -- which is why rebinding the PARAMETERS `vocab_size`
# and `stride1` is safe here and does not create the constexpr/int32 type merge
# that has bitten this file before. It also means the two textually identical
# `_top_k_per_row_job` call sites need no patch of their own.
ROW_OLD = """    row_id = tl.program_id(0) + ROW_OFFSET
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    logits_ptr += row_id * stride0
    # float4 align
    x_off_mod = (row_id * stride0 + row_start) % VEC
    skip_elems = 0 if x_off_mod == 0 else VEC - x_off_mod
    out_indices_ptr += row_id * TOPK

    s_histogram_ptr += row_id * NUM_BINS"""
ROW_NEW = """    row_id = tl.program_id(0) + ROW_OFFSET
    if VOCAB > 0:
        vocab_size = VOCAB
    if STRIDE1 > 0:
        stride1 = STRIDE1
    if FULL_ROW:
        row_start = 0
        row_end = vocab_size
    else:
        row_start = tl.load(row_starts + row_id)
        row_end = tl.load(row_ends + row_id)
    logits_ptr += row_id * stride0
    # float4 align
    x_off_mod = (row_id * stride0 + row_start) % VEC
    skip_elems = 0 if x_off_mod == 0 else VEC - x_off_mod
    out_indices_ptr += row_id * TOPK

    s_histogram_ptr += row_id * NUM_BINS"""

LAUNCH_OLD = """            TOPK=top_k,
            BLOCK_SIZE=NUM_THREADS_PER_BLOCK,
            ROW_OFFSET=0,
            num_warps=_num_warps(NUM_THREADS_PER_BLOCK),
        )"""
LAUNCH_NEW = """            TOPK=top_k,
            BLOCK_SIZE=NUM_THREADS_PER_BLOCK,
            ROW_OFFSET=0,
            FULL_ROW=bool(_SPEC_FULL_ROW),
            STRIDE1=(stride1 if _SPEC_STRIDE1 else 0),
            VOCAB=(vocab_size if _SPEC_VOCAB else 0),
            num_warps=_num_warps(NUM_THREADS_PER_BLOCK),
        )"""


def patched(src):
    assert "\nimport os\n" in src, "the generic module does not import os"
    anchor = "\nlogger = "
    if anchor not in src:
        anchor = "\ndef _use_radix_final_for_prefill"
    assert src.count(anchor) >= 1, "no place to put the knobs"
    i = src.index(anchor)
    src = src[:i] + "\n" + KNOBS + src[i:]
    for old, new in (
        (SIG_OLD, SIG_NEW),
        (ROW_OLD, ROW_NEW),
        (LAUNCH_OLD, LAUNCH_NEW),
    ):
        n = src.count(old)
        assert n == 1, f"anchor found {n} times: {old.splitlines()[0]!r}"
        src = src.replace(old, new, 1)
    return src


def env_for(arm):
    e = dict(os.environ)
    for name, v in zip(("FULL_ROW", "STRIDE1", "VOCAB"), arm[1:]):
        e[f"FLAGGEMS_SPEC_{name}"] = str(v)
    return e


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
    for m in re.finditer(
        r"SUCCESS\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[torch\.Size\(\[(\d+), (\d+)\]\)"
        r".*?, (\d+), (\d+), 1, (\d+)\]",
        out,
    ):
        rows[(int(m.group(4)), int(m.group(5)), int(m.group(8)))] = (
            float(m.group(3)),
            float(m.group(1)),
        )
    return rows


def parse_tests(out):
    m = re.search(r"(\d+) passed(?:, (\d+) skipped)?", out)
    f = re.search(r"(\d+) failed", out)
    if not m and not f:
        return "no result"
    return (
        f"{m.group(1) if m else 0} passed"
        + (f", {m.group(2)} skipped" if m and m.group(2) else "")
        + (f", {f.group(1)} FAILED" if f else "")
    )


def geo(vals):
    g = 1.0
    for v in vals:
        g *= v
    return g ** (1.0 / len(vals))


def main():
    dirty = sh("git", "status", "--porcelain", "--", str(GENERIC), str(OVERRIDE)).stdout
    if dirty.strip():
        raise SystemExit("the operator files are already modified:\n" + dirty)
    pristine = GENERIC.read_text()
    pristine_ov = OVERRIDE.read_text()
    patch = patched(pristine)
    patch_ov = patched_override(pristine_ov)
    print("### preflight: every launch of the re-signed kernel", flush=True)
    preflight(patch, patch_ov)
    occupancy("before")

    bench, tests, broken = {a[0]: [] for a in ARMS}, {}, {}
    try:
        GENERIC.write_text(patch)
        OVERRIDE.write_text(patch_ov)

        # the override rebuilds its five copies from this source; if any of its
        # own anchors moved it logs a warning and silently ships generic
        print("### override still finds its anchors?", flush=True)
        chk = subprocess.run(
            [
                sys.executable,
                "-c",
                "import logging,sys;logging.basicConfig(level=logging.WARNING);"
                "from importlib import import_module;"
                "m=import_module('flaggems_vllm.runtime.backend._hygon.fused."
                "top_k_per_row_prefill');"
                "print('ROUTES', [k for k in dir(m) if k.endswith('_NAME')])",
            ],
            capture_output=True,
            text=True,
            env=env_for(ARMS[0]),
        )
        print((chk.stdout + chk.stderr).strip()[-600:] or "(no output)", flush=True)

        for arm in ARMS:
            tag = arm[0]
            print(f"### tests, arm {tag}", flush=True)
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-rf"] + TESTS,
                capture_output=True,
                text=True,
                env=env_for(arm),
            )
            tests[tag] = parse_tests(r.stdout)
            print(f"      {tag}: {tests[tag]}", flush=True)
            for ln in [x for x in r.stdout.splitlines() if x.startswith("FAILED")][:4]:
                print(f"        {ln[:220]}", flush=True)

        for p in range(PASSES):
            for arm in ARMS:
                tag = arm[0]
                if tag in broken:
                    continue
                print(f"### benchmark pass {p + 1}, arm {tag}", flush=True)
                r = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "-s"] + BENCH,
                    capture_output=True,
                    text=True,
                    env=env_for(arm),
                )
                rows = parse_bench(r.stdout)
                if not rows:
                    text = (r.stdout + r.stderr).splitlines()
                    why = [
                        ln.strip()
                        for ln in text
                        if ("Error" in ln or "error:" in ln)
                        and "error_msg = str(e)" not in ln
                    ]
                    broken[tag] = why[-1][:220] if why else "no SUCCESS rows"
                    print(f"      ! {tag}: {broken[tag]}", flush=True)
                    for ln in text[-12:]:
                        print(f"        | {ln[:200]}", flush=True)
                    continue
                bench[tag].append(rows)
    finally:
        GENERIC.write_text(pristine)
        OVERRIDE.write_text(pristine_ov)
        left = sh(
            "git", "status", "--porcelain", "--", str(GENERIC), str(OVERRIDE)
        ).stdout
        print(f"### restored; git status: {left.strip() or 'clean'}", flush=True)

    good = [a[0] for a in ARMS if len(bench[a[0]]) == PASSES]
    print("\n  the six shapes that reach this kernel (geomean), two passes\n")
    print(f"  {'arm':>6} {'pass 1':>9} {'pass 2':>9} {'vs base':>9}   tests")
    b0 = None
    for arm in ARMS:
        tag = arm[0]
        if tag not in good:
            print(f"  {tag:>6}   FAILED: {broken.get(tag, 'incomplete')}")
            continue
        g = [
            geo([v[0] for k, v in bench[tag][p].items() if k != SAMPLED])
            for p in range(PASSES)
        ]
        if b0 is None:
            b0 = sum(g) / PASSES
        print(
            f"  {tag:>6} {g[0]:9.3f} {g[1]:9.3f} {sum(g) / PASSES / b0:9.3f}"
            f"   {tests.get(tag, '-')}"
        )

    print("\n  per shape, pass 1 / pass 2:")
    shapes = sorted({k for t in good for p in range(PASSES) for k in bench[t][p]})
    print("  " + f"{'arm':>6} " + " ".join(f"{k[0]}x{k[1]:<7}" for k in shapes))
    for tag in good:
        cells = []
        for k in shapes:
            v = [bench[tag][p].get(k, (float("nan"),))[0] for p in range(PASSES)]
            cells.append(f"{v[0]:.3f}/{v[1]:.3f}")
        print(f"  {tag:>6} " + " ".join(f"{c:>13}" for c in cells))
    print(
        f"\n  {SAMPLED[0]}x{SAMPLED[1]} takes the sampled path and is a CONTROL:"
        " it must not move"
    )

    if good:
        lat = [bench[t][p][k][1] for t in good for p in range(PASSES) for k in shapes]
        print(
            f"  vLLM latency across every arm and shape, max/min "
            f"{max(lat) / min(lat):.2f}"
        )
    occupancy("after")


if __name__ == "__main__":
    main()
