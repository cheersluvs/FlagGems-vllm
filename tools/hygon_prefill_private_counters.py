"""Private per-program counters in collect (A), no atomics in finish's emission (B).

THE COST MODEL THIS TESTS. reports/hygon_masked_atomic.txt measured, on this
card, what one partially-masked atomic to ONE address costs: ~12 ns per TAKEN
lane, serialised (2048 taken lanes -> 25.08 us per program, 1024 -> 12.58,
512 -> 5.91). Only when EVERY lane is taken does it collapse to one operation
(0.20 us). The sampled pipeline still has two same-address hot spots:

    collect  the row's candidate counter   ~1362 appends/row   ~16.6 us
    finish   the row's output slot         ~1024 appends/row   ~12.5 us

A. collect. The row's 16.6 us of serialised appends does NOT shrink as SPLIT
grows -- every program of the row queues on the same address. That is one
explanation of the SPLIT curve nobody has explained yet (2 wins by 5.2%, 4 is
flat, 8 loses 5%, 16 loses 11%): more programs shorten the read but push the
counter towards saturation, and each iteration waits on its atomic before it
can store. It is a hypothesis. The test: give every program its OWN counter
and its OWN segment of the candidate buffer, then sweep SPLIT again. If the
hypothesis holds, the curve stops turning over.

    A2 is the control, in the style of g1: same SPLIT as shipped, only the
    counter changes. If A2 == base, the counter is not a bottleneck at 2 --
    which is what the hypothesis predicts; it only bites as SPLIT grows.

    Counters are padded 32 ints (128 B) apart. A8u is A8 WITHOUT the padding:
    if A8u == A8, what matters is that the address is private, not that the
    line is -- per-address serialisation, not false sharing.

finish compacts the segments in the gather pass it already makes (it reads
every candidate index and writes its value anyway), into a new contiguous
index buffer. Retry: a segment that overflows is a lost candidate, so ANY
segment over SEG sends the row to the exact redo, where today a total over
CAP does.

B. finish's emission. finish is ONE program per row: the slot atomic was never
needed. A register running offset plus `tl.cumsum` over the take-mask gives
the same positions with no atomic and no barrier. Density here is ~75%, far
above the ~9.4% crossover this file measured for cumsum vs atomics.

    B     shipped collect, emission without atomics
    AB8   both, at SPLIT 8 -- B only touches finish, so it should add

EVERY ARM IS MEANT TO BE CORRECT, so each is checked three ways on the focus
shape -- standard normal rows, a narrow band (10 + 0.2u, which collapses the
11-bit key, overflows every segment and forces the retry), and partial rows
(random starts and ends) -- against torch.topk on the selected VALUES; then
the full prefill suite; then the real benchmark, two passes. The budget
(do_bench on our own three launches, prefixes differenced) is printed per arm
so a change can be attributed to the launch it was made in.

PREFLIGHT, before touching the card: every variant parses; no function binds
one loop name under both tl.range and tl.static_range; each _SLaunch dict
lists its kernel's constexprs in SIGNATURE ORDER (the cached runner passes
them positionally); every call in _SPlan.run passes exactly the kernel's
runtime parameters. That last one is the lesson of the fullrow probe: a
re-signed kernel needs its CALLERS audited, not just its anchors.

RECOVERY, if killed between the swap and the restore:

    git checkout -- src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py

    tools/vendor_probe.sh tools/hygon_prefill_private_counters.py hygon_prefill_private_counters
"""

import ast
import os
import pathlib
import re
import subprocess
import sys

OVERRIDE = pathlib.Path(
    "src/flaggems_vllm/runtime/backend/_hygon/fused/top_k_per_row_prefill.py"
)
PASSES = 2
BENCH = ["benchmark/test_top_k_per_row_prefill.py", "--mode", "kernel"]
TESTS = ["tests/test_top_k_per_row_prefill.py"]
FOCUS = (64, 129280, 1024)

# tag, SPLIT, private counters, counter pad (ints), emission without atomics
ARMS = [
    ("base", 2, False, 1, False),
    ("B", 2, False, 1, True),
    ("A2", 2, True, 32, False),
    ("A4", 4, True, 32, False),
    ("A8", 8, True, 32, False),
    ("A16", 16, True, 32, False),
    ("A8u", 8, True, 1, False),
    ("AB8", 8, True, 32, True),
]


# --------------------------------------------------------------------------
# function-scoped text surgery


def _fn_bounds(src, name):
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            first = (
                node.decorator_list[0].lineno if node.decorator_list else node.lineno
            )
            lines = src.splitlines(keepends=True)
            a = sum(len(x) for x in lines[: first - 1])
            b = sum(len(x) for x in lines[: node.end_lineno])
            return a, b
    raise AssertionError(f"no function {name}")


def in_fn(src, name, pairs):
    """Replace inside ONE function only; each old string must be found exactly
    as many times as stated."""
    a, b = _fn_bounds(src, name)
    body = src[a:b]
    for old, new, count in pairs:
        n = body.count(old)
        assert n == count, f"{name}: {old.strip()[:60]!r} found {n}, want {count}"
        body = body.replace(old, new)
    return src[:a] + body + src[b:]


def in_cls(src, name, pairs):
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            lines = src.splitlines(keepends=True)
            a = sum(len(x) for x in lines[: node.lineno - 1])
            b = sum(len(x) for x in lines[: node.end_lineno])
            body = src[a:b]
            for old, new, count in pairs:
                n = body.count(old)
                assert (
                    n == count
                ), f"{name}: {old.strip()[:60]!r} found {n}, want {count}"
                body = body.replace(old, new)
            return src[:a] + body + src[b:]
    raise AssertionError(f"no class {name}")


# --------------------------------------------------------------------------
# A: private counters and segments


def patch_a(src, cpad):
    anchor = "SSPLIT = max(1, int(os.environ.get("
    assert src.count(anchor) == 1
    i = src.index(anchor)
    j = src.index("\n", i) + 1
    src = (
        src[:j]
        + f"_PROBE_PRIVATE = 1\n_PROBE_CPAD = {cpad}  # ints between counters\n"
        + src[j:]
    )

    src = in_fn(
        src,
        "_s_prepare",
        [
            (
                "    BLOCK: tl.constexpr,\n):",
                "    BLOCK: tl.constexpr,\n    SPLIT: tl.constexpr,\n"
                "    CPAD: tl.constexpr,\n):",
                1,
            ),
            (
                "    tl.store(cnt_ptr + row, 0)\n",
                "    tl.store(\n"
                "        cnt_ptr + (row * SPLIT + tl.arange(0, SPLIT)) * CPAD,\n"
                "        tl.zeros([SPLIT], tl.int32),\n"
                "    )\n",
                1,
            ),
        ],
    )

    src = in_fn(
        src,
        "_s_collect",
        [
            (
                "    CHUNK: tl.constexpr,\n):",
                "    CHUNK: tl.constexpr,\n    SEG: tl.constexpr,\n"
                "    CPAD: tl.constexpr,\n):",
                1,
            ),
            (
                "    cnt2 = cnt_ptr + row + tl.zeros([BLOCK, VEC], tl.int32)\n"
                "    cnt1 = cnt_ptr + row + tl.zeros([BLOCK], tl.int32)\n",
                "    cnt2 = cnt_ptr + pid * CPAD + tl.zeros([BLOCK, VEC], tl.int32)\n"
                "    cnt1 = cnt_ptr + pid * CPAD + tl.zeros([BLOCK], tl.int32)\n",
                1,
            ),
            ('sem="relaxed", scope="gpu")', 'sem="relaxed", scope="cta")', 2),
            (
                "        keep = take & (pos >= 0) & (pos < CAP)\n"
                "        tl.store(cand_idx_ptr + row * CAP + pos,",
                "        keep = take & (pos >= 0) & (pos < SEG)\n"
                "        tl.store(cand_idx_ptr + pid * SEG + pos,",
                2,
            ),
        ],
    )

    a, b = _fn_bounds(src, "_s_finish")
    fin = src[a:b]
    g0 = fin.index("    n = tl.minimum(tl.load(cnt_ptr + row), CAP)\n")
    g1 = fin.index("\n    if n <= TOPK:")
    gather_old = fin[g0:g1]
    assert "row_base = logits_ptr + row * stride0 + s" in gather_old
    assert gather_old.rstrip().endswith("tl.debug_barrier()")
    gather_new = """    # The candidates sit in SPLIT per-program segments. Compact them into
    # one contiguous index buffer, gathering the values on the way -- the
    # same one pass over the candidates the shared-counter version made.
    ibase = cidx_ptr + row * CAP
    vbase = cand_val_ptr + row * CAP
    obase = out_ptr + row * TOPK
    cbase = counts_ptr + row * RADIX
    row_base = logits_ptr + row * stride0 + s
    n = tl.zeros((), tl.int32)
    for sg in tl.static_range(SPLIT):
        cseg = tl.minimum(tl.load(cnt_ptr + (row * SPLIT + sg) * CPAD), SEG)
        sbase = cand_idx_ptr + (row * SPLIT + sg) * SEG
        for t in tl.range(0, tl.cdiv(cseg, BLOCK)):
            p = t * BLOCK + lane
            pv = p < cseg
            ci = tl.load(sbase + p, mask=pv, other=0)
            tl.store(ibase + n + p, ci, mask=pv)
            tl.store(
                vbase + n + p, tl.load(row_base + ci, mask=pv, other=0.0), mask=pv
            )
        n += cseg
    tiles = tl.cdiv(n, BLOCK)
    tl.debug_barrier()
"""
    fin = fin[:g0] + gather_new + fin[g1:]
    src = src[:a] + fin + src[b:]

    src = in_fn(
        src,
        "_s_finish",
        [
            (
                "    cand_val_ptr,\n    out_ptr,",
                "    cand_val_ptr,\n    cidx_ptr,\n    out_ptr,",
                1,
            ),
            (
                "    BLOCK: tl.constexpr,\n):",
                "    BLOCK: tl.constexpr,\n    SPLIT: tl.constexpr,\n"
                "    SEG: tl.constexpr,\n    CPAD: tl.constexpr,\n):",
                1,
            ),
            (
                "    c = tl.load(cnt_ptr + row)\n"
                "    if (c < tl.minimum(TOPK, span)) | (c > CAP):\n",
                "    # a segment past SEG lost candidates; that is the overflow now\n"
                "    c = tl.zeros((), tl.int32)\n"
                "    over = tl.zeros((), tl.int32)\n"
                "    for sg in tl.static_range(SPLIT):\n"
                "        craw = tl.load(cnt_ptr + (row * SPLIT + sg) * CPAD)\n"
                "        c += tl.minimum(craw, SEG)\n"
                "        over += (craw > SEG).to(tl.int32)\n"
                "    if (c < tl.minimum(TOPK, span)) | (over > 0):\n",
                1,
            ),
        ],
    )

    src = in_cls(
        src,
        "_SPlan",
        [
            (
                "        self.cnt = torch.empty((num_rows,), dtype=torch.int32, device=dev)\n",
                "        self.cnt = torch.empty(\n"
                "            (num_rows * SSPLIT * _PROBE_CPAD,), dtype=torch.int32, device=dev\n"
                "        )\n"
                "        self.cidx = torch.empty((num_rows, cap), dtype=torch.int32, device=dev)\n",
                1,
            ),
            (
                '                "BLOCK": SBLOCK,\n            },\n            SWARPS,\n'
                "        )\n        schunk",
                '                "BLOCK": SBLOCK,\n                "SPLIT": SSPLIT,\n'
                '                "CPAD": _PROBE_CPAD,\n            },\n            SWARPS,\n'
                "        )\n        schunk",
                1,
            ),
            (
                '                "CHUNK": schunk,\n            },',
                '                "CHUNK": schunk,\n                "SEG": cap // SSPLIT,\n'
                '                "CPAD": _PROBE_CPAD,\n            },',
                1,
            ),
            (
                '            {"TOPK": top_k, "NB": nb, "CAP": cap, "RADIX": SRADIX, "BLOCK": SBLOCK},',
                "            {\n"
                '                "TOPK": top_k,\n'
                '                "NB": nb,\n'
                '                "CAP": cap,\n'
                '                "RADIX": SRADIX,\n'
                '                "BLOCK": SBLOCK,\n'
                '                "SPLIT": SSPLIT,\n'
                '                "SEG": cap // SSPLIT,\n'
                '                "CPAD": _PROBE_CPAD,\n'
                "            },",
                1,
            ),
            (
                "            self.cand_val,\n            indices,",
                "            self.cand_val,\n            self.cidx,\n            indices,",
                1,
            ),
        ],
    )
    return src


# --------------------------------------------------------------------------
# B: finish's emission without atomics

EMIT_OLD = """    thr_key = desired
    tl.store(slot_ptr + row, 0)
    tl.debug_barrier()
    slots = slot_ptr + row + tl.zeros([BLOCK], tl.int32)
    for equal in tl.static_range(2):
        for t in tl.range(0, tiles):
            pos = t * BLOCK + lane
            valid = pos < n
            key = _key32(tl.load(vbase + pos, mask=valid, other=0.0))
            if equal == 0:
                take = valid & (key < thr_key)
            else:
                take = valid & (key == thr_key)
            q = tl.atomic_add(slots, ones, mask=take, sem="relaxed", scope="cta")
            idx = tl.load(ibase + pos, mask=take, other=-1)
            tl.store(obase + q, idx, mask=take & (q < TOPK))
        tl.debug_barrier()
"""
EMIT_NEW = """    thr_key = desired
    # One program owns this row, so the output positions need no atomic: a
    # running offset in a register plus an exclusive prefix over the take mask.
    filled = tl.zeros((), tl.int32)
    for equal in tl.static_range(2):
        for t in tl.range(0, tiles):
            pos = t * BLOCK + lane
            valid = pos < n
            key = _key32(tl.load(vbase + pos, mask=valid, other=0.0))
            if equal == 0:
                take = valid & (key < thr_key)
            else:
                take = valid & (key == thr_key)
            ti = take.to(tl.int32)
            q = filled + tl.cumsum(ti, axis=0) - ti
            filled += tl.sum(ti, axis=0)
            idx = tl.load(ibase + pos, mask=take, other=-1)
            tl.store(obase + q, idx, mask=take & (q < TOPK))
"""


def patch_b(src):
    return in_fn(src, "_s_finish", [(EMIT_OLD, EMIT_NEW, 1)])


def variant(src, arm):
    _, split, private, cpad, emit = arm
    if private:
        src = patch_a(src, cpad)
    if emit:
        src = patch_b(src)
    return src


# --------------------------------------------------------------------------
# preflight


def _constexprs(fn):
    return [
        a.arg
        for a in fn.args.args
        if a.annotation is not None and "constexpr" in ast.unparse(a.annotation)
    ]


def _runtime(fn):
    return [
        a.arg
        for a in fn.args.args
        if a.annotation is None or "constexpr" not in ast.unparse(a.annotation)
    ]


def preflight(src, tag):
    tree = ast.parse(src)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    for name in ("_s_prepare", "_s_collect", "_s_finish"):
        kinds = {}
        for node in ast.walk(fns[name]):
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
                k = ast.unparse(node.iter.func)
                if k.endswith(("tl.range", "tl.static_range")):
                    kinds.setdefault(ast.unparse(node.target), set()).add(k)
        bad = {t: k for t, k in kinds.items() if len(k) > 1}
        assert not bad, f"{tag}/{name}: loop name under both range kinds: {bad}"

    plan = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_SPlan"
    )
    init = next(
        f for f in plan.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"
    )
    run = next(
        f for f in plan.body if isinstance(f, ast.FunctionDef) and f.name == "run"
    )
    launches = {}
    for node in ast.walk(init):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", "") == "_SLaunch"
        ):
            attr = node.targets[0].attr
            kernel = node.value.args[0].id
            keys = [k.value for k in node.value.args[2].keys]
            want = _constexprs(fns[kernel])
            assert keys == want, f"{tag}/{attr}: dict {keys} != signature {want}"
            launches[attr] = kernel
    for node in ast.walk(run):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in launches
        ):
            kernel = launches[node.func.attr]
            want = len(_runtime(fns[kernel]))
            assert len(node.args) == want, (
                f"{tag}/run: {node.func.attr} passes {len(node.args)}, "
                f"{kernel} takes {want}"
            )
    return True


# --------------------------------------------------------------------------
# the card-side child: budget + three correctness checks

CHILD = r"""
import torch, triton
from importlib import import_module

M = "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
m = import_module(M)
dev = "cuda"
num_rows, vocab, top_k, stride0, stride1 = 64, 129280, 1024, 129280, 1


def inputs(kind):
    torch.manual_seed(42)
    buf = torch.randn(
        (num_rows - 1) * stride0 + (vocab - 1) * stride1 + 1, device=dev,
        dtype=torch.float32,
    )
    if kind == "band":
        buf = 10.0 + 0.2 * torch.rand_like(buf)
    x = torch.as_strided(buf, (num_rows, vocab), (stride0, stride1))
    assert x.stride(0) == stride0 and x.stride(1) == stride1
    if kind == "partial":
        g = torch.Generator(device="cpu").manual_seed(7)
        st = torch.randint(0, 2000, (num_rows,), generator=g).to(torch.int32)
        en = (vocab - torch.randint(0, 2000, (num_rows,), generator=g)).to(torch.int32)
    else:
        st = torch.zeros(num_rows, dtype=torch.int32)
        en = torch.full((num_rows,), vocab, dtype=torch.int32)
    return x, st.to(dev), en.to(dev)


def check(kind):
    x, st, en = inputs(kind)
    out = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
    m.top_k_per_row_prefill(x, st, en, out, num_rows, stride0, stride1, top_k)
    torch.cuda.synchronize()
    col = torch.arange(vocab, device=dev)[None, :]
    inside = (col >= st[:, None].long()) & (col < en[:, None].long())
    ref = torch.topk(x.masked_fill(~inside, float("-inf")), top_k, dim=1).values
    pads = int((out < 0).sum())
    got = torch.gather(x, 1, (st[:, None].long() + out.long().clamp(min=0)))
    got = got.sort(dim=1, descending=True)[0]
    err = float((got - ref).abs().max())
    return "ok" if err == 0.0 and pads == 0 else f"WRONG(err={err:.2e},pads={pads})"


checks = {k: check(k) for k in ("normal", "band", "partial")}

x, st, en = inputs("normal")
out = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
assert m._can_sample(x, st, en, num_rows, stride0, stride1, top_k)
plan = m._SPlan(x.device, x.dtype, num_rows, vocab, top_k)


def prep():
    plan.prepare(x, st, en, plan.hist, plan.thr, plan.cnt, stride0)


def prep_coll():
    prep()
    plan.collect(x, st, en, plan.thr, plan.cnt, plan.cand_idx, plan.cand_val, stride0)


def whole():
    plan.run(x, st, en, out, stride0)


b = triton.testing.do_bench
t_p = b(prep, warmup=100, rep=300) * 1e3
t_pc = b(prep_coll, warmup=100, rep=300) * 1e3
t_a = b(whole, warmup=100, rep=300) * 1e3

whole()
torch.cuda.synchronize()
if getattr(m, "_PROBE_PRIVATE", 0):
    seg = plan.cnt.view(num_rows, m.SSPLIT, m._PROBE_CPAD)[:, :, 0].to(torch.int64)
    per_row = seg.sum(1)
    fill = f"{int(seg.max())}/{plan.cap // m.SSPLIT}"
else:
    per_row = plan.cnt.to(torch.int64)
    fill = f"{int(per_row.max())}/{plan.cap}"
print(
    f"CHILD {t_p:.1f} {t_pc - t_p:.1f} {t_a - t_pc:.1f} {t_a:.1f}"
    f" {int(per_row.min())} {int(per_row.max())} {fill}"
    f" {checks['normal']} {checks['band']} {checks['partial']} split={m.SSPLIT}"
)
"""


# --------------------------------------------------------------------------


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


def env_for(arm):
    e = dict(os.environ)
    e["FLAGGEMS_HYGON_PREFILL_SSPLIT"] = str(arm[1])
    return e


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
    return (hits[-1] if hits else "no output")[:220], text[-10:]


def main():
    dirty = sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
    if dirty.strip():
        raise SystemExit("the override is already modified:\n" + dirty)
    pristine = OVERRIDE.read_text()

    print("### preflight", flush=True)
    variants = {}
    for arm in ARMS:
        v = variant(pristine, arm)
        compile(v, f"<{arm[0]}>", "exec")
        preflight(v, arm[0])
        variants[arm[0]] = v
        print(f"      {arm[0]:>5}: ok", flush=True)
    occupancy("before")

    child, tests, bench, broken = {}, {}, {a[0]: [] for a in ARMS}, {}
    try:
        for arm in ARMS:
            tag = arm[0]
            OVERRIDE.write_text(variants[tag])
            print(f"### child (budget + checks), arm {tag}", flush=True)
            r = subprocess.run(
                [sys.executable, "-c", CHILD],
                capture_output=True,
                text=True,
                env=env_for(arm),
            )
            line = [x for x in r.stdout.splitlines() if x.startswith("CHILD")]
            if not line:
                broken[tag], tail = why(r)
                print(f"      ! {tag}: {broken[tag]}", flush=True)
                for ln in tail:
                    print(f"        | {ln[:200]}", flush=True)
                continue
            child[tag] = line[0].split()[1:]
            print(f"      {' '.join(child[tag])}", flush=True)

            print(f"### tests, arm {tag}", flush=True)
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-rf"] + TESTS,
                capture_output=True,
                text=True,
                env=env_for(arm),
            )
            tests[tag] = parse_tests(r.stdout)
            print(f"      {tests[tag]}", flush=True)
            for ln in [x for x in r.stdout.splitlines() if x.startswith("FAILED")][:3]:
                print(f"        {ln[:220]}", flush=True)

        for p in range(PASSES):
            for arm in ARMS:
                tag = arm[0]
                if tag in broken:
                    continue
                OVERRIDE.write_text(variants[tag])
                print(f"### benchmark pass {p + 1}, arm {tag}", flush=True)
                r = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "-s"] + BENCH,
                    capture_output=True,
                    text=True,
                    env=env_for(arm),
                )
                rows = parse_bench(r.stdout)
                if not rows:
                    broken[tag], tail = why(r)
                    print(f"      ! {tag}: {broken[tag]}", flush=True)
                    for ln in tail:
                        print(f"        | {ln[:200]}", flush=True)
                    continue
                bench[tag].append(rows)
    finally:
        OVERRIDE.write_text(pristine)
        left = sh("git", "status", "--porcelain", "--", str(OVERRIDE)).stdout
        print(f"### restored; git status: {left.strip() or 'clean'}", flush=True)

    print(f"\n  budget on {FOCUS[0]}x{FOCUS[1]}, do_bench on our own launches, us\n")
    print(
        f"  {'arm':>5} {'split':>5} {'prepare':>8} {'collect':>8} {'finish':>8}"
        f" {'whole':>8}   cand/row min-max   max fill   normal / band / partial"
    )
    for arm in ARMS:
        tag = arm[0]
        if tag not in child:
            print(f"  {tag:>5}   FAILED: {broken.get(tag, '?')}")
            continue
        v = child[tag]
        print(
            f"  {tag:>5} {arm[1]:>5} {v[0]:>8} {v[1]:>8} {v[2]:>8} {v[3]:>8}"
            f"   {v[4]:>6}-{v[5]:<6}      {v[6]:>9}   {v[7]} / {v[8]} / {v[9]}"
        )

    good = [a[0] for a in ARMS if len(bench[a[0]]) == PASSES]
    print(f"\n  benchmark SpeedUp on {FOCUS[0]}x{FOCUS[1]}, two passes\n")
    print(f"  {'arm':>5} {'pass 1':>9} {'pass 2':>9} {'vs base':>9}   tests")
    b0 = None
    if "base" in good:
        b0 = sum(bench["base"][p][FOCUS][0] for p in range(PASSES)) / PASSES
    for arm in ARMS:
        tag = arm[0]
        if tag not in good:
            print(f"  {tag:>5}   FAILED: {broken.get(tag, 'incomplete')}")
            continue
        v = [bench[tag][p][FOCUS][0] for p in range(PASSES)]
        rel = f"{sum(v) / PASSES / b0:9.3f}" if b0 else "        -"
        print(f"  {tag:>5} {v[0]:9.3f} {v[1]:9.3f} {rel}   {tests.get(tag, '-')}")

    print("\n  the other six shapes do not take the sampled path -- a control:")
    for tag in good:
        gm = []
        for p in range(PASSES):
            vals = [v[0] for k, v in bench[tag][p].items() if k != FOCUS]
            g = 1.0
            for x in vals:
                g *= x
            gm.append(g ** (1.0 / len(vals)))
        print(
            f"      {tag:>5}: geomean of the six " + " / ".join(f"{x:.3f}" for x in gm)
        )

    if good:
        lat = [bench[t][p][FOCUS][1] for t in good for p in range(PASSES)]
        print(
            f"\n  vLLM latency on {FOCUS[0]}x{FOCUS[1]} across arms and passes,"
            f" max/min {max(lat) / min(lat):.2f}"
        )
    occupancy("after")


if __name__ == "__main__":
    main()
