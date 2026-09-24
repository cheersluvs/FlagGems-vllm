"""The prefill override with its own radix kernel instead of patched module copies.

Before: the radix routes ran copies of the generic module, patched as source
text, written to a private directory and exec'd. After: one @triton.jit kernel
family in the override (_radix_prefill -> _histogram_step -> _bins), copied
from the generic operator's non-TLE path, with VEC / DENSE / SHORT / SKIP
constexprs. The sampled and one-read kernels are unchanged.

PART 0 -- no module copies left in sys.modules; the route each benchmark shape
takes.

PART 1 -- correctness against torch.topk on every route and every branch of
the copied kernel: aligned rows (the assume_aligned loop), unaligned rows
(skip_elems), strided rows (stride1 != 1), partial rows, rows shorter than
top_k (-1 padding), a narrow band, heavy ties, and the one-read route's retry
with every row flagged.

PART 2 -- both test suites.

PART 3 -- the benchmark, OLD (45c154a, the last card-validated patched
version, in a temporary worktree) against NEW, interleaved old/new/old/new.

    tools/vendor_probe.sh tools/hygon_prefill_own_kernels.py hygon_prefill_own_kernels
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

OLD = "45c154a"

CHILD = r"""
import sys, torch
from importlib import import_module
ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
print("OUT module copies in sys.modules:",
      [n for n in sys.modules if "top_k_per_row_prefill." in n
       or "_top_k_per_row_prefill_hygon" in n] or "none")
for rows, vocab, k in ((64, 129280, 1024), (4, 8193, 512), (4, 16385, 512),
                       (4100, 1025, 512), (12961, 4100, 512), (16380, 5115, 512),
                       (16383, 4095, 512)):
    st = torch.zeros(rows, dtype=torch.int32, device="cuda")
    s = ov._can_sample(torch.empty(rows, vocab, device="cuda"), st, st, rows, vocab, 1, k)
    d = ov._can_dense_sample(torch.empty(rows, vocab, device="cuda"), st, st, rows, 1, k)
    r = "sampled" if s else "one-read" if d else {ov._GENERIC_ROUTE: "generic",
        ov._DENSE_ROUTE: "dense", ov._SHORT_ROUTE: "short"}[ov._route(vocab, k)]
    print(f"OUT route {rows}x{vocab} k{k}: {r}")

dev = "cuda"
def check(tag, x, st, en, k):
    rows, vocab = x.shape
    out = torch.full((rows, k), -7, dtype=torch.int32, device=dev)
    ov.top_k_per_row_prefill(x, st, en, out, rows, x.stride(0), x.stride(1), k)
    torch.cuda.synchronize()
    o = out.long()
    n = (en - st).long()
    long_ = n > k
    bad = 0
    if bool(long_.any()):
        col = torch.arange(vocab, device=dev)[None, :]
        s_l, n_l, o_l = st[long_].long(), n[long_], o[long_]
        inside = (col >= s_l[:, None]) & (col < (s_l + n_l)[:, None])
        xl = x[long_]
        ref = torch.topk(xl.masked_fill(~inside, float("-inf")), k, dim=1).values
        in_range = (o_l >= 0) & (o_l < n_l[:, None])
        uniq = (o_l.sort(dim=1).values.diff(dim=1) != 0).all(dim=1)
        got = torch.gather(xl, 1, (s_l[:, None] + o_l).clamp(0, vocab - 1))
        same = (got.sort(dim=1, descending=True).values == ref).all(dim=1)
        bad += int((~(in_range.all(dim=1) & uniq & same)).sum())
    if bool((~long_).any()):
        n_s, o_s = n[~long_], o[~long_]
        pos = torch.arange(k, device=dev)[None, :]
        want = torch.where(pos < n_s[:, None], pos, -1).sort(dim=1).values
        bad += int((o_s.sort(dim=1).values != want).any(dim=1).sum())
    print(f"OUT {tag:44s} {'ok' if bad == 0 else f'WRONG rows={bad}/{rows}'}")


def rows_of(rows, vocab, stride0, kind="normal", seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, generator=g)
    if kind == "ties":
        buf = torch.round(buf * 8) / 8
    elif kind == "band":
        buf = 10.0 + 0.02 * torch.rand(buf.shape, device=dev, generator=g)
    return torch.as_strided(buf, (rows, vocab), (stride0, 1))

def full(rows, vocab):
    return (torch.zeros(rows, dtype=torch.int32, device=dev),
            torch.full((rows,), vocab, dtype=torch.int32, device=dev))

def partial(rows, vocab, k, seed=1):
    g = torch.Generator().manual_seed(seed)
    st = torch.randint(0, 300, (rows,), generator=g)
    ln = torch.randint(k // 2, vocab - 300, (rows,), generator=g)
    en = torch.clamp(st + ln, max=vocab)
    return st.int().to(dev), en.int().to(dev)

# generic route (10 < vocab / k < 64)
x = rows_of(4, 16384, 16384); check("generic aligned 4x16384", x, *full(4, 16384), 512)
x = rows_of(4, 16385, 16448); check("generic unaligned 4x16385", x, *full(4, 16385), 512)
x = rows_of(8, 8193, 8200); check("generic partial rows 8x8193", x, *partial(8, 8193, 512), 512)
base = torch.randn(8193, 8, device=dev)
check("generic strided (stride1=8) 8x8193", base.t(), *full(8, 8193), 512)
# dense route
x = rows_of(64, 4096, 4096); check("dense aligned 64x4096", x, *full(64, 4096), 512)
x = rows_of(64, 4095, 4352); check("dense unaligned 64x4095", x, *full(64, 4095), 512)
x = rows_of(64, 4100, 4360); check("dense partial rows 64x4100", x, *partial(64, 4100, 512), 512)
x = rows_of(64, 4096, 4096, "band"); check("dense narrow band 64x4096", x, *full(64, 4096), 512)
x = rows_of(64, 4096, 4096, "ties"); check("dense ties 64x4096", x, *full(64, 4096), 512)
base = torch.randn(4096, 16, device=dev)
check("dense strided (stride1=16) 16x4096", base.t(), *full(16, 4096), 512)
x = rows_of(64, 4096, 4096)
st = torch.zeros(64, dtype=torch.int32, device=dev)
en = torch.randint(1, 700, (64,), device=dev).int()
check("dense short rows (some <= top_k) 64x4096", x, st, en, 512)
# short-bins route
x = rows_of(4100, 1025, 1025); check("short 4100x1025", x, *full(4100, 1025), 512)
x = rows_of(4100, 1025, 1088); check("short partial rows 4100x1025", x, *partial(4100, 1025, 512), 512)
x = rows_of(512, 1024, 1024, "band"); check("short narrow band 512x1024", x, *full(512, 1024), 512)
# one-read route and its retry
x = rows_of(12961, 4100, 4360); check("one-read 12961x4100", x, *full(12961, 4100), 512)
x = rows_of(12961, 4100, 4360, "ties"); check("one-read ties (retry ~9%)", x, *full(12961, 4100), 512)
x = rows_of(8192, 4096, 4096, "band"); check("one-read band (every row retried)", x, *full(8192, 4096), 512)
x = rows_of(8192, 4096, 4096); check("one-read partial rows", x, *partial(8192, 4096, 512), 512)
# sampled route (unchanged kernels)
x = rows_of(64, 129280, 129280); check("sampled 64x129280", x, *full(64, 129280), 1024)
"""


def parse(out):
    rows = {}
    for m in re.finditer(
        r"SUCCESS\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[torch\.Size\(\[(\d+), (\d+)\]\)"
        r".*?, (\d+), (\d+), 1, (\d+)\]",
        out,
    ):
        rows[(int(m.group(4)), int(m.group(5)), int(m.group(8)))] = (
            float(m.group(1)),
            float(m.group(2)),
            float(m.group(3)),
        )
    return rows


def geo(v):
    g = 1.0
    for x in v:
        g *= x
    return g ** (1.0 / len(v))


def main():
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "tests"],
        capture_output=True,
        text=True,
    ).stdout
    if dirty.strip():
        raise SystemExit("the tree is modified:\n" + dirty)

    print("### part 0 + 1: routes and correctness\n", flush=True)
    r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if ln.startswith("OUT"):
            print("  " + ln[4:], flush=True)
    if r.returncode:
        print("  ! child failed:")
        for ln in r.stderr.strip().splitlines()[-15:]:
            print(f"    | {ln[:200]}")

    print("\n### part 2: tests\n", flush=True)
    for suite in ("prefill", "decode"):
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-rf",
                "-p",
                "no:cacheprovider",
                f"tests/test_top_k_per_row_{suite}.py",
            ],
            capture_output=True,
            text=True,
        )
        tail = [ln for ln in r.stdout.splitlines() if "passed" in ln or "failed" in ln]
        print(f"  {suite}: {tail[-1] if tail else r.stdout[-300:]}", flush=True)
        for ln in [x for x in r.stdout.splitlines() if x.startswith("FAILED")][:5]:
            print(f"    {ln[:200]}", flush=True)

    print(f"\n### part 3: benchmark, OLD ({OLD}) vs NEW, interleaved\n", flush=True)
    here = os.getcwd()
    old_dir = tempfile.mkdtemp(prefix="fgv_old_")
    os.rmdir(old_dir)
    subprocess.run(
        ["git", "worktree", "add", "--detach", old_dir, OLD],
        check=True,
        capture_output=True,
    )
    try:
        for tag, cwd in (("old", old_dir), ("new", here)):
            f = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
                    " as m; print(m.__file__)",
                ],
                capture_output=True,
                text=True,
                cwd=cwd,
            ).stdout.strip()
            print(f"  {tag} imports {f}", flush=True)
        res = {"old": [], "new": []}
        for i, tag in enumerate(("old", "new", "old", "new")):
            cwd = old_dir if tag == "old" else here
            r = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-s",
                    "-p",
                    "no:cacheprovider",
                    "benchmark/test_top_k_per_row_prefill.py",
                    "--mode",
                    "kernel",
                ],
                capture_output=True,
                text=True,
                cwd=cwd,
            )
            rows = parse(r.stdout)
            print(f"  run {i + 1}: {tag}, {len(rows)} shapes", flush=True)
            if not rows:
                print(r.stdout[-1500:])
                raise SystemExit("no SUCCESS rows")
            res[tag].append(rows)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", old_dir], capture_output=True
        )
        shutil.rmtree(old_dir, ignore_errors=True)

    shapes = sorted(res["new"][0], key=lambda k: k[0] * k[1])
    print(
        f"\n  {'shape':>18} {'old ms':>9} {'new ms':>9} {'new/old':>8}"
        f"   {'old SpeedUp':>13} {'new SpeedUp':>13}"
    )
    for k in shapes:
        om = min(p[k][1] for p in res["old"])
        nm = min(p[k][1] for p in res["new"])
        os_ = " / ".join(f"{p[k][2]:.3f}" for p in res["old"])
        ns_ = " / ".join(f"{p[k][2]:.3f}" for p in res["new"])
        print(
            f"  {k[0]:>6}x{k[1]:<6} k{k[2]:<4} {om:9.4f} {nm:9.4f} {nm / om:8.3f}"
            f"   {os_:>13} {ns_:>13}"
        )
    for tag in ("old", "new"):
        g = [geo([p[k][2] for k in shapes]) for p in res[tag]]
        g5 = [geo([p[k][2] for k in shapes if k[0] != 4]) for p in res[tag]]
        print(
            f"  {tag}: geomean {g[0]:.3f} / {g[1]:.3f}   without 4-row"
            f" {g5[0]:.3f} / {g5[1]:.3f}"
        )


if __name__ == "__main__":
    main()
