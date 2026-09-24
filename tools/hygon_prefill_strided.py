"""Whose bug is stride1 != 1, and did the old/new benchmark really run two codes?

hygon_prefill_own_kernels found every row WRONG on two strided inputs (a
transposed view, stride1 8 and 16), on both the generic and the dense route of
the new radix kernel. That kernel's strided branch is the generic operator's,
verbatim, and no test ever passes stride1 != 1. So: the same inputs through

    generic      flaggems_vllm.ops.top_k_per_row_prefill itself
    switch off   the override with FLAGGEMS_HYGON_TOPK_PREFILL=0
    old          the override at 45c154a (patched module copies), from a
                 temporary worktree
    new          the override as it is now

plus a contiguous copy of the same rows as a control, and the same strided
rows at stride0 = row length (a padded, non-transposed layout) to separate
"stride1 != 1" from "stride0 < stride1".

Each process first prints the file it imported the override from, with any
import error in full -- the benchmark comparison printed empty paths.

    tools/vendor_probe.sh tools/hygon_prefill_strided.py hygon_prefill_strided
"""

import os
import shutil
import subprocess
import sys
import tempfile

OLD = "45c154a"

CHILD = r"""
import torch
from importlib import import_module
ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
gen = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
print("OUT override file:", ov.__file__)
fn = gen.top_k_per_row_prefill if __import__("os").environ.get("ARM") == "generic" \
    else ov.top_k_per_row_prefill
dev = "cuda"

def run(tag, x, k):
    rows, vocab = x.shape
    st = torch.zeros(rows, dtype=torch.int32, device=dev)
    en = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
    out = torch.full((rows, k), -7, dtype=torch.int32, device=dev)
    fn(x, st, en, out, rows, x.stride(0), x.stride(1), k)
    torch.cuda.synchronize()
    o = out.long()
    ok_range = ((o >= 0) & (o < vocab)).all(dim=1)
    got = torch.gather(x, 1, o.clamp(0, vocab - 1)).sort(dim=1, descending=True).values
    ref = torch.topk(x, k, dim=1).values
    same = (got == ref).all(dim=1) & ok_range
    hit = [int(torch.isin(o[i], torch.topk(x[i], k).indices).sum()) for i in range(min(rows, 2))]
    print(f"OUT {tag:34s} strides {tuple(x.stride())!s:14s} rows ok {int(same.sum())}/{rows}"
          f"  (row 0,1: {hit} of {k} indices right)")

torch.manual_seed(0)
for rows, vocab, k in ((8, 8193, 512), (16, 4096, 512)):
    base = torch.randn(vocab, rows, device=dev)
    xt = base.t()                                       # stride (1, rows)
    run(f"transposed {rows}x{vocab}", xt, k)
    run(f"contiguous {rows}x{vocab}", xt.contiguous(), k)
    buf = torch.randn(rows * vocab * 2, device=dev)
    xs = torch.as_strided(buf, (rows, vocab), (vocab * 2, 2))   # stride1 2, stride0 big
    run(f"stride1=2, stride0>row {rows}x{vocab}", xs, k)
"""


def child(tag, cwd, env_extra):
    env = dict(os.environ)
    env.pop("FLAGGEMS_HYGON_TOPK_PREFILL", None)
    env.update(env_extra)
    r = subprocess.run(
        [sys.executable, "-c", CHILD], capture_output=True, text=True, cwd=cwd, env=env
    )
    print(f"[{tag}]", flush=True)
    for ln in r.stdout.splitlines():
        if ln.startswith("OUT"):
            print("  " + ln[4:], flush=True)
    if r.returncode:
        print("  ! failed:")
        for ln in r.stderr.strip().splitlines()[-12:]:
            print(f"    | {ln[:220]}")


def main():
    here = os.getcwd()
    old_dir = tempfile.mkdtemp(prefix="fgv_old_")
    os.rmdir(old_dir)
    subprocess.run(
        ["git", "worktree", "add", "--detach", old_dir, OLD],
        check=True,
        capture_output=True,
    )
    try:
        child("generic", here, {"ARM": "generic"})
        child("switch off", here, {"FLAGGEMS_HYGON_TOPK_PREFILL": "0"})
        child(f"old {OLD}", old_dir, {})
        child("new", here, {})
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", old_dir], capture_output=True
        )
        shutil.rmtree(old_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
