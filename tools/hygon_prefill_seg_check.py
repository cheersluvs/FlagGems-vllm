"""Verify the segment-size fix (e1a95d8), then run the standard acceptance.

PART A -- THE REGRESSION. tools/hygon_prefill_fourrow.py found that routing
4x8193 to the sampled path at SSPLIT 16 made every row retry: segments were
CAP // SSPLIT long while CHUNK's 2048 granularity left only 5 of 16 programs
with work. Segments are now min(CAP, CHUNK). This re-creates that exact
configuration through the environment -- nothing is patched -- and reports,
per row, what _s_finish decides: retry if the collected count is under top_k,
over CAP, or any segment filled. Before the fix the first line read 4/4.

    ratio 16, SSPLIT 16, STRIDE 4    4x8193 and 4x16385   (the failing case)
    ratio 16, SSPLIT 4,  STRIDE 4    the same, at the shipped split
    shipped                          64x129280, the one benchmark shape that
                                     takes the sampled path

Each is also checked on normal, narrow-band and partial rows against
torch.topk.

PART B -- tools/hygon_topk_accept.py unchanged: both suites, both benchmarks,
two passes. This is the run the upstream PR quotes, on the trimmed override
(cc8348b), whose code is AST-identical to the fix commit.

    tools/vendor_probe.sh tools/hygon_prefill_seg_check.py hygon_prefill_seg_check
"""

import os
import pathlib
import subprocess
import sys

CHILD = r"""
import torch
from importlib import import_module

m = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
dev = "cuda"


def inputs(num_rows, vocab, stride0, kind):
    torch.manual_seed(42)
    buf = torch.randn((num_rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
    if kind == "band":
        buf = 10.0 + 0.2 * torch.rand_like(buf)
    x = torch.as_strided(buf, (num_rows, vocab), (stride0, 1))
    if kind == "partial":
        g = torch.Generator(device="cpu").manual_seed(7)
        st = torch.randint(0, 500, (num_rows,), generator=g).to(torch.int32)
        en = (vocab - torch.randint(0, 500, (num_rows,), generator=g)).to(torch.int32)
    else:
        st = torch.zeros(num_rows, dtype=torch.int32)
        en = torch.full((num_rows,), vocab, dtype=torch.int32)
    return x, st.to(dev), en.to(dev)


def check(num_rows, vocab, top_k, stride0, kind):
    x, st, en = inputs(num_rows, vocab, stride0, kind)
    out = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
    m.top_k_per_row_prefill(x, st, en, out, num_rows, stride0, 1, top_k)
    torch.cuda.synchronize()
    col = torch.arange(vocab, device=dev)[None, :]
    inside = (col >= st[:, None].long()) & (col < en[:, None].long())
    ref = torch.topk(x.masked_fill(~inside, float("-inf")), top_k, dim=1).values
    pads = int((out < 0).sum())
    got = torch.gather(x, 1, st[:, None].long() + out.long().clamp(min=0))
    err = float((got.sort(dim=1, descending=True)[0] - ref).abs().max())
    return "ok" if err == 0.0 and pads == 0 else f"WRONG({err:.1e},{pads})"


for num_rows, vocab, top_k, stride0 in SHAPES:
    x, st, en = inputs(num_rows, vocab, stride0, "normal")
    routed = m._can_sample(x, st, en, num_rows, stride0, 1, top_k)
    checks = [check(num_rows, vocab, top_k, stride0, k) for k in ("normal", "band", "partial")]
    if not routed:
        print(f"CHILD {num_rows}x{vocab} NOT ROUTED to the sampled path")
        continue
    cap, chunk, seg = m._s_geometry(vocab, top_k)
    plan = m._SPlan(x.device, x.dtype, num_rows, vocab, top_k)
    out = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)
    plan.run(x, st, en, out, stride0)
    torch.cuda.synchronize()
    cnt = plan.cnt.view(num_rows, m.SSPLIT).to(torch.int64)
    c = cnt.clamp(max=seg).sum(1)
    retry = (c < top_k) | (c > cap) | (cnt > seg).any(1)
    busy = int((cnt[0] > 0).sum())
    print(
        f"CHILD {num_rows}x{vocab} split={m.SSPLIT} CAP={cap} CHUNK={chunk} SEG={seg}"
        f" busy={busy}/{m.SSPLIT} fullest={int(cnt.max())} cand={int(c.min())}-{int(c.max())}"
        f" retry={int(retry.sum())}/{num_rows} checks={'/'.join(checks)}"
    )
"""

CASES = [
    (
        "ratio 16, SSPLIT 16, STRIDE 4",
        {"SAMPLED_RATIO": 16, "SSPLIT": 16, "SSTRIDE": 4},
        [(4, 8193, 512, 8456), (4, 16385, 512, 16648)],
    ),
    (
        "ratio 16, SSPLIT 4, STRIDE 4",
        {"SAMPLED_RATIO": 16, "SSPLIT": 4, "SSTRIDE": 4},
        [(4, 8193, 512, 8456), (4, 16385, 512, 16648)],
    ),
    ("shipped", {}, [(64, 129280, 1024, 129280)]),
]


def main():
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True
    ).stdout
    if dirty.strip():
        raise SystemExit("the source tree is modified:\n" + dirty)

    print("### part A: the regression", flush=True)
    for label, knobs, shapes in CASES:
        env = dict(os.environ)
        for k, v in knobs.items():
            env[f"FLAGGEMS_HYGON_PREFILL_{k}"] = str(v)
        r = subprocess.run(
            [sys.executable, "-c", f"SHAPES = {shapes!r}\n" + CHILD],
            capture_output=True,
            text=True,
            env=env,
        )
        print(f"  {label}", flush=True)
        lines = [x for x in r.stdout.splitlines() if x.startswith("CHILD")]
        for ln in lines:
            print(f"      {ln[6:]}", flush=True)
        if len(lines) != len(shapes):
            print("      ! child failed:", flush=True)
            for ln in (r.stdout + r.stderr).strip().splitlines()[-8:]:
                print(f"        | {ln[:200]}", flush=True)

    print("\n### part B: tools/hygon_topk_accept.py\n", flush=True)
    here = pathlib.Path(__file__).resolve().parent
    r = subprocess.run([sys.executable, str(here / "hygon_topk_accept.py")], text=True)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
