"""Validate the one-read dense sampled route (cdf091a) end to end.

The route is live for rows >= 8192, top_k 512, vocab in [2048, 5120]: the three
large dense benchmark shapes. Each row is either answered exactly from the
sampled band or flagged and redone by the shipped dense copy, so correctness
has to hold on BOTH paths, and the retry path has to be exercised at scale.

PART A -- correctness and cost, through the public operator, route ON and OFF
(FLAGGEMS_HYGON_PREFILL_DENSE_SAMPLED=0), in separate processes:

    normal    standard normal rows                     ~1% take the retry
    partial   random row_starts / row_ends             short rows -> retry
    ties      logits rounded to 1/8: heavy exact ties  the tie fill at K
    band      10 + 0.2u: collapses the 11-bit key      EVERY row retries --
              the worst case for cost: the sampled kernel plus a full dense
              pass over every row, against the dense pass alone
    const     constant rows                            any k is correct

Values are compared as a multiset against torch.topk over each row's range,
and every index must lie inside that range. do_bench, min of three, per input.

PART B -- tools/hygon_topk_accept.py unchanged: both suites, both benchmarks,
two passes, route ON as it ships. The prefill suite's variable-length case at
16383 x 4095 takes this route with random row ends, i.e. the retry at scale.

PART C -- the benchmark with the route OFF, one pass, for a same-session before.

    tools/vendor_probe.sh tools/hygon_prefill_dense_route_check.py hygon_prefill_dense_route_check
"""

import os
import pathlib
import re
import subprocess
import sys

CHILD = r"""
import os, torch, triton
from importlib import import_module

ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")
dev = "cuda"
top_k = 512
on = os.environ.get("FLAGGEMS_HYGON_PREFILL_DENSE_SAMPLED", "1") != "0"


def inputs(num_rows, vocab, stride0, kind):
    torch.manual_seed(42)
    buf = torch.randn((num_rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
    if kind == "ties":
        buf = torch.round(buf * 8) / 8
    elif kind == "band":
        buf = 10.0 + 0.2 * torch.rand_like(buf)
    elif kind == "const":
        buf = torch.full_like(buf, 3.0)
    x = torch.as_strided(buf, (num_rows, vocab), (stride0, 1))
    if kind == "partial":
        g = torch.Generator(device="cpu").manual_seed(7)
        st = torch.randint(0, 300, (num_rows,), generator=g).to(torch.int32)
        en = torch.randint(0, vocab - 300, (num_rows,), generator=g).to(torch.int32)
        en = torch.clamp(st + top_k + en, max=vocab).to(torch.int32)
    else:
        st = torch.zeros(num_rows, dtype=torch.int32)
        en = torch.full((num_rows,), vocab, dtype=torch.int32)
    return x, st.to(dev), en.to(dev)


for num_rows, vocab, stride0 in ((16383, 4095, 4352), (12961, 4100, 4360), (16380, 5115, 5376)):
    for kind in ("normal", "partial", "ties", "band", "const"):
        x, st, en = inputs(num_rows, vocab, stride0, kind)
        routed = ov._can_dense_sample(x, st, en, num_rows, 1, top_k)
        out = torch.full((num_rows, top_k), -7, dtype=torch.int32, device=dev)
        ov.top_k_per_row_prefill(x, st, en, out, num_rows, stride0, 1, top_k)
        torch.cuda.synchronize()
        span = (en - st).long()
        col = torch.arange(vocab, device=dev)[None, :]
        inside = (col >= st[:, None].long()) & (col < en[:, None].long())
        ref = torch.topk(x.masked_fill(~inside, float("-inf")), top_k, dim=1).values
        o = out.long()
        in_range = bool(((o >= 0) & (o < span[:, None])).all())
        got = torch.gather(x, 1, st[:, None].long() + o.clamp(min=0))
        err = float((got.sort(dim=1, descending=True)[0] - ref).abs().max())
        ok = "ok" if err == 0.0 and in_range else f"WRONG(err={err:.1e},range={in_range})"
        flagged = "-"
        if on and routed:
            plan = next(iter(ov._DPLANS.values()))
            flagged = str(int(plan.flags.sum()))
        f = lambda: ov.top_k_per_row_prefill(x, st, en, out, num_rows, stride0, 1, top_k)
        t = min(triton.testing.do_bench(f, warmup=10, rep=200) for _ in range(3)) * 1e3
        print(f"CHILD {num_rows}x{vocab} {kind} routed={routed and on} us={t:.1f}"
              f" flagged={flagged} answer={ok}", flush=True)
        ov._DPLANS.clear()
        x = st = en = out = ref = got = inside = col = o = None
        torch.cuda.empty_cache()
"""


def parse_bench(out):
    rows = {}
    for mm in re.finditer(
        r"SUCCESS\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[torch\.Size\(\[(\d+), (\d+)\]\)"
        r".*?, (\d+), (\d+), 1, (\d+)\]",
        out,
    ):
        rows[(int(mm.group(4)), int(mm.group(5)), int(mm.group(8)))] = (
            float(mm.group(1)),
            float(mm.group(2)),
            float(mm.group(3)),
        )
    return rows


def main():
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True
    ).stdout
    if dirty.strip():
        raise SystemExit("the source tree is modified:\n" + dirty)

    print("### part A: correctness and cost, route ON / OFF", flush=True)
    res = {}
    for state in ("1", "0"):
        env = dict(os.environ)
        env["FLAGGEMS_HYGON_PREFILL_DENSE_SAMPLED"] = state
        r = subprocess.run(
            [sys.executable, "-c", CHILD], capture_output=True, text=True, env=env
        )
        lines = [x[6:] for x in r.stdout.splitlines() if x.startswith("CHILD")]
        for ln in lines:
            print(f"  [{'on ' if state == '1' else 'off'}] {ln}", flush=True)
            shape, kind = ln.split()[:2]
            res[(state, shape, kind)] = ln
        if len(lines) != 15:
            print("  ! child incomplete:", flush=True)
            for ln in (r.stdout + r.stderr).strip().splitlines()[-10:]:
                print(f"    | {ln[:200]}", flush=True)

    print("\n  route ON vs OFF, do_bench us (x = OFF / ON)\n")
    for shape in ("16383x4095", "12961x4100", "16380x5115"):
        cells = []
        for kind in ("normal", "partial", "ties", "band", "const"):
            a, b = res.get(("1", shape, kind)), res.get(("0", shape, kind))
            if a and b:
                ua = float(a.split("us=")[1].split()[0])
                ub = float(b.split("us=")[1].split()[0])
                cells.append(f"{kind} {ub:.0f}->{ua:.0f} (x{ub / ua:.2f})")
        print(f"  {shape}: " + " | ".join(cells))

    print("\n### part B: tools/hygon_topk_accept.py, route ON\n", flush=True)
    here = pathlib.Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / "hygon_topk_accept.py")], text=True)

    print("\n### part C: prefill benchmark with the route OFF, one pass\n", flush=True)
    env = dict(os.environ)
    env["FLAGGEMS_HYGON_PREFILL_DENSE_SAMPLED"] = "0"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "benchmark/test_top_k_per_row_prefill.py",
            "--mode",
            "kernel",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    rows = parse_bench(r.stdout)
    for k in sorted(rows):
        v = rows[k]
        print(
            f"  {k[0]:>6} {k[1]:>7} {k[2]:>5}   vLLM {v[0]:.4f}  Gems {v[1]:.4f}"
            f"  SpeedUp {v[2]:.3f}"
        )


if __name__ == "__main__":
    main()
