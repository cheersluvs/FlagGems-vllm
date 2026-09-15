"""Step 0 before any localisation: WHAT KIND of non-determinism, and which output is RIGHT?

Every probe so far compared N runs against ONE reference run. That design cannot
tell apart
  * the reference (first) run is the odd one and every later run agrees, from
  * every run being different,
and no probe ever checked which output is correct. The final rates were almost
all 30/30 or 0/30. A genuine race with probability p scatters binomially
(12/30, 7/30); a clean 30/30 is exactly what "the reference run was the odd
one" produces. So the property to localise is not yet known.

A second confound: shapes ran in sequence in one process, and Triton
specialises integer arguments (== 1, divisible by 16). Which shape first
triggers a fresh specialisation depends on the ORDER, which could explain why
different probes "reproduced" at different shapes.

This measures both:
  1. each (implementation, shape) in a FRESH process -- no compile history
  2. all shapes in ONE process, in the order the rate probe used
and for every run records an output hash, so the run sequence reads as
AAAAAAAAAAAA (deterministic), ABBBBBBBBBBB (first call differs) or ABCBAD...
(random). For the 20-line kernel each distinct output is also compared with a
float32 host reference, counting GROSS errors (rel > 1e-2; the defect's errors
were 1e2..1e5, while legitimate reduction-order differences are ULP-sized).

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_step0_kind.py
"""

import hashlib
import json
import os
import subprocess
import sys

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

HEAD_DIM, ROPE_DIM, HEAD_BYTES, EPS = 512, 64, 584, 1e-6
RUNS = 12
GROSS = 1e-2
SHAPES = [(17, 64), (20, 64), (12, 128), (64, 64), (1024, 64)]


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def q_arm(q, src, table, tiles, V: tl.constexpr, HH: tl.constexpr,
              W: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        tok = pid // tiles
        rows = tok * (tiles * HH) + (pid % tiles) * HH + tl.arange(0, HH)
        col = tl.arange(0, W)
        blk = tl.load(q + rows[:, None] * W + col[None, :]).to(tl.float32)
        rs = tl.rsqrt(tl.sum(blk * blk, axis=1) / W + 1e-6)
        blk = blk * rs[:, None]
        tl.store(q + rows[:, None] * W + col[None, :],
                 blk.to(tl.bfloat16), mask=col[None, :] < W - V)
        p = tl.load(src + tok)
        half = tl.arange(0, V // 2)
        c = tl.load(table + p * V + half)
        s = tl.load(table + p * V + V // 2 + half)
        po = (rows[:, None, None] * W + (W - V)
              + half[None, :, None] * 2 + tl.arange(0, 2)[None, None, :])
        pair = tl.load(q + po).to(tl.float32)
        e, o = tl.split(pair)
        e, o = e * rs[:, None], o * rs[:, None]
        tl.store(q + po, tl.join(e * c[None, :] - o * s[None, :],
                                 e * s[None, :] + o * c[None, :]).to(tl.bfloat16))

    return q_arm


def host_reference(q0, src, table, n, h):
    """The 20-line kernel in float32 on the host, same order of operations."""
    import torch

    W, V = HEAD_DIM, ROPE_DIM
    q = q0.cpu().clone()
    blk = q.float()
    rs = torch.rsqrt((blk * blk).sum(1) / W + 1e-6)
    p = src.cpu()[torch.arange(n * h) // h]          # row -> token, tiles*HH == h
    c, s = table.cpu()[p, :V // 2], table.cpu()[p, V // 2:]
    pair = blk[:, W - V:].reshape(-1, V // 2, 2)
    e, o = pair[..., 0] * rs[:, None], pair[..., 1] * rs[:, None]
    q[:, :W - V] = (blk[:, :W - V] * rs[:, None]).to(torch.bfloat16)
    q[:, W - V:] = torch.stack((e * c - o * s, e * s + o * c), -1) \
        .reshape(-1, V).to(torch.bfloat16)
    return q


def measure(impl, n, h, q_arm):
    import torch
    import flaggems_vllm

    fn = flaggems_vllm.runtime.torch_device_fn
    dev = flaggems_vllm.device
    torch.manual_seed(0)
    ref = None
    if impl == "standalone":
        tiles = h // 32
        q0 = torch.randn(n * h, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        src = torch.arange(n, dtype=torch.int64, device=dev)
        table = torch.randn(max(n, 8), ROPE_DIM, dtype=torch.float32, device=dev)
        ref = host_reference(q0, src, table, n, h)

        def run():
            q = q0.clone()
            q_arm[(n * tiles,)](q, src, table, tiles, ROPE_DIM, 32, HEAD_DIM,
                                num_warps=1, num_stages=1)
            fn.synchronize()
            return q
    else:
        impl_fn = flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
        q0 = torch.randn(n, h, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        kv = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        pos = torch.arange(n, dtype=torch.int64, device=dev)
        inv = 1.0 / (10000.0 ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32,
                                              device=dev) / ROPE_DIM))
        t = torch.arange(max(4096, n), dtype=torch.float32, device=dev)
        f = torch.einsum("i,j->ij", t, inv)
        cs = torch.cat((f.cos(), f.sin()), dim=-1)
        slot = torch.arange(n, dtype=torch.int64, device=dev)
        kc0 = torch.zeros((n + 63) // 64 + 1, 64 * HEAD_BYTES, dtype=torch.uint8,
                          device=dev)

        def run():
            q, kc = q0.clone(), kc0.clone()
            impl_fn(q, kv, kc, slot, pos, cs, EPS, 64)
            fn.synchronize()
            return q

    labels, first_of, gross = [], {}, {}
    for _ in range(RUNS):
        out = run().cpu()
        sig = hashlib.sha1(out.view(torch.int16).numpy().tobytes()).hexdigest()
        if sig not in first_of:
            first_of[sig] = chr(ord("A") + len(first_of))
            if ref is not None:
                a, b = out.float(), ref.float()
                rel = (a - b).abs() / b.abs().clamp(min=1e-6)
                gross[first_of[sig]] = int((rel > GROSS).sum())
        labels.append(first_of[sig])
        del out
    fn.empty_cache()
    return {"impl": impl, "n": n, "h": h, "seq": "".join(labels), "gross": gross}


def child(argv):
    q_arm = kernel()
    todo = ([(argv[1], int(argv[2]), int(argv[3]))] if argv[0] == "--one" else
            [(i, n, h) for n, h in SHAPES for i in ("override", "standalone")])
    for impl, n, h in todo:
        try:
            print("ROW " + json.dumps(measure(impl, n, h, q_arm)), flush=True)
        except Exception as e:
            lines = [x for x in str(e).splitlines() if x.strip()]
            print("ROW " + json.dumps({"impl": impl, "n": n, "h": h,
                                       "err": (lines[0] if lines else
                                               type(e).__name__)[:60]}), flush=True)


def spawn(args):
    p = subprocess.run([sys.executable, os.path.abspath(__file__)] + args,
                       capture_output=True, text=True, timeout=3600)
    rows = [json.loads(x[4:]) for x in p.stdout.splitlines() if x.startswith("ROW ")]
    if not rows:
        tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
        rows = [{"err": " | ".join(tail)[:80]}]
    return rows


def kind(seq):
    if len(set(seq)) == 1:
        return "deterministic"
    if len(set(seq[1:])) == 1 and seq[0] != seq[1]:
        return "FIRST CALL differs"
    return "RANDOM"


def show(title, rows):
    print("\n" + "=" * 86 + "\n" + title + "\n" + "=" * 86)
    print("  {:<11} {:>5} {:>5}  {:<14} {:<20} {}".format(
        "impl", "n", "h", "runs", "kind", "gross errors per distinct output"))
    for r in rows:
        if "err" in r:
            print("  {:<11} {:>5} {:>5}  ERROR {}".format(
                r.get("impl", "?"), r.get("n", ""), r.get("h", ""), r["err"]))
            continue
        g = ("  ".join("{}={}".format(k, v) for k, v in sorted(r["gross"].items()))
             if r["gross"] else "(no host reference)")
        print("  {:<11} {:>5} {:>5}  {:<14} {:<20} {}".format(
            r["impl"], r["n"], r["h"], r["seq"], kind(r["seq"]), g))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--one", "--seq"):
        child(sys.argv[1:])
        sys.exit(0)
    fresh = []
    for n, h in SHAPES:
        for impl in ("override", "standalone"):
            fresh += spawn(["--one", impl, str(n), str(h)])
    show("1. FRESH PROCESS per (implementation, shape) -- no compile history", fresh)
    show("2. ONE PROCESS, all shapes in sequence -- the rate probe's situation",
         spawn(["--seq"]))
    print("""
How to read it:
  * deterministic in 1, differing in 2   -> history/specialisation-order dependent;
                                            localise with fresh processes only
  * FIRST CALL differs                   -> the odd output is a first-call effect;
                                            the gross column says whether the
                                            first call or the later ones are wrong
  * RANDOM                               -> a genuine per-run race
  * deterministic with gross errors      -> a stable miscompile; the "flakiness"
                                            came from comparing across histories
""")
    print("[RESULT] STEP0_DONE")
