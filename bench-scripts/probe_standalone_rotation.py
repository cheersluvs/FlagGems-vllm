"""Narrow the rotation, test aliasing and a program-0 workaround, and pin the grid rule.

WHAT THE LOCALISATION PROBE SETTLED (20-line kernel vs float32 host reference):
  * the bad program is ALWAYS program 0
  * grid rule: multiples of 16 are NOT it -- 32 clean, 46 bad, 80 clean.
    Clean: 32 34 36 40 80.  Bad: 44 46 48 50 52 56 64 66 82 96 98 128.
  * removing the masked store (v1), the reduction (v3), bf16 storage (v4): still broken
  * removing the rotation (v2): 64x64 [0,0,0], 12x128 [0,0,44] -- nearly, not fully
  * removing the NoPE store (v5): 16384 -> 2048, i.e. with the rotation present ALL
    of program 0's output is garbage, even though blk and rs are fine without it

So the rotation is the necessary ingredient, and it contaminates everything
program 0 writes. The rotation bundles four things; this separates them, all
with the c/s table loads kept at v0's position so ordering is not a confound:

  A in-place, table cos/sin      == v0, must reproduce
  B in-place, no rotation        == v2, repeated to see whether the 44 recurs
  C in-place, split -> join, no multiply     : is 3-D split/join alone enough?
  D in-place, split/join with CONSTANT cos/sin : does multiplying by loaded c/s matter?
  E out-of-place, table cos/sin  : loads from q, stores to a separate tensor --
                                   removes every read/write alias
  F out-of-place, program 0 DUPLICATED: grid+1, pid = max(pid-1, 0), so programs
                                   0 and 1 both compute token 0 tile 0 into the
                                   same output (idempotent out-of-place). A
                                   candidate workaround if the runtime spoils
                                   program 0 specifically.

GRID RULE. Clean at <= 40 and at 80 suggests multiples of 40 might be clean
above 40, which would point at how the runtime divides the grid among cores
rather than at the code. Tested at 60, 100, 120, 160, 200 (64 heads) and 60, 80,
120, 160 (128 heads). Device properties are printed for the core count.

Five runs per cell, gross errors (rel > 1e-2) against each variant's own host
reference; 0 on every run is the only CLEAN.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_standalone_rotation.py
"""

import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

W, V, HH = 512, 64, 32
RUNS = 5
GROSS = 1e-2


@triton.jit
def k(q, out, src, table, tiles, V: tl.constexpr, HH: tl.constexpr,
      W: tl.constexpr, ROT: tl.constexpr, DUP0: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if DUP0:
        pid = tl.maximum(pid - 1, 0)
    tok = pid // tiles
    rows = tok * (tiles * HH) + (pid % tiles) * HH + tl.arange(0, HH)
    col = tl.arange(0, W)
    blk = tl.load(q + rows[:, None] * W + col[None, :]).to(tl.float32)
    rs = tl.rsqrt(tl.sum(blk * blk, axis=1) / W + 1e-6)
    blk = blk * rs[:, None]
    tl.store(out + rows[:, None] * W + col[None, :],
             blk.to(tl.bfloat16), mask=col[None, :] < W - V)
    p = tl.load(src + tok)
    half = tl.arange(0, V // 2)
    c = tl.load(table + p * V + half)
    s = tl.load(table + p * V + V // 2 + half)
    po = (rows[:, None, None] * W + (W - V)
          + half[None, :, None] * 2 + tl.arange(0, 2)[None, None, :])
    pair = tl.load(q + po).to(tl.float32)
    if ROT == 0:
        res = pair * rs[:, None, None]
    elif ROT == 1:
        e, o = tl.split(pair)
        e = e * rs[:, None]
        o = o * rs[:, None]
        res = tl.join(e * c[None, :] - o * s[None, :], e * s[None, :] + o * c[None, :])
    elif ROT == 2:
        e, o = tl.split(pair)
        e = e * rs[:, None]
        o = o * rs[:, None]
        res = tl.join(e * 0.6 - o * 0.8, e * 0.8 + o * 0.6)
    else:
        e, o = tl.split(pair)
        e = e * rs[:, None]
        o = o * rs[:, None]
        res = tl.join(e, o)
    tl.store(out + po, res.to(tl.bfloat16))


VARIANTS = [
    ("A in-place table rot (=v0)", dict(ROT=1, DUP0=0), False),
    ("B in-place no rotation (=v2)", dict(ROT=0, DUP0=0), False),
    ("C in-place split->join only", dict(ROT=3, DUP0=0), False),
    ("D in-place constant cos/sin", dict(ROT=2, DUP0=0), False),
    ("E out-of-place table rot", dict(ROT=1, DUP0=0), True),
    ("F out-of-place, prog 0 dup", dict(ROT=1, DUP0=1), True),
]


def host_ref(q0, src, table, n, h, rot):
    q = q0.cpu().clone()
    blk = q.float()
    rs = torch.rsqrt((blk * blk).sum(1) / W + 1e-6)
    p = src.cpu()[torch.arange(n * h) // h]
    c, s = table.cpu()[p, :V // 2], table.cpu()[p, V // 2:]
    pair = blk[:, W - V:].reshape(-1, V // 2, 2)
    e, o = pair[..., 0] * rs[:, None], pair[..., 1] * rs[:, None]
    if rot == 1:
        rope = torch.stack((e * c - o * s, e * s + o * c), -1)
    elif rot == 2:
        rope = torch.stack((e * 0.6 - o * 0.8, e * 0.8 + o * 0.6), -1)
    else:
        rope = torch.stack((e, o), -1)
    q[:, :W - V] = (blk[:, :W - V] * rs[:, None]).to(torch.bfloat16)
    q[:, W - V:] = rope.reshape(-1, V).to(torch.bfloat16)
    return q.float()


def measure(n, h, flags, oop, dev, sync):
    tiles = h // HH
    torch.manual_seed(0)
    q0 = torch.randn(n * h, W, dtype=torch.bfloat16, device=dev)
    src = torch.arange(n, dtype=torch.int64, device=dev)
    table = torch.randn(max(n, 8), V, dtype=torch.float32, device=dev)
    ref = host_ref(q0, src, table, n, h, flags["ROT"])
    grid = n * tiles + (1 if flags["DUP0"] else 0)
    gr, bad = [], None
    for _ in range(RUNS):
        if oop:
            q, out = q0.clone(), q0.clone()
        else:
            q = q0.clone()
            out = q
        k[(grid,)](q, out, src, table, tiles, V, HH, W, num_warps=1,
                   num_stages=1, **flags)
        sync()
        g = (out.cpu().float() - ref).abs() / ref.abs().clamp(min=1e-6) > GROSS
        gr.append(int(g.sum()))
        if bad is None and gr[-1]:
            bad = sorted(set((g.any(1).nonzero().flatten() // HH).tolist()))[:4]
        del q, out, g
    return gr, bad or []


def err(e):
    lines = [x for x in str(e).splitlines() if x.strip()]
    return (lines[0] if lines else type(e).__name__)[:40]


def main():
    import flaggems_vllm

    dev = flaggems_vllm.device
    fn = flaggems_vllm.runtime.torch_device_fn
    try:
        p = fn.get_device_properties(0)
        print("device: " + ", ".join("{}={}".format(a, getattr(p, a))
                                     for a in dir(p) if not a.startswith("_")))
    except Exception as e:
        print("device properties: " + err(e))

    print("\n" + "=" * 88)
    print("PART 1 -- grid rule (variant A), gross errors per run")
    print("=" * 88)
    for n, h in ((30, 64), (50, 64), (60, 64), (80, 64), (100, 64),
                 (15, 128), (20, 128), (30, 128), (40, 128)):
        grid = n * (h // HH)
        try:
            gr, bad = measure(n, h, VARIANTS[0][1], False, dev, fn.synchronize)
            print("  n={:>4} h={:>4} grid={:>4} {:>7}  {}  bad programs {}".format(
                n, h, grid, "(x40)" if grid % 40 == 0 else "", gr, bad), flush=True)
        except Exception as e:
            print("  n={:>4} h={:>4} grid={:>4}  ERROR {}".format(n, h, grid, err(e)))
        fn.empty_cache()

    print("\n" + "=" * 88)
    print("PART 2 -- inside the rotation; 64x64 and 12x128 fail under A, 17x64 is control")
    print("=" * 88)
    shapes = ((64, 64), (12, 128), (17, 64))
    print("  {:<30} {:<24} {:<24} {:<18} verdict".format(
        "variant", "64x64", "12x128", "17x64"))
    for name, flags, oop in VARIANTS:
        cells, clean = [], True
        for n, h in shapes:
            try:
                gr, _ = measure(n, h, flags, oop, dev, fn.synchronize)
                cells.append(str(gr))
                clean = clean and all(x == 0 for x in gr)
            except Exception as e:
                cells.append("ERR " + err(e)[:18])
                clean = False
            fn.empty_cache()
        print("  {:<30} {:<24} {:<24} {:<18} {}".format(
            name, cells[0], cells[1], cells[2], "CLEAN" if clean else ""), flush=True)

    print("""
Reading Part 2:
  C broken             -> 3-D tl.split/tl.join alone is enough; no cos/sin involved
  C clean, D broken    -> it needs the split/join AND arithmetic combining e and o
  D clean, A broken    -> the loaded cos/sin (the table read) is required
  E clean, A broken    -> in-place aliasing of q is required; out-of-place is a fix
  F clean, E broken    -> the runtime spoils program 0 itself; sacrificing it is a
                          workaround -- a diagnosis tool, not something to ship blind
""")
    print("[RESULT] ROTATION_NARROWED")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
