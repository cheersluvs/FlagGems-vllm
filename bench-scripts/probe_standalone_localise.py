"""Localise the 20-line kernel's defect: grid sweep, then one deletion at a time.

WHAT STEP 0 SETTLED.
  * Fresh process and one-process-in-sequence classified every row identically,
    so compile history / specialisation order is not the variable.
  * It is not a first-call effect. Where the standalone kernel fails, EVERY run
    is wrong, each by ~16380 gross errors -- one tile, 32 rows x 512 = 16384,
    i.e. one program computing garbage -- and the garbage differs per run,
    which is all the "random" hashes meant.
  * Where it passes (17x64, 20x64) it is deterministic and matches the float32
    host reference with zero gross errors, which validates that reference.

WHY THIS IS NOW A GOOD TARGET. The property is binary and enormous: 0 versus
~16384 gross errors against an independent reference, the same on every run.
No rates, no statistics, no comparison of runs with each other -- one run
decides, and three are taken only to show it does not wobble.

WHAT IT DOES NOT YET SAY. The shipped override failed at 17x64 and nowhere else
in step 0; this kernel fails at 12x128, 64x64 and 1024x64 and passes at 17x64.
Disjoint shapes. So this kernel demonstrates a backend defect in these
constructs, but it is NOT established to be the override's defect. Localising
it here is the best lead, and whatever fixes it must then be tried on the
override against the test oracle, not assumed to carry over.

PART 1 -- GRID SWEEP. Passing grids so far were 34 and 40; failing were 48, 128
and 2048, all multiples of 16. The grid is not a kernel argument, so Triton's
integer specialisation cannot see it; if the multiple-of-16 pattern holds, the
trigger is how the runtime partitions programs, not the compiled code. The
sweep also reports WHICH programs are wrong.

PART 2 -- DELETIONS, each against its own host reference, at two failing shapes
and one passing control. One constexpr flag per ingredient, folded at compile
time, so a disabled ingredient is absent from the IR rather than branched over:
  v0 as shipped in the 20 lines
  v1 unmasked store: read the pairs first, store all 512 cols, then the RoPE --
     the documented workaround for this backend's two open masked-store /
     masked-load defects, and never validly tested here
  v2 no rotation, so no tl.split / tl.join
  v3 no reduction, constant scale instead of rsqrt(sum)
  v4 fp32 storage instead of bf16
  v5 no NoPE store at all
A variant "fixes" it only if both failing shapes go to zero gross errors on all
runs while the control stays at zero.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_standalone_localise.py
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
RUNS = 3
GROSS = 1e-2


@triton.jit
def q_arm_v(q, src, table, tiles, V: tl.constexpr, HH: tl.constexpr,
            W: tl.constexpr, UNMASKED: tl.constexpr, ROTATE: tl.constexpr,
            REDUCE: tl.constexpr, STORE_NOPE: tl.constexpr, FP32: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    tok = pid // tiles
    rows = tok * (tiles * HH) + (pid % tiles) * HH + tl.arange(0, HH)
    col = tl.arange(0, W)
    blk = tl.load(q + rows[:, None] * W + col[None, :]).to(tl.float32)
    if REDUCE:
        rs = tl.rsqrt(tl.sum(blk * blk, axis=1) / W + 1e-6)
    else:
        rs = tl.full((HH,), 0.5, tl.float32)
    blk = blk * rs[:, None]
    p = tl.load(src + tok)
    half = tl.arange(0, V // 2)
    c = tl.load(table + p * V + half)
    s = tl.load(table + p * V + V // 2 + half)
    po = (rows[:, None, None] * W + (W - V)
          + half[None, :, None] * 2 + tl.arange(0, 2)[None, None, :])
    if UNMASKED:
        pair = tl.load(q + po).to(tl.float32)
        if STORE_NOPE:
            if FP32:
                tl.store(q + rows[:, None] * W + col[None, :], blk)
            else:
                tl.store(q + rows[:, None] * W + col[None, :], blk.to(tl.bfloat16))
    else:
        if STORE_NOPE:
            if FP32:
                tl.store(q + rows[:, None] * W + col[None, :], blk,
                         mask=col[None, :] < W - V)
            else:
                tl.store(q + rows[:, None] * W + col[None, :],
                         blk.to(tl.bfloat16), mask=col[None, :] < W - V)
        pair = tl.load(q + po).to(tl.float32)
    if ROTATE:
        e, o = tl.split(pair)
        e = e * rs[:, None]
        o = o * rs[:, None]
        out = tl.join(e * c[None, :] - o * s[None, :], e * s[None, :] + o * c[None, :])
    else:
        out = pair * rs[:, None, None]
    if FP32:
        tl.store(q + po, out)
    else:
        tl.store(q + po, out.to(tl.bfloat16))


VARIANTS = {
    "v0 as shipped":       dict(UNMASKED=0, ROTATE=1, REDUCE=1, STORE_NOPE=1, FP32=0),
    "v1 unmasked store":   dict(UNMASKED=1, ROTATE=1, REDUCE=1, STORE_NOPE=1, FP32=0),
    "v2 no split/join":    dict(UNMASKED=0, ROTATE=0, REDUCE=1, STORE_NOPE=1, FP32=0),
    "v3 no reduction":     dict(UNMASKED=0, ROTATE=1, REDUCE=0, STORE_NOPE=1, FP32=0),
    "v4 fp32 storage":     dict(UNMASKED=0, ROTATE=1, REDUCE=1, STORE_NOPE=1, FP32=1),
    "v5 no NoPE store":    dict(UNMASKED=0, ROTATE=1, REDUCE=1, STORE_NOPE=0, FP32=0),
}


def host_ref(q0, src, table, n, h, f):
    q = q0.cpu().clone()
    blk = q.float()
    rs = (torch.rsqrt((blk * blk).sum(1) / W + 1e-6) if f["REDUCE"]
          else torch.full((blk.shape[0],), 0.5))
    cast = (lambda x: x) if f["FP32"] else (lambda x: x.to(torch.bfloat16))
    p = src.cpu()[torch.arange(n * h) // h]
    c, s = table.cpu()[p, :V // 2], table.cpu()[p, V // 2:]
    pair = blk[:, W - V:].reshape(-1, V // 2, 2)
    if f["STORE_NOPE"]:
        q[:, :W - V] = cast(blk[:, :W - V] * rs[:, None])
    if f["ROTATE"]:
        e, o = pair[..., 0] * rs[:, None], pair[..., 1] * rs[:, None]
        rope = torch.stack((e * c - o * s, e * s + o * c), -1).reshape(-1, V)
    else:
        rope = (pair * rs[:, None, None]).reshape(-1, V)
    q[:, W - V:] = cast(rope)
    return q


def measure(n, h, f, dev, sync):
    tiles = h // HH
    torch.manual_seed(0)
    dt = torch.float32 if f["FP32"] else torch.bfloat16
    q0 = torch.randn(n * h, W, dtype=dt, device=dev)
    src = torch.arange(n, dtype=torch.int64, device=dev)
    table = torch.randn(max(n, 8), V, dtype=torch.float32, device=dev)
    ref = host_ref(q0, src, table, n, h, f).float()
    grosses, bad_progs = [], None
    for _ in range(RUNS):
        q = q0.clone()
        q_arm_v[(n * tiles,)](q, src, table, tiles, V, HH, W, num_warps=1,
                              num_stages=1, **f)
        sync()
        a = q.cpu().float()
        g = (a - ref).abs() / ref.abs().clamp(min=1e-6) > GROSS
        grosses.append(int(g.sum()))
        if bad_progs is None and grosses[-1]:
            rows = g.any(1).nonzero().flatten()
            bad_progs = sorted(set((rows // HH).tolist()))
        del q, a, g
    return grosses, bad_progs or []


def main():
    import flaggems_vllm

    dev = flaggems_vllm.device
    fn = flaggems_vllm.runtime.torch_device_fn

    print("=" * 80)
    print("PART 1 -- grid sweep, v0, gross errors per run (0 = matches host)")
    print("=" * 80)
    print("  {:>5} {:>5} {:>6} {:>7} {:>22} {}".format(
        "n", "h", "grid", "%16==0", "gross per run", "bad programs"))
    v0 = VARIANTS["v0 as shipped"]
    for n, h in ((16, 64), (17, 64), (20, 64), (23, 64), (24, 64), (25, 64),
                 (28, 64), (32, 64), (33, 64), (40, 64), (41, 64), (48, 64),
                 (49, 64), (64, 64), (8, 128), (9, 128), (10, 128), (11, 128),
                 (12, 128), (13, 128), (16, 128)):
        grid = n * (h // HH)
        try:
            gr, bp = measure(n, h, v0, dev, fn.synchronize)
            print("  {:>5} {:>5} {:>6} {:>7} {:>22} {}".format(
                n, h, grid, "yes" if grid % 16 == 0 else "", str(gr),
                (str(bp[:6]) + ("..." if len(bp) > 6 else "")) if bp else ""),
                flush=True)
        except Exception as e:
            lines = [x for x in str(e).splitlines() if x.strip()]
            print("  {:>5} {:>5} {:>6}   ERROR {}".format(
                n, h, grid, (lines[0] if lines else type(e).__name__)[:50]))
        fn.empty_cache()

    print("\n" + "=" * 80)
    print("PART 2 -- one deletion at a time, each against its own host reference")
    print("=" * 80)
    shapes = ((64, 64), (12, 128), (17, 64))       # two failing, one control
    print("  {:<20} {:>18} {:>18} {:>18}   verdict".format(
        "variant", "64x64", "12x128", "17x64 (control)"))
    fixes = []
    for name, f in VARIANTS.items():
        cells, ok = [], True
        for i, (n, h) in enumerate(shapes):
            try:
                gr, _ = measure(n, h, f, dev, fn.synchronize)
                cells.append(str(gr))
                ok = ok and all(x == 0 for x in gr)
            except Exception as e:
                lines = [x for x in str(e).splitlines() if x.strip()]
                cells.append("ERR " + (lines[0] if lines else type(e).__name__)[:13])
                ok = False
            fn.empty_cache()
        verdict = "CLEAN" if ok else ""
        if ok and name != "v0 as shipped":
            fixes.append(name)
        print("  {:<20} {:>18} {:>18} {:>18}   {}".format(name, *cells, verdict),
              flush=True)

    print()
    if fixes:
        print("REMOVING THIS MAKES THE 20-LINE KERNEL CORRECT: " + "; ".join(fixes))
        print("Next, separately: apply that change to the shipped override and test")
        print("17x64 against the suite's oracle over 30+ runs -- this kernel's")
        print("defect is not yet shown to be the override's.")
        print("\n[RESULT] LOCALISED")
    else:
        print("No single deletion makes it correct. Read Part 1 for the trigger")
        print("before cutting further; if only multiples of 16 fail, the cause is")
        print("the runtime's partitioning of the grid, not any operation here.")
        print("\n[RESULT] NOT_A_SINGLE_INGREDIENT")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
