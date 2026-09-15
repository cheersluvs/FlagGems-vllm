"""Confirm the mechanism, then test a zero-cost fix: pad the grid to a multiple of 40.

WHAT IS SETTLED (20-line kernel vs float32 host reference). The device reports
vector_core_num = multi_processor_count = 40, and every grid measured so far fits
one rule: grid <= 40 or grid % 40 == 0 is clean; any other grid corrupts program
0. Clean: 32 34 36 40 80 120 160 200. Bad: 44..66, 82, 96, 98, 100, 128. And an
out-of-place kernel is clean at both failing shapes, while in-place variants fail
non-monotonically in their complexity -- so in-place aliasing is required and no
particular operation is.

HYPOTHESIS, not yet a finding: the runtime fills the incomplete last group of 40
by re-running program 0 without masking it, so program 0 executes twice,
possibly concurrently. That fits "program 0 only", "only when the last group is
short", non-determinism, and out-of-place being immune (running it twice is then
idempotent). It does not obviously explain gross NoPE errors, since RMSNorm run
twice barely changes anything -- hence part 1.

PART 1 -- WITNESS. Kernels with no operator content: an in-place x = 2x + 1 over
512 elements per program, on zeros. Once gives 1; twice in sequence gives 3; two
interleaved executions give a mixture. Any element other than 1 is direct proof
of repeated execution, and its index says which program.

PART 2 -- PADDING on the 20-line kernel. Round the grid up to a multiple of 40,
with a `pid < nprog` guard so the padding programs do nothing. Control: the same
guard with an unpadded grid, so the branch itself cannot be credited.

PART 3 -- PADDING on the override kernel itself, called directly. Only runs if
the installed override has the `total_programs` guard (the 2-D grid version),
because without it padding programs would write out of bounds. Q is scored
against the same host reference, which is exactly the Q-path math with cos/sin
from the cache. If padding makes it correct, the fix costs one line in the host
and no memory, no extra launch, no extra bandwidth.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_pad40.py
"""

import importlib
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
HEAD_BYTES, EPS = 584, 1e-6
GROSS = 1e-2


@triton.jit
def witness(x, W: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    off = pid * W + tl.arange(0, W)
    tl.store(x + off, tl.load(x + off) * 2.0 + 1.0)


@triton.jit
def _body(q, src, table, pid, tiles, V: tl.constexpr, HH: tl.constexpr,
          W: tl.constexpr):
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
    e = e * rs[:, None]
    o = o * rs[:, None]
    tl.store(q + po, tl.join(e * c[None, :] - o * s[None, :],
                             e * s[None, :] + o * c[None, :]).to(tl.bfloat16))


@triton.jit
def guarded(q, src, table, tiles, nprog, V: tl.constexpr, HH: tl.constexpr,
            W: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if pid < nprog:
        _body(q, src, table, pid, tiles, V, HH, W)


def host_ref(q0, table, n, h):
    q = q0.reshape(-1, W).cpu().clone()
    blk = q.float()
    rs = torch.rsqrt((blk * blk).sum(1) / W + 1e-6)
    tok = torch.arange(n * h) // h
    c, s = table.cpu()[tok, :V // 2], table.cpu()[tok, V // 2:]
    pair = blk[:, W - V:].reshape(-1, V // 2, 2)
    e, o = pair[..., 0] * rs[:, None], pair[..., 1] * rs[:, None]
    q[:, :W - V] = (blk[:, :W - V] * rs[:, None]).to(torch.bfloat16)
    q[:, W - V:] = torch.stack((e * c - o * s, e * s + o * c), -1) \
        .reshape(-1, V).to(torch.bfloat16)
    return q.float()


def gross(out, ref):
    g = (out.reshape(-1, W).cpu().float() - ref).abs() / ref.abs().clamp(min=1e-6) > GROSS
    n = int(g.sum())
    bad = sorted(set((g.any(1).nonzero().flatten() // HH).tolist()))[:3] if n else []
    return n, bad


def err(e):
    lines = [x for x in str(e).splitlines() if x.strip()]
    return (lines[0] if lines else type(e).__name__)[:48]


def pad(n, m):
    return n if n <= m else -(-n // m) * m


def main():
    import flaggems_vllm

    dev = flaggems_vllm.device
    fn = flaggems_vllm.runtime.torch_device_fn
    sync = fn.synchronize
    try:
        M = int(fn.get_device_properties(0).multi_processor_count) or 40
    except Exception:
        M = 40
    print("group size used for padding: {}".format(M))

    # ---------------- PART 1 ----------------
    print("\n" + "=" * 84 + "\nPART 1 -- witness: in-place x = 2x+1, 20 runs; any value != 1 = re-execution\n" + "=" * 84)
    for grid in (17, 40, 41, 44, 60, 64, 80, 100, 128):
        try:
            vals0, others = set(), set()
            for _ in range(20):
                x = torch.zeros(grid * W, dtype=torch.float32, device=dev)
                witness[(grid,)](x, W, num_warps=1, num_stages=1)
                sync()
                x = x.reshape(grid, W).cpu()
                vals0 |= set(torch.unique(x[0]).tolist())
                rows = ((x != 1).any(1)).nonzero().flatten().tolist()
                others |= {r for r in rows if r != 0}
            print("  grid {:>4} {:>6}  values seen in program 0: {}   other programs != 1: {}"
                  .format(grid, "(ok)" if grid <= M or grid % M == 0 else "(bad)",
                          sorted(vals0), sorted(others)[:5]), flush=True)
        except Exception as e:
            print("  grid {:>4}  ERROR {}".format(grid, err(e)))

    # ---------------- PART 2 ----------------
    print("\n" + "=" * 84 + "\nPART 2 -- 20-line kernel, guard only vs guard + grid padded to x{}\n".format(M) + "=" * 84)
    print("  {:>9} {:>7} {:>9}  {:<34} {}".format("shape", "grid", "padded", "guard, unpadded", "guard, padded"))
    for n, h in ((17, 64), (30, 64), (64, 64), (12, 128), (1024, 64)):
        tiles = h // HH
        nprog = n * tiles
        P = pad(nprog, M)
        torch.manual_seed(0)
        q0 = torch.randn(n * h, W, dtype=torch.bfloat16, device=dev)
        src = torch.arange(n, dtype=torch.int64, device=dev)
        table = torch.randn(max(n, 8), V, dtype=torch.float32, device=dev)
        ref = host_ref(q0, table, n, h)
        cells = []
        for grid in (nprog, P):
            try:
                res = []
                for _ in range(5):
                    q = q0.clone()
                    guarded[(grid,)](q, src, table, tiles, nprog, V, HH, W,
                                     num_warps=1, num_stages=1)
                    sync()
                    res.append(gross(q, ref)[0])
                cells.append(str(res))
            except Exception as e:
                cells.append("ERR " + err(e))
        print("  {:>9} {:>7} {:>9}  {:<34} {}".format(
            "{}x{}".format(n, h), nprog, P, cells[0], cells[1]), flush=True)
        fn.empty_cache()

    # ---------------- PART 3 ----------------
    print("\n" + "=" * 84 + "\nPART 3 -- the override kernel, unpadded vs padded, Q vs host reference, 10 runs\n" + "=" * 84)
    mod = importlib.import_module("flaggems_vllm.runtime.backend._ascend.fused"
                                  ".fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")
    if not hasattr(mod, "MAX_GRID_DIM_0"):
        print("  installed override has no total_programs guard (not the 2-D grid")
        print("  version); padding would write out of bounds, so Part 3 is skipped.")
    else:
        kern = mod.fused_qnorm_rope_kv_insert_kernel
        print("  {:>9} {:>7} {:>7}  {:<44} {}".format("shape", "total", "padded", "unpadded", "padded"))
        for n, h in ((17, 64), (19, 64), (20, 64), (12, 128), (64, 64)):
            Hp = mod.q_heads_per_program(h)
            tiles = h // Hp
            qp = n * tiles
            total = qp + n
            P = pad(total, M)
            torch.manual_seed(0)
            q0 = torch.randn(n, h, W, dtype=torch.bfloat16, device=dev)
            kv = torch.randn(n, W, dtype=torch.bfloat16, device=dev)
            pos = torch.arange(n, dtype=torch.int64, device=dev)
            inv = 1.0 / (10000.0 ** (torch.arange(0, V, 2, dtype=torch.float32,
                                                  device=dev) / V))
            t = torch.arange(max(4096, n), dtype=torch.float32, device=dev)
            f = torch.einsum("i,j->ij", t, inv)
            cs = torch.cat((f.cos(), f.sin()), dim=-1)
            slot = torch.arange(n, dtype=torch.int64, device=dev)
            kc0 = torch.zeros((n + 63) // 64 + 1, 64 * HEAD_BYTES, dtype=torch.uint8, device=dev)
            ref = host_ref(q0, cs, n, h)
            cells = []
            for grid in (total, P):
                try:
                    res, bad = [], []
                    for _ in range(10):
                        q, kc = q0.clone(), kc0.clone()
                        kern[(grid, 1)](q, kv, kc, kc.view(torch.bfloat16), slot, pos, cs,
                                        EPS, 64, h, kc.stride(0), grid, qp, total, tiles, Hp,
                                        num_warps=1, num_stages=1)
                        sync()
                        g, b = gross(q, ref)
                        res.append(g)
                        bad = bad or b
                    cells.append("{} bad {}".format(res, bad) if any(res) else str(res))
                except Exception as e:
                    cells.append("ERR " + err(e))
            print("  {:>9} {:>7} {:>7}  {:<44} {}".format(
                "{}x{}".format(n, h), total, P, cells[0], cells[1]), flush=True)
            fn.empty_cache()

    print("""
Reading it:
  Part 1 values 3 or mixed at program 0 on (bad) grids only -> re-execution proven
  Part 2 padded all zero, unpadded not                      -> padding fixes the
                                                               standalone kernel
  Part 3 padded all zero, unpadded not                      -> padding fixes the
                                                               override: one host line
""")
    print("[RESULT] PAD40_DONE")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
