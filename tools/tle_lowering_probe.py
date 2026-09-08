"""Does the tle.gpu shared-memory surface actually LOWER on this card?

`has_triton_tle()` and a symbol check only prove the import resolved. The
generic top_k_per_row TLE path needs four things to generate code, in order:

    1  tle.gpu.alloc  a shared-memory buffer
    2  tle.gpu.local_ptr  into it
    3  tl.atomic_add  scatter through that pointer   <- the histogram
    4  tle.cumsum  over the result                   <- the threshold scan

MetaX C550 reports every symbol present but declares tle_enabled=False, so the
op takes the global-memory path. On Ascend the same symbols were present and
step 2 could not lower at all. This probe settles which case a card is in
before anyone flips the vendor flag.

Each case runs in its own process: on several of these backends one failed
launch poisons the context and every later result in that process is worthless.

    python tools/tle_lowering_probe.py            # all cases, one subprocess each
    python tools/tle_lowering_probe.py alloc      # just one
"""

import os
import subprocess
import sys
import traceback

CASES = ("alloc", "atomic", "cumsum", "smem_budget")


def _spawn_all():
    print(f"--- tle lowering probe | python {sys.version.split()[0]}")
    sys.stdout.flush()
    for case in CASES:
        print("\n" + "=" * 70 + f"\n=== {case}\n" + "=" * 70, flush=True)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), case],
                           capture_output=True, text=True, timeout=900)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stdout.write(r.stderr[-3000:])
            print(f"--- {case}: exit {r.returncode}")
        sys.stdout.flush()


if len(sys.argv) == 1:
    _spawn_all()
    raise SystemExit(0)

CASE = sys.argv[1]

import torch  # noqa: E402

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
import triton.experimental.tle.language as tle  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "npu"
SYNC = torch.cuda.synchronize if DEV == "cuda" else torch.npu.synchronize
NBINS = 256
BLOCK = 128


@triton.jit
def k_alloc(out_ptr, NB: tl.constexpr):
    buf = tle.gpu.alloc((NB,), tl.int32, scope=tle.gpu.smem)
    p = tle.gpu.local_ptr(buf, (0,))
    lane = tl.arange(0, NB)
    tl.store(p + lane, lane * 2)
    tl.debug_barrier()
    tl.store(out_ptr + lane, tl.load(p + lane))


@triton.jit
def k_atomic(idx_ptr, out_ptr, BLK: tl.constexpr, NB: tl.constexpr):
    """The histogram scatter, which is the whole reason to want smem."""
    buf = tle.gpu.alloc((NB,), tl.int32, scope=tle.gpu.smem)
    p = tle.gpu.local_ptr(buf, (0,))
    bins = tl.arange(0, NB)
    tl.store(p + bins, 0)
    tl.debug_barrier()
    tl.atomic_add(p + tl.load(idx_ptr + tl.arange(0, BLK)), 1)
    tl.debug_barrier()
    tl.store(out_ptr + bins, tl.load(p + bins))


@triton.jit
def k_cumsum(in_ptr, pre_ptr, tot_ptr, NB: tl.constexpr):
    bins = tl.arange(0, NB)
    prefix, total = tle.cumsum(tl.load(in_ptr + bins), axis=0, reverse=False)
    tl.store(pre_ptr + bins, prefix)
    tl.store(tot_ptr + bins, total)


@triton.jit
def k_budget(out_ptr, NB: tl.constexpr):
    buf = tle.gpu.alloc((NB,), tl.int32, scope=tle.gpu.smem)
    p = tle.gpu.local_ptr(buf, (0,))
    lane = tl.arange(0, NB)
    tl.store(p + lane, lane)
    tl.debug_barrier()
    tl.store(out_ptr + tl.arange(0, 8), tl.load(p + tl.arange(0, 8)))


def run():
    if CASE == "alloc":
        out = torch.zeros(NBINS, dtype=torch.int32, device=DEV)
        k_alloc[(1,)](out, NB=NBINS)
        SYNC()
        exp = torch.arange(NBINS, dtype=torch.int32, device=DEV) * 2
        ok = torch.equal(out, exp)
        return (f"alloc+local_ptr LOWER, value "
                f"{'CORRECT' if ok else 'WRONG ' + str(out[:8].tolist())}")

    if CASE == "atomic":
        idx = torch.randint(0, NBINS, (BLOCK,), dtype=torch.int32, device=DEV)
        out = torch.zeros(NBINS, dtype=torch.int32, device=DEV)
        k_atomic[(1,)](idx, out, BLK=BLOCK, NB=NBINS)
        SYNC()
        ref = torch.bincount(idx.cpu().long(), minlength=NBINS).to(torch.int32)
        ok = torch.equal(out.cpu(), ref)
        return (f"smem scatter {'CORRECT' if ok else 'WRONG'} | "
                f"sum={int(out.sum())} expected {BLOCK}")

    if CASE == "cumsum":
        src = torch.ones(NBINS, dtype=torch.int32, device=DEV)
        pre = torch.zeros(NBINS, dtype=torch.int32, device=DEV)
        tot = torch.zeros(NBINS, dtype=torch.int32, device=DEV)
        k_cumsum[(1,)](src, pre, tot, NB=NBINS)
        SYNC()
        return (f"tle.cumsum LOWERS | prefix[:5]={pre[:5].tolist()} "
                f"total={int(tot[0])} (expected {NBINS})")

    if CASE == "smem_budget":
        # The op needs ~21 KB (prefill) / ~25 KB (decode) at top_k=1024, and
        # ~29 / ~33 KB at top_k=2048. Walk up past both and find where this
        # card stops, so the ceiling is measured rather than assumed from the
        # 64 KB the driver reports.
        lines = []
        for kb in (4, 8, 16, 24, 32, 40, 48, 56, 64):
            n = kb * 1024 // 4
            out = torch.zeros(8, dtype=torch.int32, device=DEV)
            try:
                k_budget[(1,)](out, NB=n)
                SYNC()
                ok = out[:8].tolist() == list(range(8))
                lines.append(f"  {kb:>3} KB ({n:>6} int32)  "
                             f"{'ok' if ok else 'WRONG VALUES'}")
            except Exception as exc:  # noqa: BLE001 - the ceiling is the result
                msg = str(exc).strip().splitlines()
                lines.append(f"  {kb:>3} KB ({n:>6} int32)  FAILED: "
                             f"{msg[0][:90] if msg else type(exc).__name__}")
                break
        return "smem ceiling walk\n" + "\n".join(lines)

    return f"!! unknown case {CASE}"


print(f"--- {CASE} | triton {triton.__version__} | torch {torch.__version__} "
      f"| device {DEV}")
sys.stdout.flush()
try:
    print(f"RESULT {CASE}: {run()}")
except Exception:
    print(f"RESULT {CASE}: FAILED")
    sys.stdout.flush()
    traceback.print_exc(file=sys.stdout)
sys.stdout.flush()
