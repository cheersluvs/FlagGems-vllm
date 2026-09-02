#!/usr/bin/env python3
"""Minimal reproducer: tl.atomic_add returns non-unique per-lane old values.

Self-contained -- torch, torch_npu and triton only, no FlagGems. Intended to be
quoted verbatim in an upstream issue, so it prints everything that report needs
and nothing else.

    source /usr/local/Ascend/cann/set_env.sh
    python tools/repro_atomic_return.py
"""

import sys

import torch
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401

    DEV = "npu"
except ImportError:
    DEV = "cuda"

N = 512


@triton.jit
def all_lanes(CNT, POS, N: tl.constexpr):
    """Every lane adds 1 to one counter and records what it was handed."""
    lane = tl.arange(0, N)
    p = tl.atomic_add(
        CNT + tl.zeros([N], tl.int32), tl.full([N], 1, tl.int32), sem="relaxed"
    )
    tl.store(POS + lane, p)


@triton.jit
def half_lanes(CNT, POS, N: tl.constexpr):
    """Same, but only every other lane participates."""
    lane = tl.arange(0, N)
    m = (lane % 2) == 0
    p = tl.atomic_add(
        CNT + tl.zeros([N], tl.int32),
        tl.full([N], 1, tl.int32),
        mask=m,
        sem="relaxed",
    )
    tl.store(POS + lane, p, mask=m)


def run(kernel, n_active, label):
    cnt = torch.zeros(1, dtype=torch.int32, device=DEV)
    pos = torch.full((N,), -1, dtype=torch.int32, device=DEV)
    kernel[(1,)](cnt, pos, N=N)
    if DEV == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()
    vals = [v for v in pos.cpu().tolist() if v != -1]
    distinct = len(set(vals))
    counter = int(cnt[0])
    print(f"  {label}")
    print(f"    participating lanes : {n_active}")
    print(f"    counter after       : {counter}   (expected {n_active})")
    print(f"    distinct returns    : {distinct}   (expected {n_active})")
    print(f"    -> {'OK' if distinct == n_active else 'MISMATCH'}")
    if distinct != n_active and distinct:
        first = sorted(set(vals))[:6]
        print(f"    first distinct      : {first}")
        per = n_active / distinct
        print(f"    lanes per distinct  : {per:.2f}")
    return counter == n_active, distinct == n_active


def main():
    print("=" * 70)
    print("  tl.atomic_add: per-lane return values")
    print("=" * 70)
    print(f"  device  {DEV}")
    print(f"  triton  {triton.__version__}")
    print(f"  torch   {torch.__version__}")
    try:
        import torch_npu as tn

        print(f"  torch_npu {tn.__version__}")
        print(f"  chip    {torch.npu.get_device_properties(0)}")
    except Exception:  # noqa: BLE001
        pass
    print()
    c1, d1 = run(all_lanes, N, "all 512 lanes")
    print()
    c2, d2 = run(half_lanes, N // 2, "every other lane (256 active)")
    print()
    print("  Contract: N lanes adding 1 to one address must receive N DISTINCT")
    print("  values -- the value immediately before each lane's own addition.")
    print("  The accumulation and the returns are reported separately because")
    print("  they can, and here do, disagree.")
    return 0 if (c1 and d1 and c2 and d2) else 1


if __name__ == "__main__":
    sys.exit(main())
