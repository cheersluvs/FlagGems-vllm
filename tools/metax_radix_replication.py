"""Which step of radix final breaks on MetaX?

Bisected: USE_RADIX_FINAL off -> every forced-TLE case CORRECT; on -> duplicate
indices whenever it runs. Radix final works on 256-bin tensors. On NVIDIA
8 warps x 32 lanes is 256 threads -- one element each. On MetaX 8 x 64 is 512
threads, so every element of a 256-tensor is held by TWO threads (a
replicated layout), and that path has never run here. Candidates, in the order
radix final executes them:

    load    counts from smem  (LoadOpConversion's redundant-register branch)
    scan    tl.cumsum          (the cumsum shim)
    reduce  tl.min / tl.max    (threshold bin, count below it)

Each is checked against torch at 4 warps (256 threads, no replication), 8 and
16, with counts sourced three ways: global, smem full view, and smem through
the operator's scalar + offs form (with the shim's opaque pid >> 31).

    /data/wuyuqing/workspace/mctle-test/bin/python tools/metax_radix_replication.py
"""

import sys

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

NB = 256


@triton.jit
def k_radix(cnt_ptr, k_ptr, out_ptr, NB: tl.constexpr, SRC: tl.constexpr):
    bins = tl.arange(0, NB)
    g = tl.load(cnt_ptr + bins)
    if SRC == 0:
        counts = g
    else:
        buf = tle.gpu.alloc([NB], dtype=tl.int32, layout=None,
                            scope=tle.gpu.smem, nv_mma_shared_layout=False)
        tl.store(tle.gpu.local_ptr(buf), g)
        tl.debug_barrier()
        if SRC == 1:
            counts = tl.load(tle.gpu.local_ptr(buf))
        else:
            p = tle.gpu.local_ptr(buf, (0,)) + (tl.program_id(0) >> 31)
            counts = tl.load(p + bins)
    tl.store(out_ptr + bins, counts)                         # load
    pre = tl.cumsum(counts, axis=0) - counts
    tl.store(out_ptr + NB + bins, pre)                       # scan
    kf = tl.load(k_ptr)
    nxt = pre + counts
    m = (pre < kf) & (nxt >= kf)
    tb = tl.min(tl.where(m, bins, NB), axis=0)
    lt = tl.max(tl.where(bins == tb, pre, 0), axis=0)
    tl.store(out_ptr + 2 * NB, tb)                           # reduce
    tl.store(out_ptr + 2 * NB + 1, lt)


def main():
    torch.manual_seed(0)
    names = {0: "global", 1: "smem-view", 2: "smem-scalar"}
    print(f"NB={NB}; 8 random count vectors per config\n")
    print(f"  {'warps':>5} {'threads':>7} {'src':<12} {'load':>5} {'scan':>5} {'reduce':>6}")
    for warps in (4, 8, 16):
        for src in (0, 1, 2):
            bad = {"load": 0, "scan": 0, "reduce": 0}
            err = None
            for t in range(8):
                cnt = torch.randint(0, 20, (NB,), dtype=torch.int32, device="cuda")
                k = int(torch.randint(1, int(cnt.sum()), ()).item())
                kt = torch.tensor([k], dtype=torch.int32, device="cuda")
                out = torch.full((2 * NB + 2,), -9, dtype=torch.int32, device="cuda")
                try:
                    k_radix[(1,)](cnt, kt, out, NB=NB, SRC=src, num_warps=warps)
                    torch.cuda.synchronize()
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {str(e).strip().splitlines()[-1][:90]}"
                    break
                c, o = cnt.cpu().long(), out.cpu().long()
                pre = torch.cumsum(c, 0) - c
                nxt = pre + c
                m = (pre < k) & (nxt >= k)
                tb = int(torch.where(m, torch.arange(NB), NB).min())
                lt = int(pre[tb]) if tb < NB else 0
                bad["load"] += int(not torch.equal(o[:NB], c))
                bad["scan"] += int(not torch.equal(o[NB:2 * NB], pre))
                bad["reduce"] += int(int(o[2 * NB]) != tb or int(o[2 * NB + 1]) != lt)
            if err:
                print(f"  {warps:>5} {warps * 64:>7} {names[src]:<12} FAILED {err}")
            else:
                print(f"  {warps:>5} {warps * 64:>7} {names[src]:<12} "
                      f"{bad['load']:>5} {bad['scan']:>5} {bad['reduce']:>6}")
    print("\n  counts are wrong trials out of 8. 4 warps = 256 threads is the")
    print("  no-replication control; a column that is 0 there and not at 8/16")
    print("  is the step that breaks under replication.")


if __name__ == "__main__":
    sys.exit(main())
