"""How much does moe_align_block_size's serialized stage1 cost, and what would
tl.histogram cost instead?

Its Ascend override counts tokens per expert one token at a time:

    # Process tokens one at a time with scalar atomic_add to avoid
    # a triton-ascend bug where vectorized atomic_add to the same
    # address only executes once (drops all but one increment).
    for i in range(tokens_per_thread):
        ...
        tl.atomic_add(tokens_cnts_ptr + off_c + expert_id, 1)

The shape makes this cheaper to fix than it looks: grid is (num_experts,) and
program pid owns row pid+1 of a [num_experts+1, num_experts] matrix, so there is
no contention *between* programs.  The atomic exists only because several lanes
of one program may hit the same expert -- exactly what tl.histogram computes in
registers.  So the replacement needs no atomic at all: one vector load, one
histogram, one store of the row.

Correctness is checked against torch.bincount per program, not just A vs B.
"""

import os
import sys
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass
try:
    import torch_musa  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl


def _device():
    for name in ("npu", "musa", "cuda"):
        mod = getattr(torch, name, None)
        if mod is not None and mod.is_available():
            return name, mod
    raise RuntimeError("no accelerator found")


DEV, DEVMOD = _device()


@triton.jit
def stage1_serial(topk_ids_ptr, cnts_ptr, num_experts: tl.constexpr, numel,
                  tokens_per_thread: tl.constexpr):
    """Verbatim shape of the shipped override: one token per iteration."""
    pid = tl.program_id(0)
    start = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts
    for i in range(tokens_per_thread):
        offset = start + i
        if offset < numel:
            expert_id = tl.load(topk_ids_ptr + offset)
            tl.atomic_add(cnts_ptr + off_c + expert_id, 1)


@triton.jit
def stage1_hist(topk_ids_ptr, cnts_ptr, num_experts: tl.constexpr, numel,
                tokens_per_thread: tl.constexpr):
    """One vector load, one histogram, one store.  No atomics: the row is ours.

    Out-of-range lanes are parked in bin 0 and bin 0 is corrected, because
    Ascend does NOT drop out-of-range values from tl.histogram (measured: 511
    counted where 259 lanes were live).
    """
    pid = tl.program_id(0)
    lane = tl.arange(0, tokens_per_thread)
    offs = pid * tokens_per_thread + lane
    live = offs < numel
    eid = tl.load(topk_ids_ptr + offs, mask=live, other=0)
    counts = tl.histogram(tl.where(live, eid, 0), num_experts)
    parked = tokens_per_thread - tl.sum(live.to(tl.int32), axis=0)
    counts = counts - tl.where(tl.arange(0, num_experts) == 0, parked, 0)
    tl.store(cnts_ptr + (pid + 1) * num_experts + tl.arange(0, num_experts), counts)


def bench(fn, reps=10):
    for _ in range(3):
        fn()
    DEVMOD.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        DEVMOD.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def ceil_div(a, b):
    return (a + b - 1) // b


print(f"device {DEV} | triton {triton.__version__}")
print(f"{'num_experts':>11} {'numel':>9} {'tok/prog':>9} | "
      f"{'serial ms':>10} {'hist ms':>9} {'speedup':>8} | correctness")

for num_experts, num_tokens, topk in ((256, 16384, 8), (256, 512, 8),
                                      (256, 128, 8), (64, 4096, 6),
                                      (8, 4096, 2)):
    numel = num_tokens * topk
    tpt = triton.next_power_of_2(ceil_div(numel, num_experts))
    ids = torch.randint(0, num_experts, (numel,), dtype=torch.int32, device=DEV)

    ref = torch.zeros(num_experts + 1, num_experts, dtype=torch.int32)
    flat = ids.cpu()
    for pid in range(num_experts):
        seg = flat[pid * tpt:min((pid + 1) * tpt, numel)]
        if seg.numel():
            ref[pid + 1] = torch.bincount(seg.long(), minlength=num_experts).to(torch.int32)

    out = {}
    for name, kern in (("serial", stage1_serial), ("hist", stage1_hist)):
        c = torch.zeros(num_experts + 1, num_experts, dtype=torch.int32, device=DEV)
        kern[(num_experts,)](ids, c, num_experts, numel, tpt)
        DEVMOD.synchronize()
        out[name] = (torch.equal(c.cpu(), ref),
                     bench(lambda k=kern: k[(num_experts,)](
                         torch.zeros_like(ids) if False else ids,
                         torch.zeros(num_experts + 1, num_experts,
                                     dtype=torch.int32, device=DEV),
                         num_experts, numel, tpt)))
    (ok_s, t_s), (ok_h, t_h) = out["serial"], out["hist"]
    verdict = ("serial " + ("OK" if ok_s else "WRONG") +
               " / hist " + ("OK" if ok_h else "WRONG"))
    print(f"{num_experts:>11} {numel:>9} {tpt:>9} | "
          f"{t_s:>10.3f} {t_h:>9.3f} {t_s / t_h:>7.2f}x | {verdict}")
