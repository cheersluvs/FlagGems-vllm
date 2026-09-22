"""What does the final network's CAP cost, at a fixed candidate count?

WHY. The proposal "narrow bins x small candidate network, jointly" rests on
the final selector's cost model having changed. I argued from the source that
it had changed the wrong way -- `final_network` is a `tl.sort` bitonic network
over a CAP-wide register tile, and CAP is a constexpr -- and that was WRONG
about this implementation. `build_final_source` emits a runtime cascade:

    if   final_cnt <= 64:  final_network(CAP=64)
    elif final_cnt <= 128: final_network(CAP=128)
    elif final_cnt <= 256: final_network(CAP=256)
    else:                  <the original counting ranker>

so a row with 27 candidates takes the 64-wide branch, not a 512-wide sort. The
question cannot be settled by reading the code, only by pricing the branches.

WHAT NARROWING WOULD DO. On the dense shapes the threshold bin holds 27-50
candidates today (tools/hygon_prefill_step0.py), so they take CAP=64. At 512
bins that becomes 104-157 -> CAP=128; at 256 bins 217-322 -> CAP=256, and the
tail of that distribution falls off the cascade into the counting ranker. A
bitonic network's comparator count goes as (CAP/2)*(log2^2(CAP)+log2(CAP))/2,
i.e. about 2.6x per doubling of CAP -- but all four branches are compiled into
the kernel whether or not they run, so register pressure and code size answer
to the widest one. Both effects are measurable and neither is readable.

ARMS. The audit override, with `build_final_source`'s cascade rewritten so a
given input is forced into one branch, at the CURRENT bin count -- the
candidate count is held fixed and only the network width moves:

    audit    the cascade as that branch ships it, (64, 128, 256)
    cap64    only (64,)   -- what the dense shapes take today
    cap128   only (128,)  -- what 512 bins would push them into
    cap256   only (256,)  -- what 256 bins would push them into
    nonet    ()           -- no network at all, the original counting ranker

`nonet` also prices the network itself: the difference between it and `audit`
is how much of the dense gain is the final selector rather than the carry and
VEC2 routes.

Every arm makes the same single launch of the same kernel, so the profiler is
sound here -- unlike the sampled path, where three launches against one made
it overstate a win by 19%.

Each arm loads its own copy of the override with its own patched companion
registered first, because the override captures `build_final_source` at import.

Needs the branch fetched:  git fetch origin codex/hygon-prefill-audit

    tools/vendor_probe.sh tools/hygon_prefill_cap_cost.py hygon_prefill_cap_cost
"""

import importlib.util
import math
import os
import pathlib
import subprocess
import sys
import tempfile

import torch
from torch.profiler import ProfilerActivity, profile

FUSED_PKG = "flaggems_vllm.runtime.backend._hygon.fused"
FUSED_DIR = "src/flaggems_vllm/runtime/backend/_hygon/fused"
OVERRIDE = f"{FUSED_DIR}/top_k_per_row_prefill.py"
FINAL_SRC = "_top_k_per_row_prefill_final_source"
AUDIT_REFS = ("origin/codex/hygon-prefill-audit", "0b05008")
CASCADE = "    for i, cap in enumerate((64, 128, 256)):"
SHAPES = [
    (16383, 4095, 512, 4352),
    (12961, 4100, 512, 4360),
    (16380, 5115, 512, 5376),
    (4100, 1025, 512, 1288),
]
ROUNDS = 7
# caps None means "no network at all": an empty cascade would emit an `else`
# with no `if`, so that arm uses the route's own switch instead.
ARMS = [
    ("audit", (64, 128, 256)),
    ("cap64", (64,)),
    ("cap128", (128,)),
    ("cap256", (256,)),
    ("nonet", None),
]


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True)


def occupancy(tag):
    import shutil

    for cmd in (["hy-smi"], ["rocm-smi"]):
        exe = shutil.which(cmd[0]) or (
            f"/opt/dtk/bin/{cmd[0]}"
            if pathlib.Path(f"/opt/dtk/bin/{cmd[0]}").exists()
            else None
        )
        if not exe:
            continue
        print(f"--- card occupancy {tag}: {cmd[0]}")
        print("\n".join(sh(exe, *cmd[1:]).stdout.strip().splitlines()[:25]))
        return
    print(f"--- card occupancy {tag}: no smi tool found")


def audit_ref():
    for r in AUDIT_REFS:
        if sh("git", "show", f"{r}:{OVERRIDE}").returncode == 0:
            return r
    raise SystemExit("fetch it first:  git fetch origin codex/hygon-prefill-audit")


def arm_module(ref, tag, caps, tmp):
    """One override copy whose final-network cascade holds only `caps`."""
    # 1. the companion, with the cascade rewritten. caps None leaves it alone
    # and switches the whole route off through the env the route already reads,
    # which it does at import -- hence setting it here, before the load.
    src = sh("git", "show", f"{ref}:{FUSED_DIR}/{FINAL_SRC}.py").stdout
    assert src.count(CASCADE) == 1, "the cascade line moved; re-read the companion"
    if caps is None:
        os.environ["FLAGGEMS_HYGON_TOPK_FINAL_NETWORK"] = "0"
    else:
        os.environ.pop("FLAGGEMS_HYGON_TOPK_FINAL_NETWORK", None)
        src = src.replace(CASCADE, f"    for i, cap in enumerate({caps!r}):", 1)
    f = tmp / f"{FINAL_SRC}_{tag}.py"
    f.write_text(src)
    name = f"{FUSED_PKG}.{FINAL_SRC}"
    spec = importlib.util.spec_from_file_location(name, str(f))
    comp = importlib.util.module_from_spec(spec)
    sys.modules[name] = comp  # the override captures it at import
    spec.loader.exec_module(comp)
    # 2. the other companions, unchanged, once is enough but harmless to redo
    for other in (
        "_top_k_per_row_prefill_carry_source",
        "_top_k_per_row_prefill_final_network",
    ):
        t = sh("git", "show", f"{ref}:{FUSED_DIR}/{other}.py").stdout
        g = tmp / f"{other}.py"
        g.write_text(t)
        n2 = f"{FUSED_PKG}.{other}"
        if n2 not in sys.modules:
            s2 = importlib.util.spec_from_file_location(n2, str(g))
            m2 = importlib.util.module_from_spec(s2)
            sys.modules[n2] = m2
            s2.loader.exec_module(m2)
    # 3. the override, with its module-copy names made unique to this arm
    osrc = sh("git", "show", f"{ref}:{OVERRIDE}").stdout
    for n in (
        "_DENSE_NAME",
        "_CARRY_NAME",
        "_VEC2_NAME",
        "_VEC2_FINAL_NAME",
        "_SHORT_BINS_NAME",
        "_SPARSE_NAME",
    ):
        old = f'{n} = "flaggems_vllm.ops._top_k_per_row_prefill_hygon_'
        assert osrc.count(old) == 1, f"{n} not found once"
        osrc = osrc.replace(old, f'{n} = "flaggems_vllm.ops._cap{tag}_', 1)
    of = tmp / f"override_{tag}.py"
    of.write_text(osrc)
    oname = f"{FUSED_PKG}._cap_cost_{tag}"
    ospec = importlib.util.spec_from_file_location(oname, str(of))
    mod = importlib.util.module_from_spec(ospec)
    sys.modules[oname] = mod
    ospec.loader.exec_module(mod)
    return mod


def device_us(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = 0.0
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0)
        total += t or 0.0
    return total / iters


def main():
    ref = audit_ref()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="capcost_"))
    mods = {}
    for tag, caps in ARMS:
        mods[tag] = arm_module(ref, tag, caps, tmp)
        print(f"### arm {tag}: cascade {caps if caps else 'route off'}", flush=True)
    dev = "cuda"
    occupancy("before")
    print(
        f"\ndevice us, interleaved over {ROUNDS} rounds, each arm's FASTEST round."
        "\nThe candidate count is UNCHANGED across arms -- only the compiled"
        "\nnetwork width moves -- so this prices CAP, not the bin count.\n"
    )
    head = f"  {'shape':>18}"
    for tag, _ in ARMS:
        head += f"{tag:>10}"
    for tag, _ in ARMS[1:]:
        head += f"{'a/' + tag:>10}"
    print(head + f"{'normal':>8}{'tied':>6}")
    logs = {t: [] for t, _ in ARMS[1:]}
    for rows, vocab, top_k, stride0 in SHAPES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        tbuf = (buf * 4).round() / 4
        tied = torch.as_strided(tbuf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)
        mins, oks = {}, {"normal": True, "tied": True}
        for tag, _ in ARMS:
            m = mods[tag]

            def run(src, m=m):
                m.top_k_per_row_prefill(src, starts, ends, idx, rows, stride0, 1, top_k)

            for label, src in (("normal", logits), ("tied", tied)):
                want = torch.topk(src, top_k, dim=1).values.sort(dim=1).values
                idx.fill_(-9)
                run(src)
                torch.cuda.synchronize()
                got = src.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
                good = torch.allclose(got, want) and bool((idx >= 0).all())
                if not good:
                    print(f"      ! {tag} WRONG on {label}", flush=True)
                oks[label] = oks[label] and good
            mins[tag] = min(
                device_us(lambda run=run: run(logits)) for _ in range(ROUNDS)
            )
        line = f"  {f'{rows}x{vocab}':>18}"
        for tag, _ in ARMS:
            line += f"{mins[tag]:>10.1f}"
        for tag, _ in ARMS[1:]:
            r = mins["audit"] / mins[tag]
            logs[tag].append(math.log(r))
            line += f"{r:>10.3f}"
        line += f"{'OK' if oks['normal'] else 'WRONG':>8}"
        line += f"{'OK' if oks['tied'] else 'WRONG':>6}"
        print(line, flush=True)
    print()
    for tag, _ in ARMS[1:]:
        print(f"  geomean audit/{tag}: {math.exp(sum(logs[tag]) / len(logs[tag])):.3f}")
    occupancy("after")
    print(
        "\n  cap64 == audit means the cascade already picks the tight branch."
        "\n  cap128 and cap256 are what 512- and 256-bin histograms would force,"
        "\n  at today's candidate count: if those are flat, the network does not"
        "\n  punish narrowing and item 1 is worth building; if they fall away,"
        "\n  narrowing is priced out before the bins are even touched."
        "\n  audit/nonet is the network's own contribution to the dense gain."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
