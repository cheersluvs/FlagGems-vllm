"""Do the TLE decode kernel's shared-memory buffers overlap on MetaX?

Every remaining failure is TLE-path-only (the non-TLE path is CORRECT on the
same inputs) and NOT the atomic (masked smem atomics at the operator's
[512] and [512, 4] tiles are all correct). merge_neginf shows the lt-writes
(496 of them) overwritten by later writes, and merge_finfo faults. One
mechanism explains both: two local_alloc buffers given overlapping offsets
by the shared-memory allocator (liveness reuse gone wrong), so writes to one
land in the other.

Compile the kernels the failing cases use, then print every local_alloc in
the TTGIR with its allocation.offset and byte size, and flag overlaps
between buffers.

    PYTHONPATH=src:$PYTHONPATH /data/wuyuqing/workspace/mctle-test/bin/python \
        tools/metax_tle_smem_layout.py
"""

import os
import re
import sys

os.environ["FLAGGEMS_FORCE_TLE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module  # noqa: E402

import torch  # noqa: E402

import flaggems_vllm  # noqa: E402,F401
import metax_tle_shim  # noqa: E402

ok, msg = metax_tle_shim.install()
print(msg)
if not ok:
    sys.exit(3)
gen = import_module("flaggems_vllm.ops.top_k_per_row_decode")

dev = "cuda"
K = 512
# a CORRECT input with the same specialisation as merge_neginf (V=4096 K=512)
# and as mb_full (V=262144: multi-block + merge kernels)
for V in (4096, 262144):
    x = torch.randn(1, V, device=dev)
    lens = torch.full((1,), V, dtype=torch.int32, device=dev)
    out = torch.zeros(1, K, dtype=torch.int32, device=dev)
    gen.top_k_per_row_decode(x, 1, lens, out, 1, V, 1, K)
torch.cuda.synchronize()

ELEM = {"i32": 4, "f32": 4, "i16": 2, "f16": 2, "bf16": 2, "i8": 1, "i64": 8, "f64": 8}

fn = gen.tle_top_k_per_row_decode
caches = getattr(fn, "device_caches", None) or {}
kernels = []
for entry in caches.values():
    c = entry[0] if isinstance(entry, tuple) else entry
    kernels += list(c.values()) if isinstance(c, dict) else []
if not kernels and hasattr(fn, "cache"):
    for c in fn.cache.values():
        kernels += list(c.values())
print(f"{len(kernels)} compiled variant(s) of tle_top_k_per_row_decode\n")

for ck in kernels:
    md = ck.metadata
    ttgir = ck.asm.get("ttgir", "")
    consts = {k: v for k, v in getattr(md, "constants", {}).items()} if hasattr(md, "constants") else {}
    print(f"=== shared={getattr(md, 'shared', '?')} B  warps={md.num_warps}  "
          f"{ {k: v for k, v in consts.items() if isinstance(v, (int, bool))} }")
    bufs = []
    for line in ttgir.splitlines():
        if "local_alloc" not in line:
            continue
        off = re.search(r"allocation\.offset\s*=\s*(\d+)", line)
        ty = re.search(r"!ttg\.memdesc<([0-9x]+)x(\w+)", line)
        if not ty:
            continue
        dims = [int(d) for d in ty.group(1).split("x")]
        n = 1
        for d in dims:
            n *= d
        size = n * ELEM.get(ty.group(2), 4)
        o = int(off.group(1)) if off else None
        bufs.append((o, size, f"{ty.group(1)}x{ty.group(2)}", line.strip()[:70]))
    for o, size, t, _ in sorted(bufs, key=lambda b: (b[0] is None, b[0] or 0)):
        end = (o + size) if o is not None else None
        print(f"  offset={o!s:>7}  size={size:>6}  end={end!s:>7}  {t}")
    placed = [(o, o + s, t) for o, s, t, _ in bufs if o is not None]
    clashes = [(a, b) for i, a in enumerate(placed) for b in placed[i + 1:]
               if a[0] < b[1] and b[0] < a[1]]
    print(f"  overlapping pairs: {len(clashes)}")
    for a, b in clashes:
        print(f"    [{a[0]},{a[1]}) {a[2]}  <->  [{b[0]},{b[1]}) {b[2]}")
    if not bufs:
        print("  (no local_alloc lines found; first lines mentioning shared:)")
        for line in [l for l in ttgir.splitlines() if "shared" in l][:6]:
            print(f"    {line.strip()[:120]}")

    # Round 1: the saved TTGIR predates allocate-shared-memory, so every
    # offset read None -- "0 overlaps" meant nothing. But the multi-block
    # variant declares an extra 512xf32 (s_out_logits, 2048 B) and still
    # reports shared=18484, the same as the single-block one whose buffers
    # sum to exactly that. The real placement only survives in the LLVM IR,
    # as constant offsets from @global_smem.
    need = sum(s for _, s, _, _ in bufs)
    print(f"  buffers sum to {need} B; metadata.shared = {getattr(md, 'shared', '?')} B"
          f"{'   <-- SHORT by ' + str(need - md.shared) + ' B' if need > md.shared else ''}")
    ms = re.search(r'"?ttg\.shared"?\s*=\s*(\d+)', ttgir)
    print(f"  ttgir module ttg.shared = {ms.group(1) if ms else 'absent'}")
    llir = ck.asm.get("llir", "")
    offs = sorted({int(m.group(1)) for m in re.finditer(
        r"@global_smem\s*,\s*i(?:32|64)\s+(-?\d+)", llir)})
    print(f"  llir: constant offsets from @global_smem ({len(offs)}): {offs[:40]}")
    if not offs:
        sample = [l.strip()[:140] for l in llir.splitlines() if "global_smem" in l][:5]
        print("  llir lines mentioning global_smem:")
        for l in sample:
            print(f"    {l}")
    print()
