"""Hardware counters on the prefill kernel -- the right kernel, this time.

Round 3 reached the collector (/opt/dtk/rocprofiler/bin/rocprof) and the SQ
counter sets came back, but two things were wrong with it:

  * the rows printed were torch.randn's kernel, not the operator's. The driver
    builds its inputs on the device, so dispatch 0 is a distribution kernel and
    head -4 showed that. Every SQ number in that report is unusable.
  * the TCC sets -- FETCH_SIZE, WRITE_SIZE, L2ReadReqs -- all failed with
    "Context Create failed", and those are the ones worth having.

So: filter to the operator's kernel with rocprof's own `kernel:` line, and ask
for the memory counters ONE at a time, since a too-wide group is the likeliest
reason a context fails to create. Every set reports whether it worked.

What the numbers are for: FETCH_SIZE against the shape's row bytes says
whether the radix's second pass reaches HBM or is served by the 8 MB L2 --
which decides whether "two passes" was ever the right way to describe this
operator. SQ_WAVES with arch_vgpr and the dispatch geometry gives occupancy
directly, and SQ_WAIT_ANY over SQ_BUSY_CYCLES says how much of the kernel is
spent waiting rather than issuing.

    tools/vendor_probe.sh tools/hygon_rocprof_prefill2.py hygon_rocprof_prefill2
"""

import csv
import os
import subprocess
import sys
import tempfile

ROCPROF = "/opt/dtk/rocprofiler/bin/rocprof"
DRIVER = "tools/hygon_prefill_one.py"
KERNEL = "top_k_per_row_prefill"

# (tag, shape args, row bytes)
SHAPES = [
    ("dense", "16383 4095 512 4352 5", 16383 * 4095 * 4),
    ("sparse", "64 129280 1024 129280 5", 64 * 129280 * 4),
]
# one at a time for the memory blocks; a too-wide group is the likeliest
# reason "Context Create failed"
SETS = [
    ("fetch", "FETCH_SIZE"),
    ("write", "WRITE_SIZE"),
    ("rdreq", "TCC_EA_RDREQ_sum"),
    ("l2hit", "L2CacheHit"),
    ("waves", "SQ_WAVES SQ_BUSY_CYCLES SQ_WAIT_ANY"),
    ("insts", "SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INSTS_SALU"),
]
META = ("grd", "wgr", "lds", "scr", "arch_vgpr", "accum_vgpr", "sgpr")


def run_set(tag, shape, pmc, workdir):
    inp = os.path.join(workdir, f"{tag}.txt")
    out = os.path.join(workdir, f"{tag}.csv")
    with open(inp, "w") as fh:
        fh.write(f"pmc: {pmc}\nkernel: {KERNEL}\n")
    cmd = [ROCPROF, "-i", inp, "-o", out, sys.executable, DRIVER] + shape.split()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    err = [ln for ln in tail if "rror" in ln or "ailed" in ln]
    if not os.path.exists(out):
        return None, err[-2:] or tail[-2:]
    with open(out) as fh:
        rows = [r for r in csv.DictReader(fh) if KERNEL in r.get("KernelName", "")]
    if not rows:
        return None, ["no dispatch matched the kernel filter"] + (err[-1:] or [])
    return rows, err[-1:]


def main():
    if not os.path.exists(ROCPROF):
        print(f"no collector at {ROCPROF}")
        return 1
    workdir = tempfile.mkdtemp(prefix="rocprof_pf2_")
    print(f"counters on {KERNEL}; collector {ROCPROF}\n")
    for tag, shape, row_bytes in SHAPES:
        print(
            f"  === {tag}: {shape.rsplit(' ', 1)[0]}, "
            f"{row_bytes / 1e6:.1f} MB of row data",
            flush=True,
        )
        for stag, pmc in SETS:
            rows, err = run_set(f"{tag}_{stag}", shape, pmc, workdir)
            if rows is None:
                print(f"    {pmc:<38} -- FAILED: {'; '.join(err)[:90]}", flush=True)
                continue
            r0 = rows[0]
            meta = " ".join(f"{k}={r0.get(k, '?')}" for k in META)
            vals = {}
            for key in pmc.split():
                col = next((c for c in r0 if c.strip() == key), None)
                vals[key] = r0.get(col, "?") if col else "?"
            print(f"    {pmc:<38} {len(rows)} dispatches; {meta}", flush=True)
            for key, v in vals.items():
                note = ""
                try:
                    f = float(v)
                    if key == "FETCH_SIZE":
                        note = (
                            f"  = {f * 1024 / row_bytes:.2f}x the row bytes"
                            f" ({f / 1024:.1f} MB)"
                        )
                    elif key == "WRITE_SIZE":
                        note = f"  ({f / 1024:.1f} MB)"
                    elif key == "SQ_WAVES":
                        note = f"  ({f / 80:.0f} waves per CU over the launch)"
                except (TypeError, ValueError):
                    pass
                print(f"      {key:<22} {v}{note}", flush=True)
        print(flush=True)
    print("  FETCH_SIZE near 1x the row bytes means the second pass is served")
    print("  by the 8 MB L2, not HBM; near 2x means both passes reach memory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
