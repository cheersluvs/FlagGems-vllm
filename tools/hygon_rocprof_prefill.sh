#!/usr/bin/env bash
# Hardware counters on the REAL prefill kernel.
#
# hipprof lists SQ/TCC/CPC counters but its help exposes only tracing. The
# actual collector is the rocprof under /opt/dtk/rocprofiler/bin, which round 1
# missed by looking only on PATH.
#
# The decisive number is FETCH_SIZE on the dense shape. Its row data is 268 MB.
# If the kernel fetches ~268 MB, its second pass is served by the 8 MB L2 and
# the "second read" I spent a probe trying to eliminate never cost anything at
# HBM. If it fetches ~537 MB, both passes really do reach memory and the
# two-pass framing is right. Everything claimed about prefill's bottleneck so
# far comes from replica kernels; this is the first measurement of the kernel
# the operator actually runs.
#
# Counter sets are run one invocation each, every command printed, so a set
# the hardware rejects names itself instead of taking the others down with it.
# SQ can hold 8 counters at a time and CPC 2, per hipprof's own listing.
#
#   tools/vendor_probe.sh tools/hygon_rocprof_prefill.sh hygon_rocprof_prefill
set -uo pipefail

PY=${PY:-python}
DRIVER=tools/hygon_prefill_one.py
DENSE="16383 4095 512 4352 5"
SPARSE="64 129280 1024 129280 5"
WORK=/tmp/rocprof_pmc

say() { printf '\n=== %s\n' "$*"; }

say "the collector"
ls -la /opt/dtk/rocprofiler/bin 2>/dev/null
RP=""
for c in /opt/dtk/rocprofiler/bin/rocprof /opt/dtk/rocprofiler/bin/rocprofv2 \
         /opt/dtk/bin/rocprof /opt/dtk/rocprofiler/rocprofiler/bin/rocprof; do
    [ -x "$c" ] && { RP="$c"; break; }
done
echo "  chosen: ${RP:-NONE FOUND}"
[ -n "$RP" ] && { printf '\n--- $ %s --help\n' "$RP"; "$RP" --help 2>&1 | head -45; }

say "counter names this card defines (atomics and waves, from the XML)"
for f in /opt/dtk/share/profiler/counters/basic_counters.xml \
         /opt/dtk/rocprofiler/lib/rocprofiler/gfx_metrics.xml; do
    [ -f "$f" ] || continue
    echo "--- $f"
    grep -oE 'name="[A-Za-z0-9_]+"' "$f" | sed 's/name="//;s/"//' \
        | grep -iE 'atomic|wave|busy|stall|wait|valu|vmem|lds' | sort -u | head -40
done

[ -z "$RP" ] && { say "no collector; stopping"; exit 0; }

mkdir -p "$WORK"
collect() {  # collect <tag> <shape> <pmc line>
    local tag=$1 shape=$2 pmc=$3
    echo "pmc: $pmc" > "$WORK/$tag.txt"
    printf '\n--- $ %s -i %s.txt -o %s.csv %s %s %s\n' \
        "$RP" "$WORK/$tag" "$WORK/$tag" "$PY" "$DRIVER" "$shape"
    timeout 900 "$RP" -i "$WORK/$tag.txt" -o "$WORK/$tag.csv" \
        "$PY" $DRIVER $shape 2>&1 | tail -12
    if [ -f "$WORK/$tag.csv" ]; then
        echo "--- $tag.csv"
        head -4 "$WORK/$tag.csv"
    else
        echo "--- $tag.csv NOT produced"
    fi
}

say "dense (16383 x 4095, top_k 512): 268 MB of row data, two passes claimed"
collect dense_mem   "$DENSE"  "FETCH_SIZE WRITE_SIZE"
collect dense_l2    "$DENSE"  "L2ReadReqs L2WriteReqs TCC_BUSY_avr"
collect dense_waves "$DENSE"  "SQ_WAVES SQ_BUSY_CYCLES SQ_WAIT_ANY"
collect dense_insts "$DENSE"  "SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INSTS_SALU"

say "sparse (64 x 129280, top_k 1024): 33 MB, 64 programs on 80 CUs"
collect sparse_mem   "$SPARSE" "FETCH_SIZE WRITE_SIZE"
collect sparse_waves "$SPARSE" "SQ_WAVES SQ_BUSY_CYCLES SQ_WAIT_ANY"

say "done"
