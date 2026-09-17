#!/usr/bin/env bash
# What can actually profile a kernel on this DCU, and what can it count?
#
# Every claim about prefill's bottleneck so far -- "scattered atomics cap it at
# ~320 GB/s" -- comes from REPLICA kernels I wrote and differenced, not from
# the shipped kernel. The replicas are already known to differ from it in one
# way that matters (scalar tiles against the operator's [512,4], 167 GB/s
# against ~278), and prefill was declared exhausted on that attribution. A
# hardware-counter profile of the REAL kernel would confirm it or overturn it.
#
# This survey does NOT guess counter names: Hygon's DCU is an AMD derivative
# and its counter set need not match stock gfx9. It reports which profilers
# exist, what the agent says about itself, and what counters the profiler
# admits to having -- so the actual collection can be written against facts.
#
#   tools/vendor_probe.sh tools/hygon_rocprof_survey.sh hygon_rocprof_survey
set -uo pipefail

say() { printf '\n=== %s\n' "$*"; }

say "profilers on PATH"
for t in rocprof rocprofv2 rocprofv3 rocprof-compute omniperf rocsys \
         hipprof hpcrun; do
    p=$(command -v "$t" 2>/dev/null) && echo "  $t -> $p" || echo "  $t -- absent"
done

say "versions (best effort)"
for t in rocprof rocprofv2 rocprofv3 rocprof-compute omniperf; do
    command -v "$t" >/dev/null 2>&1 || continue
    echo "--- $t"
    "$t" --version 2>&1 | head -5
done

say "ROCm / DTK layout"
ls -d /opt/rocm* /opt/dtk* 2>/dev/null || echo "  none of /opt/rocm* /opt/dtk*"
echo "  ROCM_PATH=${ROCM_PATH:-unset}  HIP_PATH=${HIP_PATH:-unset}"
ls /opt/dtk/bin 2>/dev/null | grep -iE 'prof|trace' | head -20

say "agent: what the card reports about itself"
if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -iE \
        'Name:|gfx|Compute Unit|Max Waves|LDS|Wavefront|Chip|Cacheline|L2' \
        | head -40
else
    echo "  rocminfo absent"
fi

say "counters the profiler admits to having"
for t in rocprofv3 rocprofv2 rocprof; do
    command -v "$t" >/dev/null 2>&1 || continue
    echo "--- $t --list-basic (first 60 lines)"
    "$t" --list-basic 2>&1 | head -60
    echo "--- $t --list-derived (first 40 lines)"
    "$t" --list-derived 2>&1 | head -40
    break
done

say "kernel trace of the real operator, if any profiler works"
DRIVER="tools/hygon_prefill_one.py"
ARGS="16383 4095 512 4352 20"
for t in rocprofv3 rocprofv2 rocprof; do
    command -v "$t" >/dev/null 2>&1 || continue
    echo "--- $t --stats on $DRIVER $ARGS"
    rm -rf /tmp/rocprof_survey && mkdir -p /tmp/rocprof_survey
    ( cd "$(pwd)" && timeout 600 "$t" --stats -o /tmp/rocprof_survey/out.csv \
        "${PY:-python}" "$DRIVER" $ARGS ) 2>&1 | tail -20
    echo "--- files produced"
    ls -la /tmp/rocprof_survey 2>/dev/null | head
    for f in /tmp/rocprof_survey/*stats*.csv /tmp/rocprof_survey/*.csv; do
        [ -f "$f" ] || continue
        echo "--- $f (first 15 lines)"
        head -15 "$f"
    done
    break
done

say "survey done"
